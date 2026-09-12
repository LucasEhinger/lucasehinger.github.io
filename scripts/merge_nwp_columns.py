#!/usr/bin/env python3
"""Splice one model's columns from a narrow re-fetch into the full shard CSVs.

A partial re-fetch (``fetch_nwp_at_obs.py --models gfs --output-dir ...``)
writes narrow shards carrying only that model's columns plus the meta block.
This joins them back onto the complete shards on (valid_utc, target_lead_h),
overwriting ONLY the named models' columns and leaving every other byte alone.

Why this exists: the GFS columns in the committed shards were all sampled at
44.25N 0E -- southwestern France -- because GFS publishes longitude on 0..360
and ``sel(longitude=-71.30, method="nearest")`` silently clamps into range
instead of raising. Re-downloading all six models to fix one of them would cost
another ~12 hours of CI for no reason; the other five are correct and their
values do not depend on GFS at all.

Safety, because this rewrites files that took 12 hours to produce:

  * refuses to run unless every target shard has a matching source shard
  * refuses on ANY (valid_utc, target_lead_h) whose `split` disagrees between
    the two files -- that means the sample was regenerated between the runs and
    the shards no longer describe the same observations
  * refuses when source coverage is below --min-coverage (default 99%), since a
    partial source would blank good cells -- measured over the target rows at the
    leads the SOURCE carries, so a deliberate one-lead re-fetch is judged on the
    rows it meant to replace rather than on the whole ladder
  * refuses when ANY single column comes back with fewer cells filled than the
    target already has -- per column, not in aggregate, because an aggregate rate
    lets one field collapse to zero while the others gain enough to hide it.
    A field the model genuinely does not publish at the re-fetched forecast hours
    can be named in --accept-empty-columns, and must then come back EXACTLY
    empty; partially filled still means a throttled download.
  * writes through a temporary file and os.replace, so an interrupted run
    cannot leave a half-written shard
  * --dry-run reports all of the above and writes nothing

Usage:
    python3 scripts/merge_nwp_columns.py --models gfs \\
        --source-dir files/weather/csv/obs_partial \\
        --target-dir files/weather/csv/obs --dry-run
"""
import argparse
import csv
import glob
import os
import sys
import tempfile

KEY = ("valid_utc", "target_lead_h")
# How much lower the source's cell fill may be than the target's before the merge
# is refused. Small but non-zero: a model genuinely retiring one field should not
# block an otherwise good re-fetch.
FILL_TOLERANCE = 0.01


def model_columns(header, models):
    """Columns owned by `models`: the lead/init pair plus every suffixed field."""
    cols = []
    for m in models:
        cols += [f"lead_{m}", f"meta_init_{m}"]
    cols += [c for c in header
             if any(c.endswith(f"_{m}") for m in models)
             and not c.startswith(("lead_", "meta_init_"))]
    # Preserve the target's own ordering and drop anything it does not carry.
    seen = set(cols)
    return [c for c in header if c in seen]


def merge_shard(src_path, dst_path, models, min_coverage, dry_run,
                accept_empty=()):
    with open(dst_path, newline="") as fh:
        r = csv.DictReader(fh)
        header = list(r.fieldnames)
        target = list(r)
    with open(src_path, newline="") as fh:
        source = {tuple(row[k] for k in KEY): row for row in csv.DictReader(fh)}

    cols = model_columns(header, models)
    if not cols:
        raise SystemExit(f"no columns for {models} in {dst_path}")

    # Cell-level fill, not just row presence. `coverage` below counts rows the
    # source HAS; it says nothing about whether the cells in those rows carry
    # values. A throttled re-fetch produces a source with every row present and a
    # fraction of the cells empty, which sails past the row check and then blanks
    # good data, because the splice below is an unconditional assignment.
    value_cols = [c for c in cols if not c.startswith(("lead_", "meta_"))]

    def filled(rows):
        if not rows or not value_cols:
            return 0.0
        n = sum(1 for row in rows for c in value_cols
                if (row.get(c) or "") not in ("", "nan"))
        return n / (len(rows) * len(value_cols))

    def filled_by_col(rows):
        """Per-column fill. An AGGREGATE rate hides the case that matters: one
        field collapsing to zero while the others gain enough to cover for it."""
        if not rows:
            return {c: 0.0 for c in value_cols}
        return {c: sum(1 for row in rows
                       if (row.get(c) or "") not in ("", "nan")) / len(rows)
                for c in value_cols}

    # A lead-scoped re-fetch (``--leads 48``) writes a source holding ONLY those
    # leads. Its rows can never cover more than that slice of a shard that now
    # carries the whole ladder, so judging coverage against every target row
    # refuses a perfectly correct merge -- one lead out of seven reads as 14.3%.
    # Scope both guards to the leads the source actually carries: rows at other
    # leads are deliberately left alone, not missing. The fill comparison has to
    # be scoped for the same reason -- source fill measured over lead-48 rows
    # against target fill measured over all seven leads compares two different
    # populations and can refuse or pass for no honest reason.
    lead_key = KEY[1]
    src_leads = {row[lead_key] for row in source.values()}
    if not src_leads:
        raise SystemExit(f"{os.path.basename(src_path)}: source has no rows")
    in_scope = [row for row in target if row[lead_key] in src_leads]
    if not in_scope:
        raise SystemExit(
            f"{os.path.basename(dst_path)}: source carries leads "
            f"{sorted(src_leads)} but the target has none of them. Refusing.")

    # Both snapshots must be taken BEFORE the splice loop below, which assigns
    # source values into the target rows in place -- measured afterwards, the
    # "target" fill is really the source's own and every guard here passes
    # trivially.
    target_filled = filled(in_scope)
    t_by = filled_by_col(in_scope)

    updated = missing = out_of_scope = 0
    conflicts = []
    for row in target:
        if row[lead_key] not in src_leads:
            out_of_scope += 1
            continue
        k = tuple(row[key] for key in KEY)
        src = source.get(k)
        if src is None:
            missing += 1
            continue
        # The meta block must describe the same observation in both files.
        for check in ("split", "is_undercast"):
            if check in src and src[check] != row[check]:
                conflicts.append((k, check, row[check], src[check]))
        for c in cols:
            if c in src:
                row[c] = src[c]
        updated += 1

    coverage = updated / len(in_scope) if in_scope else 0.0
    if conflicts:
        for k, c, a, b in conflicts[:5]:
            print(f"    CONFLICT {k} {c}: target={a!r} source={b!r}")
        raise SystemExit(
            f"{os.path.basename(dst_path)}: {len(conflicts)} rows disagree on the "
            f"meta block -- the sample was regenerated between the two runs, so "
            f"these files no longer describe the same observations. Refusing.")
    if coverage < min_coverage:
        raise SystemExit(
            f"{os.path.basename(dst_path)}: source covers only {100*coverage:.1f}% "
            f"of the {len(in_scope)} target rows at leads {sorted(src_leads)} "
            f"(need {100*min_coverage:.1f}%). Merging would leave {missing} of "
            f"them carrying the OLD values while the rest are new, which is "
            f"worse than either file alone. Refusing.")

    # A re-fetch should fill at least as many cells as it replaces. Less means the
    # download was throttled, not that the model stopped publishing -- and the
    # splice would delete the difference. Deliberate blanking is not done here
    # (see prune_lead_substitutions.py), so a drop is always a fault.
    src_rows_in_scope = [source[k] for k in
                         (tuple(row[j] for j in KEY) for row in in_scope)
                         if k in source]
    s_by = filled_by_col(src_rows_in_scope)
    dropped, bad_allow = [], []
    for c in value_cols:
        if c in accept_empty:
            # Allowlisted because the model does not publish the field at these
            # forecast hours at all. Then the source must be EXACTLY empty: a
            # partially filled allowlisted column is a throttled download
            # wearing an exemption, which is the thing this guard exists to stop.
            if s_by[c] > 0.0:
                bad_allow.append((c, t_by[c], s_by[c]))
        elif s_by[c] + 1e-9 < t_by[c] - FILL_TOLERANCE:
            dropped.append((c, t_by[c], s_by[c]))
    if bad_allow:
        for c, t, sc in bad_allow:
            print(f"    {c}: target {100*t:.1f}% -> source {100*sc:.1f}%")
        raise SystemExit(
            f"{os.path.basename(dst_path)}: {len(bad_allow)} column(s) were "
            f"allowed to come back empty but came back PARTIALLY filled. That is "
            f"a throttled download, not an unpublished field. Refusing.")
    if dropped:
        for c, t, sc in sorted(dropped, key=lambda x: x[1] - x[2], reverse=True)[:10]:
            print(f"    {c}: target {100*t:.1f}% -> source {100*sc:.1f}% "
                  f"({100*(sc - t):+.1f})")
        raise SystemExit(
            f"{os.path.basename(dst_path)}: {len(dropped)} of {len(value_cols)} "
            f"{models} columns come back with FEWER cells filled than the target "
            f"already has. Splicing would BLANK the difference, which is data "
            f"loss, not a re-fetch. Almost always a throttled download -- re-run "
            f"the partial fetch with --resume until the fill rates match. If a "
            f"field genuinely is not published at these forecast hours, name it "
            f"in --accept-empty-columns and it must then come back EXACTLY empty. "
            f"Refusing.")

    print(f"    leads {sorted(src_leads)}: {len(in_scope)} of {len(target)} rows "
          f"in scope, {out_of_scope} left untouched")
    print(f"    cell fill for {models}: target {100*target_filled:.1f}% -> "
          f"source {100*filled(src_rows_in_scope):.1f}%"
          + (f" ({len(accept_empty)} column(s) allowed empty)"
             if accept_empty else ""))
    if not dry_run:
        d = os.path.dirname(dst_path) or "."
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
                w.writeheader()
                w.writerows(target)
            os.replace(tmp, dst_path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
    return updated, missing, len(target)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--target-dir", default="files/weather/csv/obs")
    p.add_argument("--min-coverage", type=float, default=0.99)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--accept-empty-columns", nargs="*", default=[],
                   metavar="COL",
                   help="Columns the model does not publish at the re-fetched "
                        "forecast hours, so they are expected to come back "
                        "empty. Each named column must come back EXACTLY empty; "
                        "partially filled means a throttled download and is "
                        "still refused. Every column NOT named here is held to "
                        "the usual rule that a re-fetch may not lose cells.")
    a = p.parse_args()

    targets = sorted(glob.glob(os.path.join(a.target_dir, "nwp_obs_shard_*.csv")))
    if not targets:
        raise SystemExit(f"no shards in {a.target_dir}")

    pairs = []
    for t in targets:
        s = os.path.join(a.source_dir, os.path.basename(t))
        if not os.path.exists(s):
            raise SystemExit(
                f"missing {s}. Every target shard needs its counterpart, or the "
                f"merge would update some shards and silently leave others on "
                f"the old values. Re-run the fetch for the missing shards first.")
        pairs.append((s, t))

    print(f"merging {a.models} from {a.source_dir} into {a.target_dir}"
          f"{' (DRY RUN)' if a.dry_run else ''}")
    tot_u = tot_m = tot_r = 0
    for s, t in pairs:
        u, m, n = merge_shard(s, t, a.models, a.min_coverage, a.dry_run,
                              accept_empty=set(a.accept_empty_columns))
        tot_u += u
        tot_m += m
        tot_r += n
    print(f"  {len(pairs)} shards | {tot_u:,} rows updated | {tot_m:,} rows had no "
          f"source row (left unchanged) | {tot_r:,} rows total")
    if a.dry_run:
        print("  dry run -- nothing written")
    else:
        print(f"  rewrote {len(pairs)} shards in {a.target_dir}")


if __name__ == "__main__":
    main()
