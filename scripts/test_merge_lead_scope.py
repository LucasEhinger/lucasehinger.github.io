#!/usr/bin/env python3
"""The splice guards must be lead-scoped, and must still catch a throttled fetch.

A one-lead re-fetch into shards that carry the whole ladder can only ever supply
1/7 of the rows. The coverage guard used to measure the source against EVERY
target row, so it refused the correct merge at 14.3%. Scoping fixes that -- but
the guard it shares the file with (fill must not drop) has to keep working, or
the fix would trade a false refusal for silent data loss.

Five properties, on synthetic shards so the real ones are never touched:
  1. a complete one-lead source merges, and only its lead's rows change
  2. a THROTTLED one-lead source is still refused
  3. a source missing rows WITHIN its own lead is still refused
  4. a field named in --accept-empty-columns may come back empty, and is blanked
     only at the re-fetched lead
  5. a field named there that comes back PARTIALLY filled is still refused, so an
     exemption cannot launder a throttled download
"""
import csv
import os
import subprocess
import sys
import tempfile

SCRIPT = os.path.join(os.path.dirname(__file__), "merge_nwp_columns.py")
LEADS = ["1", "24", "48", "72", "96", "120", "144"]
N_PER_LEAD = 40
COLS = ["valid_utc", "target_lead_h", "split", "is_undercast",
        "lead_nbm", "meta_init_nbm", "tmp_2m_nbm", "ceil_nbm",
        "lead_gfs", "meta_init_gfs", "tmp_2m_gfs"]


def target_rows():
    out = []
    for lead in LEADS:
        for i in range(N_PER_LEAD):
            out.append({
                "valid_utc": f"2020-01-{1 + i // 24:02d}T{i % 24:02d}:00:00Z",
                "target_lead_h": lead,
                "split": "train", "is_undercast": "0",
                # The OLD, off-grid NBM values the re-fetch is meant to replace.
                "lead_nbm": "36", "meta_init_nbm": "2020-01-01T00",
                "tmp_2m_nbm": "270.0", "ceil_nbm": "1500.0",
                "lead_gfs": lead, "meta_init_gfs": "2020-01-01T00",
                "tmp_2m_gfs": "271.0",
            })
    return out


def write(path, rows, cols=COLS):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def source_for(rows, lead, throttle_cells=0.0, drop_rows=0):
    """A narrow NBM source covering one lead, optionally degraded."""
    src = []
    for row in rows:
        if row["target_lead_h"] != lead:
            continue
        src.append({
            "valid_utc": row["valid_utc"], "target_lead_h": lead,
            "split": row["split"], "is_undercast": row["is_undercast"],
            "lead_nbm": "47", "meta_init_nbm": "2020-01-01T01",
            "tmp_2m_nbm": "272.5", "ceil_nbm": "1600.0",
        })
    if drop_rows:
        src = src[:-drop_rows]
    if throttle_cells:
        n = int(len(src) * throttle_cells)
        for row in src[:n]:
            row["tmp_2m_nbm"] = ""
            row["ceil_nbm"] = ""
    return src


def run(src_rows, tgt_rows, extra=()):
    d = tempfile.mkdtemp()
    sd, td = os.path.join(d, "src"), os.path.join(d, "tgt")
    os.makedirs(sd); os.makedirs(td)
    write(os.path.join(td, "nwp_obs_shard_000.csv"), tgt_rows)
    write(os.path.join(sd, "nwp_obs_shard_000.csv"), src_rows,
          ["valid_utc", "target_lead_h", "split", "is_undercast",
           "lead_nbm", "meta_init_nbm", "tmp_2m_nbm", "ceil_nbm"])
    cp = subprocess.run(
        [sys.executable, SCRIPT, "--models", "nbm",
         "--source-dir", sd, "--target-dir", td, *extra],
        capture_output=True, text=True)
    after = list(csv.DictReader(open(os.path.join(td, "nwp_obs_shard_000.csv"))))
    return cp, after


def main():
    tgt = target_rows()

    # --- 1. a complete one-lead source must MERGE ---------------------------
    cp, after = run(source_for(tgt, "48"), tgt)
    print("1. complete one-lead source")
    print("   exit", cp.returncode)
    for line in (cp.stdout + cp.stderr).strip().splitlines():
        print("  ", line)
    if cp.returncode != 0:
        raise SystemExit("FAIL: a complete one-lead re-fetch was refused")

    changed = [r for r in after if r["lead_nbm"] == "47"]
    untouched = [r for r in after if r["lead_nbm"] == "36"]
    if len(changed) != N_PER_LEAD:
        raise SystemExit(f"FAIL: {len(changed)} rows updated, want {N_PER_LEAD}")
    if {r["target_lead_h"] for r in changed} != {"48"}:
        raise SystemExit("FAIL: rows outside the source's lead were modified")
    if len(untouched) != N_PER_LEAD * (len(LEADS) - 1):
        raise SystemExit("FAIL: wrong number of untouched rows")
    # Other models must be byte-identical.
    if any(r["tmp_2m_gfs"] != "271.0" for r in after):
        raise SystemExit("FAIL: a non-target model's column changed")
    print(f"   PASS: {len(changed)} lead-48 rows updated, "
          f"{len(untouched)} rows at other leads untouched, gfs unchanged")

    # --- 2. a THROTTLED one-lead source must still be REFUSED --------------
    cp, after = run(source_for(tgt, "48", throttle_cells=0.5), tgt)
    print("\n2. throttled one-lead source (half the cells empty)")
    print("   exit", cp.returncode)
    msg = (cp.stdout + cp.stderr)
    if cp.returncode == 0:
        raise SystemExit("FAIL: a throttled source was accepted -- data loss")
    if "BLANK" not in msg:
        raise SystemExit(f"FAIL: refused for the wrong reason:\n{msg}")
    if any(r["tmp_2m_nbm"] == "" for r in after):
        raise SystemExit("FAIL: cells were blanked despite the refusal")
    print("   PASS: refused by the fill guard, target left intact")

    # --- 3. missing rows WITHIN the source's own lead must be REFUSED ------
    cp, _ = run(source_for(tgt, "48", drop_rows=12), tgt)
    print("\n3. one-lead source missing 12 of its own 40 rows")
    print("   exit", cp.returncode)
    msg = (cp.stdout + cp.stderr)
    if cp.returncode == 0:
        raise SystemExit("FAIL: an incomplete in-scope source was accepted")
    if "covers only" not in msg:
        raise SystemExit(f"FAIL: refused for the wrong reason:\n{msg}")
    print("   PASS: refused by the coverage guard, scoped to lead 48")

    # --- 4. an allowlisted field that is genuinely absent must MERGE -------
    # ceil_nbm stands in for the five NBM fields that exist at short forecast
    # hours but not at the 3-hourly ones a 48 h lead now resolves to.
    src = source_for(tgt, "48")
    for row in src:
        row["ceil_nbm"] = ""
    cp, after = run(src, tgt, extra=("--accept-empty-columns", "ceil_nbm"))
    print("\n4. allowlisted column comes back EXACTLY empty")
    print("   exit", cp.returncode)
    if cp.returncode != 0:
        raise SystemExit(f"FAIL: an allowlisted absent field was refused:\n"
                         f"{cp.stdout + cp.stderr}")
    lead48 = [r for r in after if r["target_lead_h"] == "48"]
    if any(r["ceil_nbm"] != "" for r in lead48):
        raise SystemExit("FAIL: the allowlisted column was not blanked at lead 48")
    if any(r["tmp_2m_nbm"] != "272.5" for r in lead48):
        raise SystemExit("FAIL: the other columns were not spliced")
    other = [r for r in after if r["target_lead_h"] != "48"]
    if any(r["ceil_nbm"] != "1500.0" for r in other):
        raise SystemExit("FAIL: the allowlist leaked outside the source's lead")
    print("   PASS: blanked only at lead 48, other leads keep their values, "
          "other columns spliced")

    # --- 5. an allowlisted field that comes back PARTIAL must be REFUSED ----
    src = source_for(tgt, "48")
    for row in src[:20]:
        row["ceil_nbm"] = ""
    cp, after = run(src, tgt, extra=("--accept-empty-columns", "ceil_nbm"))
    print("\n5. allowlisted column comes back PARTIALLY filled")
    print("   exit", cp.returncode)
    msg = cp.stdout + cp.stderr
    if cp.returncode == 0:
        raise SystemExit("FAIL: a partially filled allowlisted column was accepted "
                         "-- an exemption must not launder a throttled download")
    if "PARTIALLY" not in msg:
        raise SystemExit(f"FAIL: refused for the wrong reason:\n{msg}")
    print("   PASS: refused -- an exemption does not launder a throttled download")

    print("\nPASS: lead-scoped coverage accepts a correct partial re-fetch, the "
          "per-column fill guard still refuses a throttled or incomplete one, and "
          "an allowlisted field must come back exactly empty")


if __name__ == "__main__":
    main()
