#!/usr/bin/env python3
"""Download NWP forecast fields valid AT each summit observation time.

Replaces the date-based sampling in ``weather_to_csv.py``, which pinned every
Herbie init to 00:00 UTC of the label date and took fxx 1-8 -- i.e. fields valid
01-08 UTC, which is 8pm-3am local, the night BEFORE the noon webcam image the
row was labeled against. Here the valid time is the observation's own time, and
the (init, fxx) pair is solved for per model.

Three leads are sampled at every observation:

    ~1 h   near-analysis. What the model thought was happening essentially now.
    ~24 h  day-ahead. The forecast a hiker would actually plan on.
    ~48 h  two-day. Where the site's forecast is least trustworthy.

These are targets, not guarantees. NAM and GFS only run four times a day, so
their "~1 h" sample is whatever the 00/06/12/18 cycle can reach -- 1 to 6 hours
depending on the observation's hour. The lead actually achieved is written to
each row as lead_<model>, so nothing downstream has to assume it got 1 h.

Having all three on the SAME event is what separates the two explanations for
the project's poor scores: if skill is decent at 1 h and collapses by 24 h, the
models can resolve the inversion and just cannot predict it; if it is poor at
1 h too, they never resolve it at all. The page currently asserts the latter
without having measured it.

Each model publishes runs on its own cadence and each run has its own maximum
lead, so a nominal 24 h lead resolves to a different (init, fxx) for HRRR (runs
hourly, but only 00/06/12/18 reach past 18 h) than for ECMWF IFS (runs 00/12,
3-hourly steps only). Rather than trust a hand-written table of every model's
quirks, candidates are ranked by how close they land to the target lead and
tried in order until one is actually present in the archive. The lead that was
really used is written to the row, so nothing downstream has to assume.

Output is one gzipped CSV per shard, one row per (observation, lead), with every
model's columns side by side -- the shape ``train_undercast_models.py`` expects.

Usage:
    python3 scripts/fetch_nwp_at_obs.py --sample files/weather/obs/nwp_sample.csv \\
        --output-dir files/weather/csv/obs --num-shards 44 --shard 0 --workers 6
"""
import argparse
import csv
import gzip
import os
import random
import signal
import sys
import tempfile
import time
import warnings
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Importing weather_to_csv also installs its two hang guards: a default socket
# timeout, and an os.system wrapper that adds --max-time to Herbie's curl calls
# (Herbie shells out with no timeout, so a dead socket otherwise blocks a worker
# forever, unkillable).
from weather_to_csv import (  # noqa: E402
    LOCATIONS, sample_nearest, try_load, variables,
)
from herbie import Herbie  # noqa: E402

TARGET_LEADS = (1, 24, 48)

# Herbie's model name for each of our source labels. "ecmwf" columns come from
# the IFS open-data product, which Herbie calls "ifs".
HERBIE_MODEL = {"hrrr": "hrrr", "nam": "nam", "gfs": "gfs",
                "rap": "rap", "nbm": "nbm", "ecmwf": "ifs"}
HERBIE_KWARGS = {"ecmwf": {"product": "oper"}, "nbm": {"product": "co"}}

# First date each archive can actually serve, probed against the live buckets
# rather than taken from documentation (GFS and RAP both start months later than
# Herbie's docstrings claim; ECMWF starts ~10 months EARLIER than its published
# 2023-01-18 open-data launch). A too-early date here costs an index lookup and
# an empty column, not a failure, so these lean slightly generous.
MODEL_START = {
    "hrrr": datetime(2014, 7, 30, tzinfo=timezone.utc),
    "nam": datetime(2020, 5, 1, tzinfo=timezone.utc),
    "nbm": datetime(2020, 10, 1, tzinfo=timezone.utc),
    "gfs": datetime(2021, 3, 23, tzinfo=timezone.utc),
    "rap": datetime(2021, 5, 1, tzinfo=timezone.utc),
    "ecmwf": datetime(2022, 3, 1, tzinfo=timezone.utc),
}

# HRRR's maximum forecast hour grew with each model version, so a 24 h lead
# simply does not exist in the early archive and a 48 h lead does not exist
# before HRRRv4. Probed: 2017 runs stop at 18 h, 2018 reach 36 h, 2021 reach
# 48 h. Without this the pre-2018 rows would burn five index lookups each
# hunting for a file that was never produced.
HRRR_ERAS = (
    (datetime(2016, 8, 23, tzinfo=timezone.utc), 15, 15),
    (datetime(2018, 7, 12, tzinfo=timezone.utc), 18, 18),
    (datetime(2020, 12, 2, tzinfo=timezone.utc), 18, 36),
)


def lead_limits(model, init):
    """(max fxx for an ordinary run, max fxx for an extended run)."""
    spec = RUN_SPECS[model]
    if model == "hrrr":
        for cutoff, normal, long_ in HRRR_ERAS:
            if init < cutoff:
                return normal, long_
    return spec["max"], spec.get("long_max", spec["max"])

# runs: UTC hours that produce a cycle. long_runs reach long_max instead of max.
# step: forecast-hour granularity (IFS open data publishes 3-hourly only).
RUN_SPECS = {
    "hrrr": {"runs": range(24), "max": 18, "long_runs": (0, 6, 12, 18),
             "long_max": 48, "step": 1, "min_fxx": 1},
    "rap": {"runs": range(24), "max": 21, "long_runs": (3, 9, 15, 21),
            "long_max": 51, "step": 1, "min_fxx": 1},
    "nam": {"runs": (0, 6, 12, 18), "max": 84, "step": 1, "min_fxx": 1},
    "gfs": {"runs": (0, 6, 12, 18), "max": 120, "step": 1, "min_fxx": 1},
    "nbm": {"runs": range(24), "max": 36, "long_runs": (0, 6, 12, 18),
            "long_max": 192, "step": 1, "min_fxx": 1},
    # fxx 0 is the IFS analysis, which is the best possible "~1 h" sample.
    "ecmwf": {"runs": (0, 6, 12, 18), "max": 90, "long_runs": (0, 12),
              "long_max": 144, "step": 3, "min_fxx": 0},
}

# Granularity of a model's VALID times, which is a separate thing from its run
# cadence. Every IFS open-data run starts on a multiple of 3 and steps by 3, so
# IFS fields only ever exist at 00, 03, 06 ... UTC -- it can never be valid at
# 17 UTC no matter which run you pick. Observations are hourly, so two thirds of
# them would come back with empty ECMWF columns. Snapping the ECMWF valid time
# to the nearest 3-hourly step costs at most one hour of offset and keeps the
# source usable; the run actually used is recorded per row, so the offset stays
# auditable. Every other model is hourly and snaps to itself.
VALID_STEP = {"ecmwf": 3}


def snap_valid(model, valid):
    """Nearest valid time this model can actually produce (<= 1 h away)."""
    step = VALID_STEP.get(model, 1)
    if step == 1:
        return valid
    off = valid.hour % step
    if off == 0:
        return valid
    return valid + timedelta(hours=(step - off if off * 2 >= step else -off))

MODEL_VARS = {}
for _label, _spec in variables.items():
    _m = _spec["model"]
    _m = "ecmwf" if _m == "ifs" else _m
    MODEL_VARS.setdefault(_m, []).append(_label)


def candidate_runs(model, valid, target_lead, max_tries=5):
    """(init, fxx) pairs for `model` valid at `valid`, closest lead first.

    When the requested lead is beyond what this model/era can produce the list
    walks outward to the closest lead that IS achievable -- a 2015 observation
    asked for 24 h gets HRRR's 15 h ceiling instead of nothing. The row records
    the lead it actually got, so training sees the truth rather than a label
    claiming 24 h.
    """
    spec = RUN_SPECS[model]
    runs = set(spec["runs"])
    long_runs = set(spec.get("long_runs", ()))
    out = []
    for back in range(0, spec.get("long_max", spec["max"]) + 1):
        init = valid - timedelta(hours=back)
        if init.hour not in runs or back < spec["min_fxx"] or back % spec["step"]:
            continue
        normal_max, long_max = lead_limits(model, init)
        if back > (long_max if init.hour in long_runs else normal_max):
            continue
        out.append((init, back))
    # closest to the requested lead; on a tie prefer the fresher (shorter) run
    out.sort(key=lambda p: (abs(p[1] - target_lead), p[1]))
    return out[:max_tries]


def _retry(fn, tries=4, base=1.5):
    """Run fn(), retrying S3 throttling. Returns (result, ok).

    The NOAA Big Data buckets answer "503 Slow Down" under load, and this job
    runs up to 20 CI jobs x several workers against them at once. Without a
    backoff a throttled fetch looks exactly like a missing field, so a whole
    shard can come back plausibly-shaped and empty. Jitter keeps the workers
    from retrying in lockstep.
    """
    for attempt in range(tries):
        try:
            return fn(), True
        except Exception as e:
            msg = str(e)
            throttled = ("Slow Down" in msg or "503" in msg or "429" in msg
                         or "SlowDown" in msg)
            if attempt == tries - 1 or not throttled:
                return None, False
            time.sleep(base * (2 ** attempt) * (0.5 + random.random()))
    return None, False


def open_herbie(model, valid, target_lead, save_dir):
    """First archived run that lands near `target_lead`, or (None, None, None)."""
    for init, fxx in candidate_runs(model, valid, target_lead):
        h, _ = _retry(lambda: Herbie(
            init.strftime("%Y-%m-%d %H:%M"), model=HERBIE_MODEL[model],
            fxx=fxx, save_dir=save_dir, verbose=False,
            **HERBIE_KWARGS.get(model, {})))
        if h is None:
            continue
        # Herbie sets .grib only when it actually located the file in a source.
        if getattr(h, "grib", None) is None:
            continue
        return h, init, fxx
    return None, None, None


def _impl(task):
    valid_utc, target_lead, meta = task
    valid = datetime.strptime(valid_utc, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
    # Observations land at :45-:59; model fields are valid on the hour. Round to
    # the nearest hour -- within ~9 minutes of the observation either way.
    model_valid = (valid + timedelta(minutes=30)).replace(minute=0, second=0)
    loc = LOCATIONS[0]

    row = OrderedDict(meta)
    row["model_valid_utc"] = model_valid.strftime("%Y-%m-%dT%H:%M")
    row["target_lead_h"] = target_lead

    with tempfile.TemporaryDirectory() as tmp:
        for model, labels in MODEL_VARS.items():
            row[f"lead_{model}"] = ""
            row[f"meta_init_{model}"] = ""
            for lab in labels:
                row[lab] = ""
            if model_valid < MODEL_START[model]:
                continue
            mv = snap_valid(model, model_valid)
            h, init, fxx = open_herbie(model, mv, target_lead, tmp)
            if h is None:
                continue
            row[f"lead_{model}"] = fxx
            row[f"meta_init_{model}"] = init.strftime("%Y-%m-%dT%H:%M")
            for lab in labels:
                aliases = [a.replace("%n", str(fxx)) if isinstance(a, str) else a
                           for a in variables[lab]["aliases"]]
                # Check the (cached, already-fetched) GRIB index FIRST. This
                # separates the two reasons a field comes back empty, which
                # otherwise look identical: the model genuinely does not publish
                # it, versus the download was throttled. Only the second is worth
                # retrying, and conflating them is what left earlier runs at 55%
                # fill while looking healthy. Matching here also skips the
                # download attempt for aliases that cannot match at all.
                matched = None
                for a in aliases:
                    try:
                        if len(h.inventory(a)):
                            matched = a
                            break
                    except Exception:
                        continue
                if matched is None:
                    continue            # field absent from this file -- expected
                for attempt in range(3):
                    res, _ = _retry(lambda: try_load([matched], h), tries=1)
                    da = res[0] if res else None
                    if da is not None:
                        break
                    # The field IS in the index, so a failure here is transport.
                    time.sleep(1.5 * (2 ** attempt) * (0.5 + random.random()))
                if da is None:
                    print(f"\n  gave up on {model}/{lab} at {valid_utc}", flush=True)
                    continue
                try:
                    row[lab] = sample_nearest(da, loc["lat"], loc["lon"])
                except Exception:
                    pass
                finally:
                    try:
                        da.close()
                    except Exception:
                        pass
    return row


def process(task):
    """One (observation, lead) with a wall-clock backstop.

    Herbie downloads through requests/urllib3, which ignore the module-level
    socket timeout, so a dropped connection can wedge a worker in recv()
    indefinitely. SIGALRM interrupts the blocked syscall; the task is abandoned
    and --resume picks it up next pass.
    """
    def _timeout(signum, frame):
        raise TimeoutError("per-observation wall-clock timeout")

    have = hasattr(signal, "SIGALRM")
    if have:
        old = signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(900)
    try:
        return _impl(task)
    except TimeoutError:
        print(f"\ntimeout: {task[0]} lead={task[1]}", flush=True)
        return None
    except Exception as e:
        print(f"\nfailed {task[0]} lead={task[1]}: {type(e).__name__}", flush=True)
        return None
    finally:
        if have:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


META_COLS = ["valid_utc", "model_valid_utc", "target_lead_h", "split",
             "is_undercast", "year", "month", "hour_utc"]


def fieldnames():
    cols = list(META_COLS)
    for model in MODEL_VARS:
        cols += [f"lead_{model}", f"meta_init_{model}"]
    for model, labels in MODEL_VARS.items():
        cols += labels
    return cols


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", default="files/weather/obs/nwp_sample.csv")
    p.add_argument("--output-dir", default="files/weather/csv/obs")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--leads", type=int, nargs="+", default=list(TARGET_LEADS))
    p.add_argument("--limit", type=int, default=0, help="first N tasks only (smoke test)")
    p.add_argument("--resume", action="store_true",
                   help="skip (obs, lead) pairs already present in this shard's output")
    a = p.parse_args()
    if not (0 <= a.shard < a.num_shards):
        p.error("--shard must satisfy 0 <= shard < num-shards")

    os.makedirs(a.output_dir, exist_ok=True)
    out_path = os.path.join(a.output_dir, f"nwp_obs_shard_{a.shard:03d}.csv.gz")

    with open(a.sample) as fh:
        obs = list(csv.DictReader(fh))
    obs = [o for i, o in enumerate(obs) if i % a.num_shards == a.shard]

    done = set()
    existing = []
    if a.resume and os.path.exists(out_path):
        with gzip.open(out_path, "rt") as fh:
            for r in csv.DictReader(fh):
                done.add((r["valid_utc"], r["target_lead_h"]))
                existing.append(r)
        print(f"resume: {len(done)} rows already present")

    tasks = []
    for o in obs:
        meta = {
            "valid_utc": o["valid_utc"],
            "split": o["split"],
            "is_undercast": 1 if o["label"] == "undercast" else 0,
            "year": o["year"], "month": o["month"], "hour_utc": o["hour_utc"],
        }
        for lead in a.leads:
            if (o["valid_utc"], str(lead)) in done:
                continue
            tasks.append((o["valid_utc"], lead, meta))
    if a.limit:
        tasks = tasks[:a.limit]
    print(f"shard {a.shard}/{a.num_shards}: {len(obs)} observations, "
          f"{len(tasks)} (obs, lead) tasks to fetch -> {out_path}")
    if not tasks:
        print("nothing to do")
        return

    cols = fieldnames()
    n_ok = 0
    with gzip.open(out_path, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in existing:
            w.writerow(r)
        workers = max(1, min(a.workers, len(tasks)))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(process, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                if row:
                    w.writerow(row)
                    n_ok += 1
                if i % 25 == 0 or i == len(futs):
                    fh.flush()
                    print(f"\r  {i}/{len(futs)} ({100*i//len(futs)}%) ok={n_ok}",
                          end="", flush=True)
    print(f"\nwrote {out_path}: {n_ok + len(existing)} rows")

    # Fill rates per model per lead. A throttled fetch and a genuinely absent
    # field both look like an empty cell, so this is the only signal that a
    # shard came back plausibly-shaped but hollow. Anything far below its
    # neighbours means the run was rate-limited, not that the archive is short.
    with gzip.open(out_path, "rt") as fh:
        rows = list(csv.DictReader(fh))
    print("\nfill rate by model and lead (share of requested cells populated):")
    hdr = "  " + "lead".ljust(6) + "".join(m.rjust(9) for m in MODEL_VARS)
    print(hdr)
    for lead in sorted({r["target_lead_h"] for r in rows}, key=int):
        sub = [r for r in rows if r["target_lead_h"] == lead]
        cells = []
        for model, labels in MODEL_VARS.items():
            want = [r for r in sub
                    if r["model_valid_utc"] >= MODEL_START[model].strftime("%Y-%m-%dT%H:%M")]
            if not want:
                cells.append("-".rjust(9))
                continue
            filled = sum(1 for r in want for lab in labels if r[lab] not in ("", "nan"))
            cells.append(f"{100*filled/(len(want)*len(labels)):.0f}%".rjust(9))
        print("  " + lead.ljust(6) + "".join(cells))


if __name__ == "__main__":
    main()
