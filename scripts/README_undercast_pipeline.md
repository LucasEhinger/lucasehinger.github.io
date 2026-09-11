# Undercast pipeline

Two generations of this pipeline exist side by side. The **old** one is what the
live site currently serves; the **new** one is built but not yet deployed. They
must be swapped over together — see the cutover checklist.

## Old (deployed)

```
files/weather/csv/ML/all/*.csv          589 hand-labeled DATES, 24 positives
  └─ scripts/weather_to_csv.py          Herbie init pinned to 00Z, fxx 1-8
  └─ scripts/train_undercast_models.py  median imputation, date-grouped splits
  └─ files/weather/models/*.pkl         read by weather_to_json.py every 6 h
```

Two defects that motivated the rewrite:

- **Wrong time of day.** The init was pinned to 00:00 UTC with `fxx` 1–8, so the
  features are valid 01–08 UTC = 8 pm–3 am local — the night *before* the noon
  webcam image each row was labeled against.
- **24 positives.** Not enough to fit ~140 features, whatever the algorithm.

## New (built, not deployed)

```
scripts/fetch_iem_metar.py          raw KMWN METAR + RMK, 1997-present, cached
scripts/build_undercast_record.py   apply the undercast screen -> 253,317 obs
                                    (6,843 undercast, 1997-2026)
scripts/sample_undercast_obs.py     all positives + 5:1 negatives, stratified on
                                    each positive's own (year, month, hour)
.github/workflows/fetch_nwp_obs.yml forecast fields at each observation's OWN
                                    valid time, 3 leads (~1/24/48 h), 120 shards
scripts/train_undercast_obs.py      -> files/weather/models/obs/
```

Why each piece is the way it is, where it is not obvious:

- **Label source is IEM, not NOAA ISD.** ISD only carries the full METAR body
  from 1999-10, truncates remark text (losing whole `TPS LWR` groups), and its
  KMWN series stops 2025-08-24. IEM is complete and current.
- **The record starts 1997.** 1995–96 use a spaced remark variant that often
  omits the deck height, which the depth cut needs.
- **The screen** is `BKN/OVC below · vis > 40 SM · lid open-or-≥5,000 ft · deck
  ≥ 500 ft down · lowest layer above ≥ 1,000 ft`. Precision 0.70 / recall 0.67
  against all 589 hand-labeled days. It is a **proxy**: ~30% of its positives are
  something else, so models trained on it are scored against the hand labels.
- **Negatives are stratified on (year, month, hour).** The reported undercast
  rate climbs from 1.2% (1997) to 5.4% (2024) as observers adopted the remark,
  and positives peak at 10–14 UTC. Uniform negatives would let a model score
  well off the calendar and the clock.
- **Ambiguous observations are flagged, not dropped.** They leave *training*
  (calling them negative puts the worst label noise on the boundary) but stay in
  both holdouts: the webcam holdout is scored against human labels, where the
  screen's uncertainty is irrelevant. Measured: all 125 fall in the holdouts and
  none in `train`, so the sampler had already excluded them — the flag is a
  guard, not a change.
- **`nan` means clear sky.** GRIB omits cloud ceiling/base/top where there is no
  cloud. Measured: a `nan` HRRR ceiling goes with 0.0% median low-cloud cover, a
  numeric one with 39.9%. Shard CSVs are read with `keep_default_na=False` so
  `""` (never fetched) stays distinguishable from `"nan"` (model reports no
  cloud), and the cloud-geometry columns get an explicit `_no_cloud` flag.
  HRRR and RAP say "no cloud" with `nan`; NAM and GFS instead emit ~20000 m and
  NBM 88892 m. Verified against the observations rather than assumed — median
  OBSERVED visibility in the sentinel group is 96.6 km (NAM) and 128.7 km (NBM)
  against 99.8 m for rows carrying a real ceiling — so those are folded into the
  same flag.
- **Inversion strength is supplied explicitly.** An undercast *is* an inversion
  with the summit above the deck, and that is a DIFFERENCE between two columns,
  which a tree can only approximate through many axis-aligned splits.
  `T850 - T925` alone scores AUC 0.698 (HRRR), beating every raw feature present.
  Adding the profile block lifted held-out AUC on every source tested — HRRR
  webcam 0.844 -> 0.882, `all` webcam 0.901 -> 0.917. Summit temperature is
  interpolated from pressure levels, never `tmp_2m`, which sits at the model's
  smoothed terrain height rather than the real 1,917 m.
- **Lead is not a feature.** `lead_hrrr` is `{1,2,15}` for 2014–16 rows and
  `{1,23,47}` for 2021+ (HRRR's max forecast hour grew with its versions), so it
  is a proxy for the year and would smuggle the drift confound back in.
  Skill-versus-lead is measured by *grouping* the holdouts instead.
- **Folds group by ISO week**, because a multi-day inversion correlates
  consecutive days.

### Holdouts

| split | what it is | used for |
|---|---|---|
| `train` | stratified sample, ~17% positive | fitting |
| `holdout_baserate` | one full year (2022), 3-hourly, unsampled, true 2.7% rate | threshold tuning (global and per-lead) |
| `holdout_webcam` | the hand-labeled noon days | **headline metrics, scored against HUMAN labels** |

Precision measured on `train` is meaningless — negatives were subsampled 5:1, so
it is computed against a base rate that does not exist in the world.

**Splits separate by DATE, not by observation.** Each observation is one hour and
a day holds ~24 of them, so the original per-observation split held out nothing:
535 of the 589 webcam dates still carried training observations and every webcam
holdout week overlapped a training week, meaning a model could learn a day's
pattern from 02:50 and be "tested" on 03:50. `assign_splits()` drops any training
row sharing a date with a holdout, plus a one-day buffer for multi-day inversions.
Costs 17% of training observations; in exchange the holdouts become contiguous
temporal blocks, so the webcam numbers are a genuine forward test.

Training rows per source after separation (x3 for the lead rows):

| source | train obs | positives | window |
|---|---|---|---|
| hrrr | 17,511 | 2,945 | 2014-08 → 2026-09 |
| nam | 7,607 | 1,343 | 2020-05 → |
| nbm | 7,204 | 1,276 | 2020-10 → |
| gfs | 5,665 | 994 | 2021-03 → |
| rap | 5,569 | 978 | 2021-06 → |
| ecmwf / all | 4,436 | 797 | 2023-01 → |

(Counts are observations; the trainer works on 3x these as one row per lead.)

### Results, 2026-09-11

Base-rate holdout AUC by lead (XGBoost), and webcam-human precision/recall:

| source | 1 h | 24 h | 48 h | webcam P / R |
|---|---|---|---|---|
| **all** | **0.947** | **0.930** | **0.877** | **0.47 / 0.24** |
| hrrr | 0.894 | 0.872 | 0.832 | 0.24 / 0.49 |
| rap | 0.870 | 0.836 | 0.783 | 0.18 / 0.49 |
| nbm | 0.878 | 0.817 | 0.820 | 0.30 / 0.36 |
| ecmwf | 0.874 | 0.865 | 0.810 | 0.31 / 0.26 |
| nam | 0.845 | 0.820 | 0.805 | 0.21 / 0.22 |
| gfs | 0.527 | 0.515 | 0.555 | 0.04 / 0.14 |

Two things this settles:

- **Skill does not collapse with lead.** 0.947 -> 0.877 from 1 h to 48 h for
  `all`. The models both resolve inversions and largely forecast them, which is
  the opposite of what the page currently asserts.
- **Six models beat one model with 4x the data.** `all` trains on 4,435
  observations against HRRR's 17,511 and still wins decisively (base-rate AUC
  0.921 vs 0.866, webcam precision 0.47 vs 0.24). Model disagreement is
  informative.

Precision is the honest weak spot: ~0.45 at the true 2.7% base rate for `all`,
so roughly half of alerts are wrong. The screen's own precision of 0.70 caps
what is reachable, since ~30% of training positives are not undercast.

### GFS was sampled 5,400 km away

GFS publishes longitude on 0..360. Mount Washington is -71.30, and
`da.sel(longitude=-71.30, method="nearest")` does not raise — it clamps to the
nearest in-range value, 0.0 — so every GFS column came from **44.25°N, 0°E, in
southwestern France**. Detected by cross-model agreement on 850 mb temperature,
a field every model should agree on: GFS correlated 0.596 with HRRR at +6.8 K
bias, where NAM, RAP and ECMWF all correlate 0.99+.

`sample_nearest` in `weather_to_csv.py` now converts longitude into the grid's
own convention; after the fix all four models agree within 0.5 K. **The GFS
columns in the current shard CSVs are still wrong** — fixing them needs a
re-fetch. Removing them outright improves `all` (OOF 0.899 -> 0.904, webcam AUC
0.917 -> 0.930), so the ensemble was already discounting the noise.

`weather_to_json.py` holds a SECOND copy of the same buggy function. It is
deliberately left alone: the deployed models were trained on the France values,
so correcting the input without retraining would feed them out-of-distribution
data. Fix it as part of the cutover, never before.

### Known weakness: the base-rate holdout is thin

2,896 observations carrying only **86 positives**, so ~29 per lead. Enough to see
a skill-versus-lead trend, not enough to pin a per-lead threshold precisely — the
trainer falls back to the global threshold below 25 positives in a lead group and
marks it with `*`.

It is also why `ecmwf`/`all` start in 2023: holding out calendar 2022 costs ECMWF
the 10 months between its archive start (2022-03) and the end of that year.
The better design is a base-rate holdout made of weeks spread across the whole
record — more positives, every model's window represented, less cost to any one
source. That needs a second download pass, because an unbiased holdout requires
every observation in the held-out weeks and the current sample only has the
unsampled 3-hourly series for 2022.

## Cutover checklist

1. **Fix the longitude bug in `weather_to_json.py`'s own copy of
   `sample_nearest`** (see above) — must land in the same commit as the
   retrained models, never before them.
2. Decide whether GFS is re-fetched or dropped. As it stands its model is
   worthless (AUC 0.53) and must not be served.
3. Decide how the RandomForest artifacts are stored. At `min_samples_leaf=20`
   HRRR is 43 MB against the deployed 3.7 MB; XGBoost is 1.1 MB for equal or
   better skill, so serving XGBoost alone is a live option.


Nothing below can land on its own: the live `weather_to_json.py` runs every 6
hours against the deployed preprocessors, which were fitted on the **old** column
names. Changing one side alone breaks the live page on a missing column.

- [ ] **Mirror the variable-list fixes** into `weather_to_json.py` (already done
      in `weather_to_csv.py`, commit `acfab4b`):
      - rename `boundary_layer_cloud_layer` → `boundary_layer_cloud_layer_hrrr`
        (without the suffix, `select_features` drops it from every per-source
        model and it reaches only the combined one)
      - `boundary_layer_cloud_layer_gfs` alias → `:TCDC:boundary layer cloud
        layer` (GFS indexes it with no forecast-hour qualifier, so the `%n` form
        matched nothing and the column was 100% empty)
      - drop `hgt_925mb_hrrr` (only in the HRRR `prs` product) and
        `boundary_layer_cloud_layer_nam` (NAM never publishes it)
      - update the explicit `desired_columns` list further down the same file
- [ ] **Resolve each model's run independently at inference.** Today a single
      6-hourly-rounded `date_str` is used for all six models. That aligns valid
      times, but it leaves HRRR/RAP/NBM up to 6 h staler than necessary, since
      they run hourly. Reuse `candidate_runs()` / `snap_valid()` from
      `fetch_nwp_at_obs.py` so inference resolves runs the same way training did.
- [ ] **Use the per-lead thresholds.** `model_metadata_*.json` now carries
      `threshold_by_lead`; a global cut either over-fires at 48 h or under-fires
      at 1 h (measured on the fixture: 0.97 at 1 h vs 0.46 at 48 h).
- [ ] **Point the models directory** at `files/weather/models/obs/` (or copy over
      `files/weather/models/`) once the new numbers beat the old ones.
- [ ] **Update the results section** of `_pages/weather-details.html`. The
      "Determining if it's undercast" write-up is already current; the results,
      per-source table, feature-importance figures and "Next Steps" still
      describe the old models, and the write-up carries an explicit *Status*
      paragraph saying so that should be removed.
- [ ] **Retire the old path** — `regen_weather_csv.yml` and
      `train_undercast_models.py` — only after the new one is serving.
