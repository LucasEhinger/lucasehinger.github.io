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
- **Ambiguous observations are dropped, not called negative** — scattered decks
  and narrow near-misses, where the screen is least reliable.
- **`nan` means clear sky.** GRIB omits cloud ceiling/base/top where there is no
  cloud. Measured: a `nan` HRRR ceiling goes with 0.0% median low-cloud cover, a
  numeric one with 39.9%. Shard CSVs are read with `keep_default_na=False` so
  `""` (never fetched) stays distinguishable from `"nan"` (model reports no
  cloud), and the cloud-geometry columns get an explicit `_no_cloud` flag.
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
| `holdout_baserate` | one full year, 3-hourly, unsampled, true 2.7% rate | threshold tuning (global and per-lead) |
| `holdout_webcam` | the hand-labeled noon days | **headline metrics, scored against HUMAN labels** |

Precision measured on `train` is meaningless — negatives were subsampled 5:1, so
it is computed against a base rate that does not exist in the world.

## Cutover checklist

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
