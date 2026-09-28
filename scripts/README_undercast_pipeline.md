# Undercast pipeline

Forecasts whether Mount Washington's summit will sit above a cloud deck (an
*undercast*), out to six days, from six weather models. The forecast is shown on
[/weather/](https://lucasehinger.github.io/weather/) and the method is written up
on [/weather/details/](https://lucasehinger.github.io/weather/details/).

In one sentence: the summit observers' hourly METAR remarks are turned into an
undercast record, forecast fields from six models are downloaded at each sampled
observation's own time, a gradient-boosted classifier learns one from the other,
and it is scored against hand-labelled webcam days it never saw.

## At a glance

```
LABELS    fetch_iem_metar.py ──► build_undercast_record.py ──► undercast_record.csv
          (KMWN METAR + remarks,     (the cuts; 253,317 obs,        (committed)
           1997–present, IEM)          6,852 undercast)
                                              │
SAMPLE                              sample_undercast_obs.py ──► nwp_sample.csv
                                    (all positives + 5:1 negatives, (committed)
                                     two untouched holdouts)
                                              │
FORECASTS                 .github/workflows/fetch_nwp_obs.yml
                          └─ fetch_nwp_at_obs.py ──► files/weather/csv/obs/*.csv
                             (6 models × 7 leads,      (120 shards, committed)
                              ~12 h on GitHub Actions)
                                              │
MODEL                               train_undercast_obs.py ──► files/weather/models/obs/
                                                                (metadata + the served
                                                                 model are committed)
                                              │
FIGURES   plot_*.py, export_*.py, compare_undercast_ensembles.py, ... ──► /weather/details/
SERVING   .github/workflows/get_weather.yml (every 6 h)
          └─ weather_to_json.py ──► files/weather/predictions_all.json["current"] ──► /weather/
```

Webcam side, which only *evaluates* the model and illustrates the pages:

```
label_webcam_frames.py ──► files/weather/webcam/  (dated stills; gitignored, ~1 GB)
hand scores            ──► files/weather/csv/MtWashington_undercast_orig.csv
build_observation_gallery.py, build_screen_cut_gallery.py ──► /weather/observations/, /weather/cuts/
```

## Running it from start to finish

From the repository root. Every step below was run end to end on 2026-09-28
except the GRIB download (step 3), which reuses the committed shards.

```bash
# 1. Labels: download the METAR archive (~30 min), then apply the cuts.
python3 scripts/fetch_iem_metar.py --cache ~/metar_cache
python3 scripts/build_undercast_record.py --cache ~/metar_cache
#    -> files/weather/obs/undercast_record.csv

# 2. Sample: which observation times to download forecasts for.
python3 scripts/sample_undercast_obs.py
#    -> files/weather/obs/nwp_sample.csv   (see "Do not regenerate lightly")

# 3. Forecast fields at each sampled time. Run the "Fetch NWP at Observation
#    Times" workflow on GitHub (120 shards, ~12 h). Locally, one shard:
python3 scripts/fetch_nwp_at_obs.py --resume --num-shards 120 --shard 0
python3 scripts/prune_lead_substitutions.py --apply   # after any fetch
python3 scripts/audit_nwp_leads.py                    # and check it

# 4. Train (all six sources plus the combined model, three algorithms each; ~45 min).
python3 scripts/train_undercast_obs.py
#    -> files/weather/models/obs/

# 5. Figures and the numbers quoted on /weather/details/.
python3 scripts/plot_undercast_obs_models.py
python3 scripts/compare_undercast_ensembles.py
python3 scripts/export_feature_importances.py
python3 scripts/export_confusion_by_lead.py
python3 scripts/undercast_redundancy.py
python3 scripts/undercast_capacity.py         # the slow one: ~1 h of retraining
python3 scripts/headline_choice_ci.py
python3 scripts/plot_undercast_record.py
python3 scripts/ablate_undercast_screen.py
python3 scripts/plot_inversion_evidence.py
python3 scripts/baseline_undercast_rules.py

# 6. Galleries (need the webcam stills in files/weather/webcam/).
python3 scripts/build_observation_gallery.py --metar-cache ~/metar_cache
python3 scripts/build_screen_cut_gallery.py

# 7. One live forecast, exactly as the 6-hourly workflow runs it (~6 min).
python3 scripts/weather_to_json.py

# 8. Checks. All run from a fresh checkout with no arguments.
for t in scripts/test_*.py;  do python3 "$t" || echo "FAIL $t"; done
for t in scripts/test_*.mjs; do node "$t"    || echo "FAIL $t"; done
```

Python dependencies: `herbie-data numpy xarray pandas scikit-learn==1.4.2
xgboost joblib matplotlib pillow`. The served model is pickled with
scikit-learn 1.4.2 and must be loaded with the same version.

### Do not regenerate lightly

- **`nwp_sample.csv` is what the shards were fetched for.** Re-running the
  sampler today draws a different set of *negatives* (it now drops observations
  that share a date with a holdout before drawing, where the committed sample
  was drawn first and filtered at training time). The positives and the
  effective training set are unchanged, but a new sample means a new fetch.
  Never regenerate it while a fetch workflow is in flight: shards read the sample
  from their own checkout, and a mid-run change duplicates and drops rows.
- **`undercast_record.csv` is frozen at 2026-09-10.** Rebuilding it reproduces
  every committed row exactly (verified: 253,317 of 253,317 identical) and adds
  the reports filed since. Everything on the site is quoted from the committed
  version, so a rebuild changes those numbers.
- **Hand scores are edited by hand.** `MtWashington_undercast_orig.csv` is the
  one input no script produces. Changing a score changes the webcam evaluation
  and which days the observation viewer selects, and nothing else: the model is
  trained on the METAR record, not on the webcam days.

## Design decisions

Why each piece is the way it is, where that is not obvious.

**Labels come from the observers, not the webcam.** Hand-labelling a year of
webcam stills yields 589 days and 22 undercasts, too few to fit anything. The
observers' remarks yield 253,317 hourly observations and 6,852 undercasts. The
catch is that the remarks are a *proxy*: at the cuts' precision of about 0.70,
roughly three in ten training positives are not undercasts. So the model trains
on the remarks and is scored against the webcam.

**IEM, not NOAA ISD.** ISD only carries the full METAR body from 1999-10,
truncates remark text (losing whole `TPS LWR` groups), and its KMWN series stops
at 2025-08-24. IEM is complete and current. The record starts in 1997 because
1995–96 use a remark variant that often omits the deck height.

**The cuts** (`build_undercast_record.py`): a BKN/OVC deck below the summit,
visibility above 40 SM, no lid overhead below 5,000 ft, the deck top at least
500 ft below the summit, and any layer overhead at least 1,000 ft up.
Observations that miss narrowly (a SCT deck, or a looser bound on each test) are
flagged *ambiguous* and left out of training; they stay in both holdouts, where
they are scored against people rather than against the cuts.
`/weather/observations/` states the same values, and `test_observation_cuts.py`
fails if the page and the code disagree.

**Negatives are stratified on (year, month, hour).** The reported undercast rate
rises from 1.2% (1997) to 5.4% (2024) as observers adopted the remark, and
positives peak at 10–14 UTC. Drawing each positive's negatives from its own
(year, month, hour) cell matches all three out, so the forecast fields have to do
the work.

**Forecast fields are sampled at the observation's own time.** For each model the
(init, forecast hour) pair is solved per observation, at seven leads:
1, 24, 48, 72, 96, 120 and 144 h. Beyond 48 h the short-range models cannot
reach, so those rows carry a subset of the sources. That is deliberate: it is
exactly the situation the live page meets whenever one model publishes late, and
training on it teaches the model to cope rather than extrapolate.

**The lead a row claims is the lead it has.** `fetch_nwp_at_obs.py` walks to the
nearest forecast hour a model actually publishes, bounded by `LEAD_SLACK_H`
(12 h, measured) and by each model's maximum reach, so a model's 48 h ceiling is
never written into a row labelled 120 h. `prune_lead_substitutions.py` applies
the same rule to shards already on disk.

**`nan` means clear sky.** GRIB omits cloud ceiling, base and top where there is
no cloud (a `nan` HRRR ceiling goes with 0.0% median low cloud; a numeric one with
39.9%). Shards are read with `keep_default_na=False` so "never fetched" (`""`)
stays distinct from "no cloud" (`"nan"`), and cloud-geometry columns get an
explicit `_no_cloud` flag. NAM, GFS and NBM encode "no cloud" as a large sentinel
height instead, which is folded into the same flag. Median imputation would
recode every clear-sky hour as "cloud at the typical height".

**Inversion strength is supplied explicitly.** An undercast is an inversion with
the summit above the deck, which is a *difference* between two levels; trees can
only approximate that through many axis-aligned splits. `T850 − T925` alone scores
ROC-AUC 0.698 on HRRR, better than any raw field. Summit temperature is
interpolated from pressure levels, never taken from 2 m, which sits at the model's
smoothed terrain height rather than the real 1,917 m.

**Lead is not a feature.** Which leads exist changed with each model's versions,
so lead would act as a proxy for the year and bring the reporting-drift confound
back in. Skill against lead is measured by grouping the holdouts instead, and
each lead gets its own decision threshold.

**Splits separate by date, and folds by ISO week.** Each observation is one hour
and a day holds about 24, so an observation-level split would test on 03:50 after
training on 02:50 of the same day. Training rows sharing a date with a holdout are
dropped, plus a one-day buffer for multi-day inversions.

**One model is served: the combined source, Gradient Boosting.** All three
algorithms rank hours almost identically (Spearman 0.85–0.93), so a majority vote
of them adds nothing. The useful diversity is between weather *models*: combining
all six sources' fields in one model beats the best single source and beats
averaging six separate probabilities. On the webcam days, Gradient Boosting beats
the 2-of-3 vote and XGBoost, and ties both on the base-rate holdout
(`headline_choice_ci.py`, paired day-block bootstrap).

**Train and serve share code.** GRIB grid sampling lives only in
`grib_sample.py`, and feature construction is *imported* from
`train_undercast_obs.py` by `weather_to_json.py`. Both used to be duplicated; a
GFS longitude bug (every GFS value read at 44.25°N, 0°E, in southwestern France,
because xarray clamps an out-of-range `sel`) survived for months because it was
fixed in one copy only. `test_serving_features.py` replays real training rows
through the serving path and requires identical features and identical output.

### Holdouts

| split | what it is | used for |
|---|---|---|
| `train` | stratified sample, ~17% positive | fitting |
| `holdout_baserate` | 2022, every 3 h, unsampled, true ~2.7% rate | choosing the per-lead thresholds |
| `holdout_webcam` | the observation nearest noon on each hand-labelled day | **headline numbers, scored against people** |

Precision measured on `train` is meaningless: negatives were subsampled 5:1, so
it is computed against a base rate that does not exist.

## Serving

`get_weather.yml` runs `weather_to_json.py` every six hours. It:

1. picks a base time two hours behind the clock (a run still uploading passes the
   index check and then fails the downloads), and resolves each model's newest
   run that has actually published the hours needed (`resolve_run`);
2. fetches each source on its forecast-hour grid (`forecast_hour_grids`): dense
   near range, then 3-hourly out to the lesser of the source's *training* reach
   and the model's longest trained lead;
3. writes the raw fields to `weather_data_<summit>.json` (the "Individual
   Parameters" plots), then applies the model and writes
   `predictions_all.json["current"]`.

An hour is published only when every source that could reach it reports;
otherwise it is null, which the page shows as an empty slot, never as "no
undercast". If the model cannot run at all, `current` carries
`status: "unavailable"` and a reason. The `[current model]` and
"Retrieval by model" lines in the Actions log report per-source coverage and how
many hours survived.

## Known issues

- **Run resolution is per job, not per forecast hour.** A source whose newest
  run is N hours old loses the last N hours of its reach. That is why a typical
  run publishes to 138 h rather than 144. Resolving per hour, as
  `candidate_runs()` in `fetch_nwp_at_obs.py` does for training, would recover
  those ends.
- **The base-rate holdout is thin**: 2,896 observations, 86 of them undercast,
  each scored at every lead it has data for. Enough to see skill fall with lead, not to pin each lead's threshold
  precisely (the trainer falls back to the global threshold below 25 positives
  and marks it). Holdout *weeks* spread across the record would be better, but
  need an unsampled download for those weeks.
- **NBM is short in two shards at long lead** (shards 107 and 110; about 216
  rows). Found by a per-shard fill scan, not by job status. A local repair run
  hangs at a fixed task index; not yet diagnosed.
- **`cape_ecmwf` is 12.7% populated** — ECMWF renamed the field `mucape` in 2025.
  `vvel_*_ecmwf` and `dpt_2m_ecmwf` sit at exactly 40.8%, probably one shared
  cause.
- **The label ceiling.** No threshold on the remark fields separates days whose
  remarks are identical but whose skies are opposite. A field the
  remarks do not carry (how much of the horizon the deck fills, or a measured
  temperature profile up the Auto Road) is what would lift it.

## Where things live

| path | what | in git |
|---|---|---|
| `files/weather/obs/undercast_record.csv` | the labelled record | yes |
| `files/weather/obs/nwp_sample.csv` | observation times fetched | yes |
| `files/weather/csv/obs/*.csv` | forecast fields, 120 shards | yes |
| `files/weather/models/obs/` | metadata, figures' JSON, the served model | served model + JSON only |
| `files/weather/csv/MtWashington_undercast_orig.csv` | hand webcam scores | yes |
| `files/weather/csv/undercast_labels_reviewed.csv` | reviewed scores, used only by `tune_undercast_screen.py` | yes |
| `files/weather/webcam/` | dated webcam stills and their manifests | no (~1 GB) |
| `files/weather/examples/model_training_images/` | figures on /weather/details/ | yes |

`leakage_before_after.png` is the one figure no current script produces. It
shows the first model, trained on hand-labelled dates, before and after splitting
by date; the code that drew it is in the git history.
