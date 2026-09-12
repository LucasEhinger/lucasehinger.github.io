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
                                    (6,852 undercast, 1997-2026)
scripts/sample_undercast_obs.py     all positives + 5:1 negatives, stratified on
                                    each positive's own (year, month, hour)
.github/workflows/fetch_nwp_obs.yml forecast fields at each observation's OWN
                                    valid time, 3 leads (~1/24/48 h), 120 shards
scripts/train_undercast_obs.py      -> files/weather/models/obs/
scripts/undercast_eval.py           load the fitted artifacts and score them on
                                    a split; imported, never run
scripts/plot_undercast_obs_models.py    the /weather/details/ figures
scripts/compare_undercast_ensembles.py  is the 3-algorithm vote worth keeping?
```

`undercast_eval.py` exists so that nothing which merely *looks at* the models has
to refit them. A refit is a different model, and its numbers would not be the
ones the page quotes. Everything downstream loads the pickles instead.

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

### The 3-algorithm vote is not doing anything

`compare_undercast_ensembles.py`, on the untouched holdouts, using the artifacts
on disk rather than refits.

**The vote never wins.** On the combined model it ties the best single algorithm
on the base-rate holdout (F1 0.45) and loses on the webcam holdout (0.33 against
Gradient Boosting's 0.39). Unanimity buys precision 0.48 on the base-rate holdout
at the cost of 0.19 recall on the webcam one. The mechanism is that the three
members rank hours the same way 0.85-0.93 of the time (Spearman) — three tree
ensembles reading one feature set are not three opinions.

**No other algorithm does better either.** Trained on the same split: XGBoost
0.925 / GB 0.919 / RF-deeper 0.915 / RF 0.914 / HistGB 0.909 / logistic 0.896 /
ExtraTrees 0.870 (combined source, base-rate ROC-AUC). On HRRR the same seven
span 0.855-0.868. Logistic regression landing 0.006 behind on a single source is
the informative one: nearly all the signal here is linear in these features, and
the trees earn their keep only on the combined source where six models' columns
give them interactions to find.

**The diversity that pays is between weather models.** Same measurement across
the six sources, one algorithm: rank correlation 0.38-0.60, less than half the
algorithms' agreement, and HRRR/ECMWF — the two strongest — agree least (0.38).
On the rows every source covers (7,239 obs, 177 positives):

| predictor | base-rate ROC-AUC | 95% CI |
|---|---|---|
| best single source (HRRR) | 0.875 | [0.835, 0.915] |
| mean of all six probabilities | 0.906 | [0.858, 0.946] |
| combined model (feature-level fusion) | 0.925 | [0.887, 0.958] |

Paired, day-block bootstrap: combined − mean-of-six = **+0.019 [+0.003, +0.034]**,
so feature-level fusion beats probability averaging by a small but real margin.
mean-of-six − HRRR = +0.030 [−0.003, +0.063], which does not quite clear zero.

Greedy forward selection: HRRR 0.875 → +ECMWF 0.907 → +GFS 0.910 → +NBM 0.911,
then NAM and RAP each make it *worse*. Two unlike sources capture essentially the
whole gain (best pair HRRR+ECMWF 0.907; worst pair GFS+RAP 0.859). RAP finishing
last is consistent with it being the 13 km model HRRR is initialised from.

Implications: dropping to one gradient-boosted model per source costs no
measurable skill and removes 43 MB of RandomForest artifacts; effort on
algorithms has run out of room, effort on a genuinely different view of the
atmosphere has not.

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

## Cutover: done, additively

The live page now serves the new model. It was done **alongside** the old path
rather than in place of it, which is the part worth knowing if something looks
wrong later:

`weather_to_json.py` still computes and publishes every legacy per-source,
per-algorithm series exactly as before, so nothing that existed can break. On top
of that it publishes one new key, `predictions_all.json["current"]`, from the
combined-source Gradient Boosting model with per-lead thresholds. That block is
wrapped in try/except: if it raises, the key is absent, the headline panel on
/weather/ hides itself, and every other output is untouched. A missing forecast
is NOT rendered as "no undercast" — those are different statements and the front
end keeps them apart.

What was fixed to make it possible:

- **The longitude bug is gone from the serving path.** `sample_nearest`,
  `_match_lon_convention`, `_to_scalar` and `find_nearest_by_geodetic` now live
  in `scripts/grib_sample.py` and are imported by *both* fetchers. They used to
  be copy-pasted into each, which is why the fix reached the training data and
  not the live page. There is now one copy and it cannot drift.
- **Feature construction is imported, not reimplemented.** `time_features`,
  `normalize_weather_columns` and `add_profile_features` are defined once, in
  `train_undercast_obs.py`, and called by the serving code.
- **`scripts/test_serving_features.py` proves it.** Real training rows are
  replayed through the serving path and compared column by column: all 213 match
  exactly, and the model's output is identical. Run it after touching either
  side.
- **Variable-list mirror.** `boundary_layer_cloud_layer_gfs` now uses the alias
  that actually matches (the `%n hour` form matched nothing and the column was
  100% empty). `boundary_layer_cloud_layer` keeps its unsuffixed fetch key
  because the legacy preprocessors need it, and the serving code aliases the
  suffixed name onto it rather than renaming and breaking them.
- **Only two artifacts are committed**: `preprocessor_all.pkl` and
  `gradient_boosting_best_f1_all.pkl`, 0.57 MB together. CI checks out the repo,
  so the model it serves has to be in it. The RandomForests (9.6 MB for "all"
  alone) and every other source stay ignored, because nothing serves them.

Which model, and why: see "The 3-algorithm vote is not doing anything" above.
Short version — on the hand-labeled webcam days it beats the 2-of-3 vote by F1
+0.090 [+0.010, +0.175] and XGBoost by +0.084 [+0.002, +0.181], and ties both on
the base-rate holdout.

### Serving the model live turned out to be the hard part

The model was never the obstacle. Getting six weather models onto the same valid
times, from a cron job, was. Four separate faults, each of which published a
plausible number rather than failing:

1. **ECMWF was empty in every live run, for months.** IFS open data lags the
   wall clock by more than six hours, and the script asked for the most recent
   6-hourly slot -- which reliably did not exist yet. A missing column is just a
   column of nulls, so nothing complained. `resolve_run()` now finds each
   model's newest *published* run and lengthens its forecast hours to match, so a
   staler run still lands on the same valid times.
2. **The base run was the one currently uploading.** The workflow fires exactly
   on the synoptic hours. Index files appear before the data, so availability
   checks passed and the downloads then missed: measured at 18:50 UTC against the
   18Z run, HRRR, NAM and RAP all returned nothing. The base time is now lagged
   two hours before rounding down.
3. **`%n` alias substitution used the reported hour, not the fetched lead.**
   Aliases like `:APCP:surface:%n hour fcst` name the forecast hour inside the
   GRIB message. Once a model is read from an older run at a longer lead the two
   differ, and the substituted name matches nothing -- losing the variable
   silently. This was introduced by fix 1 and caught by re-probing.
4. **Hours where only some sources reported still got a forecast.** The six
   publish on different cadences, so the union of their forecast hours contains
   hours only one of them covers. The combined model is defined only where all
   six report; on a row missing five of them it returned a calm, repeated 0.095.
   Those rows are now published as null. The grids were also aligned onto common
   three-hourly steps, which took the number of fully-covered hours from nine to
   seventeen across two days.

Verified against live data end to end: all six sources populate, incomplete hours
are nulled, and the model publishes. Watch the `[current model]` lines in the
Actions log -- they report per-source population and how many hours survived.

Known limitation after all that: the panel typically reaches 24-30 h, not 48.
ECMWF publishes late and 3-hourly, so the fully-covered hours run out before the
other five sources do. A representative run: 9 usable hours at 3-hourly steps
from +3 h to +27 h. The 48 h skill numbers remain correct for the model -- they
come from training rows where each source was fetched at whatever lead it needed
-- but a single 6-hourly job cannot reproduce that. Closing the gap means
resolving runs per forecast hour, not once per job.

### Forecasting past 48 h: the lead ladder

The three leads sampled at every observation (1 / 24 / 48 h) had a consequence
nobody designed: **every training row carries all six sources**, because below
48 h all six can reach. `rows_for_source("all")` therefore required all six, and
the live guard required all six, and so the panel went quiet the moment one source
was late. ECMWF is late every run — it publishes behind the others and only
3-hourly — which is why coverage stopped near 27 h instead of 48.

Dropping ECMWF would have worked and cost the single most important feature in
the model (`dRH_925_850_ecmwf`, top of the importance ranking). The better fix is
to make the model *tolerant* of a missing source rather than independent of it,
and the leads themselves are how you teach that:

    1 h  24 h  48 h   all six reach
    72 h            HRRR (48) and RAP (51) cannot
    96 h            NAM (84) cannot either
    120 h           GFS's ceiling in RUN_SPECS
    144 h           ECMWF's ceiling, and the end of the ladder

Past 48 h the short-range models are *structurally* absent, so rows at 72 h and
beyond are exactly the partial-source rows the live page meets every run. Training
on them makes a partial forecast something the model has seen rather than
something it extrapolates into.

The ladder stops at 144 h on purpose. Past it only NBM reaches, and a
single-source forecast of a mesoscale inversion six days out is not worth the
download.

**Three things had to change together.**

1. `fetch_nwp_at_obs.TARGET_LEADS` grew to the ladder. Extending shards already on
   disk needs no re-fetch: `--resume` keeps the `(obs, lead)` pairs present and
   downloads only the new leads. Cost is about 1.7x the GRIB reads per
   observation, not 2.3x, because the short-range models contribute nothing past
   48 h.

2. **A substitution guard**, which is the part that would have silently poisoned
   everything. `candidate_runs()` deliberately walks outward to the nearest
   achievable forecast hour — right when the gap is run cadence (NBM's extended
   cycles fire only 00/06/12/18Z, so a 42 h sample honestly serves a 48 h valid
   time, and 86% of NBM's 48 h cells are such offsets) and badly wrong when the
   gap is the model's ceiling. Unguarded, asking for 120 h hands back HRRR's 48 h
   forecast and writes it into a row labelled 120 h. Nothing downstream can
   notice: the achieved hour is recorded in `lead_<model>`, but `_is_lead_col`
   drops every `lead_*` column from the features, so the model would learn that
   120 h forecasts are unusually sharp and then meet real 120 h data in
   production.

   Two independent conditions now bound the walk — `max_reachable_lead()` for "can
   this model reach that far at all", and `LEAD_SLACK_H = 12` for "is the nearest
   cycle close enough". The 12 h figure is measured, not chosen: across the 73,635
   rows on disk it rejects exactly one thing, HRRR's pre-2021 era ceilings
   substituted at lead 48, and leaves every other model's cadence offsets intact.

   That fault is already in the data, so `prune_lead_substitutions.py` applies the
   guard retroactively with no downloads — the achieved hour is in the CSV, so the
   offending cells can just be emptied. Measured: the combined model loses 27 of
   31,876 rows (0.08%), HRRR's loses 17,405 of 73,336 (23.7%). **Expect HRRR's
   published skill to fall after this.** Its 0.868 AUC was partly earned on 15-36 h
   forecasts sitting in rows labelled 24 h and 48 h; removing them is the
   correction, not a regression.

3. **`rows_for_source("all")` relaxed** from "all six present" to "every source
   that could reach this row's lead present" (`sources_expected_at`). At 1/24/48 h
   the two rules are provably identical — verified to return the same row index on
   the current data — so this is a no-op until the ladder data arrives.
   `weather_to_json` imports the same function rather than restating the table,
   because two copies of it drifting apart is exactly how the forecast would
   quietly become a forecast of something else.

**Sequencing, which matters.** The serving path refuses to apply the model past
the longest lead it was *trained* at, read from `threshold_by_lead` in the
metadata. So today, with a model trained at 1/24/48, hours past 48 h are nulled
even though the source guard would now accept them; after a retrain on the ladder
they start publishing with no further code change. `test_lead_aware_guard.py` pins
both halves of that, because the relaxed source rule on its own would happily hand
the 48 h model a 144 h row and publish a confident number.

**What this does not fix.** Still 175 independent undercast days — more leads means
more rows of the same events. It buys lead coverage and missing-source robustness,
not statistical power. See `undercast_capacity.py`.

**One design consequence to decide deliberately.** Missingness now encodes lead
almost perfectly (HRRR present iff lead <= 48 h). The model is currently
lead-agnostic on purpose — `target_lead_h` is not a feature, and declining
sharpness is handled with per-lead thresholds instead — so after this it learns its
own lead through the back door. That may be an improvement, since it could
self-calibrate rather than leaning on thresholds, but the "lead-agnostic model plus
per-lead thresholds" story on the details page stops being true.

### Reproducing every figure, number and check

Nothing the site shows is generated by hand, and nothing lives only in a scratch
directory. Every image on `/weather/` and `/weather/details/` comes from a script
below; every number quoted in those pages or in this file comes from one too.

**Figures.** From a checkout with the model artifacts present:

```bash
# Everything under files/weather/examples/model_training_images/ that the
# results sections use: per-source bars (all three algorithms), the three
# discrimination panels, 21 per-(source, algorithm) feature plots, the
# per-source confusion panels, and the served-model confusion figure.
python3 scripts/plot_undercast_obs_models.py

# The vote/diversity figure and ensemble_comparison.json.
python3 scripts/compare_undercast_ensembles.py

# Record-level figures: training_data_scale.png, undercast_rate_drift.png
python3 scripts/plot_undercast_record.py
# The screen ablation: screen_cut_ablation.png
python3 scripts/ablate_undercast_screen.py
# Why thresholds on the raw fields cannot work: inversion_evidence.png
python3 scripts/plot_inversion_evidence.py
# Physics-only baseline rules, and baseline_rules.json
python3 scripts/baseline_undercast_rules.py
```

`leakage_before_after.png` is the one exception, and deliberately so: it comes from
`plot_undercast_models.py`, the superseded first-pass script. The figure is *about*
the first pass, so only the code that had the bug can draw it. That script is kept
for exactly this reason and should not be re-run for anything else -- its docstring
says so, since it would overwrite current figures with first-pass ones.

**Analyses behind the prose.** These print tables rather than images; the pages and
this README quote them:

```bash
# Is 213 features too many for 175 undercast days? Feature/depth sweeps, a
# paired day-block bootstrap, and the seed-to-seed noise floor that decides
# how to read either.
python3 scripts/undercast_capacity.py

# Are the forecast leads in the shards what the rows claim? Achieved vs
# claimed, fill within each archive window, availability from RUN_SPECS
# cross-checked against what the trainer requires, and the guard's impact.
python3 scripts/audit_nwp_leads.py

# Which forecast hours each model really publishes. Run this before changing
# any RUN_SPECS value -- two of them were wrong from documentation.
python3 scripts/probe_model_fxx_grid.py
```

**Checks.** All runnable from a fresh checkout, no arguments:

```bash
python3 scripts/test_serving_features.py    # served features == trained features
python3 scripts/test_lead_aware_guard.py    # partial-source hours, no extrapolation
node scripts/test_weather_page_js.mjs       # figure URLs exist; CM arithmetic adds up
node scripts/test_undercast_panel.mjs       # all six headline-panel render states
```

### Still open

- [ ] **Resolve runs per forecast HOUR, not once per job.** `resolve_run()` now
      picks each model's newest complete run, which fixed the empty-ECMWF bug,
      but it still picks one run per model for the whole job. Training used
      `candidate_runs()` / `snap_valid()` in `fetch_nwp_at_obs.py` to choose, for
      each target valid time, whatever run covered it best. Doing the same here
      is what would take the panel from ~27 h of coverage to the full 48.
- [ ] **Retire the legacy path** — the per-source/per-algorithm series in
      `weather_to_json.py`, `files/weather/models/`, `regen_weather_csv.yml` and
      `train_undercast_models.py` — once `current` has run clean for a while.
      Keeping both is deliberate for now: it is the fallback.
- [ ] **Watch the first few scheduled runs.** The `[current model]` lines in the
      Actions log report feature coverage; below 80% it refuses to publish.
- [ ] **Fetch the ladder, then retrain.** The code is in; the data is not. Run
      `fetch_nwp_at_obs.py --resume` across the shards to add leads 72-144, then
      `prune_lead_substitutions.py --apply`, then retrain. Nothing past 48 h
      publishes until that happens, by design.
- [ ] **Trim the model** — 40 features and depth 2 instead of 213 and depth 3.
      Measured as free on both holdouts (every interval contains zero) and worth
      it anyway: 48 GRIB fields per run instead of 141. See
      `undercast_capacity.py` and the details page.
- [ ] **`cape_ecmwf` is 12.7% populated** — the alias is `:cape:` and ECMWF renamed
      it `mucape` in 2025. `vvel_700/850/925mb_ecmwf` and `dpt_2m_ecmwf` sit at
      exactly 40.8%, so probably one shared cause.
