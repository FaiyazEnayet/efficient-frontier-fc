<p align="center"><img src="docs/readme-banner.svg?v=type-20261017b" alt="Efficient Frontier FC: Football squad selection under a wage budget" width="1200"></p>

<h1 align="center">Efficient Frontier FC</h1>

<p align="center">
  <a href="https://github.com/FaiyazEnayet/efficient-frontier-fc/actions/workflows/ci.yml"><img src="https://github.com/FaiyazEnayet/efficient-frontier-fc/actions/workflows/ci.yml/badge.svg?label=CI&labelColor=24292f" alt="CI"></a>
  <img src="https://img.shields.io/badge/license-MIT-795b1b?labelColor=24292f" alt="License: MIT">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?labelColor=24292f" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/built%20with-SciPy%20%2B%20VAEP-0054A6?labelColor=24292f" alt="Built with SciPy + VAEP">
</p>

<p align="center">
  <a href="https://faiyazenayet.github.io/efficient-frontier-fc/docs/dashboard.html"><img src="https://img.shields.io/badge/Open-Dashboard-795b1b?style=for-the-badge&labelColor=f4edda" alt="Open Dashboard"></a>
  <a href="#method"><img src="https://img.shields.io/badge/Explore-Method-183f30?style=for-the-badge&labelColor=f4edda" alt="Explore Method"></a>
  <a href="#run-locally"><img src="https://img.shields.io/badge/Run-Locally-526554?style=for-the-badge&labelColor=f4edda" alt="Run Locally"></a>
</p>

Efficient Frontier FC treats squad construction as a constrained allocation problem. It combines event-based player value with wage proxies, then asks which feasible 23-player squad fits a specified weekly budget.


<p align="center"><a href="https://faiyazenayet.github.io/efficient-frontier-fc/docs/dashboard.html"><img src="docs/demo.gif?v=showcase-squads-20261027" alt="Efficient Frontier FC demo"></a></p>

Efficient Frontier FC covers the English Premier League, La Liga, Serie A, Bundesliga and Ligue 1, with weekly wage caps of €300,000, €500,000 and €700,000, evaluated on the corrected FEATURE_VERSION 4 results.

The optimizer beats the value baseline in 14 of 15 league-budget cells. The exception is Bundesliga at €300,000: 40.83 optimizer versus 41.53 value. The goal model has lower exact mean Brier and logloss on all 20 held-out test lines, but model and baseline confidence intervals overlap on all 20 lines. Those are mean improvements, not evidence of statistically separated intervals.

Named 23-player squads and 4-3-3 Best XIs are stored in the published results for every league at each of the three wage budgets.

## Explore

The interactive dashboard opens with a five-league comparison, then gives each league its own evidence sheet:

- **Scoreboard:** model versus train-slice base rate, with Brier and logloss means and the recorded 95% confidence intervals for all four prediction labels. Validation results are separate and explicitly report-only.
- **Frontier:** the three recorded feasible budget points, expected squad VAEP/90, actual wage bills, squad size and the run's constraint and regeneration checks.
- **Backtest:** optimizer, value, random mean and hindsight oracle under the same weekly caps, plus recorded scoring-half player unavailability.
- **Coverage:** match counts, event and lineup file availability, chronological split counts, player matching and split fingerprints.

The budget controls select recorded solutions. They do not infer an intermediate squad.

## Evaluation

The evaluation uses the `fc6_pool` configuration at FEATURE_VERSION 4, with the own-goal label correction included. Each league has a chronological train / validation / test split; the report states that the match ID sets are disjoint. The model configuration was chosen before these five-league test sets were evaluated. Results below are rounded for display from [the published JSON](runs/five_leagues/results.json); the [research report](runs/five_leagues/report.md) records the protocol and caveats.

### Coverage before performance

**Bundesliga has only 68 open-data matches, against roughly 380 for each other league.** Its held-out test contains 15 matches. Complete event and lineup files therefore do not make its evidence pool equivalent to the others.

| League | Matches | Event files present | Lineup files present | Train / validation / test | Rated players | Wage match rate |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: |
| English Premier League | 380 | 380/380 | 380/380 | 228 / 76 / 76 | 179 | 37.7% |
| La Liga | 380 | 380/380 | 380/380 | 228 / 76 / 76 | 146 | 29.4% |
| Serie A | 380 | 380/380 | 380/380 | 228 / 76 / 76 | 143 | 26.8% |
| Bundesliga | 68 | 68/68 | 68/68 | 40 / 13 / 15 | 268 | 63.2% |
| Ligue 1 | 377 | 377/377 | 377/377 | 226 / 75 / 76 | 145 | 28.5% |

Event and lineup counts above refer to match files, not individual actions. Rated-player counts cover the wage-matched subset. EA FC 24 rosters miss much of the historical player population, so a complete event pool still produces an incomplete economic player pool.

### Test scoreboard

The base rate is estimated from each league's own training slice. Both Brier and logloss are lower-is-better probability errors. The main table shows the including variants; the full four-label table follows.

| League | Test matches | Scores Brier: model / base | Concedes Brier: model / base | Scores logloss: model / base | Concedes logloss: model / base |
| :--- | ---: | :--- | :--- | :--- | :--- |
| English Premier League | 76 | 0.01045 / 0.01150 | 0.00239 / 0.00242 | 0.05263 / 0.06347 | 0.01473 / 0.01705 |
| La Liga | 76 | 0.00937 / 0.01043 | 0.00225 / 0.00228 | 0.04801 / 0.05849 | 0.01436 / 0.01617 |
| Serie A | 76 | 0.00932 / 0.01038 | 0.00221 / 0.00226 | 0.04713 / 0.05824 | 0.01355 / 0.01604 |
| Bundesliga | 15 | 0.01011 / 0.01123 | 0.00233 / 0.00238 | 0.05039 / 0.06216 | 0.01386 / 0.01679 |
| Ligue 1 | 76 | 0.00896 / 0.01006 | 0.00227 / 0.00230 | 0.04589 / 0.05680 | 0.01452 / 0.01630 |

All 20 test lines favor the model on exact Brier means and exact logloss means. **Confidence intervals overlap the base rate on all 20 lines**, including the narrow conceding margins. The dashboard retains the full stored interval endpoints instead of turning rounded mean differences into significance claims.

<details>
<summary>Full four-label scoreboard: means and 95% confidence intervals</summary>

| League / label | Model Brier [95% CI] | Base Brier [95% CI] | Model logloss [95% CI] | Base logloss [95% CI] |
| :--- | :--- | :--- | :--- | :--- |
| English Premier League / Scores including | 0.01045 [0.00926, 0.01165] | 0.01150 [0.01013, 0.01288] | 0.05263 [0.04779, 0.05780] | 0.06347 [0.05743, 0.06970] |
| English Premier League / Scores excluding | 0.01085 [0.00962, 0.01211] | 0.01110 [0.00982, 0.01242] | 0.05567 [0.05028, 0.06131] | 0.06167 [0.05546, 0.06785] |
| English Premier League / Concedes including | 0.00239 [0.00200, 0.00278] | 0.00242 [0.00204, 0.00282] | 0.01473 [0.01242, 0.01698] | 0.01705 [0.01444, 0.01955] |
| English Premier League / Concedes excluding | 0.00279 [0.00236, 0.00324] | 0.00282 [0.00239, 0.00328] | 0.01719 [0.01479, 0.01980] | 0.01945 [0.01682, 0.02223] |
| La Liga / Scores including | 0.00937 [0.00817, 0.01064] | 0.01043 [0.00902, 0.01194] | 0.04801 [0.04261, 0.05360] | 0.05849 [0.05195, 0.06506] |
| La Liga / Scores excluding | 0.00974 [0.00844, 0.01105] | 0.01002 [0.00865, 0.01140] | 0.05061 [0.04453, 0.05686] | 0.05659 [0.04977, 0.06343] |
| La Liga / Concedes including | 0.00225 [0.00182, 0.00271] | 0.00228 [0.00184, 0.00274] | 0.01436 [0.01180, 0.01690] | 0.01617 [0.01346, 0.01879] |
| La Liga / Concedes excluding | 0.00266 [0.00216, 0.00320] | 0.00269 [0.00218, 0.00323] | 0.01689 [0.01421, 0.01982] | 0.01867 [0.01581, 0.02178] |
| Serie A / Scores including | 0.00932 [0.00807, 0.01061] | 0.01038 [0.00896, 0.01184] | 0.04713 [0.04192, 0.05266] | 0.05824 [0.05165, 0.06514] |
| Serie A / Scores excluding | 0.00976 [0.00848, 0.01112] | 0.01006 [0.00873, 0.01147] | 0.05042 [0.04481, 0.05635] | 0.05675 [0.05041, 0.06354] |
| Serie A / Concedes including | 0.00221 [0.00179, 0.00267] | 0.00226 [0.00182, 0.00272] | 0.01355 [0.01121, 0.01611] | 0.01604 [0.01332, 0.01913] |
| Serie A / Concedes excluding | 0.00254 [0.00203, 0.00308] | 0.00258 [0.00206, 0.00313] | 0.01562 [0.01294, 0.01851] | 0.01801 [0.01491, 0.02132] |
| Bundesliga / Scores including | 0.01011 [0.00745, 0.01310] | 0.01123 [0.00833, 0.01455] | 0.05039 [0.03951, 0.06268] | 0.06216 [0.04949, 0.07655] |
| Bundesliga / Scores excluding | 0.01063 [0.00776, 0.01377] | 0.01090 [0.00798, 0.01412] | 0.05388 [0.04195, 0.06632] | 0.06067 [0.04799, 0.07415] |
| Bundesliga / Concedes including | 0.00233 [0.00167, 0.00302] | 0.00238 [0.00171, 0.00307] | 0.01386 [0.01004, 0.01817] | 0.01679 [0.01287, 0.02098] |
| Bundesliga / Concedes excluding | 0.00267 [0.00191, 0.00346] | 0.00271 [0.00195, 0.00350] | 0.01608 [0.01176, 0.02104] | 0.01879 [0.01459, 0.02359] |
| Ligue 1 / Scores including | 0.00896 [0.00769, 0.01021] | 0.01006 [0.00860, 0.01151] | 0.04589 [0.04039, 0.05152] | 0.05680 [0.04984, 0.06373] |
| Ligue 1 / Scores excluding | 0.00933 [0.00802, 0.01066] | 0.00966 [0.00828, 0.01104] | 0.04859 [0.04277, 0.05494] | 0.05492 [0.04816, 0.06218] |
| Ligue 1 / Concedes including | 0.00227 [0.00185, 0.00273] | 0.00230 [0.00187, 0.00276] | 0.01452 [0.01214, 0.01707] | 0.01630 [0.01364, 0.01915] |
| Ligue 1 / Concedes excluding | 0.00268 [0.00221, 0.00318] | 0.00270 [0.00223, 0.00321] | 0.01710 [0.01422, 0.01999] | 0.01876 [0.01560, 0.02183] |

</details>

Validation slices are report-only and were not used to tune this evaluation. They agree with the test direction in each league, as documented in the report; they are not additional test data.

### Efficient frontier

Expected squad VAEP/90 at each weekly cap:

| League | €300,000 | €500,000 | €700,000 |
| :--- | ---: | ---: | ---: |
| English Premier League | 11.10597 | 11.87750 | 12.26491 |
| La Liga | 11.87052 | 12.55341 | 12.99147 |
| Serie A | 13.03440 | 13.16060 | 13.21122 |
| Bundesliga | 10.58591 | 11.23844 | 11.73518 |
| Ligue 1 | 12.59603 | 13.00909 | 13.19533 |

Every recorded solution contains 23 players. The run reports passing constraints and identical regeneration at each league-budget point. A small increase in expected value as the cap rises is part of the result; it should not be disguised as a large recruitment gain.

### Backtest: 14 of 15 cells

First-half selection is evaluated on second-half total squad VAEP. **Compare the optimizer with baselines within each league only.** Raw totals scale with scoring-half match counts and are not a ranking of league quality. The random figure is a recorded mean; the oracle uses second-half hindsight and is not an attainable forecasting policy.

| League | Weekly cap | Optimizer | Value | Random mean | Oracle |
| :--- | ---: | ---: | ---: | ---: | ---: |
| English Premier League | €300,000 | 128.29 | 96.79 | 75.48 | 179.71 |
| English Premier League | €500,000 | 147.88 | 96.79 | 85.79 | 208.82 |
| English Premier League | €700,000 | 173.70 | 96.79 | 89.76 | 224.48 |
| La Liga | €300,000 | 128.55 | 98.15 | 85.86 | 183.11 |
| La Liga | €500,000 | 148.61 | 98.15 | 95.10 | 200.88 |
| La Liga | €700,000 | 159.68 | 98.15 | 102.04 | 210.75 |
| Serie A | €300,000 | 149.39 | 124.38 | 97.70 | 195.36 |
| Serie A | €500,000 | 157.73 | 124.38 | 98.20 | 209.43 |
| Serie A | €700,000 | 162.20 | 124.38 | 101.78 | 216.79 |
| Bundesliga | €300,000 | 40.83 | 41.53 | 8.23 | 55.29 |
| Bundesliga | €500,000 | 62.14 | 58.68 | 8.60 | 81.81 |
| Bundesliga | €700,000 | 82.23 | 79.43 | 12.57 | 103.55 |
| Ligue 1 | €300,000 | 122.56 | 82.98 | 86.08 | 187.32 |
| Ligue 1 | €500,000 | 150.41 | 82.98 | 90.78 | 207.07 |
| Ligue 1 | €700,000 | 150.41 | 82.98 | 93.79 | 216.69 |

Bundesliga at €300,000 is the sole optimizer-versus-value loss: **40.83 versus 41.53**. The other 14 cells favor the optimizer. The value baseline saturates below the lowest cap in the four full-season leagues, while Ligue 1's optimizer has the same realized total at the two higher caps. Neither pattern is corrected or smoothed away in the dashboard.

![Five-league FV4 backtest, coverage and held-out uncertainty](docs/results.png)

## Method

Event-based scoring and conceding probabilities feed VAEP-style player ratings. Rating shrinkage and an EA FC wage-proxy join define the modeled player pool. The squad optimizer enforces the budget, squad-size and positional constraints, while the backtest evaluates selections using the later part of the season.

FEATURE_VERSION 4 includes the own-goal correction described in the report. The evaluation used fresh feature and model caches; the resulting model fingerprint is recorded with the published results. All figures on this page come from a single evaluation run; figures from other runs are not comparable with them.

## Run locally

```sh
python -m pip install -e ".[dev]"
python -m pip install -r portfolio_ui/requirements.txt
python scripts/make_dashboard.py
python scripts/make_portfolio_gif.py
```

The dashboard command reads the five-league results through its presentation renderer. The showcase command refreshes `docs/demo.gif` and `docs/results.png` from the same FEATURE_VERSION 4 file. The prebuilt dashboard embeds its chart library, fonts and result data for local viewing.

To reproduce the research rather than only its presentation:

```sh
python scripts/run_five_leagues.py
python -m pytest tests -q
```

The research command uses the source data and caches described in the report and may download missing public lineup files. It is separate from rebuilding the presentation.

## Code and data

`src/effc/` contains the research pipeline, ratings, selection and backtesting. `runs/five_leagues/results.json` is the numeric source for the figures above; `runs/five_leagues/report.md` documents the evaluation protocol. The dashboard embeds source hashes so a displayed result can be traced back to its inputs. Presentation updates do not change model fitting, test membership, constraints or recorded performance.

## Limitations

- **Unequal samples:** Bundesliga has 68 matches in its open-data pool and 15 held-out test matches. The other league pools are 380, 380, 380 and 377 matches. Smaller samples and wide intervals limit the strength of conclusions.
- **Overlapping uncertainty:** all 20 test-line confidence intervals overlap the base rate. Lower exact means do not establish separated model performance.
- **Proxy economics:** EA FC 24 wages are not verified historical or current contracts. Wage matching restricts the modeled player population, particularly in the older-season leagues.
- **Within-league comparisons:** realized backtest totals depend on the scoring-half pool. Do not compare raw league totals as if their exposure were equal.
- **Recorded points only:** there are three evaluated caps. A connecting line is not evidence of an optimal squad at every intermediate budget.
- **Roster interpretation:** recorded names and Best XI membership are shown directly; tactical fit is not modeled.
- **Hindsight reference:** the oracle has information unavailable at selection time. Random means and backtest point totals are not confidence intervals.
- **Comparability:** figures on this page describe a single evaluation run and are not interchangeable with figures from other runs.

## License

[MIT](LICENSE). Third-party datasets, libraries and fonts retain their own terms.
