# Five-league expansion report (Stage 3 FIXED-v2)

Date: 2026-10-27. Model: improved goal-model recipe, config `fc6_pool`
(the shipped v1.5.0 winner, chosen on v1.5.0 internal validation, never
on these test sets), seed 0, FEATURE_VERSION 3 (corrected SPADL
coordinates). Model hash (first 16 hex): `7d141b0368be24af`.
Reproduce with `python scripts/run_five_leagues.py` from the repo root
(data lives under gitignored `data/`; the script downloads any missing
lineups from StatsBomb open data and caches featurized frames and the
trained models under `data/cache/`).

No version bump and no v2.0.0 tag come from this expansion. The tag
stays held until U01/U02 validation plus owner signoff.

## League set

Exactly the five nominated leagues, no more, no fewer:

| League | StatsBomb competition | Seasons | Matches |
| :--- | :--- | :--- | ---: |
| English Premier League | 2 | 2015/2016 (27) | 380 |
| La Liga | 11 | 2015/2016 (27) | 380 |
| Serie A | 12 | 2015/2016 (27) | 380 |
| Bundesliga | 9 | 2015/2016 (27), 2023/2024 (281) | 68 |
| Ligue 1 | 7 | 2015/2016 (27) | 377 |

## Data audit (reported first, before any metric)

Events were complete for every league before this run. Lineups were
present only for Bundesliga (68/68); the four 2015/2016 leagues had
0 lineups on disk because the earlier pooled download fetched events
only. This run downloaded 1,517 lineup files from the same public
StatsBomb open-data source (0 failed). Post-download, lineups are
complete for all five leagues: 380/380, 380/380, 380/380, 68/68,
377/377.

Two availability facts shape everything below, reported as-is:

- Bundesliga open data covers only 34 matches per season, so its
  per-league pool is 68 matches total, against roughly 380 for each
  other league. Its test set is 15 matches with wide intervals.
- Wage coverage is low for the 2015/2016 leagues (see ratings): EA FC
  24 rosters match only 27-38% of 2015/2016 players, so those leagues'
  frontiers and backtests run on the matched subset. Bundesliga
  matches 63.2% because its pool includes 2023/2024 players.

## Splits (train / validation / test, chronological, disjoint)

Each league's matches are ordered by (match_date, match_id): earliest
60% train, next 20% validation, latest 20% test. All three ID sets are
pairwise disjoint (asserted in the run; any overlap raises). The single
trained model set excludes every validation and test match ID from all
five leagues. Train pool: 950 league-train matches + 166 pooled
(Euro 2020, Indian Super League 2021/2022) = 1,116 matches.

| League | Train | Validation | Test |
| :--- | ---: | ---: | ---: |
| English Premier League | 228 | 76 | 76 |
| La Liga | 228 | 76 | 76 |
| Serie A | 228 | 76 | 76 |
| Bundesliga | 40 | 13 | 15 |
| Ligue 1 | 226 | 75 | 76 |

## Frozen scoreboard: test set, mean Brier, model vs base rate

Base rate = the base rate of that league's own train slice. The model
beats the base rate on all 20 test lines (5 leagues x 4 labels) on
exact means. Concedes margins are narrow and confidence intervals
overlap the base rate on some lines; the Bundesliga lines rest on 15
matches. Validation slices below are report-only and were never used
to tune anything.

| League | Scores incl | Concedes incl | Scores excl | Concedes excl |
| :--- | :--- | :--- | :--- | :--- |
| English Premier League | 0.0102 vs 0.0112 | 0.00226 vs 0.00229 | 0.0106 vs 0.0108 | 0.00266 vs 0.00269 |
| La Liga | 0.00920 vs 0.0103 | 0.00218 vs 0.00221 | 0.00957 vs 0.00985 | 0.00260 vs 0.00263 |
| Serie A | 0.00921 vs 0.0103 | 0.00214 vs 0.00218 | 0.00965 vs 0.00994 | 0.00247 vs 0.00251 |
| Bundesliga | 0.0101 vs 0.0112 | 0.00233 vs 0.00238 | 0.0106 vs 0.0109 | 0.00266 vs 0.00271 |
| Ligue 1 | 0.00860 vs 0.00970 | 0.00215 vs 0.00217 | 0.00899 vs 0.00929 | 0.00256 vs 0.00258 |

Validation (report-only), scores incl / concedes incl, model vs base:
EPL 0.00844 vs 0.00923 / 0.00193 vs 0.00195; La Liga 0.00960 vs 0.0108 /
0.00229 vs 0.00232; Serie A 0.00776 vs 0.00858 / 0.00203 vs 0.00204;
Bundesliga 0.00828 vs 0.00908 / 0.00184 vs 0.00186; Ligue 1 0.00835 vs
0.00930 / 0.00211 vs 0.00213. Validation agrees with test in every
league.

## Ratings and wage coverage

Ratings use min_minutes 90 and shrink_k 900, as in the shipped config.
Players without an EA FC 24 wage match are dropped by `build_ratings`,
so rated counts below are matched players only.

| League | Players rated | Wage match rate |
| :--- | ---: | ---: |
| English Premier League | 179 | 37.7% |
| La Liga | 146 | 29.4% |
| Serie A | 143 | 26.8% |
| Bundesliga | 268 | 63.2% |
| Ligue 1 | 145 | 28.5% |

## Frontier: expected squad VAEP/90 at each budget

| League | 0.30M | 0.50M | 0.70M |
| :--- | ---: | ---: | ---: |
| English Premier League | 10.87 | 11.57 | 11.97 |
| La Liga | 11.78 | 12.49 | 12.90 |
| Serie A | 12.72 | 12.83 | 12.83 |
| Bundesliga | 10.39 | 11.08 | 11.59 |
| Ligue 1 | 12.37 | 12.75 | 12.95 |

All squads are 23 players. Serie A's frontier is flat between 0.50M
and 0.70M: the extra budget no longer binds for that objective.

## Backtest: realized second-half squad VAEP, optimizer vs baselines

Within-season date split (first half selects, second half scores),
seed 0, same three budgets. Compare optimizer vs value only within a
league: totals scale with second-half match counts (190 matches for
the full-season leagues, 34 for Bundesliga), so cross-league totals
are not comparable. The value baseline saturates below 0.30M for the
four full-season leagues (the same cheapest squad is feasible at all
three budgets, hence identical value totals); Ligue 1's optimizer
saturates between 0.50M and 0.70M. The optimizer beats value at all
three budgets in all five leagues.

| League | Budget | Optimizer | Value | Random mean | Oracle |
| :--- | ---: | ---: | ---: | ---: | ---: |
| English Premier League | 0.30M | 113.76 | 85.32 | 73.41 | 171.52 |
| English Premier League | 0.50M | 140.29 | 85.32 | 83.30 | 199.30 |
| English Premier League | 0.70M | 161.30 | 85.32 | 87.52 | 215.46 |
| La Liga | 0.30M | 124.56 | 95.05 | 84.39 | 176.80 |
| La Liga | 0.50M | 146.78 | 95.05 | 93.77 | 195.19 |
| La Liga | 0.70M | 158.36 | 95.05 | 100.80 | 204.85 |
| Serie A | 0.30M | 142.31 | 107.81 | 95.85 | 188.91 |
| Serie A | 0.50M | 152.20 | 107.81 | 96.51 | 202.83 |
| Serie A | 0.70M | 161.33 | 107.81 | 100.49 | 209.68 |
| Bundesliga | 0.30M | 39.22 | 37.90 | 8.13 | 56.79 |
| Bundesliga | 0.50M | 57.58 | 53.09 | 8.50 | 83.55 |
| Bundesliga | 0.70M | 83.45 | 77.12 | 12.40 | 105.24 |
| Ligue 1 | 0.30M | 131.95 | 83.05 | 85.56 | 187.30 |
| Ligue 1 | 0.50M | 157.52 | 83.05 | 90.50 | 206.55 |
| Ligue 1 | 0.70M | 157.52 | 83.05 | 93.62 | 215.45 |

## Squad and determinism checks

Every league, every budget: squad constraints hold (23 players,
position minimums, wage bill within budget) and clean regeneration
(selecting twice) yields identical player IDs and hashes.

## Comparability labels

- These scoreboards are NOT comparable to the v1.5.0 FC6 figures:
  different populations (per-league chronological splits vs the
  Bundesliga 130/32 split), different test sets, and corrected
  FEATURE_VERSION 3 features (FC6 was computed pre-C1). The
  Bundesliga line above is the corrected-feature Bundesliga number on
  a new split, not a recomputation of FC6.
- These backtests are NOT comparable to the v1.5.0 Bundesliga
  backtest: different splits, different model, different ratings.
  Within-league optimizer-vs-value comparisons are the valid ones.
- Full machine-readable numbers, split ID hashes, and per-match
  summaries are in `results.json` next to this report.
