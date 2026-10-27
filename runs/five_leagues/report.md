# Five-league expansion report, recomputed at FEATURE_VERSION 4

Date: 2026-10-27. Model: improved goal-model recipe, config `fc6_pool`
(the shipped v1.5.0 winner, chosen on v1.5.0 internal validation, never
on these test sets), seed 0, FEATURE_VERSION 4 (U02 own-goal label fix).
Model hash (first 16 hex): `b86caaed47f745d4` (FV3 run: `7d141b0368be24af`).
Reproduce with `python scripts/run_five_leagues.py` from the repo root
(data lives under gitignored `data/`; the script downloads any missing
lineups from StatsBomb open data and caches featurized frames and the
trained models under `data/cache/`).

No version bump and no v2.0.0 tag come from this recompute. U01/U02
validation has landed; the tag stays held pending owner signoff on
these recomputed numbers.

## What changed from the FEATURE_VERSION 3 run (FIXED-v2, 2026-10-27)

This is a recompute of the FIXED-v2 five-league expansion under the
current code, not a new expansion: same five leagues, same chronological
splits (60% train, 20% validation, 20% test), same config `fc6_pool`,
seed 0, same three budgets. Only the feature version changed, 3 to 4,
via U02: own goals (the StatsBomb "Own Goal For" / "Own Goal Against"
pair) now count in the scores/concedes labels and in score_diff, and
the goal-conceded debit in `add_vaep` extends to them. Cache freshness
was enforced by construction: the recompute ran with an empty
`data/cache/` (raw events, lineups, match metadata, and wages reused
from the FV3 run; no featurized frame or trained model carried over),
and the resulting model hash differs from the FV3 run, confirming fresh
training on FEATURE_VERSION 4 features. The improved-models cache key
also binds the feature version, so an FV3 bundle could not have been
reused even if one had been present.

Headline verdict: the model still beats the base rate on all 20 test
lines (5 leagues x 4 labels) on exact Brier means at FEATURE_VERSION 4,
with no line flipped from the FV3 run, and on all 20 logloss lines as
well. Brier means moved up slightly almost everywhere (own goals add
positive-label rows, raising both model and base-rate errors), but
every margin survived. Per-match confidence intervals overlap the base
rate on all 20 lines, as at FV3, so these are exact-mean wins, not
separated intervals.

One honest movement: the backtest flips at one cell. Bundesliga at
0.30M now trails value (optimizer 40.83 vs value 41.53), where the FV3
run had it ahead (39.22 vs 37.90). The other 14 league-budget cells
still favor the optimizer, so the backtest claim is 14 of 15, not
15 of 15. Frontiers moved up in every league and budget (see below);
ratings counts and wage match rates are unchanged, as expected, since
the player sets and wages did not change.

### Brier delta, test set, exact means (FV3 model vs base, FV4 model vs base)

| League | Label | FV3 model vs base | FV4 model vs base | FV4 verdict |
| :--- | :--- | :--- | :--- | :--- |
| English Premier League | Scores incl | 0.01020 vs 0.01122 | 0.01045 vs 0.01150 | win |
| English Premier League | Concedes incl | 0.00226 vs 0.00229 | 0.00239 vs 0.00242 | win |
| English Premier League | Scores excl | 0.01060 vs 0.01082 | 0.01085 vs 0.01110 | win |
| English Premier League | Concedes excl | 0.00266 vs 0.00269 | 0.00279 vs 0.00282 | win |
| La Liga | Scores incl | 0.00920 vs 0.01026 | 0.00937 vs 0.01043 | win |
| La Liga | Concedes incl | 0.00218 vs 0.00221 | 0.00225 vs 0.00228 | win |
| La Liga | Scores excl | 0.00957 vs 0.00985 | 0.00974 vs 0.01002 | win |
| La Liga | Concedes excl | 0.00260 vs 0.00263 | 0.00266 vs 0.00269 | win |
| Serie A | Scores incl | 0.00921 vs 0.01026 | 0.00932 vs 0.01038 | win |
| Serie A | Concedes incl | 0.00214 vs 0.00218 | 0.00221 vs 0.00226 | win |
| Serie A | Scores excl | 0.00965 vs 0.00994 | 0.00976 vs 0.01006 | win |
| Serie A | Concedes excl | 0.00247 vs 0.00250 | 0.00254 vs 0.00258 | win |
| Bundesliga | Scores incl | 0.01011 vs 0.01123 | 0.01011 vs 0.01123 | win |
| Bundesliga | Concedes incl | 0.00233 vs 0.00238 | 0.00233 vs 0.00238 | win |
| Bundesliga | Scores excl | 0.01061 vs 0.01090 | 0.01063 vs 0.01090 | win |
| Bundesliga | Concedes excl | 0.00266 vs 0.00271 | 0.00267 vs 0.00271 | win |
| Ligue 1 | Scores incl | 0.00860 vs 0.00970 | 0.00896 vs 0.01006 | win |
| Ligue 1 | Concedes incl | 0.00215 vs 0.00217 | 0.00227 vs 0.00230 | win |
| Ligue 1 | Scores excl | 0.00899 vs 0.00929 | 0.00933 vs 0.00966 | win |
| Ligue 1 | Concedes excl | 0.00256 vs 0.00258 | 0.00268 vs 0.00270 | win |

### Backtest delta, optimizer vs value (FV3, FV4)

| League | Budget | FV3 optimizer vs value | FV4 optimizer vs value | FV4 verdict |
| :--- | ---: | :--- | :--- | :--- |
| English Premier League | 300k | 113.76 vs 85.32 | 128.29 vs 96.79 | win |
| English Premier League | 500k | 140.29 vs 85.32 | 147.88 vs 96.79 | win |
| English Premier League | 700k | 161.30 vs 85.32 | 173.70 vs 96.79 | win |
| La Liga | 300k | 124.56 vs 95.05 | 128.55 vs 98.15 | win |
| La Liga | 500k | 146.78 vs 95.05 | 148.61 vs 98.15 | win |
| La Liga | 700k | 158.36 vs 95.05 | 159.68 vs 98.15 | win |
| Serie A | 300k | 142.31 vs 107.81 | 149.39 vs 124.38 | win |
| Serie A | 500k | 152.20 vs 107.81 | 157.73 vs 124.38 | win |
| Serie A | 700k | 161.33 vs 107.81 | 162.20 vs 124.38 | win |
| Bundesliga | 300k | 39.22 vs 37.90 | 40.83 vs 41.53 | LOSS |
| Bundesliga | 500k | 57.58 vs 53.09 | 62.14 vs 58.68 | win |
| Bundesliga | 700k | 83.45 vs 77.12 | 82.23 vs 79.43 | win |
| Ligue 1 | 300k | 131.95 vs 83.05 | 122.56 vs 82.98 | win |
| Ligue 1 | 500k | 157.52 vs 83.05 | 150.41 vs 82.98 | win |
| Ligue 1 | 700k | 157.52 vs 83.05 | 150.41 vs 82.98 | win |

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

Events were complete for every league, and lineups were already complete
for all five leagues from the FV3 run, so this recompute downloaded 0
lineup files (0 failed). Post-audit, lineups are complete for all five
leagues: 380/380, 380/380, 380/380, 68/68, 377/377.

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
(Euro 2020, Indian Super League 2021/2022) = 1,116 matches. Split ID
hashes are identical to the FV3 run.

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
overlap the base rate on all 20 lines; the Bundesliga lines rest on 15
matches. Validation slices below are report-only and were never used
to tune anything.

| League | Scores incl | Concedes incl | Scores excl | Concedes excl |
| :--- | :--- | :--- | :--- | :--- |
| English Premier League | 0.01045 vs 0.01150 | 0.00239 vs 0.00242 | 0.01085 vs 0.01110 | 0.00279 vs 0.00282 |
| La Liga | 0.00937 vs 0.01043 | 0.00225 vs 0.00228 | 0.00974 vs 0.01002 | 0.00266 vs 0.00269 |
| Serie A | 0.00932 vs 0.01038 | 0.00221 vs 0.00226 | 0.00976 vs 0.01006 | 0.00254 vs 0.00258 |
| Bundesliga | 0.01011 vs 0.01123 | 0.00233 vs 0.00238 | 0.01063 vs 0.01090 | 0.00267 vs 0.00271 |
| Ligue 1 | 0.00896 vs 0.01006 | 0.00227 vs 0.00230 | 0.00933 vs 0.00966 | 0.00268 vs 0.00270 |

Validation (report-only), scores incl / concedes incl, model vs base:
English Premier League: 0.00861 vs 0.00941 / 0.00199 vs 0.00201.
La Liga: 0.01007 vs 0.01127 / 0.00245 vs 0.00249.
Serie A: 0.00800 vs 0.00884 / 0.00207 vs 0.00209.
Bundesliga: 0.00855 vs 0.00937 / 0.00212 vs 0.00216.
Ligue 1: 0.00867 vs 0.00965 / 0.00224 vs 0.00226.
Validation agrees with test in every league.

## Ratings and wage coverage

Ratings use min_minutes 90 and shrink_k 900, as in the shipped config.
Players without an EA FC 24 wage match are dropped by `build_ratings`,
so rated counts below are matched players only. Counts are unchanged
from the FV3 run.

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
| English Premier League | 11.11 | 11.88 | 12.26 |
| La Liga | 11.87 | 12.55 | 12.99 |
| Serie A | 13.03 | 13.16 | 13.21 |
| Bundesliga | 10.59 | 11.24 | 11.74 |
| Ligue 1 | 12.60 | 13.01 | 13.20 |

All squads are 23 players. Serie A's frontier is flat between 0.50M
and 0.70M at FV3 and rises slightly at FV4 (13.16 to 13.21): the extra
budget barely binds for that objective.

## Backtest: realized second-half squad VAEP, optimizer vs baselines

Within-season date split (first half selects, second half scores),
seed 0, same three budgets. Compare optimizer vs value only within a
league: totals scale with second-half match counts (190 matches for
the full-season leagues, 34 for Bundesliga), so cross-league totals
are not comparable. The value baseline saturates below 0.30M for the
four full-season leagues (the same cheapest squad is feasible at all
three budgets, hence identical value totals); Ligue 1's optimizer
saturates between 0.50M and 0.70M. The optimizer beats value at 14 of
15 league-budget cells; the exception is Bundesliga at 0.30M, where
it trails value (40.83 vs 41.53).

| League | Budget | Optimizer | Value | Random mean | Oracle |
| :--- | ---: | ---: | ---: | ---: | ---: |
| English Premier League | 300k | 128.29 | 96.79 | 75.48 | 179.71 |
| English Premier League | 500k | 147.88 | 96.79 | 85.79 | 208.82 |
| English Premier League | 700k | 173.70 | 96.79 | 89.76 | 224.48 |
| La Liga | 300k | 128.55 | 98.15 | 85.86 | 183.11 |
| La Liga | 500k | 148.61 | 98.15 | 95.10 | 200.88 |
| La Liga | 700k | 159.68 | 98.15 | 102.04 | 210.75 |
| Serie A | 300k | 149.39 | 124.38 | 97.70 | 195.36 |
| Serie A | 500k | 157.73 | 124.38 | 98.20 | 209.43 |
| Serie A | 700k | 162.20 | 124.38 | 101.78 | 216.79 |
| Bundesliga | 300k | 40.83 | 41.53 | 8.23 | 55.29 |
| Bundesliga | 500k | 62.14 | 58.68 | 8.60 | 81.81 |
| Bundesliga | 700k | 82.23 | 79.43 | 12.57 | 103.55 |
| Ligue 1 | 300k | 122.56 | 82.98 | 86.08 | 187.32 |
| Ligue 1 | 500k | 150.41 | 82.98 | 90.78 | 207.07 |
| Ligue 1 | 700k | 150.41 | 82.98 | 93.79 | 216.69 |

## Squad and determinism checks

Every league, every budget: squad constraints hold (23 players,
position minimums, wage bill within budget) and clean regeneration
(selecting twice) yields identical player IDs and hashes.

## Squads (added 2026-10-27, owner-authorized follow-up)

`results.json` now carries a `squads` section alongside the existing
figures: per league per budget, the selected 23-man squad (player,
team, pos_group, vaep_per90, wage_eur per player) and the best XI
(11 names via the existing `select_xi` logic, formation 4-3-3, as
`scripts/make_dashboard.py` does). Every name comes from the run's
own rated/selected players; nothing is generated. All five leagues
at all three budgets have a full 23-man squad and an 11-name best
XI, and every best XI name is a member of its squad. Squad counts
match the existing `squad_checks` (23 per cell).

Diff-verified against the FV4 file: every field except the new
`squads` section (and `elapsed_s` timing) is identical, including
model hash `b86caaed47f745d4`, all scoreboards, frontiers, and
backtests. No selection, split, seed, config, budget, or number
changed. This is the data the dashboard squad view renders.

## Comparability labels

- These scoreboards supersede the FEATURE_VERSION 3 five-league
  numbers from the FIXED-v2 run (same protocol, same splits,
  FEATURE_VERSION 4 features). The FV3 numbers are retained in the
  delta tables above for comparison and are not the current figures.
- These scoreboards are NOT comparable to the v1.5.0 FC6 figures:
  different populations (per-league chronological splits vs the
  Bundesliga 130/32 split), different test sets, and corrected
  features (FC6 was computed pre-C1). The Bundesliga line above is
  the corrected-feature Bundesliga number on a new split, not a
  recomputation of FC6.
- These backtests are NOT comparable to the v1.5.0 Bundesliga
  backtest: different splits, different model, different ratings.
  Within-league optimizer-vs-value comparisons are the valid ones.
- Full machine-readable numbers, split ID hashes, and per-match
  summaries are in `results.json` next to this report.
