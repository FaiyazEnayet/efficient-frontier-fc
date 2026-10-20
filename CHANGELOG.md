# Changelog

## [v1.4.0] - 2026-10-19
### Fixed
- Restored the FC2 quota validation the FC3 rewrite accidentally deleted (`src/effc/portfolio/portfolio.py`). The FC3 objective change rewrote `_milp_select`/`select_xi` and dropped the `InfeasibleError` class, `_validate_inputs` and the formation checks, so impossible quotas, empty groups, NaN inputs and MILP failures degraded to generic errors or silently skipped constraints. Restored verbatim (keeping FC3's `value_col`); the six FC2 regression tests pass again.
- `scripts/make_demo.py` was broken: it called `compute_target_vaep(None)` after FC1 changed the signature, and its backtest melt leaked the FC3 `*_unavail` diagnostic columns into the demo rules. It now reuses the pipeline's `prepare_model_data` and melts only the rule score columns.

### Added
- Held-out match-group calibration for the goal-probability models (`src/effc/vaep/calibration.py`, `src/effc/pipeline.py`). The pipeline trains on 130 matches and scores 32 held-out matches (chronological within each competition, disjoint match IDs): Brier 0.0099 (95% CI 0.0081-0.0117) for P(scores) vs 0.0113 for a base-rate baseline; 0.0030 vs 0.0033 for P(concedes). The old headline Brier is now explicitly labeled in-sample (`brier_*_in_sample`: 0.0079/0.0018).
- Chronological season-transfer evaluation: goal models trained on 2015/16, scored on 2023/24. Brier 0.0110 vs baseline 0.0107 for scores (0.0023 vs 0.0022 for concedes): the model does not beat the baseline eight seasons later. Roster stability: only 13% of 2023/24 players (24% of minutes) appeared in 2015/16.

### Changed
- Reran the full pipeline (first regeneration since before FC1): metrics and demo tables now reflect the FC3 unified total-VAEP objective and the FC6 train-split models. Backtest (total VAEP): the optimizer beats the value baseline at 0.50M (58.26 vs 57.11) and 0.70M (90.34 vs 84.22), trails at 0.30M (43.35 vs 44.12). README evidence table refreshed.

### Tests
- 12 new regression tests in `tests/test_calibration.py`; full suite 66/66 green.

## [v1.3.0] - 2026-10-19
### Fixed
- Selection and evaluation now share one objective: exposure-weighted expected total VAEP (`src/effc/backtest/runner.py`, `src/effc/portfolio/portfolio.py`). `select_squad` used to maximize summed per-90 rates while `squad_realized_score` measured a pooled minutes-weighted rate times squad size, so the "oracle" optimized a different objective than the one it was scored on and need not upper-bound anything. Selection now maximizes expected total VAEP (first-half minutes x VAEP/90); evaluation sums realized second-half VAEP over squad members. The value baseline maximizes the same objective per euro. The oracle maximizes realized second-half totals in-sample, so it genuinely solves the scored objective and upper-bounds by construction. Players with no second-half action contribute 0 and are counted explicitly (`optimizer_unavail` etc.) instead of silently dragging a pooled rate. Added regression tests: unequal-minutes hand fixture matches the summed objective, oracle upper-bounds, unavailable players counted.

### Tests
- 4 new regression tests in `tests/test_backtest.py`; full suite 47/47 green.

## [v1.2.1] - 2026-10-18
### Fixed
- Impossible position quotas are now rejected instead of silently skipped (`src/effc/portfolio/portfolio.py`). `_milp_select` used to drop a quota when the player pool had zero members of that group, so a goalkeeper minimum on a forward-only pool returned forwards instead of infeasibility. Quotas are validated against available candidates and squad size before solving, and violations raise a typed `InfeasibleError` carrying a machine-readable reason (`empty_group`, `quota_exceeds_squad`, `quotas_exceed_squad`, `pool_too_small`, `milp_infeasible`); it subclasses `RuntimeError` so existing handlers keep working. NaN wages, ratings or budgets raise clear `ValueError`s. Added regression tests: no-GK pool with GK>=1 fails, quota/squad-size contradictions fail, feasible squads meet exact size, wage cap and all minima, NaN inputs fail loudly.

## [v1.2.0] - 2026-10-18
### Fixed
- VAEP caches are now keyed by model and split (`src/effc/pipeline.py`). `compute_target_vaep` used to load a single fixed `target_vaep.pkl` no matter which model was supplied, so changing the seed, training matches, features or split silently reused another model's values. The cache filename is now versioned and content-addressed from the sorted training match IDs, the feature schema version (`FEATURE_VERSION`), the run config and a hash of the fitted models; legacy unversioned `target_vaep.pkl` files are rejected, never loaded. Added regression tests: two different dummy models never share a cache, config/train-ID changes invalidate it, identical inputs reuse it exactly.

## [v1.1.0] - 2026-10-16

### Fixed
- Backtest baselines are now exact constrained selections instead of a greedy fill. The greedy `_top_with_quotas` got trapped by tight budgets and returned short squads (7 players at 300k, 11 at 500k), whose scores were reported next to full 23-man squads. Baselines now maximize their key under the identical quotas and wage cap via the same MILP the optimizer uses, and raise when no full squad fits. Added regression tests: full squads where the greedy got stuck, RuntimeError on infeasible budgets.
- The random baseline no longer reports a silent 0.0. Rejection sampling could not reach tight budgets (it reported 0.0 at 300k), so random squads are now built by maximizing random weights under the same constraints. A budget where any rule is infeasible is skipped instead of faked. Added a regression test.
- The oracle is no longer labeled a guaranteed ceiling. It re-runs the optimizer on second-half ratings (perfect foresight of the ratings), but it maximizes expected VAEP under the model, not the realized metric, and the optimizer beat it at 300k (16.87 vs 16.76).

### Removed
- The stars baseline. With the risk penalty off, the optimizer already maximizes raw VAEP/90, so an exact stars rule solves the identical constrained problem and returns the identical squad (verified to 6 decimals). Keeping it would print one squad twice. The value baseline (max VAEP per euro) is a genuinely different selection rule and stays.

### Fixed
- Backtest baselines (stars, value) are now budget-aware. `_top_with_quotas` picked the top 23 by rating with no wage cap while the optimizer and oracle were capped, so the baselines posted identical scores at every budget. It is now a greedy quota-respecting fill under the same cap. Added regression tests: baselines never exceed a tight budget, quotas hold when the budget allows.
- Goal models now train on pre-split matches only. Training used all of `MODEL_POOL`, including the target second halves the backtest scores on. The pipeline now applies the same chronological split the backtest uses and drops post-split frames before training. Added a regression test that training data contains no post-split matches.

### Changed
- Reran the full pipeline with the fixes (goal models on 162 pre-split matches, stale VAEP cache invalidated). Backtest numbers updated in the README: value now beats the optimizer at both budgets under the same cap (16.37 vs 15.98 at 500k, 16.37 vs 13.51 at 700k); stars fall hard under tight caps.
