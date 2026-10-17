# Changelog

## [Unreleased]

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
