# Changelog

## [Unreleased]

### Fixed
- Backtest baselines (stars, value) are now budget-aware. `_top_with_quotas` picked the top 23 by rating with no wage cap while the optimizer and oracle were capped, so the baselines posted identical scores at every budget. It is now a greedy quota-respecting fill under the same cap. Added regression tests: baselines never exceed a tight budget, quotas hold when the budget allows.
- Goal models now train on pre-split matches only. Training used all of `MODEL_POOL`, including the target second halves the backtest scores on. The pipeline now applies the same chronological split the backtest uses and drops post-split frames before training. Added a regression test that training data contains no post-split matches.

### Changed
- Reran the full pipeline with the fixes (goal models on 162 pre-split matches, stale VAEP cache invalidated). Backtest numbers updated in the README: value now beats the optimizer at both budgets under the same cap (16.37 vs 15.98 at 500k, 16.37 vs 13.51 at 700k); stars fall hard under tight caps.
