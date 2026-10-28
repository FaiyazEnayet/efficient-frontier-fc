# ARCHIVED notebooks (historical, not active)

The three notebooks in this directory are historical artifacts from the
v1.5.0 era, before the five-league expansion. They are kept byte-identical
for provenance and must not be used as walkthroughs: they do not run in a
clean checkout, and their claims contradict the published results.

- 02_vaep.ipynb: trains the goal models on pooled frames loaded from the
  unversioned cache file pool_frames.pkl, then evaluates Brier scores on
  those same pooled frames and reads the in-sample score as calibration.
- 03_frontier.ipynb: loads the retired cache file target_vaep.pkl, which is
  not in the repo, and narrates a single-league frontier.
- 04_backtest.ipynb: loads the retired target_vaep.pkl, references the
  removed stars baseline, and claims the optimizer wins at every budget.

What supersedes them: runs/five_leagues/results.json and
runs/five_leagues/report.md, the five-league section of the README, and
docs/dashboard.html.

Current published numbers (five-league FV4 run, FEATURE_VERSION 4):

- Backtest: 14 of 15 league-budget cells. The single loss is Bundesliga at
  300,000 EUR/week (optimizer 40.83 vs value 41.53).
- Bundesliga wage match rate: 63.17% (README rounds this to 63.2%).
- Budgets: 300,000, 500,000 and 700,000 EUR/week in all five leagues.

Note on an older figure: the v1.5.0 era audit described the published loss
as a 500,000 EUR loss. That figure is superseded. In the current five-league
backtest the 500,000 EUR budget is a win in all five leagues; the only loss
is Bundesliga at 300,000 EUR/week, stated above.
