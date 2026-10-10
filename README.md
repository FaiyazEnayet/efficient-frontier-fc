# Efficient Frontier FC

**Squad building as portfolio optimization.** Players are assets, expected on-ball value is the return, the weekly wage is the cost. Integer programming builds the best 23-man squad at each budget. The result is an efficient frontier for football.

I kept noticing that football analytics stops at describing players (xG, ratings, radars) and never does the thing finance does with asset estimates: optimize under constraints. So I built the missing step. Estimate value honestly, then ask the optimizer what the best team money can buy looks like, and check whether it holds up out of sample.

```python
from effc.portfolio.portfolio import efficient_frontier, select_squad
import numpy as np

budgets = np.array([300_000, 500_000, 700_000, 943_000, 1_500_000])
frontier = efficient_frontier(ratings, budgets)  # MILP per budget
```

## How it works

**Value every touch (VAEP).** 437k on-ball actions from 196 matches of StatsBomb event data. Two gradient-boosted models learn P(team scores) and P(team concedes) in the next 10 actions from the game state. An action's value is the change it causes, plus 1 for a scored goal, minus 1 for a conceded one. The two label windows are asymmetric on purpose: the "before" window includes the current action so chances count, the "after" window excludes it so a scored goal does not poison the state that follows. Getting this wrong gives goals zero value or keepers infinite value. I got it wrong twice.

**Turn it into ratings.** Sum VAEP per player, divide by minutes, shrink toward position-group means (90 minutes of brilliance does not make you the best in the league). 200 rated Bundesliga 2023/24 players.

**Price them.** EA FC 24 weekly wages, joined by normalized name matching (88% hit rate; the misses are mostly kids with no minutes). Same season on both sides.

**Optimize.** Binary portfolio: sign the player or do not. Maximize expected squad VAEP subject to wage budget, 23-man squad, and position quotas. One MILP per budget point.

**Backtest.** Split the season in half by date. Ratings and squads from the first half only, scored on the second half. Benchmarks (stars, value picks, random) all respect the same quotas, so the only difference is the selection rule.

## What came out

Bundesliga 2023/24, seed 0. Rerun `python -m effc.pipeline` and check my work.

Top rated: Wirtz 0.66, Frimpong 0.61, Grimaldo 0.58 (VAEP/90, shrunk). The unbeaten champions dominate, which is either the model working or the model being obvious. The interesting part is the wages next to them: 77k, 63k, 88k a week.

The efficient frontier:

| wage budget/week | expected squad VAEP/90 |
|---|---|
| EUR 0.30M | 8.40 |
| EUR 0.50M | 9.53 |
| EUR 0.70M | 10.23 |
| EUR 0.94M | 10.70 |
| EUR 1.50M | 10.77 |

Diminishing returns, then flat. Past ~0.94M/week the budget stops binding: the unconstrained best 23 costs that much, and money cannot buy more VAEP in this pool. The best XI (4-3-3) costs EUR 672k/week. That flat top is a result, not a bug: the optimal squad here is a Moneyball squad.

The backtest (realized second-half squad VAEP/90):

| budget/week | optimizer | stars | value | random | oracle |
|---|---|---|---|---|---|
| EUR 0.50M | 14.70 | 12.23 | 6.10 | 8.51 | 18.01 |
| EUR 0.70M | 12.93 | 12.23 | 6.10 | 7.85 | 16.69 |

The optimizer beats every benchmark out of sample and lands below the oracle, which cheats by seeing the second half. That ordering is exactly what an honest method should produce.

## Data

- Events: StatsBomb open data (free). 196 matches pooled for the goal models (Bundesliga 23/24 and 15/16, World Cups 2018 and 2022); ratings on Bundesliga 2023/24 (34 matches in the open release).
- Wages: EA FC 24 database (mzafram2001/ea-fc, open source), weekly wages in EUR.
- `python -m effc.data.download` fetches everything into `data/` (gitignored, ~600MB).

## Reproduce

```bash
pip install -e ".[dev]"
python -m effc.data.download   # one-time, ~10 min
python -m effc.pipeline configs/bundesliga_2324.yaml
pytest                        # 9 tests
```

## Live refresh

A squad is a portfolio, so it should rebalance when new information arrives. `python -m effc.live_refresh` pulls finished fixtures from API-Football, turns match events into per-player value proxies, folds them into the ratings, re-runs the optimizer, and prints who enters and leaves the squad:

```bash
python -m effc.live_refresh --league 78 --season 2024 --budget 700000
```

Two things to know. First, no free feed publishes live event data, so true live VAEP is impossible without a paid feed. The proxy is deliberately crude: goals, assists, and cards, weighted to sit on the VAEP scale (a goal moves P(score) from ~0.1 to 1.0, hence 0.9). It blends into ratings with a half-life curve so a hot week moves the needle without rewriting a season. The weights live in `src/effc/live.py` and are a starting point, not science. Second, the free tier is small: seasons 2022-2024, about 2 days of date history, 100 requests/day. One daily refresh of a 9-fixture matchday costs ~20 calls.

Setup: get a free key at api-football.com and set `API_FOOTBALL_KEY`. Run it daily during a covered season and the squad rebalances itself.

## Honest limitations

- 34 Bundesliga matches is thin. Ratings lean heavily on shrinkage, and the backtest halves are 17 matches each. More data would tighten everything.
- Wages are video game wages. They correlate with real wages but they are not real wages. The framework does not care where the cost vector comes from.
- No chemistry, no injuries, no tactics. A squad is not 23 independent assets and I know it. The factor covariance model is a start, not an answer.
- CPU only, single season, one league.

## Reference

Decroos et al., *"Actions Speak Louder than Goals: Valuing Player Actions in Soccer"*, KDD 2019. The VAEP framework this builds on.

Issues and PRs welcome. If you run it on a full season, tell me where the frontier moves.
