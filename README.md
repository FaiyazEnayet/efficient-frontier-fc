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

**Turn it into ratings.** Sum VAEP per player, divide by minutes, shrink toward position-group means (90 minutes of brilliance does not make you the best in the league). 268 rated players across Bundesliga 2023/24 and 2015/16.

**Price them.** EA FC 24 weekly wages, joined by normalized name matching (63% hit rate; FC 24 has no row for most 2015/16 players, and the misses get dropped). Same game data on both seasons, which is honest about the method and loose about the history.

**Optimize.** Binary portfolio: sign the player or do not. Maximize expected squad VAEP subject to wage budget, 23-man squad, and position quotas. One MILP per budget point.

**Backtest.** Split each season in half by date and pool the halves (splitting across seasons would rate one decade's players and score them on another's, where almost nobody overlaps). Ratings and squads from the first halves only, scored on the second halves. Benchmarks (stars, value picks, random) all respect the same quotas, so the only difference is the selection rule.

## What came out

Bundesliga 2023/24 and 2015/16, seed 0. Rerun `python -m effc.pipeline configs/bundesliga_2seasons.yaml` and check my work.

Top rated: Baumann 0.67, Wirtz 0.67, Mathenia 0.66 (VAEP/90, shrunk). Keepers sit on top, which says more about VAEP than about scouting: clean sheets and saves pile up value quietly. Among outfielders the names are familiar: Wirtz 0.67, Frimpong 0.61, Grimaldo 0.58, and the unbeaten 23/24 champions still dominate the list.

![Top 10 rated players by shrunk VAEP/90](<runs/bundesliga_2seasons/Top rated players chart.png>)

The efficient frontier:

| wage budget/week | expected squad VAEP/90 |
|---|---|
| EUR 0.30M | 10.55 |
| EUR 0.50M | 11.32 |
| EUR 0.70M | 11.79 |
| EUR 0.94M | 12.02 |
| EUR 1.50M | 12.06 |

Diminishing returns, then flat. Past ~0.94M/week the budget stops binding: 12.02 at 0.94M, 12.06 at 1.50M. The best XI (4-3-3) costs EUR 571k/week. That flat top is a result, not a bug: the optimal squad here is a Moneyball squad.

![Efficient frontier: expected squad VAEP/90 at each wage budget](<runs/bundesliga_2seasons/Efficient frontier chart.png>)

The backtest (realized second-half squad VAEP/90):

| budget/week | optimizer | stars | value | random | oracle |
|---|---|---|---|---|---|
| EUR 0.50M | 15.59 | 13.03 | 13.64 | 10.73 | 17.48 |
| EUR 0.70M | 13.14 | 13.03 | 13.64 | 9.92 | 17.25 |

The optimizer beats stars and random out of sample at both budgets and lands below the oracle, which cheats by seeing the second half. It loses to the value benchmark at 0.70M. That loss is real and I am not hiding it: at a looser budget the naive value-per-euro rule found a better squad than the optimizer's risk-blind MILP. More data narrows the gap but the honest read is that the optimizer's edge is clearest when money is tight.

![Backtest: realized second-half squad VAEP/90 by selection rule](<runs/bundesliga_2seasons/Backtest chart.png>)

## Data

- Events: StatsBomb open data (free). 196 matches pooled for the goal models (Bundesliga 23/24 and 15/16, World Cups 2018 and 2022); ratings on Bundesliga 2023/24 and 2015/16 (34 matches each; that is all StatsBomb publishes for these seasons).
- Wages: EA FC 24 database (mzafram2001/ea-fc, open source), weekly wages in EUR.
- `python -m effc.data.download` fetches everything into `data/` (gitignored, ~600MB).

## Reproduce

```bash
pip install -e ".[dev]"
python -m effc.data.download   # one-time, ~10 min
python -m effc.pipeline configs/bundesliga_2seasons.yaml
pytest                        # 16 tests
```

## Live refresh

A squad is a portfolio, so it should rebalance when new information arrives. `python -m effc.live_refresh` pulls finished fixtures from API-Football, turns match events into per-player value proxies, folds them into the ratings, re-runs the optimizer, and prints who enters and leaves the squad:

```bash
python -m effc.live_refresh --league 78 --season 2024 --budget 700000
```

Two things to know. First, no free feed publishes live event data, so true live VAEP is impossible without a paid feed. The proxy is deliberately crude: goals, assists, and cards, weighted to sit on the VAEP scale (a goal moves P(score) from ~0.1 to 1.0, hence 0.9). It blends into ratings with a half-life curve so a hot week moves the needle without rewriting a season. The weights live in `src/effc/live.py` and are a starting point, not science. Second, the free tier is small: seasons 2022-2024, about 2 days of date history, 100 requests/day. One daily refresh of a 9-fixture matchday costs ~20 calls.

Setup: get a free key at api-football.com and set `API_FOOTBALL_KEY`. Run it daily during a covered season and the squad rebalances itself.

## Honest limitations

- 68 Bundesliga matches across two seasons. Better than 34, still thin. Ratings lean on shrinkage, and the backtest halves are 34 matches each (17 per season, pooled). More data would tighten everything.
- Wages are video game wages. They correlate with real wages but they are not real wages. FC 24 wages are applied to both seasons, including 2015/16 players whose actual pay that year looked nothing like these numbers. The framework does not care where the cost vector comes from, but the euro figures for the older season are fiction.
- No chemistry, no injuries, no tactics. A squad is not 23 independent assets and I know it. The factor covariance model is a start, not an answer.
- CPU only, two seasons, one league.

## Reference

Decroos et al., *"Actions Speak Louder than Goals: Valuing Player Actions in Soccer"*, KDD 2019. The VAEP framework this builds on.

Issues and PRs welcome. If you run it on a full season, tell me where the frontier moves.
