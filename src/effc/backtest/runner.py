"""Split-season backtest: do first-half ratings pick second-half value?

Estimate everything on matches up to the split date, build squads, then
score them on strictly out-of-sample matches. Every benchmark respects the
same position quotas, so the comparison is apples-to-apples; the only
difference is the selection rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from effc.portfolio.portfolio import POS_QUOTA, _milp_select, select_squad


def split_matches(matches: pd.DataFrame, frac: float = 0.5):
    """Time-split matches into first/second halves.

    Splits within each season when a season_id column is present, then pools
    the halves. This keeps the player pool overlapping between halves: a raw
    date split across seasons would rate one season's players and score them
    on another season's matches, where almost nobody overlaps.
    """
    matches = matches.sort_values("match_date").reset_index(drop=True)
    if "season_id" in matches.columns:
        firsts, seconds = [], []
        for _, g in matches.groupby("season_id", sort=False):
            g = g.sort_values("match_date").reset_index(drop=True)
            cut = int(len(g) * frac)
            firsts.append(g.iloc[:cut])
            seconds.append(g.iloc[cut:])
        first = pd.concat(firsts, ignore_index=True)
        second = pd.concat(seconds, ignore_index=True)
        return first, second
    cut = int(len(matches) * frac)
    return matches.iloc[:cut], matches.iloc[cut:]


def squad_realized_score(
    squad: pd.DataFrame, second_half_pm: pd.DataFrame, minutes: pd.DataFrame
) -> float:
    """Sum of squad members' realized second-half VAEP/90 (no shrinkage)."""
    ids = set(squad["player_id"].tolist())
    sh = second_half_pm[second_half_pm["player_id"].isin(ids)]
    if sh.empty:
        return 0.0
    m = minutes[minutes["match_id"].isin(sh["match_id"].unique())]
    tot_vaep = sh["vaep"].sum()
    tot_min = m[m["player_id"].isin(ids)]["minutes"].sum()
    if tot_min < 1e-9:
        return 0.0
    return float(tot_vaep / tot_min * 90.0 * len(ids))


def random_squads(
    ratings: pd.DataFrame, budget: float, n: int = 20, seed: int = 0,
    squad_size: int = 23,
) -> list[pd.DataFrame]:
    """Random baseline squads under the wage cap.

    Each squad maximizes a fresh random weight vector under the same
    quotas and budget the optimizer gets, so every squad is a valid
    23-man team and the rule uses no rating signal. Rejection sampling
    cannot reach tight budgets where feasible squads are rare; this
    always finds one when a feasible squad exists. Raises RuntimeError
    when no full squad fits the budget.
    """
    rng = np.random.default_rng(seed)
    wage = ratings["wage_eur"].to_numpy(dtype=float)
    groups = ratings["pos_group"].to_numpy()
    out = []
    for _ in range(n):
        sel = _milp_select(
            rng.random(len(ratings)), wage, groups, float(budget), squad_size,
            dict(POS_QUOTA), gamma=0.0, idio_var=np.zeros(len(ratings)),
        )
        out.append(ratings[sel].copy())
    return out


def _top_with_quotas(
    ratings: pd.DataFrame, key: str, budget: float, squad_size: int = 23
) -> pd.DataFrame:
    """Best 23 players by `key` that fit the wage cap.

    Maximizes the summed key under the same constraints the optimizer
    gets: position quotas, the wage budget, exactly 23 players. Solved
    exactly, so a tight budget marks the baseline infeasible (via the
    solver) instead of returning a short squad a greedy fill would get
    stuck on. Raises RuntimeError when no full squad fits.
    """
    mu = ratings[key].to_numpy(dtype=float)
    wage = ratings["wage_eur"].to_numpy(dtype=float)
    groups = ratings["pos_group"].to_numpy()
    sel = _milp_select(
        mu, wage, groups, float(budget), squad_size, dict(POS_QUOTA),
        gamma=0.0, idio_var=np.zeros(len(mu)),
    )
    return ratings[sel].copy()


def run_backtest(
    ratings_first: pd.DataFrame,
    ratings_second: pd.DataFrame,
    second_half_pm: pd.DataFrame,
    minutes: pd.DataFrame,
    budgets: list[float],
    seed: int = 0,
) -> pd.DataFrame:
    rows = []
    for b in budgets:
        try:
            opt = select_squad(ratings_first, b)
            mb = ratings_first.copy()
            mb["vpe"] = mb["vaep_per90"] / mb["wage_eur"].clip(lower=1.0)
            value = _top_with_quotas(mb, "vpe", b)
            oracle = select_squad(ratings_second, b)
            rands = random_squads(ratings_first, b, seed=seed)
        except RuntimeError:
            continue  # some rule cannot field a squad here; skip the budget
        rows.append(
            {
                "budget": b,
                "optimizer": squad_realized_score(opt, second_half_pm, minutes),
                "value": squad_realized_score(value, second_half_pm, minutes),
                "random_mean": float(np.mean(
                    [squad_realized_score(r, second_half_pm, minutes) for r in rands]
                )),
                "oracle": squad_realized_score(oracle, second_half_pm, minutes),
            }
        )
    return pd.DataFrame(rows)
