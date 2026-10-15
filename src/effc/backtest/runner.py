"""Split-season backtest: do first-half ratings pick second-half value?

Estimate everything on matches up to the split date, build squads, then
score them on strictly out-of-sample matches. Every benchmark respects the
same position quotas, so the comparison is apples-to-apples; the only
difference is the selection rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from effc.portfolio.portfolio import POS_QUOTA, select_squad


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


def _quota_sample(
    ratings: pd.DataFrame, rng: np.random.Generator, squad_size: int = 23
) -> pd.DataFrame:
    """Random squad satisfying position quotas."""
    picks = []
    for g, q in POS_QUOTA.items():
        pool = ratings[ratings["pos_group"] == g]
        picks.extend(rng.choice(pool.index.to_numpy(), size=min(q, len(pool)), replace=False))
    rest = ratings.drop(index=picks)
    need = squad_size - len(picks)
    if need > 0:
        picks.extend(rng.choice(rest.index.to_numpy(), size=need, replace=False))
    return ratings.loc[picks]


def random_squads(
    ratings: pd.DataFrame, budget: float, n: int = 20, seed: int = 0,
    squad_size: int = 23,
) -> list[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out = []
    wage = ratings["wage_eur"]
    for _ in range(n):
        for _try in range(300):
            squad = _quota_sample(ratings, rng, squad_size)
            if squad["wage_eur"].sum() <= budget:
                out.append(squad)
                break
    return out


def _top_with_quotas(
    ratings: pd.DataFrame, key: str, budget: float, squad_size: int = 23
) -> pd.DataFrame:
    """Greedy quota-respecting fill under a wage cap.

    Takes the best available players by `key` that fit the remaining
    budget, filling position quotas first. Uses the same cap the
    optimizer gets so the baselines are comparable.
    """
    picks: list = []
    remaining = float(budget)
    for g, q in POS_QUOTA.items():
        pool = ratings[ratings["pos_group"] == g].sort_values(key, ascending=False)
        taken = 0
        for idx, row in pool.iterrows():
            if taken >= q:
                break
            if row["wage_eur"] <= remaining:
                picks.append(idx)
                remaining -= row["wage_eur"]
                taken += 1
    rest = ratings.drop(index=picks).sort_values(key, ascending=False)
    for idx, row in rest.iterrows():
        if len(picks) >= squad_size:
            break
        if row["wage_eur"] <= remaining:
            picks.append(idx)
            remaining -= row["wage_eur"]
    return ratings.loc[picks]


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
        except RuntimeError:
            continue  # budget too tight for a valid 23-man squad; skip
        stars = _top_with_quotas(ratings_first, "vaep_per90", b)
        mb = ratings_first.copy()
        mb["vpe"] = mb["vaep_per90"] / mb["wage_eur"].clip(lower=1.0)
        value = _top_with_quotas(mb, "vpe", b)
        oracle = select_squad(ratings_second, b)
        rands = random_squads(ratings_first, b, seed=seed)
        rows.append(
            {
                "budget": b,
                "optimizer": squad_realized_score(opt, second_half_pm, minutes),
                "stars": squad_realized_score(stars, second_half_pm, minutes),
                "value": squad_realized_score(value, second_half_pm, minutes),
                "random_mean": float(np.mean(
                    [squad_realized_score(r, second_half_pm, minutes) for r in rands]
                )) if rands else 0.0,
                "oracle": squad_realized_score(oracle, second_half_pm, minutes),
            }
        )
    return pd.DataFrame(rows)
