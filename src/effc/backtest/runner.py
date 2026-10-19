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
    squad: pd.DataFrame, second_half_pm: pd.DataFrame
) -> tuple[float, int]:
    """Realized second-half total VAEP summed over squad members.

    The scored objective is the sum of what each selected player actually
    produced, not a pooled per-90 rate. This matches the selection
    objective (FC3): selection maximizes expected total VAEP
    (first-half minutes x VAEP/90), evaluation sums realized total VAEP.

    Returns (score, n_unavailable): players with no second-half action
    rows contribute 0 and are counted explicitly instead of silently
    dragging a pooled rate.
    """
    ids = set(squad["player_id"].tolist())
    sh = second_half_pm[second_half_pm["player_id"].isin(ids)]
    seen = set(sh["player_id"].unique().tolist())
    n_unavailable = len(ids - seen)
    if sh.empty:
        return 0.0, n_unavailable
    return float(sh["vaep"].sum()), n_unavailable


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
    """Backtest with a unified selection/evaluation objective (FC3).

    Selection maximizes expected total VAEP: first-half minutes times
    VAEP/90, i.e. the total goal-value the squad is expected to produce
    if second-half exposure matches first-half exposure. Evaluation sums
    realized second-half VAEP over squad members: the same quantity,
    measured. The value baseline maximizes the same objective per euro.
    The oracle maximizes realized second-half totals in-sample, so it
    genuinely solves the scored objective and upper-bounds by construction.
    """
    r1 = ratings_first.copy()
    r1["exp_contrib"] = (
        r1["minutes"].fillna(0.0) * r1["vaep_per90"] / 90.0
    )
    r2 = ratings_second.copy()
    # Oracle signal: raw realized second-half totals (no shrinkage), the
    # exact quantity the evaluation sums.
    r2["realized_total"] = r2["vaep"].fillna(0.0)

    rows = []
    for b in budgets:
        try:
            opt = select_squad(r1, b, value_col="exp_contrib")
            mb = r1.copy()
            mb["vpe"] = mb["exp_contrib"] / mb["wage_eur"].clip(lower=1.0)
            value = _top_with_quotas(mb, "vpe", b)
            oracle = select_squad(r2, b, value_col="realized_total")
            rands = random_squads(r1, b, seed=seed)
        except RuntimeError:
            continue  # some rule cannot field a squad here; skip the budget
        opt_score, opt_un = squad_realized_score(opt, second_half_pm)
        val_score, val_un = squad_realized_score(value, second_half_pm)
        orc_score, orc_un = squad_realized_score(oracle, second_half_pm)
        rand_scores, rand_uns = zip(
            *[squad_realized_score(r, second_half_pm) for r in rands]
        )
        rows.append(
            {
                "budget": b,
                "optimizer": opt_score,
                "value": val_score,
                "random_mean": float(np.mean(rand_scores)),
                "oracle": orc_score,
                "optimizer_unavail": opt_un,
                "value_unavail": val_un,
                "random_unavail_mean": float(np.mean(rand_uns)),
                "oracle_unavail": orc_un,
            }
        )
    return pd.DataFrame(rows)
