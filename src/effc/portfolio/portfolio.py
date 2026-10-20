"""Squad construction as constrained selection (the quant core).

A squad is a binary portfolio: x_i = 1 if the player is signed. The
efficient frontier traces max expected VAEP against the wage budget.
Risk enters as a penalty on idiosyncratic variance (linear, so the
problem stays a MILP); the full factor-model risk is reported ex post.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

SQUAD_SIZE = 23
POS_QUOTA = {"GK": 2, "DEF": 6, "MID": 6, "FWD": 4}  # minimums; sum = 18 < 23

FORMATIONS = {
    "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
    "4-2-3-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},
    "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
}


class InfeasibleError(RuntimeError):
    """The requested squad cannot be built.

    `reason` is a short machine-readable code; the message explains it.
    Subclasses RuntimeError so existing `except RuntimeError` handlers
    (e.g. the frontier's skip-on-infeasible-budget) keep working.
    """

    EMPTY_GROUP = "empty_group"
    QUOTA_EXCEEDS_SQUAD = "quota_exceeds_squad"
    QUOTAS_EXCEED_SQUAD = "quotas_exceed_squad"
    POOL_TOO_SMALL = "pool_too_small"
    MILP_INFEASIBLE = "milp_infeasible"

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        msg = f"infeasible squad ({reason})"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


def _validate_inputs(mu, wage, groups, budget, n_pick, quota, idio_var):
    """Fail fast on inputs that can never yield a valid squad.

    Every requested quota is enforced: a quota on an empty group is an
    error, never a silently skipped constraint.
    """
    n = len(mu)
    if n_pick <= 0:
        raise ValueError(f"squad size must be positive, got {n_pick}")
    if n_pick > n:
        raise InfeasibleError(
            InfeasibleError.POOL_TOO_SMALL,
            f"need {n_pick} players from a pool of {n}",
        )
    for name, arr in (("vaep_per90", mu), ("wage_eur", wage), ("idio_var", idio_var)):
        if np.isnan(np.asarray(arr, dtype=float)).any():
            raise ValueError(f"NaN in {name}: cannot optimize on missing values")
    if np.isnan(budget):
        raise ValueError("NaN budget")
    total_min = 0
    for g, q in quota.items():
        if q < 0:
            raise ValueError(f"negative quota for {g}: {q}")
        if q > n_pick:
            raise InfeasibleError(
                InfeasibleError.QUOTA_EXCEEDS_SQUAD,
                f"{g} minimum {q} exceeds squad size {n_pick}",
            )
        n_group = int((groups == g).sum())
        if q > 0 and n_group == 0:
            raise InfeasibleError(
                InfeasibleError.EMPTY_GROUP,
                f"quota needs {q} {g} but the pool has no {g} players",
            )
        total_min += q
    if total_min > n_pick:
        raise InfeasibleError(
            InfeasibleError.QUOTAS_EXCEED_SQUAD,
            f"positional minima sum to {total_min} but squad size is {n_pick}",
        )


def _milp_select(
    mu: np.ndarray,
    wage: np.ndarray,
    groups: np.ndarray,
    budget: float,
    n_pick: int,
    quota: dict[str, int],
    gamma: float,
    idio_var: np.ndarray,
) -> np.ndarray:
    _validate_inputs(mu, wage, groups, budget, n_pick, quota, idio_var)
    n = len(mu)
    c = -(mu - gamma * idio_var)  # minimize negative utility
    A_rows, lb, ub = [], [], []
    A_rows.append(wage)
    lb.append(0.0)
    ub.append(budget)
    ones = np.ones(n)
    A_rows.append(ones)
    lb.append(n_pick)
    ub.append(n_pick)
    for g, q in quota.items():
        # Every quota is enforced. Validation above guarantees a
        # positive quota always has candidates; nothing is skipped.
        row = (groups == g).astype(float)
        A_rows.append(row)
        lb.append(q)
        ub.append(n_pick)
    A = np.vstack(A_rows)
    res = milp(
        c,
        constraints=LinearConstraint(A, np.array(lb), np.array(ub)),
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
    )
    if not res.success:
        raise InfeasibleError(
            InfeasibleError.MILP_INFEASIBLE,
            f"{res.message} (budget={budget}, squad_size={n_pick})",
        )
    return (res.x > 0.5).astype(bool)


def select_squad(
    ratings: pd.DataFrame,
    budget: float,
    squad_size: int = SQUAD_SIZE,
    quota: dict[str, int] | None = None,
    gamma: float = 0.0,
    idio_var: pd.Series | None = None,
    value_col: str = "vaep_per90",
) -> pd.DataFrame:
    """Optimal squad under a wage budget. Returns selected rows.

    Maximizes the summed ``value_col``. The default (``vaep_per90``) is the
    per-90 rate objective used by the efficient frontier. The backtest
    (FC3) passes an exposure-weighted column so selection and evaluation
    share one objective: expected total VAEP.
    """
    quota = quota or POS_QUOTA
    mu = ratings[value_col].to_numpy(dtype=float)
    wage = ratings["wage_eur"].to_numpy()
    groups = ratings["pos_group"].to_numpy()
    iv = (
        idio_var.reindex(ratings["player_id"]).to_numpy()
        if idio_var is not None
        else np.zeros(len(mu))
    )
    sel = _milp_select(mu, wage, groups, budget, squad_size, quota, gamma, iv)
    return ratings[sel].copy()


def efficient_frontier(
    ratings: pd.DataFrame,
    budgets: np.ndarray,
    gamma: float = 0.0,
    idio_var: pd.Series | None = None,
    squad_size: int = SQUAD_SIZE,
) -> pd.DataFrame:
    """Max expected squad VAEP/90 at each wage budget."""
    rows = []
    for b in budgets:
        try:
            squad = select_squad(ratings, float(b), squad_size, gamma=gamma, idio_var=idio_var)
        except RuntimeError:
            continue
        rows.append(
            {
                "budget": b,
                "exp_vaep_per90": squad["vaep_per90"].sum(),
                "wage_bill": squad["wage_eur"].sum(),
                "n_players": len(squad),
            }
        )
    return pd.DataFrame(rows)


def select_xi(
    ratings: pd.DataFrame, formation: str = "4-3-3", gamma: float = 0.0,
    idio_var: pd.Series | None = None,
) -> pd.DataFrame:
    """Best XI for a formation: exact position counts, max expected VAEP."""
    quota = FORMATIONS[formation]
    mu = ratings["vaep_per90"].to_numpy()
    groups = ratings["pos_group"].to_numpy()
    iv = (
        idio_var.reindex(ratings["player_id"]).to_numpy()
        if idio_var is not None
        else np.zeros(len(mu))
    )
    n = len(mu)
    for name, arr in (("vaep_per90", mu), ("idio_var", iv)):
        if np.isnan(np.asarray(arr, dtype=float)).any():
            raise ValueError(f"NaN in {name}: cannot optimize on missing values")
    for g, q in quota.items():
        if q > 0 and int((groups == g).sum()) == 0:
            raise InfeasibleError(
                InfeasibleError.EMPTY_GROUP,
                f"formation needs {q} {g} but the pool has no {g} players",
            )
    c = -(mu - gamma * iv)
    A_rows, lb, ub = [], [], []
    for g, q in quota.items():
        A_rows.append((groups == g).astype(float))
        lb.append(q)
        ub.append(q)
    A = np.vstack(A_rows)
    res = milp(
        c,
        constraints=LinearConstraint(A, np.array(lb), np.array(ub)),
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
    )
    if not res.success:
        raise InfeasibleError(InfeasibleError.MILP_INFEASIBLE, f"XI: {res.message}")
    return ratings[res.x > 0.5].copy()


def squad_risk(selection: pd.DataFrame, Sigma: pd.DataFrame) -> float:
    """Ex-post squad risk from the full factor covariance: sqrt(w' Sigma w)
    with equal weights over selected players."""
    ids = selection["player_id"].to_numpy()
    ids = [i for i in ids if i in Sigma.index]
    if not ids:
        return float("nan")
    w = np.ones(len(ids)) / len(ids)
    S = Sigma.loc[ids, ids].to_numpy()
    return float(np.sqrt(w @ S @ w))
