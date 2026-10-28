"""Frontier stability, marginal value of budget and interpretable tradeoffs.

Stage 3 U12 diagnostics over the exact MILP squad selector
(`effc.portfolio.portfolio.select_squad`). Everything reported here is
computed from exact re-solves at stated budgets, never from a relaxation:

- Marginal value is the finite difference of the exact MILP objective
  between adjacent budgets. It is NOT an LP relaxation dual and must
  never be labelled a shadow price.
- A breakpoint is a budget at which the optimal selection changes. The
  diagnostics reconcile exactly: re-solving at a breakpoint budget must
  reproduce the reported selection set, and `reconcile_breakpoints`
  reports any mismatch instead of hiding it.
- Monotonicity is claimed only for pure max-value solves (gamma = 0),
  where the reported objective equals summed value. With a risk penalty
  (gamma > 0) the penalized objective is monotone, not the raw value sum,
  so `objective` and `value` are kept as separate fields.
- Selection frequency is computed over caller-supplied rating variants
  (for example training-side bootstrap refits of the VAEP ratings). This
  module does not refit ratings and does not invent a rating-uncertainty
  model; variants are inputs, and their provenance is the caller's.
- Nothing here projects realized future gains. A frontier point is an
  in-sample optimum at a stated budget, not a forward test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from effc.portfolio.portfolio import POS_QUOTA, select_squad

MONOTONICITY_TOL = 1e-8
WAGE_TOL = 1e-6


@dataclass(frozen=True)
class FrontierPoint:
    """One exact MILP optimum at one budget."""

    budget: float
    objective: float
    value: float
    wage_bill: float
    slack: float
    n_players: int
    position_counts: dict
    player_ids: tuple


@dataclass
class FrontierResult:
    """All feasible points on a budget grid, plus skipped budgets."""

    points: list = field(default_factory=list)
    infeasible_budgets: list = field(default_factory=list)


@dataclass
class SelectionFrequency:
    """Selection frequency across rating variants, per budget."""

    frequencies: dict = field(default_factory=dict)
    mean_jaccard: dict = field(default_factory=dict)


def _idio_by_id(ratings, idio_var):
    if idio_var is None:
        return {}
    return dict(zip(ratings["player_id"], idio_var.reindex(ratings["player_id"]).to_numpy(dtype=float)))


def _point_from_selection(selection, budget, gamma, value_col, idio_by_id):
    value = float(selection[value_col].sum())
    wage_bill = float(selection["wage_eur"].sum())
    idio_sum = float(sum(idio_by_id.get(pid, 0.0) for pid in selection["player_id"]))
    objective = value - gamma * idio_sum
    counts = selection["pos_group"].value_counts().to_dict()
    return FrontierPoint(
        budget=float(budget),
        objective=objective,
        value=value,
        wage_bill=wage_bill,
        slack=float(budget) - wage_bill,
        n_players=len(selection),
        position_counts={str(k): int(v) for k, v in counts.items()},
        player_ids=tuple(sorted(selection["player_id"].astype(str))),
    )


def solve_frontier(
    ratings,
    budgets,
    squad_size=23,
    quota=None,
    gamma=0.0,
    value_col="vaep_per90",
    idio_var=None,
):
    """Re-solve the exact MILP at each budget, ascending.

    Budgets are sorted ascending and deduplicated (documented: the
    frontier is defined on an increasing nested grid). Budgets with no
    feasible squad are recorded in `infeasible_budgets`, never silently
    dropped. Bad inputs (NaN values, missing player_id column) raise
    instead of producing a point: a diagnostic must not launder bad data.
    """
    if "player_id" not in ratings.columns:
        raise ValueError("ratings must carry a player_id column")
    budgets = sorted(set(float(b) for b in budgets))
    if not budgets:
        raise ValueError("budgets must be non-empty")
    quota = quota if quota is not None else POS_QUOTA
    idio = _idio_by_id(ratings, idio_var)
    points, infeasible = [], []
    for budget in budgets:
        try:
            selection = select_squad(
                ratings, budget, squad_size=squad_size, quota=quota,
                gamma=gamma, idio_var=idio_var, value_col=value_col,
            )
        except RuntimeError:
            infeasible.append(budget)
            continue
        points.append(_point_from_selection(selection, budget, gamma, value_col, idio))
    return FrontierResult(points=points, infeasible_budgets=infeasible)


def check_monotonicity(points, tol=MONOTONICITY_TOL):
    """Objective drops larger than `tol` between adjacent budgets.

    Returns a list of (from_budget, to_budget, drop) tuples; empty means
    the frontier is monotone. For pure max-value solves (gamma = 0) the
    objective equals summed value, so this is value monotonicity.
    """
    ordered = sorted(points, key=lambda p: p.budget)
    violations = []
    for prev, cur in zip(ordered, ordered[1:]):
        drop = prev.objective - cur.objective
        if drop > tol:
            violations.append((prev.budget, cur.budget, drop))
    return violations


def check_feasibility(points, quota, squad_size, wage_tol=WAGE_TOL):
    """Cap, squad-size and positional-minimum violations per point.

    Returns a list of human-readable violation strings; empty means every
    point satisfies its cap and quotas. `quota=None` uses the package
    default positional minima.
    """
    quota = quota if quota is not None else POS_QUOTA
    violations = []
    for p in points:
        if p.wage_bill > p.budget + wage_tol:
            violations.append(
                f"budget {p.budget}: wage bill {p.wage_bill} exceeds cap"
            )
        if p.n_players != squad_size:
            violations.append(
                f"budget {p.budget}: squad size {p.n_players} != {squad_size}"
            )
        for group, minimum in quota.items():
            if minimum > 0 and p.position_counts.get(group, 0) < minimum:
                violations.append(
                    f"budget {p.budget}: {group} count "
                    f"{p.position_counts.get(group, 0)} below minimum {minimum}"
                )
    return violations


def jaccard(a, b):
    """Jaccard similarity between two selections (sets of player ids)."""
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def breakpoints(points):
    """Adjacent budget pairs where the optimal selection changes.

    Each entry carries the entering and leaving player ids, the Jaccard
    similarity, and the value, objective and wage changes. Pairs with an
    unchanged selection are not breakpoints.
    """
    ordered = sorted(points, key=lambda p: p.budget)
    out = []
    for prev, cur in zip(ordered, ordered[1:]):
        if prev.player_ids == cur.player_ids:
            continue
        prev_set, cur_set = set(prev.player_ids), set(cur.player_ids)
        out.append(
            {
                "from_budget": prev.budget,
                "to_budget": cur.budget,
                "entering": tuple(sorted(cur_set - prev_set)),
                "leaving": tuple(sorted(prev_set - cur_set)),
                "jaccard": jaccard(prev.player_ids, cur.player_ids),
                "value_change": cur.value - prev.value,
                "objective_change": cur.objective - prev.objective,
                "wage_change": cur.wage_bill - prev.wage_bill,
            }
        )
    return out


def marginal_value_table(points):
    """Marginal integer value per adjacent budget pair.

    `marginal_per_eur` is the finite difference of the exact MILP
    objective between adjacent re-solves, divided by the budget step.
    It is an integer-program marginal, not a shadow price: no LP dual
    is computed anywhere in this module.
    """
    ordered = sorted(points, key=lambda p: p.budget)
    out = []
    for prev, cur in zip(ordered, ordered[1:]):
        step = cur.budget - prev.budget
        out.append(
            {
                "from_budget": prev.budget,
                "to_budget": cur.budget,
                "value_change": cur.value - prev.value,
                "objective_change": cur.objective - prev.objective,
                "budget_change": step,
                "marginal_per_eur": (cur.objective - prev.objective) / step if step else 0.0,
                "entering": tuple(sorted(set(cur.player_ids) - set(prev.player_ids))),
                "leaving": tuple(sorted(set(prev.player_ids) - set(cur.player_ids))),
                "jaccard": jaccard(prev.player_ids, cur.player_ids),
            }
        )
    return out


def breakpoint_intervals(points):
    """Contiguous budget runs owned by each distinct selection.

    Each entry gives the selection, the first and last grid budgets where
    it is optimal, and how many grid budgets it covers. The intervals
    tile the feasible grid exactly.
    """
    ordered = sorted(points, key=lambda p: p.budget)
    intervals = []
    for p in ordered:
        if intervals and intervals[-1]["player_ids"] == p.player_ids:
            intervals[-1]["last_budget"] = p.budget
            intervals[-1]["n_budgets"] += 1
        else:
            intervals.append(
                {
                    "player_ids": p.player_ids,
                    "first_budget": p.budget,
                    "last_budget": p.budget,
                    "n_budgets": 1,
                }
            )
    return intervals


def reconcile_breakpoints(
    ratings,
    points,
    squad_size=23,
    quota=None,
    gamma=0.0,
    value_col="vaep_per90",
    idio_var=None,
):
    """Re-solve at every point budget and compare selections exactly.

    Returns a list of mismatches (budget, expected ids, actual ids).
    Empty means the reported diagnostics reconcile exactly to fresh
    solves: no breakpoint claim survives a mismatch.
    """
    quota = quota if quota is not None else POS_QUOTA
    mismatches = []
    for p in sorted(points, key=lambda q: q.budget):
        try:
            selection = select_squad(
                ratings, p.budget, squad_size=squad_size, quota=quota,
                gamma=gamma, idio_var=idio_var, value_col=value_col,
            )
        except RuntimeError:
            mismatches.append(
                {"budget": p.budget, "expected": p.player_ids, "actual": None}
            )
            continue
        actual = tuple(sorted(selection["player_id"].astype(str)))
        if actual != p.player_ids:
            mismatches.append(
                {"budget": p.budget, "expected": p.player_ids, "actual": actual}
            )
    return mismatches


def binding_minima(point, quota):
    """Positions whose selected count sits exactly at the quota minimum.

    Only positive minima can bind; a zero minimum never binds. Returned
    sorted by position name.
    """
    quota = quota if quota is not None else POS_QUOTA
    return sorted(
        group
        for group, minimum in quota.items()
        if minimum > 0 and point.position_counts.get(group, 0) == minimum
    )


def cheap_vs_high_impact(selection, value_col="vaep_per90"):
    """Cheapest, highest-value and best/worst value-per-wage selections.

    `selection` is the selected-players frame (one row per player with
    player_id, wage_eur and the value column). All four extremes are
    reported as player ids with their numbers, so a tradeoff claim can
    be checked against the frame directly.
    """
    if len(selection) == 0:
        raise ValueError("selection must be non-empty")
    wages = selection["wage_eur"].to_numpy(dtype=float)
    values = selection[value_col].to_numpy(dtype=float)
    ids = selection["player_id"].astype(str).to_numpy()
    per_wage = values / wages
    cheapest = int(np.argmin(wages))
    highest = int(np.argmax(values))
    best = int(np.argmax(per_wage))
    worst = int(np.argmin(per_wage))
    return {
        "cheapest_id": ids[cheapest],
        "cheapest_wage": float(wages[cheapest]),
        "highest_value_id": ids[highest],
        "highest_value": float(values[highest]),
        "best_value_per_wage_id": ids[best],
        "best_value_per_wage": float(per_wage[best]),
        "worst_value_per_wage_id": ids[worst],
        "worst_value_per_wage": float(per_wage[worst]),
    }


def selection_frequency(
    variant_frames,
    budgets,
    squad_size=23,
    quota=None,
    gamma=0.0,
    value_col="vaep_per90",
    idio_var=None,
):
    """Selection frequency across caller-supplied rating variants.

    Each variant frame is one training-side bootstrap refit of the
    ratings (same player universe expected; a player missing from a
    variant counts as unselected in it). For each budget, `frequencies`
    maps player_id to the fraction of variants selecting that player,
    and `mean_jaccard` is the mean pairwise Jaccard similarity between
    variants' selections at that budget (1.0 when fewer than two
    variants are feasible there). Infeasible variant-budget pairs are
    skipped for the Jaccard mean but still count in the frequency
    denominator: a variant that cannot build a squad selects nobody.
    """
    if not variant_frames:
        raise ValueError("variant_frames must be non-empty")
    quota = quota if quota is not None else POS_QUOTA
    budgets = sorted(set(float(b) for b in budgets))
    n_variants = len(variant_frames)
    frequencies, mean_jaccard = {}, {}
    for budget in budgets:
        counts = {}
        selections = []
        for frame in variant_frames:
            try:
                selection = select_squad(
                    frame, budget, squad_size=squad_size, quota=quota,
                    gamma=gamma, idio_var=idio_var, value_col=value_col,
                )
            except RuntimeError:
                continue
            ids = tuple(sorted(selection["player_id"].astype(str)))
            selections.append(ids)
            for pid in ids:
                counts[pid] = counts.get(pid, 0) + 1
        frequencies[budget] = {pid: c / n_variants for pid, c in sorted(counts.items())}
        if len(selections) < 2:
            mean_jaccard[budget] = 1.0
        else:
            pairs = [
                jaccard(a, b)
                for i, a in enumerate(selections)
                for b in selections[i + 1:]
            ]
            mean_jaccard[budget] = float(np.mean(pairs))
    return SelectionFrequency(frequencies=frequencies, mean_jaccard=mean_jaccard)
