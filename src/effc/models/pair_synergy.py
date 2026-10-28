"""Pair-synergy estimation and selection (Stage 3 U13, Part 1).

The additive VAEP model treats combined performance as a sum of individual
contributions. This module is the disciplined test of whether pair effects
exist on top of it. It is deliberately fail-closed:

- Coverage comes first. A pair is estimable only if it coappears at least
  ``min_coappear`` times in EVERY independent period of the panel. If no
  pair qualifies, the audit says so and nothing is retained.
- Effects are cross-period: player baselines come from one period, pair
  effects are measured in another, so a pair cannot look good merely by
  fitting its own matches. Effects are shrunk toward zero by ``n / (n +
  shrink)`` and optionally capped.
- Stability comes second. A pair is retained only if its estimated effect
  has the same sign in two independent periods and clears a noise floor in
  both. Anything else is reported, not retained.
- Selection linearizes chosen pair indicators ``y_ij`` with ``y <= x_i``,
  ``y <= x_j`` and ``y >= x_i + x_j - 1`` on top of the exact additive MILP
  constraint set, so a pair bonus applies exactly when both players are
  selected, and never otherwise.

Part 1 scope: source contract, baseline capture, fixture and independent
oracle. No performance promotion is made here. In particular, no match-level
player panel is stored in this repo (the five-league results carry squads and
scoreboards, not per-match coappearances), so a real-data coverage audit can
only run where such a panel is supplied by the caller; until then the honest
Part 1 result is the estimator, the oracle, and the rule that retains
nothing without qualifying coverage. The additive selector is never modified:
with no pair effects, the synergy selector reproduces it exactly.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from effc.portfolio.portfolio import InfeasibleError

__all__ = [
    "chronological_periods",
    "coappearance_counts",
    "compare_zero_synergy",
    "coverage_audit",
    "enumerate_optimum",
    "estimate_pair_effects",
    "linearization_rows",
    "select_squad_with_synergy",
    "stability_check",
]

_MAX_ENUMERATION_PLAYERS = 20


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _validate_pairs(n_players: int, pairs, effects: np.ndarray) -> list[tuple[int, int]]:
    pairs = [tuple(p) for p in pairs]
    if len(pairs) != len(effects):
        raise ValueError(
            f"pairs ({len(pairs)}) and effects ({len(effects)}) differ in length"
        )
    if not np.all(np.isfinite(np.asarray(effects, dtype=float))):
        raise ValueError("pair effects must be finite (no NaN or inf)")
    seen: set[tuple[int, int]] = set()
    for i, j in pairs:
        if not (isinstance(i, (int, np.integer)) and isinstance(j, (int, np.integer))):
            raise ValueError(f"pair indices must be integers, got {(i, j)}")
        if i == j:
            raise ValueError(f"self-pair {(i, j)} is not a pair effect")
        if not (0 <= i < n_players and 0 <= j < n_players):
            raise ValueError(f"pair {(i, j)} out of range for {n_players} players")
        key = (min(i, j), max(i, j))
        if key in seen:
            raise ValueError(f"duplicate pair {(i, j)}")
        seen.add(key)
    return pairs


def coappearance_counts(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-pair coappearance counts from a matches x players panel.

    A player appears in a match when the panel value is not NaN (the panel
    is expected to already apply the minutes filter, as
    ``covariance.match_vaep_panel`` does). All pairs are returned, including
    zero-count pairs, so absence is explicit rather than hidden.
    """
    players = list(panel.columns)
    rows = []
    present = panel.notna().to_numpy()
    for a in range(len(players)):
        for b in range(a + 1, len(players)):
            rows.append(
                {
                    "player_a": players[a],
                    "player_b": players[b],
                    "coappearances": int(np.sum(present[:, a] & present[:, b])),
                }
            )
    return pd.DataFrame(rows, columns=["player_a", "player_b", "coappearances"])


def chronological_periods(panel: pd.DataFrame, n_periods: int = 2) -> list[pd.DataFrame]:
    """Split a panel into contiguous periods in supplied (chronological) row order."""
    if n_periods < 1:
        raise ValueError("n_periods must be at least 1")
    n = len(panel)
    bounds = [(i * n) // n_periods for i in range(n_periods + 1)]
    return [panel.iloc[bounds[i]:bounds[i + 1]] for i in range(n_periods)]


def coverage_audit(
    panel: pd.DataFrame, min_coappear: int = 10, n_periods: int = 2
) -> dict:
    """Decide whether any pair is estimable out of sample.

    A pair qualifies only if it coappears at least ``min_coappear`` times in
    EVERY period. The decision is "sufficient" only when at least one pair
    qualifies; otherwise it is "insufficient" and nothing may be retained.
    Empty panels and empty periods fail closed to "insufficient".
    """
    if panel.empty or len(panel) < n_periods:
        return {
            "decision": "insufficient",
            "reason": "panel has no usable matches for every period",
            "qualifying_pairs": [],
            "n_pairs_total": 0,
            "per_period_counts": [],
        }
    periods = chronological_periods(panel, n_periods)
    per_period: list[dict[tuple[str, str], int]] = []
    for block in periods:
        if block.empty:
            return {
                "decision": "insufficient",
                "reason": "a period has no matches",
                "qualifying_pairs": [],
                "n_pairs_total": 0,
                "per_period_counts": per_period,
            }
        counts = coappearance_counts(block)
        per_period.append(
            {
                (r.player_a, r.player_b): int(r.coappearances)
                for r in counts.itertuples()
            }
        )
    all_pairs = set().union(*[set(d) for d in per_period]) if per_period else set()
    qualifying = sorted(
        pair
        for pair in all_pairs
        if all(d.get(pair, 0) >= min_coappear for d in per_period)
    )
    if qualifying:
        return {
            "decision": "sufficient",
            "reason": f"{len(qualifying)} pair(s) meet the threshold in every period",
            "qualifying_pairs": qualifying,
            "n_pairs_total": len(all_pairs),
            "per_period_counts": per_period,
        }
    return {
        "decision": "insufficient",
        "reason": "no pair meets the threshold in every period",
        "qualifying_pairs": [],
        "n_pairs_total": len(all_pairs),
        "per_period_counts": per_period,
    }


def estimate_pair_effects(
    panel_fit: pd.DataFrame,
    panel_eval: pd.DataFrame,
    min_coappear: int = 10,
    shrink: float = 10.0,
    cap: float | None = None,
) -> pd.DataFrame:
    """Cross-period regularized pair effects.

    Baselines are player mean VAEP/90 from ``panel_fit``. For each pair, the
    raw effect is the mean over ``panel_eval`` matches where both appear of
    (joint VAEP/90 minus the sum of the two fit baselines). The reported
    effect shrinks the raw value by ``n / (n + shrink)`` and clips to
    ``[-cap, cap]`` when a cap is given. Pairs below ``min_coappear`` in the
    evaluation panel are omitted, not zero-filled, so a missing row always
    means "not estimated".
    """
    if min_coappear < 1:
        raise ValueError("min_coappear must be at least 1")
    if shrink <= 0:
        raise ValueError("shrink must be positive")
    if cap is not None and cap < 0:
        raise ValueError("cap must be non-negative")
    means = panel_fit.mean(axis=0, skipna=True)
    players = [c for c in panel_eval.columns if c in means.index and not pd.isna(means[c])]
    present = panel_eval[players].notna().to_numpy()
    values = panel_eval[players].to_numpy(dtype=float)
    rows = []
    for a in range(len(players)):
        for b in range(a + 1, len(players)):
            mask = present[:, a] & present[:, b]
            n = int(mask.sum())
            if n < min_coappear:
                continue
            joint = values[mask, a] + values[mask, b]
            raw = float(np.mean(joint - (means[players[a]] + means[players[b]])))
            effect = raw * n / (n + shrink)
            if cap is not None:
                effect = float(np.clip(effect, -cap, cap))
            rows.append(
                {
                    "player_a": players[a],
                    "player_b": players[b],
                    "n_coappear": n,
                    "raw_effect": raw,
                    "effect": effect,
                }
            )
    return pd.DataFrame(
        rows, columns=["player_a", "player_b", "n_coappear", "raw_effect", "effect"]
    )


def stability_check(
    effects_a: pd.DataFrame, effects_b: pd.DataFrame, floor: float = 0.0
) -> dict:
    """Sign agreement and retention rule across two independent periods.

    A pair is stable only if it appears in both periods with the same sign
    and with |effect| at or above ``floor`` in both. Sign agreement counts
    all common pairs regardless of the floor.
    """
    if floor < 0:
        raise ValueError("floor must be non-negative")

    def _as_map(frame: pd.DataFrame) -> dict[tuple[str, str], float]:
        out = {}
        for r in frame.itertuples():
            out[_canonical_pair(str(r.player_a), str(r.player_b))] = float(r.effect)
        return out

    map_a, map_b = _as_map(effects_a), _as_map(effects_b)
    common = sorted(set(map_a) & set(map_b))
    same_sign = sum(
        1 for p in common if np.sign(map_a[p]) == np.sign(map_b[p])
    )
    stable = sorted(
        p
        for p in common
        if np.sign(map_a[p]) == np.sign(map_b[p])
        and abs(map_a[p]) >= floor
        and abs(map_b[p]) >= floor
    )
    return {
        "n_common": len(common),
        "n_same_sign": same_sign,
        "sign_agreement": (same_sign / len(common)) if common else 0.0,
        "stable_pairs": stable,
    }


def linearization_rows(
    n_players: int, pairs
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constraint rows for pair indicators y over variables [x, y].

    For each pair (i, j) with indicator y: ``y - x_i <= 0``,
    ``y - x_j <= 0`` and ``y - x_i - x_j >= -1``. Together these force
    ``y == x_i AND x_j`` for binary x and y, so a pair bonus applies
    exactly when both players are selected.
    """
    pairs = [tuple(p) for p in pairs]
    pairs = _validate_pairs(n_players, pairs, np.zeros(len(pairs)))
    m = len(pairs)
    n_vars = n_players + m
    rows, lb, ub = [], [], []
    for k, (i, j) in enumerate(pairs):
        y = n_players + k
        for row, lo, hi in (
            ({y: 1.0, i: -1.0}, -np.inf, 0.0),
            ({y: 1.0, j: -1.0}, -np.inf, 0.0),
            ({y: 1.0, i: -1.0, j: -1.0}, -1.0, np.inf),
        ):
            vec = np.zeros(n_vars)
            for idx, val in row.items():
                vec[idx] = val
            rows.append(vec)
            lb.append(lo)
            ub.append(hi)
    if not rows:
        return (
            np.zeros((0, n_vars)),
            np.zeros(0),
            np.zeros(0),
        )
    return np.vstack(rows), np.array(lb), np.array(ub)


def select_squad_with_synergy(
    mu: np.ndarray,
    wage: np.ndarray,
    groups: np.ndarray,
    budget: float,
    n_pick: int,
    quota: dict[str, int],
    pairs,
    effects: np.ndarray,
    gamma: float = 0.0,
    idio_var: np.ndarray | None = None,
    max_pairs: int = 200,
) -> np.ndarray:
    """Exact MILP squad selection with additive pair bonuses.

    The constraint set mirrors the additive selector (wage budget, exact
    squad size, per-group minimum quotas) plus the pair-indicator
    linearization. With no pairs this reproduces the additive selector
    exactly. Interaction count is capped by ``max_pairs`` to bound solver
    complexity; anything larger is refused, not silently truncated.
    """
    mu = np.asarray(mu, dtype=float)
    wage = np.asarray(wage, dtype=float)
    groups = np.asarray(groups)
    effects = np.asarray(effects, dtype=float)
    n = len(mu)
    if not (len(wage) == n and len(groups) == n):
        raise ValueError("mu, wage and groups must have the same length")
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(wage))):
        raise ValueError("mu and wage must be finite")
    pairs = _validate_pairs(n, pairs, effects)
    if len(pairs) > max_pairs:
        raise ValueError(
            f"{len(pairs)} pairs exceed the max_pairs cap of {max_pairs}"
        )
    iv = np.zeros(n) if idio_var is None else np.asarray(idio_var, dtype=float)
    if len(iv) != n or not np.all(np.isfinite(iv)):
        raise ValueError("idio_var must be finite and match mu in length")
    m = len(pairs)
    c = np.concatenate([-(mu - gamma * iv), -effects])
    a_rows, lb, ub = [], [], []
    a_rows.append(np.concatenate([wage, np.zeros(m)]))
    lb.append(0.0)
    ub.append(float(budget))
    a_rows.append(np.concatenate([np.ones(n), np.zeros(m)]))
    lb.append(float(n_pick))
    ub.append(float(n_pick))
    for g, q in quota.items():
        a_rows.append(np.concatenate([(groups == g).astype(float), np.zeros(m)]))
        lb.append(float(q))
        ub.append(float(n_pick))
    if m:
        la, llb, lub = linearization_rows(n, pairs)
        a_rows.append(la)
        lb.extend(llb.tolist())
        ub.extend(lub.tolist())
    res = milp(
        c,
        constraints=LinearConstraint(
            np.vstack(a_rows), np.array(lb), np.array(ub)
        ),
        integrality=np.ones(n + m),
        bounds=Bounds(np.zeros(n + m), np.ones(n + m)),
    )
    if not res.success:
        raise InfeasibleError(
            InfeasibleError.MILP_INFEASIBLE,
            f"synergy MILP: {res.message} (budget={budget}, squad_size={n_pick})",
        )
    return (res.x[:n] > 0.5).astype(bool)


def _synergy_objective(
    mask: np.ndarray,
    mu: np.ndarray,
    gamma: float,
    idio_var: np.ndarray,
    pairs,
    effects: np.ndarray,
) -> float:
    value = float(np.sum(mu[mask] - gamma * idio_var[mask]))
    selected = set(np.flatnonzero(mask).tolist())
    for (i, j), s in zip(pairs, effects):
        if i in selected and j in selected:
            value += float(s)
    return value


def compare_zero_synergy(
    mu: np.ndarray,
    wage: np.ndarray,
    groups: np.ndarray,
    budget: float,
    n_pick: int,
    quota: dict[str, int],
    pairs,
    effects: np.ndarray,
    gamma: float = 0.0,
    idio_var: np.ndarray | None = None,
) -> dict:
    """Additive vs synergy selection on identical inputs, scored alike.

    Both selections are scored under the synergy objective, so the synergy
    selection must score at least as well by construction; any gap is the
    value the pair bonuses add, not a claim about realized performance.
    """
    from effc.portfolio.portfolio import _milp_select

    mu = np.asarray(mu, dtype=float)
    n = len(mu)
    iv = np.zeros(n) if idio_var is None else np.asarray(idio_var, dtype=float)
    additive = _milp_select(mu, wage, groups, budget, n_pick, quota, gamma, iv)
    synergy = select_squad_with_synergy(
        mu, wage, groups, budget, n_pick, quota, pairs, effects,
        gamma=gamma, idio_var=iv,
    )
    pairs = _validate_pairs(n, pairs, np.asarray(effects, dtype=float))
    return {
        "additive_selection": additive,
        "synergy_selection": synergy,
        "additive_value": _synergy_objective(
            additive, mu, gamma, iv, pairs, effects
        ),
        "synergy_value": _synergy_objective(
            synergy, mu, gamma, iv, pairs, effects
        ),
    }


def enumerate_optimum(
    mu: np.ndarray,
    wage: np.ndarray,
    groups: np.ndarray,
    budget: float,
    n_pick: int,
    quota: dict[str, int],
    pairs,
    effects: np.ndarray,
    gamma: float = 0.0,
    idio_var: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Brute-force optimum for tiny instances: the independent oracle.

    Enumerates every feasible squad (exact size, wage budget, per-group
    minimum quotas) and scores it under the synergy objective. Refuses
    instances larger than 20 players rather than enumerating silently for a
    long time. The MILP selector must match this value and mask exactly.
    """
    mu = np.asarray(mu, dtype=float)
    n = len(mu)
    if n > _MAX_ENUMERATION_PLAYERS:
        raise ValueError(
            f"enumeration refused for {n} players (cap {_MAX_ENUMERATION_PLAYERS})"
        )
    wage = np.asarray(wage, dtype=float)
    groups = np.asarray(groups)
    iv = np.zeros(n) if idio_var is None else np.asarray(idio_var, dtype=float)
    pairs = _validate_pairs(n, pairs, np.asarray(effects, dtype=float))
    best_value: float | None = None
    best_mask: np.ndarray | None = None
    for combo in itertools.combinations(range(n), n_pick):
        mask = np.zeros(n, dtype=bool)
        mask[list(combo)] = True
        if float(np.sum(wage[mask])) > budget:
            continue
        if any(int(np.sum(groups[mask] == g)) < q for g, q in quota.items()):
            continue
        value = _synergy_objective(mask, mu, gamma, iv, pairs, effects)
        if best_value is None or value > best_value + 1e-9:
            best_value = value
            best_mask = mask
    if best_value is None or best_mask is None:
        raise InfeasibleError(
            InfeasibleError.MILP_INFEASIBLE,
            "enumeration: no feasible squad",
        )
    return best_value, best_mask
