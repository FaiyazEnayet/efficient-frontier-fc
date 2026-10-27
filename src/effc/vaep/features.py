"""Action featurization for VAEP, in the spirit of Decroos et al. (2019).

Each on-ball action becomes a feature vector describing the game state.
Two classifiers then learn P(team scores in next k actions) and
P(team concedes in next k actions) from these states. The change in those
probabilities across an action is the action's value.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Bump whenever the feature set or schema changes: it feeds the VAEP
# cache key so stale values are never silently reused.
# v2: optional Dixon-Coles team-strength columns (team_strength.py).
# v3: StatsBomb raw coordinates (120x80) are converted to the SPADL
# frame (105x68) before any feature is computed; v2 frames used the raw
# values directly, so v2 caches must never be reused.
# v4: own goals (the "Own Goal For" / "Own Goal Against" pair) count in
# the scores/concedes labels and in score_diff, and the goal-conceded
# debit in values.py extends to them; v3 caches must never be reused.
FEATURE_VERSION = 4

# event types we value as actions
ACTION_TYPES = [
    "Pass",
    "Carry",
    "Dribble",
    "Shot",
    "Ball Recovery",
    "Interception",
    "Clearance",
    "Duel",
    "Block",
    "Goal Keeper",
    "Miscontrol",
    "Dispossessed",
]

K = 10  # lookahead window in actions

PITCH_X, PITCH_Y = 105.0, 68.0
GOAL_X, GOAL_Y = 105.0, 34.0

# Coordinate contract for this module (verified against real StatsBomb
# open data). Each event's coordinates are recorded in the ACTING
# TEAM's attacking frame: the acting team attacks toward x = 120 raw
# (x = 105 SPADL). Evidence: shots by both teams in both periods start
# in the attacking third; an opponent action after a pass sits at the
# mirror (120-x, 80-y) of the pass end (distance to the mirror about
# 0.1 m, versus 26-54 m to the shared-frame point); same-team pass
# chains are continuous (distance about 0.0 m); goalkeepers sit near
# their own goal (raw x about 2.5) in their own acting frame. Output
# features (start_x, start_y, end_x, end_y, dx, dy, distances, angles)
# are in the 105x68 SPADL frame, in metres, with the attacked goal at
# (105, 34) for every event. Conversion is scale-only (the socceraction
# StatsBomb-to-SPADL scaling) because each event is already in its
# acting team's attacking frame, so no flip by team or period is
# applied. Consequence: start_y/end_y are relative to attacking
# direction, not absolute pitch side. No feature depends on cross-action
# spatial continuity: all spatial features are within a single event
# (start to end) or relative to the attacked goal, so the per-event
# frame is exactly what those features need. The earlier comment
# describing a shared possession-normalized frame was imprecise and
# has been corrected to this verified convention.
STATSBOMB_PITCH_X, STATSBOMB_PITCH_Y = 120.0, 80.0


def _to_spadl(x, y):
    """Convert one raw StatsBomb point to the SPADL frame."""
    if np.isnan(x) or np.isnan(y):
        return np.nan, np.nan
    return x * PITCH_X / STATSBOMB_PITCH_X, y * PITCH_Y / STATSBOMB_PITCH_Y


def _loc(e):
    loc = e.get("location")
    if loc and len(loc) == 2:
        return _to_spadl(float(loc[0]), float(loc[1]))
    return np.nan, np.nan


def _end_loc(e):
    t = e["type"]["name"]
    for key in ("pass", "carry", "shot"):
        block = e.get(key)
        if isinstance(block, dict) and block.get("end_location"):
            el = block["end_location"]
            return _to_spadl(float(el[0]), float(el[1]))
    if t == "Dribble" and e.get("dribble", {}).get("overrun"):
        return _loc(e)
    return _loc(e)


def _dist_angle(x, y):
    if np.isnan(x) or np.isnan(y):
        return np.nan, np.nan
    dx = GOAL_X - x
    dy = GOAL_Y - y
    dist = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(abs(dy), dx + 1e-9))
    return dist, angle


def is_goal(e) -> bool:
    return (
        e["type"]["name"] == "Shot"
        and isinstance(e.get("shot"), dict)
        and e["shot"].get("outcome", {}).get("name") == "Goal"
    )


def featurize_match(events: list[dict], match_id: int | None = None) -> pd.DataFrame:
    """One row per action with features, labels, and actor metadata.

    Input `events` are raw StatsBomb events (120x80 frame). All location
    features in the returned frame are converted to the 105x68 SPADL
    frame by `_loc`/`_end_loc` before use; see the coordinate contract
    above the constants in this module.

    Own goals (the StatsBomb "Own Goal For" / "Own Goal Against" pair)
    count in the scores/concedes labels and in score_diff, but add no
    action rows. Each own goal's event position, benefiting team, and
    period are recorded in df.attrs["own_goals"] so the goal-conceded
    debit in values.py can find them.
    """
    events = sorted(events, key=lambda e: e["index"])
    rows = []
    # running score per team for the score-differential feature
    goals_for: dict[str, int] = {}
    # (event position, benefiting team, period) for each own goal seen
    own_goals: list[tuple[int, str, int]] = []

    for idx, e in enumerate(events):
        tname = e["type"]["name"]
        if tname not in ACTION_TYPES:
            # Own goals arrive as a consecutive pair, "Own Goal For"
            # for the benefiting team and "Own Goal Against" for the
            # conceding team, in either order. Count only the For
            # event, once, so the pair is never double-counted.
            # Record the pair's event position (idx in the sorted
            # event list, the same space as match_index) for the label
            # step below and the debit in values.py. Own goals are
            # deliberately NOT action rows: adding rows would change
            # row counts, the design matrix shape, and player VAEP
            # aggregation.
            if tname == "Own Goal For":
                team = e["team"]["name"]
                goals_for[team] = goals_for.get(team, 0) + 1
                own_goals.append((idx, team, e.get("period", 1)))
            continue
        # actor's own team (not the possession team: that changes mid-sequence
        # and would credit a player's actions to the opponent)
        team = e["team"]["name"]
        opp = None
        sx, sy = _loc(e)
        ex, ey = _end_loc(e)
        ds, angs = _dist_angle(sx, sy)
        de, ange = _dist_angle(ex, ey)
        minute = float(e.get("minute", 0)) + float(e.get("second", 0)) / 60.0
        # score differential from this team's perspective
        scored = goals_for.get(team, 0)
        conceded = sum(v for tm, v in goals_for.items() if tm != team)
        prev = rows[-1] if rows else None
        rows.append(
            {
                "match_index": idx,
                "period": e.get("period", 1),
                "team": team,
                "player_id": (e.get("player") or {}).get("id"),
                "player": (e.get("player") or {}).get("name"),
                "action": tname,
                "start_x": sx,
                "start_y": sy,
                "end_x": ex,
                "end_y": ey,
                "dx": ex - sx,
                "dy": ey - sy,
                "dist_start": ds,
                "angle_start": angs,
                "dist_end": de,
                "angle_end": ange,
                "under_pressure": 1.0 if e.get("under_pressure") else 0.0,
                "minute": minute / 95.0,
                "score_diff": scored - conceded,
                "prev_action": prev["action"] if prev else "None",
                "prev_same_team": 1.0 if (prev and prev["team"] == team) else 0.0,
                "is_goal": is_goal(e),
            }
        )
        if is_goal(e):
            goals_for[team] = goals_for.get(team, 0) + 1

    df = pd.DataFrame(rows)
    df.attrs["own_goals"] = own_goals
    if df.empty:
        return df
    if match_id is not None:
        df["match_id"] = match_id
    df["scores_incl"] = _lookahead(df, "is_goal", "team", K, True, True)
    df["scores_excl"] = _lookahead(df, "is_goal", "team", K, True, False)
    df["concedes_incl"] = _lookahead(df, "is_goal", "team", K, False, True)
    df["concedes_excl"] = _lookahead(df, "is_goal", "team", K, False, False)
    # Own goals are not action rows, so _lookahead never sees them. OR
    # them into the labels directly: for each own goal, the k rows
    # immediately before it (same period only) get scores for the
    # benefiting team or concedes for every other team, identically in
    # the incl and excl windows, because an own goal is never any
    # action's own goal. p is the insertion point: the number of rows
    # with match_index below the own-goal event position.
    if own_goals:
        match_idx = df["match_index"].to_numpy()
        teams = df["team"].to_numpy()
        periods = df["period"].to_numpy()
        s_in = df["scores_incl"].to_numpy().copy()
        s_ex = df["scores_excl"].to_numpy().copy()
        c_in = df["concedes_incl"].to_numpy().copy()
        c_ex = df["concedes_excl"].to_numpy().copy()
        for own_idx, benefit, own_period in own_goals:
            p = int((match_idx < own_idx).sum())
            for i in range(max(0, p - K), p):
                if periods[i] != own_period:
                    continue
                if teams[i] == benefit:
                    s_in[i] = s_ex[i] = 1.0
                else:
                    c_in[i] = c_ex[i] = 1.0
        df["scores_incl"] = s_in
        df["scores_excl"] = s_ex
        df["concedes_incl"] = c_in
        df["concedes_excl"] = c_ex
    return df


def _lookahead(df, goal_col, team_col, k, same_team, include_current):
    """1 if a goal by (not) the actor's team occurs in the window.

    include_current=True: window is i..i+k-1 (the action's own goal counts).
    include_current=False: window is i+1..i+k.
    Never crosses a period boundary.
    """
    goals = df[goal_col].to_numpy()
    teams = df[team_col].to_numpy()
    periods = df["period"].to_numpy()
    n = len(df)
    out = np.zeros(n, dtype=np.float32)
    next_goal_idx: dict[str, int] = {}
    for i in range(n - 1, -1, -1):
        if include_current and goals[i]:
            next_goal_idx[teams[i]] = i
        p = periods[i]
        hit = False
        lo = 0 if include_current else 1
        for tm, gi in next_goal_idx.items():
            if gi - i < lo or gi - i > k - 1 + lo or periods[gi] != p:
                continue
            if (tm == teams[i]) == same_team:
                hit = True
                break
        out[i] = 1.0 if hit else 0.0
        if not include_current and goals[i]:
            next_goal_idx[teams[i]] = i
    return out


FEATURE_COLS = [
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "dx",
    "dy",
    "dist_start",
    "angle_start",
    "dist_end",
    "angle_end",
    "under_pressure",
    "minute",
    "score_diff",
    "prev_same_team",
]


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    from effc.vaep.team_strength import STRENGTH_COLS

    X = df[FEATURE_COLS].copy()
    # Team-strength features (v2): attached to frames by the training /
    # scoring code from a frozen Dixon-Coles fit. Frames that never had
    # strengths attached (legacy caches, synthetic test frames) get the
    # neutral league-average value 0.0.
    for col in STRENGTH_COLS:
        X[col] = df[col].astype(float) if col in df.columns else 0.0
    for col in ["action", "prev_action"]:
        d = pd.get_dummies(df[col], prefix=col[:4])
        X = pd.concat([X, d], axis=1)
    return X.astype(np.float32)
