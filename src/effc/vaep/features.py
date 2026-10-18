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
FEATURE_VERSION = 1

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


def _loc(e):
    loc = e.get("location")
    if loc and len(loc) == 2:
        return float(loc[0]), float(loc[1])
    return np.nan, np.nan


def _end_loc(e):
    t = e["type"]["name"]
    for key in ("pass", "carry", "shot"):
        block = e.get(key)
        if isinstance(block, dict) and block.get("end_location"):
            el = block["end_location"]
            return float(el[0]), float(el[1])
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
    """One row per action with features, labels, and actor metadata."""
    events = sorted(events, key=lambda e: e["index"])
    rows = []
    # running score per team for the score-differential feature
    goals_for: dict[str, int] = {}

    for idx, e in enumerate(events):
        tname = e["type"]["name"]
        if tname not in ACTION_TYPES:
            if is_goal(e):
                team = e["team"]["name"]
                goals_for[team] = goals_for.get(team, 0) + 1
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
    if df.empty:
        return df
    if match_id is not None:
        df["match_id"] = match_id
    df["scores_incl"] = _lookahead(df, "is_goal", "team", K, True, True)
    df["scores_excl"] = _lookahead(df, "is_goal", "team", K, True, False)
    df["concedes_incl"] = _lookahead(df, "is_goal", "team", K, False, True)
    df["concedes_excl"] = _lookahead(df, "is_goal", "team", K, False, False)
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
    X = df[FEATURE_COLS].copy()
    for col in ["action", "prev_action"]:
        d = pd.get_dummies(df[col], prefix=col[:4])
        X = pd.concat([X, d], axis=1)
    return X.astype(np.float32)
