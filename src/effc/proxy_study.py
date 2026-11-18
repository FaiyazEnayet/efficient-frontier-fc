"""Separate study: how well does the event-summary proxy track fitted VAEP?

The live refresh folds an event-summary proxy (goals, assists and
cards, in per-90 units on the VAEP scale for readability) into the
base ratings. That proxy is not VAEP: VAEP values every on-ball action
through fitted goal models, while the proxy sees only the handful of
events a free feed publishes. This module measures the relationship on
real overlapping data instead of asserting it: the proxy is computed
from StatsBomb 2015/16 match events and lineups for one league, the
players are matched to that league's committed baseline ratings with
the same matcher the refresh uses, and the comparison reports the
correlation, the calibration slope and intercept of VAEP on the proxy,
and the error of the best linear rescaling.

The verdict is whatever the numbers say. A slope far from 1, an
intercept far from 0, or a weak correlation means the proxy is not a
calibrated VAEP estimate, and the blend stays a refresh mechanism,
not a measured improvement. Nothing here tunes the proxy weights:
they are the weights the refresh already uses, read from effc.live.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from effc.live import (
    EVENT_PROXY_W_ASSIST,
    EVENT_PROXY_W_GOAL,
    EVENT_PROXY_W_PENALTY_GOAL,
    EVENT_PROXY_W_RED,
    EVENT_PROXY_W_YELLOW,
    canonical_id,
    match_player,
    normalize_name,
)

# A lineup span with no end means the player was still on the pitch
# at the final whistle; same convention as
# effc.data.statsbomb.minutes_played.
FINAL_WHISTLE_MIN = 95.0


def _span_minutes(frm: str | None, to: str | None) -> float:
    """Minutes covered by one lineup position span ("MM:SS" strings)."""
    def parse(t):
        if not t:
            return None
        parts = t.split(":")
        return int(parts[0]) + int(parts[1]) / 60.0

    a, b = parse(frm), parse(to)
    if a is None:
        return 0.0
    if b is None:
        b = FINAL_WHISTLE_MIN
    return max(0.0, b - a)


def minutes_from_lineups(lineups: list[dict]) -> pd.DataFrame:
    """Minutes per (team, player) from StatsBomb lineup JSON.

    Sums every position span, so substitutions, injury returns and
    tactical shifts are handled the same way the main loader handles
    them; a player with no spans (unused substitute) gets 0 minutes.
    """
    rows = []
    for team in lineups or []:
        for p in team.get("lineup", []) or []:
            total = 0.0
            for s in p.get("positions") or []:
                total += _span_minutes(s.get("from"), s.get("to"))
            rows.append({
                "team": team.get("team_name", ""),
                "player": p.get("player_name", ""),
                "minutes": total,
            })
    return pd.DataFrame(rows, columns=["team", "player", "minutes"])


def statsbomb_match_proxy(events: list[dict], lineups: list[dict]) -> pd.DataFrame:
    """Per-player event-summary proxy for one StatsBomb match.

    Uses the same proxy weights as the live refresh. Goals count 0.9
    (0.7 for penalties), assists 0.4, yellow cards -0.05, red cards
    (including second yellows) -0.30. Own goals are skipped: they are
    not production by the player they are charged to, and the live
    refresh skips them the same way. A saved shot, a pass that is not
    a goal assist, and any other event type contribute nothing.
    Returns one row per player named in the lineups, plus any player
    with an event but no lineup row (0 minutes), with columns team,
    player, minutes, goals, penalty_goals, assists, yellow, red,
    contrib and proxy_per90.
    """
    minutes = minutes_from_lineups(lineups)
    by_key: dict[tuple[str, str], dict] = {}

    def ensure(team, name):
        key = (team, name)
        if key not in by_key:
            by_key[key] = {"team": team, "player": name, "goals": 0,
                           "penalty_goals": 0, "assists": 0, "yellow": 0,
                           "red": 0, "contrib": 0.0}
        return by_key[key]

    for ev in events or []:
        etype = (ev.get("type") or {}).get("name", "")
        team = (ev.get("team") or {}).get("name", "")
        name = (ev.get("player") or {}).get("name", "")
        if not name:
            continue
        if etype == "Shot":
            outcome = ((ev.get("shot") or {}).get("outcome") or {}).get("name", "")
            if outcome == "Goal":
                kind = ((ev.get("shot") or {}).get("type") or {}).get("name", "")
                row = ensure(team, name)
                if kind == "Penalty":
                    row["penalty_goals"] += 1
                    row["contrib"] += EVENT_PROXY_W_PENALTY_GOAL
                else:
                    row["goals"] += 1
                    row["contrib"] += EVENT_PROXY_W_GOAL
        elif etype == "Pass":
            if (ev.get("pass") or {}).get("goal_assist"):
                row = ensure(team, name)
                row["assists"] += 1
                row["contrib"] += EVENT_PROXY_W_ASSIST
        elif etype == "Foul Committed":
            card = ((ev.get("foul_committed") or {}).get("card") or {}).get(
                "name", "")
            if card == "Yellow Card":
                row = ensure(team, name)
                row["yellow"] += 1
                row["contrib"] += EVENT_PROXY_W_YELLOW
            elif card in ("Red Card", "Second Yellow"):
                row = ensure(team, name)
                row["red"] += 1
                row["contrib"] += EVENT_PROXY_W_RED
        # Own Goal For / Own Goal Against and all other types: no contribution

    rows = []
    seen: set[tuple[str, str]] = set()
    for r in minutes.itertuples():
        key = (r.team, r.player)
        seen.add(key)
        row = dict(by_key.get(key, {"team": r.team, "player": r.player,
                                    "goals": 0, "penalty_goals": 0,
                                    "assists": 0, "yellow": 0, "red": 0,
                                    "contrib": 0.0}))
        row["minutes"] = float(r.minutes)
        rows.append(row)
    for key, row in by_key.items():
        if key not in seen:
            row = dict(row)
            row["minutes"] = 0.0
            rows.append(row)
    df = pd.DataFrame(rows, columns=["team", "player", "minutes", "goals",
                                     "penalty_goals", "assists", "yellow",
                                     "red", "contrib"])
    df["proxy_per90"] = (df["contrib"] / df["minutes"].clip(lower=1e-9)
                         * 90.0).where(df["minutes"] > 0, 0.0)
    return df


def accumulate_proxy(match_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Sum per-match proxy rows into per-player season totals.

    One row per (team, player) with summed minutes, contributions and
    event counts, the number of matches the player appeared in, and
    the season proxy per 90.
    """
    frames = [f for f in match_frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame(columns=["team", "player", "minutes", "contrib",
                                     "goals", "penalty_goals", "assists",
                                     "yellow", "red", "matches", "proxy_per90"])
    all_rows = pd.concat(frames, ignore_index=True)
    # group by normalized name: one player is one row even when the
    # source spells the name two ways across matches (a doubled
    # apostrophe in the raw data splits N'Golo Kante in two otherwise)
    all_rows["player_key"] = all_rows["player"].map(normalize_name)
    grouped = all_rows.groupby(["team", "player_key"], as_index=False).agg(
        minutes=("minutes", "sum"), contrib=("contrib", "sum"),
        goals=("goals", "sum"), penalty_goals=("penalty_goals", "sum"),
        assists=("assists", "sum"), yellow=("yellow", "sum"),
        red=("red", "sum"), matches=("minutes", "size"))
    display = all_rows.loc[
        all_rows.groupby(["team", "player_key"])["minutes"].idxmax(),
        ["team", "player_key", "player"]].drop_duplicates()
    grouped = grouped.merge(display, on=["team", "player_key"], how="left")
    grouped = grouped.drop(columns=["player_key"])
    grouped["proxy_per90"] = (
        grouped["contrib"] / grouped["minutes"].clip(lower=1e-9) * 90.0
    ).where(grouped["minutes"] > 0, 0.0)
    return grouped


def compare_proxy_to_baseline(proxy: pd.DataFrame, ratings: pd.DataFrame,
                              min_minutes: float = 90.0) -> tuple[dict, pd.DataFrame]:
    """Match proxy players to baseline ratings and compare the two.

    Players with fewer than `min_minutes` proxy minutes are excluded:
    a per-90 rate from a few minutes is noise, not a measurement.
    Matching uses the same team-constrained matcher the refresh uses,
    so the match rate here is the match rate the refresh would see.
    Returns (stats, matched): stats carries n, the counts, Pearson
    and Spearman correlation, the OLS slope and intercept of VAEP on
    the proxy, the mean absolute error of that linear rescaling, and
    the two means; a correlation or slope that is undefined (no
    variance, too few players) is reported as None, not invented.
    Matched carries one row per matched player.
    """
    base = ratings.copy()
    base["player_id"] = base["player_id"].map(canonical_id)
    by_id = base.set_index("player_id")
    aliases: dict = {}
    rows = []
    n_proxy = len(proxy)
    n_enough_minutes = 0
    for r in proxy.itertuples():
        if float(r.minutes) < min_minutes:
            continue
        n_enough_minutes += 1
        pid = match_player(r.player, r.team, base, aliases)
        if pid is None or pid not in by_id.index:
            continue
        brow = by_id.loc[pid]
        rows.append({
            "player_id": pid,
            "player": r.player,
            "team": r.team,
            "minutes": float(r.minutes),
            "proxy_per90": float(r.proxy_per90),
            "vaep_per90": float(brow["vaep_per90"]),
        })
    matched = pd.DataFrame(rows, columns=["player_id", "player", "team",
                                          "minutes", "proxy_per90",
                                          "vaep_per90"])
    stats: dict = {
        "n": len(matched),
        "n_proxy_players": int(n_proxy),
        "n_proxy_players_min_minutes": int(n_enough_minutes),
        "n_baseline_rated": int(len(base)),
        "min_minutes": float(min_minutes),
        "pearson": None,
        "spearman": None,
        "slope": None,
        "intercept": None,
        "mae": None,
        "mean_proxy_per90": None,
        "mean_vaep_per90": None,
    }
    if len(matched) >= 2:
        x = matched["proxy_per90"].to_numpy(dtype=float)
        y = matched["vaep_per90"].to_numpy(dtype=float)
        stats["mean_proxy_per90"] = float(matched["proxy_per90"].mean())
        stats["mean_vaep_per90"] = float(matched["vaep_per90"].mean())
        if np.std(x) > 0 and np.std(y) > 0:
            pearson = matched["proxy_per90"].corr(matched["vaep_per90"])
            spearman = matched["proxy_per90"].corr(matched["vaep_per90"],
                                                   method="spearman")
            stats["pearson"] = None if pd.isna(pearson) else float(pearson)
            stats["spearman"] = None if pd.isna(spearman) else float(spearman)
        if np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            stats["slope"] = float(slope)
            stats["intercept"] = float(intercept)
            stats["mae"] = float(np.mean(np.abs(y - (slope * x + intercept))))
    return stats, matched


def load_statsbomb_json(data_dir: str, kind: str, match_id: int) -> list[dict]:
    """Load one StatsBomb events or lineups file from a data dir."""
    path = os.path.join(data_dir, kind, f"{int(match_id)}.json")
    with open(path) as f:
        return json.load(f)
