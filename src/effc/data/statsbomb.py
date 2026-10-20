"""Loading cached StatsBomb open data into workable frames."""

from __future__ import annotations

import json
import os

import pandas as pd

from effc.data.download import DATA_DIR, MODEL_POOL, TARGET


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def list_matches(pool: list[tuple[int, int]] | None = None) -> pd.DataFrame:
    """One row per match across the given (competition, season) pairs."""
    pool = pool if pool is not None else MODEL_POOL
    rows = []
    for comp_id, season_id in pool:
        path = os.path.join(DATA_DIR, f"matches_{comp_id}_{season_id}.json")
        if not os.path.exists(path):
            continue
        for m in _load_json(path):
            season = m.get("season") or {}
            rows.append(
                {
                    "match_id": m["match_id"],
                    "competition_id": comp_id,
                    "season_id": season_id,
                    "season_name": season.get("season_name", ""),
                    "match_date": m["match_date"],
                    "home_team": m["home_team"]["home_team_name"],
                    "away_team": m["away_team"]["away_team_name"],
                }
            )
    df = pd.DataFrame(rows).sort_values("match_date").reset_index(drop=True)
    return df


def load_events(match_id: int) -> list[dict]:
    path = os.path.join(DATA_DIR, "events", f"{match_id}.json")
    return _load_json(path)


def minutes_played(match_id: int) -> pd.DataFrame:
    """Minutes per player from the lineup file. Handles subs via position spans."""
    path = os.path.join(DATA_DIR, "lineups", f"{match_id}.json")
    rows = []
    for team in _load_json(path):
        for p in team["lineup"]:
            spans = p.get("positions") or []
            total = 0.0
            pos = None
            for s in spans:
                pos = pos or s.get("position")
                total += _span_minutes(s.get("from"), s.get("to"))
            rows.append(
                {
                    "match_id": match_id,
                    "player_id": p["player_id"],
                    "player": p["player_name"],
                    "team": team["team_name"],
                    "position": pos,
                    "minutes": total,
                }
            )
    return pd.DataFrame(rows)


def _span_minutes(frm: str | None, to: str | None) -> float:
    def parse(t):
        if not t:
            return None
        parts = t.split(":")
        return int(parts[0]) + int(parts[1]) / 60.0

    a, b = parse(frm), parse(to)
    if a is None:
        return 0.0
    if b is None:
        # still on the pitch at the final whistle; ~95 covers stoppage
        b = 95.0
    return max(0.0, b - a)


def download_lineups(match_ids: list[int]) -> None:
    """Fetch lineup files (small). Safe to run while events download."""
    import requests

    base = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
    outdir = os.path.join(DATA_DIR, "lineups")
    os.makedirs(outdir, exist_ok=True)
    for mid in match_ids:
        dest = os.path.join(outdir, f"{mid}.json")
        if os.path.exists(dest):
            continue
        r = requests.get(f"{base}/lineups/{mid}.json", timeout=60)
        r.raise_for_status()
        with open(dest, "w") as f:
            f.write(r.text)
