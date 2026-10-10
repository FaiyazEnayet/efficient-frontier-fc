"""Live data layer: refresh ratings from API-Football and rebalance the squad.

What this is: a daily refresh that pulls finished fixtures from the
API-Football free tier, converts match events into per-player value
proxies, folds them into the existing ratings with the same shrinkage
philosophy as the main pipeline, and re-runs the squad optimizer. The
output is a rebalance report: who comes in, who goes out.

What this is not: live VAEP. True VAEP needs every on-ball action and no
free feed publishes that. The event proxy below is deliberately crude and
calibrated to the VAEP scale so it blends honestly. Read the weights as a
starting point, not science.

Quota: the free tier allows 100 requests/day. One refresh costs
days + 2*N calls for N new fixtures. A daily run with 9 new fixtures
costs about 20 calls.

Free-tier constraints this module is built around:
- seasons 2022-2024 only (no current season),
- no `last` parameter: date queries are the only window, limited to the
  last 2 days,
- data refreshes within ~15 minutes of real events: near-live, not live.
"""

from __future__ import annotations

import json
import os
import subprocess
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

SKILL_CLI = os.environ.get(
    "API_FOOTBALL_CLI",
    os.path.expanduser("~/workspace/skills/api-football/bin/api-football"),
)
STATE_DIR = os.path.join("data", "live")
STATE_PATH = os.path.join(STATE_DIR, "state.json")

# Proxy weights in VAEP-ish units. Scoring moves P(goal) from ~0.1 to 1.0,
# so a goal is worth about 0.9 on the VAEP scale. Assists get partial
# credit. Cards are a judgment call: small, negative, documented.
W_GOAL = 0.9
W_PENALTY_GOAL = 0.7
W_ASSIST = 0.4
W_YELLOW = -0.05
W_RED = -0.30

# Live minutes needed for the live sample to carry half the weight.
LIVE_HALF_LIFE_MIN = 450.0


class ApiError(RuntimeError):
    pass


def _direct_get(endpoint: str, params: dict) -> dict:
    """Plain HTTPS call using the API_FOOTBALL_KEY env var.

    Fallback for anyone running this outside Muse, where the skill CLI
    and its stored credential are not available.
    """
    import urllib.parse
    import urllib.request

    key = os.environ.get("API_FOOTBALL_KEY", "")
    if not key:
        raise ApiError(
            "no API-Football credential: set API_FOOTBALL_KEY or install the skill CLI")
    url = ("https://v3.football.api-sports.io/" + endpoint.lstrip("/")
           + "?" + urllib.parse.urlencode(params))
    req = urllib.request.Request(url, headers={
        "Accept": "application/json", "x-apisports-key": key})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ApiError(f"API request failed: {exc}")


def api_get(endpoint: str, **params) -> list | dict:
    """Call API-Football and return the response payload.

    Prefers the Muse skill CLI (stored credential, never in env); falls
    back to a direct call with API_FOOTBALL_KEY.
    """
    data = None
    if os.path.exists(SKILL_CLI):
        cmd = [SKILL_CLI, endpoint] + [f"{k}={v}" for k, v in params.items()]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            raise ApiError(f"API request timed out: {endpoint}")
        if proc.returncode != 0:
            raise ApiError(
                f"API request failed: {proc.stdout.strip() or proc.stderr.strip()}")
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise ApiError(f"API returned non-JSON for {endpoint}")
    else:
        data = _direct_get(endpoint, params)
    if isinstance(data, dict) and data.get("error"):
        raise ApiError(f"API error: {data['error']}")
    if isinstance(data, dict) and data.get("errors"):
        raise ApiError(f"API errors: {data['errors']}")
    if isinstance(data, dict) and "response" in data:
        return data["response"]
    return data


def fetch_finished_fixtures(league: int, season: int, days: int = 2) -> list[dict]:
    """Finished fixtures from the last `days` days.

    Date queries are the only reliable window on the free tier: the `last`
    parameter is blocked and date history is limited to about 2 days.
    Costs one API call per day scanned.
    """
    from datetime import timedelta

    today = datetime.now(timezone.utc).date()
    seen: set[int] = set()
    out: list[dict] = []
    for i in range(days):
        day = (today - timedelta(days=i)).isoformat()
        resp = api_get("fixtures", league=league, season=season, date=day)
        for f in resp:
            fid = f.get("fixture", {}).get("id")
            if fid in seen:
                continue
            seen.add(fid)
            if f.get("fixture", {}).get("status", {}).get("short") == "FT":
                out.append(f)
    return out


def fetch_fixture_events(fixture_id: int) -> list[dict]:
    return api_get("fixtures/events", fixture=fixture_id)


def fetch_fixture_lineups(fixture_id: int) -> list[dict]:
    return api_get("fixtures/lineups", fixture=fixture_id)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return "".join(c for c in s.lower() if c.isalnum() or c == " ").strip()


def _team_overlap(a: str, b: str) -> float:
    ta, tb = set(_norm(a).split()), set(_norm(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def match_player(api_name: str, api_team: str, ratings: pd.DataFrame,
                 aliases: dict) -> str | None:
    """Map an API-Football player to a ratings player_id.

    Uses a persisted alias map first, then team-constrained fuzzy matching.
    Returns None when no confident match exists.
    """
    key = f"{_norm(api_team)}||{_norm(api_name)}"
    if key in aliases:
        return aliases[key]
    cands = ratings[ratings["team"].apply(lambda t: _team_overlap(t, api_team) >= 0.3)]
    best, best_score = None, 0.0
    target = _norm(api_name)
    for _, row in cands.iterrows():
        score = SequenceMatcher(None, target, _norm(row["player"])).ratio()
        if score > best_score:
            best, best_score = row["player_id"], score
    if best is not None and best_score >= 0.8:
        aliases[key] = best
        return best
    return None


def fixture_contributions(events: list[dict], lineups: list[dict]) -> pd.DataFrame:
    """Per-player proxy contributions for one finished fixture.

    Returns columns: api_team, api_player, minutes, contrib.
    """
    minutes: dict[tuple[str, str], float] = {}
    for side in lineups:
        team = side.get("team", {}).get("name", "")
        for p in side.get("startXI", []) + side.get("substitutes", []):
            name = p.get("player", {}).get("name", "")
            minutes[(team, name)] = 0.0
        for p in side.get("startXI", []):
            minutes[(team, p.get("player", {}).get("name", ""))] = 90.0

    contrib: dict[tuple[str, str], float] = {}

    def add(team, name, x):
        if name:
            contrib[(team, name)] = contrib.get((team, name), 0.0) + x

    for ev in events:
        team = ev.get("team", {}).get("name", "")
        etype, detail = ev.get("type", ""), ev.get("detail", "")
        name = (ev.get("player", {}) or {}).get("name", "")
        assist = (ev.get("assist", {}) or {}).get("name", "")
        elapsed = (ev.get("time", {}) or {}).get("elapsed") or 90
        if etype == "Goal":
            if detail == "Own Goal":
                continue
            add(team, name, W_PENALTY_GOAL if detail == "Penalty" else W_GOAL)
            add(team, assist, W_ASSIST)
        elif etype == "Card":
            if detail == "Yellow Card":
                add(team, name, W_YELLOW)
            elif detail in ("Red Card", "Second Yellow card"):
                add(team, name, W_RED)
        elif etype == "subst":
            if (team, name) in minutes:
                minutes[(team, name)] = min(minutes[(team, name)], float(elapsed))
            if assist and (team, assist) in minutes:
                minutes[(team, assist)] = max(0.0, 90.0 - float(elapsed))

    rows = [
        {"api_team": t, "api_player": n, "minutes": m,
         "contrib": contrib.get((t, n), 0.0)}
        for (t, n), m in minutes.items() if m > 0
    ]
    return pd.DataFrame(rows)


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"last_fixture_id": 0, "aliases": {}, "live_minutes": {},
            "live_contrib": {}, "watchlist": {}, "calls": {}, "history": []}


def save_state(state: dict) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=1)


def blend_ratings(ratings: pd.DataFrame, state: dict,
                  shrink_k: float = 900.0) -> pd.DataFrame:
    """Fold accumulated live proxies into the ratings.

    Same shrinkage idea as the main pipeline: total value over total
    minutes, shrunk toward the position-group mean. Live minutes enter
    with a half-life so a hot week moves the needle but does not
    rewrite a season.
    """
    out = ratings.copy()
    live_min = out["player_id"].map(state.get("live_minutes", {})).fillna(0.0)
    live_val = out["player_id"].map(state.get("live_contrib", {})).fillna(0.0)
    out["live_minutes"] = live_min

    # weight the live sample by a half-life curve: 450 live minutes carry
    # half the weight, so a hot week moves the needle but never rewrites
    # a season on its own
    w_live = (live_min / (live_min + LIVE_HALF_LIFE_MIN)).fillna(0.0)
    live_per90 = (live_val / live_min.clip(lower=1e-9) * 90.0).fillna(0.0)
    out["live_per90"] = live_per90
    out["vaep_per90_live"] = (1 - w_live) * out["vaep_per90"] + w_live * live_per90

    # re-shrink toward group means so small live samples do not explode
    group_mean = out.groupby("pos_group")["vaep_per90_live"].transform("mean")
    tot_min = out["minutes"] + live_min
    w = tot_min / (tot_min + shrink_k)
    out["vaep_per90_live"] = w * out["vaep_per90_live"] + (1 - w) * group_mean
    return out


def rebalance_report(old: pd.DataFrame, new: pd.DataFrame,
                     ratings: pd.DataFrame) -> pd.DataFrame:
    """Who leaves and who enters when the squad is re-optimized."""
    old_ids, new_ids = set(old["player_id"]), set(new["player_id"])
    r = ratings.set_index("player_id")
    rows = []
    for pid in sorted(old_ids - new_ids):
        rows.append({"action": "OUT", "player": r.loc[pid, "player"],
                     "team": r.loc[pid, "team"],
                     "vaep_per90": r.loc[pid, "vaep_per90_live"],
                     "wage_eur": r.loc[pid, "wage_eur"]})
    for pid in sorted(new_ids - old_ids):
        rows.append({"action": "IN", "player": r.loc[pid, "player"],
                     "team": r.loc[pid, "team"],
                     "vaep_per90": r.loc[pid, "vaep_per90_live"],
                     "wage_eur": r.loc[pid, "wage_eur"]})
    return pd.DataFrame(rows)


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
