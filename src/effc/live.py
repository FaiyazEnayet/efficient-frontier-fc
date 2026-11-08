"""Live data layer: refresh ratings from API-Football and rebalance the squad.

What this is: a daily refresh that pulls finished fixtures from the
API-Football free tier, converts match events into a per-player
event-summary proxy, folds that proxy into the existing ratings, and
re-runs the squad optimizer. The output is a rebalance report: who
comes in, who goes out.

What this is not: live VAEP. True VAEP needs every on-ball action and
no free feed publishes that. The event-summary proxy below is a
separately named summary of goals, assists and cards; it is not a
calibrated VAEP estimate and is not presented as one. It shares the
VAEP per-90 scale only so the blend arithmetic stays readable, and it
stays a proxy until it is calibrated on real overlapping data.

Quota: the free tier allows 100 requests/day. One refresh costs
days + 2*N calls for N new fixtures. A daily run with 9 new fixtures
costs about 20 calls.

Free-tier constraints this module is built around:
- league queries only cover seasons 2022-2024 and the `last`
  parameter is blocked, so fixtures are fetched by date (the only
  reliable window, limited to the last 2-3 days) and filtered to the
  requested league on the client side,
- data refreshes within ~15 minutes of real events: near-live, not live.

State semantics (state version 2):
- player and fixture IDs are canonical strings at every boundary, so
  a state map saved to JSON and reloaded addresses the same players
  as a ratings table whose IDs are numeric,
- processed fixtures are recorded in a processing ledger keyed by
  fixture ID, each entry carrying the fixture's source timestamp,
  its processing status and any errors; membership in the ledger,
  never a highest-ID watermark, decides what still needs processing,
  so fixtures discovered out of order are never skipped,
- updates are transactional and idempotent: a fixture's contributions
  are staged and only committed to the live totals when its events
  and lineups both arrive, and replaying a processed fixture changes
  nothing,
- a fixture whose statistics are missing or failed is recorded as
  such in the ledger and is not counted as zero production.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

SKILL_CLI = os.environ.get("API_FOOTBALL_CLI") or shutil.which("api-football")
STATE_DIR = os.path.join("data", "live")
STATE_PATH = os.path.join(STATE_DIR, "state.json")
STATE_VERSION = 2

# Event-summary proxy weights, in per-90 units on the VAEP scale for
# readability only. Scoring moves P(goal) from ~0.1 to 1.0, so a goal
# is weighted about 0.9. Assists get partial credit. Cards are a
# judgment call: small, negative, documented. These weights define the
# event_summary proxy; they are not VAEP values.
EVENT_PROXY_W_GOAL = 0.9
EVENT_PROXY_W_PENALTY_GOAL = 0.7
EVENT_PROXY_W_ASSIST = 0.4
EVENT_PROXY_W_YELLOW = -0.05
EVENT_PROXY_W_RED = -0.30

# Legacy names kept as aliases for the proxy weights above.
W_GOAL = EVENT_PROXY_W_GOAL
W_PENALTY_GOAL = EVENT_PROXY_W_PENALTY_GOAL
W_ASSIST = EVENT_PROXY_W_ASSIST
W_YELLOW = EVENT_PROXY_W_YELLOW
W_RED = EVENT_PROXY_W_RED

# Live minutes at which the live sample's shrinkage weight reaches
# one half (the proxy sample is shrunk toward the base group mean).
EVENT_PROXY_HALF_LIFE_MIN = 450.0
LIVE_HALF_LIFE_MIN = EVENT_PROXY_HALF_LIFE_MIN

# Recency is a time decay, not a minutes count: a player's accumulated
# live sample loses half its effective weight every
# EVENT_PROXY_HALF_LIFE_DAYS days since it was last seen, i.e.
# effective = raw * 0.5 ** (age_days / EVENT_PROXY_HALF_LIFE_DAYS).
EVENT_PROXY_HALF_LIFE_DAYS = 45.0


def decay_factor(age_days: float) -> float:
    """Exponential time-decay factor for a live sample of a given age."""
    if age_days is None or not np.isfinite(age_days):
        return 1.0
    return float(0.5 ** (max(0.0, float(age_days)) / EVENT_PROXY_HALF_LIFE_DAYS))


class ApiError(RuntimeError):
    pass


def canonical_id(value) -> str | None:
    """Canonical string form of a player or fixture ID.

    Numeric IDs (including integral floats and numpy integers) map to
    their integer string form, so 3053, 3053.0 and "3053" are the same
    key at every boundary.
    """
    if value is None:
        return None
    if isinstance(value, (bool,)):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return str(int(value)) if float(value).is_integer() else str(value)
    return str(value).strip()


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

    Prefers a locally installed provider CLI (its path comes from the
    API_FOOTBALL_CLI environment variable or a PATH lookup); falls back
    to a direct call with API_FOOTBALL_KEY.
    """
    data = None
    if SKILL_CLI and os.path.exists(SKILL_CLI):
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
    """Finished fixtures from the last `days` days for one league.

    Date queries are the only reliable window on the free tier: league
    queries are limited to seasons 2022-2024 and the `last` parameter
    is blocked, so each day is fetched once (all leagues) and filtered
    to `league` on the client side. `season` is kept for call
    compatibility and recorded by callers; it is not sent, because
    sending it makes the free plan reject current-season queries.
    Costs one API call per day scanned.
    """
    from datetime import timedelta

    today = datetime.now(timezone.utc).date()
    seen: set[str] = set()
    out: list[dict] = []
    for i in range(days):
        day = (today - timedelta(days=i)).isoformat()
        resp = api_get("fixtures", date=day)
        for f in resp:
            fid = canonical_id(f.get("fixture", {}).get("id"))
            if fid is None or fid in seen:
                continue
            seen.add(fid)
            if f.get("league", {}).get("id") != league:
                continue
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
    Returns the canonical string form of the player ID, or None when no
    confident match exists.
    """
    key = f"{_norm(api_team)}||{_norm(api_name)}"
    if key in aliases:
        return canonical_id(aliases[key])
    cands = ratings[ratings["team"].apply(lambda t: _team_overlap(t, api_team) >= 0.3)]
    best, best_score = None, 0.0
    target = _norm(api_name)
    for _, row in cands.iterrows():
        score = SequenceMatcher(None, target, _norm(row["player"])).ratio()
        if score > best_score:
            best, best_score = row["player_id"], score
    if best is not None and best_score >= 0.8:
        pid = canonical_id(best)
        aliases[key] = pid
        return pid
    return None


def fixture_contributions(events: list[dict], lineups: list[dict]) -> pd.DataFrame:
    """Per-player event-summary proxy contributions for one fixture.

    Returns one row per player named in the lineups, with columns:
    api_team, api_player, minutes, contrib (the event-summary proxy
    total), events_seen, and stats_status. Lineups and substitutions
    are normalized before minutes are computed: a player listed as
    both starter and substitute counts once as a starter, substitution
    chains are resolved through entry and exit minutes, and a red card
    ends a player's minutes. stats_status distinguishes what the feed
    actually showed: "observed" (the player has at least one event),
    "lineup_only" (minutes played, no events recorded), and "unused"
    (named on the bench, never entered). An unused player is a missing
    observation, not zero production.
    """
    # normalized appearance intervals: (team, name) -> [on, off or None]
    interval: dict[tuple[str, str], list] = {}
    order: list[tuple[str, str]] = []

    def ensure(team, name, on):
        key = (team, name)
        if key not in interval:
            interval[key] = [float(on), None]
            order.append(key)
        return interval[key]

    for side in lineups or []:
        team = side.get("team", {}).get("name", "")
        starters = [p.get("player", {}).get("name", "")
                    for p in side.get("startXI", []) or []]
        subs = [p.get("player", {}).get("name", "")
                for p in side.get("substitutes", []) or []]
        for name in starters:
            if name:
                ensure(team, name, 0.0)
        for name in subs:
            if name and (team, name) not in interval:
                interval[(team, name)] = [None, None]
                order.append((team, name))

    contrib: dict[tuple[str, str], float] = {}
    events_seen: dict[tuple[str, str], int] = {}

    def add(team, name, x):
        if name:
            contrib[(team, name)] = contrib.get((team, name), 0.0) + x

    for ev in events or []:
        team = ev.get("team", {}).get("name", "")
        etype, detail = ev.get("type", ""), ev.get("detail", "")
        name = (ev.get("player", {}) or {}).get("name", "")
        assist = (ev.get("assist", {}) or {}).get("name", "")
        elapsed = (ev.get("time", {}) or {}).get("elapsed")
        elapsed = float(elapsed) if elapsed is not None else None
        for who in (name, assist):
            if who:
                events_seen[(team, who)] = events_seen.get((team, who), 0) + 1
        if etype == "Goal":
            if detail == "Own Goal":
                continue
            add(team, name, EVENT_PROXY_W_PENALTY_GOAL
                if detail == "Penalty" else EVENT_PROXY_W_GOAL)
            add(team, assist, EVENT_PROXY_W_ASSIST)
        elif etype == "Card":
            if detail == "Yellow Card":
                add(team, name, EVENT_PROXY_W_YELLOW)
            elif detail in ("Red Card", "Second Yellow card"):
                add(team, name, EVENT_PROXY_W_RED)
                if name and (team, name) in interval and elapsed is not None:
                    iv = interval[(team, name)]
                    iv[1] = elapsed if iv[1] is None else min(iv[1], elapsed)
        elif etype == "subst":
            if elapsed is None:
                continue
            if name and (team, name) in interval:
                iv = interval[(team, name)]
                iv[1] = elapsed if iv[1] is None else min(iv[1], elapsed)
            if assist:
                iv = ensure(team, assist, elapsed)
                if iv[0] is None:
                    iv[0] = elapsed

    rows = []
    for team, name in order:
        on, off = interval[(team, name)]
        if on is None:
            minutes = 0.0
            status = "unused"
        else:
            minutes = max(0.0, (90.0 if off is None else off) - on)
            status = "observed" if events_seen.get((team, name), 0) else "lineup_only"
        rows.append({
            "api_team": team, "api_player": name, "minutes": minutes,
            "contrib": contrib.get((team, name), 0.0),
            "events_seen": events_seen.get((team, name), 0),
            "stats_status": status,
        })
    return pd.DataFrame(rows, columns=["api_team", "api_player", "minutes",
                                       "contrib", "events_seen", "stats_status"])


# --- state: canonical IDs, processing ledger, atomic saves -------------------

def _canonical_float_map(m: dict | None) -> dict:
    out = {}
    for k, v in (m or {}).items():
        key = canonical_id(k)
        if key is not None:
            out[key] = float(v)
    return out


def _empty_state() -> dict:
    return {"version": STATE_VERSION, "aliases": {}, "live_minutes": {},
            "live_contrib": {}, "live_last_seen": {},
            "processed_fixtures": {}, "watchlist": {}, "calls": {},
            "history": []}


def normalize_state(state: dict | None) -> dict:
    """Return `state` with every ID map keyed by canonical strings.

    Accepts legacy states (a bare fixture-ID watermark and
    int-keyed maps, as JSON round-trips produce) and upgrades them to
    state version 2 without inventing processing history: the legacy
    watermark is preserved as information only and never seeds the
    processing ledger.
    """
    src = dict(state or {})
    out = _empty_state()
    out["aliases"] = {
        str(k): canonical_id(v) for k, v in (src.get("aliases") or {}).items()
        if canonical_id(v) is not None
    }
    out["live_minutes"] = _canonical_float_map(src.get("live_minutes"))
    out["live_contrib"] = _canonical_float_map(src.get("live_contrib"))
    out["live_last_seen"] = {
        canonical_id(k): str(v) for k, v in (src.get("live_last_seen") or {}).items()
        if canonical_id(k) is not None
    }
    ledger = {}
    for k, v in (src.get("processed_fixtures") or {}).items():
        fid = canonical_id(k)
        if fid is None:
            continue
        entry = dict(v or {})
        entry["fixture_id"] = fid
        entry.setdefault("status", "ok")
        entry.setdefault("source_timestamp", None)
        entry.setdefault("errors", [])
        ledger[fid] = entry
    out["processed_fixtures"] = ledger
    out["watchlist"] = {
        str(k): float(v) for k, v in (src.get("watchlist") or {}).items()
    }
    out["calls"] = dict(src.get("calls") or {})
    out["history"] = list(src.get("history") or [])
    if src.get("last_fixture_id") is not None:
        out["last_fixture_id"] = src["last_fixture_id"]
    return out


def processed_fixture_ids(state: dict) -> set[str]:
    """Fixture IDs whose ledger status is a completed processing."""
    ledger = (state or {}).get("processed_fixtures") or {}
    return {canonical_id(k) for k, v in ledger.items()
            if isinstance(v, dict) and v.get("status") == "ok"}


def unprocessed_fixtures(fixtures: list[dict], state: dict) -> list[dict]:
    """Fixtures not yet recorded as processed, in discovery order.

    Selection is by ledger membership only; fixture IDs are never
    compared to each other, so a fixture discovered after a higher-ID
    fixture was processed is still returned.
    """
    done = processed_fixture_ids(state)
    out, seen = [], set()
    for f in fixtures:
        fid = canonical_id(f.get("fixture", {}).get("id"))
        if fid is None or fid in done or fid in seen:
            continue
        seen.add(fid)
        out.append(f)
    return out


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return normalize_state(json.load(f))
    return _empty_state()


def save_state(state: dict) -> None:
    """Save the normalized state atomically (temp file plus rename)."""
    os.makedirs(STATE_DIR, exist_ok=True)
    payload = normalize_state(state)
    fd, tmp = tempfile.mkstemp(dir=STATE_DIR, prefix=".state-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, STATE_PATH)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# --- versioned baseline ratings artifact --------------------------------------

BASELINE_VERSION = "baseline-v1"
BASELINE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "runs", "live", "baseline_ratings_v1.json")


def _ratings_sha256(ratings: list[dict]) -> str:
    blob = json.dumps(ratings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_baseline_artifact(path: str | None = None) -> dict:
    """Load the versioned baseline artifact and verify its contents hash."""
    with open(path or BASELINE_PATH) as f:
        artifact = json.load(f)
    if artifact.get("version") != BASELINE_VERSION:
        raise ValueError(
            f"baseline artifact version {artifact.get('version')!r} "
            f"!= {BASELINE_VERSION!r}")
    digest = _ratings_sha256(artifact.get("ratings") or [])
    if digest != artifact.get("ratings_sha256"):
        raise ValueError("baseline artifact ratings do not match their recorded hash")
    return artifact


def load_baseline_ratings(path: str | None = None) -> pd.DataFrame:
    """Base ratings from the immutable baseline artifact.

    The artifact is tied to one fitted model and one data snapshot
    (recorded in its metadata); reloading it never recomputes ratings
    from whatever model happens to be in memory. Player IDs are
    returned as canonical strings.
    """
    artifact = load_baseline_artifact(path)
    df = pd.DataFrame(artifact["ratings"])
    df["player_id"] = df["player_id"].map(canonical_id)
    return df


def write_baseline_artifact(ratings: pd.DataFrame, path: str, *,
                            model: str, data_snapshot: str,
                            config_hash: str | None = None,
                            feature_version: int | None = None,
                            seed: int | None = None) -> dict:
    """Write a baseline artifact from a ratings frame (build-time only)."""
    rows = []
    for _, r in ratings.iterrows():
        rows.append({
            "player_id": canonical_id(r["player_id"]),
            "player": r["player"], "team": r["team"],
            "pos_group": r["pos_group"],
            "vaep_per90": float(r["vaep_per90"]),
            "wage_eur": float(r["wage_eur"]),
            "minutes": float(r["minutes"]),
        })
    rows.sort(key=lambda r: r["player_id"])
    artifact = {
        "version": BASELINE_VERSION, "model": model,
        "data_snapshot": data_snapshot, "config_hash": config_hash,
        "feature_version": feature_version, "seed": seed,
        "ratings": rows, "ratings_sha256": _ratings_sha256(rows),
    }
    with open(path, "w") as f:
        json.dump(artifact, f, indent=1)
    return artifact


# --- blending -------------------------------------------------------------------

def _as_date(value):
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.date() if not pd.isna(ts) else None


def blend_ratings(ratings: pd.DataFrame, state: dict,
                  shrink_k: float = 900.0, as_of=None) -> pd.DataFrame:
    """Fold accumulated live event-summary proxies into the ratings.

    The base ratings are already shrunk toward their position-group
    means by the main pipeline, so the blend never shrinks them again:
    only the live proxy sample is shrunk (toward the base group mean,
    by its effective minutes), and base and shrunk proxy are combined
    in proportion to base minutes and decayed live minutes. With zero
    live minutes the result is the base rating, exactly.

    Recency enters through the time decay in `decay_factor`: each
    player's live totals are weighted by how long ago the player was
    last seen. `as_of` defaults to the latest last-seen date in the
    state, so a replayed blend is deterministic.
    """
    out = ratings.copy()
    out["player_id"] = out["player_id"].map(canonical_id)
    st = normalize_state(state)
    live_min_raw = out["player_id"].map(st["live_minutes"]).fillna(0.0)
    live_val_raw = out["player_id"].map(st["live_contrib"]).fillna(0.0)

    last_seen = st["live_last_seen"]
    reference = _as_date(as_of)
    if reference is None and last_seen:
        dates = [d for d in (_as_date(v) for v in last_seen.values()) if d]
        reference = max(dates) if dates else None
    factors = []
    for pid in out["player_id"]:
        seen = _as_date(last_seen.get(pid))
        if seen is None or reference is None:
            factors.append(1.0)
        else:
            factors.append(decay_factor((reference - seen).days))
    factor = pd.Series(factors, index=out.index, dtype=float)

    eff_min = live_min_raw * factor
    eff_val = live_val_raw * factor
    out["live_minutes"] = live_min_raw
    out["live_effective_minutes"] = eff_min

    proxy_per90 = (eff_val / eff_min.clip(lower=1e-9) * 90.0).where(
        eff_min > 0, 0.0).fillna(0.0)
    out["event_proxy_per90"] = proxy_per90
    out["live_per90"] = proxy_per90  # legacy alias

    group_mean = out.groupby("pos_group")["vaep_per90"].transform("mean")
    w_shrink = eff_min / (eff_min + shrink_k)
    proxy_shrunk = w_shrink * proxy_per90 + (1 - w_shrink) * group_mean

    denom = out["minutes"] + eff_min
    w_live = (eff_min / denom).where(denom > 0, 0.0).fillna(0.0)
    blended = (1 - w_live) * out["vaep_per90"] + w_live * proxy_shrunk
    out["event_proxy_blended_per90"] = blended
    out["vaep_per90_live"] = blended  # legacy alias for the blended column
    return out


def rebalance_report(old: pd.DataFrame, new: pd.DataFrame,
                     ratings: pd.DataFrame) -> pd.DataFrame:
    """Who leaves and who enters when the squad is re-optimized."""
    old_ids = {canonical_id(p) for p in old["player_id"]}
    new_ids = {canonical_id(p) for p in new["player_id"]}
    r = ratings.copy()
    r["player_id"] = r["player_id"].map(canonical_id)
    r = r.set_index("player_id")
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
