"""Download and cache StatsBomb open data + EA FC wage data.

Everything lands under data/ (gitignored). Run once:

    python -m effc.data.download

Uses curl (more reliable here than python requests). The VAEP
probability models train on a pooled set of competitions (goal
probability given game state is roughly competition-independent).
Player ratings are computed on the target competition only.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
FC24_URL = "https://raw.githubusercontent.com/mzafram2001/ea-fc/main/data/dataset_ea_fc_24.csv"

# Pinned upstream revisions: the branch each source URL tracks. The
# per-file SHA-256 recorded at download time (see _write_provenance)
# pins the exact artifact bytes on top of the revision name.
UPSTREAM_REVISIONS = {
    "statsbomb_open_data": "master",
    "ea_fc_24": "main",
}

# competitions whose events train the P(scores) / P(concedes) models
MODEL_POOL = [
    (9, 281),   # 1. Bundesliga 2023/2024
    (9, 27),    # 1. Bundesliga 2015/2016
    (43, 106),  # FIFA World Cup 2022
    (43, 3),    # FIFA World Cup 2018
]

# the competitions players are rated (and squads built) on: two Bundesliga
# seasons from StatsBomb open data (34 matches each; it is all they publish)
TARGET = [(9, 281), (9, 27)]  # 1. Bundesliga 2023/2024 + 2015/2016

# Additional competitions pooled ONLY as goal-model training data (the
# beat-baseline push). None of these is a target competition, and none
# contains a held-out or season-transfer test match: the test sets are
# Bundesliga matches drawn from MODEL_POOL / TARGET above. Pooled matches
# are used for training and internal validation only, never scored as a
# test set, and never used for player ratings.
#
# Coverage note (verified against the open-data match lists): StatsBomb
# publishes FULL league seasons for 2015/2016 (La Liga, Premier League,
# Serie A, Ligue 1 below). Other La Liga seasons are Barcelona-only
# samples (31-38 matches each) and Champions League seasons are the
# final only, so they are deliberately not pooled: a one-team sample
# would skew team strengths for almost no extra data.
EXTRA_POOL = [
    (55, 43),     # UEFA Euro 2020 (full tournament, 51 matches)
    (1238, 108),  # Indian Super League 2021/2022 (full season, 115)
]

# Contemporaneous 2015/2016 leagues (full seasons): the only pooled
# data the season-transfer model may train on besides its 2015/2016
# Bundesliga training matches (a model "trained in 2016" cannot use
# later data). The main models pool them too.
ERA_2015_POOL = [
    (11, 27),   # La Liga 2015/2016 (380 matches)
    (12, 27),   # Serie A 2015/2016 (380 matches)
    (7, 27),    # Ligue 1 2015/2016 (377 matches)
    (2, 27),    # Premier League 2015/2016 (380 matches)
]

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
)


def _payload_kind(dest: str) -> str:
    """Which schema a destination file must satisfy."""
    parts = dest.replace(os.sep, "/").split("/")
    base = parts[-1]
    if base.endswith(".csv"):
        return "csv"
    if "events" in parts:
        return "events"
    if "lineups" in parts:
        return "lineups"
    if base.startswith("matches_"):
        return "matches"
    return "json"


def validate_payload(raw: bytes, dest: str):
    """Parse and schema-validate downloaded bytes for `dest`.

    Returns the parsed payload. Raises ValueError on truncated or
    malformed JSON, on an HTTP error document saved as a body, and on
    JSON that parses but does not match the schema the destination
    requires. Nothing is accepted on existence/size alone.
    """
    kind = _payload_kind(dest)
    if kind == "csv":
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{dest}: wage CSV is not UTF-8 text: {exc}") from exc
        header = text.splitlines()[0].split(",") if text.splitlines() else []
        missing = {"long_name", "wage_eur"} - {h.strip() for h in header}
        if missing:
            raise ValueError(f"{dest}: wage CSV header missing {sorted(missing)}")
        return text
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{dest}: downloaded body is not valid JSON: {exc}") from exc
    if kind == "matches":
        if not isinstance(data, list) or not data:
            raise ValueError(f"{dest}: matches payload must be a non-empty list")
        for row in data:
            if not isinstance(row, dict) or "match_id" not in row or "match_date" not in row:
                raise ValueError(f"{dest}: matches row lacks match_id/match_date: {row!r:.120}")
    elif kind == "events":
        if not isinstance(data, list):
            raise ValueError(f"{dest}: events payload must be a list")
        for row in data[:50]:
            if not isinstance(row, dict) or "id" not in row or "type" not in row:
                raise ValueError(f"{dest}: event row lacks id/type: {row!r:.120}")
    elif kind == "lineups":
        if not isinstance(data, list):
            raise ValueError(f"{dest}: lineups payload must be a list")
        for row in data:
            if not isinstance(row, dict) or "lineup" not in row or (
                "team_name" not in row and "team_id" not in row
            ):
                raise ValueError(f"{dest}: lineup row lacks team/lineup: {row!r:.120}")
    return data


def validate_cached_file(dest: str) -> bool:
    """True only when an existing cache file parses and validates."""
    try:
        with open(dest, "rb") as f:
            validate_payload(f.read(), dest)
        return True
    except (OSError, ValueError):
        return False


def artifact_sha256(path: str) -> str:
    """SHA-256 hex digest of a file's bytes (the pinned artifact hash)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_provenance(dest: str, url: str, raw: bytes) -> None:
    """Sidecar provenance for an accepted download: source URL, pinned
    upstream revision, artifact SHA-256 and the validation outcome."""
    upstream = (
        UPSTREAM_REVISIONS["ea_fc_24"]
        if "ea-fc" in url
        else UPSTREAM_REVISIONS["statsbomb_open_data"]
    )
    record = {
        "url": url,
        "upstream": upstream,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "schema": _payload_kind(dest),
        "validated": True,
    }
    with open(dest + ".provenance.json", "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)


def _fetch(url: str, dest: str, retries: int = 5) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        if validate_cached_file(dest):
            return
        # A cached file that does not parse/validate (for example an
        # HTTP error document stored by an older run) is never trusted:
        # drop it and fetch a verified copy instead.
        print(f"cached file failed validation, refetching: {dest}")
        os.remove(dest)
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = dest + ".tmp"
    last_err = ""
    for attempt in range(retries):
        r = subprocess.run(
            ["curl", "-sS", "-f", "--max-time", "300", "-o", tmp, url],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            with open(tmp, "rb") as f:
                raw = f.read()
            try:
                validate_payload(raw, dest)
            except ValueError as exc:
                last_err = str(exc)[:200]
                os.remove(tmp)
                time.sleep(2 * (attempt + 1))
                continue
            # Only a parsed, schema-valid payload is atomically
            # accepted; a partial temp file never becomes the cache.
            os.replace(tmp, dest)
            _write_provenance(dest, url, raw)
            return
        last_err = r.stderr.strip()[:200]
        if os.path.exists(tmp):
            os.remove(tmp)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed {url}: {last_err}")


def download() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    match_ids: set[int] = set()

    for comp_id, season_id in MODEL_POOL:
        dest = os.path.join(DATA_DIR, f"matches_{comp_id}_{season_id}.json")
        _fetch(f"{BASE}/matches/{comp_id}/{season_id}.json", dest)
        with open(dest) as f:
            matches = json.load(f)
        for m in matches:
            match_ids.add(m["match_id"])
        print(f"competition {comp_id}/{season_id}: {len(matches)} matches")

    print(f"total unique matches: {len(match_ids)}")
    with open(os.path.join(DATA_DIR, "match_ids.json"), "w") as f:
        json.dump(sorted(match_ids), f)

    events_dir = os.path.join(DATA_DIR, "events")
    os.makedirs(events_dir, exist_ok=True)
    lineups_dir = os.path.join(DATA_DIR, "lineups")
    os.makedirs(lineups_dir, exist_ok=True)

    jobs = []
    for mid in sorted(match_ids):
        jobs.append((f"{BASE}/events/{mid}.json", os.path.join(events_dir, f"{mid}.json")))
        jobs.append((f"{BASE}/lineups/{mid}.json", os.path.join(lineups_dir, f"{mid}.json")))

    def one(job):
        url, dest = job
        try:
            _fetch(url, dest)
            return True
        except Exception as exc:
            return f"{dest}: {exc}"

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(one, j): j for j in jobs}
        for fut in tqdm(as_completed(futures), total=len(jobs), desc="files"):
            res = fut.result()
            if res is not True:
                print(f"warn: {res}")

    _fetch(FC24_URL, os.path.join(DATA_DIR, "fc24_wages.csv"))
    print(f"data ready in {DATA_DIR}")
    return DATA_DIR


def download_extra() -> list[tuple[int, int]]:
    """Fetch match lists + events for EXTRA_POOL and ERA_2015_POOL.

    Events only: pooled matches never feed player ratings, so lineups
    (minutes) are not needed. Returns the pooled (competition, season)
    pairs. Idempotent: existing files are kept.
    """
    pools = sorted(set(EXTRA_POOL) | set(ERA_2015_POOL))
    os.makedirs(DATA_DIR, exist_ok=True)
    events_dir = os.path.join(DATA_DIR, "events")
    os.makedirs(events_dir, exist_ok=True)
    match_ids: list[int] = []
    for comp_id, season_id in pools:
        dest = os.path.join(DATA_DIR, f"matches_{comp_id}_{season_id}.json")
        _fetch(f"{BASE}/matches/{comp_id}/{season_id}.json", dest)
        with open(dest) as f:
            matches = json.load(f)
        match_ids.extend(m["match_id"] for m in matches)
        print(f"competition {comp_id}/{season_id}: {len(matches)} matches")

    def one(mid):
        url = f"{BASE}/events/{mid}.json"
        dest = os.path.join(events_dir, f"{mid}.json")
        try:
            _fetch(url, dest)
            return True
        except Exception as exc:
            return f"{dest}: {exc}"

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(one, mid): mid for mid in match_ids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="extra"):
            res = fut.result()
            if res is not True:
                print(f"warn: {res}")
    print(f"extra pool ready: {len(match_ids)} matches")
    return pools


if __name__ == "__main__":
    download()
