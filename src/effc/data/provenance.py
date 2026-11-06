"""Raw-event, wage and model provenance for the demo and the pipeline.

The demo metadata used to be a bare config summary. This module records
the complete provenance the contract asks for: source URLs, retrieval
dates, licensing notes, the wage-matching audit, the config hash and the
wage units. Every fact here was verified against the live sources on
2026-11-05; anything not recorded anywhere is stated as not recorded,
never invented.

License facts (verified 2026-11-05):
- StatsBomb open data is governed by the StatsBomb Public Data User
  Agreement (last updated 8 September 2023), not by a Creative Commons
  license. It is free for non-commercial research and personal use; the
  data and any derived analysis may not be commercially exploited, and
  any publication of analysis must credit StatsBomb with its brand logo.
- The EA FC 24 wage file comes from a third-party mirror of EA Sports
  game data. Its source repository page and the raw file URL recorded in
  ``effc.data.download`` both return 404 when checked on 2026-11-05,
  and no license statement for the wage file is recorded in this repo.
  The file is therefore used as a wage proxy only and is not
  redistributed; a fresh clone cannot re-fetch it until a live source
  is found and recorded here.

Wage units: ``wage_eur`` is EUR per week, the unit EA FC 24 reports and
the unit the README budgets use. Annual figures are weekly x 52.
"""

from __future__ import annotations

import hashlib
import json

from effc.data.wages import summarize_wage_matches

WAGE_UNIT = "EUR/week"
WEEKS_PER_YEAR = 52

STATS_BOMB_URL = "https://github.com/statsbomb/open-data"
STATS_BOMB_DATA_URL = (
    "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
)
STATS_BOMB_LICENSE_URL = (
    "https://raw.githubusercontent.com/statsbomb/open-data/master/LICENSE.pdf"
)
FC24_URL = "https://raw.githubusercontent.com/mzafram2001/ea-fc/main/data/dataset_ea_fc_24.csv"

# Retrieval dates are not recorded in this repo for either source, so the
# record says so explicitly. Record a real date here only when a download
# is actually performed and dated.
RETRIEVAL_NOT_RECORDED = "not recorded in repo (verified 2026-11-05)"

SOURCES = [
    {
        "name": "statsbomb_open_data",
        "role": "raw event data (matches, events, lineups)",
        "url": STATS_BOMB_URL,
        "data_url": STATS_BOMB_DATA_URL,
        "license": "StatsBomb Public Data User Agreement (last updated 8 September 2023)",
        "license_url": STATS_BOMB_LICENSE_URL,
        "license_note": (
            "Free for non-commercial research and personal use; no "
            "commercial exploitation of the data or derived analysis; "
            "publications of analysis must credit StatsBomb with its "
            "brand logo. Not a Creative Commons license."
        ),
        "retrieval_date": RETRIEVAL_NOT_RECORDED,
    },
    {
        "name": "ea_fc_24_wages",
        "role": "wage proxy (EA FC 24 player wages)",
        "url": FC24_URL,
        "data_url": FC24_URL,
        "license": "No license statement found",
        "license_url": None,
        "license_note": (
            "No license statement for the wage file is recorded in this "
            "repo. The source repository page and the raw file URL both "
            "return 404 when checked on 2026-11-05, so the file is used "
            "as a wage proxy only and is not redistributed."
        ),
        "retrieval_date": RETRIEVAL_NOT_RECORDED,
    },
]


def annual_wage_eur(weekly_eur: float) -> float:
    """Annual EUR for a weekly EUR wage (52 weeks per year)."""
    return float(weekly_eur) * WEEKS_PER_YEAR


def config_hash(config_path: str) -> str:
    """SHA-256 hex digest of a config file's bytes."""
    digest = hashlib.sha256()
    with open(config_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matching_audit(matched) -> dict:
    """Wage-matching audit for a matched frame (see wages.match_players).

    Returns exact/fuzzy/proxy/unmatched counts, the player total and the
    matched share. Proxy rows are labeled stand-ins, not matches.
    """
    counts = summarize_wage_matches(matched)
    n_players = len(matched)
    n_matched = int(matched["matched"].sum()) if "matched" in matched else 0
    return {
        **counts,
        "n_players": int(n_players),
        "match_rate": (n_matched / n_players) if n_players else 0.0,
    }


def build_provenance(
    config_path: str,
    matched,
    feature_version: int,
    seed: int,
    run_name: str | None = None,
) -> dict:
    """Assemble the provenance record for a demo or pipeline run."""
    record = {
        "sources": [dict(s) for s in SOURCES],
        "config_path": config_path,
        "config_hash": config_hash(config_path),
        "feature_version": int(feature_version),
        "seed": int(seed),
        "wage_units": {
            "weekly": WAGE_UNIT,
            "annual": "EUR/year (weekly x 52)",
        },
        "matching_audit": matching_audit(matched),
    }
    if run_name is not None:
        record["run_name"] = run_name
    json.dumps(record)
    return record
