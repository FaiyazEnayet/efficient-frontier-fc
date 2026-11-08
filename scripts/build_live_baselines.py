"""Build the per-league live-refresh baseline artifacts.

One artifact per top-five league, in the same versioned schema as the
original Bundesliga snapshot (runs/live/baseline_ratings_v1.json, which
this script verifies but never overwrites: it comes from the
bundesliga_2seasons run and its demo snapshot).

The other four leagues come from the fitted five-league run published
in runs/five_leagues/results.json (config fc6_pool, FEATURE_VERSION 4,
seed 0). Nothing is retrained here: the script loads the fitted model
bundle from the local data cache, checks that its hash matches the
model hash recorded in the published results, applies it to the
cached featurized frames, and rebuilds each league's ratings with the
same steps scripts/run_five_leagues.py used (VAEP values, minutes from
lineups, EA FC 24 wage matching, build_ratings with min_minutes 90 and
shrink_k 900). Before an artifact is written, the rebuilt ratings are
checked against the published run: the rated-player count and wage
match rate must match, and every squad row the run published must be
present with the same rating, wage and position group.

The local data cache (data/, gitignored) is required: the fitted
bundle, the featurized five-league frames, the lineup files and the
wage file. Without it a league is reported as not built and nothing
is written for it; no ratings are invented.

Usage: PYTHONPATH=src python scripts/build_live_baselines.py
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402

from effc.data.download import DATA_DIR  # noqa: E402
from effc.data.statsbomb import minutes_played  # noqa: E402
from effc.data.wages import load_wages, match_players  # noqa: E402
from effc.leagues import LEAGUES, audit_league  # noqa: E402
from effc.live import (  # noqa: E402
    baseline_path_for_league,
    load_baseline_artifact,
    write_baseline_artifact,
)
from effc.ratings.ratings import build_ratings  # noqa: E402
from effc.vaep.values import add_vaep, player_match_vaep  # noqa: E402

RESULTS_PATH = os.path.join("runs", "five_leagues", "results.json")
FRAMES_CACHE = os.path.join(DATA_DIR, "cache", "pool_frames_five_leagues.pkl")

# Provider league IDs (the keys of effc.live.LEAGUE_BASELINES) by the
# league names used in the five-league run.
LEAGUE_IDS = {
    "English Premier League": 39,
    "La Liga": 140,
    "Serie A": 135,
    "Bundesliga": 78,
    "Ligue 1": 61,
}

SEASONS = {
    "English Premier League": "2015/2016",
    "La Liga": "2015/2016",
    "Serie A": "2015/2016",
    "Bundesliga": "2015/2016 and 2023/2024",
    "Ligue 1": "2015/2016",
}


def _load_fitted_models(expected_hash: str):
    """Find the cached fitted bundle whose model hash matches the
    hash recorded in the published five-league results."""
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "cache", "improved_models_*.pkl"))):
        with open(path, "rb") as f:
            models, _report = pickle.load(f)
        digest = hashlib.sha256(pickle.dumps(models, protocol=4)).hexdigest()[:16]
        if digest == expected_hash:
            print(f"fitted bundle: {path} (model hash {digest})")
            return models
    return None


def _check_against_published(ratings: pd.DataFrame, wage_match_rate: float,
                             entry: dict, name: str) -> None:
    if len(ratings) != entry["n_rated"]:
        raise ValueError(
            f"{name}: rebuilt {len(ratings)} rated players, "
            f"published run has {entry['n_rated']}")
    if abs(wage_match_rate - entry["wage_match_rate"]) > 1e-9:
        raise ValueError(
            f"{name}: rebuilt wage match rate {wage_match_rate} != "
            f"published {entry['wage_match_rate']}")
    by_player_team = {(r.player, r.team): r for r in ratings.itertuples()}
    for budget in entry["squads"].values():
        for row in budget.get("squad", []):
            got = by_player_team.get((row["player"], row["team"]))
            if got is None:
                raise ValueError(f"{name}: published squad player missing: {row}")
            if (abs(got.vaep_per90 - row["vaep_per90"]) > 1e-9
                    or abs(got.wage_eur - row["wage_eur"]) > 1e-6
                    or got.pos_group != row["pos_group"]):
                raise ValueError(f"{name}: published squad row differs: {row} vs {got}")


def main() -> None:
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    model_hash = results["model_hash"]

    # The Bundesliga artifact already exists and is not rebuilt here.
    bundesliga = load_baseline_artifact(baseline_path_for_league(78))
    print(f"Bundesliga: existing artifact kept "
          f"({len(bundesliga['ratings'])} rated, hash verified)")

    if not os.path.exists(FRAMES_CACHE):
        print("not built: featurized five-league frames are not in the "
              f"local data cache ({FRAMES_CACHE}); no ratings written")
        return
    models = _load_fitted_models(model_hash)
    if models is None:
        print("not built: no cached fitted bundle matches the model "
              f"hash {model_hash} recorded in {RESULTS_PATH}; "
              "no ratings written")
        return
    with open(FRAMES_CACHE, "rb") as f:
        frames = pickle.load(f)
    by_id = {int(fr["match_id"].iloc[0]): fr for fr in frames}

    wages = load_wages()
    for league in LEAGUES:
        name = league["name"]
        league_id = LEAGUE_IDS[name]
        if league_id == 78:
            continue
        entry = results["leagues"][name]
        audit = audit_league(league, DATA_DIR)
        pm_all, min_all = [], []
        for mid in sorted(audit["match_ids"]):
            if mid not in by_id:
                raise ValueError(f"{name}: no featurized frame for match {mid}")
            pm_all.append(player_match_vaep(add_vaep(by_id[mid], models)))
            min_all.append(minutes_played(mid))
        pm = pd.concat(pm_all, ignore_index=True)
        mins = pd.concat(min_all, ignore_index=True)
        sb_players = pm[["player", "team"]].drop_duplicates().reset_index(drop=True)
        wmap = match_players(sb_players, wages)
        wage_match_rate = float(wmap["matched"].mean())
        ratings = build_ratings(pm, mins, wmap, min_minutes=90, shrink_k=900)
        _check_against_published(ratings, wage_match_rate, entry, name)
        out_path = baseline_path_for_league(league_id)
        artifact = write_baseline_artifact(
            ratings, out_path,
            model=("improved goal models, config fc6_pool "
                   "(five-league run, runs/five_leagues/results.json)"),
            data_snapshot=(
                f"StatsBomb open data, {name} {SEASONS[name]} "
                f"({audit['matches']} matches), EA FC 24 wages; ratings as "
                f"published in runs/five_leagues/results.json "
                f"(n_rated {entry['n_rated']})"),
            config_hash=None,
            feature_version=results["feature_version"],
            seed=results["seed"],
            league_id=league_id, league=name, model_hash=model_hash,
        )
        print(f"{name}: wrote {out_path} "
              f"({len(artifact['ratings'])} rated, "
              f"ratings sha256 {artifact['ratings_sha256']})")


if __name__ == "__main__":
    main()
