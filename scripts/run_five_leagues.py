"""Five-league expansion driver (Stage 3 FIXED-v2).

Per league (English Premier League, La Liga, Serie A, Bundesliga,
Ligue 1), published as-is:

1. Data audit: match, event, and lineup availability, reported first.
   Missing lineups are downloaded from the same public StatsBomb
   open-data source as the existing pipeline; nothing is generated.
2. Frozen scoreboard: own chronological split (60% train, 20%
   validation, 20% test) per league. One improved goal-model set
   (config ``fc6_pool``, the shipped v1.5.0 winner, chosen on v1.5.0
   internal validation, never on these test sets) trains on the
   leagues' train slices plus UEFA Euro 2020 and Indian Super League
   2021/22, with every validation and test match ID excluded from
   training. Each league's test slice is scored exactly once against
   the base rate of that league's own train slice. Validation slices
   are scored for reporting only, never used to tune.
3. VAEP ratings, efficient frontier, and backtest at the same three
   budgets (0.30M, 0.50M, 0.70M EUR) per league, mirroring the shipped
   pipeline steps (ratings min_minutes 90, backtest halves
   min_minutes 60, seed 0).

Writes runs/five_leagues/results.json and runs/five_leagues/report.md.
No version bump and no v2.0.0 tag from this script; the tag stays held
until U01/U02 validation plus owner signoff. No dashboard HTML or
generator output is touched here.

Usage: python scripts/run_five_leagues.py  (from the repo root)
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from effc.backtest.runner import run_backtest, split_matches  # noqa: E402
from effc.data.download import DATA_DIR  # noqa: E402
from effc.data.statsbomb import list_matches, load_events, minutes_played  # noqa: E402
from effc.data.wages import load_wages, match_players  # noqa: E402
from effc.leagues import (  # noqa: E402
    BACKTEST_BUDGETS,
    LEAGUES,
    audit_league,
    chronological_split,
    league_pools,
    selection_fingerprint,
    squad_constraints_ok,
)
from effc.pipeline import featurize_matches  # noqa: E402
from effc.portfolio.portfolio import efficient_frontier, select_squad  # noqa: E402
from effc.ratings.covariance import fit_factor_model, match_vaep_panel, team_factor  # noqa: E402
from effc.ratings.ratings import build_ratings  # noqa: E402
from effc.vaep import calibration as cal  # noqa: E402
from effc.vaep import improved as imp  # noqa: E402
from effc.vaep.features import FEATURE_VERSION  # noqa: E402
from effc.vaep.values import add_vaep, player_match_vaep  # noqa: E402

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
# Pooled training-only competitions, same as the v1.5.0 EXTRA_POOL.
EXTRA_POOL = [(55, 43), (1238, 108)]
SEED = 0
CONFIG_NAME = "fc6_pool"
OUTDIR = os.path.join("runs", "five_leagues")


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _id_hash(ids) -> str:
    joined = ",".join(map(str, sorted(int(i) for i in ids)))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def _fetch_lineup(mid: int) -> bool:
    dest = os.path.join(DATA_DIR, "lineups", f"{mid}.json")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    tmp = dest + ".tmp"
    for _attempt in range(3):
        r = subprocess.run(
            ["curl", "-sS", "--max-time", "120", "-o", tmp,
             f"{BASE}/lineups/{mid}.json"],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, dest)
            return True
    return False


def download_missing_lineups(match_ids: list[int]) -> tuple[int, int]:
    """Download lineup files absent from disk, in parallel. Returns (ok, failed)."""
    missing = [
        mid for mid in match_ids
        if not (os.path.exists(os.path.join(DATA_DIR, "lineups", f"{mid}.json"))
                and os.path.getsize(os.path.join(DATA_DIR, "lineups", f"{mid}.json")) > 0)
    ]
    if not missing:
        return 0, 0
    ok = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_fetch_lineup, mid): mid for mid in missing}
        for fut in as_completed(futures):
            ok += 1 if fut.result() else 0
    return ok, len(missing) - ok


def main() -> None:
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    pools = league_pools()
    results: dict = {
        "feature_version": FEATURE_VERSION,
        "model_config": CONFIG_NAME,
        "seed": SEED,
        "budgets_eur": BACKTEST_BUDGETS,
        "leagues": {},
    }

    # 1. Audit, then download missing lineups from the same open source.
    league_meta: dict[str, pd.DataFrame] = {}
    league_audits: dict[str, dict] = {}
    for league in LEAGUES:
        name = league["name"]
        meta = list_matches(pools[name])
        league_meta[name] = meta
        audit = audit_league(league, DATA_DIR)
        league_audits[name] = audit
        print(f"[audit] {name}: {audit['matches']} matches, "
              f"events {audit['events_present']}/{audit['matches']}, "
              f"lineups {audit['lineups_present']}/{audit['matches']}")
    all_ids = sorted({mid for a in league_audits.values() for mid in a["match_ids"]})
    ok, failed = download_missing_lineups(all_ids)
    print(f"[lineups] downloaded {ok}, failed {failed} of "
          f"{len(all_ids)} league matches")
    for league in LEAGUES:
        name = league["name"]
        audit = audit_league(league, DATA_DIR)
        league_audits[name] = audit
        print(f"[audit-post] {name}: lineups "
              f"{audit['lineups_present']}/{audit['matches']}")

    # 2. Featurize every involved match once (fresh v3 features; the
    # work data dir carries no derived caches).
    extra_meta = list_matches(EXTRA_POOL)
    involved = sorted(
        set(all_ids) | set(extra_meta["match_id"].astype(int))
    )
    frames = featurize_matches(involved, "pool_frames_five_leagues.pkl")
    by_id = {}
    for f in frames:
        by_id[int(f["match_id"].iloc[0])] = f
    print(f"[featurize] {len(frames)}/{len(involved)} matches featurized")

    # 3. Per-league chronological splits; union exclusion for training.
    splits: dict[str, dict] = {}
    val_test_union: set[int] = set()
    for league in LEAGUES:
        name = league["name"]
        train, val, test = chronological_split(league_meta[name])
        splits[name] = {"train": train, "val": val, "test": test}
        val_test_union |= val | test
        print(f"[split] {name}: train {len(train)}, val {len(val)}, "
              f"test {len(test)}")
    train_pool_ids = sorted(
        {mid for s in splits.values() for mid in s["train"]}
        | set(extra_meta["match_id"].astype(int))
    )
    cal.assert_disjoint(train_pool_ids, sorted(val_test_union), "train/val-test")
    group_frames = [by_id[mid] for s in splits.values() for mid in s["train"]
                    if mid in by_id]
    pool_frames = [by_id[mid] for mid in extra_meta["match_id"].astype(int)
                   if mid in by_id]
    meta_all = pd.concat(
        list(league_meta.values()) + [extra_meta], ignore_index=True
    ).drop_duplicates("match_id").reset_index(drop=True)

    # 4. One improved model set on the union train pool (config fixed
    # from v1.5.0 internal validation; nothing is tuned on test sets).
    print(f"[train] improved models ({CONFIG_NAME}) on {len(group_frames)} "
          f"league-train + {len(pool_frames)} pooled frames")
    models, calib_report = imp.train_improved_models_cached(
        group_frames, pool_frames, meta_all, CONFIG_NAME, seed=SEED,
    )
    model_hash = hashlib.sha256(pickle.dumps(models, protocol=4)).hexdigest()[:16]
    results["model_hash"] = model_hash
    results["train_pool_matches"] = len(train_pool_ids)

    # 5-7. Per-league scoreboard, ratings, frontier, backtest.
    wages = load_wages()
    for league in LEAGUES:
        name = league["name"]
        sp = splits[name]
        frames_by_id = by_id
        train_frames = [frames_by_id[m] for m in sorted(sp["train"]) if m in frames_by_id]
        val_frames = [frames_by_id[m] for m in sorted(sp["val"]) if m in frames_by_id]
        test_frames = [frames_by_id[m] for m in sorted(sp["test"]) if m in frames_by_id]
        test_model = cal.per_match_scores(models, test_frames)
        test_base = cal.base_rate_scores(train_frames, test_frames)
        val_model = cal.per_match_scores(models, val_frames)
        val_base = cal.base_rate_scores(train_frames, val_frames)
        entry: dict = {
            "audit": {k: v for k, v in league_audits[name].items() if k != "match_ids"},
            "split": {
                "train_n": len(sp["train"]),
                "val_n": len(sp["val"]),
                "test_n": len(sp["test"]),
                "train_hash": _id_hash(sp["train"]),
                "val_hash": _id_hash(sp["val"]),
                "test_hash": _id_hash(sp["test"]),
            },
            "scoreboard_test": {
                label: {
                    "model": cal.summarize(test_model)[label],
                    "base_rate": cal.summarize(test_base)[label],
                }
                for label in cal.LABELS
            },
            "scoreboard_val_report_only": {
                label: {
                    "model": cal.summarize(val_model)[label],
                    "base_rate": cal.summarize(val_base)[label],
                }
                for label in cal.LABELS
            },
        }

        # Target VAEP + ratings on the league's own matches.
        pm_all, min_all = [], []
        for mid in sorted(league_audits[name]["match_ids"]):
            if mid not in frames_by_id:
                continue
            df = add_vaep(frames_by_id[mid], models)
            pm_all.append(player_match_vaep(df))
            try:
                min_all.append(minutes_played(mid))
            except FileNotFoundError:
                continue
        pm = pd.concat(pm_all, ignore_index=True)
        mins = pd.concat(min_all, ignore_index=True)
        sb_players = pm[["player", "team"]].drop_duplicates().reset_index(drop=True)
        wmap = match_players(sb_players, wages)
        entry["wage_match_rate"] = float(wmap["matched"].mean())
        ratings = build_ratings(pm, mins, wmap, min_minutes=90, shrink_k=900)
        entry["n_rated"] = len(ratings)
        entry["pm_matches"] = int(pm["match_id"].nunique())
        print(f"[ratings] {name}: {len(ratings)} rated, wage match "
              f"{entry['wage_match_rate']:.1%}")

        # Frontier at the same three budgets.
        panel = match_vaep_panel(pm, mins)
        player_team = ratings.set_index("player_id")["team"]
        factors = team_factor(panel, player_team)
        sigma, cov_info = fit_factor_model(panel, factors, player_team)
        idio = pd.Series(np.diag(sigma.to_numpy()), index=sigma.index)
        frontier = efficient_frontier(
            ratings, np.array(BACKTEST_BUDGETS, dtype=float),
            gamma=0.0, idio_var=idio,
        )
        entry["frontier"] = frontier.to_dict("records")

        # Backtest on the within-season date split.
        first, second = split_matches(league_meta[name], frac=0.5)
        pm1 = pm[pm["match_id"].isin(first["match_id"])]
        pm2 = pm[pm["match_id"].isin(second["match_id"])]
        r1 = build_ratings(pm1, mins, wmap, min_minutes=60, shrink_k=900)
        r2 = build_ratings(pm2, mins, wmap, min_minutes=60, shrink_k=900)
        bt = run_backtest(r1, r2, pm2, mins, list(BACKTEST_BUDGETS), seed=SEED)
        entry["backtest"] = bt.to_dict("records")
        print(f"[backtest] {name}: {len(bt)} budget rows")

        # Squad-constraint + determinism checks on the full ratings.
        checks = {}
        for b in BACKTEST_BUDGETS:
            try:
                squad = select_squad(ratings, float(b), gamma=0.0, idio_var=idio)
                twice = select_squad(ratings, float(b), gamma=0.0, idio_var=idio)
                checks[str(int(b))] = {
                    "constraints_ok": bool(squad_constraints_ok(squad, float(b))),
                    "regeneration_identical": bool(
                        selection_fingerprint(squad)
                        == selection_fingerprint(twice)
                    ),
                    "n_players": len(squad),
                }
            except RuntimeError as exc:
                checks[str(int(b))] = {"infeasible": str(exc)}
        entry["squad_checks"] = checks
        results["leagues"][name] = entry

    results["elapsed_s"] = time.time() - t0
    with open(os.path.join(OUTDIR, "results.json"), "w") as f:
        json.dump(_clean(results), f, indent=2)
    print(f"\nsaved results to {OUTDIR}/results.json")


if __name__ == "__main__":
    main()
