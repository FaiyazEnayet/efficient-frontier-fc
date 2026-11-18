"""Daily live refresh: pull new fixtures, update ratings, rebalance the squad.

    python -m effc.live_refresh [--league 78] [--season 2026]
                                [--budget 700000] [--dry-run] [--rebuild-base]

Base ratings come from the immutable, versioned baseline artifact in
runs/live/ for the requested league (tied to one fitted model and
data snapshot; see effc.live.load_baseline_ratings). A league with no
baseline artifact falls back to the default artifact, so its players
are watchlisted. With --rebuild-base they are instead
rebuilt from the pipeline: the fitted models and training match IDs
from prepare_model_data are passed to compute_target_vaep, as its
signature requires. Before any quota is spent, the refresh verifies
the baseline bundle manifest (runs/live/baseline_bundle_v1.json),
which ties all five league artifacts together with file and ratings
hashes. Fetched fixtures are folded into the ratings as
an event-summary proxy (not VAEP), the squad optimizer is re-run, and
a rebalance report is printed. State lives in data/live/state.json
and records every processed fixture in a processing ledger, so a
replay changes nothing and a fixture discovered out of order is
never skipped. Each refresh writes a reproducibility record and a
complete fixture and player audit into its runs/live/refresh_*.json
output: baseline identity and hashes, state digests before and after,
blend parameters, fixture payload hashes, per-fixture outcomes, and
old and new squad feasibility. The new squad's figure is a blended
event-proxy value, not VAEP, and is named as such.
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import pandas as pd

from effc.data.wages import load_wages, match_players
from effc.leagues import squad_constraints_ok
from effc.live import (
    ApiError,
    BASELINE_PATH,
    BUNDLE_PATH,
    EVENT_PROXY_HALF_LIFE_DAYS,
    EVENT_PROXY_HALF_LIFE_MIN,
    EVENT_PROXY_W_ASSIST,
    EVENT_PROXY_W_GOAL,
    EVENT_PROXY_W_PENALTY_GOAL,
    EVENT_PROXY_W_RED,
    EVENT_PROXY_W_YELLOW,
    blend_ratings,
    canonical_id,
    canonical_json_sha256,
    fetch_fixture_events,
    fetch_fixture_lineups,
    fetch_finished_fixtures,
    file_sha256,
    fixture_contributions,
    load_baseline_artifact,
    load_baseline_bundle,
    load_baseline_ratings,
    load_state,
    match_player,
    normalize_state,
    processed_fixture_ids,
    rebalance_report,
    save_state,
    state_digest,
    unprocessed_fixtures,
    utc_now_iso,
    utc_today,
    verify_state,
)
from effc.pipeline import compute_target_vaep, prepare_model_data
from effc.portfolio.portfolio import select_squad
from effc.ratings.ratings import build_ratings

DEFAULT_CONFIG = os.path.join("configs", "bundesliga_2seasons.yaml")


def _load_config(path: str = DEFAULT_CONFIG) -> dict:
    if os.path.exists(path):
        import yaml

        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def base_ratings(min_minutes: float = 270.0, shrink_k: float = 900.0,
                 rebuild: bool = False, config: dict | None = None,
                 league: int | None = None) -> pd.DataFrame:
    """Base ratings for the refresh.

    Default: the versioned baseline artifact for `league` (immutable,
    tied to a fitted model and data snapshot); a league with no
    baseline artifact falls back to the default artifact, so its
    players are watchlisted. With rebuild=True, ratings are rebuilt
    through the pipeline (the Bundesliga config), passing the fitted
    models and the training match IDs to compute_target_vaep.
    """
    if not rebuild:
        return load_baseline_ratings(league=league)
    cfg = config if config is not None else _load_config()
    data = prepare_model_data(cfg)
    pm, mins, _ = compute_target_vaep(
        data["models"], data["train_ids"], config=cfg)
    wages = load_wages()
    sb_players = pm[["player", "team"]].drop_duplicates().reset_index(drop=True)
    wmap = match_players(sb_players, wages)
    return build_ratings(pm, mins, wmap, min_minutes=min_minutes, shrink_k=shrink_k)


def _process_fixture(state: dict, fixture: dict, ratings: pd.DataFrame,
                     counted) -> dict:
    """Stage one fixture's contributions and commit them atomically.

    Returns the ledger entry for the fixture. Contributions are staged
    in copies of the live totals and only written back when both the
    events and the lineups arrive; on failure or missing statistics
    the live totals are untouched and the ledger records the outcome
    instead of counting the fixture as zero production. The entry
    records full availability timestamps (when the events and the
    lineups were fetched, and when the fixture was processed) and the
    latency from the fixture date to the events fetch, plus the
    per-player contributions for this fixture with their hash, so the
    live totals reconcile back to the fixtures that produced them.
    """
    fid = canonical_id(fixture["fixture"]["id"])
    entry = {
        "fixture_id": fid,
        "source_timestamp": fixture.get("fixture", {}).get("date"),
        "processed_at": utc_now_iso(),
        "status": "ok",
        "matched": 0,
        "unmatched": 0,
        "errors": [],
        "events_fetched_at": None,
        "lineups_fetched_at": None,
        "availability_latency_hours": None,
        "contributions": [],
        "contributions_sha256": canonical_json_sha256([]),
    }
    try:
        events = counted(fetch_fixture_events, fixture["fixture"]["id"])
        entry["events_fetched_at"] = utc_now_iso()
        lineups = counted(fetch_fixture_lineups, fixture["fixture"]["id"])
        entry["lineups_fetched_at"] = utc_now_iso()
    except ApiError as exc:
        entry["status"] = "error"
        entry["errors"] = [str(exc)]
        return entry
    if not events or not lineups:
        entry["status"] = "missing_statistics"
        entry["errors"] = [
            "empty events or lineups response; statistics missing, "
            "not zero production"]
        return entry

    source = entry["source_timestamp"]
    fetched = entry["events_fetched_at"]
    if source and fetched:
        try:
            delta = (pd.Timestamp(fetched) - pd.Timestamp(source)
                     ).total_seconds() / 3600.0
            entry["availability_latency_hours"] = round(float(delta), 3)
        except (ValueError, TypeError):
            entry["availability_latency_hours"] = None

    staged_minutes = dict(state["live_minutes"])
    staged_contrib = dict(state["live_contrib"])
    staged_last_seen = dict(state["live_last_seen"])
    staged_aliases = dict(state["aliases"])
    staged_watchlist = dict(state["watchlist"])
    contrib = fixture_contributions(events, lineups)
    source_day = str(entry["source_timestamp"] or utc_today())[:10]
    contributions = []
    for _, row in contrib.iterrows():
        if row["stats_status"] == "unused":
            continue
        pid = match_player(row["api_player"], row["api_team"], ratings, staged_aliases)
        if pid is None:
            key = f"{row['api_team']} || {row['api_player']}"
            staged_watchlist[key] = staged_watchlist.get(key, 0.0) + float(row["contrib"])
            entry["unmatched"] += 1
            continue
        staged_minutes[pid] = staged_minutes.get(pid, 0.0) + float(row["minutes"])
        staged_contrib[pid] = staged_contrib.get(pid, 0.0) + float(row["contrib"])
        if source_day > staged_last_seen.get(pid, ""):
            staged_last_seen[pid] = source_day
        contributions.append({"player_id": pid,
                              "minutes": float(row["minutes"]),
                              "contrib": float(row["contrib"])})
        entry["matched"] += 1
    entry["contributions"] = contributions
    entry["contributions_sha256"] = canonical_json_sha256(contributions)

    # commit: the fixture is fully processed, so its staged totals and
    # its ledger entry land together
    state["live_minutes"] = staged_minutes
    state["live_contrib"] = staged_contrib
    state["live_last_seen"] = staged_last_seen
    state["aliases"] = staged_aliases
    state["watchlist"] = staged_watchlist
    entry["processed_at"] = utc_now_iso()
    return entry


def _baseline_info(league: int | None, rebuild: bool) -> dict:
    """Baseline identity for the reproducibility record.

    For the artifact path this is the loaded artifact's own metadata
    plus its file and ratings hashes; for a pipeline rebuild there is
    no artifact, so the record says so and carries the config hash
    when the config file exists.
    """
    if rebuild:
        info: dict = {"source": "pipeline-rebuild", "league_id": league}
        if os.path.exists(DEFAULT_CONFIG):
            info["config_sha256"] = file_sha256(DEFAULT_CONFIG)
        return info
    from effc.live import baseline_path_for_league

    path = baseline_path_for_league(league) or BASELINE_PATH
    artifact = load_baseline_artifact(league=league)
    return {
        "source": "baseline-artifact",
        "league_id": league,
        "league": artifact.get("league"),
        "artifact": os.path.basename(path),
        "version": artifact.get("version"),
        "ratings_sha256": artifact.get("ratings_sha256"),
        "file_sha256": file_sha256(path),
        "model": artifact.get("model"),
        "data_snapshot": artifact.get("data_snapshot"),
        "config_hash": artifact.get("config_hash"),
        "feature_version": artifact.get("feature_version"),
        "seed": artifact.get("seed"),
        "model_hash": artifact.get("model_hash"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", type=int, default=78, help="API-Football league id (78=Bundesliga)")
    ap.add_argument("--season", type=int, default=2024,
                    help="season label recorded with the run (fixtures are fetched by date)")
    ap.add_argument("--days", type=int, default=2, help="days back to scan (free tier: max ~2)")
    ap.add_argument("--budget", type=float, default=700_000.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rebuild-base", action="store_true",
                    help="rebuild base ratings through the pipeline instead of "
                         "loading the versioned baseline artifact")
    args = ap.parse_args()

    state = normalize_state(load_state())
    state_before_digest = state_digest(state)
    today = utc_today()

    # the baseline bundle is verified before any quota is spent: a
    # tampered artifact or manifest fails the run here, not after the
    # fetch. A missing manifest is recorded, not fatal: the artifact
    # a run actually loads is still hash-verified on load.
    bundle_status = "absent"
    if os.path.exists(BUNDLE_PATH):
        try:
            load_baseline_bundle()
            bundle_status = "verified"
        except ValueError as exc:
            print(f"baseline bundle verification failed: {exc}")
            raise SystemExit(1)

    calls = state.get("calls", {})
    if calls.get("date") != today:
        calls = {"date": today, "n": 0}

    def counted(fn, *a, **k):
        calls["n"] += 1
        return fn(*a, **k)

    try:
        fixtures = counted(fetch_finished_fixtures, args.league, args.season, days=args.days)
    except ApiError as exc:
        print(f"live refresh failed: {exc}")
        raise SystemExit(1)

    new = unprocessed_fixtures(fixtures, state)
    print(f"{len(fixtures)} recent finished fixtures, {len(new)} new since last refresh")

    if args.dry_run:
        for f in new:
            fx = f["fixture"]
            print(f"  {fx['date'][:10]} {f['teams']['home']['name']} {f['goals']['home']}-"
                  f"{f['goals']['away']} {f['teams']['away']['name']}")
        return

    # quota guard: 2 calls per new fixture (events + lineups), stay under 95/day
    need = 2 * len(new)
    if calls["n"] + need > 95:
        print(f"quota guard: need ~{need} calls but only {100 - calls['n']} left today; "
              f"run again tomorrow")
        return

    ratings = base_ratings(rebuild=args.rebuild_base, league=args.league)
    baseline_info = _baseline_info(args.league, args.rebuild_base)
    working = copy.deepcopy(state)
    working.setdefault("processed_fixtures", {})
    matched, unmatched = 0, 0
    n_ok = n_error = n_missing = 0
    audit_fixtures = []

    for f in new:
        entry = _process_fixture(working, f, ratings, counted)
        working["processed_fixtures"][entry["fixture_id"]] = entry
        matched += entry["matched"]
        unmatched += entry["unmatched"]
        if entry["status"] == "ok":
            n_ok += 1
        elif entry["status"] == "error":
            n_error += 1
            print(f"  fixture {entry['fixture_id']}: skipped ({entry['errors'][0]})")
        elif entry["status"] == "missing_statistics":
            n_missing += 1
            print(f"  fixture {entry['fixture_id']}: statistics missing, recorded")
        audit_fixtures.append({
            "fixture_id": entry["fixture_id"],
            "source_timestamp": entry["source_timestamp"],
            "status": entry["status"],
            "matched": entry["matched"],
            "unmatched": entry["unmatched"],
            "fixture_sha256": canonical_json_sha256(f),
            "events_fetched_at": entry["events_fetched_at"],
            "lineups_fetched_at": entry["lineups_fetched_at"],
            "availability_latency_hours": entry["availability_latency_hours"],
            "errors": entry["errors"],
        })

    # the legacy watermark is kept as information only; the ledger is
    # the source of truth for what has been processed
    done = processed_fixture_ids(working)
    if done:
        working["last_fixture_id"] = max(int(i) for i in done)

    working["calls"] = calls
    if new:
        working.setdefault("history", []).append(
            {"date": today, "new_fixtures": len(new), "matched": matched,
             "unmatched": unmatched, "calls": calls["n"],
             "baseline_sha256": baseline_info.get("ratings_sha256")}
        )
    state = normalize_state(working)
    state_after_digest = state_digest(state)
    save_state(state)
    # verify the state as it was actually saved: reload it (which
    # re-checks the digest) and reconcile the totals to the ledger
    verification = verify_state(load_state())
    if not verification["ok"]:
        print("state verification failed after save:")
        for problem in verification["errors"]:
            print(f"  {problem}")
    for warning in verification["warnings"]:
        print(f"  state note: {warning}")
    print(f"updated {matched} player-fixtures, {unmatched} unmatched (watchlist)")

    live = blend_ratings(ratings, state)
    old_squad = select_squad(ratings, args.budget)
    live_for_opt = live.copy()
    live_for_opt["vaep_per90"] = live_for_opt["vaep_per90_live"]
    new_squad = select_squad(live_for_opt, args.budget)

    report = rebalance_report(old_squad, new_squad, live)
    print(f"\nrebalance at EUR {args.budget:,.0f}/week "
          f"(exp VAEP/90 {old_squad['vaep_per90'].sum():.2f} -> "
          f"exp blended event-proxy value/90 "
          f"{new_squad['vaep_per90_live'].sum():.2f}):")
    if report.empty:
        print("  no changes: the live data did not move the optimal squad")
    else:
        for _, r in report.iterrows():
            print(f"  {r['action']:3s} {r['player']} ({r['team']}) "
                  f"VAEP/90 {r['vaep_per90']:.3f}  wage EUR {r['wage_eur']:,.0f}/wk")

    outdir = os.path.join("runs", "live")
    os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, f"refresh_{today}.json")
    watchlist = state.get("watchlist", {})
    with open(outp, "w") as fh:
        json.dump({"date": today, "league": args.league, "season": args.season,
                   "budget": args.budget,
                   "old_exp_vaep_per90": float(old_squad["vaep_per90"].sum()),
                   "new_exp_blended_per90": float(new_squad["vaep_per90_live"].sum()),
                   "report": report.to_dict("records"),
                   "watchlist_top": sorted(watchlist.items(), key=lambda kv: -kv[1])[:10],
                   "reproducibility": {
                       "baseline": baseline_info,
                       "baseline_bundle": bundle_status,
                       "state_sha256_before": state_before_digest,
                       "state_sha256_after": state_after_digest,
                       "blend": {
                           "shrink_k": 900.0,
                           "event_proxy_half_life_min": EVENT_PROXY_HALF_LIFE_MIN,
                           "event_proxy_half_life_days": EVENT_PROXY_HALF_LIFE_DAYS,
                           "weights": {
                               "goal": EVENT_PROXY_W_GOAL,
                               "penalty_goal": EVENT_PROXY_W_PENALTY_GOAL,
                               "assist": EVENT_PROXY_W_ASSIST,
                               "yellow": EVENT_PROXY_W_YELLOW,
                               "red": EVENT_PROXY_W_RED,
                           },
                       },
                       "days": args.days,
                       "dry_run": args.dry_run,
                       "fixture_sha256": {
                           str(f["fixture"]["id"]): canonical_json_sha256(f)
                           for f in fixtures},
                   },
                   "audit": {
                       "fixtures_seen": len(fixtures),
                       "fixtures_new": len(new),
                       "fixtures_already_processed": len(fixtures) - len(new),
                       "fixtures_processed_ok": n_ok,
                       "fixtures_error": n_error,
                       "fixtures_missing_statistics": n_missing,
                       "player_fixtures_matched": matched,
                       "player_fixtures_unmatched": unmatched,
                       "watchlist_size": len(watchlist),
                       "state_verification": verification,
                       "squads": {
                           "old_feasible": bool(squad_constraints_ok(
                               old_squad, args.budget)),
                           "new_feasible": bool(squad_constraints_ok(
                               new_squad, args.budget)),
                       },
                       "fixtures": audit_fixtures,
                   }},
                  fh, indent=1)
    print(f"\nsaved {outp}")
    if not verification["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
