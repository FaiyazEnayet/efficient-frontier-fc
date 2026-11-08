"""Daily live refresh: pull new fixtures, update ratings, rebalance the squad.

    python -m effc.live_refresh [--league 78] [--season 2026]
                                [--budget 700000] [--dry-run] [--rebuild-base]

Base ratings come from the immutable, versioned baseline artifact in
runs/live/ (tied to one fitted model and data snapshot; see
effc.live.load_baseline_ratings). With --rebuild-base they are instead
rebuilt from the pipeline: the fitted models and training match IDs
from prepare_model_data are passed to compute_target_vaep, as its
signature requires. Fetched fixtures are folded into the ratings as
an event-summary proxy (not VAEP), the squad optimizer is re-run, and
a rebalance report is printed. State lives in data/live/state.json
and records every processed fixture in a processing ledger, so a
replay changes nothing and a fixture discovered out of order is
never skipped.
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import pandas as pd

from effc.data.wages import load_wages, match_players
from effc.live import (
    ApiError,
    blend_ratings,
    canonical_id,
    fetch_fixture_events,
    fetch_fixture_lineups,
    fetch_finished_fixtures,
    fixture_contributions,
    load_baseline_ratings,
    load_state,
    match_player,
    normalize_state,
    processed_fixture_ids,
    rebalance_report,
    save_state,
    unprocessed_fixtures,
    utc_today,
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
                 rebuild: bool = False, config: dict | None = None) -> pd.DataFrame:
    """Base ratings for the refresh.

    Default: the versioned baseline artifact (immutable, tied to a
    fitted model and data snapshot). With rebuild=True, ratings are
    rebuilt through the pipeline, passing the fitted models and the
    training match IDs to compute_target_vaep.
    """
    if not rebuild:
        return load_baseline_ratings()
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
    instead of counting the fixture as zero production.
    """
    fid = canonical_id(fixture["fixture"]["id"])
    entry = {
        "fixture_id": fid,
        "source_timestamp": fixture.get("fixture", {}).get("date"),
        "processed_at": utc_today(),
        "status": "ok",
        "matched": 0,
        "unmatched": 0,
        "errors": [],
    }
    try:
        events = counted(fetch_fixture_events, fixture["fixture"]["id"])
        lineups = counted(fetch_fixture_lineups, fixture["fixture"]["id"])
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

    staged_minutes = dict(state["live_minutes"])
    staged_contrib = dict(state["live_contrib"])
    staged_last_seen = dict(state["live_last_seen"])
    staged_aliases = dict(state["aliases"])
    staged_watchlist = dict(state["watchlist"])
    contrib = fixture_contributions(events, lineups)
    source_day = str(entry["source_timestamp"] or utc_today())[:10]
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
        entry["matched"] += 1

    # commit: the fixture is fully processed, so its staged totals and
    # its ledger entry land together
    state["live_minutes"] = staged_minutes
    state["live_contrib"] = staged_contrib
    state["live_last_seen"] = staged_last_seen
    state["aliases"] = staged_aliases
    state["watchlist"] = staged_watchlist
    return entry


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
    today = utc_today()
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

    ratings = base_ratings(rebuild=args.rebuild_base)
    working = copy.deepcopy(state)
    working.setdefault("processed_fixtures", {})
    matched, unmatched = 0, 0

    for f in new:
        entry = _process_fixture(working, f, ratings, counted)
        working["processed_fixtures"][entry["fixture_id"]] = entry
        matched += entry["matched"]
        unmatched += entry["unmatched"]
        if entry["status"] == "error":
            print(f"  fixture {entry['fixture_id']}: skipped ({entry['errors'][0]})")
        elif entry["status"] == "missing_statistics":
            print(f"  fixture {entry['fixture_id']}: statistics missing, recorded")

    # the legacy watermark is kept as information only; the ledger is
    # the source of truth for what has been processed
    done = processed_fixture_ids(working)
    if done:
        working["last_fixture_id"] = max(int(i) for i in done)

    working["calls"] = calls
    if new:
        working.setdefault("history", []).append(
            {"date": today, "new_fixtures": len(new), "matched": matched,
             "unmatched": unmatched, "calls": calls["n"]}
        )
    state = normalize_state(working)
    save_state(state)
    print(f"updated {matched} player-fixtures, {unmatched} unmatched (watchlist)")

    live = blend_ratings(ratings, state)
    old_squad = select_squad(ratings, args.budget)
    live_for_opt = live.copy()
    live_for_opt["vaep_per90"] = live_for_opt["vaep_per90_live"]
    new_squad = select_squad(live_for_opt, args.budget)

    report = rebalance_report(old_squad, new_squad, live)
    print(f"\nrebalance at EUR {args.budget:,.0f}/week "
          f"(exp VAEP/90 {old_squad['vaep_per90'].sum():.2f} -> "
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
                   "old_exp_vaep": float(old_squad["vaep_per90"].sum()),
                   "new_exp_vaep": float(new_squad["vaep_per90_live"].sum()),
                   "report": report.to_dict("records"),
                   "watchlist_top": sorted(watchlist.items(), key=lambda kv: -kv[1])[:10]},
                  fh, indent=1)
    print(f"\nsaved {outp}")


if __name__ == "__main__":
    main()
