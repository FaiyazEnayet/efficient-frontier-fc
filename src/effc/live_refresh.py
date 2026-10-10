"""Daily live refresh: pull new fixtures, update ratings, rebalance the squad.

    python -m effc.live_refresh [--league 78] [--season 2026] [--last 10]
                                [--budget 700000] [--dry-run]

Rebuilds the base ratings from the cached pipeline artifacts (no model
retraining), fetches finished fixtures from API-Football, folds event
proxies into the ratings, re-runs the squad optimizer, and prints a
rebalance report. State lives in data/live/state.json.
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from effc.data.wages import load_wages, match_players
from effc.live import (
    ApiError,
    blend_ratings,
    fetch_fixture_events,
    fetch_fixture_lineups,
    fetch_finished_fixtures,
    fixture_contributions,
    load_state,
    match_player,
    rebalance_report,
    save_state,
    utc_today,
)
from effc.pipeline import compute_target_vaep
from effc.portfolio.portfolio import select_squad
from effc.ratings.ratings import build_ratings


def base_ratings(min_minutes: float = 270.0, shrink_k: float = 900.0) -> pd.DataFrame:
    pm, mins, _ = compute_target_vaep()
    wages = load_wages()
    sb_players = pm[["player", "team"]].drop_duplicates().reset_index(drop=True)
    wmap = match_players(sb_players, wages)
    return build_ratings(pm, mins, wmap, min_minutes=min_minutes, shrink_k=shrink_k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", type=int, default=78, help="API-Football league id (78=Bundesliga)")
    ap.add_argument("--season", type=int, default=2024,
                    help="latest season covered by the free tier (2024 = 2024/25)")
    ap.add_argument("--days", type=int, default=2, help="days back to scan (free tier: max ~2)")
    ap.add_argument("--budget", type=float, default=700_000.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    state = load_state()
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

    new = sorted(
        [f for f in fixtures if f["fixture"]["id"] > state.get("last_fixture_id", 0)],
        key=lambda f: f["fixture"]["id"],
    )
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

    ratings = base_ratings()
    aliases = state.setdefault("aliases", {})
    live_minutes = state.setdefault("live_minutes", {})
    live_contrib = state.setdefault("live_contrib", {})
    watchlist = state.setdefault("watchlist", {})
    matched, unmatched = 0, 0

    for f in new:
        fid = f["fixture"]["id"]
        try:
            events = counted(fetch_fixture_events, fid)
            lineups = counted(fetch_fixture_lineups, fid)
        except ApiError as exc:
            print(f"  fixture {fid}: skipped ({exc})")
            continue
        contrib = fixture_contributions(events, lineups)
        for _, row in contrib.iterrows():
            pid = match_player(row["api_player"], row["api_team"], ratings, aliases)
            if pid is None:
                key = f"{row['api_team']} || {row['api_player']}"
                watchlist[key] = watchlist.get(key, 0.0) + row["contrib"]
                unmatched += 1
                continue
            live_minutes[pid] = live_minutes.get(pid, 0.0) + float(row["minutes"])
            live_contrib[pid] = live_contrib.get(pid, 0.0) + float(row["contrib"])
            matched += 1
        state["last_fixture_id"] = max(state.get("last_fixture_id", 0), fid)

    state["calls"] = calls
    state.setdefault("history", []).append(
        {"date": today, "new_fixtures": len(new), "matched": matched,
         "unmatched": unmatched, "calls": calls["n"]}
    )
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
