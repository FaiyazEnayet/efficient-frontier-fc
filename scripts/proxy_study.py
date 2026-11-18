"""Measure the event-summary proxy against fitted VAEP on real data.

The proxy the live refresh folds into the ratings is computed from
StatsBomb 2015/16 match events and lineups for one league, matched to
that league's committed baseline ratings with the same matcher the
refresh uses, and compared as correlation, calibration slope and
intercept, and linear-rescaling error. Results land in
runs/live/proxy_study/results.json with a plain-language report in
report.md. Nothing is tuned here: the proxy weights are the ones the
refresh already uses, and the verdict is whatever the numbers say.

Data: StatsBomb open data (public, no credential), downloaded into
the repo data cache layout (data/matches_<comp>_<season>.json,
data/events/<id>.json, data/lineups/<id>.json). Files already present
are reused, never refetched.

Usage:
    PYTHONPATH=src python scripts/proxy_study.py [--league 39]
        [--competition 2] [--season 27] [--season-label 2015/2016]
        [--min-minutes 90]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402
import requests  # noqa: E402

from effc.data.download import BASE, DATA_DIR  # noqa: E402
from effc.data.statsbomb import event_content_digest  # noqa: E402
from effc.live import (  # noqa: E402
    LEAGUE_NAMES,
    baseline_path_for_league,
    load_baseline_artifact,
    load_baseline_ratings,
)
from effc.proxy_study import (  # noqa: E402
    accumulate_proxy,
    compare_proxy_to_baseline,
    load_statsbomb_json,
    statsbomb_match_proxy,
)

OUT_DIR = os.path.join("runs", "live", "proxy_study")


def _download(url: str, dest: str, session: requests.Session,
              retries: int = 3) -> None:
    if os.path.exists(dest):
        return
    last = None
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
            with open(dest, "w") as f:
                f.write(resp.text)
            return
        except requests.RequestException as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed after {retries} tries: {url}: {last}")


def _fetch_all(match_ids: list[int], data_dir: str) -> None:
    from concurrent.futures import ThreadPoolExecutor

    session = requests.Session()
    jobs = []
    for mid in match_ids:
        for kind in ("events", "lineups"):
            dest = os.path.join(data_dir, kind, f"{mid}.json")
            if not os.path.exists(dest):
                jobs.append((f"{BASE}/{kind}/{mid}.json", dest))
    os.makedirs(os.path.join(data_dir, "events"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "lineups"), exist_ok=True)
    print(f"downloading {len(jobs)} files ({len(match_ids)} matches)")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_download, url, dest, session)
                   for url, dest in jobs]
        for i, fut in enumerate(futures):
            fut.result()
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(futures)} downloaded")


def _fmt(value, places: int = 3) -> str:
    if value is None:
        return "undefined"
    return f"{value:.{places}f}"


def _verdict(stats: dict) -> str:
    """Plain-language verdict from the measured numbers.

    Calibrated would mean the proxy sits on the VAEP scale: slope
    near 1, intercept near 0, and a strong correlation. Anything else
    is stated as not calibrated, with the measured reason.
    """
    slope, intercept = stats.get("slope"), stats.get("intercept")
    pearson = stats.get("pearson")
    if slope is None or intercept is None or pearson is None:
        return ("not calibrated: the comparison is undefined on this "
                "sample (no variance in the proxy or too few players).")
    if 0.8 <= slope <= 1.25 and abs(intercept) <= 0.05 and pearson >= 0.7:
        return ("calibrated within the stated tolerances: slope "
                f"{slope:.2f}, intercept {intercept:.3f}, Pearson "
                f"r {pearson:.2f}.")
    reasons = []
    if not 0.8 <= slope <= 1.25:
        reasons.append(f"the calibration slope is {slope:.2f}, not near 1")
    if abs(intercept) > 0.05:
        reasons.append(f"the intercept is {intercept:.3f}, not near 0")
    if pearson < 0.7:
        reasons.append(f"the Pearson correlation is {pearson:.2f}, below 0.7")
    return "not calibrated: " + "; ".join(reasons) + "."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", type=int, default=39,
                    help="provider league id for the baseline (39=Premier League)")
    ap.add_argument("--competition", type=int, default=2,
                    help="StatsBomb competition id (2=English Premier League)")
    ap.add_argument("--season", type=int, default=27,
                    help="StatsBomb season id (27=2015/2016)")
    ap.add_argument("--season-label", default="2015/2016")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--min-minutes", type=float, default=90.0)
    args = ap.parse_args()

    league_name = LEAGUE_NAMES[args.league]
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    matches_path = os.path.join(
        data_dir, f"matches_{args.competition}_{args.season}.json")
    if not os.path.exists(matches_path):
        session = requests.Session()
        _download(f"{BASE}/matches/{args.competition}/{args.season}.json",
                  matches_path, session)
    with open(matches_path, "rb") as f:
        matches_sha256 = hashlib.sha256(f.read()).hexdigest()
    with open(matches_path) as f:
        matches = json.load(f)
    match_ids = sorted({int(m["match_id"]) for m in matches})
    print(f"{league_name} {args.season_label}: {len(match_ids)} matches")

    _fetch_all(match_ids, data_dir)

    frames = []
    for mid in match_ids:
        events = load_statsbomb_json(data_dir, "events", mid)
        lineups = load_statsbomb_json(data_dir, "lineups", mid)
        frames.append(statsbomb_match_proxy(events, lineups))
    proxy = accumulate_proxy(frames)
    print(f"proxy computed for {len(proxy)} players "
          f"({int((proxy['minutes'] >= args.min_minutes).sum())} with "
          f">= {args.min_minutes:.0f} minutes)")

    ratings = load_baseline_ratings(league=args.league)
    artifact = load_baseline_artifact(baseline_path_for_league(args.league))
    stats, matched = compare_proxy_to_baseline(
        proxy, ratings, min_minutes=args.min_minutes)
    verdict = _verdict(stats)
    print(f"matched {stats['n']} of {stats['n_baseline_rated']} rated players; "
          f"Pearson r {stats['pearson']}; slope {stats['slope']}; "
          f"MAE {stats['mae']}")
    print(f"verdict: {verdict}")

    os.makedirs(args.out_dir, exist_ok=True)
    results = {
        "study": "event-summary proxy vs fitted VAEP",
        "league": league_name,
        "league_id": args.league,
        "season": args.season_label,
        "competition_id": args.competition,
        "season_id": args.season,
        "matches": len(match_ids),
        "matches_sha256": matches_sha256,
        "events_digest": event_content_digest(match_ids, data_dir),
        "data_source": ("StatsBomb open data, "
                        f"{BASE}/matches/{args.competition}/{args.season}.json "
                        "and per-match events/lineups"),
        "baseline": {
            "league_id": args.league,
            "artifact": os.path.basename(baseline_path_for_league(args.league)),
            "ratings_sha256": artifact["ratings_sha256"],
            "model": artifact.get("model"),
            "data_snapshot": artifact.get("data_snapshot"),
            "feature_version": artifact.get("feature_version"),
            "seed": artifact.get("seed"),
            "model_hash": artifact.get("model_hash"),
        },
        "comparison": stats,
        "verdict": verdict,
        "matched_players": matched.to_dict("records"),
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=1)

    top = matched.sort_values("proxy_per90", ascending=False).head(10)
    lines = [
        "# Event-summary proxy vs fitted VAEP",
        "",
        f"League: {league_name}, season {args.season_label} "
        f"({len(match_ids)} matches, StatsBomb open data).",
        f"Baseline: {artifact['data_snapshot']}",
        "",
        "## What was measured",
        "",
        "The event-summary proxy is the live refresh's proxy: goals "
        "0.9 (penalties 0.7), assists 0.4, yellow cards -0.05, red cards "
        "-0.30, per 90 minutes, with minutes from lineups and "
        "substitution spans. It was computed from the same season's "
        "match events and lineups, and each player was matched to the "
        "committed baseline ratings with the same team-constrained "
        "matcher the refresh uses. Only players with at least "
        f"{args.min_minutes:.0f} proxy minutes are compared.",
        "",
        "## Result",
        "",
        f"- Players compared: {stats['n']} of "
        f"{stats['n_baseline_rated']} rated baseline players "
        f"({stats['n_proxy_players']} players had any proxy minutes).",
        f"- Mean proxy per 90: {_fmt(stats['mean_proxy_per90'])}; "
        f"mean fitted VAEP per 90: {_fmt(stats['mean_vaep_per90'])}.",
        f"- Pearson r: {_fmt(stats['pearson'])}; "
        f"Spearman rho: {_fmt(stats['spearman'])}.",
        f"- Calibration of VAEP on the proxy: slope "
        f"{_fmt(stats['slope'])}, intercept {_fmt(stats['intercept'])}.",
        f"- Mean absolute error of the best linear rescaling: "
        f"{_fmt(stats['mae'])} VAEP per 90.",
        "",
        f"Verdict: the proxy is {verdict}",
        "",
        "## What this means for the live refresh",
        "",
        "The blended rating the refresh produces is a refresh "
        "mechanism, not a measured improvement, until this study says "
        "otherwise. The proxy and VAEP move together only as far as "
        "the numbers above show; the blend weights and the shrinkage "
        "were not tuned to this study, and the study was not used to "
        "tune them.",
        "",
        "## Highest proxy per 90 (compared players)",
        "",
    ]
    for r in top.itertuples():
        lines.append(f"- {r.player} ({r.team}): proxy {r.proxy_per90:.3f}, "
                     f"VAEP {r.vaep_per90:.3f}, {r.minutes:.0f} minutes")
    lines.append("")
    lines.append("Full per-player table: results.json (matched_players).")
    lines.append("")
    with open(os.path.join(args.out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {os.path.join(args.out_dir, 'results.json')}")
    print(f"wrote {os.path.join(args.out_dir, 'report.md')}")


if __name__ == "__main__":
    main()
