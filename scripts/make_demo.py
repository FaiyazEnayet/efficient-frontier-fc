"""Export small derived CSVs under demo/ for the dashboard.

Rebuilds ratings, the frontier on a fine budget grid, the selected squad
at each grid point, the best XI, and the backtest using the same functions
and seed as the pipeline, then writes demo/*.csv + demo/meta.json.

The goal models are retrained exactly as the pipeline trains them via
effc.pipeline.prepare_model_data (deterministic given the seed), so the
versioned VAEP cache hits and no event re-processing is needed. The demo
tables therefore always reflect the pipeline's deployed models.

Run from the repo root after the pipeline has been run at least once:

    python scripts/make_demo.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from effc.backtest.runner import run_backtest, split_matches  # noqa: E402
from effc.data.wages import load_wages, match_players  # noqa: E402
from effc.pipeline import compute_target_vaep, prepare_model_data  # noqa: E402
from effc.portfolio.portfolio import efficient_frontier, select_squad, select_xi  # noqa: E402
from effc.ratings.ratings import build_ratings  # noqa: E402

CONFIG = "configs/bundesliga_2seasons.yaml"
DEMO_DIR = "demo"
GRID_MIN, GRID_MAX, GRID_STEP = 300_000, 1_500_000, 50_000


def main() -> None:
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    # target-competition VAEP from the pipeline's deployed goal models.
    # Retraining is deterministic given the seed, so the versioned VAEP
    # cache hits and the demo tables always match the pipeline output.
    data = prepare_model_data(cfg)
    pm, mins, matches = compute_target_vaep(
        data["models"], data["train_ids"], config=cfg
    )

    wages = load_wages()
    sb_players = pm[["player", "team"]].drop_duplicates().reset_index(drop=True)
    wmap = match_players(sb_players, wages)
    ratings = build_ratings(
        pm, mins, wmap,
        min_minutes=cfg.get("min_minutes", 90),
        shrink_k=cfg.get("shrink_k", 900),
    )
    print(f"rated players: {len(ratings)}")

    os.makedirs(DEMO_DIR, exist_ok=True)

    ratings[["player_id", "player", "team", "pos_group", "vaep_per90", "wage_eur", "minutes"]].to_csv(
        os.path.join(DEMO_DIR, "players.csv"), index=False
    )

    grid = np.arange(GRID_MIN, GRID_MAX + 1, GRID_STEP, dtype=float)
    grid = np.array(sorted(set(grid.tolist()) | {943_000.0}))  # headline budget
    frontier = efficient_frontier(ratings, grid)
    frontier[["budget", "exp_vaep_per90", "wage_bill", "n_players"]].to_csv(
        os.path.join(DEMO_DIR, "frontier.csv"), index=False
    )
    print(f"frontier grid: {len(frontier)} budgets")

    squad_rows = []
    for b in frontier["budget"]:
        squad = select_squad(ratings, float(b))
        for _, r in squad.iterrows():
            squad_rows.append(
                {
                    "budget": b,
                    "player_id": r["player_id"],
                    "player": r["player"],
                    "team": r["team"],
                    "pos_group": r["pos_group"],
                    "vaep_per90": r["vaep_per90"],
                    "wage_eur": r["wage_eur"],
                }
            )
    pd.DataFrame(squad_rows).to_csv(os.path.join(DEMO_DIR, "squads.csv"), index=False)
    print(f"squads: {len(squad_rows)} rows")

    xi_budget = float(cfg.get("xi_budget_eur", 943_000))
    formation = cfg.get("formation", "4-3-3")
    xi_pool = select_squad(ratings, xi_budget)
    xi = select_xi(xi_pool, formation=formation)
    xi_out = xi[["player", "team", "pos_group", "vaep_per90", "wage_eur"]].copy()
    xi_out.to_csv(os.path.join(DEMO_DIR, "xi.csv"), index=False)
    print(f"best XI at {xi_budget:,.0f}: wage bill {xi_out['wage_eur'].sum():,.0f}")

    first, second = split_matches(matches, frac=cfg.get("backtest_split", 0.5))
    pm1 = pm[pm["match_id"].isin(first["match_id"])]
    pm2 = pm[pm["match_id"].isin(second["match_id"])]
    r1 = build_ratings(pm1, mins, wmap, min_minutes=60, shrink_k=cfg.get("shrink_k", 900))
    r2 = build_ratings(pm2, mins, wmap, min_minutes=60, shrink_k=cfg.get("shrink_k", 900))
    bt = run_backtest(r1, r2, pm2, mins, [300_000.0, 500_000.0, 700_000.0], seed=cfg.get("seed", 0))
    # melt only the rule score columns; the *_unavail availability counts
    # are pipeline diagnostics, not demo rules
    bt_long = bt.melt(
        id_vars="budget",
        value_vars=["optimizer", "value", "random_mean", "oracle"],
        var_name="rule",
        value_name="score",
    )
    bt_long.to_csv(os.path.join(DEMO_DIR, "backtest.csv"), index=False)
    print(bt.to_string(index=False))

    meta = {
        "config": CONFIG,
        "run_name": cfg.get("run_name"),
        "seed": cfg.get("seed", 0),
        "formation": formation,
        "xi_budget_eur": xi_budget,
        "n_rated": len(ratings),
        "wage_match_rate": float(wmap["matched"].mean()),
        "grid": {"min": GRID_MIN, "max": GRID_MAX, "step": GRID_STEP},
        "rules": ["optimizer", "value", "random_mean", "oracle"],
        "note": "Derived from the real pipeline output; regenerate with scripts/make_demo.py",
    }
    with open(os.path.join(DEMO_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {DEMO_DIR}/")


if __name__ == "__main__":
    main()
