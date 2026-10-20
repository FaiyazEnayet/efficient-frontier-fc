"""Beat-baseline experiment driver (frozen FC6 scoreboard).

Selects a goal-model configuration on internal validation, cross-fits
Platt calibration + base-rate blending on training-group matches only,
retrains on the full training pool, and scores the FC6 test sets
exactly once:

- held-out: the same 32 match IDs FC6 held out (train group: the same
  130), baseline = base rate of the 130 training matches;
- season transfer: train 2015/2016 (17 matches + contemporaneous
  2015/2016 leagues), score the same 17 2023/2024 matches FC6 scored.

Leakage guards are asserted by match ID before any scoring: pooled
matches are disjoint from both test sets and from the held-out group,
and no 2023/2024 Bundesliga match appears in any training pool.

Writes runs/beat_baseline/results.json. The deployed pipeline
(effc.pipeline) uses the same recipe via effc.vaep.improved.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from effc.backtest.runner import split_matches  # noqa: E402
from effc.data.download import MODEL_POOL, TARGET  # noqa: E402
from effc.data.statsbomb import list_matches  # noqa: E402
from effc.pipeline import (  # noqa: E402
    era2015_pool,
    extra_pool,
    featurize_pool,
    pre_split_frames,
)
from effc.vaep import calibration as cal  # noqa: E402
from effc.vaep import improved as imp  # noqa: E402


def fc6_split(cfg):
    """Reproduce prepare_model_data's split steps (no model training)."""
    target_matches = list_matches(TARGET)
    _first, second = split_matches(
        target_matches, frac=cfg.get("backtest_split", 0.5)
    )
    post_split_ids = set(second["match_id"].astype(int))
    frames = pre_split_frames(featurize_pool(), post_split_ids)
    meta = list_matches(MODEL_POOL)
    train_frames, calib_frames = cal.match_group_split(
        frames, meta, calib_frac=cfg.get("calib_frac", 0.2)
    )
    return {
        "frames": frames,
        "meta": meta,
        "target_matches": target_matches,
        "train_frames": train_frames,
        "calib_frames": calib_frames,
        "train_ids": sorted({int(f["match_id"].iloc[0]) for f in train_frames}),
        "calib_ids": sorted({int(f["match_id"].iloc[0]) for f in calib_frames}),
    }

LABELS = ("scores_incl", "concedes_incl")


def frame_ids(frames):
    return sorted({int(f["match_id"].iloc[0]) for f in frames})


def score_block(models, train_frames, test_frames):
    """Model vs FC6-definition baseline on a test set (called once)."""
    held_model = cal.per_match_scores(models, test_frames)
    held_base = cal.base_rate_scores(train_frames, test_frames)
    ms, bs = cal.summarize(held_model), cal.summarize(held_base)
    return {
        label: {
            "model": ms[label]["brier"],
            "base_rate_baseline": bs[label]["brier"],
            "model_logloss": ms[label]["logloss"],
            "baseline_logloss": bs[label]["logloss"],
        }
        for label in cal.LABELS
    }


RESULTS_PATH = "runs/beat_baseline/results.json"


def load_results() -> dict:
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_results(results: dict) -> None:
    os.makedirs("runs/beat_baseline", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)


def main() -> None:
    t0 = time.time()
    with open("configs/bundesliga_2seasons.yaml") as f:
        cfg = yaml.safe_load(f)
    seed = cfg.get("seed", 0)

    data = fc6_split(cfg)
    train_frames, calib_frames = data["train_frames"], data["calib_frames"]
    train_ids, calib_ids = set(data["train_ids"]), set(data["calib_ids"])
    meta_pool = data["meta"]

    extra_frames, extra_meta = extra_pool()
    era_frames, era_meta = era2015_pool()
    meta_all = (
        pd.concat([meta_pool, extra_meta], ignore_index=True)
        .drop_duplicates("match_id")
        .reset_index(drop=True)
    )
    extra_ids = set(frame_ids(extra_frames))
    era_ids = set(frame_ids(era_frames))

    # ---- leakage guards (by match ID, before anything is scored) ----
    target_meta = meta_pool[meta_pool["competition_id"] == 9]
    ids_2024 = set(
        target_meta.loc[target_meta["season_id"] == 281, "match_id"].astype(int)
    )
    assert not (extra_ids & calib_ids), "extra pool leaks held-out matches"
    assert not (era_ids & calib_ids), "era pool leaks held-out matches"
    assert not (extra_ids & ids_2024), "extra pool contains 2023/24 Bundesliga"
    assert not (era_ids & ids_2024), "era pool contains 2023/24 Bundesliga"
    assert not (train_ids & calib_ids)
    print(
        f"pools: train group {len(train_ids)}, held-out {len(calib_ids)}, "
        f"extra {len(extra_ids)}, era2015 {len(era_ids)}; guards passed"
    )

    results: dict = load_results()
    results["pool_counts"] = {
        "train_group": len(train_ids),
        "held_out": len(calib_ids),
        "extra_pool_matches": len(extra_ids),
        "era2015_pool_matches": len(era_ids),
    }
    save_results(results)

    # ================= main arm =================
    # Selection mirrors imp.select_config exactly (same split, same
    # pool cap) but saves each config's scores as it finishes, so an
    # interrupted run resumes mid-selection.
    sel = results.setdefault("main_selection", {"scores": {}})
    if "selected" in sel:
        best_main = sel["selected"]
        print(f"\n== main arm: selection already done -> {best_main} ==")
    else:
        print("\n== main arm: config selection on internal validation ==")
        sub_train, val = cal.match_group_split(
            train_frames, meta_all, calib_frac=0.25
        )
        match_dates = dict(
            zip(
                meta_all["match_id"].astype(int),
                meta_all["match_date"].astype(str),
            )
        )
        pool_capped = imp.subsample_frames(
            extra_frames, imp.SELECTION_POOL_CAP
        )
        for name, config in imp.CONFIGS.items():
            if name in sel["scores"]:
                print(f"  {name}: (cached)")
                continue
            bundle = imp.train_bundle(
                pool_capped + sub_train, meta_all, config, seed=seed,
                match_dates=match_dates, labels=LABELS,
            )
            per_label = {}
            for label in LABELS:
                ys, ps = [], []
                for f in val:
                    ys.append(f[label].to_numpy())
                    ps.append(imp.raw_predict(bundle, f)[label])
                from effc.vaep.models import brier_score

                per_label[label] = brier_score(
                    np.concatenate(ys), np.concatenate(ps)
                )
            sel["scores"][name] = per_label
            save_results(results)
            print(
                f"  {name}: "
                + ", ".join(f"{k}={v:.5f}" for k, v in per_label.items())
            )
            del bundle
        best_main = min(
            sel["scores"],
            key=lambda n: sel["scores"][n]["scores_incl"]
            + sel["scores"][n]["concedes_incl"],
        )
        sel["selected"] = best_main
        save_results(results)
        print(f"  -> selected {best_main}")

    if "held_out" in results:
        print("== main arm: held-out already scored ==")
    else:
        print("== main arm: cross-fit calibration + final training ==")
        models_main, calib_rep = imp.train_improved_models_cached(
            train_frames, extra_frames, meta_all, best_main, seed=seed
        )
        results["main_calibration"] = calib_rep
        save_results(results)

        print("== main arm: scoring held-out ONCE ==")
        results["held_out"] = score_block(models_main, train_frames, calib_frames)
        save_results(results)
        for label in LABELS:
            r = results["held_out"][label]
            print(
                f"  {label}: model {r['model']['mean']:.5f} vs baseline "
                f"{r['base_rate_baseline']['mean']:.5f}"
            )

    # ================= transfer arm =================
    print("\n== transfer arm: config selection by cross-fitted OOF ==")
    season_of = dict(
        zip(meta_pool["match_id"].astype(int), meta_pool["season_id"].astype(int))
    )
    comp_of = dict(
        zip(
            meta_pool["match_id"].astype(int),
            meta_pool["competition_id"].astype(int),
        )
    )
    tr_frames = [
        f for f in data["frames"]
        if season_of.get(int(f["match_id"].iloc[0])) == 27
        and comp_of.get(int(f["match_id"].iloc[0])) == 9
    ]
    sc_frames = [
        f for f in data["frames"]
        if season_of.get(int(f["match_id"].iloc[0])) == 281
        and comp_of.get(int(f["match_id"].iloc[0])) == 9
    ]
    tr_ids = set(frame_ids(tr_frames))
    sc_ids = set(frame_ids(sc_frames))
    assert not (era_ids & sc_ids), "era pool leaks transfer test matches"
    assert not (tr_ids & sc_ids)
    print(f"  transfer train {len(tr_ids)} matches, scored {len(sc_ids)}")

    sel_t = results.setdefault("transfer_selection", {"scores": {}})
    if "selected" in sel_t:
        best_tr = sel_t["selected"]
        print(f"== transfer arm: selection already done -> {best_tr} ==")
    else:
        # Config selection for the transfer arm: internal chronological
        # validation split of the ERA pool (latest 25% per competition),
        # bundles trained on era sub-train + the 17 Bundesliga matches.
        # Cross-fitting is reserved for calibration on the 17 (below).
        era_sub, era_val = cal.match_group_split(
            era_frames, era_meta, calib_frac=0.25
        )
        era_sub_capped = imp.subsample_frames(era_sub, imp.SELECTION_POOL_CAP)
        for name, config in imp.CONFIGS.items():
            if name in sel_t["scores"]:
                print(f"  {name}: (cached)")
                continue
            bundle = imp.train_bundle(
                era_sub_capped + tr_frames, meta_all, config, seed=seed,
                match_dates=dict(
                    zip(
                        meta_all["match_id"].astype(int),
                        meta_all["match_date"].astype(str),
                    )
                ),
                labels=LABELS,
            )
            tot = {}
            for label in LABELS:
                ys, ps = [], []
                for f in era_val:
                    ys.append(f[label].to_numpy())
                    ps.append(imp.raw_predict(bundle, f)[label])
                from effc.vaep.models import brier_score

                tot[label] = brier_score(np.concatenate(ys), np.concatenate(ps))
            sel_t["scores"][name] = tot
            save_results(results)
            print(f"  {name}: " + ", ".join(f"{k}={v:.5f}" for k, v in tot.items()))
            del bundle
        best_tr = min(
            sel_t["scores"], key=lambda n: sum(sel_t["scores"][n].values())
        )
        sel_t["selected"] = best_tr
        save_results(results)
        print(f"  -> selected {best_tr}")

    if "season_transfer" in results:
        print("== transfer arm: 2023/24 already scored ==")
    else:
        print("== transfer arm: cross-fit calibration + final training ==")
        models_tr, calib_rep_tr = imp.train_improved_models_cached(
            tr_frames, era_frames, meta_all, best_tr, seed=seed
        )
        results["transfer_calibration"] = calib_rep_tr
        save_results(results)

        print("== transfer arm: scoring 2023/24 ONCE ==")
        results["season_transfer"] = score_block(models_tr, tr_frames, sc_frames)
        save_results(results)
        for label in LABELS:
            r = results["season_transfer"][label]
            print(
                f"  {label}: model {r['model']['mean']:.5f} vs baseline "
                f"{r['base_rate_baseline']['mean']:.5f}"
            )

    results["elapsed_s"] = time.time() - t0
    save_results(results)
    print(f"\nwrote runs/beat_baseline/results.json in {results['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
