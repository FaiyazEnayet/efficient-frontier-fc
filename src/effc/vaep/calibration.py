"""Held-out calibration for the VAEP goal-probability models.

The pipeline used to train the goal models on all pre-split action frames
and then report Brier scores computed on those same frames: an in-sample
fit presented as the headline accuracy number. This module implements the
FC6 fix:

- match_group_split: split the pre-split frames into a training group and
  a held-out calibration group at the match level, chronological within
  each competition (calibration matches are never earlier than training
  matches, so no future information leaks into training);
- per_match_scores / base_rate_scores: Brier score and log loss per match
  for the fitted models and for a matched feasible baseline that predicts
  the training-set base rate (no tuning on the scored half);
- summarize: mean, standard deviation and bootstrap 95% intervals of the
  mean across matches, for the model and the baseline side by side;
- transfer_seasons / roster_overlap: the inputs for a genuine
  chronological season-transfer evaluation (train on the earlier season,
  score on the later one) and the player-pool stability between the
  training and deployment seasons.

Every comparison of trained-vs-scored data asserts disjoint match IDs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from effc.vaep.models import brier_score, predict_proba

LABELS = ("scores_incl", "scores_excl", "concedes_incl", "concedes_excl")


def assert_disjoint(train_ids, scored_ids, what: str = "match") -> None:
    """Raise when any ID was both trained and scored on."""
    train = {int(i) for i in train_ids}
    scored = {int(i) for i in scored_ids}
    overlap = train & scored
    if overlap:
        raise ValueError(
            f"{len(overlap)} {what} IDs are in both the training and the "
            f"scored set (e.g. {sorted(overlap)[:5]}); held-out evaluation "
            "requires disjoint IDs"
        )


def _frame_match_id(frame: pd.DataFrame) -> int:
    return int(frame["match_id"].iloc[0])


def match_group_split(
    frames: list[pd.DataFrame], meta: pd.DataFrame, calib_frac: float = 0.2
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Chronological per-competition train/calibration split of match frames.

    `frames` are action frames with a match_id column; `meta` has
    match_id, competition_id, season_id and match_date. Within each
    (competition_id, season_id) group, matches are ordered by
    (match_date, match_id) and the latest `calib_frac` become the
    calibration group. Groups with fewer than two matches stay entirely
    in training. Frames whose match is missing from `meta` go to
    training with a warning.

    Returns (train_frames, calib_frames); the two groups' match IDs are
    disjoint by construction.
    """
    by_match = {}
    for f in frames:
        by_match.setdefault(_frame_match_id(f), []).append(f)
    meta_idx = meta.set_index("match_id")
    groups: dict[tuple, list] = {}
    train_ids: set[int] = set()
    for mid in by_match:
        if mid not in meta_idx.index:
            print(f"[warn] match {mid} missing from metadata; keeping in training")
            train_ids.add(mid)
            continue
        row = meta_idx.loc[mid]
        key = (row["competition_id"], row["season_id"])
        groups.setdefault(key, []).append((str(row["match_date"]), mid))
    calib_ids: set[int] = set()
    for key, members in groups.items():
        members.sort()
        n_calib = int(round(len(members) * calib_frac)) if len(members) >= 2 else 0
        calib_ids.update(mid for _, mid in members[len(members) - n_calib :])
        train_ids.update(mid for _, mid in members[: len(members) - n_calib])
    assert_disjoint(train_ids, calib_ids)
    train_frames = [f for mid in train_ids for f in by_match[mid]]
    calib_frames = [f for mid in calib_ids for f in by_match[mid]]
    return train_frames, calib_frames


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    """Natural-log log loss with clipped probabilities."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def per_match_scores(
    models: dict, frames: list[pd.DataFrame], labels: tuple = LABELS
) -> pd.DataFrame:
    """Brier score and log loss per match for fitted goal models.

    Returns one row per match: match_id, n_actions, brier_<label>,
    logloss_<label>.
    """
    rows = []
    for f in frames:
        preds = predict_proba(models, f)
        row: dict = {"match_id": _frame_match_id(f), "n_actions": len(f)}
        for label in labels:
            y = f[label].to_numpy()
            row[f"brier_{label}"] = brier_score(y, preds[label])
            row[f"logloss_{label}"] = log_loss(y, preds[label])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("match_id").reset_index(drop=True)


def base_rate_scores(
    train_frames: list[pd.DataFrame],
    scored_frames: list[pd.DataFrame],
    labels: tuple = LABELS,
) -> pd.DataFrame:
    """Matched feasible baseline: predict the training-set base rate.

    Scored on the same matches as the model, with the constant prediction
    P(label) estimated on the training frames only: no tuning on the
    scored half, and feasible for anyone to reproduce.
    """
    train = pd.concat(train_frames, ignore_index=True)
    rows = []
    for f in scored_frames:
        row: dict = {"match_id": _frame_match_id(f), "n_actions": len(f)}
        y_len = len(f)
        for label in labels:
            p = float(train[label].mean())
            y = f[label].to_numpy()
            row[f"brier_{label}"] = brier_score(y, np.full(y_len, p))
            row[f"logloss_{label}"] = log_loss(y, np.full(y_len, p))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("match_id").reset_index(drop=True)


def summarize(
    per_match: pd.DataFrame, labels: tuple = LABELS, n_boot: int = 2000, seed: int = 0
) -> dict:
    """Mean, std and bootstrap 95% CI of the mean across matches.

    Returns {label: {metric: {mean, std, n_matches, ci95}}}.
    """
    rng = np.random.default_rng(seed)
    out: dict = {}
    for label in labels:
        out[label] = {}
        for metric in ("brier", "logloss"):
            vals = per_match[f"{metric}_{label}"].to_numpy(dtype=float)
            n = len(vals)
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            boots = rng.choice(vals, size=(n_boot, n), replace=True).mean(axis=1)
            lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
            out[label][metric] = {
                "mean": mean,
                "std": std,
                "n_matches": n,
                "ci95": [lo, hi],
            }
    return out


def transfer_seasons(
    meta: pd.DataFrame, competition_id: int = 9
) -> tuple[int, int, dict[int, str]]:
    """Pick the earliest and latest seasons of a competition.

    Returns (train_season_id, scored_season_id, {season_id: season_name}).
    Training on the earlier season and scoring on the later one is a
    genuine chronological deployment simulation. Raises ValueError when
    fewer than two seasons are present.
    """
    sub = meta[meta["competition_id"] == competition_id]
    if sub.empty:
        raise ValueError(f"no matches for competition {competition_id}")
    first_seen = sub.groupby("season_id")["match_date"].min().sort_values()
    if len(first_seen) < 2:
        raise ValueError(
            f"season transfer needs two seasons, found {len(first_seen)}"
        )
    train_season = int(first_seen.index[0])
    scored_season = int(first_seen.index[-1])
    if "season_name" in sub.columns:
        names = (
            sub.drop_duplicates("season_id")
            .set_index("season_id")["season_name"]
            .to_dict()
        )
        names = {int(k): str(v) for k, v in names.items()}
    else:
        names = {}
    return train_season, scored_season, names


def roster_overlap(
    train_player_ids, scored_player_ids, scored_minutes: pd.DataFrame
) -> dict:
    """Player-pool stability between a training season and a scored season.

    Reports the fraction of the scored season's players (and of its played
    minutes) whose player_id was already seen in the training season: the
    share of the deployment roster the training data actually covers.
    """
    train_player_ids = set(train_player_ids)
    scored_player_ids = set(scored_player_ids)
    n_scored = len(scored_player_ids)
    n_overlap = len(scored_player_ids & train_player_ids)
    player_frac = n_overlap / n_scored if n_scored else 0.0
    total_minutes = float(scored_minutes["minutes"].sum())
    seen_minutes = float(
        scored_minutes[scored_minutes["player_id"].isin(train_player_ids)][
            "minutes"
        ].sum()
    )
    return {
        "n_train_players": len(train_player_ids),
        "n_scored_players": n_scored,
        "n_overlap_players": n_overlap,
        "player_overlap_frac": player_frac,
        "minutes_overlap_frac": seen_minutes / total_minutes
        if total_minutes > 0
        else 0.0,
    }


def build_report(
    *,
    train_ids,
    calib_ids,
    in_sample_scores: dict,
    model_summary: dict,
    baseline_summary: dict,
) -> dict:
    """Assemble the metrics.json calibration block.

    `in_sample_scores` maps label -> {"brier": x, "logloss": y} measured on
    the training frames. Every emitted Brier key names the data it was
    computed on: in-sample values carry an explicit `_in_sample` suffix
    and held-out values live under `held_out`, so no bare Brier number
    can be read as a generalization claim.
    """
    assert_disjoint(train_ids, calib_ids)
    report: dict = {
        "n_train_matches": len({int(i) for i in train_ids}),
        "n_calib_matches": len({int(i) for i in calib_ids}),
    }
    for label, scores in in_sample_scores.items():
        for metric, value in scores.items():
            report[f"{metric}_{label}_in_sample"] = float(value)
    report["held_out"] = {
        label: {
            "model": model_summary[label],
            "base_rate_baseline": baseline_summary[label],
        }
        for label in model_summary
    }
    return report
