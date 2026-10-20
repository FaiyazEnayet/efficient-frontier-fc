"""P(scores) and P(concedes) estimators: gradient boosting on game states.

Two window conventions (see vaep.values for why both are needed):
  incl: goal in actions i..i+k-1 (the action's own goal counts)
  excl: goal in actions i+1..i+k
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

LABELS = ["scores_incl", "scores_excl", "concedes_incl", "concedes_excl"]


def train_goal_models(
    frames: list[pd.DataFrame], seed: int = 0
) -> dict:
    """Train on pooled action frames. Returns {label: clf} + columns."""
    from effc.vaep.features import design_matrix

    df = pd.concat(frames, ignore_index=True)
    X = design_matrix(df)
    models: dict = {}
    for label in LABELS:
        y = df[label].to_numpy()
        clf = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=seed,
        )
        clf.fit(X, y)
        models[label] = clf
    models["columns"] = list(X.columns)
    return models


def predict_proba(models: dict, df: pd.DataFrame) -> dict[str, np.ndarray]:
    from effc.vaep.features import design_matrix

    if "strengths" in models:
        # Improved bundle: attach the frozen Dixon-Coles strengths
        # (fit on training matches only) before featurizing. Pure
        # inference on this frame; nothing is fit here.
        from effc.vaep.team_strength import attach_strength_features

        home = models.get("home_team_by_match", {}).get(
            int(df["match_id"].iloc[0]), ""
        )
        df = attach_strength_features(df, models["strengths"], home)
    X = design_matrix(df).reindex(columns=models["columns"], fill_value=0.0)
    return {label: models[label].predict_proba(X)[:, 1] for label in LABELS}


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))
