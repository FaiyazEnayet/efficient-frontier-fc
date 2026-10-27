"""VAEP: value each action by the change in goal probabilities it causes.

For action i by team T, with S_i the state after the action:

    vaep_off  = P_excl(T scores | S_i) - P_incl(T scores | S_{i-1}) + goal_i
    vaep_def  = P_incl(T concedes | S_{i-1}) - P_excl(T concedes | S_i)
    vaep      = vaep_off + vaep_def

The incl/excl asymmetry is deliberate: the "before" window includes the
current action so chances (and saves) are credited where they belong,
while the "after" window excludes it so a scored goal doesn't poison the
state that follows. A scored goal adds +1 (realization above expectation).

When possession changes hands between actions, the before-state
probabilities are read from the opponent's perspective: P_T(scores) =
P_opp(concedes) and vice versa. This keeps the accounting zero-sum across
turnovers instead of punishing players for the other team's chances.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_vaep(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    from effc.vaep.models import predict_proba

    df = df.copy()
    preds = predict_proba(models, df)
    df["ps_excl"] = preds["scores_excl"]
    df["ps_incl"] = preds["scores_incl"]
    df["pc_excl"] = preds["concedes_excl"]
    df["pc_incl"] = preds["concedes_incl"]

    grp = df.groupby(["match_id", "period"])
    df["team_prev"] = grp["team"].shift(1)
    same = (df["team"] == df["team_prev"]).fillna(True)

    for col in ("ps_incl", "pc_incl"):
        df[col + "_prev"] = grp[col].shift(1)

    # before-state from the current actor's perspective; on a turnover the
    # opponent's "scores" is our "concedes"
    df["ps_before"] = np.where(
        same, df["ps_incl_prev"], df["pc_incl_prev"]
    )
    df["pc_before"] = np.where(
        same, df["pc_incl_prev"], df["ps_incl_prev"]
    )
    # first action of a period: fall back to base rates
    df["ps_before"] = df["ps_before"].fillna(df["ps_incl"].mean())
    df["pc_before"] = df["pc_before"].fillna(df["pc_incl"].mean())

    df["vaep_off"] = df["ps_excl"] - df["ps_before"] + df["is_goal"].astype(float)
    df["vaep_def"] = df["pc_before"] - df["pc_excl"]
    df["vaep"] = df["vaep_off"] + df["vaep_def"]

    # Goal-conceded debit. Scoring is +1 above expectation; conceding is
    # symmetric. Without this, the keeper's post-goal event earns positive
    # defensive value (P(concedes) "drops" only because the goal already
    # happened). The -1 lands on the conceding team's first action after
    # the goal, which in practice is the keeper's event.
    for j in df.index[df["is_goal"]]:
        scorer = df.at[j, "team"]
        mid, per, mi = df.at[j, "match_id"], df.at[j, "period"], df.at[j, "match_index"]
        mask = (
            (df["match_id"] == mid)
            & (df["period"] == per)
            & (df["match_index"] > mi)
            & (df["match_index"] <= mi + 3)
            & (df["team"] != scorer)
        )
        cands = df.index[mask]
        if len(cands):
            df.loc[cands[0], "vaep_def"] -= 1.0
            df.loc[cands[0], "vaep"] -= 1.0
    # Same debit after own goals. Own goals are not action rows, so they
    # have no is_goal row and the loop above never sees them; the
    # artifact it corrects applies to them exactly as to regular goals
    # (the keeper's pre-goal concedes probability now includes the own
    # goal through the labels). Own goals come from df.attrs, set by
    # featurize_match; frames built any other way simply have none.
    # Deliberate asymmetry, documented: no +1 realization is assigned
    # for an own goal, because no action row scored it; the benefiting
    # team's players earn nothing for it, and the conceding side bears
    # the debit. Assigning the +1 to an arbitrary preceding action
    # would fabricate credit.
    for own_idx, benefit, own_period in df.attrs.get("own_goals", []):
        if not len(df) or "match_id" not in df.columns:
            break
        mid = df["match_id"].iloc[0]
        mask = (
            (df["match_id"] == mid)
            & (df["period"] == own_period)
            & (df["match_index"] > own_idx)
            & (df["match_index"] <= own_idx + 3)
            & (df["team"] != benefit)
        )
        cands = df.index[mask]
        if len(cands):
            df.loc[cands[0], "vaep_def"] -= 1.0
            df.loc[cands[0], "vaep"] -= 1.0
    return df


def player_match_vaep(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate VAEP to one row per (match, player)."""
    g = df.groupby(["match_id", "team", "player_id", "player"], dropna=False)
    out = g.agg(
        vaep=("vaep", "sum"),
        vaep_off=("vaep_off", "sum"),
        vaep_def=("vaep_def", "sum"),
        actions=("vaep", "size"),
    ).reset_index()
    return out
