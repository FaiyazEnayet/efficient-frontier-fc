"""Candidate goal-probability features for U07 part 2.

Part 2 only. This module implements the richer feature family (b)
from the U07 source as trial code: event result, body part, pass
height, pass type and carry context, plus the prior two or three
actions. It also builds the lagged action-window matrix family (f)
trains on.

Deliberate boundary: nothing here changes the shipped featurization.
``vaep.features.featurize_match`` and ``design_matrix`` are untouched,
so the shipped design matrix, FEATURE_VERSION and every cache key
stay exactly as part 1 froze them. These richer columns live only in
trial frames built by ``extend_frame_with_richer_context``; if a
candidate ever earns promotion into the shipped recipe, that move
belongs to a follow-up change with a feature-version bump, not to
this experiment.

Availability: every column here describes the completed current
action or an earlier action only. The raw fields read are the ones
the frozen catalog allows, and ``validate_candidate_spec`` runs the
catalog availability check on the family (b) spec before any
extraction. No label, lookahead window or future outcome is read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from effc.models import feature_catalog as catalog
from effc.vaep.features import design_matrix

# Family (b) columns added to a trial frame. Numeric columns carry
# 0.0 where the concept does not apply (a non-pass has no pass
# outcome); categorical columns carry "NA" there, so a missing value
# can never be confused with a real event value.
RICH_NUMERIC_COLUMNS = [
    "rich_is_pass",
    "rich_pass_complete",
    "rich_is_carry",
    "rich_carry_length",
    "rich_is_shot",
    "rich_prev_prev_same_team",
]

RICH_CATEGORICAL_COLUMNS = [
    "rich_pass_height",
    "rich_pass_type",
    "rich_pass_body_part",
    "rich_shot_body_part",
    "rich_shot_type",
    "rich_prev_prev_action",
]

RICH_COLUMNS = RICH_NUMERIC_COLUMNS + RICH_CATEGORICAL_COLUMNS

# Raw StatsBomb fields family (b) reads, taken from the frozen
# catalog spec so the availability check below and the catalog agree.
FAMILY_B_FIELDS = [
    fam["required_raw_fields"]
    for fam in catalog.CANDIDATE_FAMILIES
    if fam["key"] == "b"
][0]


def validate_candidate_spec() -> bool:
    """Availability check for family (b) against the frozen catalog.

    Raises ValueError if any declared field is unknown or encodes a
    future window outcome. Returns True when the spec is allowed.
    """
    spec = [fam for fam in catalog.CANDIDATE_FAMILIES if fam["key"] == "b"][0]
    return catalog.validate_feature_availability(spec)


def _event_by_position(events: list[dict]) -> dict[int, dict]:
    """Raw events keyed by position in the sorted event list.

    ``featurize_match`` numbers action rows by position in the sorted
    list (its ``match_index``), not by the StatsBomb ``index`` field,
    so the same position key aligns raw events to frame rows.
    """
    ordered = sorted(events, key=lambda e: e["index"])
    return {pos: ev for pos, ev in enumerate(ordered)}


def extend_frame_with_richer_context(
    frame: pd.DataFrame, events: list[dict]
) -> pd.DataFrame:
    """Add family (b) richer-context columns to a featurized frame.

    Each row describes the completed current action (pass outcome,
    height, type and body part; shot body part and type; carry length)
    and the action two places back (``rich_prev_prev_action`` and
    whether it was the same team). Nothing is read from any later
    action, so flipping a future goal cannot change an earlier row's
    richer columns. Raises ValueError when the frame has no
    ``match_index`` column or an event position is missing.
    """
    validate_candidate_spec()
    if "match_index" not in frame.columns:
        raise ValueError(
            "frame has no match_index column; richer context needs the "
            "action position to align raw events"
        )
    by_pos = _event_by_position(events)
    actions = frame["action"].to_numpy()
    teams = frame["team"].to_numpy()
    out = frame.copy()
    n = len(frame)
    numeric = {col: np.zeros(n, dtype=float) for col in RICH_NUMERIC_COLUMNS}
    categorical = {col: np.full(n, "NA", dtype=object)
                   for col in RICH_CATEGORICAL_COLUMNS}
    for i, (_, row) in enumerate(frame.iterrows()):
        pos = int(row["match_index"])
        if pos not in by_pos:
            raise ValueError(
                f"no raw event at position {pos}; events do not cover "
                "the featurized frame"
            )
        ev = by_pos[pos]
        tname = str(row["action"])
        if tname == "Pass":
            numeric["rich_is_pass"][i] = 1.0
            outcome = (ev.get("pass") or {}).get("outcome") or {}
            # StatsBomb omits the outcome for a completed pass.
            numeric["rich_pass_complete"][i] = (
                0.0 if outcome.get("name") else 1.0
            )
            height = (ev.get("pass") or {}).get("height") or {}
            categorical["rich_pass_height"][i] = str(
                height.get("name") or "Ground"
            )
            ptype = (ev.get("pass") or {}).get("type") or {}
            categorical["rich_pass_type"][i] = str(
                ptype.get("name") or "Normal"
            )
            body = (ev.get("pass") or {}).get("body_part") or {}
            categorical["rich_pass_body_part"][i] = str(
                body.get("name") or "Foot"
            )
        elif tname == "Carry":
            numeric["rich_is_carry"][i] = 1.0
            numeric["rich_carry_length"][i] = float(
                np.hypot(row["dx"], row["dy"])
            )
        elif tname == "Shot":
            numeric["rich_is_shot"][i] = 1.0
            body = (ev.get("shot") or {}).get("body_part") or {}
            categorical["rich_shot_body_part"][i] = str(
                body.get("name") or "Foot"
            )
            stype = (ev.get("shot") or {}).get("type") or {}
            categorical["rich_shot_type"][i] = str(
                stype.get("name") or "Normal"
            )
        if i >= 2:
            categorical["rich_prev_prev_action"][i] = str(actions[i - 2])
            numeric["rich_prev_prev_same_team"][i] = (
                1.0 if teams[i - 2] == teams[i] else 0.0
            )
        else:
            categorical["rich_prev_prev_action"][i] = "None"
    for col in RICH_NUMERIC_COLUMNS:
        out[col] = numeric[col]
    for col in RICH_CATEGORICAL_COLUMNS:
        out[col] = categorical[col]
    return out


def extended_design_matrix(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    """Design matrix for a trial family.

    Family "a" is the shipped matrix unchanged. Family "b" is the
    shipped matrix plus the richer numeric columns and one-hot
    dummies for the richer categorical columns. Raises ValueError for
    an unknown family, or for family "b" when the frame was not
    extended with richer context first.
    """
    if family == "a":
        return design_matrix(frame)
    if family != "b":
        raise ValueError(
            f"unknown trial family {family!r}; expected 'a' or 'b'"
        )
    missing = [c for c in RICH_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            "frame lacks richer-context columns "
            f"{missing}; run extend_frame_with_richer_context first"
        )
    X = design_matrix(frame)
    for col in RICH_NUMERIC_COLUMNS:
        X[col] = frame[col].astype(np.float32).to_numpy()
    for col in RICH_CATEGORICAL_COLUMNS:
        dummies = pd.get_dummies(frame[col], prefix=col)
        X = pd.concat([X, dummies], axis=1)
    return X.astype(np.float32)


def sequence_window_matrix(frame: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Lagged action-window matrix for family (f).

    The shipped design matrix for the current action plus the same
    columns shifted by one and two actions (zero-filled at the start
    of a match), so a compact sequence model sees the current action
    and the two before it. The window may cross a period boundary;
    that is recorded as a limitation of this trial, not hidden.
    """
    if window != 3:
        raise ValueError(
            f"sequence window is fixed at 3 actions, got {window}"
        )
    X = design_matrix(frame)
    parts = [X]
    for lag in (1, 2):
        shifted = X.shift(lag).fillna(0.0)
        shifted.columns = [f"lag{lag}_{c}" for c in X.columns]
        parts.append(shifted)
    return pd.concat(parts, axis=1).astype(np.float32)
