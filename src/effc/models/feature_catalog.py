"""Frozen source contract for U07: stronger goal-probability features
and model search.

Part 1 only. This module freezes the shipped fc6_pool baseline (the
exact config, feature schema, labels and lookahead that the deployed
goal models use) and records the candidate families (a) to (f) from
the U07 source as proposed specifications. Nothing here is evaluated
or selected: every candidate carries status "proposed" and
evaluated False, and the leakage gate in evaluation/development_trials
refuses any selection that used either original FC6 scored arm.

Discovered paths: the contract names models/feature_catalog.py and
evaluation/development_trials.py; this repo lays code out under
src/effc/, so the real paths are src/effc/models/feature_catalog.py
and src/effc/evaluation/development_trials.py. No contract path is
invented: vaep/features.py, vaep/improved.py and vaep/models.py all
live under src/effc/vaep/ at HEAD.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from effc.vaep.features import FEATURE_COLS, FEATURE_VERSION, K
from effc.vaep.improved import CONFIGS
from effc.vaep.models import LABELS
from effc.vaep.team_strength import STRENGTH_COLS

CATALOG_VERSION = 1
BASELINE_CONFIG_NAME = "fc6_pool"
BASELINE_SEED = 0

# The design-matrix columns a shipped frame must carry: the numeric
# state features plus the Dixon-Coles strength columns (zero for the
# fc6_pool config, which trains without strengths).
STRENGTH_COLUMNS = list(STRENGTH_COLS)

FC6_BASELINE = {
    "config_name": BASELINE_CONFIG_NAME,
    "catalog_version": CATALOG_VERSION,
    "gbm_params": dict(CONFIGS[BASELINE_CONFIG_NAME]["gbm"]),
    "use_strengths": bool(CONFIGS[BASELINE_CONFIG_NAME]["use_strengths"]),
    "dc_half_life": CONFIGS[BASELINE_CONFIG_NAME]["dc_half_life"],
    "recency_half_life": CONFIGS[BASELINE_CONFIG_NAME]["recency_half_life"],
    "feature_version": FEATURE_VERSION,
    "lookahead_k": K,
    "labels": list(LABELS),
    "seed": BASELINE_SEED,
    "numeric_features": list(FEATURE_COLS),
    "strength_features": list(STRENGTH_COLS),
}

# Candidate families from the U07 source, in increasing complexity.
# Each entry is a specification only: required_raw_fields names the
# StatsBomb event fields the family would read, and availability
# states what the family may describe. Post-action features may
# describe the completed current action but must not encode any
# future window outcome.
CANDIDATE_FAMILIES = [
    {
        "key": "a",
        "name": "corrected-geometry existing GBM",
        "description": (
            "The shipped HistGradientBoosting recipe on the corrected "
            "SPADL geometry (feature version 4): no new fields, the "
            "reference point for every richer family."
        ),
        "required_raw_fields": ["location", "pass.end_location",
                                "carry.end_location", "shot.end_location"],
        "availability": "within-action geometry only",
        "status": "proposed",
        "evaluated": False,
        "selected_on_fc6": False,
    },
    {
        "key": "b",
        "name": "richer event context",
        "description": (
            "Event result, body part, pass height, pass type and carry "
            "context, plus the prior two or three actions."
        ),
        "required_raw_fields": ["pass.outcome.name", "pass.height.name",
                                "pass.type.name", "pass.body_part.name",
                                "shot.body_part.name", "shot.type.name",
                                "carry.end_location", "prev_action",
                                "prev_prev_action"],
        "availability": "completed current action and prior actions only",
        "status": "proposed",
        "evaluated": False,
        "selected_on_fc6": False,
    },
    {
        "key": "c",
        "name": "tuned histogram GBM",
        "description": (
            "HistGradientBoosting with tuned hyperparameters (max_iter, "
            "learning rate, leaf count, minimum leaf size, L2) chosen "
            "on the internal development split only."
        ),
        "required_raw_fields": ["location", "pass.end_location",
                                "carry.end_location", "shot.end_location"],
        "availability": "within-action geometry only",
        "status": "proposed",
        "evaluated": False,
        "selected_on_fc6": False,
    },
    {
        "key": "d",
        "name": "LightGBM/CatBoost",
        "description": (
            "LightGBM or CatBoost where permitted and justified: the "
            "dependency must be available in the training environment "
            "and the choice justified by a measured benefit, not assumed."
        ),
        "required_raw_fields": ["location", "pass.end_location",
                                "carry.end_location", "shot.end_location"],
        "availability": "within-action geometry only",
        "status": "proposed",
        "evaluated": False,
        "selected_on_fc6": False,
    },
    {
        "key": "e",
        "name": "regularized logistic comparator",
        "description": (
            "Regularized logistic regression on the same design matrix: "
            "an interpretable comparator that bounds what linear "
            "structure the richer models must beat."
        ),
        "required_raw_fields": ["location", "pass.end_location",
                                "carry.end_location", "shot.end_location"],
        "availability": "within-action geometry only",
        "status": "proposed",
        "evaluated": False,
        "selected_on_fc6": False,
    },
    {
        "key": "f",
        "name": "compact sequence model",
        "description": (
            "A compact sequence model over the action stream, only "
            "after the simpler feature candidates are evaluated."
        ),
        "required_raw_fields": ["location", "pass.end_location",
                                "carry.end_location", "shot.end_location",
                                "prev_action", "prev_prev_action"],
        "availability": "completed current action and prior actions only",
        "status": "proposed",
        "evaluated": False,
        "selected_on_fc6": False,
    },
]

# Raw event fields this repo reads (vaep/features.py) plus the fields
# candidate family (b) would add, and the derived prior-action context
# tokens. Anything outside this set is unknown to the contract.
ALLOWED_RAW_FIELDS = frozenset([
    "location",
    "pass.end_location",
    "carry.end_location",
    "shot.end_location",
    "shot.outcome.name",
    "dribble.overrun",
    "type.name",
    "team.name",
    "player.id",
    "player.name",
    "minute",
    "second",
    "period",
    "index",
    "under_pressure",
    "pass.outcome.name",
    "pass.height.name",
    "pass.type.name",
    "pass.body_part.name",
    "shot.body_part.name",
    "shot.type.name",
    "prev_action",
    "prev_prev_action",
    "prev_same_team",
])

# Substrings that mark a field as encoding a future window outcome.
# Labels and lookahead windows are outcomes, never features.
FORBIDDEN_FIELD_TOKENS = (
    "scores_",
    "concedes_",
    "lookahead",
    "next_goal",
    "future",
    "window",
    "goal_in_",
    "is_goal",
)


def validate_feature_availability(spec: dict) -> bool:
    """Check a candidate feature spec against the availability rule.

    Every declared field must be a known raw event field (or a prior
    action context token) and must not encode a future window outcome.
    Raises ValueError on the first violation; returns True when the
    spec is allowed. Post-action fields describe the completed
    current action; prior-action fields describe earlier actions.
    """
    for field in spec.get("required_raw_fields", []):
        lowered = str(field).lower()
        for token in FORBIDDEN_FIELD_TOKENS:
            if token in lowered:
                raise ValueError(
                    f"field {field!r} encodes a future window outcome "
                    f"(matched {token!r}); post-action features must not "
                    "encode future outcomes"
                )
        if field not in ALLOWED_RAW_FIELDS:
            raise ValueError(
                f"field {field!r} is not a known raw event field in this "
                "contract; record the discovered field before using it"
            )
    return True


def expected_design_columns(df: pd.DataFrame) -> list[str]:
    """The design-matrix columns a frame must produce.

    Numeric state features, strength columns, and one dummy column
    per action and prior-action value present in the frame (prefixes
    acti_ and prev_, as built by vaep.features.design_matrix).
    """
    cols = list(FEATURE_COLS) + list(STRENGTH_COLS)
    for source, prefix in (("action", "acti"), ("prev_action", "prev")):
        for value in sorted({str(v) for v in df[source].unique()}):
            cols.append(f"{prefix}_{value}")
    return cols


def check_schema(X: pd.DataFrame) -> bool:
    """Schema check for a design matrix: exact expected columns.

    Raises ValueError naming any missing numeric feature or any
    unexpected column (a label or lookahead outcome smuggled in as a
    feature is caught here). Returns True when the schema matches.
    """
    cols = list(X.columns)
    missing = [c for c in FEATURE_COLS + list(STRENGTH_COLS)
               if c not in cols]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    allowed_prefixes = ("acti_", "prev_")
    unexpected = [
        c for c in cols
        if c not in FEATURE_COLS
        and c not in STRENGTH_COLS
        and not c.startswith(allowed_prefixes)
    ]
    if unexpected:
        raise ValueError(f"unexpected feature columns: {unexpected}")
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(f"non-numeric feature columns: {non_numeric}")
    return True


def leakage_gate_statement() -> str:
    """The U07 leakage gate, as a frozen sentence."""
    return (
        "No tuned feature, cutoff, architecture, or model is selected "
        "using either original FC6 scored arm; selection uses the "
        "internal development split only, and the frozen separate "
        "benchmark is scored once after selection."
    )


def catalog_digest() -> str:
    """SHA-256 digest of the frozen baseline and candidate statuses.

    Binds a baseline capture to the exact contract it was captured
    under: any change to the baseline config or to a candidate's
    evaluation status changes the digest.
    """
    payload = {
        "baseline": FC6_BASELINE,
        "candidates": [
            {k: fam[k] for k in ("key", "name", "status", "evaluated",
                                 "selected_on_fc6")}
            for fam in CANDIDATE_FAMILIES
        ],
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
