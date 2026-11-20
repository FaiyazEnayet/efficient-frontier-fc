"""Forward-only, point-in-time evaluation.

Stage 3 U04 part 1: the source contract and independent oracle for a
strict historical deployment simulation. The existing backtest scores
selections on a later half of the season, but the goal models behind the
ratings train on pooled matches that can postdate the decisions being
scored (UEFA Euro 2020 and the Indian Super League 2021/22 are pooled
training data for 2015/16 decisions). This module does not retrain those
models and does not replace any published result. It defines the
discipline a forward backtest must satisfy, enforces it at the
interface, and records every decision so an independent check can
recompute it from the raw rows.
"""

from effc.evaluation.forward_backtest import (
    POLICIES,
    ForwardDecision,
    PointInTimeError,
    assert_pool_point_in_time,
    build_decision_schedule,
    canonical_hash,
    check_pool_point_in_time,
    decision_cache_key,
    decision_record,
    file_sha256,
    ledger_artifact_paths,
    run_forward_backtest,
    score_squad_with_penalty,
    validate_decision,
    verify_ledger,
    verify_record_hash,
)

__all__ = [
    "POLICIES",
    "ForwardDecision",
    "PointInTimeError",
    "assert_pool_point_in_time",
    "build_decision_schedule",
    "canonical_hash",
    "check_pool_point_in_time",
    "decision_cache_key",
    "decision_record",
    "file_sha256",
    "ledger_artifact_paths",
    "run_forward_backtest",
    "score_squad_with_penalty",
    "validate_decision",
    "verify_ledger",
    "verify_record_hash",
]
