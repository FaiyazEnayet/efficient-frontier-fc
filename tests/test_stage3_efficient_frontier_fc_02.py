"""Stage 3 U02 tests: raw-event spatial geometry and action semantics.

Fast, no data downloads. Synthetic events replicate structures verified
against real StatsBomb open data (Chelsea vs Arsenal 2015/16 and a second
sampled match): own goals arrive as an "Own Goal For" / "Own Goal Against"
pair, and each event's coordinates sit in the acting team's attacking
frame (an opponent action after a pass sits at the mirror of the pass
end, while same-team chains are continuous).
"""

import numpy as np
import pytest

from effc.vaep import features as F
from effc.vaep.features import FEATURE_VERSION, _to_spadl, design_matrix, featurize_match


def ev(index, team, tname, period=1, location=None, minute=1, second=0, extra=None):
    e = {
        "index": index,
        "period": period,
        "minute": minute,
        "second": second,
        "type": {"name": tname},
        "team": {"name": team},
        "player": {"id": 1, "name": "P1"},
    }
    if location is not None:
        e["location"] = location
    if extra:
        e.update(extra)
    return e


def passing(index, team, start, end, period=1):
    return ev(
        index, team, "Pass", period=period, location=start,
        extra={"pass": {"end_location": end}},
    )


def shot_goal(index, team, start, end, period=1):
    return ev(
        index, team, "Shot", period=period, location=start,
        extra={"shot": {"outcome": {"name": "Goal"}, "end_location": end}},
    )


def own_goal_pair(index, benefit, conceder, period=1):
    # Real-data shape: the pair arrives consecutively, in either order.
    return [
        ev(index, benefit, "Own Goal For", period=period, location=[106.3, 35.7]),
        ev(index + 1, conceder, "Own Goal Against", period=period, location=[13.8, 44.4]),
    ]


def test_stage3_efficient_frontier_fc_02_feature_version_bumped():
    assert FEATURE_VERSION == 4


def test_stage3_efficient_frontier_fc_02_own_goal_labels_and_score_diff():
    events = [
        passing(0, "A", [60.0, 40.0], [70.0, 40.0]),
        passing(1, "A", [70.0, 40.0], [80.0, 40.0]),
        passing(2, "B", [80.0, 40.0], [70.0, 40.0]),
        *own_goal_pair(3, "A", "B"),
        passing(5, "A", [60.0, 40.0], [70.0, 40.0]),
        passing(6, "B", [60.0, 40.0], [70.0, 40.0]),
    ]
    df = featurize_match(events, match_id=1)
    # Own goals add no action rows.
    assert len(df) == 5
    # Rows before the goal: benefiting team scores, conceding team concedes,
    # identically in the incl and excl windows (an own goal is never any
    # action's own goal, so the two conventions agree on it).
    assert df.loc[0, "scores_incl"] == 1.0
    assert df.loc[0, "scores_excl"] == 1.0
    assert df.loc[1, "scores_incl"] == 1.0
    assert df.loc[1, "scores_excl"] == 1.0
    assert df.loc[2, "concedes_incl"] == 1.0
    assert df.loc[2, "concedes_excl"] == 1.0
    assert df.loc[2, "scores_incl"] == 0.0
    # Rows after the goal see nothing from it (it is in the past).
    assert df.loc[3, "scores_incl"] == 0.0
    assert df.loc[3, "concedes_incl"] == 0.0
    assert df.loc[4, "scores_incl"] == 0.0
    assert df.loc[4, "concedes_incl"] == 0.0
    # score_diff counts the own goal for every later action.
    assert df.loc[0, "score_diff"] == 0.0
    assert df.loc[2, "score_diff"] == 0.0
    assert df.loc[3, "score_diff"] == 1.0
    assert df.loc[4, "score_diff"] == -1.0


def test_stage3_efficient_frontier_fc_02_own_goal_pair_order_independent():
    base = [
        passing(0, "A", [60.0, 40.0], [70.0, 40.0]),
        passing(1, "B", [60.0, 40.0], [70.0, 40.0]),
    ]
    for_first = base + own_goal_pair(2, "A", "B") + [passing(4, "A", [60.0, 40.0], [70.0, 40.0])]
    against_first = base + list(reversed(own_goal_pair(2, "A", "B"))) + [
        passing(4, "A", [60.0, 40.0], [70.0, 40.0])
    ]
    d1 = featurize_match(for_first, match_id=1)
    d2 = featurize_match(against_first, match_id=1)
    for col in ("scores_incl", "scores_excl", "concedes_incl", "concedes_excl", "score_diff"):
        assert (d1[col].to_numpy() == d2[col].to_numpy()).all()


def test_stage3_efficient_frontier_fc_02_own_goal_period_boundary():
    events = [
        passing(0, "A", [60.0, 40.0], [70.0, 40.0], period=1),
        passing(1, "B", [60.0, 40.0], [70.0, 40.0], period=1),
        passing(2, "A", [60.0, 40.0], [70.0, 40.0], period=2),
        *own_goal_pair(3, "A", "B", period=2),
        passing(5, "B", [60.0, 40.0], [70.0, 40.0], period=2),
    ]
    df = featurize_match(events, match_id=1)
    # Period-1 rows must not see the period-2 own goal.
    assert df.loc[0, "scores_incl"] == 0.0
    assert df.loc[1, "concedes_incl"] == 0.0
    # Period-2 row before the goal does.
    assert df.loc[2, "scores_incl"] == 1.0
    assert df.loc[2, "scores_excl"] == 1.0
    # Period-2 row after the goal does not.
    assert df.loc[3, "scores_incl"] == 0.0


def test_stage3_efficient_frontier_fc_02_orientation_both_teams_both_periods():
    events = []
    idx = 0
    for period in (1, 2):
        for team in ("A", "B"):
            events.append(
                ev(idx, team, "Shot", period=period, location=[100.0, 40.0],
                   extra={"shot": {"outcome": {"name": "Off T"}, "end_location": [120.0, 40.0]}})
            )
            idx += 1
    df = featurize_match(events, match_id=1)
    # Every shot, either team, either period, starts in the attacking half
    # of the SPADL frame: each event is recorded in the acting team's
    # attacking frame, so the attacked goal is always (105, 34).
    assert (df["start_x"].to_numpy() > 52.5).all()
    assert (df["dist_start"].to_numpy() < 105.0).all()


def test_stage3_efficient_frontier_fc_02_spadl_preserves_turnover_mirror():
    # Real-data property: an opponent action after a pass sits at the
    # mirror (120-x, 80-y) of the pass end. The scale-only conversion
    # must preserve that mirror relation exactly.
    for x, y in ((69.6, 48.9), (76.0, 33.1), (83.5, 45.1)):
        mx, my = _to_spadl(120.0 - x, 80.0 - y)
        cx, cy = _to_spadl(x, y)
        assert mx == pytest.approx(105.0 - cx)
        assert my == pytest.approx(68.0 - cy)


def test_stage3_efficient_frontier_fc_02_goal_window_period_boundary():
    events = [
        passing(0, "A", [60.0, 40.0], [70.0, 40.0], period=1),
        passing(1, "A", [70.0, 40.0], [80.0, 40.0], period=1),
        shot_goal(2, "A", [100.0, 40.0], [120.0, 40.0], period=1),
        passing(3, "B", [60.0, 40.0], [70.0, 40.0], period=2),
        passing(4, "A", [60.0, 40.0], [70.0, 40.0], period=2),
    ]
    df = featurize_match(events, match_id=1)
    # The goal counts for its own state (incl) but not the after-state.
    assert df.loc[2, "scores_incl"] == 1.0
    assert df.loc[2, "scores_excl"] == 0.0
    # Period-1 rows before it see it.
    assert df.loc[0, "scores_incl"] == 1.0
    # Period-2 rows must not: labels never cross a period boundary.
    assert df.loc[3, "scores_incl"] == 0.0
    assert df.loc[3, "concedes_incl"] == 0.0
    assert df.loc[4, "scores_incl"] == 0.0


def test_stage3_efficient_frontier_fc_02_no_self_label_in_design_matrix():
    events = [
        passing(0, "A", [60.0, 40.0], [70.0, 40.0]),
        shot_goal(1, "A", [100.0, 40.0], [120.0, 40.0]),
        passing(2, "B", [60.0, 40.0], [70.0, 40.0]),
    ]
    df = featurize_match(events, match_id=1)
    X = design_matrix(df)
    for col in ("scores_incl", "scores_excl", "concedes_incl", "concedes_excl", "is_goal"):
        assert col not in X.columns


def test_stage3_efficient_frontier_fc_02_shot_end_location_with_height():
    # Real StatsBomb shots carry a 3-element end_location [x, y, z].
    events = [
        ev(0, "A", "Shot", location=[115.1, 45.5],
           extra={"shot": {"outcome": {"name": "Saved"}, "end_location": [117.0, 40.5, 0.2]}})
    ]
    df = featurize_match(events, match_id=1)
    assert df.loc[0, "end_x"] == pytest.approx(117.0 * 105.0 / 120.0)
    assert df.loc[0, "end_y"] == pytest.approx(40.5 * 68.0 / 80.0)


def test_stage3_efficient_frontier_fc_02_own_goal_debit_in_add_vaep():
    from effc.vaep.values import add_vaep
    import effc.vaep.models as M

    events = [
        passing(0, "A", [60.0, 40.0], [70.0, 40.0]),
        *own_goal_pair(1, "A", "B"),
        passing(3, "B", [60.0, 40.0], [70.0, 40.0]),
    ]
    df = featurize_match(events, match_id=1)
    orig = M.predict_proba
    M.predict_proba = lambda models, d: {k: np.full(len(d), v) for k, v in
        [("scores_incl", 0.02), ("scores_excl", 0.02),
         ("concedes_incl", 0.01), ("concedes_excl", 0.01)]}
    try:
        out = add_vaep(df, {})
    finally:
        M.predict_proba = orig
    # B's first action after the own goal: turnover before-state gives
    # pc_before 0.02, pc_excl 0.01, then the symmetric -1.0 debit.
    assert out.loc[1, "vaep_def"] == pytest.approx(0.02 - 0.01 - 1.0)
