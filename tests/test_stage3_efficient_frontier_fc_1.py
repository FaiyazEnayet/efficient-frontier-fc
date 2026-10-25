"""Stage 3 C1 regression tests: StatsBomb raw coordinates are 120x80.

The contract under test: `featurize_match` receives raw StatsBomb event
locations and must convert them to the 105x68 SPADL frame (x * 105/120,
y * 68/80) before computing any distance, angle, or displacement feature.
Every event type goes through the same conversion, for both start and end
locations. StatsBomb already normalises events so the team in possession
always attacks toward x = 120, so no flip by team or period is applied:
the single goal at SPADL (105, 34) is correct for every team.

All fixtures below are written in RAW StatsBomb coordinates. Before the
C1 fix these tests fail: the old code treated raw values as SPADL values
directly (for example, a shot from raw (108, 40) reported a distance of
about 6.7 m instead of 10.5 m, and landed outside the SPADL pitch).
"""

import math

import pytest

from effc.vaep.features import ACTION_TYPES, FEATURE_VERSION, featurize_match


def _ev(idx, type_name, team="A", loc=(60.0, 40.0), end=None, dribble_overrun=False):
    ev = {
        "id": str(idx),
        "index": idx,
        "period": 1,
        "timestamp": "00:10:00.000",
        "minute": 10,
        "second": 0,
        "type": {"id": 1, "name": type_name},
        "possession": 1,
        "possession_team": {"id": 1, "name": team},
        "team": {"id": 1, "name": team},
        "player": {"id": 1, "name": "P1"},
        "location": list(loc),
    }
    if type_name == "Pass" and end is not None:
        ev["pass"] = {"end_location": list(end)}
    if type_name == "Carry" and end is not None:
        ev["carry"] = {"end_location": list(end)}
    if type_name == "Shot" and end is not None:
        ev["shot"] = {"outcome": {"name": "Saved"}, "end_location": list(end)}
    if type_name == "Dribble" and dribble_overrun:
        ev["dribble"] = {"overrun": True}
    return ev


def test_stage3_efficient_frontier_fc_1_shot_distance_from_raw_goal():
    # Raw (108, 40) converts to SPADL (94.5, 34): exactly 10.5 m from goal.
    # Raw goal (120, 40) converts to SPADL (105, 34): distance zero.
    shot = _ev(1, "Shot", loc=(108.0, 40.0), end=(120.0, 40.0))
    df = featurize_match([shot], match_id=1)
    row = df.iloc[0]
    assert row["start_x"] == pytest.approx(94.5)
    assert row["start_y"] == pytest.approx(34.0)
    assert row["dist_start"] == pytest.approx(10.5)
    assert row["angle_start"] == pytest.approx(0.0)
    assert row["end_x"] == pytest.approx(105.0)
    assert row["end_y"] == pytest.approx(34.0)
    assert row["dist_end"] == pytest.approx(0.0)


def test_stage3_efficient_frontier_fc_1_pass_start_end_converted():
    # Raw (60, 40) -> SPADL (52.5, 34); raw (84, 40) -> SPADL (73.5, 34).
    ev = _ev(1, "Pass", loc=(60.0, 40.0), end=(84.0, 40.0))
    df = featurize_match([ev], match_id=1)
    row = df.iloc[0]
    assert row["start_x"] == pytest.approx(52.5)
    assert row["start_y"] == pytest.approx(34.0)
    assert row["end_x"] == pytest.approx(73.5)
    assert row["end_y"] == pytest.approx(34.0)
    assert row["dx"] == pytest.approx(21.0)
    assert row["dy"] == pytest.approx(0.0)
    assert row["dist_start"] == pytest.approx(52.5)
    assert row["dist_end"] == pytest.approx(31.5)


def test_stage3_efficient_frontier_fc_1_every_action_type_same_frame():
    # The same raw location must give the same converted distance for
    # every action type, including types with no end-location block.
    for tname in ACTION_TYPES:
        kwargs = {}
        if tname in ("Pass", "Carry", "Shot"):
            kwargs["end"] = (108.0, 40.0)
        if tname == "Dribble":
            kwargs["dribble_overrun"] = True
        ev = _ev(1, tname, loc=(96.0, 40.0), **kwargs)
        df = featurize_match([ev], match_id=1)
        row = df.iloc[0]
        assert row["start_x"] == pytest.approx(84.0), tname
        assert row["start_y"] == pytest.approx(34.0), tname
        assert row["dist_start"] == pytest.approx(21.0), tname
        if tname in ("Pass", "Carry", "Shot"):
            assert row["end_x"] == pytest.approx(94.5), tname
            assert row["dist_end"] == pytest.approx(10.5), tname
        else:
            # No end location: the end state is the start state.
            assert row["end_x"] == pytest.approx(84.0), tname
            assert row["dist_end"] == pytest.approx(21.0), tname


def test_stage3_efficient_frontier_fc_1_bounds_and_corners():
    # Raw pitch corners must land exactly on the SPADL pitch corners,
    # and no converted coordinate may leave the 105x68 frame.
    corners = [(0.0, 0.0), (120.0, 0.0), (0.0, 80.0), (120.0, 80.0)]
    events = [_ev(i + 1, "Pass", loc=c, end=c) for i, c in enumerate(corners)]
    df = featurize_match(events, match_id=1)
    expected = [(0.0, 0.0), (105.0, 0.0), (0.0, 68.0), (105.0, 68.0)]
    for row, (ex, ey) in zip(df.itertuples(), expected):
        assert row.start_x == pytest.approx(ex)
        assert row.start_y == pytest.approx(ey)
        assert row.end_x == pytest.approx(ex)
        assert row.end_y == pytest.approx(ey)
    assert df["start_x"].between(0.0, 105.0).all()
    assert df["start_y"].between(0.0, 68.0).all()
    # Far corner (120, 80) -> (105, 68): distance is the goal-centre offset.
    assert df.iloc[3]["dist_start"] == pytest.approx(34.0)


def test_stage3_efficient_frontier_fc_1_angle_off_centre():
    # Raw (96, 56) -> SPADL (84, 47.6): dx 21, dy -13.6 from the goal.
    shot = _ev(1, "Shot", loc=(96.0, 56.0), end=(96.0, 56.0))
    df = featurize_match([shot], match_id=1)
    row = df.iloc[0]
    assert row["start_x"] == pytest.approx(84.0)
    assert row["start_y"] == pytest.approx(47.6)
    assert row["dist_start"] == pytest.approx(math.hypot(21.0, 13.6))
    assert row["angle_start"] == pytest.approx(math.degrees(math.atan2(13.6, 21.0)))


def test_stage3_efficient_frontier_fc_1_both_teams_same_goal_distance():
    # StatsBomb normalises the attacking direction, so a shot from raw
    # (108, 40) is 10.5 m out for either team, with no flip applied.
    events = [
        _ev(1, "Shot", team="A", loc=(108.0, 40.0), end=(120.0, 40.0)),
        _ev(2, "Shot", team="B", loc=(108.0, 40.0), end=(120.0, 40.0)),
    ]
    df = featurize_match(events, match_id=1)
    assert df["dist_start"].tolist() == pytest.approx([10.5, 10.5])
    assert df["dist_end"].tolist() == pytest.approx([0.0, 0.0])


def test_stage3_efficient_frontier_fc_1_feature_version_bumped():
    # The schema changed with the coordinate fix, so v2 VAEP caches must
    # never be reused: the pipeline cache key includes FEATURE_VERSION.
    assert FEATURE_VERSION == 3
