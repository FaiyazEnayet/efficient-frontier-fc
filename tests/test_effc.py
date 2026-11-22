"""Tests for Efficient Frontier FC. Fast, no data downloads."""

import numpy as np
import pandas as pd
import pytest

from effc.data.wages import norm, position_group
from effc.portfolio.portfolio import efficient_frontier, select_squad, select_xi
from effc.ratings.ratings import build_ratings
from effc.vaep.features import design_matrix, featurize_match


def _ev(idx, type_name, team="A", player="P1", pid=1, loc=(50.0, 34.0), minute=10):
    return {
        "id": str(idx),
        "index": idx,
        "period": 1,
        "timestamp": "00:10:00.000",
        "minute": minute,
        "second": 0,
        "type": {"id": 1, "name": type_name},
        "possession": 1,
        "possession_team": {"id": 1, "name": team},
        "team": {"id": 1, "name": team},
        "player": {"id": pid, "name": player},
        "location": list(loc),
    }


def test_featurize_labels_goal_within_k():
    # pass, then a shot that scores 3 actions later.
    # incl labels count the action's own goal; excl labels do not.
    events = [
        _ev(1, "Pass", loc=(60.0, 30.0)),
        _ev(2, "Carry", loc=(70.0, 30.0)),
        _ev(3, "Dribble", loc=(80.0, 34.0)),
    ]
    shot = _ev(4, "Shot", loc=(85.0, 34.0))
    shot["shot"] = {"outcome": {"name": "Goal"}, "end_location": [105.0, 34.0]}
    events.append(shot)
    df = featurize_match(events, match_id=1)
    assert len(df) == 4
    assert df.loc[0, "scores_incl"] == 1.0
    assert df.loc[0, "scores_excl"] == 1.0
    assert df.loc[3, "scores_incl"] == 1.0  # the goal counts for its own state
    assert df.loc[3, "scores_excl"] == 0.0  # but not for the after-state
    assert df.loc[0, "concedes_incl"] == 0.0


def test_featurize_concedes_opponent_goal():
    events = [_ev(1, "Pass", team="A", loc=(60.0, 30.0))]
    shot = _ev(2, "Shot", team="B", player="P9", pid=9, loc=(85.0, 34.0))
    shot["shot"] = {"outcome": {"name": "Goal"}, "end_location": [105.0, 34.0]}
    events.append(shot)
    df = featurize_match(events, match_id=1)
    assert df.loc[0, "concedes_incl"] == 1.0
    assert df.loc[0, "scores_incl"] == 0.0


def test_design_matrix_numeric():
    events = [_ev(1, "Pass"), _ev(2, "Carry")]
    df = featurize_match(events, match_id=1)
    X = design_matrix(df)
    assert X.shape[0] == 2
    assert X.dtypes.apply(lambda d: np.issubdtype(d, np.number)).all()


def test_vaep_values_shift():
    from effc.vaep.values import add_vaep

    events = [_ev(i, "Pass", loc=(50 + i, 34.0)) for i in range(1, 6)]
    df = featurize_match(events, match_id=7)

    class FakeModels(dict):
        pass

    # constant models: ps_after = 0.02 for all -> vaep_off == 0 except first row
    import effc.vaep.models as M

    orig = M.predict_proba
    M.predict_proba = lambda models, d: {k: np.full(len(d), v) for k, v in
        [("scores_incl", 0.02), ("scores_excl", 0.02),
         ("concedes_incl", 0.01), ("concedes_excl", 0.01)]}
    try:
        out = add_vaep(df, {})
    finally:
        M.predict_proba = orig
    assert np.allclose(out["vaep_off"].iloc[1:], 0.0)
    assert "vaep" in out.columns


def test_ratings_shrinkage():
    pm = pd.DataFrame(
        {
            "match_id": [1, 1, 1, 2, 2, 2],
            "team": ["A"] * 6,
            "player_id": [1, 2, 3, 1, 2, 3],
            "player": ["P1", "P2", "P3", "P1", "P2", "P3"],
            "vaep": [0.5, -0.1, -0.4, 0.3, 0.0, -0.2],
            "vaep_off": [0.4, 0.0, -0.3, 0.2, 0.0, -0.1],
            "vaep_def": [0.1, -0.1, -0.1, 0.1, 0.0, -0.1],
            "actions": [50, 40, 30, 45, 42, 33],
        }
    )
    mins = pd.DataFrame(
        {
            "match_id": [1, 1, 1, 2, 2, 2],
            "player_id": [1, 2, 3, 1, 2, 3],
            "minutes": [90.0] * 6,
            "position": ["Striker", "Midfielder", "Striker"] * 2,
        }
    )
    wmap = pd.DataFrame(
        {
            "player": ["P1", "P2", "P3"],
            "team": ["A"] * 3,
            "wage_eur": [100000.0, 50000.0, 40000.0],
            "value_eur": [50e6, 20e6, 10e6],
            "pos_group": ["FWD", "MID", "FWD"],
            "matched": [True] * 3,
        }
    )
    r = build_ratings(pm, mins, wmap, min_minutes=100, shrink_k=900)
    assert len(r) == 3
    # P1 raw = 0.4/90; group (FWD) mean is lower -> shrinkage pulls P1 down
    p1 = r[r["player"] == "P1"].iloc[0]
    assert p1["vaep_per90"] < p1["vaep_per90_raw"]
    assert p1["vaep_per90"] > r[r["player"] == "P3"]["vaep_per90"].iloc[0]
    assert p1["shrink_w"] == pytest.approx(180 / (180 + 900))


def _fake_ratings(n=30, seed=0):
    rng = np.random.default_rng(seed)
    groups = (["GK"] * 3 + ["DEF"] * 10 + ["MID"] * 10 + ["FWD"] * 7)[:n]
    return pd.DataFrame(
        {
            "player_id": np.arange(n),
            "player": [f"P{i}" for i in range(n)],
            "team": ["A"] * n,
            "pos_group": groups,
            "vaep_per90": rng.normal(0.05, 0.15, n),
            "wage_eur": rng.uniform(20_000, 300_000, n),
            "minutes": np.full(n, 900.0),
        }
    )


def test_select_squad_respects_budget_and_size():
    ratings = _fake_ratings()
    squad = select_squad(ratings, budget=5_000_000, squad_size=23)
    assert len(squad) == 23
    assert squad["wage_eur"].sum() <= 5_000_000 + 1e-6
    assert (squad["pos_group"] == "GK").sum() >= 2


def test_select_xi_formation():
    ratings = _fake_ratings()
    xi = select_xi(ratings, formation="4-3-3")
    counts = xi["pos_group"].value_counts()
    assert len(xi) == 11
    assert counts["GK"] == 1 and counts["DEF"] == 4
    assert counts["MID"] == 3 and counts["FWD"] == 3


def test_frontier_monotone_budgets():
    ratings = _fake_ratings()
    fr = efficient_frontier(ratings, np.array([4_000_000.0, 6_000_000.0, 9_000_000.0]))
    assert len(fr) == 3
    # more budget cannot hurt the optimum
    assert fr["exp_vaep_per90"].is_monotonic_increasing


def test_norm_and_positions():
    assert norm("Kylian Mbappé Lottin") == "kylian mbappe lottin"
    assert norm("E. Haaland") == "e haaland"
    assert position_group("ST, LW") == "FWD"
    assert position_group("CDM") == "MID"
    assert position_group("GK") == "GK"


def test_no_gk_pool_with_gk_quota_raises():
    # The reported defect: a pool with no goalkeepers and a GK minimum
    # must fail, not silently return players that violate the quota.
    from effc.portfolio.portfolio import InfeasibleError
    ratings = _fake_ratings()
    no_gk = ratings[ratings["pos_group"] != "GK"].reset_index(drop=True)
    assert (no_gk["pos_group"] == "GK").sum() == 0
    with pytest.raises(InfeasibleError) as exc:
        select_squad(no_gk, budget=5_000_000, squad_size=11,
                     quota={"GK": 1, "FWD": 1})
    assert exc.value.reason == InfeasibleError.EMPTY_GROUP


def test_quota_exceeding_squad_size_raises():
    from effc.portfolio.portfolio import InfeasibleError
    ratings = _fake_ratings()
    with pytest.raises(InfeasibleError) as exc:
        select_squad(ratings, budget=5_000_000, squad_size=11,
                     quota={"GK": 12})
    assert exc.value.reason == InfeasibleError.QUOTA_EXCEEDS_SQUAD


def test_quotas_summing_past_squad_size_raise():
    from effc.portfolio.portfolio import InfeasibleError
    ratings = _fake_ratings()
    with pytest.raises(InfeasibleError) as exc:
        select_squad(ratings, budget=5_000_000, squad_size=11,
                     quota={"GK": 6, "DEF": 6})
    assert exc.value.reason == InfeasibleError.QUOTAS_EXCEED_SQUAD


def test_feasible_squad_meets_all_constraints():
    ratings = _fake_ratings()
    quota = {"GK": 2, "DEF": 4, "MID": 4, "FWD": 2}
    squad = select_squad(ratings, budget=5_000_000, squad_size=23, quota=quota)
    assert len(squad) == 23
    assert squad["wage_eur"].sum() <= 5_000_000 + 1e-6
    counts = squad["pos_group"].value_counts()
    for g, q in quota.items():
        assert counts.get(g, 0) >= q, f"{g} minimum {q} not met"


def test_nan_wage_raises_clear_error():
    ratings = _fake_ratings()
    ratings.loc[0, "wage_eur"] = float("nan")
    with pytest.raises(ValueError, match="NaN in wage_eur"):
        select_squad(ratings, budget=5_000_000, squad_size=23)


def test_nan_rating_raises_clear_error():
    ratings = _fake_ratings()
    ratings.loc[0, "vaep_per90"] = float("nan")
    with pytest.raises(ValueError, match="NaN in vaep_per90"):
        select_squad(ratings, budget=5_000_000, squad_size=23)


def test_infeasible_budget_raises_typed_error():
    # A budget no 23-man squad can meet surfaces as infeasible, and the
    # typed error still satisfies `except RuntimeError` handlers.
    from effc.portfolio.portfolio import InfeasibleError
    ratings = _fake_ratings()
    with pytest.raises(InfeasibleError):
        select_squad(ratings, budget=1.0, squad_size=23)
    with pytest.raises(RuntimeError):
        select_squad(ratings, budget=1.0, squad_size=23)


# ---------------------------------------------------------------------------
# Stage 3 U06: formal action-value reference replication and attribution audit
# ---------------------------------------------------------------------------
# The repo VAEP formula and its goal-conceded debit are checked against an
# independent replication built on socceraction SPADL / Atomic-SPADL. The
# fixture is a real StatsBomb open-data excerpt: Bayer Leverkusen vs
# VfB Stuttgart, 2024-04-27, match 3895320, event indices 1990-2130
# (141 events, one Stuttgart goal at index 2003).


def _u06_excerpt_events():
    import json
    from pathlib import Path

    path = Path(__file__).parent / "fixtures" / "statsbomb" / "events_3895320_excerpt.json"
    with open(path) as f:
        return json.load(f)


def _u06_const_probs(monkeypatch_probs=None):
    """Stub predict_proba with constant probabilities, as existing tests do."""
    import effc.vaep.models as M

    orig = M.predict_proba
    M.predict_proba = lambda models, d: {
        k: np.full(len(d), v)
        for k, v in [
            ("scores_incl", 0.02),
            ("scores_excl", 0.02),
            ("concedes_incl", 0.01),
            ("concedes_excl", 0.01),
        ]
    }
    return orig


def test_stage3_efficient_frontier_fc_06_reference_formula_matches_add_vaep():
    """Known-answer fixture: hand-computed VAEP for a goal plus the debit.

    Constant probabilities (ps 0.02, pc 0.01) make every value hand
    computable: the goal action earns exactly +1, the turnover swap moves
    the before-state to the opponent view, and the goal-conceded debit
    lands exactly once on the conceding team's next action.
    """
    from effc.analysis.action_audit import reference_vaep
    from effc.vaep.values import add_vaep

    events = [
        _ev(1, "Pass", team="A", player="P1", pid=1, loc=(50.0, 34.0)),
        _ev(2, "Carry", team="A", player="P1", pid=1, loc=(60.0, 34.0)),
        _ev(3, "Shot", team="A", player="P1", pid=1, loc=(85.0, 34.0)),
        _ev(4, "Pass", team="B", player="P2", pid=2, loc=(50.0, 34.0)),
        _ev(5, "Pass", team="B", player="P2", pid=2, loc=(55.0, 34.0)),
    ]
    events[2]["shot"] = {"outcome": {"name": "Goal"}, "end_location": [105.0, 34.0]}
    df = featurize_match(events, match_id=7)
    assert df["is_goal"].tolist() == [False, False, True, False, False]

    orig = _u06_const_probs()
    try:
        valued = add_vaep(df, {})
    finally:
        import effc.vaep.models as M

        M.predict_proba = orig

    # Hand-computed: goal earns +1; row 4 (first B action after the goal)
    # carries the -1 debit on vaep_def; everything else is zero.
    assert np.allclose(valued["vaep_off"].to_numpy(), [0.0, 0.0, 1.0, 0.01, 0.0])
    assert np.allclose(valued["vaep_def"].to_numpy(), [0.0, 0.0, 0.0, -0.99, 0.0])
    assert np.allclose(valued["vaep"].to_numpy(), [0.0, 0.0, 1.0, -0.98, 0.0])

    # The independent re-implementation must agree row for row.
    ref = reference_vaep(valued)
    assert np.allclose(ref["vaep_off_ref"].to_numpy(), valued["vaep_off"].to_numpy())
    assert np.allclose(ref["vaep_def_ref"].to_numpy(), valued["vaep_def"].to_numpy())
    assert np.allclose(ref["vaep_ref"].to_numpy(), valued["vaep"].to_numpy())


def test_stage3_efficient_frontier_fc_06_invariants_hold_on_valued_frame():
    from effc.analysis.action_audit import check_vaep_invariants
    from effc.vaep.values import add_vaep

    events = [
        _ev(1, "Pass", team="A", player="P1", pid=1, loc=(50.0, 34.0)),
        _ev(2, "Carry", team="A", player="P1", pid=1, loc=(60.0, 34.0)),
        _ev(3, "Shot", team="A", player="P1", pid=1, loc=(85.0, 34.0)),
        _ev(4, "Pass", team="B", player="P2", pid=2, loc=(50.0, 34.0)),
        _ev(5, "Pass", team="B", player="P2", pid=2, loc=(55.0, 34.0)),
        _ev(6, "Shot", team="B", player="P2", pid=2, loc=(85.0, 34.0)),
        _ev(7, "Pass", team="A", player="P1", pid=1, loc=(50.0, 34.0)),
    ]
    events[2]["shot"] = {"outcome": {"name": "Goal"}, "end_location": [105.0, 34.0]}
    events[5]["shot"] = {"outcome": {"name": "Goal"}, "end_location": [105.0, 34.0]}
    df = featurize_match(events, match_id=8)
    orig = _u06_const_probs()
    try:
        valued = add_vaep(df, {})
    finally:
        import effc.vaep.models as M

        M.predict_proba = orig
    results = check_vaep_invariants(valued)
    failed = [r for r in results if not r["passed"]]
    assert not failed, failed
    names = {r["name"] for r in results}
    assert {"adds_up", "probabilities_in_range", "goal_realization",
            "debit_exactly_once_per_goal", "no_nan_values", "debit_total"} <= names


def test_stage3_efficient_frontier_fc_06_own_goal_debit_without_plus_one():
    """An own goal debits the conceding side but credits nobody with +1."""
    from effc.analysis.action_audit import reference_vaep
    from effc.vaep.values import add_vaep

    events = [
        _ev(1, "Pass", team="A", player="P1", pid=1, loc=(50.0, 34.0)),
        _ev(2, "Carry", team="A", player="P1", pid=1, loc=(60.0, 34.0)),
        _ev(3, "Own Goal For", team="B", player="P2", pid=2, loc=(90.0, 34.0)),
        _ev(4, "Pass", team="A", player="P1", pid=1, loc=(50.0, 34.0)),
        _ev(5, "Pass", team="A", player="P1", pid=1, loc=(55.0, 34.0)),
    ]
    df = featurize_match(events, match_id=9)
    assert len(df) == 4  # own goals add no action rows
    assert df.attrs["own_goals"] == [(2, "B", 1)]
    orig = _u06_const_probs()
    try:
        valued = add_vaep(df, {})
    finally:
        import effc.vaep.models as M

        M.predict_proba = orig
    # No action scored, so no +1 anywhere; the debit lands once, on the
    # first conceding-team (A) action after the own goal.
    assert valued["vaep_off"].max() <= 0.011
    assert np.allclose(valued["vaep_def"].to_numpy(), [0.0, 0.0, -1.0, 0.0])
    ref = reference_vaep(valued)
    assert np.allclose(ref["vaep_ref"].to_numpy(), valued["vaep"].to_numpy())


def test_stage3_efficient_frontier_fc_06_flatten_roundtrip():
    from effc.analysis.action_audit import flatten_statsbomb_events

    events = _u06_excerpt_events()
    flat = flatten_statsbomb_events(events, match_id=3895320)
    assert len(flat) == len(events)
    assert flat["event_id"].tolist() == [e["id"] for e in events]
    assert flat["type_name"].tolist() == [e["type"]["name"] for e in events]
    assert set(flat["team_id"]) == {904, 174}
    assert (flat["game_id"] == 3895320).all()


def test_stage3_efficient_frontier_fc_06_coverage_table_on_excerpt():
    from effc.analysis.action_audit import coverage_table
    from effc.vaep.features import ACTION_TYPES

    events = _u06_excerpt_events()
    table = coverage_table(events, match_id=3895320, home_team_id=904)
    assert {"event_type", "raw_count", "repo_rows", "spadl_actions",
            "matched", "repo_only", "spadl_only"} <= set(table.columns)
    # Reconciliation identities: every row and action is accounted for.
    assert table["matched"].sum() + table["repo_only"].sum() == table["repo_rows"].sum()
    assert table["matched"].sum() + table["spadl_only"].sum() == table["spadl_actions"].sum()
    assert table["repo_rows"].sum() == sum(
        1 for e in events if e["type"]["name"] in ACTION_TYPES
    )
    assert table["matched"].sum() > 50
    by_type = table.set_index("event_type")
    # The repo values Ball Recovery, Block and non-tackle Duels; the
    # reference SPADL pipeline drops them as non-actions.
    assert by_type.loc["Ball Recovery", "repo_only"] == by_type.loc["Ball Recovery", "repo_rows"]
    assert by_type.loc["Block", "repo_only"] == by_type.loc["Block", "repo_rows"]
    # The reference values fouls as actions; the repo does not value them.
    assert by_type.loc["Foul Committed", "spadl_only"] == by_type.loc["Foul Committed", "spadl_actions"]
    assert by_type.loc["Foul Committed", "repo_rows"] == 0
    # Shots and passes match on both sides.
    assert by_type.loc["Shot", "matched"] == by_type.loc["Shot", "repo_rows"]
    assert by_type.loc["Pass", "matched"] > 30


def test_stage3_efficient_frontier_fc_06_orientation_mirror_beats_shared_frame():
    """StatsBomb coordinates are in the acting team's attacking frame.

    Across a possession change, the next action starts at the mirror of
    the previous action's end, not at the shared-frame point. The
    reference pipeline assumes a fixed frame, so this decides which
    conversion is right for this data.
    """
    from effc.analysis.action_audit import orientation_check

    events = _u06_excerpt_events()
    res = orientation_check(events)
    assert res["turnover_pairs"] >= 5
    assert res["median_mirror_m"] < res["median_shared_m"]
    assert res["share_mirror_closer"] > 0.9
    # Every team that shoots in the excerpt shoots from the same end of
    # the raw frame (the excerpt's shots are all Stuttgart's).
    assert res["shot_x_min"], "excerpt has no shots at all"
    for team, xmin in res["shot_x_min"].items():
        assert xmin > 80.0, team


def test_stage3_efficient_frontier_fc_06_spadl_away_shots_at_wrong_end():
    """The reference direction fix mirrors every away-team action.

    Away shots sit at raw x 91-114, so the repo puts them at the attacked
    goal; the reference flips them to x 5-25, the wrong end of the pitch.
    """
    from effc.analysis.action_audit import attach_event_ids, spadl_actions
    from effc.vaep.features import featurize_match

    events = _u06_excerpt_events()
    repo = attach_event_ids(featurize_match(events, match_id=3895320), events)
    spadl = spadl_actions(events, match_id=3895320, home_team_id=904)
    away_spadl_shots = spadl[(spadl["team_id"] == 174) & (spadl["type_name"] == "shot")]
    assert len(away_spadl_shots) >= 2
    assert away_spadl_shots["start_x"].median() < 30.0
    away_repo_shots = repo[(repo["team"] == "VfB Stuttgart") & (repo["action"] == "Shot")]
    assert len(away_repo_shots) >= 2
    assert away_repo_shots["start_x"].median() > 80.0


def test_stage3_efficient_frontier_fc_06_spadl_home_y_mirrored_repo_y():
    """The reference also mirrors y; the repo does not.

    For home-team actions x agrees and y is mirrored (68 - y); for
    away-team actions the two flips cancel on y but x is mirrored.
    """
    from effc.analysis.action_audit import attach_event_ids, spadl_actions
    from effc.vaep.features import featurize_match

    events = _u06_excerpt_events()
    repo = attach_event_ids(featurize_match(events, match_id=3895320), events)
    spadl = spadl_actions(events, match_id=3895320, home_team_id=904)
    m = spadl.merge(repo, left_on="original_event_id", right_on="event_id",
                    suffixes=("_sa", "_repo"))
    home = m[m["team_id"] == 904]
    assert len(home) > 30
    assert (home["start_x_sa"] - home["start_x_repo"]).abs().median() < 0.5
    assert (home["start_y_sa"] - (68.0 - home["start_y_repo"])).abs().median() < 0.5
    assert (home["start_y_sa"] - home["start_y_repo"]).abs().median() > 10.0


def test_stage3_efficient_frontier_fc_06_atomic_receival_credit():
    """Atomic-SPADL splits actions and adds receival rows the repo lacks.

    A completed pass becomes a pass plus a receival for the receiver, and
    a scored goal becomes its own atomic goal row. The repo credits
    neither the receiver nor the goal as separate actions.
    """
    from effc.analysis.action_audit import atomic_spadl_actions, spadl_actions

    events = _u06_excerpt_events()
    spadl = spadl_actions(events, match_id=3895320, home_team_id=904)
    atomic = atomic_spadl_actions(spadl)
    assert len(atomic) > len(spadl)
    names = atomic["type_name"]
    assert (names == "receival").sum() > 20
    assert (names == "goal").sum() == 1
    # Every atomic row traces to a real SPADL action, not thin air.
    assert set(atomic["original_event_id"]) <= set(spadl["original_event_id"])


def test_stage3_efficient_frontier_fc_06_disagreement_distribution_on_excerpt():
    """Matched values disagree exactly where the pipelines diverge.

    Both pipelines share the repo probability source here, so the
    disagreement isolates the transformation: actions the reference
    drops shift the before-state and the lookahead window of the actions
    that remain.
    """
    from effc.analysis.action_audit import audit_match
    from effc.vaep.models import train_goal_models
    from effc.vaep.values import add_vaep
    from effc.vaep.features import featurize_match

    events = _u06_excerpt_events()
    frame = featurize_match(events, match_id=3895320)
    models = train_goal_models([frame], seed=0)
    valued = add_vaep(frame, models)
    result = audit_match(events, match_id=3895320, home_team_id=904,
                         repo_valued=valued)
    dis = result["disagreement"]
    assert dis["n_matched"] > 50
    assert dis["max_abs_diff"] > 0.005
    assert dis["share_gt_0.01"] > 0.0
    assert set(dis) >= {"n_matched", "mean_abs_diff", "median_abs_diff",
                        "p90_abs_diff", "max_abs_diff", "share_gt_0.01",
                        "share_gt_0.05", "share_gt_0.10"}
    # Per-player aggregates differ for at least one player: the dropped
    # actions carried value that the reference never sees.
    players = result["player_aggregates"]
    assert (players["diff"].abs() > 1e-9).any()
    assert result["coverage"]["matched"].sum() == dis["n_matched"]


def test_stage3_efficient_frontier_fc_06_empty_match_returns_empty():
    from effc.analysis.action_audit import (
        audit_match,
        coverage_table,
        orientation_check,
        reference_vaep,
    )

    assert coverage_table([], match_id=1, home_team_id=1).empty
    assert reference_vaep(pd.DataFrame()).empty
    assert orientation_check([])["turnover_pairs"] == 0
    result = audit_match([], match_id=1, home_team_id=1, repo_valued=pd.DataFrame())
    assert result["disagreement"]["n_matched"] == 0
