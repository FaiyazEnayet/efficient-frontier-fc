"""Stage 3 efficient-frontier-fc U05: real-player entity matching and
selection coverage audit.

C6 rebuilt the matcher and U06 audited action values; this unit covers
what neither of them did: a player ID crosswalk, confidence categories
for every match type, a reviewed mapping registry that records human
decisions on the uncertain matches, per-position coverage measured from
the event side, and the optimization impact of the exclusions, measured
by re-solving squads under wage-interval and proxy-inclusion scenarios.

The committed registry (data/mapping_registry.csv) holds 43 reviewed
rows: the 27 fuzzy matches, Max Meyer (ambiguous, resolved to the FC
Luzern row), 12 same-club name-variant matches plus the second
Choupo-Moting pool row, and Dante twice (both pool rows to the Nice
row). The real-pool numbers these tests pin behavior for are published
in docs/entity_matching_audit.md; the tests below run on synthetic
frames and the committed registry, never on the uncommitted wage file.
"""

import os

import numpy as np
import pandas as pd
import pytest

from effc.analysis.entity_audit import (
    MATCH_CONFIDENCE,
    SB_POSITION_GROUPS,
    apply_mapping_registry,
    build_player_crosswalk,
    coverage_by_position,
    load_mapping_registry,
    match_confidence,
    selection_sensitivity,
    statsbomb_position_group,
    wage_interval_sensitivity,
    with_match_confidence,
)
from effc.data.wages import (
    MATCH_TYPES,
    WAGE_INTERVAL_HALF_WIDTH,
    score_gold_set,
    summarize_wage_matches,
    wage_uncertainty_intervals,
)
from effc.portfolio.portfolio import select_squad

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _matched_frame(rows):
    return pd.DataFrame(rows)


def _base_matched():
    # One player at two clubs (a transfer, same person), one fuzzy
    # match, one unmatched row and one quarantined row.
    return _matched_frame([
        {"player_id": 1, "player": "Aaron Ramsey", "team": "Burnley",
         "person_key": "aaron ramsey", "wage_row_id": "fc24:aaa",
         "wage_eur": 18000.0, "value_eur": 3800000.0, "pos_group": "MID",
         "fc_club": "Burnley", "match_type": "exact", "matched": True,
         "match_note": "", "wage_source": "EA FC 24 (fc24_wages.csv)",
         "wage_source_sha256": "sha", "wage_source_mtime": "2026-01-01"},
        {"player_id": 1, "player": "Aaron Ramsey", "team": "Cardiff City",
         "person_key": "aaron ramsey", "wage_row_id": "fc24:bbb",
         "wage_eur": 17000.0, "value_eur": 3300000.0, "pos_group": "MID",
         "fc_club": "Cardiff City", "match_type": "exact", "matched": True,
         "match_note": "", "wage_source": "EA FC 24 (fc24_wages.csv)",
         "wage_source_sha256": "sha", "wage_source_mtime": "2026-01-01"},
        {"player_id": 2, "player": "Yussuf Yurary Poulsen",
         "team": "RB Leipzig", "person_key": "yussuf yurary poulsen",
         "wage_row_id": "fc24:ccc", "wage_eur": 53000.0,
         "value_eur": 10500000.0, "pos_group": "FWD",
         "fc_club": "RB Leipzig", "match_type": "fuzzy", "matched": True,
         "match_note": "", "wage_source": "EA FC 24 (fc24_wages.csv)",
         "wage_source_sha256": "sha", "wage_source_mtime": "2026-01-01"},
        {"player_id": 3, "player": "No Such Player", "team": "Test FC",
         "person_key": "no such player", "wage_row_id": pd.NA,
         "wage_eur": float("nan"), "value_eur": float("nan"),
         "pos_group": pd.NA, "fc_club": pd.NA, "match_type": "unmatched",
         "matched": False, "match_note": "",
         "wage_source": pd.NA, "wage_source_sha256": pd.NA,
         "wage_source_mtime": pd.NA},
        {"player_id": 4, "player": "Max Meyer", "team": "Schalke 04",
         "person_key": "max meyer", "wage_row_id": pd.NA,
         "wage_eur": float("nan"), "value_eur": float("nan"),
         "pos_group": pd.NA, "fc_club": pd.NA, "match_type": "ambiguous",
         "matched": False, "match_note": "homonym",
         "wage_source": pd.NA, "wage_source_sha256": pd.NA,
         "wage_source_mtime": pd.NA},
    ])


def _wages_frame():
    return pd.DataFrame([
        {"long_name": "Test Player", "short_name": "T. Player",
         "club_name": "Other FC", "positions": "CM", "age": 25,
         "overall": 75, "value_eur": 5000000.0, "wage_eur": 15000.0,
         "key": "test player", "club_key": "other fc",
         "club_norm": "other", "pos_group": "MID",
         "person_key": "test player", "wage_row_id": "fc24:test1",
         "wage_source": "EA FC 24 (fc24_wages.csv)",
         "wage_source_sha256": "sha", "wage_source_mtime": "2026-01-01"},
        {"long_name": "Max Meyer", "short_name": "M. Meyer",
         "club_name": "FC Luzern", "positions": "CAM, CM", "age": 27,
         "overall": 74, "value_eur": 1000000.0, "wage_eur": 11000.0,
         "key": "max meyer", "club_key": "fc luzern",
         "club_norm": "luzern", "pos_group": "MID",
         "person_key": "max meyer", "wage_row_id": "fc24:luzern",
         "wage_source": "EA FC 24 (fc24_wages.csv)",
         "wage_source_sha256": "sha", "wage_source_mtime": "2026-01-01"},
    ])


def _registry_frame(rows):
    return pd.DataFrame(rows, columns=[
        "player", "team", "player_id", "prior_match_type", "decision",
        "wage_row_id", "wage_eur", "evidence", "reviewer",
        "reviewed_date",
    ])


def test_stage3_efficient_frontier_fc_05_confidence_all_types():
    # Every match type, including the new "reviewed" type, maps to
    # exactly one confidence category; an unknown type fails closed.
    assert set(MATCH_CONFIDENCE) == set(MATCH_TYPES)
    assert match_confidence("exact") == "high"
    assert match_confidence("exact_other_club") == "medium"
    assert match_confidence("reviewed") == "medium"
    assert match_confidence("alias") == "low"
    assert match_confidence("fuzzy") == "low"
    assert match_confidence("ambiguous") == "uncertain"
    assert match_confidence("proxy") == "proxy"
    assert match_confidence("unmatched") == "none"
    with pytest.raises(ValueError, match="unknown match_type"):
        match_confidence("bogus")
    # The reviewed interval is the other-club interval: the identity is
    # verified, but the wage still comes from another club and season.
    assert WAGE_INTERVAL_HALF_WIDTH["reviewed"] == 0.15
    assert WAGE_INTERVAL_HALF_WIDTH["exact"] == 0.05
    assert WAGE_INTERVAL_HALF_WIDTH["fuzzy"] == 0.25


def test_stage3_efficient_frontier_fc_05_with_match_confidence():
    frame = with_match_confidence(_base_matched())
    by_player = dict(zip(frame["player"], frame["confidence"]))
    assert by_player["Aaron Ramsey"] == "high"
    assert by_player["Yussuf Yurary Poulsen"] == "low"
    assert by_player["No Such Player"] == "none"
    assert by_player["Max Meyer"] == "uncertain"
    with pytest.raises(ValueError, match="match_type"):
        with_match_confidence(pd.DataFrame([{"player": "x"}]))


def test_stage3_efficient_frontier_fc_05_crosswalk_transfer_one_row():
    cross = build_player_crosswalk(_base_matched())
    # Two player IDs have a wage row; the unmatched and quarantined
    # rows are excluded and counted, not silently dropped.
    assert len(cross) == 2
    assert set(cross["player_id"]) == {1, 2}
    ramsey = cross[cross["player_id"] == 1].iloc[0]
    assert ramsey["person_key"] == "aaron ramsey"
    assert ramsey["n_teams"] == 2
    assert ramsey["wage_row_ids"] == "fc24:aaa; fc24:bbb"
    assert ramsey["confidence"] == "high"
    report = cross.attrs["player_crosswalk_report"]
    assert report["n_player_rows"] == 5
    assert report["n_player_ids"] == 2
    assert report["n_excluded_no_wage_row"] == 2
    assert report["n_player_ids_multi_club"] == 1


def test_stage3_efficient_frontier_fc_05_crosswalk_conflicts_fail_closed():
    frame = _base_matched()
    frame.loc[0, "person_key"] = "aaron ramsey"
    frame.loc[1, "person_key"] = "someone else"
    with pytest.raises(ValueError, match="more than one person_key"):
        build_player_crosswalk(frame)
    frame = _base_matched()
    frame.loc[1, "player"] = "Aaron Smith"
    with pytest.raises(ValueError, match="more than one normalized name"):
        build_player_crosswalk(frame)
    frame = _base_matched()
    frame.loc[0, "wage_row_id"] = pd.NA
    with pytest.raises(ValueError, match="without a wage row ID"):
        build_player_crosswalk(frame)
    with pytest.raises(ValueError, match="needs a player_id column"):
        build_player_crosswalk(pd.DataFrame([{"player": "x"}]))


def test_stage3_efficient_frontier_fc_05_registry_loads_committed():
    reg = load_mapping_registry()
    assert len(reg) == 43
    assert set(reg["decision"]) == {"accept"}
    assert not reg.duplicated(subset=["player", "team"]).any()
    assert set(reg["prior_match_type"]) == {"fuzzy", "ambiguous", "unmatched"}
    assert (reg["prior_match_type"] == "fuzzy").sum() == 27
    assert (reg["prior_match_type"] == "ambiguous").sum() == 1
    assert (reg["prior_match_type"] == "unmatched").sum() == 15
    meyer = reg[reg["player"] == "Max Meyer"].iloc[0]
    assert meyer["team"] == "Schalke 04"
    assert meyer["wage_row_id"] == "fc24:f92b143a8134"
    assert meyer["wage_eur"] == 11000.0
    dante = reg[reg["player"] == "Dante Bonfim da Costa Santos"]
    assert len(dante) == 2
    assert dante["wage_row_id"].nunique() == 1
    assert set(dante["team"]) == {"Bayern Munich", "Wolfsburg"}
    choupo = reg[reg["player"] == "Jean-Eric Maxim Choupo-Moting"]
    assert len(choupo) == 2
    assert choupo["wage_row_id"].nunique() == 1
    assert choupo["player_id"].nunique() == 1
    assert reg["evidence"].str.strip().str.len().min() > 0
    assert reg["wage_row_id"].str.startswith("fc24:").all()


def test_stage3_efficient_frontier_fc_05_registry_validation(tmp_path):
    good = _registry_frame([{
        "player": "Test Player", "team": "Test FC", "player_id": 1,
        "prior_match_type": "unmatched", "decision": "accept",
        "wage_row_id": "fc24:test1", "wage_eur": 15000.0,
        "evidence": "same club, only candidate", "reviewer": "review",
        "reviewed_date": "2026-11-23",
    }])
    path = tmp_path / "registry.csv"
    good.to_csv(path, index=False)
    assert len(load_mapping_registry(str(path))) == 1

    def _expect_error(frame, match):
        bad_path = tmp_path / "bad.csv"
        frame.to_csv(bad_path, index=False)
        with pytest.raises(ValueError, match=match):
            load_mapping_registry(str(bad_path))

    dup = pd.concat([good, good], ignore_index=True)
    _expect_error(dup, "more than one decision")
    bad = good.copy()
    bad.loc[0, "decision"] = "maybe"
    _expect_error(bad, "accept or reject")
    bad = good.copy()
    bad.loc[0, "evidence"] = ""
    _expect_error(bad, "evidence is required")
    bad = good.copy()
    bad.loc[0, "wage_row_id"] = ""
    _expect_error(bad, "an accept needs a wage_row_id")
    bad = good.copy()
    bad.loc[0, "wage_eur"] = 0.0
    _expect_error(bad, "a positive wage_eur")
    bad = good.drop(columns=["reviewer"])
    _expect_error(bad, "missing columns")


def test_stage3_efficient_frontier_fc_05_registry_apply_accept_reject():
    matched = _matched_frame([
        {"player_id": 1, "player": "Test Player", "team": "Test FC",
         "person_key": "test player", "wage_row_id": pd.NA,
         "wage_eur": float("nan"), "value_eur": float("nan"),
         "pos_group": pd.NA, "fc_club": pd.NA, "match_type": "unmatched",
         "matched": False, "match_note": "",
         "wage_source": pd.NA, "wage_source_sha256": pd.NA,
         "wage_source_mtime": pd.NA},
        {"player_id": 2, "player": "Yussuf Yurary Poulsen",
         "team": "RB Leipzig", "person_key": "yussuf yurary poulsen",
         "wage_row_id": "fc24:ccc", "wage_eur": 53000.0,
         "value_eur": 10500000.0, "pos_group": "FWD",
         "fc_club": "RB Leipzig", "match_type": "fuzzy", "matched": True,
         "match_note": "", "wage_source": "EA FC 24 (fc24_wages.csv)",
         "wage_source_sha256": "sha", "wage_source_mtime": "2026-01-01"},
    ])
    registry = _registry_frame([
        {"player": "Test Player", "team": "Test FC", "player_id": 1,
         "prior_match_type": "unmatched", "decision": "accept",
         "wage_row_id": "fc24:test1", "wage_eur": 15000.0,
         "evidence": "same club, only candidate", "reviewer": "review",
         "reviewed_date": "2026-11-23"},
        {"player": "Yussuf Yurary Poulsen", "team": "RB Leipzig",
         "player_id": 2, "prior_match_type": "fuzzy", "decision": "reject",
         "wage_row_id": "", "wage_eur": float("nan"),
         "evidence": "different person after all", "reviewer": "review",
         "reviewed_date": "2026-11-23"},
    ])
    out = apply_mapping_registry(matched, registry, _wages_frame())
    accepted = out[out["player"] == "Test Player"].iloc[0]
    assert accepted["match_type"] == "reviewed"
    assert accepted["matched"]
    assert accepted["wage_eur"] == 15000.0
    # The wage row ID comes from the registry value, not from the wage
    # row (where it is the index, not a column).
    assert accepted["wage_row_id"] == "fc24:test1"
    assert accepted["fc_club"] == "Other FC"
    assert accepted["pos_group"] == "MID"
    assert accepted["person_key"] == "test player"
    assert accepted["prior_match_type"] == "unmatched"
    assert accepted["match_note"].startswith("reviewed: ")
    rejected = out[out["player"] == "Yussuf Yurary Poulsen"].iloc[0]
    assert rejected["match_type"] == "ambiguous"
    assert not rejected["matched"]
    assert pd.isna(rejected["wage_eur"])
    assert rejected["prior_match_type"] == "fuzzy"
    # A registry row for a player the frame never saw fails closed,
    # as does an accept naming a wage row the file does not hold, and
    # a registry wage that disagrees with the file.
    ghost = _registry_frame([{
        "player": "Ghost", "team": "Test FC", "player_id": 9,
        "prior_match_type": "unmatched", "decision": "accept",
        "wage_row_id": "fc24:test1", "wage_eur": 15000.0,
        "evidence": "no such player", "reviewer": "review",
        "reviewed_date": "2026-11-23",
    }])
    with pytest.raises(ValueError, match="has no matching"):
        apply_mapping_registry(matched, ghost, _wages_frame())
    bad_id = _registry_frame([{
        "player": "Test Player", "team": "Test FC", "player_id": 1,
        "prior_match_type": "unmatched", "decision": "accept",
        "wage_row_id": "fc24:nope", "wage_eur": 15000.0,
        "evidence": "wrong id", "reviewer": "review",
        "reviewed_date": "2026-11-23",
    }])
    with pytest.raises(ValueError, match="not in the wage table"):
        apply_mapping_registry(matched, bad_id, _wages_frame())
    bad_wage = _registry_frame([{
        "player": "Test Player", "team": "Test FC", "player_id": 1,
        "prior_match_type": "unmatched", "decision": "accept",
        "wage_row_id": "fc24:test1", "wage_eur": 99999.0,
        "evidence": "wrong wage", "reviewer": "review",
        "reviewed_date": "2026-11-23",
    }])
    with pytest.raises(ValueError, match="disagrees with the wage file"):
        apply_mapping_registry(matched, bad_wage, _wages_frame())


def test_stage3_efficient_frontier_fc_05_position_groups():
    # The 23 StatsBomb position names in the target pool map exactly;
    # anything else is unknown, never guessed.
    expected = {
        "Goalkeeper": "GK",
        "Center Back": "DEF", "Left Center Back": "DEF",
        "Right Center Back": "DEF", "Left Back": "DEF",
        "Right Back": "DEF", "Left Wing Back": "DEF",
        "Right Wing Back": "DEF",
        "Center Defensive Midfield": "MID",
        "Left Defensive Midfield": "MID",
        "Right Defensive Midfield": "MID", "Center Midfield": "MID",
        "Left Center Midfield": "MID", "Right Center Midfield": "MID",
        "Left Midfield": "MID", "Right Midfield": "MID",
        "Center Attacking Midfield": "MID",
        "Left Attacking Midfield": "MID",
        "Right Attacking Midfield": "MID",
        "Left Wing": "FWD", "Right Wing": "FWD",
        "Center Forward": "FWD", "Left Center Forward": "FWD",
        "Right Center Forward": "FWD",
    }
    assert SB_POSITION_GROUPS == expected
    for name, group in expected.items():
        assert statsbomb_position_group(name) == group
    assert statsbomb_position_group("Sweeper") == "unknown"
    assert statsbomb_position_group(None) == "unknown"
    assert statsbomb_position_group(float("nan")) == "unknown"


def test_stage3_efficient_frontier_fc_05_coverage_by_position():
    matched = _matched_frame([
        {"player_id": 1, "player": "A", "team": "T1", "matched": True},
        {"player_id": 2, "player": "B", "team": "T1", "matched": True},
        {"player_id": 3, "player": "C", "team": "T1", "matched": False},
        {"player_id": 4, "player": "D", "team": "T1", "matched": False},
        {"player_id": 5, "player": "E", "team": "T1", "matched": True},
        {"player_id": 6, "player": "F", "team": "T1", "matched": False},
    ])
    positions = pd.Series(
        ["Goalkeeper", "Center Back", "Center Forward", "Left Wing",
         "Center Midfield", None],
        index=[1, 2, 3, 4, 5, 6])
    result = coverage_by_position(matched, positions)
    assert result["n_players"] == 6
    assert result["n_matched"] == 3
    assert result["match_rate"] == pytest.approx(0.5)
    assert result["by_position"]["GK"] == {
        "n": 1, "matched": 1, "match_rate": 1.0}
    assert result["by_position"]["DEF"] == {
        "n": 1, "matched": 1, "match_rate": 1.0}
    assert result["by_position"]["FWD"] == {
        "n": 2, "matched": 0, "match_rate": 0.0}
    assert result["by_position"]["MID"] == {
        "n": 1, "matched": 1, "match_rate": 1.0}
    assert result["by_position"]["unknown"] == {
        "n": 1, "matched": 0, "match_rate": 0.0}
    # A DataFrame of per-match positions is reduced to the modal one.
    frame_positions = pd.DataFrame([
        {"player_id": 1, "position": "Goalkeeper"},
        {"player_id": 1, "position": "Goalkeeper"},
        {"player_id": 2, "position": "Center Back"},
    ])
    sub = coverage_by_position(
        matched[matched["player_id"].isin([1, 2])], frame_positions)
    assert sub["by_position"]["GK"]["n"] == 1
    with pytest.raises(ValueError, match="player_id and matched"):
        coverage_by_position(pd.DataFrame([{"player": "x"}]), positions)


def _ratings_frame():
    # Six players with distinct values so the MILP optimum is unique.
    return pd.DataFrame([
        {"player_id": 1, "player": "GK One", "team": "T1",
         "pos_group": "GK", "vaep_per90": 0.30, "wage_eur": 10000.0,
         "match_type": "exact"},
        {"player_id": 2, "player": "GK Two", "team": "T1",
         "pos_group": "GK", "vaep_per90": 0.20, "wage_eur": 5000.0,
         "match_type": "exact"},
        {"player_id": 3, "player": "DEF One", "team": "T1",
         "pos_group": "DEF", "vaep_per90": 0.40, "wage_eur": 20000.0,
         "match_type": "exact"},
        {"player_id": 4, "player": "DEF Two", "team": "T1",
         "pos_group": "DEF", "vaep_per90": 0.10, "wage_eur": 3000.0,
         "match_type": "fuzzy"},
        {"player_id": 5, "player": "MID One", "team": "T1",
         "pos_group": "MID", "vaep_per90": 0.50, "wage_eur": 30000.0,
         "match_type": "exact_other_club"},
        {"player_id": 6, "player": "MID Two", "team": "T1",
         "pos_group": "MID", "vaep_per90": 0.45, "wage_eur": 25000.0,
         "match_type": "exact"},
    ])


def test_stage3_efficient_frontier_fc_05_selection_sensitivity():
    # Six players, pick 3 with one of each group, budget 40000. The
    # base optimum is GK One + DEF Two + MID Two (0.30 + 0.10 + 0.45 =
    # 0.85, bill 38000); after MID Two gets cheaper and better, the
    # optimum is GK One + DEF One + MID Two (0.30 + 0.40 + 0.60 = 1.30).
    quota = {"GK": 1, "DEF": 1, "MID": 1}
    base = _ratings_frame()
    alt = base.copy()
    alt.loc[alt["player_id"] == 6, "wage_eur"] = 10000.0
    alt.loc[alt["player_id"] == 6, "vaep_per90"] = 0.60
    result = selection_sensitivity(
        base, alt, [40000.0], value_col="vaep_per90",
        squad_size=3, quota=quota)
    cell = result["budgets"][40000.0]
    assert "error" not in cell
    assert cell["base_objective"] == pytest.approx(0.85)
    assert cell["alt_objective"] == pytest.approx(1.30)
    assert cell["objective_delta"] == pytest.approx(0.45)
    assert cell["jaccard"] == pytest.approx(2 / 4)
    assert [p["player"] for p in cell["entering"]] == ["DEF One"]
    assert [p["player"] for p in cell["leaving"]] == ["DEF Two"]
    # An infeasible budget is reported, not raised and not skipped.
    result = selection_sensitivity(base, alt, [1.0], squad_size=3,
                                   quota=quota)
    assert "error" in result["budgets"][1.0]
    # Identical frames change nothing.
    result = selection_sensitivity(base, base.copy(), [40000.0],
                                   squad_size=3, quota=quota)
    assert result["budgets"][40000.0]["jaccard"] == 1.0
    assert result["budgets"][40000.0]["objective_delta"] == 0.0
    with pytest.raises(ValueError, match="need a wage_eur column"):
        selection_sensitivity(base.drop(columns=["wage_eur"]), alt,
                              [40000.0], squad_size=3, quota=quota)


def test_stage3_efficient_frontier_fc_05_wage_interval_sensitivity():
    quota = {"GK": 1, "DEF": 1, "MID": 1}
    ratings = _ratings_frame()
    result = wage_interval_sensitivity(ratings, [40000.0],
                                      squad_size=3, quota=quota)
    # Oracle: the low and high re-solves must match fresh select_squad
    # calls on the same interval wages.
    with_intervals = wage_uncertainty_intervals(ratings)
    for side in ("low", "high"):
        col = f"wage_{side}"
        alt = with_intervals.copy()
        alt["wage_eur"] = with_intervals[col]
        expected = select_squad(alt, 40000.0, squad_size=3, quota=quota)
        cell = result[side]["budgets"][40000.0]
        assert "error" not in cell
        assert cell["alt_objective"] == pytest.approx(
            float(expected["vaep_per90"].sum()))
        assert cell["alt_wage_bill"] == pytest.approx(
            float(expected["wage_eur"].sum()))
    # The low-wage squad is worth at least the point squad, and the
    # high-wage squad at most it, on this frame.
    base_obj = float(select_squad(ratings, 40000.0, squad_size=3,
                                  quota=quota)["vaep_per90"].sum())
    assert result["low"]["budgets"][40000.0]["alt_objective"] >= base_obj
    assert result["high"]["budgets"][40000.0]["alt_objective"] <= base_obj
    # A type with no interval half width fails closed.
    bad = ratings.copy()
    bad.loc[0, "match_type"] = "bogus"
    with pytest.raises(ValueError, match="without a wage interval"):
        wage_interval_sensitivity(bad, [40000.0], squad_size=3,
                                  quota=quota)


def test_stage3_efficient_frontier_fc_05_gold_set_with_reviewed():
    # The gold-set scorer flows the new type through unchanged: a
    # reviewed row scores like any other matched row, and a corrupt
    # expected wage still drops precision.
    gold = pd.DataFrame([
        {"player": "Kevin De Bruyne", "team": "Manchester City",
         "expected_wage_eur": 350000.0, "expected_match_type": "exact"},
        {"player": "Max Meyer", "team": "Schalke 04",
         "expected_wage_eur": 11000.0, "expected_match_type": "reviewed"},
        {"player": "No Such Player", "team": "Test FC",
         "expected_wage_eur": np.nan, "expected_match_type": "unmatched"},
    ])
    matched = pd.DataFrame([
        {"player": "Kevin De Bruyne", "team": "Manchester City",
         "wage_eur": 350000.0, "match_type": "exact"},
        {"player": "Max Meyer", "team": "Schalke 04",
         "wage_eur": 11000.0, "match_type": "reviewed"},
        {"player": "No Such Player", "team": "Test FC",
         "wage_eur": np.nan, "match_type": "unmatched"},
    ])
    result = score_gold_set(matched, gold)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["n_correct"] == 2
    counts = summarize_wage_matches(matched)
    assert counts["reviewed"] == 1
    intervals = wage_uncertainty_intervals(matched)
    meyer = intervals[intervals["player"] == "Max Meyer"].iloc[0]
    assert meyer["wage_low"] == pytest.approx(11000.0 * 0.85)
    assert meyer["wage_high"] == pytest.approx(11000.0 * 1.15)
