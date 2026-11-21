"""Stage 3 efficient-frontier-fc C6: wage/entity matching and eligibility bias.

Before this item, load_wages deduplicated globally on normalized name,
match_players matched exact names without team agreement, the fuzzy
fallback took the first ambiguous row, and build_ratings dropped
unmatched players with no audit. These tests pin the fix:

- homonyms are disambiguated by club through a club-name crosswalk,
  and quarantined when the club cannot disambiguate them;
- an exact name at a different club is never silently treated as the
  same player: a unique name is the same person after a transfer and
  is flagged as such, while a homonym with no club agreement is
  quarantined;
- invalid wages are quarantined at load, never matched;
- aliases need evidence (a short name in the same source row, or an
  explicit alias entry) and conflicting duplicate aliases are rejected;
- crosswalk IDs are deterministic and stable across seasons;
- every optimizer row carries one traceable wage source, and the join
  is audited by position, club, season, minutes and value quantile.

Real-player gold set: the wage figures below are real EA FC 24 rows,
read directly from data/fc24_wages.csv on 2026-11-20 with a standalone
pandas script (not through the matcher): Aaron Ramsey Burnley 18000 /
Cardiff City 17000, Adam Smith AFC Bournemouth 36000 / Morecambe 1000,
Adama Traore Ferencvaros 1000 / Hull City 13000 / Melbourne Victory
2000, Mohamed Salah Liverpool 260000, Mohammed Salah Punjab FC 500,
Kevin De Bruyne Manchester City 350000, Erling Haaland Manchester
City 340000, Harry Kane Bayern Muenchen 170000, Granit Xhaka Bayer 04
Leverkusen 74000, Leroy Sane Bayern Muenchen 105000, Nadiem Amiri
1. FSV Mainz 05 29000, Timo Werner Tottenham Hotspur 63000,
Kevin Vogt 1. FC Union Berlin 29000, Yussuf Poulsen
RB Leipzig 53000 (StatsBomb spells him Yussuf Yurary Poulsen). The test
derives its expectations from the embedded fixture table by direct
lookup, never by calling the matcher.
"""

import os

import numpy as np
import pandas as pd
import pytest

from effc.data.wages import (
    build_alias_map,
    load_wages,
    match_players,
    normalize_club,
    score_gold_set,
    summarize_wage_matches,
    wage_join_audit,
    assign_proxy_wages,
    norm,
    position_group,
)
from effc.ratings.ratings import build_ratings

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Real rows (see module docstring for the source and verification).
# Synthetic rows are marked as such where they appear.
_REAL_WAGE_ROWS = [
    ("Aaron Ramsey", "A. Ramsey", "Cardiff City", "CM, CAM", 32, 74, 3_300_000.0, 17_000.0),
    ("Aaron Ramsey", "A. Ramsey", "Burnley", "LM, CM, CAM", 20, 70, 3_800_000.0, 18_000.0),
    ("Adam Smith", "A. Smith", "AFC Bournemouth", "RB", 32, 74, 2_900_000.0, 36_000.0),
    ("Adam Smith", "A. Smith", "Morecambe", "GK", 30, 60, 160_000.0, 1_000.0),
    ("Adama Traoré", "A. Traoré", "Ferencvárosi Torna Club", "RW, CAM", 28, 74, 4_200_000.0, 1_000.0),
    ("Adama Traoré", "A. Traoré", "Hull City", "CAM, CM, RW", 28, 71, 1_900_000.0, 13_000.0),
    ("Adama Traoré", "A. Traoré", "Melbourne Victory", "LB", 33, 64, 300_000.0, 2_000.0),
    ("Mohamed Salah", "M. Salah", "Liverpool FC", "RW", 31, 89, 85_500_000.0, 260_000.0),
    ("Mohammed Salah", "M. Salah", "Punjab FC", "LB", 28, 53, 80_000.0, 500.0),
    ("Kevin De Bruyne", "K. De Bruyne", "Manchester City", "CM, CAM", 32, 91, 103_000_000.0, 350_000.0),
    ("Erling Haaland", "E. Haaland", "Manchester City", "ST", 22, 91, 185_000_000.0, 340_000.0),
    ("Harry Kane", "H. Kane", "Bayern München", "ST", 29, 90, 119_500_000.0, 170_000.0),
    ("Granit Xhaka", "G. Xhaka", "Bayer 04 Leverkusen", "CDM", 30, 84, 35_500_000.0, 74_000.0),
    ("Leroy Sané", "L. Sané", "Bayern München", "RM, LM", 27, 86, 72_500_000.0, 105_000.0),
    ("Nadiem Amiri", "N. Amiri", "1. FSV Mainz 05", "CM, CAM", 26, 77, 12_500_000.0, 29_000.0),
    ("Timo Werner", "T. Werner", "Tottenham Hotspur", "LW, LM, ST", 27, 80, 22_500_000.0, 63_000.0),
    ("Kevin Vogt", "K. Vogt", "1. FC Union Berlin", "CB", 31, 76, 5_000_000.0, 29_000.0),
    ("Yussuf Poulsen", "Y. Poulsen", "RB Leipzig", "ST", 29, 77, 10_500_000.0, 53_000.0),
]


def _wage_frame(rows=_REAL_WAGE_ROWS):
    df = pd.DataFrame(
        rows,
        columns=["long_name", "short_name", "club_name", "positions",
                 "age", "overall", "value_eur", "wage_eur"],
    )
    df["key"] = df["long_name"].map(norm)
    df["club_key"] = df["club_name"].map(norm)
    df["pos_group"] = df["positions"].map(position_group)
    return df


def _sb_players(pairs):
    return pd.DataFrame(pairs, columns=["player", "team"])


def _fixture_lookup(long_name, club_name):
    """Independent lookup: the wage of a real row, by direct filtering."""
    df = _wage_frame()
    hit = df[(df["long_name"] == long_name) & (df["club_name"] == club_name)]
    assert len(hit) == 1, f"fixture must hold exactly one {long_name} at {club_name}"
    return float(hit["wage_eur"].iloc[0])


def test_stage3_efficient_frontier_fc_6_gold_literals_match_fixture():
    # Guards the gold set itself: every literal wage in the tests below
    # must equal the value the embedded fixture table holds for that
    # real (name, club) row, checked by direct lookup, not by matching.
    assert _fixture_lookup("Aaron Ramsey", "Burnley") == 18_000.0
    assert _fixture_lookup("Aaron Ramsey", "Cardiff City") == 17_000.0
    assert _fixture_lookup("Adam Smith", "AFC Bournemouth") == 36_000.0
    assert _fixture_lookup("Adam Smith", "Morecambe") == 1_000.0
    assert _fixture_lookup("Adama Traoré", "Hull City") == 13_000.0
    assert _fixture_lookup("Adama Traoré", "Ferencvárosi Torna Club") == 1_000.0
    assert _fixture_lookup("Adama Traoré", "Melbourne Victory") == 2_000.0
    assert _fixture_lookup("Mohamed Salah", "Liverpool FC") == 260_000.0
    assert _fixture_lookup("Harry Kane", "Bayern München") == 170_000.0
    assert _fixture_lookup("Kevin De Bruyne", "Manchester City") == 350_000.0
    assert _fixture_lookup("Nadiem Amiri", "1. FSV Mainz 05") == 29_000.0
    assert _fixture_lookup("Timo Werner", "Tottenham Hotspur") == 63_000.0


def test_stage3_efficient_frontier_fc_6_homonym_collision():
    # Same name, several clubs: the club must pick the row, and a club
    # with no row for the name must quarantine instead of guessing.
    sb = _sb_players([
        ("Aaron Ramsey", "Burnley"),
        ("Aaron Ramsey", "Cardiff City"),
        ("Aaron Ramsey", "Arsenal"),
        ("Adam Smith", "AFC Bournemouth"),
        ("Adam Smith", "Morecambe"),
        ("Adama Traoré", "Ferencvárosi Torna Club"),
        ("Adama Traoré", "Hull City"),
        ("Adama Traoré", "Melbourne Victory"),
    ])
    matched = match_players(sb, _wage_frame())
    by_key = {
        (r["player"], r["team"]): r for _, r in matched.iterrows()
    }
    assert by_key[("Aaron Ramsey", "Burnley")]["wage_eur"] == 18_000.0
    assert by_key[("Aaron Ramsey", "Burnley")]["match_type"] == "exact"
    assert by_key[("Aaron Ramsey", "Cardiff City")]["wage_eur"] == 17_000.0
    assert by_key[("Aaron Ramsey", "Cardiff City")]["match_type"] == "exact"
    assert by_key[("Adam Smith", "AFC Bournemouth")]["wage_eur"] == 36_000.0
    assert by_key[("Adam Smith", "Morecambe")]["wage_eur"] == 1_000.0
    assert by_key[("Adama Traoré", "Hull City")]["wage_eur"] == 13_000.0
    assert by_key[("Adama Traoré", "Ferencvárosi Torna Club")]["wage_eur"] == 1_000.0
    assert by_key[("Adama Traoré", "Melbourne Victory")]["wage_eur"] == 2_000.0
    # No Ramsey row at Arsenal: quarantined, no wage, never the first row.
    arsenal = by_key[("Aaron Ramsey", "Arsenal")]
    assert arsenal["match_type"] == "ambiguous"
    assert not arsenal["matched"]
    assert pd.isna(arsenal["wage_eur"])


def test_stage3_efficient_frontier_fc_6_club_crosswalk():
    # Naming variants for the same club must agree; reserve and
    # differently-named clubs must not.
    assert normalize_club("Bayer Leverkusen") == normalize_club("Bayer 04 Leverkusen")
    assert normalize_club("Bayern Munich") == normalize_club("Bayern München")
    assert normalize_club("FC Köln") == normalize_club("1. FC Köln")
    assert normalize_club("Darmstadt 98") == normalize_club("SV Darmstadt 98")
    assert normalize_club("FC Heidenheim") == normalize_club("1. FC Heidenheim 1846")
    assert normalize_club("FSV Mainz 05") == normalize_club("1. FSV Mainz 05")
    assert normalize_club("Freiburg") == normalize_club("SC Freiburg")
    assert normalize_club("Hoffenheim") == normalize_club("TSG 1899 Hoffenheim")
    assert normalize_club("Wolfsburg") == normalize_club("VfL Wolfsburg")
    assert normalize_club("Bochum") == normalize_club("VfL Bochum 1848")
    assert normalize_club("Union Berlin") == normalize_club("1. FC Union Berlin")
    assert normalize_club("Augsburg") == normalize_club("FC Augsburg")
    assert normalize_club("Freiburg") != normalize_club("SC Freiburg II")
    assert normalize_club("Bayern Munich") != normalize_club("Borussia Dortmund")
    assert normalize_club(None) == ""
    assert normalize_club(float("nan")) == ""
    sb = _sb_players([
        ("Granit Xhaka", "Bayer Leverkusen"),
        ("Leroy Sané", "Bayern Munich"),
        ("Harry Kane", "Bayern Munich"),
    ])
    matched = match_players(sb, _wage_frame())
    by_player = dict(zip(matched["player"], matched.itertuples()))
    assert by_player["Granit Xhaka"].wage_eur == 74_000.0
    assert by_player["Granit Xhaka"].match_type == "exact"
    assert by_player["Leroy Sané"].wage_eur == 105_000.0
    assert by_player["Leroy Sané"].match_type == "exact"
    assert by_player["Harry Kane"].wage_eur == 170_000.0
    assert by_player["Harry Kane"].match_type == "exact"


def test_stage3_efficient_frontier_fc_6_wrong_club_exact_name():
    # A unique name at another club is the same person after a transfer:
    # the wage is his, but the match is flagged as other-club, not exact.
    sb = _sb_players([
        ("Nadiem Amiri", "Bayer Leverkusen"),
        ("Timo Werner", "RB Leipzig"),
        ("Kevin Vogt", "Hoffenheim"),
    ])
    matched = match_players(sb, _wage_frame())
    by_player = dict(zip(matched["player"], matched.itertuples()))
    assert by_player["Nadiem Amiri"].wage_eur == 29_000.0
    assert by_player["Nadiem Amiri"].match_type == "exact_other_club"
    assert by_player["Nadiem Amiri"].matched
    assert by_player["Timo Werner"].wage_eur == 63_000.0
    assert by_player["Timo Werner"].match_type == "exact_other_club"
    assert by_player["Kevin Vogt"].wage_eur == 29_000.0
    assert by_player["Kevin Vogt"].match_type == "exact_other_club"


def test_stage3_efficient_frontier_fc_6_invalid_wage(tmp_path):
    # A zero, negative or implausible wage is quarantined at load and
    # reported, never matched; a frame carrying one fails closed.
    csv = tmp_path / "fc24_wages.csv"
    csv.write_text(
        "long_name,short_name,club_name,positions,age,overall,value_eur,wage_eur\n"
        "Good Player,G. Player,Test FC,ST,25,80,10000000,50000\n"
        "Zero Wage,Z. Wage,Test FC,CM,25,70,1000000,0\n"
        "Negative Wage,N. Wage,Test FC,CM,25,70,1000000,-100\n"
        "Huge Wage,H. Wage,Test FC,CM,25,70,1000000,5000000\n"
    )
    wages, report = load_wages(str(csv), with_report=True)
    assert len(wages) == 1
    assert wages.iloc[0]["long_name"] == "Good Player"
    assert report["n_invalid_wage"] == 3
    assert report["n_rows_read"] == 4
    assert report["n_rows_kept"] == 1
    assert report["wage_source_sha256"]
    assert report["wage_source_mtime"]
    bad = _wage_frame()
    bad.loc[len(bad)] = [
        "Bad Wage", "B. Wage", "Test FC", "ST", 25, 70,
        1_000_000.0, 0.0, norm("Bad Wage"), norm("Test FC"), "FWD",
    ]
    sb = _sb_players([("Bad Wage", "Test FC")])
    with pytest.raises(ValueError, match="invalid wage"):
        match_players(sb, bad)


def test_stage3_efficient_frontier_fc_6_load_wages_homonym_dedup(tmp_path):
    # The same person at one club twice keeps the highest-rated row;
    # the same name at two clubs keeps both rows.
    csv = tmp_path / "fc24_wages.csv"
    csv.write_text(
        "long_name,short_name,club_name,positions,age,overall,value_eur,wage_eur\n"
        "Aaron Ramsey,A. Ramsey,Cardiff City,CM,25,74,3300000,17000\n"
        "Aaron Ramsey,A. Ramsey,Burnley,LM,25,70,3800000,18000\n"
        "Aaron Ramsey,A. Ramsey,Cardiff City,CM,25,72,3000000,16000\n"
    )
    wages = load_wages(str(csv))
    assert len(wages) == 2
    assert set(wages["club_name"]) == {"Cardiff City", "Burnley"}
    cardiff = wages[wages["club_name"] == "Cardiff City"].iloc[0]
    assert cardiff["overall"] == 74
    assert cardiff["wage_eur"] == 17_000.0


def test_stage3_efficient_frontier_fc_6_duplicate_alias():
    # One alias, two canonicals: rejected. The same alias twice for
    # the same canonical is fine. Evidence is required, never blank.
    with pytest.raises(ValueError, match="conflicting"):
        build_alias_map([
            ("m salah", "mohamed salah", "Liverpool FC squad list"),
            ("m salah", "mohammed salah", "Punjab FC squad list"),
        ])
    with pytest.raises(ValueError, match="evidence"):
        build_alias_map([("m salah", "mohamed salah", "")])
    ok = build_alias_map([
        ("m salah", "mohamed salah", "Liverpool FC squad list"),
        ("m salah", "mohamed salah", "Liverpool FC squad list"),
    ])
    assert ok == {"m salah": ("mohamed salah", "Liverpool FC squad list")}
    sb = _sb_players([("Mohamed Salah", "Liverpool FC")])
    with pytest.raises(ValueError, match="conflicting"):
        match_players(sb, _wage_frame(), aliases=[
            ("mohamed salah", "mohamed salah ghaly", "FC long name"),
            ("mohamed salah", "mohammed salah", "Punjab FC squad list"),
        ])


def test_stage3_efficient_frontier_fc_6_alias_match():
    # A short name from the same source row is verified alias evidence,
    # but only inside the club and only when it points at one player.
    sb = _sb_players([
        ("M. Salah", "Liverpool FC"),
        ("M. Salah", "Punjab FC"),
    ])
    matched = match_players(sb, _wage_frame())
    by_team = dict(zip(matched["team"], matched.itertuples()))
    assert by_team["Liverpool FC"].wage_eur == 260_000.0
    assert by_team["Liverpool FC"].match_type == "alias"
    assert by_team["Punjab FC"].wage_eur == 500.0
    assert by_team["Punjab FC"].match_type == "alias"
    # An explicit alias table resolves a name the file spells longer.
    # The plain Mohamed Salah row is left out here so the alias path
    # is what resolves the name, not an exact match.
    rows = _wage_frame()
    rows = rows[rows["long_name"] != "Mohamed Salah"].reset_index(drop=True)
    long_row = pd.DataFrame([{
        "long_name": "Mohamed Salah Ghaly", "short_name": "M. Salah",
        "club_name": "Liverpool FC", "positions": "RW", "age": 31,
        "overall": 89, "value_eur": 85_500_000.0, "wage_eur": 260_000.0,
        "key": norm("Mohamed Salah Ghaly"), "club_key": norm("Liverpool FC"),
        "pos_group": "FWD",
    }])
    rows = pd.concat([rows, long_row], ignore_index=True)
    aliases = build_alias_map([(
        "mohamed salah", "mohamed salah ghaly",
        "FC long name adds the family name Ghaly; same club and position",
    )])
    matched = match_players(
        _sb_players([("Mohamed Salah", "Liverpool FC")]), rows,
        aliases=aliases,
    )
    assert matched.iloc[0]["wage_eur"] == 260_000.0
    assert matched.iloc[0]["match_type"] == "alias"


def test_stage3_efficient_frontier_fc_6_ambiguous_fuzzy_quarantine():
    # Two wage rows in one club share a last name and first initial:
    # a name variant must quarantine, never take the first row.
    rows = pd.DataFrame([
        {"long_name": "Tim Müller", "short_name": "T. Müller",
         "club_name": "Test FC", "positions": "CM", "age": 25,
         "overall": 75, "value_eur": 5_000_000.0, "wage_eur": 20_000.0,
         "key": norm("Tim Müller"), "club_key": norm("Test FC"),
         "pos_group": "MID"},
        {"long_name": "Tom Müller", "short_name": "T. Müller",
         "club_name": "Test FC", "positions": "CM", "age": 26,
         "overall": 74, "value_eur": 4_000_000.0, "wage_eur": 18_000.0,
         "key": norm("Tom Müller"), "club_key": norm("Test FC"),
         "pos_group": "MID"},
    ])
    matched = match_players(
        _sb_players([("Terry Müller", "Test FC")]), rows)
    assert matched.iloc[0]["match_type"] == "ambiguous"
    assert not matched.iloc[0]["matched"]
    assert pd.isna(matched.iloc[0]["wage_eur"])
    # A unique fuzzy candidate still matches, as before: StatsBomb
    # spells him Yussuf Yurary Poulsen, the file Yussuf Poulsen.
    matched = match_players(
        _sb_players([("Yussuf Yurary Poulsen", "RB Leipzig")]),
        _wage_frame())
    assert matched.iloc[0]["match_type"] == "fuzzy"
    assert matched.iloc[0]["wage_eur"] == 53_000.0


def test_stage3_efficient_frontier_fc_6_stable_ids_across_seasons():
    # IDs come from content, not row order: the same person at the same
    # club keeps the same row ID across seasons, and a move changes it.
    wages = _wage_frame()
    sb = _sb_players([
        ("Aaron Ramsey", "Burnley"),
        ("Aaron Ramsey", "Cardiff City"),
        ("Kevin De Bruyne", "Manchester City"),
    ])
    first = match_players(sb, wages)
    shuffled = match_players(sb, wages.sample(frac=1.0, random_state=7))
    merged = first.merge(
        shuffled, on=["player", "team"], suffixes=("_a", "_b"))
    assert (merged["wage_row_id_a"] == merged["wage_row_id_b"]).all()
    assert (merged["person_key_a"] == merged["person_key_b"]).all()
    by_key = {
        (r["player"], r["team"]): r for _, r in first.iterrows()
    }
    assert by_key[("Aaron Ramsey", "Burnley")]["wage_row_id"] != (
        by_key[("Aaron Ramsey", "Cardiff City")]["wage_row_id"])
    assert by_key[("Aaron Ramsey", "Burnley")]["person_key"] == (
        by_key[("Aaron Ramsey", "Cardiff City")]["person_key"])
    assert first["wage_row_id"].is_unique
    assert first["wage_row_id"].str.startswith("fc24:").all()


def test_stage3_efficient_frontier_fc_6_gold_set_precision_recall():
    # The real-player gold set, scored independently of the matcher:
    # expected wages and types are literals checked against the CSV.
    gold = pd.DataFrame([
        {"player": "Aaron Ramsey", "team": "Burnley",
         "expected_wage_eur": 18_000.0, "expected_match_type": "exact"},
        {"player": "Aaron Ramsey", "team": "Cardiff City",
         "expected_wage_eur": 17_000.0, "expected_match_type": "exact"},
        {"player": "Aaron Ramsey", "team": "Arsenal",
         "expected_wage_eur": np.nan, "expected_match_type": "ambiguous"},
        {"player": "Adam Smith", "team": "AFC Bournemouth",
         "expected_wage_eur": 36_000.0, "expected_match_type": "exact"},
        {"player": "Adam Smith", "team": "Morecambe",
         "expected_wage_eur": 1_000.0, "expected_match_type": "exact"},
        {"player": "Adama Traoré", "team": "Hull City",
         "expected_wage_eur": 13_000.0, "expected_match_type": "exact"},
        {"player": "Adama Traoré", "team": "Melbourne Victory",
         "expected_wage_eur": 2_000.0, "expected_match_type": "exact"},
        {"player": "Granit Xhaka", "team": "Bayer Leverkusen",
         "expected_wage_eur": 74_000.0, "expected_match_type": "exact"},
        {"player": "Leroy Sané", "team": "Bayern Munich",
         "expected_wage_eur": 105_000.0, "expected_match_type": "exact"},
        {"player": "Harry Kane", "team": "Bayern Munich",
         "expected_wage_eur": 170_000.0, "expected_match_type": "exact"},
        {"player": "Nadiem Amiri", "team": "Bayer Leverkusen",
         "expected_wage_eur": 29_000.0,
         "expected_match_type": "exact_other_club"},
        {"player": "Timo Werner", "team": "RB Leipzig",
         "expected_wage_eur": 63_000.0,
         "expected_match_type": "exact_other_club"},
        {"player": "Kevin De Bruyne", "team": "Manchester City",
         "expected_wage_eur": 350_000.0, "expected_match_type": "exact"},
        {"player": "Erling Haaland", "team": "Manchester City",
         "expected_wage_eur": 340_000.0, "expected_match_type": "exact"},
        {"player": "Yussuf Yurary Poulsen", "team": "RB Leipzig",
         "expected_wage_eur": 53_000.0, "expected_match_type": "fuzzy"},
        {"player": "No Such Player", "team": "Test FC",
         "expected_wage_eur": np.nan, "expected_match_type": "unmatched"},
    ])
    sb = gold[["player", "team"]]
    matched = match_players(sb, _wage_frame())
    result = score_gold_set(matched, gold)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["n_gold"] == 16
    assert result["n_wrong_wage"] == 0
    assert result["n_wrong_club"] == 0
    # The scorer measures rather than rubber-stamps: corrupt one
    # expected wage and precision must fall.
    corrupt = gold.copy()
    corrupt.loc[corrupt["player"] == "Harry Kane", "expected_wage_eur"] = 1.0
    bad = score_gold_set(matched, corrupt)
    assert bad["precision"] < 1.0
    assert bad["n_wrong_wage"] == 1


def test_stage3_efficient_frontier_fc_6_join_audit_slices():
    sb = _sb_players([
        ("Aaron Ramsey", "Burnley"),
        ("Aaron Ramsey", "Cardiff City"),
        ("Aaron Ramsey", "Arsenal"),
        ("Adam Smith", "AFC Bournemouth"),
        ("Kevin De Bruyne", "Manchester City"),
        ("Erling Haaland", "Manchester City"),
        ("No Such Player", "Test FC"),
    ])
    matched = match_players(sb, _wage_frame())
    minutes = pd.DataFrame([
        {"player": "Aaron Ramsey", "team": "Burnley", "minutes": 1200.0},
        {"player": "Aaron Ramsey", "team": "Cardiff City", "minutes": 900.0},
        {"player": "Aaron Ramsey", "team": "Arsenal", "minutes": 300.0},
        {"player": "Adam Smith", "team": "AFC Bournemouth", "minutes": 1800.0},
        {"player": "Kevin De Bruyne", "team": "Manchester City", "minutes": 2500.0},
        {"player": "Erling Haaland", "team": "Manchester City", "minutes": 2700.0},
        {"player": "No Such Player", "team": "Test FC", "minutes": 100.0},
    ])
    audit = wage_join_audit(
        matched, minutes=minutes,
        season=pd.Series(["2023/24"] * 7, index=matched.index),
    )
    assert audit["n_players"] == 7
    assert audit["counts"]["exact"] == 5
    assert audit["counts"]["ambiguous"] == 1
    assert audit["counts"]["unmatched"] == 1
    assert audit["match_rate"] == pytest.approx(5 / 7)
    assert audit["by_position"]["MID"]["n"] == 3
    assert audit["by_position"]["MID"]["matched"] == 3
    assert audit["by_club"]["Manchester City"]["n"] == 2
    assert audit["by_club"]["Manchester City"]["matched"] == 2
    assert audit["by_season"]["2023/24"]["n"] == 7
    assert sum(b["n"] for b in audit["by_minutes"].values()) == 7
    assert audit["by_minutes"]["1800+"]["n"] == 3
    assert sum(b["n"] for b in audit["by_value_quantile"].values()) == 5
    assert audit["by_value_quantile"]["Q1"]["n"] >= 1
    # Quintiles are contiguous ranked slices, not strided samples: the
    # cheapest matched player sits in Q1 and the dearest in Q5.
    assert audit["by_value_quantile"]["Q1"]["median_value_eur"] == pytest.approx(
        2_900_000.0)
    assert audit["by_value_quantile"]["Q5"]["median_value_eur"] == pytest.approx(
        185_000_000.0)
    # Quarantined rows are listed for review, with the reason attached.
    assert len(audit["quarantine"]) == 1
    assert audit["quarantine"][0]["player"] == "Aaron Ramsey"
    assert audit["quarantine"][0]["team"] == "Arsenal"


def test_stage3_efficient_frontier_fc_6_build_ratings_traceable():
    pm = pd.DataFrame([
        {"match_id": 1, "player_id": 1, "player": "Kevin De Bruyne",
         "team": "Manchester City", "vaep": 0.5},
        {"match_id": 1, "player_id": 2, "player": "Erling Haaland",
         "team": "Manchester City", "vaep": 0.8},
        {"match_id": 1, "player_id": 3, "player": "No Such Player",
         "team": "Test FC", "vaep": 0.1},
    ])
    minutes = pd.DataFrame([
        {"match_id": 1, "player_id": 1, "minutes": 90.0, "position": "CM"},
        {"match_id": 1, "player_id": 2, "minutes": 90.0, "position": "ST"},
        {"match_id": 1, "player_id": 3, "minutes": 90.0, "position": "ST"},
    ])
    sb = _sb_players([
        ("Kevin De Bruyne", "Manchester City"),
        ("Erling Haaland", "Manchester City"),
        ("No Such Player", "Test FC"),
    ])
    wmap = match_players(sb, _wage_frame())
    ratings = build_ratings(pm, minutes, wmap, min_minutes=60)
    assert len(ratings) == 2
    assert ratings["wage_source"].notna().all()
    assert ratings["wage_row_id"].notna().all()
    assert set(ratings["player"]) == {"Kevin De Bruyne", "Erling Haaland"}
    audit = ratings.attrs["wage_join_audit"]
    assert audit["n_dropped_unmatched"] == 1
    assert audit["n_dropped_below_minutes"] == 0
    assert audit["n_retained"] == 2
    # A wage map with two rows for one player fails closed, and a
    # retained row without a wage source fails closed too.
    dup = pd.concat([wmap, wmap.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_ratings(pm, minutes, dup, min_minutes=60)
    nosource = wmap.copy()
    nosource.loc[nosource["player"] == "Kevin De Bruyne", "wage_source"] = None
    with pytest.raises(ValueError, match="wage source"):
        build_ratings(pm, minutes, nosource, min_minutes=60)


def test_stage3_efficient_frontier_fc_6_proxy_timestamps_and_quarantine():
    sb = _sb_players([
        ("Kevin De Bruyne", "Manchester City"),
        ("Aaron Ramsey", "Arsenal"),
        ("No Such Player", "Test FC"),
    ])
    wages, report = load_wages(
        os.path.join(REPO_ROOT, "tests", "fixtures", "c6_wages.csv"),
        with_report=True,
    )
    matched = match_players(sb, wages)
    proxied = assign_proxy_wages(matched)
    assert proxied.loc[
        proxied["player"] == "No Such Player", "match_type"].iloc[0] == "proxy"
    proxy = proxied.loc[proxied["player"] == "No Such Player"].iloc[0]
    assert proxy["wage_source"].startswith("proxy")
    assert proxy["wage_source_sha256"] == report["wage_source_sha256"]
    assert proxy["wage_source_mtime"] == report["wage_source_mtime"]
    # Quarantined rows get no proxy wage: the ambiguity stays visible.
    assert proxied.loc[
        proxied["player"] == "Aaron Ramsey", "match_type"].iloc[0] == "ambiguous"
    assert pd.isna(proxied.loc[
        proxied["player"] == "Aaron Ramsey", "wage_eur"].iloc[0])


def test_stage3_efficient_frontier_fc_6_pipeline_records_audit():
    text = open(os.path.join(
        REPO_ROOT, "src", "effc", "pipeline.py")).read()
    assert "wage_match_audit" in text
    assert "wage_source" in text


def test_stage3_efficient_frontier_fc_6_summarize_counts_all_types():
    counts = summarize_wage_matches(pd.DataFrame({
        "match_type": ["exact", "exact_other_club", "alias", "fuzzy",
                       "ambiguous", "proxy", "unmatched"],
    }))
    assert counts == {
        "exact": 1, "exact_other_club": 1, "alias": 1, "fuzzy": 1,
        "ambiguous": 1, "proxy": 1, "unmatched": 1,
    }
