"""EA FC 24 wages joined to StatsBomb players by normalized name matching.

Wage units: ``wage_eur`` is EUR per week, the unit EA FC 24 reports and
the unit the README budgets use. Annual figures are weekly x 52; see
``annual_wage_eur`` and the provenance record in ``effc.data.provenance``.

Matching rules (Stage 3 C6). An exact name match is only accepted when
the club agrees through the club crosswalk (``normalize_club`` plus the
``CLUB_ALIASES`` overrides), or when the name is unique in the wage
file, in which case the player is the same person after a transfer and
the match is flagged ``exact_other_club`` instead of ``exact``. A name
at several clubs whose club agrees with none is quarantined as
``ambiguous``: no wage is assigned and no row is guessed. The
club-constrained fuzzy fallback (last name plus first initial) only
matches a unique candidate; several candidates quarantine the row.
Aliases need evidence: a short name from the same source row, or an
explicit alias entry built with ``build_alias_map``, which rejects
conflicting duplicates. Invalid wages (zero, negative, non-finite, or
above ``MAX_PLAUSIBLE_WAGE_EUR_WEEK``) are quarantined at load and
reported, never matched.

Every matched row carries ``wage_row_id`` (a deterministic crosswalk
ID, stable across seasons for the same person at the same club),
``person_key`` (the normalized name, stable across seasons and clubs),
and the wage source: ``wage_source`` plus the source file digest and
version timestamp (``wage_source_sha256``, ``wage_source_mtime``), so
each optimizer row traces to one wage source. Proxy rows carry the
source version their median was computed from.
"""

from __future__ import annotations

import hashlib
import os
import unicodedata
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from effc.data.download import DATA_DIR

# Wage unit for every wage_eur value in this module: EUR per week.
WAGE_UNIT = "EUR/week"
WEEKS_PER_YEAR = 52

# The wage file is a weekly-wage proxy from EA FC 24. The highest weekly
# wage in the file is 350,000 EUR, so the plausibility bound sits well
# above any real row and only catches corrupt or mis-scaled values.
MAX_PLAUSIBLE_WAGE_EUR_WEEK = 1_000_000.0

WAGE_SOURCE_NAME = "EA FC 24 (fc24_wages.csv)"
WAGE_SOURCE_FILE = "fc24_wages.csv"
CALLER_SUPPLIED_SOURCE = (
    "caller-supplied wage table (no source file recorded)"
)

# Generic club-name tokens dropped by normalize_club. These are legal
# or league markers, never the distinguishing part of a club name:
# "1. FC Koeln" and "FC Koeln" are the same club, as are "SV Darmstadt
# 98" and "Darmstadt 98". Pure-number tokens ("04", "1899", "1846")
# are dropped by the digit rule in normalize_club.
CLUB_GENERIC_TOKENS = {
    "fc", "sv", "vfl", "sc", "tsg", "fsv", "cf", "afc", "ac", "as",
    "rc", "ssc", "us", "ud", "cd", "sd", "spvgg", "sg", "kv", "de",
}
# English exonyms for club-name tokens, verified against the wage file:
# StatsBomb says "Bayern Munich", the file says "Bayern Muenchen".
# Add an entry only with a verified same-club pair behind it.
CLUB_EXONYMS = {"munich": "munchen"}
# Explicit club crosswalk overrides: normalized StatsBomb team name to
# normalized wage-file club name, for pairs the token rules above
# cannot resolve. Each entry needs a verified same-club pair (shared
# player names between the two spellings). Empty today: the token rules
# resolve every Bundesliga team in the local data (22 of 23 map to
# exactly one wage-file club; Hertha Berlin has no wage-file club), so
# no override is needed yet. Pass overrides to match_players through
# its club_aliases argument to use them without editing this module.
CLUB_ALIASES: dict[str, str] = {}

# Match types. "exact" and "exact_other_club", "alias" and "fuzzy" are
# matches (matched True). "ambiguous" is quarantine (matched False, no
# wage). "proxy" is a labeled stand-in (matched False). "unmatched"
# has no wage row at all.
MATCH_TYPES = (
    "exact", "exact_other_club", "alias", "fuzzy",
    "ambiguous", "proxy", "unmatched",
)
_MATCHED_TYPES = ("exact", "exact_other_club", "alias", "fuzzy")


def annual_wage_eur(weekly_eur: float) -> float:
    """Annual EUR for a weekly EUR wage (52 weeks per year)."""
    return float(weekly_eur) * WEEKS_PER_YEAR

POSITION_GROUPS = {
    "GK": "GK",
    "RB": "DEF",
    "RWB": "DEF",
    "CB": "DEF",
    "LB": "DEF",
    "LWB": "DEF",
    "CDM": "MID",
    "CM": "MID",
    "CAM": "MID",
    "RM": "MID",
    "LM": "MID",
    "RW": "FWD",
    "LW": "FWD",
    "CF": "FWD",
    "ST": "FWD",
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = "".join(c.lower() if c.isalnum() or c == " " else " " for c in s)
    return " ".join(s.split())


def normalize_club(name) -> str:
    """Club name reduced to its distinguishing tokens.

    Strips accents, lowercases, drops generic markers (FC, SV, VfL,
    TSG, ...), drops pure-number tokens (04, 1899, 1846, 98, 05) and
    maps verified exonyms. Returns "" for a missing name. Two spellings
    of the same club agree; a reserve side ("SC Freiburg II") does not
    agree with its first team, and distinct clubs never agree: the
    reduction is injective over the wage file's club list.
    """
    if name is None:
        return ""
    try:
        if pd.isna(name):
            return ""
    except (TypeError, ValueError):
        pass
    tokens = []
    for token in norm(name).split():
        token = CLUB_EXONYMS.get(token, token)
        if token in CLUB_GENERIC_TOKENS or token.isdigit():
            continue
        tokens.append(token)
    return " ".join(tokens)


def position_group(positions: str) -> str:
    """First listed FC position mapped to GK/DEF/MID/FWD."""
    first = str(positions).split(",")[0].strip().upper()
    return POSITION_GROUPS.get(first, "MID")


def _wage_row_id(person_key: str, club_norm: str) -> str:
    """Deterministic crosswalk ID for one (person, club) wage row.

    Content-based, so the same person at the same club keeps the same
    ID across seasons and across row orderings, and a transfer (new
    club) gets a new row ID while ``person_key`` stays stable.
    """
    digest = hashlib.sha1(
        f"{person_key}|{club_norm}".encode("utf-8")).hexdigest()[:12]
    return f"fc24:{digest}"


def _source_file_record(path: str) -> dict:
    """SHA-256 digest and version timestamp of a wage source file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    mtime = datetime.fromtimestamp(
        os.path.getmtime(path), tz=timezone.utc).isoformat()
    return {
        "wage_source": WAGE_SOURCE_NAME,
        "wage_source_file": os.path.basename(path),
        "wage_source_sha256": digest.hexdigest(),
        "wage_source_mtime": mtime,
    }


def _invalid_wage_mask(wages: pd.DataFrame) -> pd.Series:
    """True for wage rows that cannot be a weekly EUR wage."""
    value = pd.to_numeric(wages["wage_eur"], errors="coerce")
    return ~(
        value.notna()
        & (value > 0)
        & (value <= MAX_PLAUSIBLE_WAGE_EUR_WEEK)
    )


def validate_wage_values(wages: pd.DataFrame) -> None:
    """Fail closed on any invalid wage in a table about to be matched.

    ``load_wages`` quarantines invalid rows before this point; a caller
    that built a wage table by hand must clean it first instead of
    silently matching a zero or corrupt wage.
    """
    if "wage_eur" not in wages.columns:
        raise ValueError("validate_wage_values needs a wage_eur column")
    bad = wages[_invalid_wage_mask(wages)]
    if len(bad):
        names = bad["long_name"].tolist() if "long_name" in bad else []
        raise ValueError(
            f"invalid wage_eur in {len(bad)} wage row(s)"
            + (f": {names[:5]}" if names else "")
            + f" (must be finite, over 0 and at most "
              f"{MAX_PLAUSIBLE_WAGE_EUR_WEEK:,.0f} EUR/week)"
        )


def load_wages(path: str | None = None, with_report: bool = False):
    """Load the EA FC 24 wage file with homonym-safe dedup.

    Deduplicates per (normalized name, normalized club), keeping the
    highest-rated row: the same name at two clubs is two people (or
    one person at two clubs across roster updates) and both rows are
    kept. Rows with an invalid wage are quarantined: dropped from the
    returned frame and counted in the report, never matched. Every
    returned row carries its crosswalk ID and wage source record.
    """
    path = path or os.path.join(DATA_DIR, WAGE_SOURCE_FILE)
    record = _source_file_record(path)
    df = pd.read_csv(path, usecols=[
        "long_name", "short_name", "club_name", "positions", "age", "overall",
        "value_eur", "wage_eur",
    ])
    n_read = len(df)
    df = df.dropna(subset=["long_name", "wage_eur"])
    n_dropped_missing = n_read - len(df)
    df["key"] = df["long_name"].map(norm)
    df["club_key"] = df["club_name"].map(norm)
    df["club_norm"] = df["club_name"].map(normalize_club)
    df["pos_group"] = df["positions"].map(position_group)
    invalid = _invalid_wage_mask(df)
    n_invalid = int(invalid.sum())
    df = df[~invalid]
    # Keep the highest-rated row per person per club (dedupes across
    # roster updates); a name at two clubs keeps both rows.
    df = df.sort_values("overall", ascending=False).drop_duplicates(
        ["key", "club_norm"])
    df = df.reset_index(drop=True)
    df["person_key"] = df["key"]
    df["wage_row_id"] = [
        _wage_row_id(k, c) for k, c in zip(df["key"], df["club_norm"])
    ]
    df["wage_source"] = record["wage_source"]
    df["wage_source_sha256"] = record["wage_source_sha256"]
    df["wage_source_mtime"] = record["wage_source_mtime"]
    n_homonym_names = int(
        df.groupby("key")["club_norm"].nunique().gt(1).sum())
    report = {
        **record,
        "n_rows_read": int(n_read),
        "n_rows_kept": int(len(df)),
        "n_dropped_missing": int(n_dropped_missing),
        "n_invalid_wage": n_invalid,
        "n_homonym_names": n_homonym_names,
    }
    df.attrs["wage_load_report"] = report
    if with_report:
        return df, report
    return df


def build_alias_map(pairs) -> dict:
    """Build a verified alias table from (alias, canonical, evidence).

    Alias and canonical are normalized names. Every entry needs
    non-empty evidence: the fact that justifies treating the alias as
    the canonical player (a squad list, a source row, a name-variant
    note). The same alias pointing at two different canonicals is a
    conflict and raises; the same alias twice for one canonical is
    fine. Returns {alias: (canonical, evidence)}.
    """
    table: dict[str, tuple[str, str]] = {}
    for alias, canonical, evidence in pairs:
        alias_key = norm(alias)
        canonical_key = norm(canonical)
        if not str(evidence or "").strip():
            raise ValueError(
                f"alias {alias!r} needs evidence: record what justifies "
                f"treating it as {canonical!r}")
        if alias_key in table and table[alias_key][0] != canonical_key:
            raise ValueError(
                f"conflicting alias {alias!r}: already maps to "
                f"{table[alias_key][0]!r}, cannot also map to "
                f"{canonical_key!r}")
        table[alias_key] = (canonical_key, str(evidence).strip())
    return table


def _normalize_aliases(aliases) -> dict:
    """Accept an alias table as built by build_alias_map, or triples."""
    if aliases is None:
        return {}
    if isinstance(aliases, dict):
        table = {}
        for alias, value in aliases.items():
            if isinstance(value, (tuple, list)):
                canonical, evidence = value[0], value[1]
            else:
                raise ValueError(
                    f"alias {alias!r} needs evidence: pass a "
                    f"(canonical, evidence) pair or use build_alias_map")
            table[norm(alias)] = (norm(canonical), str(evidence))
        return table
    return build_alias_map(aliases)


def _prepare_wage_frame(wages: pd.DataFrame) -> pd.DataFrame:
    """Fill the derived columns match_players needs, when absent."""
    w = wages.copy()
    if "key" not in w.columns:
        if "long_name" not in w.columns:
            raise ValueError("wage table needs long_name or key")
        w["key"] = w["long_name"].map(norm)
    if "club_key" not in w.columns:
        if "club_name" not in w.columns:
            raise ValueError("wage table needs club_name or club_key")
        w["club_key"] = w["club_name"].map(norm)
    if "club_norm" not in w.columns:
        if "club_name" in w.columns:
            w["club_norm"] = w["club_name"].map(normalize_club)
        else:
            w["club_norm"] = w["club_key"].map(normalize_club)
    if "pos_group" not in w.columns:
        if "positions" in w.columns:
            w["pos_group"] = w["positions"].map(position_group)
        else:
            raise ValueError("wage table needs positions or pos_group")
    if "person_key" not in w.columns:
        w["person_key"] = w["key"]
    if "wage_row_id" not in w.columns:
        w["wage_row_id"] = [
            _wage_row_id(k, c) for k, c in zip(w["key"], w["club_norm"])
        ]
    if "wage_source" not in w.columns:
        w["wage_source"] = CALLER_SUPPLIED_SOURCE
    if "wage_source_sha256" not in w.columns:
        w["wage_source_sha256"] = ""
    if "wage_source_mtime" not in w.columns:
        w["wage_source_mtime"] = ""
    if "short_name" not in w.columns:
        w["short_name"] = pd.NA
    return w


def _disambiguate(key, club_norm, by_name, exact_by_club):
    """Pick the wage row for a normalized name at a normalized club.

    Returns (row or None, match type, note). A club-agreeing row
    wins; a name unique in the file is the same person at another club
    (a transfer) and matches flagged as other-club; a name at several
    clubs with no club agreement is quarantined as ambiguous.
    """
    hit = exact_by_club.get((key, club_norm))
    if hit is not None:
        return hit, "exact", ""
    rows = by_name.get(key, [])
    if len(rows) == 1:
        return rows[0], "exact_other_club", (
            "name is unique in the wage file; club differs, so this is "
            "the same player after a transfer, not a same-club match")
    if len(rows) > 1:
        clubs = sorted({str(getattr(r, "club_name")) for r in rows})
        return None, "ambiguous", (
            "homonym: name appears at several clubs "
            f"({', '.join(clubs)}) and the club does not disambiguate it")
    return None, "unmatched", ""


def match_players(sb_players: pd.DataFrame, wages: pd.DataFrame,
                  aliases=None, club_aliases=None) -> pd.DataFrame:
    """Left-join StatsBomb players to wages. Returns sb_players + wage columns.

    sb_players needs columns: player, team. Matching runs in order:
    exact name plus club agreement through the crosswalk; a unique
    name at another club (``exact_other_club``); an explicit alias
    from ``aliases``; a short name from the same wage row (``alias``);
    then the club-constrained last-name plus first-initial fallback
    (``fuzzy``), which only matches a unique candidate. Anything else
    is ``ambiguous`` (quarantined, no wage) or ``unmatched``. The
    match_type column records how each row matched, and every matched
    row carries its crosswalk ID and wage source. Unmatched rows carry
    NaN wages; use assign_proxy_wages to give the unmatched ones an
    explicit, flagged proxy wage (quarantined rows get none).
    """
    alias_table = _normalize_aliases(aliases)
    club_alias_map = dict(CLUB_ALIASES)
    if club_aliases:
        for sb_club, fc_club in club_aliases.items():
            club_alias_map[norm(sb_club)] = norm(fc_club)
    w = _prepare_wage_frame(wages)
    validate_wage_values(w)
    dupes = w.duplicated(subset=["key", "club_norm"], keep=False)
    if dupes.any():
        names = sorted(w.loc[dupes, "key"].unique().tolist())
        raise ValueError(
            f"duplicate wage rows for the same person at the same club: "
            f"{names[:5]} (load through load_wages, which dedupes per "
            f"person per club)")

    sb = sb_players.copy()
    sb["key"] = sb["player"].map(norm)
    sb["club_key"] = sb["team"].map(norm)
    sb["club_norm"] = sb["team"].map(normalize_club)

    rows = list(w.itertuples(index=False))
    col = {name: i for i, name in enumerate(w.columns)}
    by_name: dict[str, list] = {}
    exact_by_club: dict[tuple, object] = {}
    short_by_club: dict[tuple, list] = {}
    fuzzy_by_club: dict[tuple, list] = {}
    for row in rows:
        key = row[col["key"]]
        club_norm = row[col["club_norm"]]
        by_name.setdefault(key, []).append(row)
        exact_by_club[(key, club_norm)] = row
        short = row[col["short_name"]]
        if pd.notna(short) and norm(short) and norm(short) != key:
            short_by_club.setdefault((norm(short), club_norm), []).append(row)
        parts = key.split()
        if len(parts) >= 2:
            fuzzy_by_club.setdefault(
                (club_norm, parts[-1], parts[0][0]), []).append(row)

    out_rows = []
    for sb_row in sb.itertuples(index=False):
        sb_col = {name: i for i, name in enumerate(sb.columns)}
        key = sb_row[sb_col["key"]]
        club_norm = sb_row[sb_col["club_norm"]]
        club_key = sb_row[sb_col["club_key"]]
        # An explicit club alias overrides the club spelling first.
        aliased_club = club_alias_map.get(club_key)
        eff_club = aliased_club if aliased_club is not None else club_norm
        match_type = "unmatched"
        note = ""
        row = None
        idx, mtype, note = _disambiguate(
            key, eff_club, by_name, exact_by_club)
        if idx is not None:
            row, match_type = idx, mtype
        elif mtype == "ambiguous":
            match_type, note = mtype, note
        else:
            alias_hit = alias_table.get(key)
            if alias_hit is not None:
                canonical, evidence = alias_hit
                idx, mtype, note = _disambiguate(
                    canonical, eff_club, by_name, exact_by_club)
                if idx is not None:
                    row = idx
                    match_type = "alias"
                    note = f"alias evidence: {evidence}"
                elif mtype == "ambiguous":
                    match_type, note = mtype, note
            if row is None and match_type == "unmatched":
                short_rows = short_by_club.get((key, eff_club), [])
                if len(short_rows) == 1:
                    row = short_rows[0]
                    match_type = "alias"
                    note = ("alias evidence: short name in the same wage "
                            "row")
                elif len(short_rows) > 1:
                    match_type = "ambiguous"
                    note = ("alias collision: short name points at "
                            "several players in this club")
            if row is None and match_type == "unmatched":
                parts = key.split()
                if len(parts) >= 2:
                    fuzzy_rows = fuzzy_by_club.get(
                        (eff_club, parts[-1], parts[0][0]), [])
                    if len(fuzzy_rows) == 1:
                        row = fuzzy_rows[0]
                        match_type = "fuzzy"
                    elif len(fuzzy_rows) > 1:
                        match_type = "ambiguous"
                        note = ("ambiguous fuzzy fallback: several wage "
                                "rows share this club, last name and "
                                "first initial")
        rec = {c: sb_row[sb_col[c]] for c in sb.columns
               if c not in ("key", "club_key", "club_norm")}
        if row is not None:
            rec["wage_eur"] = row[col["wage_eur"]]
            rec["value_eur"] = row[col["value_eur"]]
            rec["pos_group"] = row[col["pos_group"]]
            rec["fc_club"] = row[col["club_name"]]
            rec["person_key"] = row[col["person_key"]]
            rec["wage_row_id"] = row[col["wage_row_id"]]
            rec["wage_source"] = row[col["wage_source"]]
            rec["wage_source_sha256"] = row[col["wage_source_sha256"]]
            rec["wage_source_mtime"] = row[col["wage_source_mtime"]]
        else:
            rec["wage_eur"] = float("nan")
            rec["value_eur"] = float("nan")
            rec["pos_group"] = pd.NA
            rec["fc_club"] = pd.NA
            rec["person_key"] = key
            rec["wage_row_id"] = pd.NA
            rec["wage_source"] = pd.NA
            rec["wage_source_sha256"] = pd.NA
            rec["wage_source_mtime"] = pd.NA
        rec["match_type"] = match_type
        rec["match_note"] = note
        rec["matched"] = match_type in _MATCHED_TYPES
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


# Half-widths of the wage uncertainty interval by match type, as a
# fraction of the point wage. Exact matches are only exposed to
# reporting rounding; other-club exact matches may carry a stale wage
# from before a transfer; alias and fuzzy matches may be the wrong
# player; proxy wages are a group median standing in for an unknown
# wage.
WAGE_INTERVAL_HALF_WIDTH = {
    "exact": 0.05,
    "exact_other_club": 0.15,
    "alias": 0.25,
    "fuzzy": 0.25,
    "proxy": 0.50,
}


def summarize_wage_matches(matched: pd.DataFrame) -> dict:
    """Counts by match_type: exact, exact_other_club, alias, fuzzy,
    ambiguous, proxy, unmatched."""
    counts = matched["match_type"].value_counts().to_dict()
    return {
        kind: int(counts.get(kind, 0)) for kind in MATCH_TYPES
    }


def assign_proxy_wages(matched: pd.DataFrame) -> pd.DataFrame:
    """Fill unmatched rows with an explicit proxy wage, flagged as such.

    The proxy is the median wage of matched players in the same
    position group, falling back to the overall matched median when
    the player has no position group or the group has no matched wage.
    Proxy rows are marked match_type "proxy" and matched stays False:
    a proxy is a labeled stand-in, not a match. Proxy rows carry the
    wage source version their median was computed from. Quarantined
    (ambiguous) rows get no proxy: the ambiguity stays visible.
    """
    out = matched.copy()
    if "match_type" not in out.columns:
        out["match_type"] = "unmatched"
    for column, default in (
            ("wage_source", CALLER_SUPPLIED_SOURCE),
            ("wage_source_sha256", ""),
            ("wage_source_mtime", ""),
            ("match_note", "")):
        if column not in out.columns:
            out[column] = default
    group_median = out.loc[out["matched"], :].groupby("pos_group")["wage_eur"].median()
    overall_median = float(out.loc[out["matched"], "wage_eur"].median())
    digest = ""
    mtime = ""
    source = WAGE_SOURCE_NAME
    if "wage_source_sha256" in out.columns:
        present = out.loc[out["matched"], "wage_source_sha256"].dropna()
        present = present[present != ""]
        if len(present):
            digest = str(present.iloc[0])
    if "wage_source_mtime" in out.columns:
        present = out.loc[out["matched"], "wage_source_mtime"].dropna()
        present = present[present != ""]
        if len(present):
            mtime = str(present.iloc[0])
    for i in out.loc[out["match_type"] == "unmatched"].index:
        group = out.at[i, "pos_group"]
        fill = group_median.get(group, float("nan"))
        if not pd.notna(fill):
            fill = overall_median
        out.at[i, "wage_eur"] = fill
        out.at[i, "match_type"] = "proxy"
        group_label = group if pd.notna(group) else "overall"
        out.at[i, "wage_source"] = (
            f"proxy: {group_label} median of {source}")
        out.at[i, "wage_source_sha256"] = digest
        out.at[i, "wage_source_mtime"] = mtime
        out.at[i, "match_note"] = (
            "proxy wage: position-group median of matched wages")
    return out


def wage_uncertainty_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Add wage_low / wage_high columns from the match type.

    Unmatched rows (no wage, no proxy assigned) get NaN intervals:
    call assign_proxy_wages first if an interval is needed for them.
    """
    out = df.copy()
    if "match_type" not in out.columns:
        raise ValueError("wage_uncertainty_intervals needs a match_type column")
    half = out["match_type"].map(WAGE_INTERVAL_HALF_WIDTH)
    out["wage_low"] = out["wage_eur"] * (1.0 - half)
    out["wage_high"] = out["wage_eur"] * (1.0 + half)
    return out


def _slice_counts(frame: pd.DataFrame) -> dict:
    """Per-slice player counts and match counts for one audit slice."""
    result = {}
    for name, sub in frame.groupby("slice", dropna=False):
        n = len(sub)
        matched = int(sub["matched"].sum()) if "matched" in sub else 0
        result[str(name)] = {
            "n": int(n),
            "matched": matched,
            "match_rate": (matched / n) if n else 0.0,
        }
    return result


def wage_join_audit(matched: pd.DataFrame, minutes=None, season=None,
                    value_col: str = "value_eur") -> dict:
    """Coverage and false-match audit for a matched frame.

    Overall counts and match rate, plus breakdowns by position group,
    club, season, available minutes and ranked-value quantile. ``minutes``
    is a DataFrame with player/team/minutes, or a Series aligned with
    ``matched``; ``season`` is a Series aligned with ``matched``, a
    column name in it, or a list of season labels. ``quarantine``
    lists the ambiguous rows for review. All values are plain Python
    types, so the result is JSON-serializable.
    """
    frame = matched.copy()
    counts = summarize_wage_matches(frame)
    n_players = len(frame)
    n_matched = int(frame["matched"].sum()) if "matched" in frame else 0
    audit: dict = {
        "counts": counts,
        "n_players": int(n_players),
        "n_matched": n_matched,
        "match_rate": (n_matched / n_players) if n_players else 0.0,
    }
    frame["slice"] = frame["pos_group"].fillna("unknown")
    audit["by_position"] = _slice_counts(frame)
    frame["slice"] = frame["team"].fillna("unknown")
    audit["by_club"] = _slice_counts(frame)
    if season is not None:
        if isinstance(season, str):
            labels = frame[season].fillna("unknown").tolist()
        elif isinstance(season, pd.Series):
            labels = season.reindex(frame.index).fillna("unknown").tolist()
        else:
            labels = list(season)
        frame["slice"] = labels
        audit["by_season"] = _slice_counts(frame)
    else:
        audit["by_season"] = {}
    if minutes is not None:
        if isinstance(minutes, pd.DataFrame):
            mins = minutes[["player", "team", "minutes"]].drop_duplicates(
                ["player", "team"])
            frame = frame.merge(mins, on=["player", "team"], how="left")
            values = frame["minutes"].fillna(0.0).tolist()
        elif isinstance(minutes, pd.Series):
            values = minutes.reindex(frame.index).fillna(0.0).tolist()
        else:
            values = [float(v) for v in minutes]
        labels = []
        for value in values:
            if value == 0.0:
                label = "0"
            elif value < 270.0:
                label = "<270"
            elif value < 900.0:
                label = "270-900"
            elif value < 1800.0:
                label = "900-1800"
            else:
                label = "1800+"
            labels.append(label)
        frame["slice"] = labels
        audit["by_minutes"] = _slice_counts(frame)
    else:
        audit["by_minutes"] = {}
    by_value: dict = {}
    if value_col in frame.columns:
        have = frame.loc[frame["matched"] & frame[value_col].notna(),
                         value_col]
        if len(have):
            order = have.sort_values().to_numpy()
            n_bins = min(5, len(order))
            chunks = np.array_split(order, n_bins)
            for i, chunk in enumerate(chunks):
                by_value[f"Q{i + 1}"] = {
                    "n": int(len(chunk)),
                    "median_value_eur": float(pd.Series(chunk).median()),
                }
    audit["by_value_quantile"] = by_value
    quarantine = []
    if "match_type" in frame.columns:
        for row in frame.loc[
                frame["match_type"] == "ambiguous"].itertuples():
            quarantine.append({
                "player": row.player,
                "team": row.team,
                "match_type": "ambiguous",
                "match_note": getattr(row, "match_note", ""),
            })
    audit["quarantine"] = quarantine
    return audit


def score_gold_set(matched: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """Score a matched frame against an independently checked gold set.

    ``gold`` needs player, team, expected_wage_eur and
    expected_match_type. Precision is the share of predicted matches
    whose wage and type agree with gold; recall is the share of gold
    matches recovered with the right wage and type. A predicted wage
    agrees when it is within 1 EUR/week of the expected wage.
    """
    if gold.duplicated(subset=["player", "team"]).any():
        raise ValueError("gold set needs one row per (player, team)")
    merged = gold.merge(
        matched[["player", "team", "wage_eur", "match_type"]],
        on=["player", "team"], how="left", suffixes=("", "_pred"))
    merged = merged.rename(columns={
        "wage_eur": "pred_wage_eur", "match_type": "pred_match_type"})
    details = []
    n_predicted = 0
    n_expected = 0
    n_correct = 0
    n_wrong_wage = 0
    n_wrong_club = 0
    n_missing = 0
    n_false_match = 0
    for row in merged.itertuples():
        expected_matched = (
            row.expected_match_type in _MATCHED_TYPES
            and pd.notna(row.expected_wage_eur))
        predicted_matched = (
            row.pred_match_type in _MATCHED_TYPES
            and pd.notna(row.pred_wage_eur))
        wage_ok = False
        if expected_matched and predicted_matched:
            wage_ok = abs(float(row.pred_wage_eur)
                          - float(row.expected_wage_eur)) <= 1.0
        type_ok = row.pred_match_type == row.expected_match_type
        correct = bool(expected_matched and predicted_matched
                       and wage_ok and type_ok)
        if predicted_matched:
            n_predicted += 1
        if expected_matched:
            n_expected += 1
        if correct:
            n_correct += 1
        if predicted_matched and expected_matched and not wage_ok:
            n_wrong_wage += 1
        if row.pred_match_type != row.expected_match_type:
            n_wrong_club += 1
        if expected_matched and not predicted_matched:
            n_missing += 1
        if predicted_matched and not expected_matched:
            n_false_match += 1
        details.append({
            "player": row.player,
            "team": row.team,
            "expected_match_type": row.expected_match_type,
            "expected_wage_eur": (
                None if pd.isna(row.expected_wage_eur)
                else float(row.expected_wage_eur)),
            "pred_match_type": row.pred_match_type,
            "pred_wage_eur": (
                None if pd.isna(row.pred_wage_eur)
                else float(row.pred_wage_eur)),
            "correct": correct,
        })
    return {
        "precision": (n_correct / n_predicted) if n_predicted else 0.0,
        "recall": (n_correct / n_expected) if n_expected else 0.0,
        "n_gold": int(len(merged)),
        "n_predicted_matched": int(n_predicted),
        "n_expected_matched": int(n_expected),
        "n_correct": int(n_correct),
        "n_wrong_wage": int(n_wrong_wage),
        "n_wrong_club": int(n_wrong_club),
        "n_missing": int(n_missing),
        "n_false_match": int(n_false_match),
        "details": details,
    }
