"""
micromet.format.amf_schema
==========================

AmeriFlux base variable schema & helpers used across MicroMet.

This module defines:
- Base variable names and canonical units
- Allowed/disallowed qualifiers (suffixes)
- Common alias → base-name normalization (e.g., T_SONIC → TA, Rn → NETRAD)
- Validation utilities for upload readiness (structure, cadence, qualifiers)
- Column reordering helpers (TIMESTAMP_* first)

Notes
-----
* AmeriFlux half-hourly (HH) files must include TIMESTAMP_START and TIMESTAMP_END.
* Missing data sentinel for upload CSVs is -9999 (numeric fields). Analytic CSVs keep NaN.
* Qualifiers: site uploads should generally avoid `_PI` and `_QC` suffixes. `_F` is allowed
  for gap-filled series when explicitly intended (do not overuse in upload-bound files).

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


# =====================================================================
# Canonical names, units, qualifiers, time rules
# =====================================================================

TIMESTAMP_START = "TIMESTAMP_START"
TIMESTAMP_END = "TIMESTAMP_END"

# Required for a minimal HH energy/radiation/temperature set
AMF_REQUIRED: List[str] = [
    TIMESTAMP_START,
    TIMESTAMP_END,
    "SW_IN",
    "SW_OUT",
    "LW_IN",
    "LW_OUT",
    "NETRAD",
    "H",
    "LE",
    "G",
    "TA",
]

# Canonical units for common variables (extend as needed)
AMF_UNITS: Dict[str, str] = {
    TIMESTAMP_START: "",
    TIMESTAMP_END: "",
    "SW_IN": "W/m2",
    "SW_OUT": "W/m2",
    "LW_IN": "W/m2",
    "LW_OUT": "W/m2",
    "NETRAD": "W/m2",
    "H": "W/m2",
    "LE": "W/m2",
    "G": "W/m2",
    "TA": "C",
}

# Allowed/Disallowed qualifiers
ALLOWED_QUALIFIERS = {"_F"}  # site-upload acceptable (gap-filled series)
DISALLOWED_QUALIFIERS = {"_PI", "_QC"}  # network-reserved; block in upload outputs

# Time cadence and missing sentinel
HALFHOUR_MINUTES = 30
MISSING_SENTINEL = -9999
ILLEGAL_SENTINELS = {-8888, -6999}  # commonly encountered but not allowed


# =====================================================================
# Aliases / Normalization
# =====================================================================

# Common alias→base name mappings seen in field loggers & other toolchains
# Note: Matching is case-insensitive; normalization will compare upper-cased keys.
ALIAS_TO_BASE_UPPER: Dict[str, str] = {
    # Timestamps
    "TIMESTAMP": TIMESTAMP_START,
    "DATE_TIME": TIMESTAMP_START,
    "DATETIME": TIMESTAMP_START,
    "DATE": TIMESTAMP_START,
    # Shortwave radiation
    "SWIN": "SW_IN",
    "SW_IN_1_1_1": "SW_IN",
    "SW_IN_1_1_1_Avg": "SW_IN",
    "RG": "SW_IN",  # common pyranometer shorthand
    "SWGLOBAL": "SW_IN",
    # Shortwave out
    "SWOUT": "SW_OUT",
    "SW_OUT_1_1_1": "SW_OUT",
    # Longwave in/out
    "LWIN": "LW_IN",
    "LW_IN_1_1_1": "LW_IN",
    "LWOUT": "LW_OUT",
    "LW_OUT_1_1_1": "LW_OUT",
    # Net radiation
    "NETRAD_1_1_1": "NETRAD",
    "RN": "NETRAD",
    "RNET": "NETRAD",
    # Turbulent fluxes
    "SENSIBLE_HEAT_FLUX": "H",
    "SH": "H",
    "H_1_1_1": "H",
    "LATENT_HEAT_FLUX": "LE",
    "LH": "LE",
    "LE_1_1_1": "LE",
    # Ground heat flux
    "G0": "G",
    "SHF": "G",
    "G_1_1_1": "G",
    # Air temperature
    "T_SONIC": "TA",
    "TA_1_1_1": "TA",
    "TA_AIR": "TA",
    "TAIR": "TA",
}

# =====================================================================
# Specs & helpers
# =====================================================================


@dataclass(frozen=True)
class VariableSpec:
    """Specification for a single variable."""

    name: str
    units: str = ""
    dtype: str = "number"  # "number" | "string" (timestamps are string in upload CSV)
    min_val: Optional[float] = None
    max_val: Optional[float] = None


# Optional per-variable soft bounds (not enforced in writer; may be used in QA)
SOFT_LIMITS: Dict[str, VariableSpec] = {
    "SW_IN": VariableSpec("SW_IN", "W/m2", "number", 0.0, 1400.0),
    "SW_OUT": VariableSpec("SW_OUT", "W/m2", "number", 0.0, 1000.0),
    "LW_IN": VariableSpec("LW_IN", "W/m2", "number", 100.0, 500.0),
    "LW_OUT": VariableSpec("LW_OUT", "W/m2", "number", 100.0, 700.0),
    "NETRAD": VariableSpec("NETRAD", "W/m2", "number", -300.0, 1000.0),
    "H": VariableSpec("H", "W/m2", "number", -400.0, 1000.0),
    "LE": VariableSpec("LE", "W/m2", "number", -400.0, 1000.0),
    "G": VariableSpec("G", "W/m2", "number", -300.0, 300.0),
    "TA": VariableSpec("TA", "C", "number", -50.0, 50.0),
}

BASE_NAMES: set[str] = set(AMF_UNITS.keys())


def split_base_and_qualifier(col_name: str) -> Tuple[str, str]:
    """
    Split a column name into (base, qualifier_suffix).
    Example: "LE_F" -> ("LE", "_F"); "H" -> ("H", "")
    """
    name = col_name.strip()
    for q in sorted(ALLOWED_QUALIFIERS | DISALLOWED_QUALIFIERS, key=len, reverse=True):
        if name.endswith(q):
            return name[: -len(q)], q
    return name, ""


def normalize_column_name(name: str) -> str:
    """
    Normalize a column name to AmeriFlux base if possible using alias table.
    Keeps qualifier suffix (e.g., _F) intact.

    Examples:
        "t_sonic" -> "TA"
        "Rn" -> "NETRAD"
        "SW_IN_1_1_1" -> "SW_IN"
        "LE_F" -> "LE_F"   (base normalized, qualifier preserved)
    """
    base, q = split_base_and_qualifier(name)
    key = base.strip().upper()
    new_base = ALIAS_TO_BASE_UPPER.get(key, base.strip())  # default to original base
    return f"{new_base}{q}"


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with normalized column names using alias table
    (keeps original order).
    """
    mapping = {c: normalize_column_name(c) for c in df.columns}
    return df.rename(columns=mapping)


def reorder_with_timestamps_first(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns to have TIMESTAMP_START, TIMESTAMP_END first.
    """
    first = [c for c in [TIMESTAMP_START, TIMESTAMP_END] if c in df.columns]
    rest = [c for c in df.columns if c not in first]
    return df.loc[:, first + rest]


# =====================================================================
# Validation utilities
# =====================================================================


@dataclass
class ValidationResult:
    ok: bool
    issues: Dict[str, object]

    def __bool__(self) -> bool:  # so if result: ...
        return self.ok


def _check_required_columns(df: pd.DataFrame, required: Iterable[str]) -> List[str]:
    return [c for c in required if c not in df.columns]


def _check_illegal_sentinels(df: pd.DataFrame) -> bool:
    """
    True if any illegal sentinel values appear anywhere in df (numeric or string-cast).
    """
    # Check numeric
    illegal_found = False
    for v in ILLEGAL_SENTINELS:
        try:
            if (df == v).any(numeric_only=False).any():  # type: ignore[arg-type]
                illegal_found = True
                break
        except Exception:
            # Some pandas versions require per-column checks; fall through
            pass
        # string-cast scan (for text columns)
        if df.astype(str).eq(str(v)).any().any():
            illegal_found = True
            break
    return illegal_found


def _check_timestamps(df: pd.DataFrame) -> Dict[str, object]:
    """
    Validate timestamp presence and cadence.
    Assumes timestamps already localized to local STANDARD time (no DST).
    """
    issues: Dict[str, object] = {}

    if TIMESTAMP_START not in df.columns:
        issues["timestamp_start_present"] = False
        return issues

    try:
        ts = pd.to_datetime(df[TIMESTAMP_START], errors="coerce")
        issues["timestamp_start_present"] = True
        issues["timestamp_start_nulls"] = int(ts.isna().sum())
    except Exception as e:
        issues["timestamp_start_present"] = False
        issues["timestamp_parse_error"] = str(e)
        return issues

    # 30-min cadence check (ignoring NA)
    diffs = ts.diff().dropna().dt.total_seconds().div(60)
    unique = sorted(set(int(x) for x in diffs.unique() if np.isfinite(x)))
    issues["cadence_minutes_unique"] = unique
    issues["uniform_30min"] = len(unique) == 1 and unique[0] == HALFHOUR_MINUTES

    # End timestamps if present
    if TIMESTAMP_END in df.columns:
        te = pd.to_datetime(df[TIMESTAMP_END], errors="coerce")
        dt = (te - ts).dt.total_seconds().div(60)
        # Expect exactly 30
        ok_end = bool(np.nan_to_num(dt, nan=HALFHOUR_MINUTES).eq(HALFHOUR_MINUTES).all())  # type: ignore[return-value]
        issues["end_aligned_30min"] = ok_end
        issues["timestamp_end_nulls"] = int(te.isna().sum())
    else:
        issues["timestamp_end_present"] = False

    return issues


def _check_qualifiers(df: pd.DataFrame) -> Dict[str, object]:
    """
    Return any columns with disallowed qualifiers, and summary of qualifier usage.
    """
    disallowed: List[str] = []
    allowed: List[str] = []
    bare: List[str] = []
    for c in df.columns:
        base, q = split_base_and_qualifier(c)
        if q in DISALLOWED_QUALIFIERS:
            disallowed.append(c)
        elif q in ALLOWED_QUALIFIERS:
            allowed.append(c)
        else:
            bare.append(c)
    return {
        "disallowed_qualifier_cols": disallowed,
        "allowed_qualifier_cols": allowed,
        "bare_cols": bare,
    }


def _check_units_row(
    units_row: Optional[List[str]], cols: List[str]
) -> Dict[str, object]:
    """
    Optionally check a supplied units row (second header) against AMF_UNITS.
    """
    if units_row is None:
        return {"units_row_checked": False}

    mismatches: Dict[str, Tuple[str, str]] = {}
    for c, u in zip(cols, units_row):
        exp = AMF_UNITS.get(c, "")
        # Be lenient: only report when both sides are non-empty and differ
        if exp and u and u != exp:
            mismatches[c] = (u, exp)
    return {
        "units_row_checked": True,
        "units_mismatches": mismatches,
    }


def validate_amf_dataframe(
    df: pd.DataFrame,
    *,
    required: Iterable[str] = AMF_REQUIRED,
    check_units: bool = False,
    units_row: Optional[List[str]] = None,
) -> ValidationResult:
    """
    Validate a DataFrame against core AmeriFlux structural expectations.

    Parameters
    ----------
    df : pd.DataFrame
    required : iterable of required columns to enforce
    check_units : bool
        If True, compare a provided units_row with AMF_UNITS (lenient).
    units_row : Optional[List[str]]
        Sequence of units strings matching df.columns order (for 2nd header row check).

    Returns
    -------
    ValidationResult with issues dict containing:
      - missing_required: list[str]
      - illegal_sentinels_present: bool
      - timestamps: dict (presence, cadence, etc.)
      - qualifiers: dict (disallowed, allowed, bare counts)
      - units_row_checked/units_mismatches (if check_units)
    """
    issues: Dict[str, object] = {}

    # Normalize columns for alias handling before checking required
    dfn = normalize_dataframe_columns(df)

    # Required columns
    missing = _check_required_columns(dfn, required)
    issues["missing_required"] = missing

    # Missing sentinel sanity (for already-sentinelized frames)
    issues["illegal_sentinels_present"] = _check_illegal_sentinels(dfn)

    # Timestamps
    issues["timestamps"] = _check_timestamps(dfn)

    # Qualifiers
    issues["qualifiers"] = _check_qualifiers(dfn)

    # Units (optional)
    if check_units:
        issues |= _check_units_row(units_row, list(dfn.columns))

    ok = (
        len(missing) == 0
        and not issues["illegal_sentinels_present"]
        and issues["timestamps"].get("uniform_30min", False)  # type: ignore[return-value]
    )  # type: ignore[return-value]

    return ValidationResult(ok=bool(ok), issues=issues)


# =====================================================================
# Convenience
# =====================================================================


def suggest_upload_column_order(df: pd.DataFrame) -> List[str]:
    """
    Suggest a column order with TIMESTAMP_* first, followed by other columns in
    current order (normalized names).
    """
    dfn = normalize_dataframe_columns(df)
    first = [c for c in [TIMESTAMP_START, TIMESTAMP_END] if c in dfn.columns]
    rest = [c for c in dfn.columns if c not in first]
    return first + rest


def has_disallowed_upload_columns(df: pd.DataFrame) -> bool:
    """
    True if the DataFrame includes columns with disallowed qualifiers.
    """
    q = _check_qualifiers(df)
    return len(q.get("disallowed_qualifier_cols", [])) > 0  # type: ignore[return-value]


def normalize_and_reorder(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize aliases to AmeriFlux base names and reorder with timestamps first.
    """
    return reorder_with_timestamps_first(normalize_dataframe_columns(df))


# =====================================================================
# Example of expanding schema (future-proofing)
# =====================================================================

# You can extend the schema by adding more base variables and units here,
# then updating AMF_REQUIRED as appropriate for your upload target.

# Example:
# AMF_UNITS.update({
#     "PA": "kPa",    # air pressure
#     "RH": "%",      # relative humidity
#     "WS": "m/s",    # wind speed
#     "WD": "degrees" # wind direction
# })
# Then optionally add to AMF_REQUIRED if needed for a given pipeline.
