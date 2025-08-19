"""
micromet.format.checks
======================

AmeriFlux-format validation routines for MicroMet.

This module orchestrates structural checks on a pandas DataFrame intended for
AmeriFlux half-hourly (HH) uploads. It builds on `amf_schema` definitions
and returns a structured report that you can print, serialize, or fail on.

Key entry points
----------------
- run_amf_upload_checks(df, options=...)
- format_report_text(report)
- format_report_json(report)

Optional: units-row validation for CSVs that include a second header line.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Any

import io
import json
import numpy as np
import pandas as pd

from . import amf_schema as schema


# ============================================================================
# Options & Report types
# ============================================================================


@dataclass
class CheckOptions:
    """Configuration for upload validation."""

    required: Iterable[str] = tuple(schema.AMF_REQUIRED)
    check_units: bool = False  # compare provided units row to expected (lenient)
    units_row: Optional[List[str]] = None  # if you captured it from CSV
    missing_sentinel: float | int = schema.MISSING_SENTINEL
    illegal_sentinels: Iterable[float | int] = tuple(schema.ILLEGAL_SENTINELS)
    check_soft_limits: bool = False  # soft physics bounds (warn-level)
    radiation_consistency_tol_wm2: float = (
        75.0  # |NETRAD - components| tolerance for flag
    )
    enforce_timestamps_first: bool = True  # suggest/order TIMESTAMP_* first
    allow_disallowed_qualifiers: bool = False  # if False, mark as error
    normalize_aliases: bool = True  # normalize alias names before checks


@dataclass
class CheckReport:
    """Structured validation report."""

    ok: bool
    issues: Dict[str, Any]
    suggestions: Dict[str, Any]


# ============================================================================
# Units row helpers (optional)
# ============================================================================


def sniff_units_row_from_csv(
    path: str | Path, delimiter: str = ","
) -> Optional[List[str]]:
    """
    Attempt to read the *second* line of a CSV as a potential units row.
    Returns a list of strings if it looks like units, otherwise None.
    """
    p = Path(path)
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
        second = f.readline()
    if not first or not second:
        return None
    # crude heuristic: contains common unit tokens or looks non-alphanumeric
    maybe_units = second.strip().split(delimiter)
    joined = ",".join(maybe_units).lower()
    tokens = ("w/m2", "c", "kpa", "m/s", "deg", "unit")
    if any(tok in joined for tok in tokens):
        return maybe_units
    return None


# ============================================================================
# Low-level checks (beyond amf_schema.validate_amf_dataframe)
# ============================================================================


def check_missing_value_policy(
    df: pd.DataFrame,
    missing_sentinel: float | int,
    illegal_sentinels: Iterable[float | int],
) -> Dict[str, Any]:
    """
    Validate that only the expected missing sentinel appears (for already-sentinelized frames),
    and that illegal sentinels do not appear.
    """
    issues: Dict[str, Any] = {"illegal": [], "other_sentinels": []}

    # Illegal sentinels present?
    for v in illegal_sentinels:
        try:
            if (df == v).any(numeric_only=False).any():  # type: ignore[arg-type]
                issues["illegal"].append(v)
        except Exception:
            # Fallback string scan
            if df.astype(str).eq(str(v)).any().any():
                issues["illegal"].append(v)

    # Identify other sentinel-like constants present besides the chosen one
    # (heuristic: frequent negative constants)
    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        # Count values per column (cheap heuristic)
        others: set[float] = set()
        for c in numeric.columns:
            vc = numeric[c].value_counts(dropna=True)
            # pick small negative modes that are not the chosen sentinel
            for val, cnt in vc.items():
                if pd.isna(val):  # type: ignore
                    continue
                if (
                    isinstance(val, (int, float))
                    and val < 0
                    and val != missing_sentinel
                    and cnt > 0
                ):
                    if val in illegal_sentinels:
                        continue
                    # consider -9998, -6999, etc. as suspicious
                    if abs(val) >= 6000:
                        others.add(float(val))
        issues["other_sentinels"] = sorted(others)

    issues["ok"] = len(issues["illegal"]) == 0
    return issues


def check_soft_physical_limits(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Soft QA: counts values outside recommended plausible ranges (not strict errors).
    Uses SOFT_LIMITS from amf_schema (if available for a variable).
    """
    out_of_range: Dict[str, int] = {}
    for name, spec in schema.SOFT_LIMITS.items():
        if name not in df.columns:
            continue
        s = pd.to_numeric(df[name], errors="coerce")
        mask = pd.Series(False, index=s.index)
        if spec.min_val is not None:
            mask |= s < spec.min_val
        if spec.max_val is not None:
            mask |= s > spec.max_val
        n = int(mask.sum())
        if n > 0:
            out_of_range[name] = n
    return {"out_of_range_counts": out_of_range, "ok": len(out_of_range) == 0}


def check_radiation_component_consistency(
    df: pd.DataFrame, tol_wm2: float = 75.0
) -> Dict[str, Any]:
    """
    Compare measured NETRAD to computed (SW_IN - SW_OUT + LW_IN - LW_OUT).
    Flags rows where |diff| > tol_wm2.
    """
    req = {"SW_IN", "SW_OUT", "LW_IN", "LW_OUT", "NETRAD"}
    if not req.issubset(df.columns):
        return {"checked": False, "n_flagged": 0}
    sw_in = pd.to_numeric(df["SW_IN"], errors="coerce")
    sw_out = pd.to_numeric(df["SW_OUT"], errors="coerce")
    lw_in = pd.to_numeric(df["LW_IN"], errors="coerce")
    lw_out = pd.to_numeric(df["LW_OUT"], errors="coerce")
    r_meas = pd.to_numeric(df["NETRAD"], errors="coerce")

    comp = sw_in - sw_out + lw_in - lw_out
    diff = (r_meas - comp).abs()
    flags = diff > tol_wm2
    return {"checked": True, "n_flagged": int(flags.sum()), "tolerance_wm2": tol_wm2}


# ============================================================================
# Orchestration
# ============================================================================


def run_amf_upload_checks(
    df: pd.DataFrame, options: CheckOptions | None = None
) -> CheckReport:
    """
    Run a suite of structural and light-physics checks to assess AmeriFlux readiness.

    Returns a CheckReport with:
      - ok: overall pass/fail
      - issues: structured dict with details
      - suggestions: ordered column names, alias rename hints, etc.
    """
    opts = options or CheckOptions()

    # Normalize aliases (e.g., T_SONIC -> TA) before checking
    dfn = (
        schema.normalize_dataframe_columns(df) if opts.normalize_aliases else df.copy()
    )

    # Base validation from amf_schema
    base = schema.validate_amf_dataframe(
        dfn,
        required=opts.required,
        check_units=opts.check_units,
        units_row=opts.units_row,
    )

    issues: Dict[str, Any] = dict(base.issues)  # copy
    suggestions: Dict[str, Any] = {}

    # Missing sentinel & illegal sentinels
    mv = check_missing_value_policy(
        dfn,
        missing_sentinel=opts.missing_sentinel,
        illegal_sentinels=opts.illegal_sentinels,
    )
    issues["missing_value_policy"] = mv

    # Optional soft physics bounds
    if opts.check_soft_limits:
        soft = check_soft_physical_limits(dfn)
        issues["soft_limits"] = soft

    # Radiation component consistency
    rad = check_radiation_component_consistency(
        dfn, tol_wm2=opts.radiation_consistency_tol_wm2
    )
    issues["radiation_consistency"] = rad

    # Qualifier policy
    if not opts.allow_disallowed_qualifiers:
        has_bad = schema.has_disallowed_upload_columns(dfn)
        issues["qualifiers"]["disallowed_error"] = has_bad

    # Suggestions
    if opts.enforce_timestamps_first:
        suggestions["column_order"] = schema.suggest_upload_column_order(dfn)
    else:
        suggestions["column_order"] = list(dfn.columns)

    # Alias rename hints (only if normalization changed any names)
    alias_hints: Dict[str, str] = {}
    for c in df.columns:
        norm = schema.normalize_column_name(c) if opts.normalize_aliases else c
        if norm != c:
            alias_hints[c] = norm
    if alias_hints:
        suggestions["alias_renames"] = alias_hints

    # Qualifier cleanup suggestions
    disallowed = issues.get("qualifiers", {}).get("disallowed_qualifier_cols", [])
    if disallowed:
        suggestions["drop_or_rename"] = disallowed

    # Compose overall OK
    ok = bool(base.ok)
    ok = ok and mv.get("ok", True)
    if not opts.allow_disallowed_qualifiers:
        ok = ok and not issues["qualifiers"].get("disallowed_error", False)

    # Do not fail overall on soft physics or radiation mismatch; treat as warnings
    report = CheckReport(ok=ok, issues=issues, suggestions=suggestions)
    return report


# ============================================================================
# Pretty-printers
# ============================================================================


def format_report_text(report: CheckReport) -> str:
    """
    Render a human-readable validation summary.
    """
    lines: List[str] = []
    status = "PASS ✅" if report.ok else "FAIL ❌"
    lines.append(f"AmeriFlux Format Validation: {status}\n")

    # Missing columns
    miss = report.issues.get("missing_required", [])
    if miss:
        lines.append(f"- Missing required columns: {miss}")
    else:
        lines.append("- Required columns: OK")

    # Timestamps
    ts = report.issues.get("timestamps", {})
    lines.append(
        f"- Timestamps: uniform_30min={ts.get('uniform_30min', False)}, "
        f"cadence_unique={ts.get('cadence_minutes_unique', [])}, "
        f"end_aligned_30min={ts.get('end_aligned_30min', None)}"
    )

    # Qualifiers
    q = report.issues.get("qualifiers", {})
    disallowed = q.get("disallowed_qualifier_cols", [])
    if disallowed:
        lines.append(f"- Disallowed qualifiers present: {disallowed}")
    else:
        lines.append("- Qualifiers: OK")

    # Missing-value policy
    mv = report.issues.get("missing_value_policy", {})
    if mv:
        lines.append(
            f"- Missing value policy: illegal={mv.get('illegal', [])}, "
            f"other_sentinels={mv.get('other_sentinels', [])}"
        )

    # Units (optional)
    if "units_row_checked" in report.issues:
        mism = report.issues.get("units_mismatches", {})
        lines.append(f"- Units row mismatches: {mism}")

    # Soft limits (warnings)
    soft = report.issues.get("soft_limits", {})
    oor = soft.get("out_of_range_counts", {}) if soft else {}
    if oor:
        lines.append(f"- Soft limit warnings (out-of-range counts): {oor}")

    # Radiation consistency
    rad = report.issues.get("radiation_consistency", {})
    if rad.get("checked", False):
        lines.append(
            f"- Radiation component consistency: flagged={rad.get('n_flagged')} "
            f"(tol={rad.get('tolerance_wm2')} W/m²)"
        )

    # Suggestions
    if report.suggestions:
        lines.append("\nSuggestions:")
        order = report.suggestions.get("column_order")
        if order:
            lines.append(f"  * Suggested column order (timestamps first): {order}")
        if "alias_renames" in report.suggestions:
            lines.append(f"  * Alias renames: {report.suggestions['alias_renames']}")
        if "drop_or_rename" in report.suggestions:
            lines.append(
                f"  * Drop or rename disallowed qualifier columns: {report.suggestions['drop_or_rename']}"
            )

    return "\n".join(lines)


def format_report_json(report: CheckReport, indent: int = 2) -> str:
    """
    Serialize the report to JSON (strings for non-serializable objects).
    """

    def _default(o):
        if isinstance(o, (np.generic,)):
            return o.item()
        return str(o)

    return json.dumps(
        {"ok": report.ok, "issues": report.issues, "suggestions": report.suggestions},
        indent=indent,
        default=_default,
    )


# ============================================================================
# Convenience: run checks from a CSV on disk
# ============================================================================


def run_checks_from_csv(
    path: str | Path, *, delimiter: str = ",", options: CheckOptions | None = None
) -> CheckReport:
    """
    Load a CSV (optionally with a second header as units) and run checks.
    The function will attempt to sniff a units row if options.check_units is True.
    """
    p = Path(path)
    header = 0  # assume first row is names
    df = pd.read_csv(p, header=header)

    opts = options or CheckOptions()
    units_row = opts.units_row
    if opts.check_units and units_row is None:
        maybe = sniff_units_row_from_csv(p, delimiter=delimiter)
        # If a units row exists, drop it from df (we read the entire file again without it)
        if maybe is not None:
            units_row = maybe
            # Reload skipping the second line (header + units)
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                first = f.readline()
                _ = f.readline()  # units row
                rest = f.read()
            df = pd.read_csv(io.StringIO(first + rest), header=0)

    # Run checks (units_row provided if found)
    run_opts = CheckOptions(**{**asdict(opts), "units_row": units_row})
    return run_amf_upload_checks(df, options=run_opts)
