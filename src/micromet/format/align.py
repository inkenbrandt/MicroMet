"""
micromet.format.align
=====================

Timestamp alignment utilities for AmeriFlux half-hourly (HH) workflows.

Goals
-----
* Strict 30-minute cadence (TIMESTAMP_START, TIMESTAMP_END)
* No DST jumps (work in local STANDARD time; fixed UTC offset)
* No overlaps or duplicates
* Deterministic rounding/snap to boundary with small tolerance
* Non-destructive by default; repair is opt-in

Key API
-------
- AlignmentOptions: knobs for snapping & repair
- AlignmentReport: structured summary of what was found/fixed
- detect_cadence(df): quick cadence inspection
- snap_to_halfhour(df, ...): snap TIMESTAMP_START to exact 30-min boundaries
- enforce_halfhour_cadence(df, options): full repair (reindex + END rebuild)
- audit_alignment(df): rich diagnostics without changing the data
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any, List, Tuple

import numpy as np
import pandas as pd

TIMESTAMP_START = "TIMESTAMP_START"
TIMESTAMP_END = "TIMESTAMP_END"
HALFHOUR_MINUTES = 30


# =====================================================================
# Options & Report
# =====================================================================


@dataclass
class AlignmentOptions:
    """Behavior flags for time alignment."""

    # Snap tolerance: values within ±tolerance_secs of a perfect boundary will be snapped.
    tolerance_secs: int = 60
    # If True, reindex to a perfect 30-min grid between min/max START, inserting missing rows (NaN values).
    reindex_to_grid: bool = True
    # If True, drop duplicate START stamps (keep first).
    drop_duplicates: bool = True
    # If True, rebuild TIMESTAMP_END as START + 30 min (always recommended for AMF).
    rebuild_end: bool = True
    # If True, enforce monotonic non-decreasing START after snapping (sort by START if needed).
    sort_by_start: bool = True


@dataclass
class AlignmentReport:
    """Structured summary of alignment results."""

    ok: bool
    n_rows_in: int
    n_rows_out: int
    start_min: Optional[pd.Timestamp]
    start_max: Optional[pd.Timestamp]
    cadence_unique_minutes: List[int]
    uniform_30min: bool
    n_duplicates_dropped: int
    n_rows_inserted: int
    n_rows_removed: int
    n_snapped: int
    suspected_dst_jumps: int
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "n_rows_in": self.n_rows_in,
            "n_rows_out": self.n_rows_out,
            "start_min": str(self.start_min) if self.start_min is not None else None,
            "start_max": str(self.start_max) if self.start_max is not None else None,
            "cadence_unique_minutes": self.cadence_unique_minutes,
            "uniform_30min": self.uniform_30min,
            "n_duplicates_dropped": self.n_duplicates_dropped,
            "n_rows_inserted": self.n_rows_inserted,
            "n_rows_removed": self.n_rows_removed,
            "n_snapped": self.n_snapped,
            "suspected_dst_jumps": self.suspected_dst_jumps,
            "notes": self.notes,
        }


# =====================================================================
# Internals
# =====================================================================


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _snap_to_halfhour_vectorized(
    ts: pd.Series, tolerance_secs: int
) -> Tuple[pd.Series, int]:
    """
    Snap timestamps to the nearest 30-min boundary if within tolerance.
    Returns (snapped_series, n_snapped).
    """
    # Compute minutes since day start
    s = pd.to_datetime(ts, errors="coerce")
    # Epoch seconds for integer math
    epoch = (s.view("int64") // 10**9).astype(
        "float64"
    )  # seconds since epoch (NaN-safe)
    # 30-min in seconds
    step = HALFHOUR_MINUTES * 60
    # Nearest boundary in seconds
    nearest = np.round(epoch / step) * step
    delta = np.abs(epoch - nearest)
    mask = (delta <= tolerance_secs) & s.notna()
    snapped = s.copy()
    snapped.loc[mask] = pd.to_datetime(nearest[mask], unit="s", utc=True).tz_convert(None)  # type: ignore
    return snapped, int(mask.sum())


def _detect_dst_like_jumps(ts: pd.Series) -> int:
    """
    Count diffs that look like DST transitions (±60 minutes off the nominal cadence).
    This should be zero if everything is already in local STANDARD time (fixed offset).
    """
    diffs = ts.diff().dt.total_seconds().div(60).dropna()
    # Any 60-min anomalies near 90 or 0 minutes that imply a ±60 shift from 30?
    # We simply count diffs that are not {30} and are multiples of 60.
    suspect = diffs[(diffs % 60 == 0) & (diffs != 30)]
    return int(suspect.size)


def _unique_cadence_minutes(ts: pd.Series) -> List[int]:
    diffs = ts.diff().dropna().dt.total_seconds().div(60)
    uniq = sorted({int(v) for v in diffs[np.isfinite(diffs)]})
    return uniq


# =====================================================================
# Public helpers
# =====================================================================


def detect_cadence(
    df: pd.DataFrame, start_col: str = TIMESTAMP_START
) -> Dict[str, Any]:
    """
    Inspect timestamp cadence and duplicates without modifying data.
    """
    issues: Dict[str, Any] = {}
    if start_col not in df.columns:
        issues.update(
            {
                "present": False,
                "uniform_30min": False,
                "cadence_unique_minutes": [],
                "n_duplicates": 0,
                "suspected_dst_jumps": 0,
            }
        )
        return issues

    ts = _to_datetime(df[start_col])
    uniq = _unique_cadence_minutes(ts)
    issues["present"] = True
    issues["cadence_unique_minutes"] = uniq
    issues["uniform_30min"] = len(uniq) == 1 and uniq[0] == HALFHOUR_MINUTES

    # Duplicates
    issues["n_duplicates"] = int(ts.duplicated().sum())

    # DST-like jumps
    issues["suspected_dst_jumps"] = _detect_dst_like_jumps(ts)

    return issues


def snap_to_halfhour(
    df: pd.DataFrame,
    *,
    start_col: str = TIMESTAMP_START,
    options: AlignmentOptions | None = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Snap TIMESTAMP_START to exact 30-min boundaries within a given tolerance.
    Returns (df_out, n_snapped). Does NOT reindex or drop duplicates.
    """
    opts = options or AlignmentOptions()
    if start_col not in df.columns:
        return df.copy(), 0
    out = df.copy()
    s = _to_datetime(out[start_col])
    snapped, n_snapped = _snap_to_halfhour_vectorized(s, opts.tolerance_secs)
    out[start_col] = snapped
    if opts.sort_by_start:
        out = out.sort_values(start_col, kind="mergesort", ignore_index=True)
    return out, n_snapped


def enforce_halfhour_cadence(
    df: pd.DataFrame,
    *,
    start_col: str = TIMESTAMP_START,
    end_col: str = TIMESTAMP_END,
    options: AlignmentOptions | None = None,
) -> Tuple[pd.DataFrame, AlignmentReport]:
    """
    Full alignment pass:
      1) Snap START to exact 30-min boundaries within tolerance
      2) Sort by START (stable)
      3) Drop duplicate START (keep first) if requested
      4) Reindex to perfect 30-min grid from min to max START (if requested)
      5) Rebuild END = START + 30 min (if requested)

    Returns (df_out, report).
    """
    opts = options or AlignmentOptions()
    notes: List[str] = []

    if start_col not in df.columns:
        rep = AlignmentReport(
            ok=False,
            n_rows_in=len(df),
            n_rows_out=len(df),
            start_min=None,
            start_max=None,
            cadence_unique_minutes=[],
            uniform_30min=False,
            n_duplicates_dropped=0,
            n_rows_inserted=0,
            n_rows_removed=0,
            n_snapped=0,
            suspected_dst_jumps=0,
            notes=["Missing TIMESTAMP_START"],
        )
        return df.copy(), rep

    # Input stats
    n_in = len(df)
    ts_in = _to_datetime(df[start_col])
    start_min, start_max = (ts_in.min(), ts_in.max()) if n_in else (None, None)

    # 1) Snap
    out, n_snapped = snap_to_halfhour(df, start_col=start_col, options=opts)
    if n_snapped:
        notes.append(
            f"Snapped {n_snapped} timestamps to 30-min boundaries (±{opts.tolerance_secs}s)."
        )

    # 2) Sort
    if opts.sort_by_start:
        out = out.sort_values(start_col, kind="mergesort", ignore_index=True)

    # 3) Drop duplicates (keep first)
    s = _to_datetime(out[start_col])
    dup_mask = s.duplicated(keep="first")
    n_dups = int(dup_mask.sum()) if opts.drop_duplicates else 0
    if opts.drop_duplicates and n_dups:
        out = out.loc[~dup_mask].reset_index(drop=True)
        notes.append(f"Dropped {n_dups} duplicate START rows (kept first occurrence).")

    # 4) Reindex to perfect 30-min grid between min and max
    n_inserted = 0
    n_removed = n_dups  # removed so far
    if opts.reindex_to_grid and len(out):
        s = _to_datetime(out[start_col])
        idx = pd.date_range(s.min(), s.max(), freq=f"{HALFHOUR_MINUTES}T")
        before = len(out)
        out = out.set_index(s).reindex(idx).rename_axis(start_col).reset_index()
        n_inserted = len(out) - before
        if n_inserted:
            notes.append(
                f"Inserted {n_inserted} missing rows to enforce perfect 30-min cadence."
            )

    # 5) Rebuild END
    if opts.rebuild_end:
        out[end_col] = pd.to_datetime(out[start_col]) + pd.Timedelta(
            minutes=HALFHOUR_MINUTES
        )

    # Post-checks
    ts_out = _to_datetime(out[start_col])
    uniq = _unique_cadence_minutes(ts_out)
    uniform = len(uniq) == 1 and uniq[0] == HALFHOUR_MINUTES
    suspected_dst = _detect_dst_like_jumps(ts_out)
    if suspected_dst:
        notes.append(
            f"Detected {suspected_dst} DST-like jump(s); ensure input is local STANDARD time."
        )

    rep = AlignmentReport(
        ok=uniform and suspected_dst == 0,
        n_rows_in=n_in,
        n_rows_out=len(out),
        start_min=ts_out.min() if len(out) else None,
        start_max=ts_out.max() if len(out) else None,
        cadence_unique_minutes=uniq,
        uniform_30min=uniform,
        n_duplicates_dropped=n_dups,
        n_rows_inserted=n_inserted,
        n_rows_removed=n_removed,
        n_snapped=n_snapped,
        suspected_dst_jumps=suspected_dst,
        notes=notes,
    )
    return out, rep


def audit_alignment(
    df: pd.DataFrame, start_col: str = TIMESTAMP_START
) -> AlignmentReport:
    """
    Read-only audit of cadence/duplicates/DST-like jumps. Does not modify data.
    """
    if start_col not in df.columns:
        return AlignmentReport(
            ok=False,
            n_rows_in=len(df),
            n_rows_out=len(df),
            start_min=None,
            start_max=None,
            cadence_unique_minutes=[],
            uniform_30min=False,
            n_duplicates_dropped=0,
            n_rows_inserted=0,
            n_rows_removed=0,
            n_snapped=0,
            suspected_dst_jumps=0,
            notes=["Missing TIMESTAMP_START"],
        )
    s = _to_datetime(df[start_col])
    uniq = _unique_cadence_minutes(s)
    uniform = len(uniq) == 1 and uniq[0] == HALFHOUR_MINUTES
    suspected_dst = _detect_dst_like_jumps(s)
    return AlignmentReport(
        ok=uniform and suspected_dst == 0,
        n_rows_in=len(df),
        n_rows_out=len(df),
        start_min=s.min(),
        start_max=s.max(),
        cadence_unique_minutes=uniq,
        uniform_30min=uniform,
        n_duplicates_dropped=0,
        n_rows_inserted=0,
        n_rows_removed=0,
        n_snapped=0,
        suspected_dst_jumps=suspected_dst,
        notes=(
            []
            if uniform and suspected_dst == 0
            else ["Cadence not uniform 30min or DST-like jumps detected."]
        ),
    )


# =====================================================================
# Convenience printer
# =====================================================================


def format_report_text(report: AlignmentReport) -> str:
    status = "PASS ✅" if report.ok else "WARN ⚠️"
    lines = [
        f"Alignment Report: {status}",
        f"- Rows in/out: {report.n_rows_in} → {report.n_rows_out}",
        f"- Start range: {report.start_min} … {report.start_max}",
        f"- Cadence unique (min): {report.cadence_unique_minutes}",
        f"- Uniform 30-min: {report.uniform_30min}",
        f"- Duplicates dropped: {report.n_duplicates_dropped}",
        f"- Rows inserted (grid): {report.n_rows_inserted}",
        f"- Rows removed: {report.n_rows_removed}",
        f"- Timestamps snapped: {report.n_snapped}",
        f"- Suspected DST jumps: {report.suspected_dst_jumps}",
    ]
    if report.notes:
        lines.append("- Notes:")
        lines.extend([f"  * {n}" for n in report.notes])
    return "\n".join(lines)
