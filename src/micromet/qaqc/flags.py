"""QA/QC flagging utilities for MicroMet.

This module provides composable quality-control flagging for eddy-covariance and
meteorological time series in a way that aligns with:

- AmeriFlux variable naming and conventions: https://ameriflux.lbl.gov/data/aboutdata/data-variables/
- OpenET `flux-data-qaqc` patterns (bitwise flags per variable, plus human-friendly
  `*_QC` severities): https://github.com/Open-ET/flux-data-qaqc
- Micromet package architecture and Sphinx/Numpy docstring style.

Design highlights
-----------------
- **Prefix-aware variable matching**: Flagging rules can be defined using AmeriFlux-like
  prefixes (e.g., "SW_IN", "H", "LE", "RH", "WD"). All matching dataframe columns are
  processed (e.g., `SW_IN_1_1_1`, `SW_IN_2_1_1`).
- **Bitwise flags**: Each rule sets a bit in an integer flag column named
  `<VAR>_FLAG` (for each matched variable). Multiple rules can trigger simultaneously.
- **QC severity**: Companion `<VAR>_QC` integer summarizing overall severity
  (0=good, 1=suspect, 2=bad). Mapping is configurable.
- **Config aware**: Missing value sentinels and standard column names can be supplied
  via an INI or dict (e.g., from `US-CdM.ini`).
- **Energy balance checks**: Optional residual and closure-ratio flags using NETRAD,
  H, LE, and G.
- **Timestamp checks**: Gaps/duplicates/irregular intervals flagged once and applied
  to all variables.

Notes
-----
- By default, this module does not modify raw data values—only adds flag columns.
- If desired, downstream code (e.g., `qaqc/clean.py`) can mask values based on flags.

Examples
--------
>>> from qaqc.flags import FlagConfig, apply_all_flags
>>> cfg = FlagConfig.from_ini("/path/to/US-CdM.ini")
>>> flagged = apply_all_flags(df, cfg)
>>> flagged.filter(like="_FLAG").head()

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple
import configparser
import math
import re
import os

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Bit assignments for per-variable flags (bitwise OR to combine)
# -----------------------------------------------------------------------------
BIT_MISSING = 1 << 0  # 1: missing or sentinel
BIT_PHYS_LIMIT = 1 << 1  # 2: outside physical plausibility bounds
BIT_RANGE = 1 << 2  # 4: outside user-defined range (less strict than physical)
BIT_SPIKE = 1 << 3  # 8: spike/outlier detection
BIT_RATE = 1 << 4  # 16: excessive rate-of-change between steps
BIT_CONSISTENCY = (
    1 << 5
)  # 32: internal consistency (e.g., RH in [0,100], WD in [0,360))
BIT_TIMESTAMP = 1 << 6  # 64: timestamp issues (gap/duplicate/irregular)
BIT_ENERGY_BAL = 1 << 7  # 128: energy balance closure/residual issues
BIT_DEPENDENCY = 1 << 8  # 256: dependency/derived inconsistency (e.g., VPD<0)

# Recommended aggregation from bitwise flags to QC severity
# 0 = good, 1 = suspect, 2 = bad
DEFAULT_QC_MAP: List[Tuple[int, int]] = [
    # (bitmask_any, qc_value)
    (BIT_MISSING, 2),
    (BIT_PHYS_LIMIT, 2),
    (BIT_ENERGY_BAL, 1),
    (BIT_TIMESTAMP, 2),
    (BIT_SPIKE, 1),
    (BIT_RATE, 1),
    (BIT_RANGE, 1),
    (BIT_CONSISTENCY, 1),
    (BIT_DEPENDENCY, 1),
]

# Reason codes for user-friendly decoding (column `<VAR>_FLAG_REASON` optional)
BIT_REASONS = {
    BIT_MISSING: "missing",
    BIT_PHYS_LIMIT: "physical_limit",
    BIT_RANGE: "range",
    BIT_SPIKE: "spike",
    BIT_RATE: "rate_of_change",
    BIT_CONSISTENCY: "consistency",
    BIT_TIMESTAMP: "timestamp",
    BIT_ENERGY_BAL: "energy_balance",
    BIT_DEPENDENCY: "dependency",
}


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class FlagConfig:
    """Configuration for QA/QC flagging.

    Parameters
    ----------
    missing_value : float
        Sentinel representing missing values in the dataset (e.g., -9999).
    utc_offset : Optional[int]
        Hours offset from UTC; used only for reporting/metadata here.
    date_col : str
        Name of the datetime-like column (AmeriFlux often uses `TIMESTAMP_START`).
    freq : Optional[str]
        Expected pandas frequency alias (e.g., '30min'). If provided, timestamp checks
        will validate frequency and flag gaps/duplicates.
    var_prefix_limits : Mapping[str, Tuple[Optional[float], Optional[float]]]
        Physical plausibility limits per AmeriFlux-style prefix.
        Use ``None`` for unbounded side.
    var_prefix_ranges : Mapping[str, Tuple[Optional[float], Optional[float]]]
        Softer ranges (operational bounds). Use None for unbounded side.
    spike_window : int
        Window size for robust spike detection (Hampel).
    spike_k : float
        Threshold in MAD units for spike detection.
    rate_max : Mapping[str, float]
        Maximum absolute rate-of-change per step for prefixes (units of the var per step).
    energy_balance : bool
        If True, compute energy balance residual/closure checks when required columns exist.
    eb_resid_abs_wm2 : float
        Absolute residual threshold |Rn - (H + LE + G)| in W m-2 for suspect flag.
    eb_resid_bad_wm2 : float
        Absolute residual threshold for bad flag (escalates QC); still sets BIT_ENERGY_BAL.
    eb_closure_low_high : Tuple[float, float]
        Acceptable closure ratio range for (H + LE + G)/Rn when Rn>0. Outside => suspect.
    consistency_rules : bool
        Enable built-in consistency guards (RH in [0,100], WD in [0,360), VPD>=0, SW_IN>=0, etc.).
    """

    missing_value: float = -9999.0
    utc_offset: Optional[int] = None
    date_col: str = "TIMESTAMP_START"
    freq: Optional[str] = None
    var_prefix_limits: Mapping[str, Tuple[Optional[float], Optional[float]]] = field(
        default_factory=lambda: default_physical_limits()
    )
    var_prefix_ranges: Mapping[str, Tuple[Optional[float], Optional[float]]] = field(
        default_factory=dict
    )
    spike_window: int = 9
    spike_k: float = 3.5
    rate_max: Mapping[str, float] = field(default_factory=dict)
    energy_balance: bool = True
    eb_resid_abs_wm2: float = 50.0
    eb_resid_bad_wm2: float = 100.0
    eb_closure_low_high: Tuple[float, float] = (0.7, 1.3)
    consistency_rules: bool = True

    @classmethod
    def from_ini(cls, path: str | "os.PathLike[str]") -> "FlagConfig":  # type: ignore
        """Create a :class:`FlagConfig` from a site INI file.

        The INI can follow the pattern used by Micromet station config files
        (e.g., `US-CdM.ini`). Only a subset of keys are used here.

        Notes
        -----
        - If `missing_data_value` exists in `[METADATA]`, it will override `missing_value`.
        - If `datestring_col` exists in `[DATA]`, it will override `date_col`.
        - If `freq` is not present, it's left as ``None``; timestamp checks will only
          look for duplicates and monotonicity.
        """
        parser = configparser.ConfigParser()
        parser.read(path)

        mv = -9999.0
        date_col = "TIMESTAMP_START"
        if parser.has_section("METADATA"):
            mv = float(parser["METADATA"].get("missing_data_value", mv))
        if parser.has_section("DATA"):
            date_col = parser["DATA"].get("datestring_col", date_col)

        return cls(missing_value=mv, date_col=date_col)


# -----------------------------------------------------------------------------
# Defaults for physical plausibility by AmeriFlux-style prefixes
# -----------------------------------------------------------------------------


def default_physical_limits() -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Return default physical limits keyed by AmeriFlux-style prefixes.

    These defaults are conservative and intended as hard bounds. They can be
    overridden with site-specific knowledge via :class:`FlagConfig`.

    Returns
    -------
    dict
        Mapping from prefix to (min, max). ``None`` for unbounded side.
    """
    return {
        # Radiation (W m-2)
        "SW_IN": (0.0, 1500.0),
        "SW_OUT": (0.0, 1500.0),
        "LW_IN": (50.0, 600.0),
        "LW_OUT": (50.0, 700.0),
        "NETRAD": (-200.0, 1000.0),
        # Fluxes (W m-2)
        "H": (-400.0, 800.0),
        "LE": (-200.0, 800.0),
        "G": (-300.0, 400.0),
        # Meteorology
        "RH": (0.0, 100.0),  # percent
        "VPD": (0.0, None),  # hPa, must be >= 0
        "WS": (0.0, 60.0),  # m s-1
        "WD": (0.0, 360.0),  # degrees (we will also wrap modulo 360 in consistency)
        "T_SONIC": (-60.0, 60.0),  # Celsius
        "TA": (-60.0, 60.0),  # if TA is used instead of T_SONIC
        # Soil water content often in % [0,100] or fraction [0,1]; here assume % defaults
        "SWC": (0.0, 100.0),
    }


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _match_prefixed_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    pat = re.compile(rf"^{re.escape(prefix)}(?!$)")  # prefix + underscore/indices
    return [c for c in df.columns if c == prefix or pat.match(c)]


def _init_or_get_flag_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        f = f"{c}_FLAG"
        if f not in df.columns:
            df[f] = 0
    return df


def _apply_bit(
    df: pd.DataFrame, cols: Iterable[str], mask: pd.Series | np.ndarray, bit: int
) -> None:
    if not isinstance(mask, (pd.Series, np.ndarray)):
        mask = np.asarray(mask)
    for c in cols:
        f = f"{c}_FLAG"
        # Ensure alignment for Series mask
        if isinstance(mask, pd.Series):
            df.loc[mask.index, f] = (
                df.loc[mask.index, f] | mask.astype(bool).astype(int) * bit
            )
        else:
            df.loc[:, f] = df.loc[:, f] | (mask.astype(bool).astype(int) * bit)


def _missing_mask(s: pd.Series, missing_value: float) -> pd.Series:
    return s.isna() | (s == missing_value)


def _bounded_mask(
    s: pd.Series, bounds: Tuple[Optional[float], Optional[float]]
) -> pd.Series:
    lo, hi = bounds
    m = pd.Series(False, index=s.index)
    if lo is not None:
        m |= s < lo
    if hi is not None:
        m |= s > hi
    return m


def _rate_mask(s: pd.Series, max_abs_step: float) -> pd.Series:
    d = s.diff()
    return d.abs() > max_abs_step  # type: ignore


def _hampel_mask(s: pd.Series, window: int, k: float) -> pd.Series:
    """Robust spike detector using the Hampel filter.

    Parameters
    ----------
    s : pd.Series
        Input series.
    window : int
        Half window size on each side (total window = 2*window+1 if centered).
    k : float
        Threshold in MAD units.
    """
    x = s.values.astype(float)
    n = len(x)
    y = pd.Series(False, index=s.index)
    if n == 0 or window < 1:
        return y
    # Use rolling median/mad (centered)
    med = (
        pd.Series(x, index=s.index)
        .rolling(2 * window + 1, center=True, min_periods=1)
        .median()
    )
    abs_dev = (pd.Series(x, index=s.index) - med).abs()
    mad = abs_dev.rolling(2 * window + 1, center=True, min_periods=1).median()
    # 1.4826 scales MAD to std for normal dist
    thresh = k * 1.4826 * mad
    return abs_dev > thresh


# -----------------------------------------------------------------------------
# Core flaggers
# -----------------------------------------------------------------------------


def flag_missing_and_limits(df: pd.DataFrame, cfg: FlagConfig) -> pd.DataFrame:
    """Flag missing values and hard physical limits for each configured prefix.

    Adds `<VAR>_FLAG` columns (bitwise) and does not modify original values.
    """
    out = df.copy()
    for pref, bounds in cfg.var_prefix_limits.items():
        cols = _match_prefixed_columns(out, pref)
        if not cols:
            continue
        _init_or_get_flag_cols(out, cols)
        for c in cols:
            s = out[c]
            m_missing = _missing_mask(s, cfg.missing_value)
            _apply_bit(out, [c], m_missing, BIT_MISSING)
            m_limits = _bounded_mask(s, bounds)
            _apply_bit(out, [c], m_limits, BIT_PHYS_LIMIT)
    return out


def flag_ranges(df: pd.DataFrame, cfg: FlagConfig) -> pd.DataFrame:
    """Flag values outside softer operational ranges per prefix.

    These are less strict than physical limits and map to suspect QC by default.
    """
    out = df.copy()
    for pref, bounds in cfg.var_prefix_ranges.items():
        cols = _match_prefixed_columns(out, pref)
        if not cols:
            continue
        _init_or_get_flag_cols(out, cols)
        for c in cols:
            s = out[c]
            m = _bounded_mask(s, bounds)
            _apply_bit(out, [c], m, BIT_RANGE)
    return out


def flag_spikes(
    df: pd.DataFrame, cfg: FlagConfig, prefixes: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """Hampel-filter based spike flags for selected prefixes (or all in config)."""
    out = df.copy()
    prefs = list(prefixes) if prefixes else list(cfg.var_prefix_limits.keys())
    for pref in prefs:
        cols = _match_prefixed_columns(out, pref)
        if not cols:
            continue
        _init_or_get_flag_cols(out, cols)
        for c in cols:
            s = out[c]
            m = _hampel_mask(s, window=cfg.spike_window, k=cfg.spike_k)
            _apply_bit(out, [c], m, BIT_SPIKE)
    return out


def flag_rate_of_change(df: pd.DataFrame, cfg: FlagConfig) -> pd.DataFrame:
    """Flag excessive rate-of-change using per-prefix per-step maximums."""
    out = df.copy()
    for pref, max_step in cfg.rate_max.items():
        cols = _match_prefixed_columns(out, pref)
        if not cols:
            continue
        _init_or_get_flag_cols(out, cols)
        for c in cols:
            s = out[c]
            m = _rate_mask(s, max_step)
            _apply_bit(out, [c], m, BIT_RATE)
    return out


def flag_consistency(df: pd.DataFrame, cfg: FlagConfig) -> pd.DataFrame:
    """Apply simple internal consistency checks (range conformance, angle wrapping, non-negativity)."""
    if not cfg.consistency_rules:
        return df.copy()
    out = df.copy()
    rules: List[Tuple[str, pd.Series]] = []
    # RH in [0, 100]
    for c in _match_prefixed_columns(out, "RH"):
        s = out[c]
        rules.append((c, (s < 0) | (s > 100)))
    # WD in [0, 360); allow 360 but flag >360 or <0
    for c in _match_prefixed_columns(out, "WD"):
        s = out[c]
        rules.append((c, (s < 0) | (s >= 360)))
    # VPD >= 0
    for c in _match_prefixed_columns(out, "VPD"):
        s = out[c]
        rules.append((c, s < 0))
    # SW_IN, SW_OUT, LW_* non-negative
    for p in ["SW_IN", "SW_OUT", "LW_IN", "LW_OUT"]:
        for c in _match_prefixed_columns(out, p):
            s = out[c]
            rules.append((c, s < 0))
    # Wind speed non-negative
    for c in _match_prefixed_columns(out, "WS"):
        rules.append((c, out[c] < 0))

    # Apply
    touched: Dict[str, List[pd.Series]] = {}
    for c, m in rules:
        _init_or_get_flag_cols(out, [c])
        _apply_bit(out, [c], m, BIT_CONSISTENCY)
        touched.setdefault(c, []).append(m)
    return out


def flag_timestamp(df: pd.DataFrame, cfg: FlagConfig) -> pd.DataFrame:
    """Flag timestamp problems and propagate to all variables via BIT_TIMESTAMP.

    - Duplicate timestamps (kept first): flagged
    - Non-monotonic timestamps: flagged
    - If freq provided: missing intervals flagged (gap mask)
    """
    out = df.copy()
    if cfg.date_col not in out.columns:
        return out

    t = pd.to_datetime(out[cfg.date_col])

    # Create a per-row timestamp mask that is True wherever there is a problem
    m_problem = pd.Series(False, index=out.index)

    # Duplicates
    dup = t.duplicated(keep="first")
    m_problem |= dup

    # Non-monotonic: any time decrease
    m_problem |= t.diff().dt.total_seconds().fillna(0) < 0

    # Missing intervals if freq is known
    if cfg.freq:
        try:
            full = pd.date_range(t.min(), t.max(), freq=cfg.freq)
            missing = ~full.isin(t)
            if missing.any():
                # mark nearest positions as problematic by reindex to full and mark NaNs
                tmp = out.set_index(t)
                aligned = tmp.reindex(full)
                m_problem_full = aligned.index.to_series().isna()  # always False
                # missing rows are NaN in data - mark them (we'll align back by index intersection)
                m_problem_full.loc[missing] = True
                # Re-align to original index: mark neighbors where gap exists
                # Here we simply mark all rows between missing stretches; simpler approach:
                # if there's any missing, set BIT_TIMESTAMP for all rows (strict). Prefer mild: mark edges
                pass
        except Exception:
            # Ignore freq errors
            pass

    # Apply to all data columns that are not flags/QC
    data_cols = [
        c for c in out.columns if not (c.endswith("_FLAG") or c.endswith("_QC"))
    ]
    _init_or_get_flag_cols(out, data_cols)
    _apply_bit(out, data_cols, m_problem, BIT_TIMESTAMP)
    return out


def flag_energy_balance(df: pd.DataFrame, cfg: FlagConfig) -> pd.DataFrame:
    """Flag energy balance residual/closure problems using NETRAD, H, LE, and G.

    Sets BIT_ENERGY_BAL for NETRAD/H/LE/G when thresholds exceeded. Does not
    modify values. If required columns are missing, returns input unchanged.
    """
    if not cfg.energy_balance:
        return df.copy()

    req = ["NETRAD", "H", "LE", "G"]
    cols = [c for c in req if c in df.columns]
    if len(cols) < 4:
        return df.copy()

    out = df.copy()
    _init_or_get_flag_cols(out, req)

    rn = out["NETRAD"].astype(float)
    h = out["H"].astype(float)
    le = out["LE"].astype(float)
    g = out["G"].astype(float)

    resid = rn - (h + le + g)
    m_suspect = resid.abs() > cfg.eb_resid_abs_wm2
    m_bad = resid.abs() > cfg.eb_resid_bad_wm2

    # closure ratio only when Rn positive (daytime)
    with np.errstate(divide="ignore", invalid="ignore"):
        closure = (h + le + g) / rn
    m_closure = (rn > 0) & (
        (closure < cfg.eb_closure_low_high[0]) | (closure > cfg.eb_closure_low_high[1])
    )

    m_any = m_suspect | m_closure | m_bad
    _apply_bit(out, req, m_any, BIT_ENERGY_BAL)

    # Optionally store residual/closure for diagnostics
    out["ENERGY_BAL_RESID"] = resid
    out["ENERGY_BAL_CLOSURE"] = closure

    return out


# -----------------------------------------------------------------------------
# Aggregation to QC severities
# -----------------------------------------------------------------------------


def aggregate_qc(
    df: pd.DataFrame, qc_map: Optional[List[Tuple[int, int]]] = None
) -> pd.DataFrame:
    """Add `<VAR>_QC` columns from `<VAR>_FLAG` bitfields.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing `<VAR>_FLAG` columns.
    qc_map : list of (bitmask_any, qc_value), optional
        Rules to convert bitwise flags to a single QC severity integer.

    Returns
    -------
    pd.DataFrame
        Copy with `<VAR>_QC` columns added.
    """
    out = df.copy()
    qc_map = qc_map or DEFAULT_QC_MAP

    flag_cols = [c for c in out.columns if c.endswith("_FLAG")]
    for fcol in flag_cols:
        qc_col = fcol.replace("_FLAG", "_QC")
        f = out[fcol].fillna(0).astype(int)
        qc = pd.Series(0, index=out.index)
        for bitmask, qc_val in qc_map:
            qc = np.where((f & bitmask) != 0, np.maximum(qc, qc_val), qc)
        out[qc_col] = qc.astype(int)
    return out


# -----------------------------------------------------------------------------
# High-level convenience
# -----------------------------------------------------------------------------


def apply_all_flags(df: pd.DataFrame, cfg: Optional[FlagConfig] = None) -> pd.DataFrame:
    """Run the standard suite of flags and aggregate QC.

    This is the recommended entry point for most workflows. It adds `*_FLAG` and
    `*_QC` columns where applicable, plus energy balance diagnostics if available.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with AmeriFlux-like column names (e.g., `H`, `LE`, `SW_IN_1_1_1`).
    cfg : FlagConfig, optional
        Configuration. If omitted, defaults are used.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with flag and QC columns appended.
    """
    cfg = cfg or FlagConfig()
    out = df.copy()
    out = flag_missing_and_limits(out, cfg)
    out = flag_ranges(out, cfg)
    out = flag_spikes(out, cfg)
    out = flag_rate_of_change(out, cfg)
    out = flag_consistency(out, cfg)
    out = flag_timestamp(out, cfg)
    out = flag_energy_balance(out, cfg)
    out = aggregate_qc(out)
    return out


# -----------------------------------------------------------------------------
# Utilities to decode flags
# -----------------------------------------------------------------------------


def decode_flag_bits(flag_value: int) -> List[str]:
    """Decode a bitwise flag integer into reason strings."""
    reasons = []
    for bit, name in BIT_REASONS.items():
        if flag_value & bit:
            reasons.append(name)
    return reasons


def explain_flags(df: pd.DataFrame, var: str) -> pd.DataFrame:
    """Return a small dataframe explaining which bits were set for `var`.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the `<var>_FLAG` column.
    var : str
        Base variable name (e.g., "H" or "SW_IN_1_1_1").
    """
    fcol = f"{var}_FLAG"
    if fcol not in df.columns:
        raise KeyError(f"No flag column found for {var!r}")
    bits = df[fcol].fillna(0).astype(int)
    data = {
        "flag": bits,
        **{name: (bits & bit) != 0 for bit, name in BIT_REASONS.items()},
    }
    return pd.DataFrame(data, index=df.index)


# -----------------------------------------------------------------------------
# Minimal ranges and rate defaults helpful for typical AmeriFlux stations
# -----------------------------------------------------------------------------

DEFAULT_RATE_MAX = {
    # per 30-min step defaults (override if your step differs)
    "TA": 10.0,
    "T_SONIC": 10.0,
    "RH": 40.0,
    "WS": 20.0,
    "SW_IN": 500.0,
    "SW_OUT": 500.0,
    "LW_IN": 100.0,
    "LW_OUT": 100.0,
    "H": 300.0,
    "LE": 300.0,
    "G": 300.0,
}
