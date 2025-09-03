# qaqc/detect.py
"""
Quality-control detection utilities for Micromet.

This module provides robust, composable detectors that return boolean masks
(True = flagged) for common EC/meteorological QA/QC checks and AmeriFlux-
style conventions. All detectors are vectorized and side-effect free.

The functions are designed to interoperate with:
- Micromet (https://github.com/inkenbrandt/MicroMet)
- AmeriFlux variable naming and expectations
  (https://ameriflux.lbl.gov/data/aboutdata/data-variables/)
- flux-data-qaqc (https://github.com/Open-ET/flux-data-qaqc)

Design principles
-----------------
1. Stateless, functional detectors that accept/return DataFrames/Series of
   booleans (True => flag/bad/suspect).
2. Consistent handling of missing sentinels (e.g., -9999) → NaN upstream
   or via `coerce_missing`.
3. Prefix-aware range checks that honor AmeriFlux variable families (e.g.,
   "SW_IN_*", "LE", "H", etc.).
4. Optional compact bitmask encoding if `qaqc.flags` is installed.
5. Numpy-style docstrings; SciPy for regression/outlier goodness.

Notes
-----
- Do *not* mutate inputs. All functions return new objects.
- Column selection is caller's responsibility unless a detector is explicitly
  prefix-aware.
- For pairwise/relationship outliers, SciPy is used (no sklearn dependency).

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def coerce_missing(
    df: pd.DataFrame,
    missing_values: Union[float, int, Sequence[Union[float, int]]] = (-9999,),
) -> pd.DataFrame:
    """
    Replace sentinel missing values with NaN (copy).

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    missing_values : float | int | sequence of float | int, default (-9999,)
        Sentinel(s) that should be treated as NaN.

    Returns
    -------
    pandas.DataFrame
        Copy with sentinels replaced by NaN.
    """
    out = df.copy()
    if isinstance(missing_values, (int, float)):
        missing_values = [missing_values]
    for mv in missing_values:
        out = out.replace(mv, np.nan)
    return out


def _ensure_series(x: Union[pd.Series, np.ndarray, pd.DataFrame]) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(
                "Expected a single-column DataFrame for a Series detector."
            )
        return x.iloc[:, 0]
    if isinstance(x, np.ndarray):
        return pd.Series(x)
    return x


def _prefix_select(df: pd.DataFrame, prefixes: Iterable[str]) -> List[str]:
    """Return columns whose names start with any of the `prefixes`."""
    pref = tuple(prefixes)
    return [c for c in df.columns if c.startswith(pref)]


# ---------------------------------------------------------------------
# Missing / NaN / Duplicate / Time checks
# ---------------------------------------------------------------------


def detect_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag missing (NaN) values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.

    Returns
    -------
    pandas.DataFrame (bool)
        True where values are missing.
    """
    return df.isna()


def detect_constant_segments(
    s: Union[pd.Series, pd.DataFrame],
    window: int = 6,
    atol: float = 0.0,
) -> pd.Series:
    """
    Flag flatlined segments (low variability) using rolling range <= atol.

    Parameters
    ----------
    s : Series or single-column DataFrame
        Input series.
    window : int, default 6
        Rolling window length (samples).
    atol : float, default 0.0
        Absolute tolerance on (max - min) within the window.

    Returns
    -------
    pandas.Series (bool)
        True where flatlined.
    """
    x = _ensure_series(s).astype(float)
    roll_max = x.rolling(window, min_periods=window).max()
    roll_min = x.rolling(window, min_periods=window).min()
    flat = (roll_max - roll_min).le(atol)
    # Expand back to all points in each flat window with forward-fill:
    return flat.reindex_like(x).fillna(False).rolling(window, min_periods=1).max().astype(bool)  # type: ignore


def detect_duplicate_timestamps(idx: pd.DatetimeIndex) -> pd.Series:
    """
    Flag duplicate timestamps in a DatetimeIndex.

    Parameters
    ----------
    idx : pandas.DatetimeIndex
        Time index.

    Returns
    -------
    pandas.Series (bool)
        Aligned to idx, True where a timestamp is a duplicate.
    """
    dup = idx.duplicated(keep="first")
    return pd.Series(dup, index=idx)


def detect_timestamp_gaps(
    idx: pd.DatetimeIndex,
    expected_freq: Union[str, pd.Timedelta] = "30min",
) -> pd.Series:
    """
    Flag locations adjacent to missing time steps (gaps) relative to an expected frequency.

    Parameters
    ----------
    idx : pandas.DatetimeIndex
        Time index.
    expected_freq : str or Timedelta, default "30min"
        Expected sampling interval.

    Returns
    -------
    pandas.Series (bool)
        Aligned to idx, True for samples immediately following a gap.
    """
    if not isinstance(expected_freq, pd.Timedelta):
        expected_freq = pd.Timedelta(expected_freq)
    diffs = idx.to_series().diff()
    gaps = diffs.gt(expected_freq) | diffs.lt(
        expected_freq
    )  # includes early/late steps
    return gaps.fillna(False)


# ---------------------------------------------------------------------
# Range/physical-limit checks (prefix-aware)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class RangeSpec:
    """Simple min/max spec."""

    vmin: Optional[float] = None
    vmax: Optional[float] = None


def detect_out_of_range_prefix(
    df: pd.DataFrame,
    limits: Mapping[str, RangeSpec],
) -> pd.DataFrame:
    """
    Prefix-aware range check. Each key in `limits` is a prefix (e.g., "SW_IN", "LE"),
    and the RangeSpec applies to all columns whose names start with that prefix.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    limits : mapping[str, RangeSpec]
        Prefix → (vmin, vmax) constraints.

    Returns
    -------
    pandas.DataFrame (bool)
        Same shape as df (non-matching columns are False). True = out of range.
    """
    flagged = pd.DataFrame(False, index=df.index, columns=df.columns)
    for pref, spec in limits.items():
        cols = _prefix_select(df, [pref])
        if not cols:
            continue
        sub = df[cols].astype(float)
        bad = pd.DataFrame(False, index=sub.index, columns=sub.columns)
        if spec.vmin is not None:
            bad |= sub.lt(spec.vmin)
        if spec.vmax is not None:
            bad |= sub.gt(spec.vmax)
        flagged.loc[:, cols] = bad
    return flagged


# ---------------------------------------------------------------------
# Spike / step detectors
# ---------------------------------------------------------------------


def detect_spikes_mad(
    s: Union[pd.Series, pd.DataFrame],
    window: int = 9,
    z: float = 6.0,
) -> pd.Series:
    """
    Median Absolute Deviation (MAD) spike detector.

    Flags points whose absolute deviation from the rolling median exceeds
    `z * 1.4826 * MAD`.

    Parameters
    ----------
    s : Series or single-column DataFrame
        Input.
    window : int, default 9
        Rolling window (odd recommended).
    z : float, default 6.0
        Threshold multiplier.

    Returns
    -------
    pandas.Series (bool)
    """
    x = _ensure_series(s).astype(float)
    med = x.rolling(window, center=True, min_periods=max(3, window // 2)).median()
    mad = (
        (x - med)
        .abs()
        .rolling(window, center=True, min_periods=max(3, window // 2))
        .median()
    )
    scale = 1.4826 * mad
    return (x - med).abs().gt(z * scale).fillna(False)


def detect_steps(
    s: Union[pd.Series, pd.DataFrame],
    thresh: float,
) -> pd.Series:
    """
    Simple step change detector via first difference magnitude.

    Parameters
    ----------
    s : Series or single-column DataFrame
        Input.
    thresh : float
        Absolute difference threshold to flag a step.

    Returns
    -------
    pandas.Series (bool)
        True where |diff| >= thresh (aligned to current index).
    """
    x = _ensure_series(s).astype(float)
    d = x.diff().abs()
    return d.ge(thresh).fillna(False)


# ---------------------------------------------------------------------
# Relationship-based outliers (pairwise linear fit)
# ---------------------------------------------------------------------


def detect_pairwise_outliers(
    x: Union[pd.Series, pd.DataFrame],
    y: Union[pd.Series, pd.DataFrame],
    z: float = 3.5,
    min_n: int = 30,
) -> pd.Series:
    """
    Detect outliers from a linear relationship y ~ a*x + b using robust residuals.

    The procedure:
      1) Fit OLS (scipy.stats.linregress) on available pairs.
      2) Compute residuals r = y - (a*x+b).
      3) Use robust scale via MAD; flag |r| > z * 1.4826 * MAD.

    Parameters
    ----------
    x, y : Series or single-column DataFrame
        Paired variables.
    z : float, default 3.5
        Threshold in robust sigma units.
    min_n : int, default 30
        Minimum valid pairs to attempt a fit; otherwise returns all False.

    Returns
    -------
    pandas.Series (bool)
        True where outlier relative to the fitted line (index is intersection).
    """
    xs = _ensure_series(x).astype(float)
    ys = _ensure_series(y).astype(float)
    valid = xs.notna() & ys.notna()
    if valid.sum() < min_n:
        return pd.Series(False, index=xs.index)
    fit = stats.linregress(xs[valid], ys[valid])
    yhat = fit.slope * xs + fit.intercept  # type: ignore
    resid = ys - yhat
    med = resid.median()
    mad = (resid - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(False, index=xs.index)
    robust_sigma = 1.4826 * mad
    return resid.abs().gt(z * robust_sigma).fillna(False)


# ---------------------------------------------------------------------
# Energy balance quick checks (optional but handy for flux work)
# ---------------------------------------------------------------------


def detect_energy_imbalance(
    netrad: pd.Series,
    h: pd.Series,
    le: pd.Series,
    g: Optional[pd.Series] = None,
    tol_wm2: float = 50.0,
) -> pd.Series:
    """
    Flag energy balance residuals exceeding tolerance:
        Rn - G - (H + LE) > tol_wm2

    Parameters
    ----------
    netrad : pandas.Series
        Net radiation (W m-2).
    h : pandas.Series
        Sensible heat flux (W m-2).
    le : pandas.Series
        Latent heat flux (W m-2).
    g : pandas.Series, optional
        Ground heat flux (W m-2). If None, uses G=0.
    tol_wm2 : float, default 50.0
        Absolute residual tolerance (W m-2).

    Returns
    -------
    pandas.Series (bool)
        True where |residual| > tol_wm2.
    """
    Rn = _ensure_series(netrad).astype(float)
    H = _ensure_series(h).astype(float)
    LE = _ensure_series(le).astype(float)
    G = (
        pd.Series(0.0, index=Rn.index)
        if g is None
        else _ensure_series(g).astype(float).reindex(Rn.index)
    )
    resid = Rn - G - (H + LE)
    return resid.abs().gt(tol_wm2).fillna(False)


# ---------------------------------------------------------------------
# Column-wise drivers: build a full mask bundle
# ---------------------------------------------------------------------


def bundle_basic_detections(
    df: pd.DataFrame,
    *,
    missing_values: Union[float, int, Sequence[Union[float, int]]] = (-9999,),
    constant_window: int = 6,
    constant_atol_by_prefix: Optional[Mapping[str, float]] = None,
    spike_window_by_prefix: Optional[Mapping[str, int]] = None,
    spike_z_by_prefix: Optional[Mapping[str, float]] = None,
    range_specs: Optional[Mapping[str, RangeSpec]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Apply a common set of detectors and return a dict of boolean DataFrames.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    missing_values : scalar or sequence, default (-9999,)
        Values to treat as missing.
    constant_window : int, default 6
        Window for constant/flatline detection.
    constant_atol_by_prefix : mapping[str, float], optional
        Per-prefix flatline tolerances.
    spike_window_by_prefix : mapping[str, int], optional
        Per-prefix MAD spike windows.
    spike_z_by_prefix : mapping[str, float], optional
        Per-prefix MAD z thresholds.
    range_specs : mapping[str, RangeSpec], optional
        Prefix-aware range constraints.

    Returns
    -------
    dict[str, pandas.DataFrame(bool)]
        Keys include: "missing", "range", "constant", "spike".
        Only populated for columns matched by the respective detectors.
    """
    clean = coerce_missing(df, missing_values=missing_values)

    out: Dict[str, pd.DataFrame] = {}

    # Missing
    out["missing"] = clean.isna()

    # Range
    if range_specs:
        out["range"] = detect_out_of_range_prefix(clean, range_specs)
    else:
        out["range"] = pd.DataFrame(False, index=df.index, columns=df.columns)

    # Constant/flatline
    const_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    if constant_atol_by_prefix:
        for pref, atol in constant_atol_by_prefix.items():
            cols = _prefix_select(clean, [pref])
            for c in cols:
                const_mask[c] = detect_constant_segments(
                    clean[c], window=constant_window, atol=atol
                )
    out["constant"] = const_mask

    # Spikes (MAD)
    spike_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    if spike_window_by_prefix or spike_z_by_prefix:
        prefs = set()
        if spike_window_by_prefix:
            prefs |= set(spike_window_by_prefix.keys())
        if spike_z_by_prefix:
            prefs |= set(spike_z_by_prefix.keys())
        for pref in prefs:
            cols = _prefix_select(clean, [pref])
            w = spike_window_by_prefix.get(pref, 9) if spike_window_by_prefix else 9
            z = spike_z_by_prefix.get(pref, 6.0) if spike_z_by_prefix else 6.0
            for c in cols:
                spike_mask[c] = detect_spikes_mad(clean[c], window=w, z=z)
    out["spike"] = spike_mask

    return out


# ---------------------------------------------------------------------
# Optional: compact bitmask encoding (if qaqc.flags is present)
# ---------------------------------------------------------------------


def encode_bitmasks(
    masks: Mapping[str, Union[pd.Series, pd.DataFrame]],
    prefer_series: bool = False,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Encode a dict of named boolean masks into a single integer bitmask per element.

    Requires `qaqc.flags` with an IntFlag-like enum named `QCFlag` that defines bits:
        MISSING, RANGE, CONSTANT, SPIKE, DUP_TS, GAP, ENERGY

    If not available, this function raises ImportError.

    Parameters
    ----------
    masks : mapping[str, Series | DataFrame]
        Named masks. Supported keys: 'missing', 'range', 'constant', 'spike',
        'dup_ts', 'gap', 'energy'.
    prefer_series : bool, default False
        If True and all inputs are Series with matching index, return a Series.

    Returns
    -------
    pandas.Series or pandas.DataFrame (int)
        Bitmask per element.

    Raises
    ------
    ImportError
        If `qaqc.flags` is not importable.
    """
    try:
        from .flags import QCFlag  # type: ignore
    except Exception as e:
        raise ImportError("encode_bitmasks requires qaqc.flags.QCFlag") from e

    # Normalize all to DataFrames aligned on unioned index/columns
    def to_df(x):
        if isinstance(x, pd.Series):
            return x.to_frame()
        return x

    dfs = {k: to_df(v).astype(bool) for k, v in masks.items()}
    # Build a unioned frame
    all_cols: List[str] = sorted({c for v in dfs.values() for c in v.columns})
    all_idx = pd.Index(sorted({i for v in dfs.values() for i in v.index}))
    out = pd.DataFrame(0, index=all_idx, columns=all_cols, dtype=np.int64)

    name_to_flag = {
        "missing": QCFlag.MISSING,
        "range": QCFlag.RANGE,
        "constant": QCFlag.CONSTANT,
        "spike": QCFlag.SPIKE,
        "dup_ts": QCFlag.DUP_TS,
        "gap": QCFlag.GAP,
        "energy": QCFlag.ENERGY,
    }

    for name, dfm in dfs.items():
        flag = name_to_flag.get(name)
        if flag is None:
            continue
        dfm = dfm.reindex(index=all_idx, columns=all_cols, fill_value=False)
        out |= dfm.astype(np.int64) * int(flag)

    if prefer_series and out.shape[1] == 1:
        return out.iloc[:, 0]
    return out


# ---------------------------------------------------------------------
# Reasonable defaults (AmeriFlux-ish) for quick use
# ---------------------------------------------------------------------


def default_range_specs() -> Dict[str, RangeSpec]:
    """
    Provide conservative physical limits for common variables (AmeriFlux-ish).

    Returns
    -------
    dict[str, RangeSpec]
        Prefix → RangeSpec
    """
    return {
        # Radiation (W m-2)
        "SW_IN": RangeSpec(0.0, 1500.0),
        "SW_OUT": RangeSpec(0.0, 1500.0),
        "LW_IN": RangeSpec(50.0, 700.0),
        "LW_OUT": RangeSpec(50.0, 700.0),
        "NETRAD": RangeSpec(-200.0, 1000.0),
        # Fluxes (W m-2)
        "LE": RangeSpec(-100.0, 800.0),
        "H": RangeSpec(-200.0, 800.0),
        "G": RangeSpec(-300.0, 500.0),
        # Meteorology
        "T_": RangeSpec(-60.0, 60.0),  # °C sonic/air temp (use T_ prefix)
        "RH": RangeSpec(0.0, 100.0),  # %
        "WS": RangeSpec(0.0, 50.0),  # m s-1
        "WD": RangeSpec(0.0, 360.0),  # deg
        "VPD": RangeSpec(0.0, 70.0),  # hPa (conservative)
        "SWC": RangeSpec(0.0, 100.0),  # % volumetric water content (0-100)
    }


def default_constant_tolerances() -> Dict[str, float]:
    """
    Flatline tolerances for families (units-respecting coarse defaults).

    Returns
    -------
    dict[str, float]
    """
    return {
        "SW_IN": 1.0,
        "SW_OUT": 1.0,
        "LW_IN": 1.0,
        "LW_OUT": 1.0,
        "H": 1.0,
        "LE": 1.0,
        "G": 0.5,
        "WS": 0.1,
        "T_": 0.05,
        "RH": 0.1,
        "SWC": 0.05,
    }


def default_spike_windows() -> Dict[str, int]:
    """
    MAD windows for spike detection by family.

    Returns
    -------
    dict[str, int]
    """
    return {
        "SW_IN": 9,
        "SW_OUT": 9,
        "LW_IN": 9,
        "LW_OUT": 9,
        "H": 9,
        "LE": 9,
        "G": 9,
        "WS": 9,
        "T_": 9,
        "RH": 9,
        "SWC": 9,
    }


def default_spike_z() -> Dict[str, float]:
    """
    MAD z thresholds by family.

    Returns
    -------
    dict[str, float]
    """
    return {
        "SW_IN": 6.0,
        "SW_OUT": 6.0,
        "LW_IN": 6.0,
        "LW_OUT": 6.0,
        "H": 6.0,
        "LE": 6.0,
        "G": 6.0,
        "WS": 6.0,
        "T_": 6.0,
        "RH": 6.0,
        "SWC": 6.0,
    }


# ---------------------------------------------------------------------
# Turnkey helper
# ---------------------------------------------------------------------


def run_basic_qaqc(
    df: pd.DataFrame,
    *,
    missing_values: Union[float, int, Sequence[Union[float, int]]] = (-9999,),
    range_specs: Optional[Mapping[str, RangeSpec]] = None,
    constant_window: int = 6,
    constant_atol_by_prefix: Optional[Mapping[str, float]] = None,
    spike_window_by_prefix: Optional[Mapping[str, int]] = None,
    spike_z_by_prefix: Optional[Mapping[str, float]] = None,
    expected_freq: Union[str, pd.Timedelta] = "30min",
    energy_cols: Tuple[str, str, str, Optional[str]] = ("NETRAD", "H", "LE", "G"),
    tol_energy_wm2: float = 50.0,
    encode_bits: bool = False,
) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """
    Run a sensible default QA/QC suite and return a bundle of masks (and optionally a bitmask).

    Parameters
    ----------
    df : pandas.DataFrame
        Input (AmeriFlux-like columns preferred).
    missing_values : scalar or sequence, default (-9999,)
        Values to treat as missing.
    range_specs : mapping[str, RangeSpec], optional
        If None, uses `default_range_specs()`.
    constant_window : int, default 6
        Window for flatline detection.
    constant_atol_by_prefix : mapping[str, float], optional
        If None, uses `default_constant_tolerances()`.
    spike_window_by_prefix : mapping[str, int], optional
        If None, uses `default_spike_windows()`.
    spike_z_by_prefix : mapping[str, float], optional
        If None, uses `default_spike_z()`.
    expected_freq : str or Timedelta, default "30min"
        Expected sampling interval for gap checks.
    energy_cols : (str, str, str, str|None), default ("NETRAD", "H", "LE", "G")
        Column names for energy balance residual (G can be None).
    tol_energy_wm2 : float, default 50.0
        Absolute residual tolerance for energy balance.
    encode_bits : bool, default False
        If True, also return an integer bitmask DataFrame under key "bitmask".
        Requires `qaqc.flags.QCFlag`.

    Returns
    -------
    dict
        {
          "missing": DataFrame(bool),
          "range": DataFrame(bool),
          "constant": DataFrame(bool),
          "spike": DataFrame(bool),
          "dup_ts": Series(bool),
          "gap": Series(bool),
          "energy": Series(bool),
          "bitmask": DataFrame(int)  # only if encode_bits=True and flags available
        }
    """
    rng = range_specs or default_range_specs()
    const_atol = constant_atol_by_prefix or default_constant_tolerances()
    sp_win = spike_window_by_prefix or default_spike_windows()
    sp_z = spike_z_by_prefix or default_spike_z()

    # Core column-wise bundle
    masks = bundle_basic_detections(
        df,
        missing_values=missing_values,
        constant_window=constant_window,
        constant_atol_by_prefix=const_atol,
        spike_window_by_prefix=sp_win,
        spike_z_by_prefix=sp_z,
        range_specs=rng,
    )

    # Index/time checks
    if not isinstance(df.index, pd.DatetimeIndex):
        dup_ts = pd.Series(False, index=df.index)
        gap = pd.Series(False, index=df.index)
    else:
        dup_ts = detect_duplicate_timestamps(df.index)
        gap = detect_timestamp_gaps(df.index, expected_freq=expected_freq)

    masks["dup_ts"] = dup_ts  # type: ignore
    masks["gap"] = gap  # type: ignore

    # Energy balance residual
    rn_col, h_col, le_col, g_col = energy_cols
    rn = df.get(rn_col)
    h = df.get(h_col)
    le = df.get(le_col)
    g = df.get(g_col) if g_col is not None else None

    if rn is not None and h is not None and le is not None:
        masks["energy"] = detect_energy_imbalance(rn, h, le, g, tol_wm2=tol_energy_wm2)  # type: ignore
    else:
        masks["energy"] = pd.Series(False, index=df.index)  # type: ignore

    # Optional compact encoding
    if encode_bits:
        try:
            bitmask = encode_bitmasks(masks)
            masks["bitmask"] = bitmask  # type: ignore
        except ImportError:
            # Silently skip bitmask if flags are not wired yet
            pass

    return masks  # type: ignore
