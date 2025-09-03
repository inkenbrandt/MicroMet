"""
micromet.qaqc.physics
=====================

Physics-based QA/QC helpers for micrometeorological towers using AmeriFlux
base variable names.

Expected columns (when applicable):
- TIMESTAMP_START (datetime-like)
- SW_IN, SW_OUT, LW_IN, LW_OUT (W/m2)
- NETRAD, H, LE, G (W/m2)
- TA (degC)

Key functions
-------------
- radiation_consistency(df, tol_wm2=75): flag |NETRAD - (SW_IN-SW_OUT+LW_IN-LW_OUT)| > tol
- energy_closure_ratio(df): (H+LE)/(NETRAD-G)
- energy_closure_residual(df): (NETRAD-G) - (H+LE)
- albedo(df): SW_OUT/SW_IN
- albedo_flags(df, tol=0.02): outside [0 - tol, 1 + tol]
- bowen_ratio(df, min_le_wm2=1.0): H/LE with guards
- daylight_mask(df, sw_threshold=5): SW_IN > threshold
- midday_mask(df, start_hour=10, end_hour=14): local time window
- apparent_surface_temp_from_lw_out(df, emissivity=0.98): K & C
- sky_temp_from_lw_in(df, emissivity=1.0): K & C
- daily_energy_closure_summary(df): daily stats of closure ratio & residual
- build_physics_flags(df, ...): compile useful boolean flags

Notes
-----
- All computations assume inputs already in **local standard time** (fixed UTC offset).
- Units: W/m² for fluxes, °C for TA, timestamps in TIMESTAMP_START.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

SIGMA_SB = 5.670374419e-8  # Stefan–Boltzmann constant [W m^-2 K^-4]
HALFHOUR_MINUTES = 30

COL_TSTART = "TIMESTAMP_START"
COL_TEND = "TIMESTAMP_END"

AMF_VARS = {
    "SW_IN": "SW_IN",
    "SW_OUT": "SW_OUT",
    "LW_IN": "LW_IN",
    "LW_OUT": "LW_OUT",
    "NETRAD": "NETRAD",
    "H": "H",
    "LE": "LE",
    "G": "G",
    "TA": "TA",
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _num(s: pd.Series) -> pd.Series:
    """Coerce to numeric (float) with NaN where invalid."""
    return pd.to_numeric(s, errors="coerce")


def _has(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    return set(cols).issubset(df.columns)


def _dt_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    if COL_TSTART in df.columns:
        return pd.to_datetime(df[COL_TSTART], errors="coerce")  # type: ignore
    # fallback: try the index
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index
    return pd.to_datetime(pd.Series([pd.NaT] * len(df)))  # type: ignore


# ---------------------------------------------------------------------
# Radiation components & checks
# ---------------------------------------------------------------------


def compute_rnet_components(df: pd.DataFrame) -> pd.Series:
    """
    Compute Rn components as (SW_IN - SW_OUT + LW_IN - LW_OUT).
    Returns a Series aligned to df.
    """
    req = ["SW_IN", "SW_OUT", "LW_IN", "LW_OUT"]
    if not _has(df, req):
        return pd.Series(np.nan, index=df.index)
    return (
        _num(df["SW_IN"]) - _num(df["SW_OUT"]) + _num(df["LW_IN"]) - _num(df["LW_OUT"])
    )


def radiation_consistency(df: pd.DataFrame, tol_wm2: float = 75.0) -> pd.Series:
    """
    Flag rows where measured NETRAD differs from component sum by more than tol (W/m2).

    Returns
    -------
    pd.Series (bool): True where |NETRAD - components| > tol (i.e., inconsistent)
    """
    if not _has(df, ["NETRAD", "SW_IN", "SW_OUT", "LW_IN", "LW_OUT"]):
        return pd.Series(False, index=df.index)
    comp = compute_rnet_components(df)
    diff = (_num(df["NETRAD"]) - comp).abs()
    return (diff > tol_wm2).fillna(False)


def radiation_difference(df: pd.DataFrame) -> pd.Series:
    """
    Return the signed difference NETRAD - (SW_IN - SW_OUT + LW_IN - LW_OUT) [W/m2].
    Positive means measured NETRAD exceeds components.
    """
    if not _has(df, ["NETRAD", "SW_IN", "SW_OUT", "LW_IN", "LW_OUT"]):
        return pd.Series(np.nan, index=df.index)
    return _num(df["NETRAD"]) - compute_rnet_components(df)


# ---------------------------------------------------------------------
# Energy balance closure
# ---------------------------------------------------------------------


def energy_closure_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Compute the classic energy balance closure ratio:
        ECR = (H + LE) / (NETRAD - G)

    Returns NaN where denominator is near zero or missing.
    """
    if not _has(df, ["NETRAD", "H", "LE", "G"]):
        return pd.Series(np.nan, index=df.index)
    rn = _num(df["NETRAD"])
    g = _num(df["G"])
    num = _num(df["H"]) + _num(df["LE"])
    den = rn - g
    with np.errstate(divide="ignore", invalid="ignore"):
        ecr = num / den
    # Avoid huge spikes where den ~ 0
    ecr[(den.abs() < 1e-6)] = np.nan
    return ecr


def energy_closure_residual(df: pd.DataFrame) -> pd.Series:
    """
    Residual of energy balance:
        R = (NETRAD - G) - (H + LE)   [W/m2]
    Positive means available energy exceeds turbulent fluxes.
    """
    if not _has(df, ["NETRAD", "H", "LE", "G"]):
        return pd.Series(np.nan, index=df.index)
    rn = _num(df["NETRAD"])
    g = _num(df["G"])
    h = _num(df["H"])
    le = _num(df["LE"])
    return (rn - g) - (h + le)


def energy_closure_flags(
    df: pd.DataFrame,
    *,
    residual_tol_wm2: float = 100.0,
    ratio_bounds: Tuple[float, float] = (0.6, 1.4),
) -> pd.Series:
    """
    Flag rows where energy balance closure is suspect, using either:
      - |residual| > residual_tol_wm2  OR
      - ratio outside ratio_bounds

    Returns boolean Series (True = suspect).
    """
    res = energy_closure_residual(df).abs() > residual_tol_wm2
    ecr = energy_closure_ratio(df)
    low, high = ratio_bounds
    ratio_flag = (ecr < low) | (ecr > high)
    return (res | ratio_flag).fillna(False)


# ---------------------------------------------------------------------
# Albedo & simple radiative sanity checks
# ---------------------------------------------------------------------


def albedo(df: pd.DataFrame) -> pd.Series:
    """Return α = SW_OUT / SW_IN; NaN where SW_IN <= 0."""
    if not _has(df, ["SW_IN", "SW_OUT"]):
        return pd.Series(np.nan, index=df.index)
    sw_in = _num(df["SW_IN"])
    sw_out = _num(df["SW_OUT"])
    with np.errstate(divide="ignore", invalid="ignore"):
        a = sw_out / sw_in
    a[(sw_in <= 0)] = np.nan
    return a


def albedo_flags(df: pd.DataFrame, tol: float = 0.02) -> pd.Series:
    """
    Flag α outside [0 - tol, 1 + tol].
    """
    a = albedo(df)
    return ((a < (0 - tol)) | (a > (1 + tol))).fillna(False)


def daylight_mask(df: pd.DataFrame, sw_threshold: float = 5.0) -> pd.Series:
    """Return True where SW_IN > sw_threshold (W/m2)."""
    if "SW_IN" not in df.columns:
        return pd.Series(False, index=df.index)
    return (_num(df["SW_IN"]) > sw_threshold).fillna(False)


def midday_mask(
    df: pd.DataFrame, start_hour: int = 10, end_hour: int = 14
) -> pd.Series:
    """
    Return True for rows whose TIMESTAMP_START local hour is within [start_hour, end_hour).
    """
    dt = _dt_index(df)
    if dt.isna().all():
        return pd.Series(False, index=df.index)
    hrs = dt.dt.hour  # type: ignore
    return ((hrs >= start_hour) & (hrs < end_hour)).reindex(df.index, fill_value=False)


def midday_negative_netrad_flag(
    df: pd.DataFrame, *, sw_in_threshold: float = 200.0
) -> pd.Series:
    """
    Flag cases where NETRAD < 0 during bright midday conditions (SW_IN >= threshold).
    """
    if not _has(df, ["NETRAD", "SW_IN"]):
        return pd.Series(False, index=df.index)
    mid = midday_mask(df)
    bright = _num(df["SW_IN"]) >= sw_in_threshold
    return (mid & bright & (_num(df["NETRAD"]) < 0)).fillna(False)


# ---------------------------------------------------------------------
# Derived temperatures from longwave
# ---------------------------------------------------------------------


def _kelvin_from_c(t_c: pd.Series) -> pd.Series:
    return _num(t_c) + 273.15


def apparent_surface_temp_from_lw_out(
    df: pd.DataFrame, emissivity: float = 0.98
) -> pd.DataFrame:
    """
    Estimate apparent surface temperature from LW_OUT (emissivity ε ~ 0.95–0.99):
        T_surf = (LW_OUT / (ε σ))^(1/4)

    Returns DataFrame with columns: Tsurf_K, Tsurf_C
    """
    if "LW_OUT" not in df.columns:
        return pd.DataFrame(
            {
                "Tsurf_K": pd.Series(np.nan, index=df.index),
                "Tsurf_C": pd.Series(np.nan, index=df.index),
            }
        )
    lw = _num(df["LW_OUT"])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_k = (lw / (emissivity * SIGMA_SB)) ** 0.25
    t_c = t_k - 273.15
    return pd.DataFrame({"Tsurf_K": t_k, "Tsurf_C": t_c}, index=df.index)


def sky_temp_from_lw_in(df: pd.DataFrame, emissivity: float = 1.0) -> pd.DataFrame:
    """
    Estimate effective sky temperature from LW_IN (ε≈1 for optically thick sky):
        T_sky = (LW_IN / (ε σ))^(1/4)

    Returns DataFrame with columns: Tsky_K, Tsky_C
    """
    if "LW_IN" not in df.columns:
        return pd.DataFrame(
            {
                "Tsky_K": pd.Series(np.nan, index=df.index),
                "Tsky_C": pd.Series(np.nan, index=df.index),
            }
        )
    lw = _num(df["LW_IN"])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_k = (lw / (emissivity * SIGMA_SB)) ** 0.25
    t_c = t_k - 273.15
    return pd.DataFrame({"Tsky_K": t_k, "Tsky_C": t_c}, index=df.index)


# ---------------------------------------------------------------------
# Bowen ratio and helpers
# ---------------------------------------------------------------------


def bowen_ratio(df: pd.DataFrame, min_le_wm2: float = 1.0) -> pd.Series:
    """
    Bowen ratio: β = H / LE; NaN when |LE| < min_le_wm2 to avoid blow-ups.
    """
    if not _has(df, ["H", "LE"]):
        return pd.Series(np.nan, index=df.index)
    h = _num(df["H"])
    le = _num(df["LE"])
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = h / le
    beta[le.abs() < min_le_wm2] = np.nan
    return beta


# ---------------------------------------------------------------------
# Daily summaries
# ---------------------------------------------------------------------


def daily_energy_closure_summary(
    df: pd.DataFrame, *, ratio_quantiles=(0.1, 0.5, 0.9)
) -> pd.DataFrame:
    """
    Daily stats of energy balance closure:
      - ratio_q10, ratio_q50, ratio_q90 (median-based)
      - resid_med, resid_iqr
      - n (count per day)

    Returns a DataFrame indexed by date.
    """
    dt = _dt_index(df)
    if dt.isna().all():
        # Cannot resample; return empty frame
        return pd.DataFrame(
            columns=[
                "ratio_q10",
                "ratio_q50",
                "ratio_q90",
                "resid_med",
                "resid_iqr",
                "n",
            ]
        )

    ec_ratio = energy_closure_ratio(df)
    resid = energy_closure_residual(df)

    grp = pd.DataFrame({"ratio": ec_ratio, "resid": resid}, index=dt).resample("D")
    q10, q50, q90 = ratio_quantiles
    out = pd.DataFrame(
        {
            "ratio_q10": grp["ratio"].quantile(q10),
            "ratio_q50": grp["ratio"].quantile(q50),
            "ratio_q90": grp["ratio"].quantile(q90),
            "resid_med": grp["resid"].median(),
            "resid_iqr": grp["resid"].quantile(0.75) - grp["resid"].quantile(0.25),
            "n": grp["ratio"].count(),
        }
    )
    return out


# ---------------------------------------------------------------------
# Flag aggregator
# ---------------------------------------------------------------------


@dataclass
class PhysicsFlagOptions:
    """
    Options controlling physics-based flags.
    """

    rad_diff_tol_wm2: float = 75.0
    closure_residual_tol_wm2: float = 100.0
    closure_ratio_bounds: Tuple[float, float] = (0.6, 1.4)
    albedo_tol: float = 0.02
    midday_sw_threshold: float = 200.0


def build_physics_flags(
    df: pd.DataFrame, options: PhysicsFlagOptions | None = None
) -> Dict[str, pd.Series]:
    """
    Build a dictionary of boolean flag Series keyed by variable/metric name.

    Returns keys:
      - "NETRAD_consistency"  : radiation component mismatch flags
      - "ALBEDO_bounds"       : albedo outside [0 - tol, 1 + tol]
      - "EBC_suspect"         : energy balance closure suspect
      - "NETRAD_midday_neg"   : negative NETRAD during bright midday
    """
    opts = options or PhysicsFlagOptions()
    flags: Dict[str, pd.Series] = {}

    # Net radiation component check
    flags["NETRAD_consistency"] = radiation_consistency(
        df, tol_wm2=opts.rad_diff_tol_wm2
    )

    # Albedo bounds
    flags["ALBEDO_bounds"] = albedo_flags(df, tol=opts.albedo_tol)

    # Energy balance closure
    flags["EBC_suspect"] = energy_closure_flags(
        df,
        residual_tol_wm2=opts.closure_residual_tol_wm2,
        ratio_bounds=opts.closure_ratio_bounds,
    )

    # Midday negative NETRAD under bright solar
    flags["NETRAD_midday_neg"] = midday_negative_netrad_flag(
        df, sw_in_threshold=opts.midday_sw_threshold
    )

    # Ensure boolean dtype and alignment
    for k, s in list(flags.items()):
        flags[k] = s.reindex(df.index).fillna(False).astype(bool)

    return flags
