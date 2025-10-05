"""
AmeriFlux-like Timestamp Alignment QA/QC

Implements the core ideas from the AmeriFlux Timestamp Alignment Module:
- Compute potential incoming shortwave at the top of atmosphere (SW_IN_POT)
  in local standard time (no DST).
- Build 15-day non-overlapping "maximum diurnal composites".
- Compute (1) % of time composite observed radiation exceeds potential and
  (2) lag (in time steps) at which cross-correlation between observed and
      potential is maximized.

Also provides heuristic flags for:
- Time zone mismatch / DST usage
- Timestamp START vs END mis-specification
- Stream desynchronization between SW_IN and PPFD_IN
- Possible radiation sensor issues (shading/not level/unexpectedly high)

References
----------
AmeriFlux Data QA/QC: Timestamp Alignment Module
(Design: 15-day max diurnal composite; local standard time; exceedance &
 cross-correlation interpretation). See:
https://ameriflux.lbl.gov/data/flux-data-products/data-qaqc/timestamp-alignment-module/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import pytz
from math import sin, cos, radians, pi, asin

# Constants
SOLAR_CONSTANT = 1367  # W/m²
ALBEDO = 0.25  # average for natural terrain
EMISSIVITY = 0.98  # ground emissivity
STEFAN_BOLTZMANN = 5.67e-8  # W/m²/K⁴
LATITUDE = 39.5  # Utah average latitude in degrees
LONGITUDE = -111.5  # Utah average longitude in degrees


def solar_declination(doy: int) -> float:
    """
    Calculate the solar declination angle for a given day of the year.

    The solar declination is the angle between the Earth's equatorial
    plane and the line connecting the centers of the Earth and the Sun.

    Parameters
    ----------
    doy : int
        The day of the year (1-365 or 1-366).

    Returns
    -------
    float
        The solar declination angle in radians.
    """
    return radians(23.44) * sin(2 * pi * (284 + doy) / 365)


def hour_angle(hour: int) -> float:
    """
    Calculate the hour angle for a given hour of the day.

    The hour angle is the angular displacement of the Sun east or west
    of the local meridian due to Earth's rotation.

    Parameters
    ----------
    hour : int
        The hour of the day (0-23).

    Returns
    -------
    float
        The hour angle in radians.
    """
    return radians(15 * (hour - 12))


def solar_elevation(doy: int, hour: int, latitude: float = LATITUDE) -> float:
    """
    Calculate the solar elevation angle.

    The solar elevation angle is the angle of the Sun above the horizon.

    Parameters
    ----------
    doy : int
        The day of the year.
    hour : int
        The hour of the day.
    latitude : float, optional
        The latitude in degrees. Defaults to LATITUDE.

    Returns
    -------
    float
        The solar elevation angle in radians.
    """
    decl = solar_declination(doy)
    lat_rad = radians(latitude)
    ha = hour_angle(hour)

    sin_elev = sin(lat_rad) * sin(decl) + cos(lat_rad) * cos(decl) * cos(ha)
    return max(0, asin(sin_elev))  # radians


def clear_sky_radiation(doy: int, 
                        hour: int, 
                        latitude: float = LATITUDE) -> float:
    """
    Estimate the clear-sky incoming shortwave radiation.

    This function provides a simplified estimation of the incoming shortwave
    radiation on a clear day, taking into account the solar constant,
    Earth-Sun distance, and atmospheric transmissivity.

    Parameters
    ----------
    doy : int
        The day of the year.
    hour : int
        The hour of the day.
    latitude : float, optional
        The latitude in degrees. Defaults to LATITUDE.

    Returns
    -------
    float
        The estimated clear-sky radiation in W/m^2.
    """
    elev_rad = solar_elevation(doy, hour, latitude)
    if elev_rad <= 0:
        return 0.0

    d_r = 1 + 0.033 * cos(2 * pi * doy / 365)  # Earth-Sun distance factor
    cos_zenith = cos(pi / 2 - elev_rad)
    rs = SOLAR_CONSTANT * d_r * cos_zenith * 0.75  # ~0.75 = clear sky transmissivity
    return rs


def longwave_radiation(T_kelvin):
    """
    Estimate the longwave radiation using the Stefan-Boltzmann law.

    This function calculates the longwave radiation emitted by a surface
    based on its temperature, using the Stefan-Boltzmann law.

    Parameters
    ----------
    T_kelvin : float
        The temperature of the surface in Kelvin.

    Returns
    -------
    float
        The estimated longwave radiation in W/m^2.
    """
    return EMISSIVITY * STEFAN_BOLTZMANN * T_kelvin**4


def estimate_net_radiation_range(doy, hour):
    """
    Estimate the min/max net radiation for a given hour and day of the year.

    This function calculates a plausible range for net radiation by
    estimating the components of the radiation budget (shortwave and
    longwave radiation) under typical conditions.

    Parameters
    ----------
    doy : int
        The day of the year.
    hour : int
        The hour of the day.

    Returns
    -------
    tuple[float, float]
        A tuple containing the minimum and maximum estimated net radiation
        in W/m^2.
    """
    rs_down = clear_sky_radiation(doy, hour)

    # Net shortwave
    rs_net = rs_down * (1 - ALBEDO)

    # Assume typical diurnal surface temperature range in K
    Tmin_K = 273.15 + 5  # typical min in early morning
    Tmax_K = 273.15 + 40  # hot afternoon summer temp

    # Incoming longwave (simplified as ~cloudless sky)
    lw_down_min = longwave_radiation(Tmin_K - 5)  # colder sky at night
    lw_down_max = longwave_radiation(Tmax_K - 15)  # warmer sky in afternoon

    # Outgoing longwave from surface
    lw_up_min = longwave_radiation(Tmin_K)
    lw_up_max = longwave_radiation(Tmax_K)

    # Net radiation range
    Rn_min = rs_net + lw_down_min - lw_up_max
    Rn_max = rs_net + lw_down_max - lw_up_min

    return Rn_min, Rn_max


def add_buffer(min_max: tuple, buffer: float = 100):
    """
    Add a buffer to a min/max tuple, with a hard lower limit.

    This function expands a range defined by a min/max tuple by adding
    a buffer. The minimum value is not allowed to go below -200.

    Parameters
    ----------
    min_max : tuple
        A tuple containing the minimum and maximum values.
    buffer : float, optional
        The buffer to add to the max value and subtract from the min
        value. Defaults to 100.

    Returns
    -------
    tuple[float, float]
        The buffered minimum and maximum values.
    """
    if min_max[0] - buffer <= -200:
        minv = -200
    else:
        minv = min_max[0] - buffer
    maxv = min_max[1] + buffer
    return minv, maxv


if __name__ == "__main__":
    # Example usage
    doy = 172  # summer solstice ~June 21
    hour = 14  # 2 PM

    Rn_min, Rn_max = estimate_net_radiation_range(doy, hour)
    print(f"DOY: {doy}, Hour: {hour}")
    print(f"Estimated Net Radiation Range: {Rn_min:.1f} W/m² to {Rn_max:.1f} W/m²")

    def calc_diurnal_range(doy):
        """
        Calculate the diurnal range of net radiation for a given day.

        This function calculates the estimated minimum and maximum net
        radiation for each hour of a given day of the year.

        Parameters
        ----------
        doy : int
            The day of the year.

        Returns
        -------
        tuple[list[float], list[float]]
            A tuple of two lists, where the first list contains the
            minimum net radiation for each hour and the second list
            contains the maximum.
        """
        hours = np.arange(0, 24)
        rn_min_list, rn_max_list = zip(
            *[estimate_net_radiation_range(doy, h) for h in hours]
        )
        return rn_min_list, rn_max_list


# ----------------------------- Utilities ------------------------------------ #


def _infer_freq_minutes(dt: pd.DatetimeIndex) -> int:
    """
    Infer the sampling interval in whole minutes from a DatetimeIndex.

    This function calculates the median difference between consecutive
    timestamps in a DatetimeIndex and returns the result in whole minutes.

    Parameters
    ----------
    dt : pd.DatetimeIndex
        The DatetimeIndex from which to infer the frequency.

    Returns
    -------
    int
        The inferred sampling interval in minutes.

    Raises
    ------
    ValueError
        If the index contains fewer than two timestamps or if the
        inferred interval is not positive.
    """
    diffs = np.diff(dt.view("i8"))  # nanoseconds
    if len(diffs) == 0:
        raise ValueError("Need at least two timestamps to infer frequency.")
    med = np.median(diffs)
    minutes = int(round(med / 1e9 / 60.0))
    if minutes <= 0:
        raise ValueError("Non-positive inferred sampling interval.")
    return minutes


def _to_local_standard_time(
    dt: pd.DatetimeIndex,
    std_utc_offset_hours: float,
    assume_naive_is_local: bool = False,
) -> pd.DatetimeIndex:
    """
    Convert timestamps to local standard time (no DST) using a fixed UTC offset.

    This function takes a DatetimeIndex and converts it to a fixed-offset
    timezone, effectively removing any daylight saving time adjustments.

    Parameters
    ----------
    dt : pd.DatetimeIndex
        Input timestamps, which can be timezone-naive or timezone-aware.
    std_utc_offset_hours : float
        The UTC offset for the local standard time (e.g., -7 for MST).
    assume_naive_is_local : bool, optional
        If True and the input is timezone-naive, it is assumed to be in
        local standard time. If False, it is assumed to be in UTC.
        Defaults to False.

    Returns
    -------
    pd.DatetimeIndex
        A DatetimeIndex localized to a fixed-offset timezone (no DST).
    """
    offset = pd.Timedelta(hours=std_utc_offset_hours)
    fixed_tz = pytz.FixedOffset(int(offset.total_seconds() // 60))
    if dt.tz is None:
        if assume_naive_is_local:
            return dt.tz_localize(fixed_tz)
        else:
            # Assume input is UTC, convert to fixed offset
            return dt.tz_localize("UTC").tz_convert(fixed_tz)
    else:
        return dt.tz_convert(fixed_tz)


# ------------------------ Solar / SW_IN_POT model --------------------------- #


def sw_in_pot_noaa(
    dt_local_standard: pd.DatetimeIndex,
    lat_deg: float | None = 40.7607,
    lon_deg: float | None = -111.8939,
    std_utc_offset_hours: int | None = -7,
) -> pd.Series:
    """
    Compute top-of-atmosphere shortwave irradiance using NOAA's approximations.

    This function calculates the potential incoming shortwave radiation
    at the top of the atmosphere on a horizontal surface, using solar
    position approximations from NOAA.

    Parameters
    ----------
    dt_local_standard : pd.DatetimeIndex
        A DatetimeIndex in local standard time (fixed UTC offset, no DST).
    lat_deg : float
        Latitude in degrees (positive for Northern Hemisphere).
    lon_deg : float
        Longitude in degrees (positive for Eastern Hemisphere).
    std_utc_offset_hours : float
        The local standard time UTC offset (e.g., -7 for MST).

    Returns
    -------
    pd.Series
        A Series of potential incoming shortwave radiation (W/m^2),
        indexed by the input DatetimeIndex.

    Raises
    ------
    ValueError
        If the input DatetimeIndex is not timezone-aware.
    """
    if dt_local_standard.tz is None:
        raise ValueError("dt_local_standard must be tz-aware fixed-offset (no DST).")

    # Constants
    G_SC = 1361.0  # W m^-2, solar constant

    # Vectorized time components
    ts = dt_local_standard.tz_convert(None)  # drop tz for numeric ops
    doy = ts.dayofyear.values
    hour = ts.hour.values
    minute = ts.minute.values
    second = ts.second.values

    # Fractional year (radians), NOAA formulation
    gamma = 2.0 * np.pi * (doy - 1 + (hour - 12) / 24.0) / 365.0

    # Equation of time (minutes) and solar declination (radians)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )

    # Time offset (minutes)
    if lon_deg is None or std_utc_offset_hours is None:
        raise ValueError("Must provide lon_deg and std_utc_offset_hours.")
    else:
        time_offset = eqtime + 4.0 * lon_deg - 60.0 * std_utc_offset_hours

    # True solar time (minutes)
    tst = hour * 60.0 + minute + second / 60.0 + time_offset

    # Hour angle (degrees -> radians)
    ha_deg = (tst / 4.0) - 180.0
    ha = np.deg2rad(ha_deg)

    if lat_deg is None:
        raise ValueError("Must provide lat_deg.")
    else:
        lat = np.deg2rad(lat_deg)

    # Solar zenith cosine
    cos_zen = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)

    # Earth-Sun distance correction factor (Spencer)
    e0 = (
        1.00011
        + 0.034221 * np.cos(gamma)
        + 0.00128 * np.sin(gamma)
        + 0.000719 * np.cos(2 * gamma)
        + 0.000077 * np.sin(2 * gamma)
    )

    i0 = G_SC * e0 * np.maximum(0.0, cos_zen)  # W m^-2, 0 at night

    return pd.Series(i0, index=dt_local_standard)


# ----------------- Maximum diurnal composite in 15-day windows -------------- #


@dataclass
class WindowComposite:
    """
    A container for the results of a 15-day window analysis.

    This dataclass holds the composite data and statistical results for a
    single 15-day analysis window.

    Attributes
    ----------
    year : int
        The year of the analysis window.
    window_id : int
        The 15-day window number within the year.
    step_minutes : int
        The sampling interval in minutes.
    steps_per_day : int
        The number of time steps per day.
    comp_pot : np.ndarray
        The maximum diurnal composite for potential incoming shortwave radiation.
    comp_sw : np.ndarray, optional
        The maximum diurnal composite for observed shortwave radiation.
    comp_ppfd : np.ndarray, optional
        The maximum diurnal composite for observed PPFD.
    pct_exceed_sw : float, optional
        The percentage of time the observed SW composite exceeds the potential.
    pct_exceed_ppfd : float, optional
        The percentage of time the observed PPFD composite exceeds the potential.
    lag_sw : int, optional
        The lag (in time steps) at which the SW cross-correlation is maximized.
    corr_sw : float, optional
        The maximum cross-correlation for shortwave radiation.
    lag_ppfd : int, optional
        The lag (in time steps) at which the PPFD cross-correlation is maximized.
    corr_ppfd : float, optional
        The maximum cross-correlation for PPFD.
    """

    year: int
    window_id: int
    step_minutes: int
    steps_per_day: int
    comp_pot: np.ndarray
    comp_sw: Optional[np.ndarray]
    comp_ppfd: Optional[np.ndarray]
    pct_exceed_sw: Optional[float]
    pct_exceed_ppfd: Optional[float]
    lag_sw: Optional[int]
    corr_sw: Optional[float]
    lag_ppfd: Optional[int]
    corr_ppfd: Optional[float]


def _max_diurnal_composite(
    s: pd.Series,
    step_minutes: int,
) -> np.ndarray:
    """
    Build a 'maximum diurnal composite' over a 15-day window.

    This function calculates the maximum value for each time slot of the
    day over a 15-day period, creating a composite diurnal cycle.

    Parameters
    ----------
    s : pd.Series
        A Series of values with a DatetimeIndex in local standard time.
    step_minutes : int
        The sampling interval in minutes.

    Returns
    -------
    np.ndarray
        An array of length `steps_per_day` containing the maximum value
        for each time slot.
    """
    steps_per_day = int(round(1440 / step_minutes))
    # Slot index within day [0 .. steps_per_day-1]
    slot = (s.index.hour * 60 + s.index.minute) // step_minutes  # type: ignore
    # Group by day + slot, take max by slot:
    df = pd.DataFrame({"slot": slot, "val": s.values})
    grp = df.groupby("slot")["val"].max()
    comp = np.full(steps_per_day, np.nan, dtype=float)
    comp[grp.index.values] = grp.values
    return comp


def _xcorr_best_lag(
    a: np.ndarray, b: np.ndarray, max_lag: int
) -> Tuple[Optional[int], Optional[float]]:
    """
    Find the lag of maximum Pearson correlation between two arrays.

    This function calculates the Pearson correlation between two arrays at
    different lags (from -max_lag to +max_lag) and returns the lag that
    results in the highest correlation.

    Parameters
    ----------
    a : np.ndarray
        The first array.
    b : np.ndarray
        The second array.
    max_lag : int
        The maximum lag (in steps) to check in each direction.

    Returns
    -------
    tuple[Optional[int], Optional[float]]
        A tuple containing the best lag and the corresponding correlation,
        or (None, None) if there is insufficient data.
    """
    best_lag, best_corr = None, None
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x = a[-lag:]
            y = b[: len(b) + lag]
        elif lag > 0:
            x = a[: len(a) - lag]
            y = b[lag:]
        else:
            x = a
            y = b
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 6:  # need enough points
            r = np.corrcoef(x[mask], y[mask])[0, 1]
            if (best_corr is None) or (r > best_corr):
                best_corr = float(r)
                best_lag = int(lag)
    return best_lag, best_corr


def _fifteen_day_window_id(doy: int) -> int:
    """
    Calculate the 1-based 15-day window number for a given day of the year.

    Parameters
    ----------
    doy : int
        The day of the year (1-365 or 1-366).

    Returns
    -------
    int
        The 15-day window number (1-25).
    """
    return 1 + (doy - 1) // 15


def analyze_timestamp_alignment(
    df: pd.DataFrame,
    *,
    lat: float | None = LATITUDE,
    lon: float | None = LONGITUDE,
    std_utc_offset_hours: int = -7,
    time_from: str = "CENTER",
    start_col: str = "TIMESTAMP_START",
    end_col: str = "TIMESTAMP_END",
    time_col: Optional[str] = None,
    sw_col: str = "SW_IN",
    ppfd_col: str = "PPFD_IN",
    assume_naive_is_local: bool = False,
    max_lag_steps: int = 6,
) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], WindowComposite]]:
    """
    Perform the main timestamp alignment analysis.

    This function analyzes the timestamp alignment of radiation data by
    comparing observed values against potential top-of-atmosphere radiation.
    It computes composites, cross-correlations, and other statistics for
    15-day windows.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing timestamp and radiation data.
    lat : float
        Latitude of the site.
    lon : float
        Longitude of the site.
    std_utc_offset_hours : float
        The UTC offset for local standard time.
    time_from : str, optional
        How to interpret the timestamp ('CENTER', 'START', 'END').
        Defaults to 'CENTER'.
    start_col : str, optional
        Name of the start timestamp column. Defaults to 'TIMESTAMP_START'.
    end_col : str, optional
        Name of the end timestamp column. Defaults to 'TIMESTAMP_END'.
    time_col : str, optional
        Name of a single datetime column. Defaults to None.
    sw_col : str, optional
        Name of the shortwave radiation column. Defaults to 'SW_IN'.
    ppfd_col : str, optional
        Name of the PPFD column. Defaults to 'PPFD_IN'.
    assume_naive_is_local : bool, optional
        Whether to assume naive timestamps are in local time. Defaults to False.
    max_lag_steps : int, optional
        Maximum lag for cross-correlation. Defaults to 6.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        A tuple containing a summary DataFrame of the analysis and a
        dictionary of WindowComposite objects for each window.
    """
    # 1) Build DateTimeIndex
    idx = None
    if time_col and time_col in df.columns:
        idx = pd.to_datetime(df[time_col], errors="coerce")
    else:
        if (start_col in df.columns) and (end_col in df.columns):
            start = pd.to_datetime(df[start_col].astype(str), errors="coerce")
            end = pd.to_datetime(df[end_col].astype(str), errors="coerce")
            if time_from.upper() == "CENTER":
                dt = start + (end - start) / 2
            elif time_from.upper() == "START":
                dt = start
            elif time_from.upper() == "END":
                dt = end
            else:
                raise ValueError("time_from must be one of {'CENTER','START','END'}")
            idx = dt
        else:
            raise ValueError("Provide a time column or TIMESTAMP_START/END columns.")

    # Attach index & keep only needed columns
    work = df.copy()
    work.index = pd.DatetimeIndex(idx)
    cols = [c for c in [sw_col, ppfd_col] if c in work.columns]
    work = work[cols].sort_index()

    # 2) Convert to local standard time (fixed offset)
    dt_lst = _to_local_standard_time(
        work.index,  # type: ignore
        std_utc_offset_hours=std_utc_offset_hours,
        assume_naive_is_local=assume_naive_is_local,
    )
    work.index = dt_lst

    # 3) Infer sampling step and compute SW_IN_POT
    step_minutes = _infer_freq_minutes(work.index)
    pot = sw_in_pot_noaa(
        work.index, lat_deg=lat, lon_deg=lon, std_utc_offset_hours=std_utc_offset_hours
    )

    # 4) Organize by year and 15-day window
    out_rows = []
    composites: Dict[Tuple[int, int], WindowComposite] = {}
    steps_per_day = int(round(1440 / step_minutes))

    df_all = pd.concat(
        {"SW_IN": work.get(sw_col), "PPFD_IN": work.get(ppfd_col), "SW_IN_POT": pot},
        axis=1,
    )

    # Year/window grouping
    years = work.index.year
    windows = _fifteen_day_window_id(work.index.dayofyear)  # type: ignore
    gkey = pd.MultiIndex.from_arrays([years, windows], names=["year", "window_id"])  # type: ignore

    for (year, window_id), sub in df_all.groupby(gkey):
        # Build composites
        comp_pot = _max_diurnal_composite(
            sub["SW_IN_POT"].dropna(), step_minutes=step_minutes
        )

        comp_sw = None
        if "SW_IN" in sub and sub["SW_IN"].notna().any():
            comp_sw = _max_diurnal_composite(
                sub["SW_IN"].dropna(), step_minutes=step_minutes
            )

        comp_ppfd = None
        if "PPFD_IN" in sub and sub["PPFD_IN"].notna().any():
            comp_ppfd = _max_diurnal_composite(
                sub["PPFD_IN"].dropna(), step_minutes=step_minutes
            )

        # Percent exceedance (observed > potential) using composites
        pct_exceed_sw = None
        lag_sw = None
        corr_sw = None
        if comp_sw is not None:
            mask = np.isfinite(comp_sw) & np.isfinite(comp_pot)
            if mask.any():
                pct_exceed_sw = 100.0 * np.mean(comp_sw[mask] > comp_pot[mask])
            lag_sw, corr_sw = _xcorr_best_lag(comp_pot, comp_sw, max_lag=max_lag_steps)

        pct_exceed_ppfd = None
        lag_ppfd = None
        corr_ppfd = None
        if comp_ppfd is not None:
            mask = np.isfinite(comp_ppfd) & np.isfinite(comp_pot)
            if mask.any():
                pct_exceed_ppfd = 100.0 * np.mean(comp_ppfd[mask] > comp_pot[mask])
            lag_ppfd, corr_ppfd = _xcorr_best_lag(
                comp_pot, comp_ppfd, max_lag=max_lag_steps
            )

        composites[(year, window_id)] = WindowComposite(
            year=year,
            window_id=window_id,
            step_minutes=step_minutes,
            steps_per_day=steps_per_day,
            comp_pot=comp_pot,
            comp_sw=comp_sw,
            comp_ppfd=comp_ppfd,
            pct_exceed_sw=pct_exceed_sw,  # type: ignore
            pct_exceed_ppfd=pct_exceed_ppfd,  # type: ignore
            lag_sw=lag_sw,
            corr_sw=corr_sw,
            lag_ppfd=lag_ppfd,
            corr_ppfd=corr_ppfd,
        )

        out_rows.append(
            {
                "year": year,
                "window_id": window_id,
                "step_minutes": step_minutes,
                "pct_exceed_sw": pct_exceed_sw,
                "pct_exceed_ppfd": pct_exceed_ppfd,
                "lag_steps_sw": lag_sw,
                "corr_sw": corr_sw,
                "lag_steps_ppfd": lag_ppfd,
                "corr_ppfd": corr_ppfd,
            }
        )

    summary_df = (
        pd.DataFrame(out_rows).sort_values(["year", "window_id"]).reset_index(drop=True)
    )
    return summary_df, composites


# ---------------------------- Heuristic flags -------------------------------- #


def flag_issues(summary: pd.DataFrame) -> Dict[str, str]:
    """
    Apply simple heuristics to flag likely timestamp and data quality issues.

    This function uses a set of heuristics to identify potential issues
    such as timezone mismatches, DST usage, and sensor problems based on
    the summary of the timestamp alignment analysis.

    Parameters
    ----------
    summary : pd.DataFrame
        A DataFrame containing the summary of the timestamp alignment analysis.

    Returns
    -------
    dict[str, str]
        A dictionary of flagged issues, where keys are issue types and
        values are descriptive messages.
    """
    notes = {}

    # Overall stats across windows
    sw_lags = summary["lag_steps_sw"].dropna().astype(int)
    ppfd_lags = summary["lag_steps_ppfd"].dropna().astype(int)
    sw_ex = summary["pct_exceed_sw"].dropna()
    ppfd_ex = summary["pct_exceed_ppfd"].dropna()

    def majority_abs_ge(series: pd.Series, thr: int, frac=0.6) -> bool:
        return (series.abs() >= thr).mean() >= frac if len(series) else False

    # Time zone mismatch (≈ 1h) if many windows at |lag| >= 2 steps (30-min data)
    if majority_abs_ge(sw_lags, 2) or majority_abs_ge(ppfd_lags, 2):
        notes["timezone_or_dst"] = (
            "Many 15-day windows show ~1-hour offset (|lag|≥2 steps for 30-min data): "
            "possible time-zone mismatch or DST usage."
        )

    # START vs END mis-specification (~30 min)
    if majority_abs_ge(sw_lags, 1) and (sw_lags.abs() == 1).mean() > 0.5:
        notes["start_vs_end"] = (
            "Cross-correlation peaks at ±1 step (~30 min) in most windows: "
            "possible START vs END timestamp mis-specification."
        )

    # DST heuristic: look for lag jumping by ~2 steps around typical DST months
    if len(sw_lags):
        df = summary.dropna(subset=["lag_steps_sw"]).copy()
        # Approximate month from window center
        df["month_est"] = (
            ((df["window_id"] - 0.5) * 15 / 30.5 + 1).clip(1, 12).round().astype(int)
        )
        spring = df[df["month_est"].between(3, 4)]["lag_steps_sw"].astype(int)
        fall = df[df["month_est"].between(10, 11)]["lag_steps_sw"].astype(int)
        if len(spring) and len(fall) and (spring.mean() - fall.mean()) >= 2:
            notes["dst"] = (
                "Lag changes by ~2 steps between spring and fall windows: "
                "data may include DST timestamps."
            )

    # Stream desynchronization: SW vs PPFD lags disagree
    if len(sw_lags) and len(ppfd_lags):
        if (sw_lags - ppfd_lags).abs().median() >= 1:
            notes["desync"] = (
                "SW_IN and PPFD_IN peak at different lags: streams may be desynchronized."
            )

    # Radiation sensor anomalies: frequent exceedance of potential
    if (len(sw_ex) and (sw_ex > 5).mean() > 0.5) or (
        len(ppfd_ex) and (ppfd_ex > 5).mean() > 0.5
    ):
        notes["radiation_qc"] = (
            "Observed radiation frequently exceeds potential (early/late-day exceedance): "
            "check leveling, shading, and calibration; also verify timestamp alignment."
        )

    return notes


# ------------------------------- Plotting ------------------------------------ #


def plot_summary(
    summary: pd.DataFrame,
    composites: Dict[Tuple[int, int], WindowComposite],
    which_year: Optional[int] = None,
    outfile_prefix: Optional[str] = None,
):
    """
    Create and display summary plots of the timestamp alignment analysis.

    This function generates a set of plots to visualize the results of the
    timestamp alignment analysis, including percent exceedance, lag times,
    and a composite overlay for the "worst" window.

    Parameters
    ----------
    summary : pd.DataFrame
        A DataFrame containing the summary of the analysis.
    composites : dict
        A dictionary of WindowComposite objects for each analysis window.
    which_year : int, optional
        The year to plot. If None, all years are plotted. Defaults to None.
    outfile_prefix : str, optional
        A prefix for the output plot filenames. If provided, plots are
        saved to files. Defaults to None.

    Returns
    -------
    dict
        A dictionary of matplotlib Figure handles for the generated plots.
    """
    figs = {}
    data = summary.copy()
    if which_year is not None:
        data = data[data["year"] == which_year]
        if data.empty:
            raise ValueError(f"No data found for year={which_year}")

    # Panel A: exceedance bars
    figA, axA = plt.subplots(figsize=(10, 4))
    x = np.arange(len(data))
    axA.bar(x - 0.2, data["pct_exceed_sw"].values, width=0.4, label="% exceed SW_IN")  # type: ignore
    if "pct_exceed_ppfd" in data:
        axA.bar(
            x + 0.2, data["pct_exceed_ppfd"].values, width=0.4, label="% exceed PPFD_IN"  # type: ignore
        )
    axA.set_xticks(x)
    axA.set_xticklabels([f"W{w}" for w in data["window_id"]], rotation=0)
    axA.set_ylabel("% of composite steps with obs > POT")
    axA.set_title("Percent exceedance by 15-day window")
    axA.legend()
    figs["exceedance"] = figA
    if outfile_prefix:
        figA.savefig(f"{outfile_prefix}_exceedance.png", dpi=150, bbox_inches="tight")

    # Panel B: lag bars
    figB, axB = plt.subplots(figsize=(10, 4))
    axB.bar(x - 0.2, data["lag_steps_sw"].values, width=0.4, label="Lag steps (SW_IN)")  # type: ignore
    if "lag_steps_ppfd" in data:
        axB.bar(
            x + 0.2,
            data["lag_steps_ppfd"].values,  # type: ignore
            width=0.4,
            label="Lag steps (PPFD_IN)",
        )
    axB.axhline(0, linewidth=1)
    axB.set_xticks(x)
    axB.set_xticklabels([f"W{w}" for w in data["window_id"]])
    axB.set_ylabel("Lag (steps); for 30-min data, 2 steps ≈ 1 hour")
    axB.set_title("Best cross-correlation lag by 15-day window")
    axB.legend()
    figs["lags"] = figB
    if outfile_prefix:
        figB.savefig(f"{outfile_prefix}_lags.png", dpi=150, bbox_inches="tight")

    # Panel C: worst window overlay
    # Choose the window with max absolute lag (preferring SW), else max exceedance
    cand = data.copy()
    cand["_score"] = cand["lag_steps_sw"].abs().fillna(0)
    if cand["_score"].max() == 0:
        cand["_score"] = cand["pct_exceed_sw"].fillna(0)
    yr, w = int(cand.iloc[cand["_score"].idxmax()]["year"]), int(  # type: ignore
        cand.iloc[cand["_score"].idxmax()]["window_id"]  # type: ignore
    )
    wc = composites[(yr, w)]
    t = np.arange(wc.steps_per_day) * wc.step_minutes / 60.0  # hours since midnight

    figC, axC = plt.subplots(figsize=(10, 4))
    axC.plot(t, wc.comp_pot, label="SW_IN_POT (TOA)")
    if wc.comp_sw is not None:
        axC.plot(t, wc.comp_sw, label="SW_IN (max diurnal composite)")
    if wc.comp_ppfd is not None:
        axC.plot(t, wc.comp_ppfd, label="PPFD_IN (max diurnal composite)")
    axC.set_xlabel("Local standard time (hours)")
    axC.set_ylabel("Irradiance / PPFD (units as provided)")
    axC.set_title(f"Window W{w} in {yr}: composite overlay (worst window)")
    axC.legend()
    figs["overlay"] = figC
    if outfile_prefix:
        figC.savefig(
            f"{outfile_prefix}_overlay_W{w}_{yr}.png", dpi=150, bbox_inches="tight"
        )

    plt.show()
    return figs
