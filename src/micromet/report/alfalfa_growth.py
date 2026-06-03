from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd


# -----------------------------
# Utilities: time axis + inputs
# -----------------------------


def to_datetime_index(dates: Union[Sequence, pd.DatetimeIndex]) -> pd.DatetimeIndex:
    """Convert array-like to a normalized (midnight), sorted DatetimeIndex."""
    idx = pd.to_datetime(pd.Index(dates))
    # Normalize to midnight to avoid time-of-day misalignment
    idx = pd.DatetimeIndex(idx).normalize()
    idx = idx.sort_values()
    return idx


def make_daily_index(dates: Union[Sequence, pd.DatetimeIndex]) -> pd.DatetimeIndex:
    """Ensure daily index (continuous). If input is already daily, returns the full daily range."""
    idx = to_datetime_index(dates)
    return pd.date_range(idx.min(), idx.max(), freq="D")


def clean_cut_dates(
    cut_dates: Sequence,
    date_min: pd.Timestamp,
    date_max: pd.Timestamp,
    allow_outside_range: bool = True,
) -> List[pd.Timestamp]:
    """Normalize, sort, and deduplicate cut dates. Optionally keep those outside the modeled window."""
    cuts = pd.to_datetime(pd.Index(cut_dates)).normalize()
    cuts = cuts.dropna()
    cuts = pd.DatetimeIndex(sorted(set(cuts)))

    if not allow_outside_range:
        cuts = cuts[(cuts >= date_min) & (cuts <= date_max)]

    return list(cuts)


# -----------------------------
# GDD computation (optional)
# -----------------------------


def daily_gdd_simple(
    tmin: Union[np.ndarray, pd.Series],
    tmax: Union[np.ndarray, pd.Series],
    tbase_c: float = 5.0,
    tcap_c: Optional[float] = None,
) -> np.ndarray:
    """
    Daily GDD using simple temperature averaging:
        GDD = max(0, ((Tmax + Tmin)/2) - Tbase)

    If tcap_c is provided, Tmax and Tmin are capped at tcap_c before averaging.
    """
    tmin = np.asarray(tmin, dtype=float)
    tmax = np.asarray(tmax, dtype=float)

    if tcap_c is not None:
        tmin = np.minimum(tmin, tcap_c)
        tmax = np.minimum(tmax, tcap_c)

    tmean = 0.5 * (tmin + tmax)
    gdd = np.maximum(0.0, tmean - tbase_c)
    return gdd


def compute_gdd_series(
    dates: pd.DatetimeIndex,
    weather: Optional[pd.DataFrame],
    tbase_c: float = 5.0,
    tcap_c: Optional[float] = None,
    tmin_col: str = "tmin_c",
    tmax_col: str = "tmax_c",
    gdd_col: str = "gdd",
) -> Optional[pd.Series]:
    """
    Return a daily GDD series indexed by dates if possible.

    Accepts either:
      - precomputed 'gdd' column, or
      - Tmin/Tmax columns to compute GDD.

    If weather is None or missing required columns, returns None.
    """
    if weather is None:
        return None

    w = weather.copy()
    if not isinstance(w.index, pd.DatetimeIndex):
        raise ValueError("weather must have a DatetimeIndex")

    w.index = w.index.normalize()

    # Reindex to full model dates
    w = w.reindex(dates)

    if gdd_col in w.columns and w[gdd_col].notna().any():
        return w[gdd_col].astype(float)

    if (
        tmin_col in w.columns
        and tmax_col in w.columns
        and w[tmin_col].notna().any()
        and w[tmax_col].notna().any()
    ):
        gdd = daily_gdd_simple(w[tmin_col], w[tmax_col], tbase_c=tbase_c, tcap_c=tcap_c)
        return pd.Series(gdd, index=dates, name="gdd")

    return None


# -----------------------------
# Environmental modifiers (optional)
# -----------------------------


def temperature_stress_piecewise(
    tmean_c: Union[np.ndarray, pd.Series],
    t_min_c: float = 0.0,
    t_opt_low_c: float = 15.0,
    t_opt_high_c: float = 27.0,
    t_stop_c: float = 30.0,
) -> np.ndarray:
    """
    Simple piecewise temperature stress factor in [0, 1].
    - 0 below t_min_c
    - 1 between [t_opt_low_c, t_opt_high_c]
    - linearly declines to 0 between t_opt_high_c and t_stop_c
    - 0 at/above t_stop_c

    Notes:
      - Use when you want an explicit hot-temperature slowdown.
      - Defaults are illustrative; calibrate for Utah fields if you have data.
    """
    t = np.asarray(tmean_c, dtype=float)
    f = np.ones_like(t)

    f[t <= t_min_c] = 0.0

    # below optimal range: ramp up (optional, can keep 1.0 if you prefer)
    mask_low = (t > t_min_c) & (t < t_opt_low_c)
    if t_opt_low_c > t_min_c:
        f[mask_low] = (t[mask_low] - t_min_c) / (t_opt_low_c - t_min_c)

    # optimal range
    mask_opt = (t >= t_opt_low_c) & (t <= t_opt_high_c)
    f[mask_opt] = 1.0

    # above optimal: ramp down
    mask_high = (t > t_opt_high_c) & (t < t_stop_c)
    if t_stop_c > t_opt_high_c:
        f[mask_high] = 1.0 - (t[mask_high] - t_opt_high_c) / (t_stop_c - t_opt_high_c)

    f[t >= t_stop_c] = 0.0
    return np.clip(f, 0.0, 1.0)


def default_water_stress_none(n: int) -> np.ndarray:
    """Default 'no water limitation' factor."""
    return np.ones(n, dtype=float)


# -----------------------------
# Growth functions
# -----------------------------


def growth_linear(time: np.ndarray, h0: float, hmax: float, r: float) -> np.ndarray:
    return np.minimum(hmax, h0 + r * time)


def growth_exp_asymptotic(
    time: np.ndarray, h0: float, hmax: float, k: float
) -> np.ndarray:
    return hmax - (hmax - h0) * np.exp(-k * time)


def growth_logistic(time: np.ndarray, h0: float, hmax: float, k: float) -> np.ndarray:
    if h0 <= 0:
        raise ValueError("Logistic growth requires h0 > 0 to anchor the curve.")
    A = (hmax - h0) / h0
    return hmax / (1.0 + A * np.exp(-k * time))


GrowthFn = Callable[[np.ndarray, float, float, float], np.ndarray]


def choose_growth_fn(model: str) -> GrowthFn:
    """
    model:
      - 'linear'
      - 'exp'
      - 'logistic'
    """
    m = model.lower().strip()
    if m == "linear":
        return growth_linear
    if m in ("exp", "exponential", "asymptotic_exp", "monomolecular"):
        return growth_exp_asymptotic
    if m == "logistic":
        return growth_logistic
    raise ValueError(f"Unknown model: {model}")


# -----------------------------
# Dormancy helpers
# -----------------------------


def is_active_season_by_doy(
    dates: pd.DatetimeIndex,
    start_mmdd: Tuple[int, int] = (3, 1),  # Mar 1
    end_mmdd: Tuple[int, int] = (10, 31),  # Oct 31
) -> np.ndarray:
    """
    Simple Utah-oriented active season proxy when weather is missing.
    Returns boolean array aligned to dates.
    """
    start_month, start_day = start_mmdd
    end_month, end_day = end_mmdd
    years = dates.year

    active = np.zeros(len(dates), dtype=bool)
    for y in np.unique(years):
        start = pd.Timestamp(year=y, month=start_month, day=start_day)
        end = pd.Timestamp(year=y, month=end_month, day=end_day)
        mask = (dates >= start) & (dates <= end)
        active[mask] = True
    return active


def is_active_season_by_temp(
    dates: pd.DatetimeIndex,
    tmean_c: pd.Series,
    tbase_c: float = 5.0,
    consecutive_days: int = 5,
) -> np.ndarray:
    """
    Active season defined by temperature:
      active if rolling mean(tmean_c) over 'consecutive_days' >= tbase_c.

    This is a pragmatic approximation; adjust for your agronomic definition of green-up and dormancy.
    """
    tm = tmean_c.reindex(dates).astype(float)
    roll = tm.rolling(consecutive_days, min_periods=consecutive_days).mean()
    active = (roll >= tbase_c).fillna(False).to_numpy()
    return active


# -----------------------------
# Main simulation
# -----------------------------


@dataclass
class AlfalfaHeightParams:
    h_resid_cm: float = 7.5  # ~3 inches
    h_max_cm: float = 75.0  # management-dependent
    rate: Union[float, Sequence[float]] = 0.008 # cm/day (linear) OR day^-1 (k) depending on model
    model: str = "exp"  # 'linear', 'exp', 'logistic'
    time_mode: str = "days"  # 'days' or 'gdd'
    tbase_c: float = 5.0  # 41°F 
    tcap_c: Optional[float] = None  # optional high-temp cap for GDD calc; iif provided, Tmax and Tmin are capped at tcap_c before averaging
    enforce_bounds: bool = True
    grazing_height: Optional[Union[float, list, tuple, np.ndarray]] = None
    # Dormancy behavior
    dormancy_mode: str = "doy"  # 'doy', 'temp', or 'none'; under temp, active when rolling mean(tmean_c) over 'greenup_consecutive_days'  >= tbase_c
    doy_start: Tuple[int, int] = (3, 1) # start date of active season if dormancy_mode is 'doy'
    doy_end: Tuple[int, int] = (10, 31) # end date of active season if dormancy_mode is 'doy'
    tmean_col: str = "tmean_c"  # used if dormancy_mode='temp' and weather provided
    greenup_consecutive_days: int = 5

    # Optional stress
    use_temp_stress: bool = False
    tmin_c: float = 0.0
    topt_low_c: float = 15.0
    topt_high_c: float = 27.0
    tstop_c: float = 30.0


def simulate_alfalfa_height_single_field(
    dates: Union[Sequence, pd.DatetimeIndex],
    cut_dates: Sequence,
    params: AlfalfaHeightParams,
    weather: Optional[pd.DataFrame] = None,
    cut_effect: str = "post",  # 'post' means height on cut date is stubble; 'pre' means apply cut next day
) -> pd.DataFrame:  
    """
    Simulate daily canopy height for one field with dynamic k-rates and dynamic grazing height ceilings.
    """
    idx = make_daily_index(dates)
    date_min, date_max = idx.min(), idx.max()

    cuts = clean_cut_dates(cut_dates, date_min, date_max, allow_outside_range=True)

    # Build GDD series if needed
    gdd = None
    if params.time_mode.lower() == "gdd":
        gdd = compute_gdd_series(
            dates=idx,
            weather=weather,
            tbase_c=params.tbase_c,
            tcap_c=params.tcap_c,
            tmin_col="tmin_c",
            tmax_col="tmax_c",
            gdd_col="gdd",
        )

    # Determine active/dormant days
    if params.dormancy_mode.lower() == "none":
        active = np.ones(len(idx), dtype=bool)
    elif (
        params.dormancy_mode.lower() == "temp"
        and weather is not None
        and params.tmean_col in weather.columns
    ):
        active = is_active_season_by_temp(
            idx,
            tmean_c=weather[params.tmean_col].copy(),
            tbase_c=params.tbase_c,
            consecutive_days=params.greenup_consecutive_days,
        )
    else:
        active = is_active_season_by_doy(idx, params.doy_start, params.doy_end)

    # Optional temperature stress factor
    if (
        params.use_temp_stress
        and weather is not None
        and params.tmean_col in weather.columns
    ):
        temp_stress = temperature_stress_piecewise(
            tmean_c=weather[params.tmean_col].reindex(idx),
            t_min_c=params.tmin_c,
            t_opt_low_c=params.topt_low_c,
            t_opt_high_c=params.topt_high_c,
            t_stop_c=params.tstop_c,
        )
    else:
        temp_stress = np.ones(len(idx), dtype=float)

    growth_fn = choose_growth_fn(params.model)
    cut_set = set(pd.DatetimeIndex(cuts))
    
    # --- Set up arrays for BOTH height and the active k-rate ---
    heights = np.full(len(idx), np.nan, dtype=float)
    k_values = np.full(len(idx), np.nan, dtype=float)  

    last_reset_pos = 0
    current_cut_idx = 0  
    
    heights[0] = params.h_resid_cm
    # Initialize the first day's k-rate
    k_values[0] = float(params.rate[0]) if isinstance(params.rate, (list, tuple, np.ndarray, pd.Series)) else float(params.rate)

    end_date = idx[-1]
    drawdown_window = 14  # Days to transition down
    drawdown_start_date = end_date - pd.Timedelta(days=drawdown_window)
    height_at_drawdown_start = None

    for i in range(1, len(idx)):
        d = idx[i]
        
        # Late fall drawdown execution
        if d >= drawdown_start_date:
            if height_at_drawdown_start == None:
                height_at_drawdown_start = float(heights[i - 1])
                
            days_into_drawdown = (d - drawdown_start_date).days
            fraction = max(0.0, (drawdown_window - days_into_drawdown) / drawdown_window)
            
            heights[i] = params.h_resid_cm + (height_at_drawdown_start - params.h_resid_cm) * fraction
            k_values[i] = 0.0
            continue

        is_cut = d in cut_set

        # --- DYNAMIC K-SELECTION ---
        if isinstance(params.rate, (list, tuple, np.ndarray, pd.Series)):
            if current_cut_idx < len(params.rate):
                current_k = float(params.rate[current_cut_idx])
            else:
                current_k = float(params.rate[-1])
        else:
            current_k = float(params.rate)
            
        k_values[i] = current_k  

        # --- NEW: DYNAMIC H_MAX CEILING SELECTION ---
        if hasattr(params, 'grazing_height') and params.grazing_height is not None:
            if isinstance(params.grazing_height, (list, tuple, np.ndarray, pd.Series)):
                if current_cut_idx < len(params.grazing_height):
                    cycle_target = params.grazing_height[current_cut_idx]
                else:
                    cycle_target = params.grazing_height[-1]
            else:
                cycle_target = params.grazing_height
            
            # If the current cycle has a grazing cap assigned, use it. Otherwise, use standard h_max_cm.
            current_h_max = float(cycle_target) if cycle_target is not None else params.h_max_cm
        else:
            current_h_max = params.h_max_cm

        # Apply cut event
        if cut_effect.lower() == "post":
            if is_cut:
                heights[i] = params.h_resid_cm
                last_reset_pos = i
                current_cut_idx += 1  
                continue
        elif cut_effect.lower() == "pre":
            if idx[i - 1] in cut_set:
                heights[i] = params.h_resid_cm
                last_reset_pos = i
                current_cut_idx += 1  
                continue
        else:
            raise ValueError("cut_effect must be 'post' or 'pre'")

        # Dormancy: hold constant (no growth) if not active
        if not active[i]:
            heights[i] = max(params.h_resid_cm, heights[i - 1])
            continue

        # Compute time-since-reset
        if params.time_mode.lower() == "gdd" and gdd is not None:
            tt_today = float(gdd.iloc[last_reset_pos + 1 : i + 1].sum())
            tt_yesterday = float(gdd.iloc[last_reset_pos + 1 : i].sum())
            time_var_today = np.array([tt_today], dtype=float)
            time_var_prev = np.array([tt_yesterday], dtype=float)
        else:
            dt_days_today = float(i - last_reset_pos)
            dt_days_prev = float(i - 1 - last_reset_pos)
            time_var_today = np.array([dt_days_today], dtype=float)
            time_var_prev = np.array([dt_days_prev], dtype=float)

        # Get ideal curve heights (Updated to use current_h_max)
        h_ideal_today = float(growth_fn(time_var_today, params.h_resid_cm, current_h_max, current_k)[0])
        h_ideal_prev = float(growth_fn(time_var_prev, params.h_resid_cm, current_h_max, current_k)[0])

        delta_growth = max(0.0, h_ideal_today - h_ideal_prev)
        delta_stressed = delta_growth * float(temp_stress[i])
        h_stressed = heights[i - 1] + delta_stressed

        if params.enforce_bounds:
            h_stressed = max(params.h_resid_cm, min(current_h_max, h_stressed))

        heights[i] = h_stressed

    return pd.DataFrame({
        "height_cm": heights,
        "active_k": k_values
    }, index=idx)


def simulate_alfalfa_height_multi_field(
    dates: Union[Sequence, pd.DatetimeIndex],
    cut_dates_by_field: Mapping[str, Sequence],
    params_by_field: Optional[Mapping[str, AlfalfaHeightParams]] = None,
    default_params: Optional[AlfalfaHeightParams] = None,
    weather_by_field: Optional[Mapping[str, pd.DataFrame]] = None,
    cut_effect: str = "post",
) -> pd.DataFrame:
    """
    Simulate daily canopy height for multiple fields.

    cut_dates_by_field: dict(field_id -> list of cut dates)
    params_by_field: optional dict(field_id -> AlfalfaHeightParams)
    weather_by_field: optional dict(field_id -> daily weather DataFrame)

    Returns DataFrame indexed by daily date, columns=field_id, values in cm.
    """
    idx = make_daily_index(dates)

    if default_params is None:
        default_params = AlfalfaHeightParams()

    out = {}
    for field_id, cuts in cut_dates_by_field.items():
        p = default_params
        if params_by_field is not None and field_id in params_by_field:
            p = params_by_field[field_id]

        w = None
        if weather_by_field is not None and field_id in weather_by_field:
            w = weather_by_field[field_id]

        s = simulate_alfalfa_height_single_field(
            dates=idx,
            cut_dates=cuts,
            params=p,
            weather=w,
            cut_effect=cut_effect,
        )
        out[field_id] = s

    df = pd.DataFrame(out, index=idx)
    df.index.name = "date"
    return df
