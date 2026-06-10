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
    """Configuration parameters for the alfalfa canopy height simulation model.

    Attributes:
        h_resid_cm (float): Post-harvest residual or stubble height in centimeters.
            Serves as the baseline minimum height. Defaults to 5 cm (~2 inches).
        h_max_cm (float): Default management-dependent maximum potential height 
            of the alfalfa canopy in centimeters. Defaults to 75.0.
        rate (float or Sequence[float]): Growth rate parameter. Interpreted as 
            cm/day for the linear model, or day⁻¹ (k) for exponential/logistic 
            models. Can accept a single value or a sequence of values mapped 
            chronologically to each growth cycle. Defaults to 0.008.
        model (str): The growth mathematical model to deploy. Accepted values 
            are 'linear', 'exp' (or 'exponential'), and 'logistic'. 
            Defaults to "exp".
        time_mode (str): Driving variable for time progression. Use 'days' for 
            calendar days or 'gdd' for Growing Degree Days. Defaults to "days".
        tbase_c (float): Base temperature threshold in Celsius below which 
            alfalfa growth ceases. Used for GDD accumulation and temperature-based 
            dormancy calculations. Defaults to 5.0 (41°F).
        tcap_c (float, optional): High-temperature ceiling in Celsius for the 
            GDD calculation. If provided, daily maximum and minimum temperatures 
            are capped at this value before averaging. Defaults to None.
        enforce_bounds (bool): If True, forces the simulated height array to be 
            strictly clamped between `h_resid_cm` and the current maximum height. 
            Defaults to True.
        grazing_height (float or Sequence[float], optional): Target maximum height 
            ceilings used to cap growth specifically during grazing intervals. Can 
            be a single value or sequence per harvest cycle. Defaults to None.
        dormancy_mode (str): Strategy for identifying active vs. dormant periods. 
            Options are 'doy' (Day of Year range), 'temp' (rolling temperature 
            threshold), or 'none' (active year-round). Defaults to "temp".
        doy_start (Tuple[int, int]): Month and day marking the beginning of the 
            active growing season (Month, Day) when `dormancy_mode` is 'doy'. 
            Defaults to (3, 1) (March 1st).
        doy_end (Tuple[int, int]): Month and day marking the end of the active 
            growing season (Month, Day) when `dormancy_mode` is 'doy'. 
            Defaults to (10, 31) (October 31st).
        tmean_col (str): Column name in the input `weather` DataFrame representing 
            the daily mean temperature. Required if `dormancy_mode` is 'temp'. 
            Defaults to "tmean_c".
        greenup_consecutive_days (int): Number of consecutive days that the rolling 
            mean temperature must remain above `tbase_c` to trigger spring 
            green-up when using 'temp' dormancy. Defaults to 5.
    """
    h_resid_cm: float = 5
    h_max_cm: float = 75.0  
    rate: Union[float, Sequence[float]] = 0.008 
    model: str = "exp"  
    time_mode: str = "days"  
    tbase_c: float = 5.0  
    tcap_c: Optional[float] = None  
    enforce_bounds: bool = True
    grazing_height: Optional[Union[float, list, tuple, np.ndarray]] = None
    # Dormancy behavior
    dormancy_mode: str = "temp"  
    doy_start: Tuple[int, int] = (3, 1) 
    doy_end: Tuple[int, int] = (10, 31) 
    tmean_col: str = "tmean_c"  
    greenup_consecutive_days: int = 5

def simulate_alfalfa_height_single_field(
    dates: Union[Sequence, pd.DatetimeIndex],
    cut_dates: Sequence,
    params: AlfalfaHeightParams,
    weather: Optional[pd.DataFrame] = None,
    cut_effect: str = "post",
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

    growth_fn = choose_growth_fn(params.model)
    cut_set = set(pd.DatetimeIndex(cuts))
    
    heights = np.full(len(idx), np.nan, dtype=float)
    k_values = np.full(len(idx), np.nan, dtype=float)  

    last_reset_pos = 0
    current_cut_idx = 0  
    
    heights[0] = params.h_resid_cm
    k_values[0] = float(params.rate[0]) if isinstance(params.rate, (list, tuple, np.ndarray, pd.Series)) else float(params.rate)

    end_date = idx[-1]
    drawdown_window = 14
    drawdown_start_date = end_date - pd.Timedelta(days=drawdown_window)
    height_at_drawdown_start = None

    for i in range(1, len(idx)):
        d = idx[i]
        
        # Late fall drawdown execution
        if d >= drawdown_start_date:
            if height_at_drawdown_start is None:
                height_at_drawdown_start = float(heights[i - 1])
                
            days_into_drawdown = (d - drawdown_start_date).days
            fraction = max(0.0, (drawdown_window - days_into_drawdown) / drawdown_window)
            
            heights[i] = params.h_resid_cm + (height_at_drawdown_start - params.h_resid_cm) * fraction
            k_values[i] = 0.0
            continue

        is_cut = d in cut_set

        # Dynamic K-Selection
        if isinstance(params.rate, (list, tuple, np.ndarray, pd.Series)):
            if current_cut_idx < len(params.rate):
                current_k = float(params.rate[current_cut_idx])
            else:
                current_k = float(params.rate[-1])
        else:
            current_k = float(params.rate)
            
        k_values[i] = current_k  

        # Dynamic H_MAX Ceiling Selection
        if hasattr(params, 'grazing_height') and params.grazing_height is not None:
            if isinstance(params.grazing_height, (list, tuple, np.ndarray, pd.Series)):
                if current_cut_idx < len(params.grazing_height):
                    cycle_target = params.grazing_height[current_cut_idx]
                else:
                    cycle_target = params.grazing_height[-1]
            else:
                cycle_target = params.grazing_height
            
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

        # Dormancy: hold constant if not active
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

        # Get ideal curve heights
        h_ideal_today = float(growth_fn(time_var_today, params.h_resid_cm, current_h_max, current_k)[0])
        h_ideal_prev = float(growth_fn(time_var_prev, params.h_resid_cm, current_h_max, current_k)[0])

        delta_growth = max(0.0, h_ideal_today - h_ideal_prev)
        h_new = heights[i - 1] + delta_growth

        if params.enforce_bounds:
            h_new = max(params.h_resid_cm, min(current_h_max, h_new))

        heights[i] = h_new

    return pd.DataFrame({
        "height_cm": heights,
        "active_k": k_values
    }, index=idx)


def generate_field2_heights(field1_data, field1_cuts, field2_cuts, h_resid_cm, catchup_days=14):
    """
    Generate a daily crop height profile for a secondary field (Field 2) by 
    copying and adjusting the simulated growth curve of a primary field (Field 1).
    
    This function synchronizes two field curves by evaluating cutting timelines 
    cycle-by-cycle. It resolves mismatches caused by staggered cutting dates 
    using a linear blend to eliminate unnatural vertical "jumps" in height data.

    Logic Scenarios:
    ----------------
    Scenario A (Field 1 cut first, c1 < c2):
        - Field 2 holds Field 1's last pre-cut maximum height during the gap.
        - On its actual cut date (c2), Field 2 drops to `h_resid_cm`.
        - For the next `catchup_days`, Field 2's height is linearly scaled from 
          residual height back up to match Field 1's ongoing active growth curve.

    Scenario B (Field 2 cut first, c2 < c1):
        - Field 2 drops immediately to `h_resid_cm` on its early cut date (c2).
        - Field 2 stays flat at baseline residual height until Field 1 is cut (c1).
        - Once both fields have been cut, they naturally re-align and grow together.

    Args:
        field1_data (pd.Series): Daily height data for Field 1, indexed by datetime.
        field1_cuts (iterable): Chronological collection of cutting dates for Field 1 
            (can be strings, datetimes, or Timestamps).
        field2_cuts (iterable): Chronological collection of cutting dates for Field 2 
            (must pair 1:1 in sequence with Field 1 cuts).
        h_resid_cm (float): The baseline post-harvest residual/stubble height (minimum height).
        catchup_days (int, optional): The number of days allowed for Field 2 to smoothly 
            ramp up and catch up to Field 1's growth line in Scenario A. Defaults to 14.

    Returns:
        pd.Series: A new Pandas Series containing the daily calculated heights for Field 2, 
            matching the exact index of `field1_data`.

    Raises:
        IndexError: If the number of cuts in `field1_cuts` and `field2_cuts` do not match.
    """
    # Start by making a clean copy of Field 1's height series
    field2_heights = field1_data.copy()
    
    # Ensure all cut dates are sorted pandas Timestamps
    f1_cuts = sorted([pd.to_datetime(d) for d in field1_cuts])
    f2_cuts = sorted([pd.to_datetime(d) for d in field2_cuts])
    
    # Iterate through pairs of cuts cycle-by-cycle
    for c1, c2 in zip(f1_cuts, f2_cuts):
        
        # --- SCENARIO A: Field 1 is cut first ---
        if c1 < c2:
            # 1. Get Field 1's height right before it was cut
            idx_before_c1 = field1_data.index[field1_data.index < c1]
            if len(idx_before_c1) > 0:
                h_pre_cut = float(field1_data.loc[idx_before_c1[-1]])
            else:
                h_pre_cut = h_resid_cm
                
            # 2. Field 2 holds that high pre-cut value until the day before it is cut
            mismatch_days = field2_heights.index[(field2_heights.index >= c1) & (field2_heights.index < c2)]
            field2_heights.loc[mismatch_days] = h_pre_cut
            
            # 3. Drop Field 2 to residue on its actual cut date
            if c2 in field2_heights.index:
                field2_heights.loc[c2] = h_resid_cm
                
            # 4. LINEAR FIX: Smoothly blend from h_resid_cm back up to Field 1's active curve
            for step in range(1, catchup_days + 1):
                blend_date = c2 + pd.Timedelta(days=step)
                
                if blend_date in field2_heights.index:
                    fraction = step / catchup_days
                    f1_height_today = float(field1_data.loc[blend_date])
                    
                    # Blend formula
                    field2_heights.loc[blend_date] = h_resid_cm + (f1_height_today - h_resid_cm) * fraction
                
        # --- SCENARIO B: Field 2 is cut first ---
        elif c2 < c1:
            # Field 2 drops to minimum height immediately and stays there until Field 1 is cut.
            mismatch_days = field2_heights.index[(field2_heights.index >= c2) & (field2_heights.index <= c1)]
            field2_heights.loc[mismatch_days] = h_resid_cm

    return field2_heights
