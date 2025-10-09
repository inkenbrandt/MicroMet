"""
Data validation and quality control functions for the reformatter pipeline.

This module handles applying physical limits to data values and detecting
stuck or anomalous sensor readings.
"""

import math
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd

import micromet.qaqc.variable_limits as variable_limits


def apply_physical_limits(
    df: pd.DataFrame,
    how: str = "mask",
    inplace: bool = False,
    prefer_longest_key: bool = True,
    return_mask: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """
    Apply physical Min/Max bounds to columns in a DataFrame.

    This function applies physical limits (minimum and maximum) to the columns
    of a DataFrame. It can either mask out-of-bounds values with NaN or clip
    them to the limits.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to which the limits will be applied.
    how : str, optional
        The method to use for applying limits: 'mask' (default) or 'clip'.
    inplace : bool, optional
        If True, modify the DataFrame in place. Defaults to False.
    prefer_longest_key : bool, optional
        If True, prefer longer matching keys from the limits dictionary.
        Defaults to True.
    return_mask : bool, optional
        If True, return a boolean mask of the values that were flagged.
        Defaults to False.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]
        A tuple containing:
        - The DataFrame with physical limits applied.
        - A boolean mask of flagged values (if `return_mask` is True).
        - A report summarizing the number of flagged values for each column.
    """
    if how not in {"mask", "clip"}:
        raise ValueError("how must be 'mask' or 'clip'")

    limits_dict = variable_limits.limits

    out = df if inplace else df.copy()
    no_limits = ['CO2_DENSITY_SIGMA', 'FC_SAMPLES', 
                 'H_SAMPLES','LE_SAMPLES', 'RECORD',
                 'TAU_QC']
    col_list = [i for i in out.columns if i not in no_limits]


    keys = list(limits_dict.keys())
    if prefer_longest_key:
        keys.sort(key=len, reverse=True)

    col_map = {}
    for key in keys:
        matching_cols = [c for c in col_list if str(c).startswith(key)]
        if not matching_cols:
            continue
        lim = limits_dict[key]
        mn = lim.get("Min", np.nan)
        mx = lim.get("Max", np.nan)
        for col in matching_cols:
            if col not in col_map or (
                prefer_longest_key and len(key) > len(col_map[col]["key"])
            ):
                col_map[col] = {"key": key, "Min": mn, "Max": mx}

    mask_df = pd.DataFrame(False, index=out.index, columns=out.columns)
    records = []

    NA_PLACEHOLDER = -9999 
    PLACEHOLDER2 =  -999900

    for col, info in col_map.items():
        key = info["key"]
        mn = info["Min"]
        mx = info["Max"]
        ser = pd.to_numeric(out[col], errors="coerce")
        is_na_placeholder = ser == NA_PLACEHOLDER
        ser = ser.mask(is_na_placeholder, np.nan)
        is_na_placeholder2 = ser == PLACEHOLDER2
        ser = ser.mask(is_na_placeholder2, np.nan)

        lower_ok = (
            ser >= mn
            if not (pd.isna(mn) or (isinstance(mn, float) and math.isnan(mn)))
            else pd.Series(True, index=ser.index)
        )
        upper_ok = (
            ser <= mx
            if not (pd.isna(mx) or (isinstance(mx, float) and math.isnan(mx)))
            else pd.Series(True, index=ser.index)
        )
        ok = lower_ok & upper_ok
        oor = ~ok
        n_below = int((~lower_ok & ser.notna()).sum())
        n_above = int((~upper_ok & ser.notna()).sum())
        n_oor = int((oor & ser.notna()).sum())
        if how == "mask":
            ser_out = ser.where(ok)
        else:
            ser_out = ser
            if not pd.isna(mn):
                ser_out = ser_out.clip(lower=mn)
            if not pd.isna(mx):
                ser_out = ser_out.clip(upper=mx)
        out[col] = ser_out.astype(float) if ser_out.isna().any() else ser_out
        mask_df[col] = oor
        records.append(
            {
                "column": col,
                "matched_key": key,
                "min": mn,
                "max": mx,
                "n_below": n_below,
                "n_above": n_above,
                "n_flagged": n_oor,
                "pct_flagged": (n_oor / ser.notna().sum() * 100.0) if ser.notna().sum() else 0.0,
            }
        )
    report = pd.DataFrame.from_records(records).sort_values(
        ["n_flagged", "column"], ascending=[False, True]
    )
    return (out, (mask_df if return_mask else None), report)


def mask_stuck_values(
    df: pd.DataFrame,
    threshold: Union[int, str, pd.Timedelta],
    columns: Optional[Iterable[str]] = None,
    tolerance: Optional[float] = None,
    mask_value=np.nan,
    return_mask: bool = False,
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
]:
    """
    Detect and mask 'stuck' values in a datetime-indexed DataFrame.

    A run is considered 'stuck' when the series does not change (within an optional
    numeric tolerance) for at least `threshold`. Threshold can be a count of rows
    (int) or a time duration (str like '30min' / '2H' or pd.Timedelta).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex (required).
    threshold : int | str | pd.Timedelta
        Minimum length of a non-changing run to be masked.
        - If int: count of consecutive rows (e.g., 5).
        - If str or Timedelta: minimum duration (e.g., '30min', pd.Timedelta('2H')).
    columns : iterable[str], optional
        Subset of columns to check. Defaults to all columns.
    tolerance : float, optional
        For numeric columns only: treat changes with absolute difference <= tolerance
        as 'no change'. If None, exact equality is used.
    mask_value : any, default np.nan
        Value to assign to masked entries.
    return_mask : bool, default False
        If True, also return a boolean DataFrame mask where True marks masked cells.

    Returns
    -------
    masked_df : pd.DataFrame
        Copy of `df` with stuck runs masked.
    report : pd.DataFrame
        Tidy report with one row per masked run, columns:
        ['column','value','start','end','n_rows','duration','threshold_type','threshold_value']
    mask_df : pd.DataFrame (optional)
        Boolean DataFrame (same shape as `df[columns]`) with True where values were masked.

    Notes
    -----
    - NaNs act as boundaries and are never considered part of a 'stuck' run.
    - For irregular time steps and time-based thresholds, the run 'duration'
      is computed as end_time - start_time (inclusive of row timestamps).
    - Entire runs that meet/exceed the threshold are masked (not just the tail beyond threshold).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex.")

    # Normalize inputs
    cols = list(columns) if columns is not None else list(df.columns)

    if isinstance(threshold, int):
        thresh_type = "count"
        thresh_count = threshold
        thresh_delta = None
    else:
        thresh_type = "time"
        thresh_delta = pd.to_timedelta(threshold)
        thresh_count = None

    # Prepare mask and report accumulator
    mask_df = pd.DataFrame(False, index=df.index, columns=cols)
    report_rows = []

    for col in cols:
        s = df[col]

        # Boundaries: treat NaNs as breaking runs
        notna = s.notna()

        # Determine "change points"
        if pd.api.types.is_numeric_dtype(s) and tolerance is not None:
            # consider 'no change' if difference <= tolerance
            # mark a change when |diff| > tol
            diff = s.diff().abs()
            changed = (diff > tolerance) | (~notna) | (~notna.shift(1, fill_value=False))  # type: ignore
        else:
            # exact equality
            # change occurs when current != previous OR either is NaN
            prev = s.shift(1)
            changed = (s != prev) | (~notna) | (~prev.notna())

        # Group by segments of constant value (between change points)
        group_id = changed.cumsum()

        # Iterate groups that are non-NaN and constant
        for gid, idx in s.groupby(group_id).groups.items():
            # idx is an index of row positions (labels)
            block = s.loc[idx]
            if block.isna().any():
                # skip blocks with NaN; we don't mask NaNs and they break runs
                continue

            # For safety, verify constancy within tolerance/equality
            if pd.api.types.is_numeric_dtype(block) and tolerance is not None:
                is_const = (block.max() - block.min()) <= tolerance
            else:
                is_const = block.nunique(dropna=False) == 1

            if not is_const:
                continue  # shouldn't happen often, but keep it robust

            # Compute run stats
            start_time = block.index[0]
            end_time = block.index[-1]
            n_rows = block.size
            duration = end_time - start_time  # timedelta

            meets = False
            if thresh_type == "count":
                meets = n_rows >= thresh_count  # type: ignore
            else:
                # For single-row runs, duration == 0; interpret as < threshold
                meets = duration >= thresh_delta

            if meets:
                # Mask the entire run
                mask_df.loc[block.index, col] = True

                # Stuck value for report (representative)
                val = block.iloc[0]

                report_rows.append(
                    {
                        "column": col,
                        "value": val,
                        "start": start_time,
                        "end": end_time,
                        "n_rows": n_rows,
                        "duration": duration,
                        "threshold_type": thresh_type,
                        "threshold_value": (
                            thresh_count if thresh_type == "count" else thresh_delta
                        ),
                    }
                )

    # Build outputs
    masked_df = df.copy()
    for col in cols:
        masked_df.loc[mask_df[col], col] = mask_value

    report = (
        pd.DataFrame(report_rows)
        .sort_values(["column", "start"])
        .reset_index(drop=True)
    )

    return (masked_df, report, mask_df) if return_mask else (masked_df, report)


__all__ = [
    "apply_physical_limits",
    "mask_stuck_values",
]