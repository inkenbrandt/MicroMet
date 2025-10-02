"""
This module contains data transformation functions that are used in the
reformatting pipeline.
"""

import pandas as pd
import numpy as np
import re
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union, Iterable

import logging


from micromet.utils import logger_check
import micromet.qaqc.variable_limits as variable_limits
import micromet.format.reformatter_vars as reformatter_vars


# Constants
MISSING_VALUE: int = -9999
SOIL_SENSOR_SKIP_INDEX: int = 3
DEFAULT_SOIL_DROP_LIMIT: int = 4

# SoilVUE Depth/orientation conversion tables
_DEPTH_MAP = {5: 1, 10: 2, 20: 3, 30: 4, 40: 5, 50: 6, 60: 7, 75: 8, 100: 9}
_ORIENT_MAP = {"N": 3, "S": 4}
_LEGACY_RE = re.compile(
    r"^(?P<prefix>(SWC|TS|EC|K|T))_(?P<depth>\d{1,3})cm_(?P<orient>[NS])_.*$",
    re.IGNORECASE,
)
_PREFIX_PATTERNS: Dict[re.Pattern[str], str] = {
    re.compile(r"^BulkEC_", re.IGNORECASE): "EC_",
    re.compile(r"^VWC_", re.IGNORECASE): "SWC_",
    re.compile(r"^Ka_", re.IGNORECASE): "K_",
}


def infer_datetime_col(df: pd.DataFrame, logger: logging.Logger) -> str | None:
    """
    Infer the name of the timestamp column in a DataFrame.

    This function searches for a timestamp column in the DataFrame by
    checking a list of common names (e.g., 'TIMESTAMP_END'). If a
    matching column is found, its name is returned. Otherwise, it logs
    a warning and returns the name of the first column.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to search for a timestamp column.
    logger : logging.Logger
        The logger to use for warning messages.

    Returns
    -------
    str or None
        The name of the timestamp column if found, otherwise the name of
        the first column.
    """
    datetime_col_options = ["TIMESTAMP_END", "TIMESTAMP_END_1"]
    datetime_col_options += [col.lower() for col in datetime_col_options]
    for cand in datetime_col_options:
        if cand in df.columns:
            return cand
    logger.warning("No TIMESTAMP column in dataframe")
    return df.iloc[:, 0].name  # type: ignore


def fix_timestamps(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Convert the timestamp column to datetime objects and handle missing values.

    This function identifies the timestamp column, converts it to datetime
    objects, and removes any rows where the timestamp could not be parsed.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with a timestamp column.
    logger : logging.Logger
        The logger for tracking progress and warnings.

    Returns
    -------
    pd.DataFrame
        The DataFrame with a 'DATETIME_END' column of datetime objects.
    """
    df = df.copy()
    if "TIMESTAMP" in df.columns:
        df = df.drop(["TIMESTAMP"], axis=1)
    ts_col = infer_datetime_col(df, logger)
    if ts_col is None:
        return df

    logger.debug(f"TS col {ts_col}")
    logger.debug(f"TIMESTAMP_END col {df[ts_col][0]}")
    ts_format = "%Y%m%d%H%M"
    df["DATETIME_END"] = pd.to_datetime(df[ts_col], format=ts_format, errors="coerce")
    logger.debug(f"Len of unfixed timestamps {len(df)}")
    df = df.dropna(subset=["DATETIME_END"])
    logger.debug(f"Len of fixed timestamps {len(df)}")
    return df


def resample_timestamps(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Resample a DataFrame to 30-minute intervals.

    This function resamples the DataFrame to a fixed 30-minute frequency
    based on the 'DATETIME_END' column. It also handles duplicate
    timestamps and interpolates missing data.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with a 'DATETIME_END' column.
    logger : logging.Logger
        The logger for tracking progress.

    Returns
    -------
    pd.DataFrame
        The resampled DataFrame with a 30-minute frequency index.
    """
    today = pd.Timestamp("today").floor("D")
    df = df[df["DATETIME_END"] <= today]
    df = (
        df.drop_duplicates(subset=["DATETIME_END"])
        .set_index("DATETIME_END")
        .sort_index()
    )
    df = df.resample("30min").first().interpolate(limit=1)
    logger.debug(f"Len of resampled timestamps {len(df)}")
    return df


def timestamp_reset(df, minutes=30):
    """
    Reset TIMESTAMP_START and TIMESTAMP_END columns based on the DataFrame index.

    This function generates new 'TIMESTAMP_START' and 'TIMESTAMP_END' columns
    based on the DataFrame's datetime index. The 'TIMESTAMP_START' is calculated
    by subtracting a specified number of minutes to the start time.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with a datetime index.
    minutes : int, optional
        The number of minutes to add to the start time to calculate the
        end time. Defaults to 30.

    Returns
    -------
    pd.DataFrame
        The DataFrame with updated 'TIMESTAMP_START' and 'TIMESTAMP_END' columns.
    """
    df["TIMESTAMP_END"] = df.index.strftime("%Y%m%d%H%M").astype(int)
    df["TIMESTAMP_START"] = (
        (df.index - pd.Timedelta(minutes=minutes)).strftime("%Y%m%d%H%M").astype(int)
    )
    return df


def rename_columns(
    df: pd.DataFrame, data_type: str, config: dict, logger: logging.Logger
) -> pd.DataFrame:
    """
    Rename DataFrame columns based on configuration and standardize their names.

    This function renames columns using a predefined mapping from the
    configuration, normalizes soil and temperature-related prefixes,
    and converts all column names to uppercase.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with columns to be renamed.
    data_type : str
        The type of data ('eddy' or 'met'), which determines which
        renaming map to use.
    config : dict
        The configuration dictionary containing the renaming maps.
    logger : logging.Logger
        The logger for tracking the renaming process.

    Returns
    -------
    pd.DataFrame
        The DataFrame with renamed and standardized column names.
    """
    mapping = config.get("renames_eddy" if data_type == "eddy" else "renames_met", {})
    logger.debug(f"Renaming columns from {df.columns} to {mapping}")
    df.columns = df.columns.str.strip()
    df = df.rename(columns=mapping)
    df = normalize_prefixes(df, logger)
    df = modernize_soil_legacy(df, logger)
    df.columns = df.columns.str.upper()
    logger.debug(f"Len of renamed cols {len(df)}")
    return df


def normalize_prefixes(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Normalize column name prefixes for soil and temperature measurements.

    This function standardizes column name prefixes by renaming them based
    on a set of predefined patterns. For example, it can change 'BulkEC_'
    to 'EC_'.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with columns to be normalized.
    logger : logging.Logger
        The logger for tracking the normalization process.

    Returns
    -------
    pd.DataFrame
        The DataFrame with normalized column name prefixes.
    """
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        for patt, repl in _PREFIX_PATTERNS.items():
            if patt.match(col):
                rename_map[col] = patt.sub(repl, col)
                break
        else:
            if re.match(r"^T_\d{1,3}cm_", col, flags=re.IGNORECASE):
                rename_map[col] = re.sub(r"^T_", "Ts_", col, flags=re.IGNORECASE)
    if rename_map:
        logger.debug("Prefix normalisation: %s", rename_map)
        df = df.rename(columns=rename_map)
    logger.debug(f"Len of normalized prefix cols {len(df)}")
    return df


def modernize_soil_legacy(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Update legacy soil sensor column names to a standardized format.

    This function identifies and renames legacy soil sensor columns to a
    modern, standardized format based on predefined mapping rules for
    depth and orientation.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with legacy soil sensor column names.
    logger : logging.Logger
        The logger for tracking the modernization process.

    Returns
    -------
    pd.DataFrame
        The DataFrame with updated soil sensor column names.
    """
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        m = _LEGACY_RE.match(col)
        if not m:
            continue
        prefix = m.group("prefix").upper()
        if prefix == "T":
            prefix = "TS"
        depth_cm = int(m.group("depth"))
        orient = m.group("orient").upper()
        depth_idx = _DEPTH_MAP.get(depth_cm)
        if depth_idx is None:
            continue
        replic = _ORIENT_MAP[orient]
        new_name = f"{prefix}_{replic}_{depth_idx}_1"
        rename_map[col] = new_name
    if rename_map:
        logger.info(f"Legacy soil columns modernised: {rename_map}")
        df = df.rename(columns=rename_map)
    return df


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
    keys = list(limits_dict.keys())
    if prefer_longest_key:
        keys.sort(key=len, reverse=True)

    col_map = {}
    for key in keys:
        matching_cols = [c for c in out.columns if str(c).startswith(key)]
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

    for col, info in col_map.items():
        key = info["key"]
        mn = info["Min"]
        mx = info["Max"]
        ser = pd.to_numeric(out[col], errors="coerce")
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
                "pct_flagged": (n_oor / len(ser) * 100.0) if len(ser) else 0.0,
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


def apply_fixes(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Apply a set of minor, variable-specific data corrections.

    This function serves as a pipeline for applying several small, targeted
    fixes to the data, such as correcting 'TAU' values, converting soil
    water content to percent, and scaling SSITC test values.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be fixed.
    logger : logging.Logger
        The logger for tracking the fixes being applied.

    Returns
    -------
    pd.DataFrame
        The DataFrame with all fixes applied.
    """
    df = tau_fixer(df)
    df = fix_swc_percent(df, logger)
    df = ssitc_scale(df, logger)
    return df


def tau_fixer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace zero values in the 'TAU' column with NaN.

    This function checks for zero values in the 'TAU' column and replaces
    them with NaN. This is often done to handle cases where zero represents
    a missing or invalid measurement.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with a 'TAU' column.

    Returns
    -------
    pd.DataFrame
        The DataFrame with zero values in 'TAU' replaced by NaN.
    """
    if "TAU" in df.columns and "U_STAR" in df.columns:
        bad_idx = df["TAU"] == 0
        df.loc[bad_idx, "TAU"] = np.nan
    return df


def fix_swc_percent(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Convert fractional soil water content (SWC) values to percentages.

    This function checks soil water content columns (those starting with
    'SWC_') and, if the values appear to be fractional (<= 1.5),
    multiplies them by 100 to convert them to percentages.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with SWC columns.
    logger : logging.Logger
        The logger for tracking the conversion process.

    Returns
    -------
    pd.DataFrame
        The DataFrame with SWC values converted to percentages where applicable.
    """
    df = df.copy()

    def _fix_one(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        m = s.max(skipna=True)
        if pd.notna(m) and m <= 1.5:
            s = s * 100.0
            logger.debug(f"Converted {s.name} from fraction to percent")
        return s

    for name in [c for c in df.columns if str(c).startswith("SWC_")]:
        obj = df.loc[:, name]
        if isinstance(obj, pd.DataFrame):
            for sub in obj.columns:
                df[sub] = _fix_one(df[sub])
        else:
            df[name] = _fix_one(obj)
    return df


def fill_na_drop_dups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge any number of duplicate columns with numeric suffixes (``.1``, ``.2``, ...),
    treating ``-9999`` as missing, and drop redundant duplicates.

    This function groups columns by their base name (the part before a trailing
    ``.<number>`` suffix). For each group, it merges values across the base column
    (if present) and all suffixed duplicates by preferring the first non-missing
    value at each row. During merging, the sentinel value ``-9999`` is treated as
    missing (converted to ``NaN``). After merging, remaining missing values are
    filled back with ``-9999`` and all duplicate suffixed columns are dropped,
    preserving the base column as the canonical result.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame that may contain duplicate columns named with numeric
        suffixes (e.g., ``"A.1"``, ``"A.2"``, ...). The unsuffixed base column
        (e.g., ``"A"``) is optional. Sentinel missing values are expected to be
        encoded as ``-9999``.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame where, for each base column, all suffixed duplicates have
        been merged into the base column and the duplicates removed. Any remaining
        missing values are filled with ``-9999``.

    Notes
    -----
    - Columns are grouped by the regex pattern ``r"^(?P<base>.+?)\\.(?P<idx>\\d+)$"``.
      Columns not matching this pattern are treated as base columns.
    - Merge precedence follows ascending numeric suffix order, with the base column
      (if present) considered first.
    - The input DataFrame is not modified in place; a copy is returned.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     "A":   [1, -9999, 3, -9999],
    ...     "A.1": [np.nan,  2,   -9999, 4],
    ...     "A.2": [-9999,   9,   np.nan, -9999],
    ...     "B.1": [10, -9999, np.nan, 13],   # no base 'B' column present
    ...     "B.3": [np.nan, 11, 12, -9999]
    ... })
    >>> fill_na_drop_dups(df)
         A     B
    0    1  10.0
    1    2  11.0
    2    3  12.0
    3    4  13.0
    """
    df_out = df.copy()
    pattern = re.compile(r"^(?P<base>.+?)\.(?P<idx>\d+)$")

    # Group columns by base name with numeric suffixes collected and sorted
    groups: dict[str, list[tuple[int, str]]] = {}
    for col in df_out.columns:
        m = pattern.match(col)
        if m:
            base = m.group("base")
            idx = int(m.group("idx"))
            groups.setdefault(base, []).append((idx, col))
        else:
            # Ensure singleton group for base-only column
            groups.setdefault(col, []).append((0, col))

    to_drop: list[str] = []

    for base, items in groups.items():
        # Sort by numeric suffix (base column, if present, has idx==0)
        items_sorted = sorted(items, key=lambda t: t[0])

        merged = None
        for _, col in items_sorted:
            s = df_out[col].replace(-9999, np.nan)
            merged = s if merged is None else merged.combine_first(s)

        # Re-impose sentinel for any remaining NaNs
        merged = merged.fillna(-9999)

        # Write back to base column (create if it didn't exist)
        df_out[base] = merged

        # Drop all duplicates except the base
        for _, col in items_sorted:
            if col != base:
                to_drop.append(col)

    if to_drop:
        # Deduplicate in case of overlap
        df_out = df_out.drop(columns=list(dict.fromkeys(to_drop)))

    return df_out


def ssitc_scale(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Scale SSITC (Signal Strength and Integrity Test) columns.

    This function checks specific SSITC columns and, if their values
    exceed a certain threshold (3), applies a scaling and rating
    transformation to them.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with SSITC columns.
    logger : logging.Logger
        The logger for tracking the scaling process.

    Returns
    -------
    pd.DataFrame
        The DataFrame with SSITC columns scaled where applicable.
    """
    ssitc_columns = [
        "FC_SSITC_TEST",
        "LE_SSITC_TEST",
        "ET_SSITC_TEST",
        "H_SSITC_TEST",
        "TAU_SSITC_TEST",
    ]
    for column in ssitc_columns:
        if column in df.columns:
            if df[column].max() > 3:
                df[column] = scale_and_convert(df[column])
                logger.debug(f"Scaled SSITC {column}")
    logger.debug(f"Scaled SSITC len: {len(df)}")
    return df


def scale_and_convert(column: pd.Series) -> pd.Series:
    """
    Apply a rating transformation and convert the column to float type.

    This function applies a 'rating' function to each element of the
    Series and then converts the entire Series to float.

    Parameters
    ----------
    column : pd.Series
        The input Series to be transformed.

    Returns
    -------
    pd.Series
        The transformed and converted Series.
    """
    column = column.apply(rating)
    return column


def rating(x):
    """
    Categorize a numeric value into a discrete rating level (0, 1, or 2).

    This function categorizes a numeric value into one of three levels:
    - 0 for values between 0 and 3.
    - 1 for values between 4 and 6.
    - 2 for all other values.

    Parameters
    ----------
    x : numeric or None
        The input value to be rated.

    Returns
    -------
    int
        The rating level (0, 1, or 2).
    """
    if x is None or np.isnan(x):
        x = 0
    else:
        if 0 <= x <= 3:
            x = 0
        elif 4 <= x <= 6:
            x = 1
        else:
            x = 2
    return x


def drop_extra_soil_columns(
    df: pd.DataFrame, config: dict, logger: logging.Logger
) -> pd.DataFrame:
    """
    Drop redundant or unused soil-related columns from the DataFrame.

    This function identifies and removes soil-related columns that are
    considered extra or redundant based on the provided configuration.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with soil-related columns.
    config : dict
        The configuration dictionary containing lists of columns to drop.
    logger : logging.Logger
        The logger for tracking the column dropping process.

    Returns
    -------
    pd.DataFrame
        The DataFrame with extra soil columns removed.
    """
    df = df.copy()
    math_soils: Sequence[str] = config.get("math_soils_v2", [])
    to_drop: List[str] = []

    for col in df.columns:
        parts = col.split("_")
        if len(parts) >= 3 and parts[0] in {"SWC", "TS", "EC", "K"}:
            try:
                if int(parts[1]) >= SOIL_SENSOR_SKIP_INDEX:
                    to_drop.append(col)
                    continue
            except ValueError:
                pass
        if col in math_soils[:-DEFAULT_SOIL_DROP_LIMIT]:
            to_drop.append(col)
            continue
        if parts[0] in {"VWC", "Ka"} or col.endswith("cm_N") or col.endswith("cm_S"):
            to_drop.append(col)

    if to_drop:
        logger.info("Dropping %d redundant soil columns", len(to_drop))
        df = df.drop(columns=to_drop, errors="ignore")
    return df


def make_unique(cols):
    """
    Make a list of column names unique by appending numeric suffixes to duplicates.

    This function takes a list of column names and ensures that all names
    are unique by appending a numeric suffix (e.g., '.1', '.2') to any
    duplicate names.

    Parameters
    ----------
    cols : list
        A list of column names.

    Returns
    -------
    list
        A list of unique column names.
    """
    seen = {}
    out = []
    for c in cols:
        c = str(c)
        if c in seen:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out


def make_unique_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that all column names in a DataFrame are unique.

    This function uses the `make_unique` helper function to append numeric
    suffixes to any duplicate column names, ensuring that every column
    has a unique identifier.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with unique column names.
    """
    df = df.copy()
    df.columns = make_unique(df.columns)
    return df


def set_number_types(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Convert columns in a DataFrame to the appropriate numeric types.

    This function iterates through the columns of a DataFrame and converts
    them to numeric types (integer or float) where appropriate. It handles
    special cases for certain columns and logs warnings for duplicate columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    logger : logging.Logger
        The logger for tracking the type conversion process.

    Returns
    -------
    pd.DataFrame
        The DataFrame with columns converted to numeric types.
    """
    logger.debug(f"Setting number types: {df.head(3)}")
    dupes = pd.Series(df.columns).value_counts()
    logger.debug(dupes[dupes > 1])

    for col in df.columns:
        logger.debug(f"Setting number types {col}")
        pos = np.where(df.columns == col)[0]
        if len(pos) == 1:
            if col in ["MO_LENGTH", "RECORD", "FILE_NO", "DATALOGGER_NO"]:
                df[col] = pd.to_numeric(df[col], downcast="integer", errors="coerce")
            elif col in ["DATETIME_END"]:
                df[col] = df[col]
            elif col in ["TIMESTAMP_START", "TIMESTAMP_END", "SSITC"]:
                df[col] = pd.to_numeric(df[col], downcast="integer", errors="coerce")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            logger.warning(f"Column {col} appears multiple times in DataFrame")
            for p in pos:
                s = df.iloc[:, p]
                if col in [
                    "MO_LENGTH",
                    "RECORD",
                    "FILE_NO",
                    "DATALOGGER_NO",
                    "TIMESTAMP_START",
                    "TIMESTAMP_END",
                    "SSITC",
                ]:
                    df.iloc[:, p] = pd.to_numeric(
                        s, downcast="integer", errors="coerce"
                    )
                elif col == "DATETIME_END":
                    continue
                else:
                    df.iloc[:, p] = pd.to_numeric(s, errors="coerce")
    logger.debug(f"Set number types: {len(df)}")
    return df


def drop_extras(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Drop extra or unwanted columns from the DataFrame based on configuration.

    This function removes columns from the DataFrame that are listed in the
    'drop_cols' section of the configuration dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    config : dict
        The configuration dictionary containing the list of columns to drop.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the specified columns removed.
    """
    return df.drop(columns=config.get("drop_cols", []), errors="ignore")


def col_order(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Reorder DataFrame columns to place priority columns at the beginning.

    This function moves specified columns ('TIMESTAMP_END', 'TIMESTAMP_START')
    to the front of the DataFrame for better readability and consistency.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    logger : logging.Logger
        The logger for tracking the reordering process.

    Returns
    -------
    pd.DataFrame
        The DataFrame with columns reordered.
    """
    first_cols = ["TIMESTAMP_END", "TIMESTAMP_START"]
    for col in first_cols:
        if col in df.columns:
            ncol = df.pop(col)
            df.insert(0, col, ncol)
    logger.debug(f"Column Order: {df.columns}")
    return df
