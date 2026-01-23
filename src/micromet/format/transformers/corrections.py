"""
Data correction functions for the reformatter pipeline.

This module contains variable-specific corrections and data value fixes,
including handling special values, unit conversions, and merging duplicate columns.
"""

import logging
import re

import numpy as np
import pandas as pd


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
    ssitc_bases = [
        "FC_SSITC_TEST",
        "LE_SSITC_TEST",
        "ET_SSITC_TEST",
        "H_SSITC_TEST",
        "TAU_SSITC_TEST",
    ]
    ssitc_columns = [
        col for col in df.columns 
        if any(col.startswith(base) for base in ssitc_bases)
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


__all__ = [
    "apply_fixes",
    "tau_fixer",
    "fix_swc_percent",
    "ssitc_scale",
    "scale_and_convert",
    "rating",
    "fill_na_drop_dups",
]