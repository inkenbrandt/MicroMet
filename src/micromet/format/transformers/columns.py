"""
Column naming and organization functions for the reformatter pipeline.

This module handles column renaming, prefix normalization, legacy format
updates, and column ordering operations.
"""

import logging
import re
from typing import Dict

import pandas as pd

def create_suffix_map(df, col_list, suffix):
    """
    Filters a list of columns based on what's actually in the DataFrame,
    then creates a dictionary for renaming with a user-provided suffix.
    """
    # 1. Only include columns that actually exist in your current dataframe
    existing_cols = [col for col in col_list if col in df.columns]
    
    # 2. Create the renaming dictionary
    # Example: 'CO2_SIGMA' -> 'CO2_SIGMA_1_1_1'
    rename_dict = {col: f"{col}{suffix}" for col in existing_cols}
    
    return rename_dict


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


__all__ = [
    "rename_columns",
    "normalize_prefixes",
    "modernize_soil_legacy",
    "make_unique",
    "make_unique_cols",
    "col_order",
]