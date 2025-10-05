"""
Column cleanup and type conversion functions for the reformatter pipeline.

This module handles dropping unwanted columns, setting proper data types,
and filtering soil-related columns.
"""

import logging
from typing import List, Sequence, Union

import numpy as np
import pandas as pd


# Constants for soil column filtering
SOIL_SENSOR_SKIP_INDEX: int = 3
DEFAULT_SOIL_DROP_LIMIT: int = 4


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


def process_and_match_columns(
    df_full: pd.DataFrame,
    amflux: Union[pd.DataFrame, pd.Series]
) -> pd.DataFrame:
    """
    Cleans column names of df_full by removing '_1', '_2', '_3', and '_4' 
    suffixes, compares the cleaned names against an 'amflux' variable list, 
    and returns a DataFrame of the results, along with printing the unmatched columns.

    Args:
        df_full: The DataFrame whose columns need to be cleaned and matched.
        amflux: A DataFrame or Series that contains the 'Variable' column 
                or is the Series of variables to match against.

    Returns:
        A DataFrame containing the original columns, the cleaned columns, 
        and a boolean indicating if the cleaned column is in the amflux list.
    """
    
    # 1. Column Cleaning Logic
    clean_columns = list(df_full.columns)
    
    # Iteratively remove suffixes: '_1', '_2', '_3', '_4'
    # This loop is a condensed way to achieve the same result as the four 
    # separate list comprehensions in the original code.
    suffixes_to_remove = ['_1', '_2', '_3', '_4']
    
    for suffix in suffixes_to_remove:
        clean_columns = [item.split(suffix)[0] for item in clean_columns]

    clean_columns_series = pd.Series(clean_columns)
    
    # 2. Determine the AMERIFLUX Variable List for Matching
    # Handle both Series and DataFrame inputs for amflux
    if isinstance(amflux, pd.DataFrame) and 'Variable' in amflux.columns:
        amflux_variables = amflux['Variable']
    elif isinstance(amflux, pd.Series):
        amflux_variables = amflux
    else:
        raise ValueError("The 'amflux' argument must be a pandas Series or a DataFrame with a 'Variable' column.")

    # 3. Matching
    is_in_amflux = clean_columns_series.isin(amflux_variables)
    
    # 4. Create Results DataFrame
    results_df = pd.DataFrame({
        'all_columns': df_full.columns,
        'clean_columns': clean_columns,
        'is_in_amflux': is_in_amflux
    })

    # 5. Print and Return
    unmatched_df = results_df[results_df.is_in_amflux == False].sort_values('clean_columns')
    
    print('COLUMNS NOT IN AMERIFLUX VARIABLE LIST\n')
    print(unmatched_df)
    
    return results_df


__all__ = [
    "SOIL_SENSOR_SKIP_INDEX",
    "DEFAULT_SOIL_DROP_LIMIT",
    "drop_extra_soil_columns",
    "set_number_types",
    "drop_extras",
    "process_and_match_columns",
]