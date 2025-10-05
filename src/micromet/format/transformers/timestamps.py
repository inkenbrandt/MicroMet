"""
Timestamp transformation functions for the reformatter pipeline.

This module handles all datetime-related operations including timestamp
detection, conversion, resampling, and formatting.
"""

import logging
import pandas as pd


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


def timestamp_reset(df: pd.DataFrame, minutes: int = 30) -> pd.DataFrame:
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


__all__ = [
    "infer_datetime_col",
    "fix_timestamps",
    "resample_timestamps",
    "timestamp_reset",
]