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

def resample_timestamps_interval(df: pd.DataFrame, logger: logging.Logger,
                                 data_type: str, stationid: str
                                 ) -> pd.DataFrame:
    """
    Resample a DataFrame to 30-minute or 60-minute intervals.

    This function resamples the DataFrame to a frequency
    based on the 'DATETIME_END' column. The frequency is determined based on 
    comparison between the date range of the data and the interval change date
    specified in interval_updates.py.
    
    This function also handles duplicate timestamps.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with a 'DATETIME_END' column.
    logger : logging.Logger
        The logger for tracking progress.
    datatype : datatype

    Returns
    -------
    pd.DataFrame
        The resampled DataFrame with the resampled index and new field with datainterval
    """
    interval_update_dict = interval_updates.interval_update_dict

    today = pd.Timestamp("today").floor("D")
    df = df[df["DATETIME_END"] <= today]
    df = (
        df.drop_duplicates(subset=["DATETIME_END"])
        .set_index("DATETIME_END")
        .sort_index()
    )
    
    if key in interval_update_dict.keys():
        if (data_type=="eddy") & (interval_update_dict[key][0]!=None):
            change_date = pd.to_datetime(interval_update_dict[key][0])
        elif (data_type=="met") & (interval_update_dict[key][1]!=None):
            change_date = pd.to_datetime(interval_update_dict[key][1])
        else:
            logger.debug(f"Sampling interval not changed for {key} {data_type} data")
            change_date = None
    if change_date:
        if (df.index.max()<change_date):
            logger.debug("Resampling all data to 30 minutes")
            df = df.resample("30min").agg('first')
            df['datainterval'] = 30
        elif (df.index.min()>change_date):
            logger.debug("Resampling all data to 60 minutes")
            df = df.resample("60min").agg('first')
            df['datainterval'] = 60
        elif (df.index.max()>change_date) & (df.index.min()<change_date):
            # just a check on the data interval switch date
            time_diff_td = df.index.to_series().diff()
            df['timediff'] = time_diff_td.dt.total_seconds() / 60
            check60_date = (change_date + pd.Timedelta(hours=1)).floor('h')
            check30_date = (change_date.floor('h'))
            check30 = df.loc[df.index==check30_date, 'timediff'].iloc[0]
            check60 = df.loc[df.index==check60_date, 'timediff'].iloc[0]
            if (check30!=30) | (check60 != 60):
                logger.warning("Date when sampling interval changed may be incorrect based on index differences")

            logger.debug(f"Resampling data to 30 minutes before {change_date} and 60 minutes after")
            df60 = df[df.index>change_date]
            df60 = df60.resample("60min").agg('first')
            df60['datainterval'] = 60
            df30 = df[df.index<=change_date]
            df30 = df30.resample("30min").agg('first')
            df30['datainterval'] = 30
            df = pd.concat([df30, df60])
            df.drop(columns=['timediff'], inplace=True)
    else:
        df = df.resample("30min").agg('first')
        df['datainterval'] = 30
        logger.debug("Resampling all data to 30 minutes")

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
    "resample_timestamps_interval",
    "timestamp_reset",
]