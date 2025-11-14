import pandas as pd
import numpy as np
from typing import Union

def prep_parquet(station, df, install_date):
    not_subset = ['TIMESTAMP_START', 'TIMESTAMP_END',
               'FC_SSITC_TEST', 'LE_SSITC_TEST', 
               'ET_SSITC_TEST', 'H_SSITC_TEST','TAU_SSITC_TEST' ]
    
    df = df.loc[station]
    df = df.replace(-9999, np.nan)
    filtered_cols = [col for col in df.columns if col not in not_subset]
    df = df.dropna(axis=1, how='all')
    df = df.sort_index()

    return(df)

def set_range_to_nan(
    df: pd.DataFrame,
    column_name: str,
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
    index_is_datetime: bool = True,
    date_col: str = None
) -> pd.DataFrame:
    """
    Sets values in a specified column to np.nan within a given datetime range.

    Args:
        df: The pandas DataFrame.
        column_name: The name of the column whose values will be set to NaN.
        start_date: The start of the datetime range (inclusive). Can be a string
                    (e.g., '2025-10-01') or a pd.Timestamp.
        end_date: The end of the datetime range (inclusive). Can be a string
                  (e.g., '2025-10-02 12:00:00') or a pd.Timestamp.
        index_is_datetime: If True (default), the function uses the DataFrame's index
                           for filtering.
        date_col: If index_is_datetime is False, provide the name of the column
                  containing the datetime information for filtering.

    Returns:
        The modified pandas DataFrame.
    """
    df_copy = df.copy() # Work on a copy to prevent SettingWithCopyWarning

    if index_is_datetime:
        # 1. Create the boolean mask using the DatetimeIndex
        # The .loc accessor works well with slicing on a DatetimeIndex
        date_mask = (df_copy.index >= start_date) & (df_copy.index <= end_date)
    else:
        if date_col is None:
            raise ValueError("Must provide 'date_col' if 'index_is_datetime' is False.")

        # Ensure the date column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
             df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')

        # 1. Create the boolean mask using the specified date column
        date_mask = (df_copy[date_col] >= start_date) & (df_copy[date_col] <= end_date)

    # 2. Apply the mask and set the target column values to np.nan
    # Use .loc[rows, columns] for reliable assignment
    df_copy.loc[date_mask, column_name] = np.nan

    return df_copy

# find optimal shift between two sets of data, using your desired frequency and 
# min and max lags to inspect
# created to align met data with very bad timestamps (e.g., off by months)
# did not return a perfect lag for me, but got me close
def find_optimal_shift(
    df1, 
    df2, 
    value_col1, 
    value_col2, 
    freq='h',  # Coarse detection unit (e.g., 'D' for days)
    min_lag_units=100, # Minimum lag magnitude to check (e.g., 30 days)
    max_lag_units=500, # Maximum lag magnitude to check (e.g., 180 days)
    dropna_threshold=0.75
):
    """
    Identifies the optimal time shift (lag) required to align two datetime-indexed 
    pandas DataFrames by maximizing the cross-correlation between two specified columns.

    This version searches for shifts between +/- min_lag_units and +/- max_lag_units.

    Interpreting the Lag:
    - Positive Lag: df2 is behind df1 (df2 needs to be shifted FORWARD).
    - Negative Lag: df2 is ahead of df1 (df2 needs to be shifted BACKWARD).

    Parameters:
    - df1, df2 (pd.DataFrame): DataFrames with a datetime index.
    - value_col1 (str): Name of the column in df1 to compare.
    - value_col2 (str): Name of the column in df2 to compare.
    - freq (str): Resampling frequency (e.g., 'D' for daily, 'H' for hourly). 
      This determines the unit of the returned lag.
    - min_lag_units (int): The absolute minimum lag magnitude (in units of 'freq') to test.
    - max_lag_units (int): The absolute maximum lag magnitude (in units of 'freq') to test.
    - dropna_threshold (float): Minimum required fraction of non-NaN values 
      after alignment for the data to be processed (e.g., 0.75 = 75% non-NaN data).

    Returns:
    - tuple: (best_lag, max_correlation)
    """
    
    # 1. Resample and Select Series
    s1 = df1[value_col1].resample(freq).mean()
    s2 = df2[value_col2].resample(freq).mean()

    # 2. Align Data across the Full Time Span (using outer join)
    # Outer join creates a combined index spanning both datasets, inserting NaNs.
    s1_aligned, s2_aligned = s1.align(s2, join='outer')
    
    # Use the full series (containing NaNs) for correlation.
    s1_full = s1_aligned
    s2_full = s2_aligned

    # 3. Validation Check
    if len(s1_full) < max_lag_units:
        print("Warning: Combined data span is too short for the specified max_lag_units.")
        return 0, np.nan

    # 4. Calculate Cross-Correlation for Lags
    
    # 4a. Negative lags (df2 is ahead of df1): Check from -max down to -min
    negative_lags = np.arange(-max_lag_units, -min_lag_units) 
    
    # 4b. Positive lags (df2 is behind df1): Check from +min up to +max
    positive_lags = np.arange(min_lag_units, max_lag_units + 1)
    
    # Combine the two directional searches
    lags = np.concatenate([negative_lags, positive_lags])
    
    # Calculate correlation for each lag
    # pd.Series.corr() handles NaNs resulting from the shift operation.
    correlations = [s1_full.corr(s2_full.shift(lag)) for lag in lags]

    if all(pd.isna(correlations)):
        print("Warning: All correlations resulted in NaN. Data may be constant or invalid.")
        return 0, np.nan

    # 5. Find the Best Lag
    max_correlation = np.nanmax(correlations)
    best_lag_index = np.nanargmax(correlations)
    best_lag = lags[best_lag_index]

    return best_lag, max_correlation

# apply lag to a dataframe based on the find_optimal_shift function
def apply_lag_shift(df, detected_lag, freq_unit):
    """
    Applies the inverse of the detected lag to a DataFrame's datetime index 
    to align it with the reference dataset.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be shifted (e.g., df1 from find_optimal_shift function).
    - detected_lag (int): The lag detected by find_optimal_shift (e.g., -60).
    - freq_unit (str): The frequency unit used for the lag (e.g., 'D', 'H', '30T').

    Returns:
    - pd.DataFrame: The DataFrame with the adjusted datetime index.
    """
    # The shift required for DF1 to match DF2 is the negative (inverse) of 
    # the detected lag (which shifts DF2 to match DF1).
    required_shift_units = -detected_lag
    
    # Create the time offset
    time_offset = pd.Timedelta(required_shift_units, unit=freq_unit) 

    # Apply the offset to the index
    df_aligned = df.set_index(df.index + time_offset)
    
    return df_aligned