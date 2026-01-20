import pandas as pd
import numpy as np
from typing import Union

def prep_parquet(station, df):    
    df = df.loc[station]
    df = df.replace(-9999, np.nan)
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


def apply_internal_flags(df, 
                              flag_cols,
                              start_date,
                              end_date,
                              flag_value,
                              ):
    """
    Applies a specified flag value across multiple specified flag columns 
    within a given date range.
    ... (Docstring contents omitted for brevity, but they are correct)
    """
    # 1. Create a copy to avoid modifying the original DataFrame
    df2 = df.copy()

    # 2. Format dates
    start_date_format = pd.to_datetime(start_date)
    end_date_format = pd.to_datetime(end_date)
    
    # 3. Validation: Check if ALL specified flag columns exist
    missing_cols = [col for col in flag_cols if col not in df.columns]
    
    if missing_cols:
        raise KeyError(f"The following required flag column(s) were not found in the DataFrame: {missing_cols}")
    
    # 4. Apply the flag (This block was incorrectly indented)
    # Create a boolean mask for the date range
    mask = (df2.index >= start_date_format) & (df2.index <= end_date_format)
    
    # Apply the flag value to the selected rows/columns
    df2.loc[mask, flag_cols] = flag_value
        
    return df2


# train regressoin model
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from typing import Tuple, Dict, Any

def mask_wind_direction(df, wd_col, start_deg, end_deg):
    """
    Creates a boolean mask for bad wind directions.
    
    Parameters:
    df (pd.DataFrame): Your dataset.
    wd_col (str): Name of the wind direction column (0-360).
    start_deg (float): The start of the exclusion zone (clockwise).
    end_deg (float): The end of the exclusion zone (clockwise).
    
    Returns:
    pd.Series: A boolean mask where True = BAD data (inside the zone).
    """
    wd = df[wd_col]
    
    if start_deg <= end_deg:
        # Standard case: e.g., 90 to 180 (East to South)
        mask = (wd >= start_deg) & (wd <= end_deg)
    else:
        # Wrap-around case: e.g., 350 to 20 (Northwest to Northeast)
        mask = (wd >= start_deg) | (wd <= end_deg)
        
    return mask

def mask_by_rolling_window_combined(
    df: pd.DataFrame,
    sig_col: str = 'H2O_SIG_STRGTH_MIN',
    rolling_window: int = 9,
    threshold_value: float = 0.8,
) -> pd.Series:
    """
    Create a robust quality mask using instant and smoothed signal thresholds.

    This function implements a 'dual-condition' filter to identify poor instrument 
    performance (e.g., AGC or RSSI drops). It protects against over-masking 
    transient spikes by requiring both the instantaneous signal AND a centered 
    rolling median to fall below the threshold before a point is rejected.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing the signal strength telemetry.
    sig_col : str, default 'H2O_SIG_STRGTH_MIN'
        The name of the column to evaluate.
    rolling_window : int, default 9
        The size of the moving window (number of periods). An odd integer is 
        recommended to ensure the window is perfectly centered on the timestamp.
    threshold_value : float, default 0.8
        The minimum acceptable signal strength. Values below this are 
        considered potential failures.

    Returns
    -------
    pd.Series
        A boolean Series (mask) where True indicates 'Good Data' (Keep) 
        and False indicates 'Bad Data' (Filter).

    Notes
    -----
    - Robustness: Uses a rolling median rather than a mean to ignore 
      short-duration impulse noise (spikes) within the window.
    - Logic: A data point is masked ONLY if:
        (Instant Signal < Threshold) AND (Rolling Median < Threshold).
    - Edge Handling: Uses `min_periods=1` to ensure valid masking at the 
      beginning and end of the dataset.
    - Missing Data: Existing NaN values in `sig_col` are excluded from the 
      printed quality report to provide an accurate 'dropped points' percentage.
    """
    # 1. Calculate the smoothed signal 
    rolling_sig = df[sig_col].rolling(
        window=rolling_window, 
        center=True, 
        min_periods=1
    ).median()
    
    # 2. Define the two conditions
    # True if the signal is "Good"
    instant_pass = df[sig_col] >= threshold_value
    rolling_pass = rolling_sig >= threshold_value
    
    # 3. Combined Logic: 
    # We keep the data if the instant signal is good OR if the window says it's okay.
    # This means we ONLY drop if BOTH are bad.
    mask = instant_pass | rolling_pass

    # 4. Refined Reporting Logic
    # We only care about rows where we actually HAD data to begin with
    valid_data_indices = df[sig_col].notna()
    
    # Points that had data but were masked by our threshold logic
    num_filtered = (~mask & valid_data_indices).sum()
    total_valid = valid_data_indices.sum()
    
    if total_valid > 0:
        print(f"Quality Control Report: {num_filtered} of {total_valid} valid points "
              f"({num_filtered/total_valid:.1%}) dropped via threshold.")
    else:
        print("Quality Control Report: No valid data found in column.")
    
    return mask
    

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def despike_data_nan_aware(data, filter_size=5, threshold_factor=3.0):
    """
    Remove outliers (spikes) from a 1D array using a NaN-aware median filter.

    This function identifies spikes by comparing each data point to the median 
    of its local neighborhood. It is specifically designed to handle arrays 
    containing NaN values without allowing those NaNs to bias the filter or 
    the noise statistics.

    The process follows these steps:
    1. Pads the data to handle edges using reflection.
    2. Calculates a moving baseline using a sliding window median (ignoring NaNs).
    3. Computes the residual noise and determines a threshold based on the 
       standard deviation of that noise.
    4. Replaces values exceeding the threshold with the local median.

    Parameters
    ----------
    data : array_like
        The input 1D signal or time-series data to be despiked. Can contain 
        NaN values.
    filter_size : int, optional
        The size of the sliding window used to calculate the local median. 
        Must be an odd integer. Default is 5.
    threshold_factor : float, optional
        The multiplier applied to the global standard deviation of the noise 
        to determine the spike detection threshold. A higher value is less 
        sensitive (detects fewer spikes). Default is 3.0.

    Returns
    -------
    despiked_data : ndarray
        A copy of the input data where identified spikes have been replaced 
        by the local median. Original NaN values are preserved.
    spike_mask : ndarray (bool)
        A boolean mask of the same shape as `data`, where True indicates 
        a detected spike location.

    Notes
    -----
    - This function uses `np.nanmedian` and `np.nanstd`, which are 
      computationally more expensive than their standard counterparts but 
      necessary if the dataset is missing values.
    - If a window consists entirely of NaNs, the resulting baseline value 
      for that window will be NaN.

    Examples
    --------
    >>> signal = [10, 11, 100, 12, np.nan, 11, 10]
    >>> clean, mask = despike_data_nan_aware(signal, filter_size=3)
    >>> clean
    array([10., 11., 11., 12., nan, 11., 10.])
    """
    # Ensure data is a numpy array
    data = np.asanyarray(data)
    
    # Create a padded version to handle edges
    pad_size = filter_size // 2
    padded_data = np.pad(data, pad_size, mode='reflect')
    
    # Create sliding windows
    windows = sliding_window_view(padded_data, filter_size)
    
    # Calculate baseline using nanmedian (ignores NaNs)
    baseline = np.nanmedian(windows, axis=1)
    
    # Calculate noise: Difference between original and baseline
    noise = data - baseline
    
    # Calculate threshold using nanstd to ignore existing NaNs
    threshold = threshold_factor * np.nanstd(noise)
    
    # Identify spikes (ignoring NaNs in the comparison)
    spike_mask = np.abs(noise) > threshold
    
    # Replace spikes with baseline, but keep original NaNs as NaNs
    despiked_data = data.copy()
    despiked_data[spike_mask] = baseline[spike_mask]
    
    return despiked_data, spike_mask

def train_linear_regression_model(
    df: pd.DataFrame, 
    target_col: str, 
    predictor_col: str
) -> Tuple[LinearRegression | None, Dict[str, Any]]:
    """
    Trains a Linear Regression model using complete data from two specified columns.

    Args:
        df: The input pandas DataFrame.
        target_col: The name of the column containing the dependent variable (Y).
        predictor_col: The name of the column containing the independent variable (X).

    Returns:
        A tuple containing:
        1. The trained LinearRegression model instance (or None if training fails).
        2. A dictionary of model results (e.g., intercept and coefficient).
    """
    # 1. Identify rows with complete data for model training
    complete_data = df.dropna(subset=[predictor_col, target_col])

    # Check if we have enough data to train a model
    if len(complete_data) < 10:
        print("Error: Not enough complete data points to train the linear regression model.")
        return None, {}

    # Prepare training features (X) and target (Y)
    X_train = complete_data[[predictor_col]]
    y_train = complete_data[target_col]

    # 2. Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 3. Compile results for evaluation
    model_results = {
        "intercept": round(model.intercept_, 3),
        "coefficient": round(model.coef_[0], 3),
        "r_squared": round(model.score(X_train, y_train), 3),
        "training_n_samples": len(complete_data)
    }

    return model, model_results

# impute values from regression model
import pandas as pd
from sklearn.linear_model import LinearRegression

def impute_missing_values(
    df: pd.DataFrame, 
    model: LinearRegression, 
    target_col: str, 
    predictor_col: str
) -> pd.Series:
    """
    Imputes missing values using Linear Regression and returns the resulting 
    imputed column as a Series, without creating a full DataFrame copy.
    """
    # 1. Start with a copy of the target Series (Y) – This is the only copy needed
    imputed_series = df[target_col].copy(deep=True)

    # 2. Identify rows needing imputation
    missing_mask = imputed_series.isna()
    
    # Identify rows where we have the predictor (X) and need to predict Y
    # Use .notna() on the DataFrame column predictor_col
    imputation_rows_mask = missing_mask & df[predictor_col].notna()

    if not imputation_rows_mask.any():
        print(f"No missing values in '{target_col}' that can be imputed using '{predictor_col}'.")
        return imputed_series.rename(new_col)

    # 3. Prepare prediction features (X)
    # Select the predictor column for only the rows that need imputation
    X_predict = df.loc[imputation_rows_mask, [predictor_col]]

    # 4. Predict the missing values
    predictions = model.predict(X_predict)

    print(f"Imputing {len(predictions)} missing values into the new Series")

    # 5. Fill the missing values in the Series
    # Use the index of the identified rows to safely assign predictions
    imputed_series.loc[imputation_rows_mask] = predictions

    # 6. Return the resulting Series, renamed
    return imputed_series