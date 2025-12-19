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

def mask_by_rolling_window(
    df: pd.DataFrame,
    sig_col: str = 'H2O_SIG_STRGTH_MIN',
    rolling_window: int = 8,
    threshold_value: float = 0.8,
    threshold_direction: str = 'gt'
) -> pd.Series:
    """
    Create a boolean mask based on a rolling average of a signal column.

    This function is commonly used in flux processing (like Eddy Covariance) 
    to filter out data periods where the instrument signal strength (e.g., AGC 
    or RSSI) drops below a quality threshold, smoothed over a specific window 
    to prevent over-flagging transient spikes.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the signal strength column.
    sig_col : str, default 'H2O_SIG_STRGTH_MIN'
        The column name to perform the rolling average and thresholding on.
    rolling_window : int, default 8
        The size of the moving window (number of periods) for the mean calculation.
    threshold_value : float, default 0.8
        The numerical value used to determine the mask boundary.
    threshold_direction : {'gt', 'lt'}, default 'gt'
        The comparison operator for the threshold:
        - 'gt': Mask is True where rolling average is Greater Than threshold.
        - 'lt': Mask is True where rolling average is Less Than threshold.

    Returns
    -------
    pd.Series
        A boolean Series (mask) where True indicates the data passed the 
        threshold criteria and False indicates it should be filtered out.

    Notes
    -----
    The rolling window is centered (`center=True`), meaning the average for 
    any given point is calculated using both preceding and following data.
    """
    df2 = df.copy()
    
    # Calculate smoothed signal
    df2['Signal_Rolling'] = df2[sig_col].rolling(
        window=rolling_window, 
        center=True
    ).mean()
    
    # Determine masking logic
    if threshold_direction == 'gt':
        mask = df2['Signal_Rolling'] > threshold_value
    elif threshold_direction == 'lt':
        mask = df2['Signal_Rolling'] < threshold_value
    else:
        raise ValueError("threshold_direction must be either 'lt' or 'gt'")

    # Report masking statistics
    num_masked = len(mask[mask == False])
    print(f"Masking Report: {num_masked} of {len(df)} points ({num_masked/len(df):.1%}) outside threshold.")
    
    return mask


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