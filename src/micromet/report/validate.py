


import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
import plotly.graph_objects as go


# validate test variables to equal 0, 1, 2
def validate_flags(df: pd.DataFrame, 
                   flag_columns: List[str] = ['FC_SSITC_TEST', 'LE_SSITC_TEST', 'ET_SSITC_TEST', 'H_SSITC_TEST',
       'TAU_SSITC_TEST'], 
                   allowed_values: List[int] = [0, 1, 2]) -> Dict[str, List]:
    """
    Checks specified DataFrame columns for values outside of the allowed set,
    including checking for NaN (missing) values.

    This is typically used for quality control (QC) flag columns which should 
    only contain specific integer values (like 0, 1, 2).

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the flag columns.
    flag_columns : List[str]
        A list of column names to check.
    allowed_values : List[int]
        The list of values considered valid (defaults to [0, 1, 2]).

    Returns
    -------
    Dict[str, List]
        A dictionary where keys are the column names that failed validation,
        and values are a list of the unique, invalid values found in that column,
        including the string "NaN" if missing values are present.
    """
    
    # Convert allowed_values to a set for faster lookup
    allowed_set = set(allowed_values)
    
    # Dictionary to store results for columns that fail the validation
    invalid_columns = {}

    print(f"--- Starting Validation ---")
    print(f"Checking columns: {flag_columns}")
    print(f"Allowed values: {allowed_set}")

    for col in flag_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame.")
            continue

        # 1. Find all unique values in the series, including NaNs
        unique_values = df[col].unique()

        # 2. Separate NaNs, valid flags, and invalid numeric flags
        invalid_numeric_flags = []
        nan_present = False
        
        for val in unique_values:
            if pd.isna(val):
                nan_present = True
            elif val not in allowed_set:
                invalid_numeric_flags.append(val)

        # 3. Construct the final report list (numeric values first, then "NaN" indicator)
        final_report_list = sorted(invalid_numeric_flags)
        
        if nan_present:
            final_report_list.append("NaN")
            
        if final_report_list:
            invalid_columns[col] = final_report_list
            print(f"FAIL: Column '{col}' contains unexpected values: {final_report_list}")
        else:
            print(f"PASS: Column '{col}' contains only valid values.")

    print(f"--- Validation Complete ---")
    return invalid_columns

# Compare field names between dataframe and amerriflux variable names
def compare_names_to_ameriflux(
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

# compare alignment between two files (one raw that is read in and one from micromet)
def compare_to_raw(raw_file_path, micromet_df, test_var = 'NETRAD', threshold=0.1):
    '''Compares a specific variable between a raw data file and a micromet DataFrame.

    The function reads a 'raw' DAT or CSV file from the provided path, merges it with the 
    'micromet' DataFrame based on TIMESTAMP to DATETIME_END fields, and calculates the absolute
    difference for a specified variable (`test_var`) between the two sources. It 
    returns only the rows where this absolute difference is greater than the given 
    `threshold`.

    Args:
        raw_file_path (str): The file path to the raw data CSV file. This file is 
                             assumed to have a specific format (header on row 1, with 
                             rows 2 and 3 skipped).
        micromet_df (pd.DataFrame): DataFrame containing the micrometeorological data.
        test_var (str, optional): The variable to compare (e.g., 'LE' for Latent Energy). 
                                  Defaults to 'LE'. The function assumes the raw 
                                  column is named '{test_var}_1_1_1' and the micromet 
                                  column is named '{test_var}'.
        threshold (float, optional): The absolute difference threshold. Rows where 
                                     |raw_value - micromet_value| > threshold are returned. 
                                     Defaults to 0.1.

    Returns:
        pd.DataFrame: A DataFrame containing the 'DATETIME_END' and the values of the 
                      `test_var` from both sources ('{test_var}_1_1_1' and '{test_var}') 
                      for all rows where the absolute difference exceeds the `threshold`.
    '''
    raw = pd.read_csv(raw_file_path, skiprows=[2,3], header=1, low_memory=False)
    raw['TIMESTAMP'] = pd.to_datetime(raw['TIMESTAMP'])

    combo = raw.merge(micromet_df, how='inner', left_on='TIMESTAMP', right_on='DATETIME_END',
                      suffixes=['_raw', '_micromet'])

    le_diff = combo[f'{test_var}_1_1_1'] -combo[f'{test_var}'].astype('float')
    value_differences = combo.loc[(le_diff.abs()>threshold), ['DATETIME_END',f'{test_var}_1_1_1', f'{test_var}']]
    return(value_differences)

# check for consistency between DATETIME_END and TIMESTAMP_END fields
def validate_timestamp_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks for consistency between a standardized datetime column (DATETIME_END)
    and a string/integer timestamp column (TIMESTAMP_START) formatted as YYYYMMDDHHMM.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the columns to check.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the rows where the DATETIME_END and
        the converted TIMESTAMP_END columns do not match, along with both columns
        for inspection. Returns an empty DataFrame if all rows match.
    """
    df = df.copy()
    
    REQUIRED_COLS = ['DATETIME_END', 'TIMESTAMP_END']

    if not all(col in df.columns for col in REQUIRED_COLS):
        print(f"Error: DataFrame must contain both {REQUIRED_COLS} columns.")
        return pd.DataFrame()

    print("\n--- Starting Timestamp Consistency Validation ---")

    # Ensure DATETIME_END is properly parsed datetime object
    df['DATETIME_END_DT'] = pd.to_datetime(df['DATETIME_END'], errors='coerce')

    # Convert TIMESTAMP_END (e.g., 202406241430) to a datetime object
    # We convert to string first to handle both int and string inputs
    df['TIMESTAMP_END_DT'] = pd.to_datetime(
        df['TIMESTAMP_END'].astype(str), 
        format='%Y%m%d%H%M', 
        errors='coerce'
    )

    # Compare the two generated datetime columns
    # We use .notna() to ignore rows where either conversion failed (coerced to NaT)
    mismatch_mask = (df['DATETIME_END_DT'] != df['TIMESTAMP_END_DT']) & \
                    (df['DATETIME_END_DT'].notna()) & \
                    (df['TIMESTAMP_END_DT'].notna())

    # Filter for mismatches and report
    mismatch_report = df.loc[mismatch_mask, REQUIRED_COLS + ['DATETIME_END_DT', 'TIMESTAMP_END_DT']].copy()
    
    if mismatch_report.empty:
        print("PASS: DATETIME_END and TIMESTAMP_END are perfectly consistent (where both are valid).")
    else:
        print(f"FAIL: Found {len(mismatch_report)} inconsistent rows.")
        
    print("--- Timestamp Consistency Validation Complete ---")


    
    return mismatch_report

# Find extended periods of time where sensor read 0 values (used with precip data)
def find_zero_chunks(
    df: pd.DataFrame,
    var_name: str,
    days_threshold: int,
    aggregation_method: str = 'sum', # New parameter to determine daily aggregation
    tolerance: float = 1e-6
) -> pd.DataFrame:
    """
    Identifies continuous chunks of time where a variable is effectively zero or NaN,
    treating NaNs as part of the zero gap.

    The function first resamples the high-frequency data to daily ('D') frequency
    using the specified aggregation method before checking for long zero periods.

    Args:
        df: The pandas DataFrame with a DatetimeIndex (any frequency).
        var_name: The name of the column to check for zero values.
        days_threshold: The minimum number of consecutive days required to be
                        identified as a "long zero chunk".
        aggregation_method: The method used to aggregate high-frequency data to daily.
                            Options: 'sum' (default) or 'max'.
        tolerance: A small value used to check if a float is close to zero.

    Returns:
        A DataFrame listing the 'Start Day', 'End Day', and 'Duration (Days)'
        for each identified long zero chunk.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Error: DataFrame index must be a DatetimeIndex.")
        return pd.DataFrame()

    if aggregation_method not in ['sum', 'max']:
        print(f"Error: Invalid aggregation_method '{aggregation_method}'. Must be 'sum' or 'max'.")
        return pd.DataFrame()

    # 1. Resample the data to daily frequency ('D') using the specified method
    # The resulting index represents the start of each day.
    try:
        df_daily = df[var_name].resample('D').agg(aggregation_method)
    except Exception as e:
        print(f"Error during resampling with method '{aggregation_method}': {e}")
        return pd.DataFrame()

    # The period threshold is now simply the number of days, as we are working with daily periods.
    # period_threshold is kept for clarity but is equal to days_threshold
    period_threshold = days_threshold

    # 2. Create a boolean mask: True if daily aggregated value is near zero OR is NaN
    is_zero_or_na = (df_daily.abs() < tolerance) | (df_daily.isna())

    # 3. Calculate consecutive count of 'is_zero_or_na' being True
    # (~is_zero_or_na).cumsum() creates a group ID that changes only when a non-zero day is found.
    consecutive_zero_count = is_zero_or_na.astype(int).groupby(
        (~is_zero_or_na).cumsum()
    ).cumsum()

    # 4. Determine chunk end points using boolean mask differences
    # We create a mask for where the streak is broken (non-zero value occurs)
    is_streak_broken = (~is_zero_or_na).astype(int).diff().fillna(0)

    # An end occurs on the period *before* the streak is broken (where is_streak_broken == 1)
    # The current day is the first non-zero day, so we shift back by one day to get the end of the zero chunk.
    end_indices_before_transition = is_streak_broken[is_streak_broken == 1].index - pd.Timedelta(days=1)

    # If the DataFrame ends while still in a zero chunk, the end time is the last index
    if is_zero_or_na.iloc[-1]:
        end_indices_before_transition = end_indices_before_transition.append(pd.Index([df_daily.index[-1]]))

    # Use unique, sorted list of all valid end points
    all_end_indices = end_indices_before_transition.unique().sort_values()

    # 5. Calculate the Start Day by backtracking from the End Day
    chunks: List[Dict[str, Union[pd.Timestamp, float]]] = []
    
    # Keep track of starts to avoid processing overlapping chunks (if any)
    processed_start_days = set()

    for end in all_end_indices:
        # Get the length of the consecutive zero run ending on this day
        # We must ensure the index 'end' is in the index for lookup
        if end in consecutive_zero_count.index:
            streak_length = consecutive_zero_count.loc[end]

            # Only process if the streak length meets the threshold
            if streak_length >= days_threshold:
                # Calculate the TRUE start day: Start = End - (Length - 1 day)
                # This correctly identifies the absolute first day of the zero streak.
                start = end - pd.Timedelta(days=int(streak_length) - 1)
                
                # Duration is simply the streak length
                duration = float(streak_length)
                
                normalized_start = start.normalize()
                
                # Check for overlapping chunks before appending (shouldn't happen with this logic, but safe)
                if normalized_start not in processed_start_days:
                    chunks.append({
                        'Start Day': normalized_start,
                        'End Day': end.normalize(),
                        'Duration (Days)': duration
                    })
                    processed_start_days.add(normalized_start)

    return pd.DataFrame(chunks)

# preps two dataframes to run the compare function
def prep_for_comparison(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares two pandas DataFrames for comparison by:
    1. Finding the intersection of columns.
    2. Finding the intersection of indices.
    3. Returning new DataFrames with only the common columns and indices.

    Args:
        df1: The first pandas DataFrame.
        df2: The second pandas DataFrame.

    Returns:
        A tuple of two pandas DataFrames (df1_prep, df2_prep) ready for comparison.
    """
    # 1. Find the intersection of columns
    common_cols = df1.columns.intersection(df2.columns)

    # 2. Find the intersection of indices
    common_indices = df1.index.intersection(df2.index)

    # 3. Create new DataFrames with only common columns and indices
    df1_prep = df1.loc[common_indices, common_cols]
    df2_prep = df2.loc[common_indices, common_cols]

    return df1_prep, df2_prep

# check differences between dataframes column by column, ignoring Na values
def data_diff_check(df1, df2):
    """
    Calculates the percent of non-null fields that differ between two 
    dataframes, for all column pairs with identical names.

    Note: It can be helpful to round the dataframes first if you only
    want to note larger differences

    Parameters
    ----------
    df1 : DataFrame with a DatatimeIndex
    df2 : DataFrame with a DatatimeIndex

    Returns
    -------
    pd.DataFrame
        Dataframe with column names as index and percent (not proportion!)
        of values that differ in that column between dataframes
    """
    common_cols = df1.columns.intersection(df2.columns)
    percent_diff = {}
    if len(common_cols)>0:
        for col in common_cols:
            df1 = df1.dropna(subset = [col])
            df2 = df2.dropna(subset = [col])
            df1_prep, df2_prep = prep_for_comparison(df1, df2)
            x = df1_prep[col].compare(df2_prep[col])
            if len(df1_prep)>0:
                diff = len(x)/len(df1_prep)*100
                percent_diff[col] = diff
            else:
                percent_diff[col] = -9999
    result_df = pd.DataFrame.from_dict(
        percent_diff, 
        orient='index', 
        columns=['percent_different']
    )
    return(result_df)

# determine the optimal lag between two series
def review_lags(data1, data2, max_lag=4):

    """
    Calculates the Cross-Correlation Function (CCF) to find the optimal time 
    lag between two time series.

    The optimal lag is the time shift that results in the maximum absolute 
    correlation between the two series.

    Parameters
    ----------
    data1 : pd.Series
        The primary time series. Must have a DatetimeIndex and be the same 
        frequency as data2.
    data2 : pd.Series
        The secondary time series, which is shifted (lagged) relative to data1. 
        Must have a DatetimeIndex and be the same frequency as data1.
    max_lag : int
        The maximum number of periods (in both positive and negative directions) 
        to test for the lag. The function tests lags from -max_lag to +max_lag.

    Returns
    -------
    pd.Series
        A Series containing the cross-correlation values. 
        The index is the lag (in periods), and the values are the correlation 
        coefficients.

    Notes
    -----
    - **Lag Interpretation:**
      - A **positive lag** (k > 0) means **data2 leads data1** by k periods.
      - A **negative lag** (k < 0) means **data1 leads data2** by |k| periods.
    - **Missing Data:** Pandas' `.corr()` uses pairwise complete observation, 
      meaning it only correlates non-NA values that align by date/time index.
      Shifting `data2` introduces NAs at the start/end, automatically reducing 
      the sample size, which is an expected behavior of lagged correlation.
    """
    try:
        lags = np.arange(-max_lag, max_lag + 1)
        cross_correlations = []
        for lag in lags:
            corr = data1.corr(data2.shift(lag))
            cross_correlations.append(corr)
        
        ccf_series = pd.Series(cross_correlations, index=lags)
        optimal_lag = ccf_series.abs().idxmax()
        max_correlation_value = ccf_series[optimal_lag]
        
        # Printing the results for quick review is maintained as requested
        if optimal_lag == 0 and max_correlation_value.round(3)==1:
            print("Data identical")
        else:
            print(f"Optimal Lag: {optimal_lag} periods")
            print(f"Max Cross-Correlation: {max_correlation_value.round(3)}")
    except Exception as E:
        print(E)
    return cross_correlations, optimal_lag, max_correlation_value

# performs more timeseries validation steps
def validate_timeseries_data(df: pd.DataFrame, interval_minutes: int, date_format: str = '%Y%m%d%H%M') -> Dict[str, Union[bool, str]]:
    """
    Performs several validation checks on a time-series DataFrame with a DatetimeIndex.

    This version includes a robust type coercion step (astype(str) + regex cleanup) 
    to handle the scenario where the START/END columns contain unparsed numeric data
    (like floats ending in .0) which causes comparison failures.

    Args:
        df: The input DataFrame, expected to have a DatetimeIndex and columns 
            named 'TIMESTAMP_START' and 'TIMESTAMP_END' containing datetime-like data.
        interval_minutes: The expected interval between index entries (e.g., 30 or 60).
        date_format: The format string for converting string/numeric dates (default is '%Y%m%d%H%M').

    Returns:
        A dictionary summarizing the results of the three validation checks.
    """
    
    results = {}
    
    # 0. Data Standardization for Robustness (Fixing the Type Mismatch Issue)
    
    # Check if required columns exist
    if 'TIMESTAMP_END' not in df.columns or 'TIMESTAMP_START' not in df.columns:
        return {
            'error': True,
            'message': "Required columns 'TIMESTAMP_START' and 'TIMESTAMP_END' not found. Please ensure your DataFrame columns are named exactly 'TIMESTAMP_START' and 'TIMESTAMP_END'."
        }

    try:
        # Robust Conversion: Handles original data being string, int, or float (like 202406190000.0)
        # 1. astype(str): Converts any numeric type to string.
        # 2. str.replace: Removes trailing '.0' from float conversions.
        # 3. pd.to_datetime: Converts the clean string to a proper datetime object.
        df['END_dt'] = pd.to_datetime(
            df['TIMESTAMP_END'].astype(str).str.replace(r'\.0$', '', regex=True), 
            format=date_format, 
            errors='coerce'
        )
        df['START_dt'] = pd.to_datetime(
            df['TIMESTAMP_START'].astype(str).str.replace(r'\.0$', '', regex=True), 
            format=date_format, 
            errors='coerce'
        )
    except Exception as e:
        return {
            'error': True,
            'message': f"Data conversion failed. Check your 'TIMESTAMP_START'/'TIMESTAMP_END' data and 'date_format' argument. Error: {e}"
        }

    # Check for NaT (Not a Time) values resulting from failed conversions
    if df['END_dt'].isna().any() or df['START_dt'].isna().any():
        return {
            'error': True,
            'message': "Data conversion resulted in NaT values (unparsable dates). Check your input data consistency."
        }


    # 1. Define Timedelta objects for comparison
    # Both index interval and duration must match this Timedelta
    interval_td = pd.Timedelta(minutes=interval_minutes)
    
    # --- CHECK 1: Index Interval Validation ---
    index_diff = df.index.to_series().diff().dropna()
    
    if index_diff.empty:
        is_index_consistent = True
        index_status = "Index consistency check skipped (1 or 0 rows)."
    else:
        is_index_consistent = (index_diff == interval_td).all()
        if is_index_consistent:
            index_status = f"PASS: All index intervals are exactly {interval_minutes} minutes."
        else:
            first_fail = index_diff[index_diff != interval_td].iloc[0]
            fail_index = index_diff[index_diff != interval_td].index[0] 
            index_status = f"FAIL: Index interval is inconsistent. First discrepancy ends at {fail_index}: Found {first_fail} (Expected {interval_td})."

    results['index_interval_check'] = is_index_consistent
    results['index_interval_status'] = index_status


    # --- CHECK 2: End Value Match (The type mismatch fix) ---
    # Does the 'END_dt' value match the index?
    is_end_matching_index = (df['END_dt'] == df.index).all()
    if is_end_matching_index:
        end_status = "PASS: All 'TIMESTAMP_END' values exactly match the DatetimeIndex."
    else:
        mismatch_series = df[df['END_dt'] != df.index].iloc[0]
        # Using the original column name 'TIMESTAMP_END' and 'Index' for better clarity in the fail message
        end_status = f"FAIL: First mismatch at index {mismatch_series.name}. TIMESTAMP_END={mismatch_series['TIMESTAMP_END']}, Index={mismatch_series.name}. (Comparison failed due to mismatched time or type)."
        
    results['end_match_check'] = is_end_matching_index
    results['end_match_status'] = end_status


    # --- CHECK 3: Start-End Difference Validation (UPDATED) ---
    # Is the START value exactly 'interval_minutes' before the END value?
    duration = df['END_dt'] - df['START_dt']
    # Check against the general interval_td (e.g., 30 min or 60 min)
    is_duration_consistent = (duration == interval_td).all() 
    
    if is_duration_consistent:
        duration_status = f"PASS: All TIMESTAMP_START-TIMESTAMP_END durations are exactly {interval_minutes} minutes."
    else:
        mismatch_index = duration[duration != interval_td].index[0]
        first_fail = duration[duration != interval_td].iloc[0]
        duration_status = f"FAIL: Duration inconsistent. First mismatch at index {mismatch_index}: Found {first_fail} (Expected {interval_minutes} minutes)."

    results['duration_check'] = is_duration_consistent
    results['duration_status'] = duration_status
        
    return results

# evaluate offset in time series data in rolling sections
def detect_sectional_offsets_indexed(
    df1, df2, value_col1, value_col2,
    freq='h', max_lag=24, window_size='7D'
):
    """
    Evaluates time offsets between two time series data frames ((datetime-indexed) in
    rolling sections. Returns the best lag with the best offset for each time window.

    Parameters:
    - df1, df2: DataFrames with datetime index.
    - value_col1: name of the column with numerical values to compare for df1
    - value_col2: name of the column with numerical values to compare for df2
    - freq: resampling frequency (e.g., 'h' for hourly).
    - max_lag: maximum lag (in units of freq) to test.
    - window_size: time window for sectional comparison (e.g., '7D' or '12H').

    Returns:
    - DataFrame with lag information per window.
    """

    # Resample both series to ensure regular intervals
    s1 = df1[value_col1].resample(freq).mean()
    s2 = df2[value_col2].resample(freq).mean()

    # Align both series to ensure same timestamps and drop NaNs introduced by resampling
    # Keeping only timestamps present in BOTH resampled series.
    s1, s2 = s1.align(s2, join='inner')
    
    # Drop any remaining NaNs (from initial data gaps) for cleaner segmenting
    # This prevents forward-filling over large gaps, which can be misleading.
    combined = pd.DataFrame({'s1': s1, 's2': s2}).dropna()
    s1 = combined['s1']
    s2 = combined['s2']
    
    # Check if any data remains after cleaning
    if len(s1) == 0:
        return pd.DataFrame()

    # Create window start times
    window_starts = pd.date_range(s1.index.min(), s1.index.max(), freq=window_size)
    results = []

    for start in window_starts:
        end = start + pd.to_timedelta(window_size)
        
        # Select the segment from the cleaned (aligned and dropped NA) series
        seg1 = s1.loc[start:end] 
        seg2 = s2.loc[start:end]

        # Check for sufficient data points in the window
        if len(seg1) < max_lag * 2 or len(seg2) < max_lag * 2:
            continue  # Skip short or empty windows

        lags = np.arange(-max_lag, max_lag + 1)
        
        # Calculate correlations. pd.Series.corr() automatically handles NaNs
        # that might arise from shifting (e.g., when aligning a lagged series)
        correlations = [seg1.corr(seg2.shift(lag)) for lag in lags]

        if all(pd.isna(correlations)):
            continue

        best_lag = lags[np.nanargmax(correlations)]
        best_corr = np.nanmax(correlations)

        results.append({
            'window_start': start,
            'best_lag': best_lag,
            'correlation': best_corr
        })

    result_df = pd.DataFrame(results)

    return result_df

# plots the results of detect_sectional_offsets_indexed
def plot_sectional_lags_plotly(corr_check, height=400):
    """
    Plots the results of the detect_sectional_offsets_indexed function,
    showing the best lag for each timeperiod
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=corr_check['window_start'],
        y=corr_check['best_lag'],
        mode='lines+markers',
        name='Best Lag',
        line=dict(color='royalblue'),
        marker=dict(size=6)
    ))

    # Add zero-lag reference line
    fig.add_trace(go.Scatter(
        x=[corr_check['window_start'].min(), corr_check['window_start'].max()],
        y=[0, 0],
        mode='lines',
        name='Zero Lag',
        line=dict(color='gray', dash='dash')
    ))

    fig.update_layout(
        title='Sectional Time Lag Detection',
        xaxis_title='Window Start Time',
        yaxis_title=f'Best Time Lag',
        template='plotly_white',
        hovermode='x unified',
        height=height
    )

    fig.show()
