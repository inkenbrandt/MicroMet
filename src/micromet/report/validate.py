

# validate test variables to equal 0, 1, 2

import pandas as pd
import numpy as np
from typing import List, Dict, Union


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
            diff = len(x)/len(df1_prep)*100
            percent_diff[col] = diff
    result_df = pd.DataFrame.from_dict(
        percent_diff, 
        orient='index', 
        columns=['percent_different']
    )
    return(result_df)


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