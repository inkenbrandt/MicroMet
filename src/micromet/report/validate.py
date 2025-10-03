

# validate test variables to equal 0, 1, 2

import pandas as pd
from typing import List, Dict

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