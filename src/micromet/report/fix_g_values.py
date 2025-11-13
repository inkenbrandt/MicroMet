import pandas as pd
import numpy as np


def correct_vars_by_factor(df, correction_factor=0.05/0.16, 
                           vars_to_correct=['SG_1_1_1','SG_2_1_1'], 
                           min_correction_date=pd.to_datetime('2010-01-01'),
                           max_correction_date=pd.to_datetime('2030-01-01')):
    """
    Applies a multiplicative correction factor to specified variables within a 
    defined time window.
    The default min and max correction dates are intended to correct the full 
    range of values.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame, expected to have a DatetimeIndex.
    correction_factor : float, optional
        The factor by which the variables should be multiplied. 
        Default is 0.05 / 0.16.
    vars_to_correct : list of str, optional
        List of column names to apply the correction to.
    min_correction_date : datetime, optional
        Start date (inclusive) for the correction window.
    max_correction_date : datetime, optional
        End date (inclusive) for the correction window.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the specified columns corrected within the date 
        range.
    """
    df_out = df.copy()
    mask = (df_out.index >= min_correction_date) & (
        df_out.index<=max_correction_date)
    df_out.loc[mask, vars_to_correct] = df_out.loc[
        mask, vars_to_correct]*correction_factor
    return(df_out)


def apply_limits_to_vars(df, limit_check_vars, limits):
    """
    Sets values in specified columns that fall outside a given [min, max] range 
    to NaN.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    limit_check_vars : list of str
        List of column names to apply the limits to.
    limits : list or tuple
        A two-element sequence [min_value, max_value].

    Returns
    -------
    pd.DataFrame
        A new DataFrame with out-of-range values set to NaN.
    """
    df_out = df.copy()
    for col in limit_check_vars:
        mask = (df_out[col]<limits[0]) | (df_out[col]>limits[1])
        df_out.loc[mask, col] = np.nan
        print(f'{mask.sum()} values dropped from {col} b/c value out of range') 
    return(df_out)   


def calculate_new_g_value(df, plate_num):
    """
    Calculates the new G value (G_{plate_num}__1_1) by summing the G_PLATE and SG components.
    
    Note: The sum operation automatically results in NaN if either source value is NaN.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    plate_num : str
        The plate number (e.g., '1' or '2') used to construct column names.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the calculated G value.
    """
    df_out = df.copy()
    col_g = f'G_{plate_num}_1_1'
    col_g_plate = f'G_PLATE_{plate_num}_1_1'
    col_sg = f'SG_{plate_num}_1_1'

    # Mask identifies records where G will become missing because G_PLATE or SG is missing,
    # but G currently has a value. This is purely for the print statement.
    mask = ((df_out[col_g_plate].isna()) | (df_out[col_sg].isna())) & (
        ~df_out[col_g].isna()
    )
    print(f'{mask.sum()} new records will having missing G values b/c SG or G_PLATE missing') 
    
    # Calculate the new G value. The addition automatically propagates NaNs if either 
    # source is NaN, which is generally the desired behavior.
    df_out[col_g] = df_out[col_g_plate] + df_out[col_sg]
    
    return df_out

def calc_mean_value_for_soil(df, var='G'):
    """
    Calculates the mean of two related variables (var_1_1_1 and var_2_1_1) 
    and stores the result in a third variable (var_1_1_A).

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    var : str, optional
        The variable prefix (e.g., 'G', 'SG'). Default is 'G'.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the calculated mean value in the 'var_1_1_A' column.
    """
    df_out = df.copy()
    col1 = f'{var}_1_1_1'
    col2 = f'{var}_2_1_1'
    col_mean = f'{var}_1_1_A'
    
    # FIX 1: Corrected the mask to check col1 AND col2 (it was checking col1 twice)
    # This mask is for the print statement, checking if mean will become NaN when it 
    # previously had a value.
    mask_print = ((df_out[col1].isna()) | (df_out[col2].isna())) & (
        ~df_out[col_mean].isna()
    )
    print(f'{mask_print.sum()} new records will having missing Mean {var} b/c {var}1 or {var}2 is missing') 
    
    # Initialize the mean column to NaN (to clear previous values)
    df_out[col_mean] = np.nan 

    # Calculation mask: only calculate the mean where BOTH inputs are NOT NaN
    calc_mask = (~df_out[col1].isna()) & (~df_out[col2].isna())
    
    # Apply the calculation only to the rows where both inputs are valid
    df_out.loc[calc_mask, col_mean] = (df_out[col1] + df_out[col2]) / 2
    
    return df_out

def run_soil_data_pipeline(df_input, 
                            sg_correction_factor=0.05/0.16,
                            sg_limits=[-100, 250], 
                            g_limits=[-250, 400]):
    """
    Executes the full, seven-step data processing pipeline for soil data.

    The steps include:
    1. Applying correction factor to SG variables.
    2. Applying limits/quality control to SG variables.
    3. Calculating G values for plate 1 and plate 2 based on SG plus .
    4. Applying limits/quality control to calculated G variables.
    5. Calculating the mean G value (G_1_1_A).
    6. Applying limits/quality control to the mean G variable.

    Parameters
    ----------
    df_input : pd.DataFrame
        The initial input DataFrame (e.g., 'final_eddy').
    sg_correction_factor : float
        Correction factor for SG variables.
    sg_limits : list or tuple
        Min/max limits for SG variables.
    g_limits : list or tuple
        Min/max limits for G variables (G_1_1_1, G_2_1_1, G_1_1_A).

    Returns
    -------
    pd.DataFrame
        The final processed DataFrame.
    """
    print("--- Starting G Fix Data Pipeline ---")
    
    # STEP 1: Apply correction factor (SG variables)
    print("Step 1: SG correction applied.")
    temp1 = correct_vars_by_factor(
        df_input, 
        correction_factor=sg_correction_factor, 
        vars_to_correct = ['SG_1_1_1', 'SG_2_1_1', 'SG_1_1_A']
    )
    

    # STEP 2: Apply limits to corrected SG variables
    print('\n')
    print("Step 2: SG limits applied.")
    temp2 = apply_limits_to_vars(
        temp1, 
        limit_check_vars=['SG_1_1_1', 'SG_2_1_1', 'SG_1_1_A'], 
        limits=sg_limits
    )
    

    # STEP 3 & 4: Calculate new G values for plate 1 and plate 2
    print('\n')
    print("Step 3 & 4: G values calculated (G_1_1_1 and G_2_1_1).")
    temp3 = calculate_new_g_value(temp2, '1')
    temp4 = calculate_new_g_value(temp3, '2')
    

    # STEP 5: Apply limits to the new G variables
    print('\n')
    print("Step 5: G limits applied to individual plates.")
    temp5 = apply_limits_to_vars(
        temp4, 
        limit_check_vars=['G_1_1_1', 'G_2_1_1'], 
        limits=g_limits
    )
    
    # STEP 6: Calculate the mean G value
    print('\n')
    print("Step 6: Mean G value (G_1_1_A) calculated.")
    temp6 = calc_mean_value_for_soil(temp5, 'G')

    # STEP 7: Apply limits to the mean G variable
    print('\n')
    print("Step 7: Final mean G limits applied.")
    temp7 = apply_limits_to_vars(
        temp6, 
        limit_check_vars=['G_1_1_A'], 
        limits=g_limits
    )
    
    print("--- Pipeline Finished ---")
    return temp7