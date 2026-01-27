import pandas as pd
import numpy as np

def update_albedo(df, suffix):
    """
    Calculates the Shortwave Albedo percentage based on incoming and outgoing radiation.

    Following the EasyFlux methodology, this function computes albedo where solar 
    radiation is sufficient and ensures that missing sensor data or physical 
    impossibilities (like night time or sensor shading) are handled correctly.

    Args:
        df (pd.DataFrame): The dataset containing radiation measurements.
        suffix (str): The sensor or level identifier used in column naming 
                      (e.g., '1' for 'SW_IN_1').

    Returns:
        np.ndarray: Calculated Albedo values as a percentage (0-100%). 
                    Returns 0 during night/invalid conditions and NaN if inputs are missing.
    """
    sw_in_col = f'SW_IN_{suffix}'
    sw_out_col = f'SW_OUT_{suffix}'

    # Logic: SWin must be above threshold and greater than reflected radiation
    daylight_mask = (df[sw_in_col] > 0.1) & (df[sw_in_col] >= df[sw_out_col])
    
    # Check for missing data in either component
    missing_mask = df[sw_in_col].isna() | df[sw_out_col].isna()

    calculated_albedo = np.where(
        missing_mask, 
        np.nan, 
        np.where(daylight_mask, (df[sw_out_col] / df[sw_in_col]) * 100.0, 0)
    )
    
    return calculated_albedo