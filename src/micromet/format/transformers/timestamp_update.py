import pandas as pd
import numpy as np

def resample_alternating_frequency_with_other(df, min_records_threshold=24):
    """
    Identifies contiguous blocks of data, resamples 30min/60min blocks,
    and assigns 'OTHER' to the timestep for unclassified (non-gap) blocks.
    """
    
    # --- Step 1: Calculate Time Differences and Classify ---
    df['time_diff'] = df.index.to_series().diff().dt.total_seconds() / 60
    df['time_diff_min'] = df['time_diff'].round()

    def classify_frequency(diff):
        if pd.isna(diff):
            return 'UNKNOWN'
        if 25 <= diff <= 35:
            return '30MIN'
        elif 55 <= diff <= 65:
            return '60MIN'
        else:
            # All other differences are 'OTHER' frequencies or true gaps
            return 'OTHER'

    df['frequency_class'] = df['time_diff_min'].apply(classify_frequency)

    # --- Step 2: Create Block Categories and IDs ---
    
    # 'other_to_nan' column: ONLY UNKNOWN and sustained GAPS become NaN
    # The 'OTHER' string is preserved here for temporary identification.
    df['block_category_temp'] = df['frequency_class'].replace('UNKNOWN', np.nan)
    
    # 1. Forward fill the very first NaN (UNKNOWN).
    df['block_category_temp'] = df['block_category_temp'].ffill()

    # 2. Backward fill with limit=1. Corrects the first row of any block.
    df['block_category_temp'] = df['block_category_temp'].bfill(limit=1)

    # This column will contain '30MIN', '60MIN', 'OTHER', or NaN (for true gaps)
    df['block_category'] = df['block_category_temp']

    # Create Block IDs for both defined blocks, 'OTHER' blocks, and sustained gaps
    df['block_id'] = (df['block_category'].fillna('GAP_BLOCK') != df['block_category'].fillna('GAP_BLOCK').shift(1)).cumsum()

    # --- Step 3: Separate and Iterate Over Defined Blocks and 'OTHER' Blocks ---

    # Filter for blocks that are NOT true NaNs (i.e., NOT sustained gaps)
    defined_or_other_blocks = df[df['block_category'].notna()]

    resampled_list = []
    previous_freq = None
    
    for block_id, block in defined_or_other_blocks.groupby('block_id'):
        
        current_category = block['block_category'].iloc[0]

        # Handle 30MIN and 60MIN blocks
        if current_category in ['30MIN', '60MIN']:
            current_freq = int(current_category.replace('MIN', ''))

            # Apply the minimum record threshold logic
            if len(block) >= min_records_threshold:
                final_freq_for_resample = current_freq
                previous_freq = current_freq
            elif previous_freq is not None:
                final_freq_for_resample = previous_freq
            else:
                final_freq_for_resample = current_freq
                
            # Resample and assign timestep
            freq_str = f"{final_freq_for_resample}min"
            
            original_cols = block.columns.drop(['time_diff', 'time_diff_min', 'frequency_class', 'block_category_temp', 'block_category', 'block_id'], errors='ignore')
            resampled_block = block[original_cols].resample(freq_str).last()
            resampled_block['timestep'] = final_freq_for_resample

        # Handle 'OTHER' blocks
        else: # current_category == 'OTHER'
            # Do NOT resample 'OTHER' blocks, just assign the timestep and keep the original data
            resampled_block = block.drop(columns=['time_diff', 'time_diff_min', 'frequency_class', 'block_category_temp', 'block_category', 'block_id'], errors='ignore').copy()
            resampled_block['timestep'] = -1
        
        resampled_list.append(resampled_block)


    # --- Step 4: Recombine with Original Gap Rows ---
    
    final_resampled_blocks = pd.concat(resampled_list)

    # Get the original true gap rows (block_category is NaN)
    gap_rows = df[df['block_category'].isna()].drop(
        columns=['time_diff', 'time_diff_min', 'frequency_class', 'block_category_temp', 'block_category', 'block_id'], 
        errors='ignore'
    )

    # Concatenate resampled data, 'OTHER' data, and the true gap rows, then sort
    final_df = pd.concat([final_resampled_blocks, gap_rows]).sort_index()

    # Clean up the timestep column for the true gap rows (they should be NaN)
    final_df['timestep'] = final_df['timestep'].replace('', np.nan).fillna('TRUE_GAP')

    # Final cleanup of temporary columns
    final_df = final_df.drop(columns=['time_diff', 'time_diff_min', 'frequency_class', 'block_category_temp', 'block_category', 'block_id'], errors='ignore')

    return final_df


import pandas as pd
import logging
import numpy as np

# Set logging for demonstration purposes (can be removed in production)
# logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def resample_single_frequency_switch(df, sample_size=100):
    """
    Resamples a DataFrame based on a single detected frequency switch (30min to 60min).
    It uses the mode of the first 100 records to robustly determine the initial frequency,
    handling minor clock jitter and occasional gaps.

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex.
        sample_size (int): The number of initial records to analyze for the starting routine.

    Returns:
        pd.DataFrame: Resampled DataFrame with a 'timestep' column.
    """
    
    # 1. Input Validation and Preparation
    if not isinstance(df.index, pd.DatetimeIndex):
        logging.error("DataFrame index must be a DatetimeIndex.")
        raise ValueError("DataFrame index must be a DatetimeIndex.")
        
    if len(df) < 2:
        logging.warning("DataFrame has fewer than 2 rows; cannot determine frequency.")
        df['timestep'] = pd.NA
        return df

    # Calculate Time Differences and Round to nearest minute (handles jitter)
    df['time_diff'] = df.index.to_series().diff().dt.total_seconds() / 60
    df['time_diff_rounded'] = df['time_diff'].round()
    
    # Floor the Index to the nearest minute to prepare for accurate resampling
    df.index = df.index.floor('min')
    
    # 2. Determine Initial and Final Routines (Robust Check)

    N = len(df)
    
    # Safely get a sample of the initial time differences (skipping the first NaN)
    analysis_size = min(sample_size + 1, N) # Sample size + 1 to account for the skip
    
    # Calculate the mode of the initial routine
    initial_mode = df['time_diff_rounded'].iloc[1:analysis_size].mode()
    initial_routine = initial_mode.iloc[0] if not initial_mode.empty else None

    # The final routine is the last valid time difference
    final_routine = df['time_diff_rounded'].iloc[-1]

    # --- 3. Conditional Resampling Logic ---
    
    # Case 1: All data is 60-minute
    if (final_routine == 60) and (initial_routine == 60):
        df = df.resample('60min').last()
        df['timestep'] = 60
        logging.debug('All data hourly.')

    # Case 2: All data is 30-minute
    elif (final_routine == 30) and (initial_routine == 30):
        df = df.resample('30min').last()
        df['timestep'] = 30
        logging.debug('All data half-hourly.')

    # Case 3: Switch from 30-minute to 60-minute
    elif (final_routine == 60) and (initial_routine == 30):
        
        # Find the timestamp of the last *valid* 30-minute rounded difference
        max30 = df[df['time_diff_rounded'] == 30].index.max()
        
        # Split the data into 30min and 60min sections
        mask = df.index >= max30
        
        # Resample 30-minute section
        df30 = df[~mask].copy()
        df30 = df30.resample('30min').last()
        df30['timestep'] = 30
        
        # Resample 60-minute section
        df60 = df[mask].copy()
        df60 = df60.resample('60min').last()
        df60['timestep'] = 60
        
        # Concatenate and sort
        df = pd.concat([df30, df60], axis=0).sort_index()
        logging.debug(f'Mixed timestamps. Hourly data starts on {max30}.')

    # Case 4: Other (e.g., 60 to 30, or initial mode is neither 30 nor 60)
    else:
        logging.warning(f"Unhandled routine pattern: Initial={initial_routine}, Final={final_routine}. No resampling performed.")
        # If unhandled, drop temp columns and return the original DataFrame
        df.drop(columns=['time_diff', 'time_diff_rounded'], inplace=True, errors='ignore')
        df['timestep'] = pd.NA
        return df

    # Final cleanup and return
    df.drop(columns=['time_diff', 'time_diff_rounded'], inplace=True, errors='ignore')
    return df