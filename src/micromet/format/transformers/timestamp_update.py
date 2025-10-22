''' various scripts
for trying to address timestamp issues in the data
'''


'''
I haven't gotten this script to work properly yet. The script has
errored out at the part where I process the df60 dataframe, though the
df30 processing worked fine.

Note that this script requires that you identify the datetime column 
before running "process" which may be problematic for some files.

Here is an example of how I was running the script:

am_data = micromet.Reformatter(drop_soil=True,logger=logger)
in_df = pd.read_csv(file_to_read,skiprows=[0,2,3],
                    na_values=[-9999,"NAN","NaN","nan"])
in_df['TIMESTAMP'] = pd.to_datetime(in_df['TIMESTAMP'])
in_df["TIMESTAMP_END"] = in_df.TIMESTAMP.dt.strftime("%Y%m%d%H%M").astype(int)
df, report = process_by_interval(in_df, key, interval_dict, datatype)

'''
def process_by_interval(in_df, key, interval_dict, datatype):
    '''
    The goal of this script is to use the interval_updates dictionary to 
    identify when data switched from 30 to 60 minute sampling and then process
    the data correctly. 
    '''
    if key in interval_dict.keys():
        if (datatype=="eddy") & (interval_dict[key][0]!=None):
            change_date = pd.to_datetime(interval_dict[key][0])
        elif (datatype=="met") & (interval_dict[key][1]!=None):
            change_date = pd.to_datetime(interval_dict[key][1])
        else:
            logger.debug(f"Station not in interval dictionary for {key} {datatype} data")
            change_date = None
    if change_date:
        if (in_df.TIMESTAMP.max()<change_date):
            logger.debug("Processing all data at 30 minutes")
            df, report, checktime = am_data.process(in_df, interval=30, data_type=datatype)
        elif (in_df.TIMESTAMP.min()>change_date):
            logger.debug("Processing all data at 60 minutes")
            df, report, checktime = am_data.process(in_df, interval=60, data_type=datatype)
        elif (in_df.TIMESTAMP.max()>change_date) & (in_df.TIMESTAMP.min()<change_date):
            #just a check on the data interval switch date
            time_diff_td = in_df.TIMESTAMP.diff()
            in_df['timediff'] = time_diff_td.dt.total_seconds() / 60
            check60_date = (change_date + pd.Timedelta(hours=1)).floor('h')
            check30_date = (change_date.floor('h'))
            check30 = in_df.loc[in_df.TIMESTAMP==check30_date, 'timediff'].iloc[0]
            check60 = in_df.loc[in_df.TIMESTAMP==check60_date, 'timediff'].iloc[0]
            if (check30!=30) | (check60 != 60):
                logger.warning("Date when sampling interval changed may be incorrect based on index differences")
            in_df.drop(columns=['timediff'], inplace=True)

            logger.debug(f"Processing data at 30 minutes before {change_date} and 60 minutes after")
            df60 = in_df[in_df.TIMESTAMP>change_date]
            df60_process, report60, checktime = am_data.process(df60, interval=60, data_type=datatype)
            df30 = in_df[in_df.TIMESTAMP<=change_date]
            df30_process, report30, checktime = am_data.process(df30, interval=30, data_type=datatype)
            df = pd.concat([df60_process, df30_process])
            report = pd.concat([report30, report60])
    else:
        logger.warning("Site not found in interval dictionary; processing all data to 30 minutes")
        df, report, checktime = am_data.process(in_df, interval=30, data_type=datatype)
    return(df, report)



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