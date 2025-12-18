"""
This module contains a dictionary of the datetime when sampling freuency was 
updated from 30 minutes to 60 minutes for eddy data (first item in list)
and met data (second item in list).

It also contains a funtion that subsets out data to only include data from before 
or after the interval switch, for a dataframe with a multindex of STATIONID and 
DATETIME_END

"""

interval_update_dict = {
    "US-UTE":	["10/9/2025 11:20",	"10/9/2025 11:20"], # verified; note that first ameriflux statistics record after break needs to be dropped
    "US-UTJ":	["10/9/2025 12:04",	"10/9/2025 12:05"], # verified 10/28/2025
    "US-UTB":	["10/9/2025 13:08",	None],
    'US-UTD':   ["10/22/2025 16:30", "10/22/2025 16:30"], # verified that 30 minutes thru this date; could also check for hourly after this date
    "US-CdM":	["10/9/2025 13:20",	"10/9/2025 13:27"], 
    "US-UTM":	["10/9/2025 13:57",	"10/10/2025 8:15"], # DIFFERENT DAYS!
    "US-UTW":	["10/10/2025 10:24", "10/10/2025 10:37"],
    "US-UTV":	["10/14/25 6:22",	"10/14/2025 11:35"], # verified met and eddy times
    "US-UTP":	["10/13/25 12:13",	"10/13/2025 12:25"],
    "US-UTL":	["10/10/25 13:58",	"10/10/25 13:58"], #written as pm but assuming a.m.
    "US-UTZ":	["10/10/25 1:46", None],
    "US-UTG":	["10/10/2025 7:46",	"10/10/2025 7:54"]
}

from typing import Dict, List, Union
import pandas as pd

def subset_interval(
    df: pd.DataFrame, 
    date_dict: Dict[Union[int, str], List[str]], 
    interval: int, 
    data_type: str
) -> pd.DataFrame:
    """
    Subsets a MultiIndex DataFrame based on station ID, a date cutoff, 
    and a data_type, using a single vectorized boolean mask.

    Args:
        df (pd.DataFrame): MultiIndex DataFrame with levels 'STATIONID' 
                           and 'DATETIME_END'.
        date_dict (dict): Dictionary where keys are 'STATIONID' and values 
                          are a list of two date strings [date1, date2].
        interval (int): Condition for subsetting. 30 for dates <= cutoff, 
                        60 for dates > cutoff.
        data_type (str): Determines which date to use as the cutoff:
                         'eddy' uses the first date (index 0).
                         'met' uses the second date (index 1).

    Returns:
        pd.DataFrame: The subsetted DataFrame containing data from all relevant stations.
    """
    
    id_level = df.index.get_level_values('STATIONID')
    date_level = df.index.get_level_values('DATETIME_END')

    if data_type.lower() == 'eddy':
        date_index = 0 
    elif data_type.lower() == 'met':
        date_index = 1 
    else:
        raise ValueError(f"Unsupported data_type: {data_type}. Must be 'eddy' or 'met'.")
    
    df_station_ids = set(id_level.unique())

    stations_to_process = df_station_ids.intersection(date_dict.keys())

    final_mask = pd.Series(False, index=df.index)

    for station_id in stations_to_process:
        dates = date_dict[station_id]
        try:
            cutoff_date = pd.to_datetime(dates[date_index])
        except IndexError:
            print(f"Warning: Date list for station {station_id} does not have an element at index {date_index}. Skipping.")
            continue
        
        station_mask = (id_level == station_id)
        
        if interval == 30:
            date_condition = date_level <= cutoff_date
        elif interval == 60:
            date_condition = date_level > cutoff_date
        else:
            print(f"Warning: Unsupported interval value {interval}. Skipping station {station_id}.")
            continue

        combined_mask = station_mask & date_condition
        
        final_mask = final_mask | combined_mask
    
    return df.loc[final_mask]