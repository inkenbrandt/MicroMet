import pandas as pd
import numpy as np

def prep_parquet(station, df, install_date):
    not_subset = ['TIMESTAMP_START', 'TIMESTAMP_END',
               'FC_SSITC_TEST', 'LE_SSITC_TEST', 
               'ET_SSITC_TEST', 'H_SSITC_TEST','TAU_SSITC_TEST' ]
    
    df = df.loc[station]
    df = df.replace(-9999, np.nan)
    filtered_cols = [col for col in df.columns if col not in not_subset]
    df = df.dropna(axis=1, how='all')
    df = df[df.index.floor('D')>install_date]
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