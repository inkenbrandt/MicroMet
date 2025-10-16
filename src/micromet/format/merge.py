import pandas as pd
import numpy as np

def fillna_with_second_df(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    suffix1: str = '_df1',
    suffix2: str = '_df2'
) -> pd.DataFrame:
    """
    Merges two DataFrames by index, prioritizing data from df1 and using df2 
    to fill any missing (NaN) values introduced by the outer merge for any 
    columns that match between the two dataframes.

    Parameters
    ----------
    df1 : pd.DataFrame
        The primary DataFrame whose index and values are prioritized.
    df2 : pd.DataFrame
        The secondary DataFrame used to fill NaN values in df1's columns.
    suffix1 : str, optional
        The suffix to apply to columns from df1 during the merge. 
        The default is '_df1'. This suffix is removed from the output. 
        Select a suffix that is not a string in a column name in either dataframe
    suffix2 : str, optional
        The suffix to apply to columns from df2 during the merge. 
        The default is '_df2'. These columns are dropped from the output.
        Select a suffix that is not a string in a column name in either dataframe

    Returns
    -------
    pd.DataFrame
        A merged DataFrame containing the union of both indices. Columns 
        are filled: df1's value if present, otherwise df2's value.
        The final column names are stripped of suffix1.
    
    Notes
    -----
    This function assumes that the column names (excluding suffixes) 
    in both DataFrames are the same for matching purposes.
    """
    # Check df1 columns for suffix1 or suffix2
    if any(df1.columns.str.contains(suffix1, regex=False)) or \
       any(df1.columns.str.contains(suffix2, regex=False)):
        raise ValueError(
            f"Error: Columns in df1 already contain '{suffix1}' or '{suffix2}'. "
            "Please select different suffix values."
        )

    # Check df2 columns for suffix1 or suffix2
    if any(df2.columns.str.contains(suffix1, regex=False)) or \
       any(df2.columns.str.contains(suffix2, regex=False)):
        raise ValueError(
            f"Error: Columns in df2 already contain '{suffix1}' or '{suffix2}'. "
            "Please select different suffix values."
        )
    
    # Merge datasets and identify column sets
    mergedat = df1.merge(df2, left_index=True, right_index=True, how='outer', suffixes=[suffix1, suffix2])
    df1_cols = mergedat.columns[mergedat.columns.str.contains(suffix1, regex=False)]
    df2_cols = mergedat.columns[mergedat.columns.str.contains(suffix2, regex=False)]

    # 3. Coalesce Values (Fill df1's NaN with df2's values)
    for col1 in df1_cols:
        base_name = col1.removesuffix(suffix1)
        col2 = base_name + suffix2
        mergedat[col1] = mergedat[col1].fillna(mergedat[col2])

    mergedat = mergedat.drop(columns=df2_cols, errors='ignore')
    
    mergedat = mergedat.rename(columns=lambda x: x.removesuffix(suffix1))
    
    return mergedat