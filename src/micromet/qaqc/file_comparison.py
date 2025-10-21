import pandas as pd

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