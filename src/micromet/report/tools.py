from scipy.signal import find_peaks
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from typing import Union, List, Dict

import matplotlib.pyplot as plt


def find_irr_dates(
    df, swc_col="SWC_1_1_1", do_plot=False, dist=20, height=30, prom=0.6
):
    """
    Detect irrigation events from soil water content time series.

    Identifies peaks in soil water content (SWC) data that are likely associated
    with irrigation events. Detection is based on peak prominence, height, and distance.
    Optionally plots the full time series with detected events highlighted.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing time-indexed soil water content data.
    swc_col : str, optional
        Column name containing soil water content values in percent
        (not fractional). Default is 'SWC_1_1_1'.
    do_plot : bool, optional
        If True, generates a plot showing SWC time series and identified peaks.
        Default is False.
    dist : int, optional
        Minimum number of time steps between detected peaks. Default is 20.
    height : float, optional
        Minimum height of peaks to be considered irrigation events. Default is 30 (%).
    prom : float, optional
        Minimum prominence of the peaks. Default is 0.6.

    Returns
    -------
    dates_of_irr : pandas.DatetimeIndex
        Timestamps corresponding to detected irrigation events.
    swc_during_irr : pandas.Series
        Soil water content values at the detected peaks.

    Notes
    -----
    - Only data from April through October (inclusive) is considered as the irrigation season.
    - Uses `scipy.signal.find_peaks` for peak detection.
    - Intended for use with SWC data measured in percent, not fractional values.
    """
    df_irr_season = df[df.index.month.isin([4, 5, 6, 7, 8, 9, 10])]
    peaks, _ = find_peaks(
        df_irr_season[swc_col], distance=dist, height=height, prominence=(prom, None)
    )
    dates_of_irr = df_irr_season.iloc[peaks].index
    swc_during_irr = df_irr_season[swc_col].iloc[peaks]
    if do_plot:
        plt.plot(df.index, df[swc_col])
        plt.plot(dates_of_irr, swc_during_irr, "x")
        plt.show()
    return dates_of_irr, swc_during_irr


def find_gaps(df, columns, missing_value=-9999, min_gap_periods=1):
    """
    Find gaps in time series data where values are either NaN or equal to missing_value
    for longer than min_gap_periods.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a time series index of regular frequency.
    columns : str or list of str
        Column(s) to check for gaps.
    missing_value : numeric, default -9999
        Value to consider as missing data alongside NaN.
    min_gap_periods : int, default 1
        Minimum number of consecutive missing periods to be considered a gap.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing gap information with columns:
        - gap_start: start datetime of gap
        - gap_end: end datetime of gap
        - duration_hours: duration of gap in hours
        - missing_records: number of missing records in gap
        - column: name of column where gap was found
    """
    if isinstance(columns, str):
        columns = [columns]

    # Initialize list to store gap information
    gaps = []

    for col in columns:
        # Create boolean mask for missing values
        is_missing = df[col].isna() | (df[col] == missing_value)

        # Get the frequency of the time series as a pandas timedelta
        freq = pd.tseries.frequencies.to_offset(pd.infer_freq(df.index))

        # Find runs of missing values
        missing_runs = (is_missing != is_missing.shift()).cumsum()[is_missing]

        if len(missing_runs) == 0:
            continue

        # Group consecutive missing values
        for run_id in missing_runs.unique():
            run_mask = missing_runs == run_id
            run_indices = missing_runs[run_mask].index

            # Only consider runs longer than min_gap_periods
            if len(run_indices) > min_gap_periods:
                gap_start = run_indices[0]
                gap_end = run_indices[-1]

                # Calculate duration in hours
                duration_hours = (gap_end - gap_start).total_seconds() / 3600

                gaps.append(
                    {
                        "gap_start": gap_start,
                        "gap_end": gap_end,
                        "duration_hours": duration_hours,
                        "missing_records": len(run_indices),
                        "column": col,
                    }
                )

    if not gaps:
        return pd.DataFrame(
            columns=[
                "gap_start",
                "gap_end",
                "duration_hours",
                "missing_records",
                "column",
            ]
        )

    return pd.DataFrame(gaps).sort_values("gap_start").reset_index(drop=True)


def plot_gaps(gaps_df, title="Time Series Data Gaps"):
    """
    Create a Gantt chart visualization of gaps in time series data.

    Parameters
    ----------
    gaps_df : pandas.DataFrame
        DataFrame containing gap information as returned by `find_gaps()`.
    title : str, default "Time Series Data Gaps"
        Title for the plot.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure showing gaps as a Gantt chart.
    """
    if len(gaps_df) == 0:
        print("No gaps found to plot.")
        return None

    # Create figure
    fig = go.Figure()

    # Get unique columns and assign colors
    unique_columns = gaps_df["column"].unique()
    # Define a set of colors manually
    colors = [
        "rgb(166,206,227)",
        "rgb(31,120,180)",
        "rgb(178,223,138)",
        "rgb(51,160,44)",
        "rgb(251,154,153)",
        "rgb(227,26,28)",
        "rgb(253,191,111)",
        "rgb(255,127,0)",
        "rgb(202,178,214)",
    ]
    # Cycle through colors if more variables than colors
    color_map = dict(
        zip(
            unique_columns,
            [colors[i % len(colors)] for i in range(len(unique_columns))],
        )
    )

    # Add gaps as horizontal bars
    for idx, row in gaps_df.iterrows():
        fig.add_trace(
            go.Bar(
                x=[row["duration_hours"]],
                y=[row["column"]],
                orientation="h",
                base=[(row["gap_start"] - pd.Timestamp.min).total_seconds() / 3600],
                marker_color=color_map[row["column"]],
                name=row["column"],
                showlegend=False,
                hovertemplate=(
                    f"Column: {row['column']}<br>"
                    + f"Start: {row['gap_start']}<br>"
                    + f"End: {row['gap_end']}<br>"
                    + f"Duration: {row['duration_hours']:.1f} hours<br>"
                    + f"Missing Records: {row['missing_records']}"
                ),
            )
        )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Variables",
        barmode="overlay",
        height=max(200, 100 * len(unique_columns)),
        showlegend=False,
        xaxis=dict(tickformat="%Y-%m-%d %H:%M", type="date"),
    )

    return fig


def detect_extreme_variations(
    df: pd.DataFrame,
    fields: Union[str, List[str]] = None,
    frequency: str = "D",
    variation_threshold: float = 3.0,
    null_value: Union[float, int] = -9999,
    min_periods: int = 2,
) -> Dict[str, pd.DataFrame]:
    """
    Detect extreme variations in specified fields of a datetime-indexed DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with datetime index.
    fields : str or list of str, optional
        Column names to analyze. If None, analyzes all numeric columns.
    frequency : str, default 'D'
        Frequency to analyze variations over (e.g., 'D' for daily, 'H' for hourly).
    variation_threshold : float, default 3.0
        Number of standard deviations beyond which a variation is considered extreme.
    null_value : float or int, default -9999
        Value to be treated as null.
    min_periods : int, default 2
        Minimum number of valid observations required to calculate variation.

    Returns
    -------
    dict
        Dictionary with the following keys:
        - 'variations': DataFrame with calculated variation metrics.
        - 'extreme_points': DataFrame with boolean flags for extreme values.
        - 'summary': Summary statistics for each field.
    """
    # Validate input DataFrame
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a datetime index")

    # Create copy of DataFrame
    df_copy = df.copy()

    # Replace null values
    df_copy = df_copy.replace(null_value, np.nan)

    # Select fields to analyze
    if fields is None:
        fields = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    elif isinstance(fields, str):
        fields = [fields]

    # Initialize results
    variations = pd.DataFrame(index=df_copy.index)
    extreme_points = pd.DataFrame(index=df_copy.index)
    summary_stats = []

    # Calculate variations and detect extremes for each field
    for field in fields:
        # Group by frequency and calculate statistics
        grouped = df_copy[field].groupby(pd.Grouper(freq=frequency))

        # Calculate variation metrics
        field_var = f"{field}_variation"
        variations[field_var] = grouped.transform(
            lambda x: (
                np.abs(x - x.mean()) / x.std()
                if len(x.dropna()) >= min_periods
                else np.nan
            )
        )

        # Flag extreme variations
        extreme_points[f"{field}_extreme"] = variations[field_var] > variation_threshold

        # Calculate summary statistics
        field_summary = {
            "field": field,
            "total_observations": len(df_copy[field].dropna()),
            "extreme_variations": extreme_points[f"{field}_extreme"].sum(),
            "mean_variation": variations[field_var].mean(),
            "max_variation": variations[field_var].max(),
            "std_variation": variations[field_var].std(),
        }
        summary_stats.append(field_summary)

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_stats)

    return {
        "variations": variations,
        "extreme_points": extreme_points,
        "summary": summary_df,
    }


def clean_extreme_variations(
    df: pd.DataFrame,
    fields: Union[str, List[str]] = None,
    frequency: str = "D",
    variation_threshold: float = 3.0,
    null_value: Union[float, int] = -9999,
    min_periods: int = 2,
    replacement_method: str = "nan",
) -> Dict[str, Union[pd.DataFrame, pd.DataFrame]]:
    """
    Clean extreme variations from specified fields in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with datetime index.
    fields : str or list of str, optional
        Column names to clean. If None, processes all numeric columns.
    frequency : str, default 'D'
        Frequency to analyze variations over (e.g., 'D' for daily, 'H' for hourly).
    variation_threshold : float, default 3.0
        Number of standard deviations beyond which a variation is considered extreme.
    null_value : float or int, default -9999
        Value to be treated as null.
    min_periods : int, default 2
        Minimum number of valid observations required to calculate variation.
    replacement_method : str, default 'nan'
        Method to handle extreme values:
        - 'nan': Replace with NaN
        - 'interpolate': Linear interpolation
        - 'mean': Replace with frequency mean
        - 'median': Replace with frequency median

    Returns
    -------
    dict
        Dictionary with the following keys:
        - 'cleaned_data': DataFrame with cleaned data.
        - 'cleaning_summary': Summary of cleaning operations.
        - 'removed_points': DataFrame showing removed or replaced values.
    """
    # Validate replacement method
    valid_methods = ["nan", "interpolate", "mean", "median"]
    if replacement_method not in valid_methods:
        raise ValueError(f"replacement_method must be one of {valid_methods}")

    # Detect extreme variations
    variation_results = detect_extreme_variations(
        df=df,
        fields=fields,
        frequency=frequency,
        variation_threshold=variation_threshold,
        null_value=null_value,
        min_periods=min_periods,
    )

    # Create copy of input DataFrame
    cleaned_df = df.copy()

    # Initialize summary statistics
    cleaning_summary = []
    removed_points = pd.DataFrame(index=df.index)

    # Process each field
    if fields is None:
        fields = df.select_dtypes(include=[np.number]).columns.tolist()
    elif isinstance(fields, str):
        fields = [fields]

    for field in fields:
        # Get extreme points for this field
        extreme_mask = variation_results["extreme_points"][f"{field}_extreme"]

        # Store removed values
        removed_points[field] = np.where(extreme_mask, cleaned_df[field], np.nan)

        # Apply replacement method
        if replacement_method == "nan":
            cleaned_df.loc[extreme_mask, field] = np.nan

        elif replacement_method == "interpolate":
            temp_series = cleaned_df[field].copy()
            temp_series[extreme_mask] = np.nan
            cleaned_df[field] = temp_series.interpolate(method="time")

        elif replacement_method in ["mean", "median"]:
            grouped = cleaned_df[field].groupby(pd.Grouper(freq=frequency))
            if replacement_method == "mean":
                replacements = grouped.transform("mean")
            else:
                replacements = grouped.transform("median")
            cleaned_df.loc[extreme_mask, field] = replacements[extreme_mask]

        # Calculate cleaning summary
        cleaning_stats = {
            "field": field,
            "points_removed": extreme_mask.sum(),
            "percent_removed": (extreme_mask.sum() / len(df)) * 100,
            "replacement_method": replacement_method,
        }
        cleaning_summary.append(cleaning_stats)

    return {
        "cleaned_data": cleaned_df,
        "cleaning_summary": pd.DataFrame(cleaning_summary),
        "removed_points": removed_points,
    }


def polar_to_cartesian_dataframe(df, wd_column="WD", dist_column="Dist"):
    """
    Convert polar coordinates from a DataFrame to Cartesian coordinates.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing polar coordinates.
        wd_column (str): Column name for degrees from north.
        dist_column (str): Column name for distance from origin.

    Returns:
        pd.DataFrame: A DataFrame with added 'X' and 'Y' columns.
    """
    # Create copies of the input columns to avoid modifying original data
    wd = df[wd_column].copy()
    dist = df[dist_column].copy()

    # Identify invalid values (-9999 or NaN)
    invalid_mask = (wd == -9999) | (dist == -9999) | wd.isna() | dist.isna()

    # Convert degrees from north to standard polar angle (radians) where valid
    theta_radians = np.radians(90 - wd)

    # Calculate Cartesian coordinates, setting invalid values to NaN
    df[f"X_{dist_column}"] = np.where(
        invalid_mask, np.nan, dist * np.cos(theta_radians)
    )
    df[f"Y_{dist_column}"] = np.where(
        invalid_mask, np.nan, dist * np.sin(theta_radians)
    )

    return df


def aggregate_to_daily_centroid(
    df,
    date_column="Timestamp",
    x_column="X",
    y_column="Y",
    weighted=True,
):
    """
    Aggregate half-hourly coordinate data to daily centroids.

    Parameters:
        df (pd.DataFrame): DataFrame containing timestamp and coordinates.
        date_column (str): Column containing datetime values.
        x_column (str): Column name for X coordinate.
        y_column (str): Column name for Y coordinate.
        weighted (bool): Weighted by ET column or not (default: True).

    Returns:
        pd.DataFrame: Aggregated daily centroids.
    """
    # Define a lambda function to compute the weighted mean:
    wm = lambda x: np.average(x, weights=df.loc[x.index, "ET"])

    # Ensure datetime format
    df[date_column] = pd.to_datetime(df[date_column])

    # Group by date (ignoring time component)
    df["Date"] = df[date_column].dt.date

    # Calculate centroid (mean of X and Y)
    if weighted:

        # Compute weighted average using ET as weights
        daily_centroids = (
            df.groupby("Date")
            .apply(
                lambda g: pd.Series(
                    {
                        x_column: (g[x_column] * g["ET"]).sum() / g["ET"].sum(),
                        y_column: (g[y_column] * g["ET"]).sum() / g["ET"].sum(),
                    }
                )
            )
            .reset_index()
        )
    else:
        daily_centroids = (
            df.groupby("Date").agg({x_column: "mean", y_column: "mean"}).reset_index()
        )
    # Groupby and aggregate with namedAgg [1]:
    return daily_centroids


def compute_Cw(sigma_w, u_star, target=1.25):
    """
    Compute vertical velocity correction factor C_w so that (C_w*sigma_w)/(sqrt(C_w)*u_star) = target.
    Uses Eq. (5) from Paw U et al. (2025): Cw = (target/(sigma_w/u_star))^2.
    If sigma_w/u_star >= target, returns C_w=1 (no correction needed).
    """
    ratio = sigma_w / u_star
    if np.isnan(ratio) or np.isclose(u_star, 0):
        return np.nan
    # Only apply correction if measured ratio < target
    if ratio < target:
        return (target / ratio) ** 2
    else:
        return 1.0


def filter_near_neutral(z_over_L, lower=-0.1, upper=0.0):
    """
    Return a boolean mask (or filtered array) for near-neutral stability: lower < z/L < upper.
    Default uses -0.1 < z/L < 0 as in Paw U et al. (2025):contentReference[oaicite:13]{index=13}.
    """
    z_over_L = np.asarray(z_over_L, dtype=float)
    mask = (z_over_L > lower) & (z_over_L < upper)
    return mask


# Example usage:
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start="2024-01-01", end="2024-01-02", freq="30min")
    df = pd.DataFrame(index=dates)

    # Create multiple columns with gaps
    df["temperature"] = 20 + np.random.randn(len(dates))
    df["humidity"] = 60 + np.random.randn(len(dates))
    df["pressure"] = 1013 + np.random.randn(len(dates))

    # Insert some gaps
    df.loc["2024-01-01 10:00":"2024-01-01 12:00", "temperature"] = -9999
    df.loc["2024-01-01 15:00":"2024-01-01 16:00", "humidity"] = np.nan
    df.loc["2024-01-01 18:00":"2024-01-01 20:00", "pressure"] = -9999

    # Find gaps
    gaps_df = find_gaps(df, ["temperature", "humidity", "pressure"], min_gap_periods=1)

    # Create and show plot
    fig = plot_gaps(gaps_df, "Sample Data Gaps")
    fig.show()
