import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import logging


def logger_check(logger: logging.Logger | None) -> logging.Logger:
    """
    Check if a logger is provided, and if not, create one.

    Args:
        logger: Logger to check

    Returns:
        Logger to use
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.WARNING)

        # Create console handler and set level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(ch)

    return logger


def energy_sankey(df, date_text="2024-06-19 12:00", logger: logging.Logger = None):
    """
    Create a Sankey diagram representing the energy balance for a specific date and time.

    This function generates a Sankey diagram to visualize the flow of energy in a system,
    typically used in meteorological or environmental studies. It calculates various
    energy balance components and creates a diagram showing their relationships.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame with time series data and the following columns:
        'SW_IN', 'LW_IN', 'SW_OUT', 'LW_OUT', 'NETRAD', 'G', 'LE', 'H'.
        The index must be a pandas.DatetimeIndex.
    date_text : str, optional
        Date and time to extract from the DataFrame (default is "2024-06-19 12:00").

    Returns
    -------
    plotly.graph_objs._figure.Figure
        A Plotly Figure object containing the Sankey diagram.

    Notes
    -----
    - The energy balance ratio (EBR) is calculated as (H + LE) / (NETRAD - G).
    - The residual term is computed as: NETRAD - (G + H + LE).

    Energy Balance Components
    -------------------------
    - SW_IN: Incoming Shortwave Radiation
    - LW_IN: Incoming Longwave Radiation
    - SW_OUT: Outgoing Shortwave Radiation
    - LW_OUT: Outgoing Longwave Radiation
    - NETRAD: Net Radiation
    - G: Ground Heat Flux
    - LE: Latent Heat
    - H: Sensible Heat

    Examples
    --------
    >>> import pandas as pd
    >>> fig = energy_sankey(df, "2024-06-19 12:00")
    >>> fig.show()

    See Also
    --------
    plotly.graph_objs.Sankey : For creating Sankey diagrams with Plotly.
    """
    select_date = pd.to_datetime(date_text)
    swi = df.loc[select_date, "SW_IN"]
    lwi = df.loc[select_date, "LW_IN"]
    swo = df.loc[select_date, "SW_OUT"]
    lwo = df.loc[select_date, "LW_OUT"]
    nr = df.loc[select_date, "NETRAD"]
    shf = df.loc[select_date, "G"]
    le = df.loc[select_date, "LE"]
    h = df.loc[select_date, "H"]

    # Define the energy balance terms and their indices
    labels = [
        "Incoming Shortwave Radiation",
        "Incoming Longwave Radiation",
        "Total Incoming Radiation",
        "Outgoing Shortwave Radiation",
        "Outgoing Longwave Radiation",
        "Net Radiation",
        "Ground Heat Flux",
        "Sensible Heat",
        "Latent Heat",
        "Residual",
    ]

    logger = logger_check(logger)
    logger.debug(f"Sensible Heat: {h}")
    rem = nr - (shf + h + le)

    ebr = (h + le) / (nr - shf)

    # Define the source and target nodes and the corresponding values for the energy flow
    source = [0, 1, 2, 2, 2, 5, 5, 5, 5]  # Indices of the source nodes
    target = [2, 2, 5, 3, 4, 6, 7, 8, 9]  # Indices of the target nodes

    # Define the source and target nodes and the corresponding values for the energy flow
    # source = [0, 1, 2, 2, 2, 5, 5, 5, 5]  # Indices of the source nodes
    # target = [2, 2, 5, 3, 4, 6, 7, 8, 9]  # Indices of the target nodes
    values = [lwi, swi, nr, swo, lwo, shf, h, le, rem]  # Values of the energy flow

    # Create the Sankey diagram
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=values,
                ),
            )
        ]
    )

    # Update layout and title
    fig.update_layout(
        title_text=f"Energy Balance {ebr:0.2f} on {select_date:%Y-%m-%d}", font_size=10
    )

    # Show the figure
    # fig.show()
    return fig


def scatterplot_instrument_comparison(edmet, compare_dict, station, logger: logging.Logger = None):
    """
    Generate a scatterplot comparing two instrument measurements with regression analysis.

    This function compares two instruments specified in `compare_dict` by plotting
    their 1-hour mean values, interpolating missing data, and performing linear regression.
    It includes a 1:1 reference line and displays key regression statistics.

    Parameters
    ----------
    edmet : pandas.DataFrame
        DataFrame containing environmental measurements, indexed by datetime.
    compare_dict : dict
        Dictionary where keys are instrument column names and values are tuples of
        metadata in the format (variable, instrument label, units).
    station : str
        Identifier for the station to annotate the plot title.

    Returns
    -------
    slope : float
        Slope of the linear regression line.
    intercept : float
        Intercept of the linear regression line.
    r_squared : float
        Coefficient of determination ($R^2$) from the regression.
    p_value : float
        Two-sided p-value for a hypothesis test whose null hypothesis is that the slope is zero.
    std_err : float
        Standard error of the estimated slope.
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The matplotlib axes object containing the plot.

    Notes
    -----
    - Missing values (-9999) are replaced with NaN before processing.
    - The data is resampled to hourly frequency and linearly interpolated.
    - The regression line and 1:1 line are plotted for visual comparison.
    - X and Y axes are labeled using metadata from `compare_dict`.
    - The function both displays the plot and prints regression statistics.
    """
    # Compare two instruments
    instruments = list(compare_dict.keys())
    df = edmet[instruments].replace(-9999, np.nan).dropna()
    df = df.resample("1h").mean().interpolate(method="linear")
    df = df.dropna()

    x = df[instruments[0]]
    y = df[instruments[1]]

    xinfo = compare_dict[instruments[0]]
    yinfo = compare_dict[instruments[1]]

    # one to one line
    xline = np.arange(df.min().min(), df.max().max(), 0.1)
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    # Predict y values
    y_pred = slope * x + intercept
    # R-squared
    r_squared = r_value**2

    fig, ax = plt.subplots(figsize=(10, 8))
    # Plot
    ax.scatter(x, y, alpha=0.5, s=1, label="Data points")
    ax.set_title(f"{xinfo[1]} Comparison: {station}")
    ax.plot(xline, xline, label="1:1 line", color="green", linestyle="--")
    ax.plot(
        x,
        y_pred,
        color="red",
        label=f"Fit: y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_squared:.3f}",
    )
    plt.legend()
    plt.grid(True)

    ax.set_xlabel(f"{xinfo[0]} {xinfo[1]} ({xinfo[2]})")
    ax.set_ylabel(f"{yinfo[0]} {yinfo[1]} ({yinfo[2]})")

    plt.show()

    # Log results
    logger = logger_check(logger)
    logger.info(f"Slope: {slope:.3f}")
    logger.info(f"Intercept: {intercept:.3f}")
    logger.info(f"R-squared: {r_squared:.3f}")
    return slope, intercept, r_squared, p_value, std_err, fig, ax


def mean_squared_error(series1: pd.Series, series2: pd.Series) -> float:
    """
    Calculate the mean squared error between two pandas Series.

    Parameters
    ----------
    series1 : pd.Series
        The first data series.
    series2 : pd.Series
        The second data series.

    Returns
    -------
    float
        The mean squared error between the two series.

    Raises
    ------
    ValueError
        If the input Series are not the same length.
    """
    if len(series1) != len(series2):
        raise ValueError("Input Series must be of the same length.")

    return np.mean((series1 - series2) ** 2)


def mean_diff_plot(
    m1,
    m2,
    sd_limit=1.96,
    ax=None,
    scatter_kwds=None,
    mean_line_kwds=None,
    limit_lines_kwds=None,
):
    """
    Construct a Tukey/Bland-Altman Mean Difference Plot.

    Tukey's Mean Difference Plot (also known as a Bland-Altman plot) is a
    graphical method to analyze the differences between two methods of
    measurement. The mean of the measures is plotted against their difference.

    For more information see
    https://en.wikipedia.org/wiki/Bland-Altman_plot

    Parameters
    ----------
    m1 : array_like
        A 1-d array.
    m2 : array_like
        A 1-d array.
    sd_limit : float
        The limit of agreements expressed in terms of the standard deviation of
        the differences. If `md` is the mean of the differences, and `sd` is
        the standard deviation of those differences, then the limits of
        agreement that will be plotted are md +/- sd_limit * sd.
        The default of 1.96 will produce 95% confidence intervals for the means
        of the differences. If sd_limit = 0, no limits will be plotted, and
        the ylimit of the plot defaults to 3 standard deviations on either
        side of the mean.
    ax : AxesSubplot
        If `ax` is None, then a figure is created. If an axis instance is
        given, the mean difference plot is drawn on the axis.
    scatter_kwds : dict
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.scatter plotting method
    mean_line_kwds : dict
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method
    limit_lines_kwds : dict
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method

    Returns
    -------
    Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    References
    ----------
    Bland JM, Altman DG (1986). "Statistical methods for assessing agreement
    between two methods of clinical measurement"

    Examples
    --------

    Load relevant libraries.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    Making a mean difference plot.

    >>> # Seed the random number generator.
    >>> # This ensures that the results below are reproducible.
    >>> np.random.seed(9999)
    >>> m1 = np.random.random(20)
    >>> m2 = np.random.random(20)
    >>> f, ax = plt.subplots(1, figsize = (8,5))
    >>> mean_diff_plot(m1, m2, ax = ax)
    >>> plt.show()

    .. plot:: plots/graphics-mean_diff_plot.py
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if len(m1) != len(m2):
        raise ValueError("m1 does not have the same length as m2.")
    if sd_limit < 0:
        raise ValueError(f"sd_limit ({sd_limit}) is less than 0.")

    means = np.mean([m1, m2], axis=0)
    diffs = m1 - m2
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)

    scatter_kwds = scatter_kwds or {}
    if "s" not in scatter_kwds:
        scatter_kwds["s"] = 20
    mean_line_kwds = mean_line_kwds or {}
    limit_lines_kwds = limit_lines_kwds or {}
    for kwds in [mean_line_kwds, limit_lines_kwds]:
        if "color" not in kwds:
            kwds["color"] = "gray"
        if "linewidth" not in kwds:
            kwds["linewidth"] = 1
    if "linestyle" not in mean_line_kwds:
        kwds["linestyle"] = "--"
    if "linestyle" not in limit_lines_kwds:
        kwds["linestyle"] = ":"

    ax.scatter(means, diffs, **scatter_kwds)  # Plot the means against the diffs.
    ax.axhline(mean_diff, **mean_line_kwds)  # draw mean line.

    # Annotate mean line with mean difference.
    ax.annotate(
        f"mean diff:\n{np.round(mean_diff, 2)}",
        xy=(0.99, 0.5),
        horizontalalignment="right",
        verticalalignment="center",
        fontsize=14,
        xycoords="axes fraction",
    )

    if sd_limit > 0:
        half_ylim = (1.5 * sd_limit) * std_diff
        ax.set_ylim(mean_diff - half_ylim, mean_diff + half_ylim)
        limit_of_agreement = sd_limit * std_diff
        lower = mean_diff - limit_of_agreement
        upper = mean_diff + limit_of_agreement
        for j, lim in enumerate([lower, upper]):
            ax.axhline(lim, **limit_lines_kwds)
        ax.annotate(
            f"-{sd_limit} SD: {lower:0.2g}",
            xy=(0.99, 0.07),
            horizontalalignment="right",
            verticalalignment="bottom",
            fontsize=14,
            xycoords="axes fraction",
        )
        ax.annotate(
            f"+{sd_limit} SD: {upper:0.2g}",
            xy=(0.99, 0.92),
            horizontalalignment="right",
            fontsize=14,
            xycoords="axes fraction",
        )

    elif sd_limit == 0:
        half_ylim = 3 * std_diff
        ax.set_ylim(mean_diff - half_ylim, mean_diff + half_ylim)

    ax.set_ylabel("Difference", fontsize=15)
    ax.set_xlabel("Means", fontsize=15)
    ax.tick_params(labelsize=13)
    fig.tight_layout()
    return fig


def bland_alt_plot(edmet, compare_dict, station, alpha=0.5, logger: logging.Logger = None):
    """
    Create a Bland–Altman plot to assess agreement between two instruments.

    This function compares two instruments specified in `compare_dict` by plotting the
    mean of their measurements against their difference. It reports the root mean square error
    (RMSE), bias, and spread, and visualizes the limits of agreement.

    Parameters
    ----------
    edmet : pandas.DataFrame
        DataFrame containing environmental measurements, indexed by datetime.
    compare_dict : dict
        Dictionary where keys are instrument column names and values are tuples of
        metadata in the format (variable, instrument label, units).
    station : str
        Name or identifier for the measurement station, used in plot titles.
    alpha : float, optional
        Transparency of the plot elements. Default is 0.5.

    Returns
    -------
    f : matplotlib.figure.Figure
        The matplotlib figure object containing the plot.
    ax : matplotlib.axes.Axes
        The matplotlib axes object for the Bland–Altman plot.

    Notes
    -----
    - Missing values (-9999) are replaced with NaN before processing.
    - Data is resampled to hourly frequency and linearly interpolated.
    - Bias is calculated as the mean difference between instruments.
    - Spread is the standard deviation of the difference.
    - Limits of agreement are shown at ±1.96 × standard deviation.
    - The `mean_diff_plot` utility is used for plotting.
    """
    # Compare two instruments
    instruments = list(compare_dict.keys())
    df = edmet[instruments].replace(-9999, np.nan).dropna()
    df = df.resample("1h").mean().interpolate(method="linear")
    df = df.dropna()

    x = df[instruments[0]]
    y = df[instruments[1]]
    rmse = np.sqrt(mean_squared_error(x, y))
    logger = logger_check(logger)
    logger.info(f"RMSE: {rmse:.3f}")

    mean_vals = df[instruments].mean(axis=1)
    diff_vals = x - y
    bias = diff_vals.mean()
    spread = diff_vals.std()
    logger.info(f"Bias = {bias:.3f}, Spread = {spread:.3f}")
    top = diff_vals.mean() + 1.96 * diff_vals.std()
    bottom = diff_vals.mean() - 1.96 * diff_vals.std()

    f, ax = plt.subplots(1, figsize=(8, 5), alpha=alpha)
    mean_diff_plot(x, y, ax=ax)
    ax.text(
        mean_vals.mean(),
        top,
        s=compare_dict[instruments[0]][0],
        verticalalignment="top",
        fontweight="bold",
    )
    ax.text(
        mean_vals.mean(),
        bottom,
        s=compare_dict[instruments[1]][0],
        verticalalignment="bottom",
        fontweight="bold",
    )
    ax.set_title(
        f"{compare_dict[instruments[0]][0]} vs {compare_dict[instruments[1]][0]} at {station}"
    )
    ax.set_xlabel(
        f"Mean {compare_dict[instruments[0]][1]} ({compare_dict[instruments[0]][2]})"
    )
    ax.set_ylabel(
        f"Difference ({compare_dict[instruments[0]][2]})\n({compare_dict[instruments[0]][0]} - {compare_dict[instruments[1]][0]})",
        fontsize=10,
    )

    return f, ax


# Example of filtering by date range
def plot_timeseries_daterange(
    input_df, selected_station, selected_field, start_date, end_date
):
    """
    Plot a time series for a specific station and variable within a date range.

    This function extracts and visualizes a single time series from a multi-station
    DataFrame, limited to a specified date range and filtered by station ID and
    data field. It handles missing values and generates a time series plot.

    Parameters
    ----------
    input_df : pandas.DataFrame
        DataFrame containing a multi-index (station, timestamp) with time series data.
    selected_station : str
        Station identifier to select from the DataFrame.
    selected_field : str
        Column name of the variable to plot (e.g., temperature, humidity).
    start_date : str or pandas.Timestamp
        Start date of the time range to include in the plot.
    end_date : str or pandas.Timestamp
        End date of the time range to include in the plot.

    Returns
    -------
    None
        Displays a matplotlib plot but does not return any value.

    Notes
    -----
    - Assumes `input_df` is indexed by station and datetime (MultiIndex).
    - Missing values equal to -9999 are replaced with NaN prior to plotting.
    - The function sets global variables `fig` and `ax`, which may overwrite
      other plots in an interactive session.
    """
    global fig, ax
    # ax.clear()
    fig, ax = plt.subplots(figsize=(10, 8))

    # Filter data by date range
    filtered_df = input_df.loc[selected_station].loc[start_date:end_date]
    filtered_df = filtered_df.loc[:, selected_field].replace(-9999, np.nan)

    # Plot each selected category
    ax.plot(filtered_df.index, filtered_df, label=selected_station, linewidth=2)

    plt.title(f"{selected_station} {selected_field}\n{start_date} to {end_date}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def save_plot(b):
    """
    Saves plot for an interactive notebook button function
    """
    # This line saves the plot as a .png file. Change it to .pdf to save as pdf.
    fig.savefig("plot.png")
