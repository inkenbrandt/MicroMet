import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import logging


def logger_check(logger: logging.Logger | None) -> logging.Logger:
    """
    Initialize and return a logger instance if none is provided.

    This function checks if a logger object is provided. If not, it
    creates a new logger with a default warning level and a stream
    handler that outputs to the console.

    Parameters
    ----------
    logger : logging.Logger or None
        An existing logger instance. If None, a new logger is created.

    Returns
    -------
    logging.Logger
        A configured logger instance.
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
    Create a Sankey diagram of energy balance for a specific time.

    This function generates a Sankey diagram to visualize the flow of
    energy components in a system. It is typically used in
    meteorological or environmental studies.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with a DatetimeIndex and columns for energy
        components: 'SW_IN', 'LW_IN', 'SW_OUT', 'LW_OUT', 'NETRAD',
        'G', 'LE', 'H'.
    date_text : str, optional
        The date and time for which to plot the energy balance, in a
        format recognized by `pd.to_datetime`. Defaults to
        "2024-06-19 12:00".
    logger : logging.Logger, optional
        A logger for outputting debug information. If None, a default
        logger is created.

    Returns
    -------
    go.Figure
        A Plotly Figure object containing the Sankey diagram.

    Notes
    -----
    The energy balance ratio (EBR) is calculated as:
    (H + LE) / (NETRAD - G)
    The residual term is computed as:
    NETRAD - (G + H + LE)
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
    Generate a scatter plot comparing two instrument measurements.

    This function creates a scatter plot to compare measurements from two
    instruments, including a linear regression fit and a 1:1 reference
    line.

    Parameters
    ----------
    edmet : pd.DataFrame
        A DataFrame with a DatetimeIndex containing the measurement data.
    compare_dict : dict
        A dictionary mapping the instrument column names to their
        metadata (variable, label, units).
    station : str
        The identifier for the station, used in the plot title.
    logger : logging.Logger, optional
        A logger for outputting regression statistics.

    Returns
    -------
    slope : float
        The slope of the linear regression line.
    intercept : float
        The intercept of the linear regression line.
    r_squared : float
        The R-squared value of the regression.
    p_value : float
        The p-value of the regression.
    std_err : float
        The standard error of the regression estimate.
    fig : plt.Figure
        The matplotlib Figure object.
    ax : plt.Axes
        The matplotlib Axes object.
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
    Calculate the Mean Squared Error (MSE) between two series.

    Parameters
    ----------
    series1 : pd.Series
        The first data series.
    series2 : pd.Series
        The second data series.

    Returns
    -------
    float
        The Mean Squared Error between the two series.

    Raises
    ------
    ValueError
        If the input series are not of the same length.
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

    This plot shows the difference between two measurements against
    their mean, which is useful for assessing the agreement between
    two measurement methods.

    Parameters
    ----------
    m1 : array_like
        A 1-D array of measurements.
    m2 : array_like
        A 1-D array of measurements.
    sd_limit : float, optional
        The number of standard deviations to use for the limits of
        agreement. Defaults to 1.96 (for 95% confidence).
    ax : plt.Axes, optional
        An existing matplotlib Axes to draw the plot on. If None, a
        new figure and axes are created.
    scatter_kwds : dict, optional
        Keyword arguments for the scatter plot.
    mean_line_kwds : dict, optional
        Keyword arguments for the mean difference line.
    limit_lines_kwds : dict, optional
        Keyword arguments for the limits of agreement lines.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object.

    References
    ----------
    Bland, J. M., & Altman, D. G. (1986). Statistical methods for
    assessing agreement between two methods of clinical measurement.
    The lancet, 327(8476), 307-310.
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
    Create a Bland-Altman plot to assess agreement between instruments.

    This function generates a Bland-Altman plot to visualize the
    agreement between two instruments, including the bias and limits
    of agreement.

    Parameters
    ----------
    edmet : pd.DataFrame
        A DataFrame with a DatetimeIndex containing measurement data.
    compare_dict : dict
        A dictionary mapping instrument column names to their metadata.
    station : str
        The identifier for the station, used in the plot title.
    alpha : float, optional
        The transparency level for the plot elements. Defaults to 0.5.
    logger : logging.Logger, optional
        A logger for outputting statistics.

    Returns
    -------
    f : plt.Figure
        The matplotlib Figure object.
    ax : plt.Axes
        The matplotlib Axes object.
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
    Plot a time series for a specific station and variable over a date range.

    This function filters a DataFrame by station and date range, and then
    plots the selected variable over time.

    Parameters
    ----------
    input_df : pd.DataFrame
        A DataFrame with a MultiIndex (station, timestamp).
    selected_station : str
        The identifier of the station to plot.
    selected_field : str
        The name of the column (variable) to plot.
    start_date : str or pd.Timestamp
        The start date of the time range.
    end_date : str or pd.Timestamp
        The end date of the time range.

    Returns
    -------
    None
        This function displays a plot but does not return any objects.
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
    Save the current matplotlib figure to a file.

    This function is intended to be used as a callback for an
    interactive widget, such as a button in a Jupyter notebook.

    Parameters
    ----------
    b : object
        The triggering widget event (not used in the function).

    Returns
    -------
    None
    """
    # This line saves the plot as a .png file. Change it to .pdf to save as pdf.
    fig.savefig("plot.png")
