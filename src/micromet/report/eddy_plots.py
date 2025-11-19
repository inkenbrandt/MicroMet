from statsmodels.formula.api import ols
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from scipy.stats import linregress
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plotly.offline import iplot


def create_grouped_boxplot(df, value_col, category_col):
    """
    Creates an interactive Plotly Graph Objects boxplot grouped by a category.

    Args:
        df (pd.DataFrame): The input DataFrame.
        value_col (str): The name of the numeric column to plot on the Y-axis.
        category_col (str): The name of the categorical column to group the boxplots by.

    Returns:
        go.Figure: The Plotly Figure object.
    """
    fig = go.Figure()

    # Get a sorted list of unique categories to iterate over
    unique_categories = sorted(df[category_col].unique())

    for category in unique_categories:
        # Filter the data for the current category
        category_data = df[df[category_col] == category]

        # Create the custom hover text for this subset (using the DataFrame's index)
        # We ensure the index is used, even if it's a DatetimeIndex
        category_hover_text = ["Index: {}".format(i) for i in category_data.index]

        # Add a Box trace for the current category
        fig.add_trace(go.Box(
            # X-axis is the category
            x=[category] * len(category_data),
            # Y-axis is the numeric value
            y=category_data[value_col],
            name=str(category), # Set the trace name to the category
            boxpoints='all',
            # Assign the custom hover text
            text=category_hover_text,
            # Configure the hover label template
            hovertemplate=(
                f'{category_col}: {category}<br>' +
                f'{value_col}: ' + '%{y}<br>' +
                '%{text}<br>' +
                '<extra></extra>'
            )
        ))

    # Layout Customization
    fig.update_layout(
        title=f'Boxplots of {value_col} Grouped by {category_col}',
        xaxis_title=category_col,
        yaxis_title=value_col,
        boxgap=0.2
    )

    return fig

## this is an old versoi of this plot that I used for the May 2025 comparisons!
def ols_plot(x, y,xlabel, ylabel, title):
    '''
    Create a scatterplot between two arrays, `x` and `y`, visualizing their relationship
    along with an Ordinary Least Squares (OLS) regression line and a 1:1 reference line.

    This function calculates the OLS regression line for the given data, plots the
    scattered data points, the regression line with its equation and R-squared value,
    and a 1:1 diagonal line for comparison. It also adds a grid, custom labels, and a title.

    Parameters
    ----------
    x : array-like
        The independent variable data (e.g., predicted values).
    y : array-like
        The dependent variable data (e.g., actual values).
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    title : str
        The title for the plot.

    Returns
    -------
    None
        This function does not return any value; it displays the plot directly.

    Dependencies
    ------------
    - matplotlib.pyplot as plt
    - numpy as np
    - scipy.stats.linregress    '''
    
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())

    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    y_pred = slope * x + intercept

    plt.figure(figsize=(10,5))
    plt.plot(x, y_pred, color='red', label=f'Fit: y = {slope:.2f}x + {intercept:.2f}\nR2={round(r_value**2, 2)}')
    plt.plot(np.arange(min_val,max_val,0.1),np.arange(min_val,max_val,0.1),color='black',linestyle=":",label='1:1 Line')
    plt.scatter(x,y, label='Record', alpha=0.5)
    plt.grid(True)
    plt.xlabel(f'{xlabel}')
    plt.ylabel(f'{ylabel}')
    plt.title(f'{title}')
    plt.legend()


# this plot is a great way to view how residuals in a linear model vary over time
def student_resid_plot(df, var1, var2, title):
    '''
    Generates an interactive scatter plot of studentized residuals from an OLS regression.

    This function performs a simple Ordinary Least Squares (OLS) regression using `var1`
    as the independent variable and `var2` as the dependent variable from the input
    DataFrame `df`. It then calculates the studentized residuals and plots them against
    the DataFrame's index (assumed to be temporal, e.g., 'Date').
    The plot includes horizontal lines indicating a the 1.96 threshold
    and highlights points that exceed these thresholds as outliers.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data for regression.
        The DataFrame's index is used for the x-axis in the plot.
    var1 : str
        The name of the column in `df` to be used as the independent variable in the OLS regression.
    var2 : str
        The name of the column in `df` to be used as the dependent variable in the OLS regression.
    title : str
        The title for the plot.

    Returns
    -------
    None
        This function does not return any value; it displays an interactive Plotly graph directly.

    Dependencies
    ------------
    - statsmodels.formula.api.ols
    - plotly.express as px
    - numpy as np (for np.abs)
    '''
    simple_regression_model = ols(f'{var2} ~ {var1}', data=df).fit()
    stud_res = simple_regression_model.outlier_test()
    outlier_threshold = 1.96

    fig = px.scatter(stud_res,
        x=stud_res.index, 
        y='student_resid',
        hover_data={'Date': stud_res.index.strftime('%Y-%m-%d'),
                    'Studentized Residual': stud_res['student_resid']},
        height=350,
        labels = {
            stud_res.index.name: "Date",
            'student_resid':'Studentized Residual'
        },
        title=f'{title}'
    )

    fig.add_hline(y=outlier_threshold, line_dash="dash", line_color="red",
                annotation_text=f"Outlier Threshold (+{outlier_threshold})",
                annotation_position="top right")
    fig.add_hline(y=-outlier_threshold, line_dash="dash", line_color="red",
                annotation_text=f"Outlier Threshold (-{outlier_threshold})",
                annotation_position="bottom right")

    # Optionally, you can also mark individual outlier points if you want them to stand out
    identified_outliers = stud_res[np.abs(stud_res['student_resid']) > outlier_threshold]
    if not identified_outliers.empty:
        fig.add_scatter(x=identified_outliers.index, y=identified_outliers['student_resid'],
                        mode='markers', name='Outliers', marker=dict(color='red', size=8, symbol='x'))
        
    fig.update_layout(
        margin=dict(
            l=20,  
            r=20,  
            b=20,  
            t=50
        )
    )

    fig.show()


# this is the function I am using for the chapman conference
def comparison_plot (df, var1, var2, title, xlabel, ylabel, output_path, print_plot=True):
    '''
    Generates a scatter plot to compare two variables from a DataFrame, including a linear regression line and a 1:1 reference line.

    This function performs a linear regression on var1 (independent variable) and var2 (dependent variable), and visualizes the relationship. It drops any rows with missing values in these two columns before plotting. The plot includes several key features:

    Data points are displayed as hollow circles with a blue outline.
    A red line shows the best-fit linear regression.
    A black dashed line represents the ideal 1:1 relationship for comparison.
    The legend provides key statistics, including the slope and R-squared value of the linear fit.
    The plot is saved to a file and displayed.

    Parameters
    df : pandas.DataFrame
    The input DataFrame containing the data for the plot.
    var1 : str
    The name of the column in df to be used for the x-axis and linear regression.
    var2 : str
    The name of the column in df to be used for the y-axis and linear regression.
    title : str
    The title for the plot.
    xlabel : str
    The label for the x-axis.
    ylabel : str
    The label for the y-axis.
    output_path: str
    The path for where to export the plot

    Returns
    None
    This function does not return any value; it displays and saves a plot directly.

    Dependencies
    pandas as pd
    numpy as np
    matplotlib.pyplot as plt
    scipy.stats as stats
    '''

    scatterdf = df[[var1, var2]].dropna()

    x = scatterdf[var1]
    y = scatterdf[var2]

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Print the results
    print(f"Slope: {slope}")
    print(f"Intercept: {intercept}")
    print(f"R-value (correlation coefficient): {r_value}")
    print(f"P-value: {p_value}")
    print(f"Standard error of the estimate: {std_err}")



    # Plot the data and the regression line
    plt.scatter(x, y, label='Data points', s=20, facecolors='none', edgecolors='blue')
    plt.plot(x, slope * x + intercept, color='red')

    linear_fit_label = f'Linear fit (Slope={slope:.2f}, R$^{2}$={r_value**2:.2f})'
    plt.plot(x, slope * x + intercept, color='red', label=linear_fit_label)

    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    plt.plot([min_val, max_val], [min_val, max_val], 
            color='black', linestyle='--', label='1:1 Line')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if print_plot==True:
        plt.savefig(output_path)
    plt.show()


def plot_linear_regression_with_color(data, x_col, y_col, color_col, output_path=None, print_plot=False):
    
    """
    Generates a scatter plot with a linear regression line and a 1:1 line for data analysis.

    This function is designed for plotting any three numerical columns from a pandas DataFrame.
    It performs a linear regression between the specified x and y columns and uses a third column
    to color the data points.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing the data to be plotted. It must contain the columns specified
        by `x_col`, `y_col`, and `color_col`.

    x_col : str
        The name of the column for the x-axis, representing the independent variable.

    y_col : str
        The name of the column for the y-axis, representing the dependent variable.

    color_col : str
        The name of the column used to color the scatter plot points, useful for visualizing
        a third variable.

    Returns
    -------
    None
        This function displays a plot using matplotlib.

    Dependencies
    ------------
    - matplotlib.pyplot
    - scipy.stats
    - pandas (assumed for the input 'data' DataFrame)

    The plot includes:
    - Scatter points of `y_col` vs. `x_col`.
    - A colorbar representing `color_col`. The `twilight` colormap is used, which is ideal
      for cyclical data.
    - A linear regression best-fit line with its slope and R-squared value.
    - A 1:1 line for visual comparison.
    - A legend, a grid, and auto-adjusted axis labels based on the input column names.
    """
    # Data Preparation
    scatterdf = data[[x_col, y_col, color_col]].dropna().copy()
    x = scatterdf[x_col]
    y = scatterdf[y_col]

    # Perform linear regression
    slope, intercept, r_value, _, _ = stats.linregress(x, y)

    # Plotting
    scatter = plt.scatter(x, y, label='Data points', s=20,
                          c=scatterdf[color_col],
                          cmap='twilight')

    # Plot the regression line
    linear_fit_label = f'Linear fit (Slope={slope:.2f}, R$^{2}$={r_value**2:.2f})'
    plt.plot(x, slope * x + intercept, color='red', label=linear_fit_label)

    # Plot the 1:1 line
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    plt.plot([min_val, max_val], [min_val, max_val],
             color='black', linestyle='--', label='1:1 Line')

    # Add and label the colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(f'{color_col}')

    # Add labels and other plot elements
    plt.xlabel(f'{x_col}')
    plt.ylabel(f'{y_col}')
    plt.title(f'Scatter Plot of {y_col} vs. {x_col}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if print_plot:
        plt.savefig(output_path)

    plt.show()


import matplotlib.pyplot as plt
from windrose import WindroseAxes

def plot_wind_rose_from_df(df, wd_col, ws_col, title=None, save_path=None):
    """
    Generates and plots a wind rose from a pandas DataFrame.

    This function creates a wind rose plot using the specified wind direction and
    wind speed columns from a DataFrame. The plot displays the frequency of wind
    coming from different directions and the distribution of wind speeds.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the wind data.
    
    wd_col : str
        The name of the column in `df` that contains the wind direction data (in degrees).
    
    ws_col : str
        The name of the column in `df` that contains the wind speed data.
    
    title : str, optional
        The title for the wind rose plot. If not provided, no title will be set.
    
    save_path : str, optional
        The file path to save the plot. If not provided, the plot will not be saved.
        Example: 'my_wind_rose_plot.png'

    Returns
    -------
    None
        This function displays and/or saves a plot.
    """
    # Ensure the required columns exist in the DataFrame
    if wd_col not in df.columns or ws_col not in df.columns:
        raise ValueError(f"DataFrame must contain both '{wd_col}' and '{ws_col}' columns.")

    # Drop any rows with missing data for the specified columns
    df_clean = df.dropna(subset=[wd_col, ws_col]).copy()
    
    # Create the figure and WindroseAxes object
    fig = plt.figure(figsize=(8, 8), dpi=80)
    ax = WindroseAxes.from_ax(fig=fig)
    
    # Plot the wind rose
    ax.bar(df_clean[wd_col], df_clean[ws_col], normed=True, opening=0.8, edgecolor='white')

    # Set the legend and title
    ax.set_legend()
    if title:
        ax.set_title(title)
    
    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path)
    
    # Show the plot
    plt.show()




def plot_interactive_regression_with_color(
    
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    plot_size: int = 500 # New parameter for plot size

) -> None:
    """
    Generates an interactive scatter plot with a linear regression line, a 1:1 line,
    and color-coding using Plotly. Index and variable values appear on hover.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data. DatetimeIndex is automatically handled for hover.
    x_col : str
        The name of the column for the x-axis.
    y_col : str
        The name of the column for the y-axis.
    color_col : str
        The name of the column used to color the scatter plot points.
    """

    # 1. Data Preparation and Cleaning
    # Drop rows with NaN in any of the three required columns
    cols_to_use = [x_col, y_col, color_col]
    scatterdf = df[cols_to_use].dropna().copy()

    # Convert index to a column for hover data
    index_name = df.index.name if df.index.name is not None else 'Index'
    scatterdf = scatterdf.reset_index().rename(columns={'index': index_name})

    x = scatterdf[x_col]
    y = scatterdf[y_col]
    color_data = scatterdf[color_col]

    # 2. Perform linear regression
    if len(x) < 2:
        print("Not enough data points remaining after dropping NaNs to perform regression.")
        return

    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    r_squared = r_value**2
    
    # Calculate regression line points
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept

    # Determine 1:1 line boundaries
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())

    # 3. Create the Plotly Figure

    fig = go.Figure()

    # --- Trace 1: Scatter Points (Colored and Hoverable) ---
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=6, # Increased size slightly to make the outline clearer
            color=color_data,  # This will determine the *outline* color,
            opacity=0.6, # <-- Set transparency (0.0 is fully transparent, 1.0 is fully opaque)
            colorscale='Twilight',
            colorbar=dict(title=color_col),
            showscale=True,
            # ------------------------------------------------------------------
            # ADD/CHANGE THESE LINES:
            symbol='circle-open', # <-- Set the marker shape to an unfilled circle
            line=dict(width=2, color=color_data) # <-- Set the outline color/width
            # ------------------------------------------------------------------
        ),
        name='Data points',
        # Define what shows up on hover
        hovertext=scatterdf[index_name].astype(str),
        hoverinfo='text+x+y',
        customdata=scatterdf[[index_name, color_col]], # Use customdata for better label control
        hovertemplate=(
            f'<b>{index_name}: %{{hovertext}}</b><br>' +
            f'{x_col}: %{{x:.2f}}<br>' +
            f'{y_col}: %{{y:.2f}}<br>' +
            f'{color_col}: %{{customdata[1]}}<extra></extra>' # use customdata for color
        )
    ))

    # --- Trace 2: Linear Regression Line ---
    fig.add_trace(go.Scatter(
        x=x_fit,
        y=y_fit,
        mode='lines',
        line=dict(color='red', width=2),
        name=f'Linear fit (Slope={slope:.2f}, R\u00B2={r_squared:.2f})',
        hoverinfo='skip' # Don't show hover data for the line
    ))

    # --- Trace 3: 1:1 Line ---
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='1:1 Line',
        hoverinfo='skip'
    ))

    # 4. Configure Layout
    fig.update_layout(
        title=f'Scatter Plot of {y_col} vs. {x_col}',
        xaxis_title=x_col,
        yaxis_title=y_col,
        hovermode='closest',
        plot_bgcolor='white',
        # --- KEY CHANGE: Set width and height to the same value ---
        width=plot_size,
        height=plot_size,
        # --------------------------------------------------------
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    # Ensure axis ranges match for the 1:1 line to be square
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')

    fig.show()


def plotlystuff(datasets, colnames, chrttypes=None, datatitles=None, chrttitle='', colors=None,
                two_yaxes=False, axisdesig=None, axislabels=['Levels', 'Barometric Pressure'], opac=None, 
                plot_height=300):
    '''Plots one or more datasets on a shared set of axes
    datasets: list of one or more datasets to plot, must have datetime index
    colnames: list of one or more column names to plot on the y-axis; must be one column name per dataset
    chrttypes: list of types of characters to plot; defaults to line; can include lines and markers (points)
    colors: list of colors to use in plots; defaults to ['#228B22', '#FF1493', '#5acafa', '#663399', '#FF0000']
    two_yaxes: presumably whether data should show up with two axes or one
    axisdesig:uncertain
    axislabels: list of names to for legend to label y-value on each dataset
    opac:list of values for opacity setting of datasets; default is 0.8
    plot_height: integer value for height of plot; default is 300
    '''
    
    if chrttypes is None:
        chrttypes = ['lines'] * len(datasets)

    if opac is None:
        opac = [0.8] * len(datasets)
        
    if datatitles is None:
        datatitles = colnames
    
    if axisdesig is None:
        axisdesig = ['y1'] * len(datasets)
        
    if colors is None:
        if len(datasets) <= 5: 
            colors = ['#228B22', '#FF1493', '#5acafa', '#663399', '#FF0000']
        else:
            colors = []
            for i in range(len(datasets)):
                colors.append('#{:02x}{:02x}{:02x}'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    
    modetypes = ['markers', 'lines+markers', 'lines']
    datum = {}
    
    # Plotting the line charts for the datasets
    for i in range(len(datasets)):
        datum['d' + str(i)] = go.Scatter(
            x=datasets[i].index,
            y=datasets[i][colnames[i]],
            name=datatitles[i],
            line=dict(color=colors[i]),
            mode=chrttypes[i],
            opacity=opac[i],
            yaxis=axisdesig[i]
        )
    
    # Combine the data for plotting
    data = list(datum.values())

    # Calculate dynamic y-axis range
    y_min = min([datasets[i][colnames[i]].min() for i in range(len(datasets))])
    y_max = max([datasets[i][colnames[i]].max() for i in range(len(datasets))])
    
    # Layout definition with adjustments for vertical space and axis range
    layout = dict(
        title=chrttitle,
        xaxis=dict(
            rangeslider=dict(visible=True),
            type='date',
            tickformat='%Y-%m-%d %H:%M'
        ),
        yaxis=dict(
            title=axislabels[0],
            titlefont=dict(color='#1f77b4'),
            tickfont=dict(color='#1f77b4'),
            range=[y_min, y_max]  # Set dynamic y-axis range
        ),
        height=plot_height,  # Increase the height for more vertical space
        margin=dict(t=50, b=50, l=60, r=60)  # Adjust margins
    )
    
    if two_yaxes:
        layout['yaxis2'] = dict(
            title=axislabels[1],
            titlefont=dict(color='#ff7f0e'),
            tickfont=dict(color='#ff7f0e'),
            anchor='x',
            overlaying='y',
            side='right',
            position=0.15
        )

    fig = dict(data=data, layout=layout)
    iplot(fig, filename='well')
    return


def compare_to_sig_strength(df, var, signal_var='H2O_SIG_STRGTH_MIN', cutoff=0.8, scaling_factor=1, sig_plot=False):
    '''
    Create plotlystuff plots to view all data for a variable over time and 
    values for that variable when the signal strength is below the indicated 
    cutoff value.

    Args:
    df (pd.DataFrame): Dataframe with datetime index.
    var (str): Name of variable to plot
    signal_var (str): Name of variable representing signal strength to plot
        Should be either H2O_SIG_STRGTH_MIN or CO2_SIG_STRGTH_MIN
    cutoff (float): Cutoff value to investigate for signal strength
    scaling_factor (int): value to scale the signal_var by so that signal 
        strength and variable of interest can be co-plot
    sig_plot (bolean): If True, will plot second plot showing variable alongside 
        scaled signal strength
    """


    '''
    temp = df.copy()
    sig_name = f'{signal_var}_SCALED'
    temp[sig_name] = temp[signal_var]*scaling_factor
    mask = temp[signal_var]<cutoff
    var_name = f'{var}_BELOW_CUTOFF'
    temp[var_name] = temp[var]
    temp.loc[~mask, var_name] = np.nan
    if sig_plot:
        plotlystuff([temp, temp, temp], [var, var_name, sig_name], chrttitle=f'{var} with {cutoff} cutoff')
    plotlystuff([temp, temp], [var, var_name], chrttitle=f'{var} with {cutoff} cutoff')
    return(temp)
