# MicroMet

[![Read the Docs](https://img.shields.io/readthedocs/micromet)](https://micromet.readthedocs.io/en/latest/)
[![PyPI - Version](https://img.shields.io/pypi/v/micromet)](https://pypi.org/project/micromet/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/micromet.svg)](https://anaconda.org/conda-forge/micromet)

A Python toolkit for meteorological data processing.

## Description

MicroMet is a comprehensive Python toolkit for processing, analyzing, and visualizing micrometeorological data. It is particularly well-suited for handling half-hourly Eddy Covariance data from Campbell Scientific CR6 dataloggers running EasyFluxDL, and for preparing data for submission to the AmeriFlux Data Portal.

The toolkit provides a suite of tools for common data processing tasks, including reading various file formats, reformatting and standardizing data, performing quality assurance checks, and generating insightful plots and reports.

## Features

-   **Data Reading**: Read Campbell Scientific TOA5 and AmeriFlux output files.
-   **Data Reformatting**: A flexible pipeline for cleaning and standardizing data, including timestamp correction, column renaming, and unit conversion.
-   **Quality Assurance**: Tools for applying physical limits to variables, detecting and handling outliers, and assessing timestamp alignment.
-   **Data Visualization**: A range of plotting functions for visualizing data, including time series plots, scatter plots, energy balance Sankey diagrams, and Bland-Altman plots.
-   **Data Reporting**: Utilities for generating reports on data quality and analysis results.
-   **Station Data Management**: Tools for downloading data directly from stations and managing data in a database.

## Installation

You can install MicroMet using pip:

```bash
pip install micromet
```

Or via conda-forge:

```bash
conda install -c conda-forge micromet
```

## Setup for Development

To set up the project for development, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/micromet.git
    cd micromet
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the package in editable mode with development dependencies:**
    ```bash
    pip install -e .[dev]
    ```
4.  **Run the tests:**
    ```bash
    pytest
    ```

## Usage

Here are some examples of how to use the MicroMet package.

### Reading Data

The `AmerifluxDataProcessor` class can be used to read AmeriFlux-style data files.

```python
from micromet.reader import AmerifluxDataProcessor

processor = AmerifluxDataProcessor()
df = processor.to_dataframe("path/to/your/data.dat")
```

### Reformatting Data

The `Reformatter` class is the main entry point for cleaning and standardizing your data.

```python
from micromet.format.reformatter import Reformatter
import pandas as pd

# Assuming you have a DataFrame `df` with your raw data
# and a `data_type` of 'eddy' or 'met'
reformatter = Reformatter()
cleaned_df, report = reformatter.prepare(df, data_type='eddy')
```

### Generating Reports and Plots

The `report` module provides tools for generating various plots and reports.

#### Energy Balance Sankey Diagram

```python
from micromet.report.graphs import energy_sankey
import pandas as pd

# Assuming `df` is a DataFrame with the required energy balance components
fig = energy_sankey(df, date_text="2024-06-19 12:00")
fig.show()
```

#### Instrument Comparison Scatter Plot

```python
from micromet.report.graphs import scatterplot_instrument_comparison

# Assuming `edmet` is a DataFrame with instrument data and `compare_dict`
# defines the instruments to compare.
slope, intercept, r_squared, p_value, std_err, fig, ax = scatterplot_instrument_comparison(
    edmet, compare_dict, station="MyStation"
)
```

## Modules

The `micromet` package is organized into the following modules:

-   `reader`: Contains the `AmerifluxDataProcessor` for reading data files.
-   `format`: A subpackage with modules for data formatting, including:
    -   `reformatter`: The main `Reformatter` class for cleaning and standardizing data.
    -   `transformers`: A collection of data transformation functions.
    -   `add_header_from_peer`: Tools for fixing files with missing headers.
    -   `compare`: Functions for comparing two time series.
    -   `file_compile`: Utilities for compiling multiple files.
    -   `headers`: Helper functions for working with file headers.
-   `qaqc`: A subpackage for quality assurance and control, including:
    -   `netrad_limits`: Tools for quality assurance of timestamp alignment.
    -   `variable_limits`: A dictionary defining physical and plausible ranges for variables.
-   `report`: A subpackage for generating reports and plots, with:
    -   `graphs`: Functions for creating various plots.
    -   `tools`: A collection of utility functions for analysis and reporting.
-   `station_data_pull`: Classes for downloading and processing data from stations.
-   `station_info`: Configuration data for stations.
-   `utils`: A collection of miscellaneous utility functions.

## Contributing

Contributions are welcome! If you would like to contribute to the project, please follow these steps:

1.  Fork the repository on GitHub.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with a clear and descriptive message.
4.  Push your changes to your fork.
5.  Create a pull request to the main repository.

Please ensure that your code follows the existing style and that you add or update tests as appropriate.

## Documentation

For more detailed information, the full documentation can be found on [Read the Docs](https://micromet.readthedocs.io/en/latest/).
