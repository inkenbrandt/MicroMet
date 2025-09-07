# MicroMet

[![Read the Docs](https://img.shields.io/readthedocs/micromet)](https://micromet.readthedocs.io/en/latest/)
[![PyPI - Version](https://img.shields.io/pypi/v/micromet)](https://pypi.org/project/micromet/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/micromet.svg)](https://anaconda.org/conda-forge/micromet)

A Python toolkit for meteorological data processing.

## Description

MicroMet is a collection of scripts and modules designed to process half-hourly Eddy Covariance data from [Campbell Scientific CR6 dataloggers](https://www.campbellsci.com/cr6) running [EasyFluxDL](https://www.campbellsci.com/easyflux-dl). It is particularly useful for preparing data for submission to the AmeriFlux Data Portal.

The toolkit can handle common data issues such as missing headers and can compile multiple data files into a single, standardized format that conforms to AmeriFlux standards.

## Installation

You can install MicroMet using pip:

```bash
pip install micromet
```

Or via conda-forge:

```bash
conda install -c conda-forge micromet
```

## Usage

Here are some examples of how to use the MicroMet package.

### Reformatting Data

The `Reformatter` class is the main entry point for cleaning and standardizing your data.

```python
from micromet.converter import Reformatter
import pandas as pd

# Assuming you have a DataFrame `df` with your raw data
# and a `data_type` of 'eddy' or 'met'
reformatter = Reformatter()
cleaned_df, report = reformatter.prepare(df, data_type='eddy')
```

### Creating Plots

The `graphs` module provides functions for creating various plots, such as energy balance Sankey diagrams and instrument comparison plots.

```python
from micromet.graphs import energy_sankey
import pandas as pd

# Assuming `df` is a DataFrame with the required energy balance components
fig = energy_sankey(df, date_text="2024-06-19 12:00")
fig.show()
```

## Modules

The `micromet` package is organized into the following modules:

-   `add_header_from_peer`: Tools for fixing files with missing headers by borrowing from similar files.
-   `compare`: Functions for comparing two time series, including linear regression and outlier detection.
-   `converter`: The main module containing the `AmerifluxDataProcessor` and `Reformatter` classes for reading and cleaning data.
-   `file_compile`: Utilities for compiling multiple files into a single directory with duplicate handling.
-   `graphs`: Functions for creating various plots, such as Sankey diagrams and scatter plots.
-   `headers`: Helper functions for working with file headers.
-   `netrad_limits`: Tools for quality assurance of timestamp alignment.
-   `reformatter_vars`: Configuration dictionary for the data reformatter.
-   `station_data_pull`: Classes and functions for downloading data from stations.
-   `tools`: A collection of miscellaneous utility functions.
-   `variable_limits`: A dictionary defining the physical and plausible ranges for variables.

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
