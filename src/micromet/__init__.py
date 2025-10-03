"""
Micromet: A package for processing and analyzing micrometeorological data.

This package provides a collection of tools for reading, reformatting,
performing quality control, and generating reports from micrometeorological
and flux data, particularly from AmeriFlux-style data sources.

The main components of the package are:
- `AmerifluxDataProcessor`: For reading and parsing data files.
- `Reformatter`: For cleaning and standardizing data.
- `tools`: A collection of utility functions for analysis.
- `graphs`: For creating various plots and visualizations.
- `StationDataDownloader`: For downloading data from stations.
- `StationDataProcessor`: For processing and managing station data.
"""
from .reader import AmerifluxDataProcessor
from .format.reformatter import Reformatter
from .report import tools
from .report import graphs
from .station_data_pull import StationDataDownloader, StationDataProcessor
from .format import headers
from .format import reformatter_vars
from .qaqc import variable_limits, netrad_limits
from .format import add_header_from_peer
from .format import compare
from .report import validate
from .report import gap_summary

__version__ = "0.3.1"

__all__ = [
    "AmerifluxDataProcessor",
    "Reformatter",
    "tools",
    "graphs",
    "StationDataDownloader",
    "StationDataProcessor",
    "headers",
    "reformatter_vars",
    "variable_limits",
    "add_header_from_peer",
    "compare",
]
