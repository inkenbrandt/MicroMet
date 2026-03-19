Getting Started with Micromet
=============================

Micromet is a Python package for processing micrometeorological data from environmental monitoring stations. It provides utilities for formatting, validating, and analyzing data from sources such as AmeriFlux and Campbell Scientific systems.

This guide walks you through installation, basic usage, and how to start working with your own data.

Installation
------------

Micromet is available on PyPI and can be installed using pip:

.. code-block:: bash

    pip install micromet

Alternatively, if you want to use the latest development version, you can install it directly from GitHub:

.. code-block:: bash

    pip install git+https://github.com/inkenbrandt/MicroMet.git

MicroMet is also available as a conda package. You can install it using the following command:

.. code-block:: bash

    conda install -c conda-forge micromet


Micromet can be installed from source. First, clone the repository:

.. code-block:: bash

    git clone https://github.com/inkenbrandt/MicroMet.git
    cd MicroMet

It's recommended to use a virtual environment:

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

Install the package with optional documentation and development dependencies:

.. code-block:: bash

    pip install -e .[docs,test]

Usage Overview
--------------

Micromet consists of modular tools for reformatting and analyzing micrometeorological data. Here's a basic example to get started:

.. code-block:: python

    from micromet.format.reformatter import Reformatter

    # Load your raw data into a pandas DataFrame
    import pandas as pd
    df = pd.read_csv("path/to/your/data.csv")

    # Create a Reformatter instance and process the data
    ref = Reformatter()
    cleaned_df, report = ref.prepare(df, data_type='eddy')

    # Now cleaned_df contains the cleaned and normalized data
    print(cleaned_df.head())

Modules
-------

The key subpackages and modules in Micromet are:

- ``format`` — Data formatting subpackage:

  - ``reformatter`` — Main :class:`~micromet.format.reformatter.Reformatter` class for cleaning and standardizing data
  - ``file_compile`` — Utilities for compiling multiple raw ``.dat`` files into a single dataset
  - ``merge`` — Functions for merging eddy covariance and met data streams
  - ``compare`` — Instrument comparison and cross-correlation functions
  - ``headers`` — Utilities for detecting and applying missing headers across files
  - ``transformers`` — Data transformation submodule (``columns``, ``timestamps``, ``corrections``, ``validation``, ``cleanup``, ``interval_updates``)

- ``qaqc`` — Quality assurance and control subpackage:

  - ``variable_limits`` — Physical and plausible range definitions for all variables
  - ``netrad_limits`` — Net radiation QA/QC and timestamp alignment
  - ``data_cleaning`` — QC flag application and data cleaning

- ``report`` — Reporting and visualization subpackage:

  - ``graphs`` — Plotting functions (energy Sankey diagrams, instrument comparison scatter plots)
  - ``tools`` — Utility functions for analysis (irrigation event detection, gap finding)
  - ``validate`` — Data validation, lag detection, and sensor intercomparison
  - ``fix_g_values`` — Soil heat flux storage corrections
  - ``recalculate_albedo`` — Albedo recalculation utilities
  - ``gap_summary`` — Gap analysis and reporting
  - ``eddy_plots`` — Eddy covariance diagnostic plots
  - ``easyflux_footprint`` — Flux footprint analysis
  - ``alfalfa_growth`` — Alfalfa height modeling from growing degree days

- ``reader`` — :class:`~micromet.reader.AmerifluxDataProcessor` for reading AmeriFlux and TOA5 data files
- ``station_data_pull`` — :class:`~micromet.station_data_pull.StationDataDownloader` and :class:`~micromet.station_data_pull.StationDataProcessor` for managing station data

.. note::

   The ``notebooks/`` directory contains worked examples and is not part of the core API. See :doc:`notebooks` for a guide to the available notebooks.

Contributing
------------

We welcome contributions! If you have suggestions, bug reports, or would like to add features:

1. Fork the repository
2. Create a new branch
3. Submit a pull request

Please make sure to add unit tests for new functionality and follow PEP8 standards.

Further Reading
---------------

- :doc:`modules`
- :doc:`data_processing`
- :doc:`flux_workflow_summary`
- `Micromet on GitHub <https://github.com/inkenbrandt/MicroMet>`_
- `Full documentation on Read the Docs <https://micromet.readthedocs.io/en/latest/>`_
