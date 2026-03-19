Example Notebooks
=================

The ``docs/notebooks/`` directory contains Jupyter Notebooks that demonstrate the MicroMet processing workflow and package capabilities. Each notebook corresponds to a stage of the end-to-end pipeline described in :doc:`flux_workflow_summary`.

.. note::

   Notebooks are provided for reference and are not re-executed during documentation builds. Data files required by some notebooks (large CSVs, station ``.dat`` files) are not included in the repository.

Processing Workflow Notebooks
------------------------------

These notebooks implement the numbered processing steps for a flux station (see :doc:`flux_processing_workflow` for full technical details).

.. toctree::
   :maxdepth: 1

   notebooks/cs_files
   notebooks/Appending Data From Dataloggers
   notebooks/netrad_limits_getting_started
   notebooks/Sensor Comparisons
   notebooks/DL_forAmeriFluxOutputOnly
   notebooks/footprint_recalculation

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Notebook
     - Description
   * - ``cs_files.ipynb``
     - Reads and compiles Campbell Scientific ``.dat`` files; demonstrates ``file_compile`` and ``Reformatter.preprocess()`` for CS-format eddy data (workflow Step 1).
   * - ``Appending Data From Dataloggers.ipynb``
     - Merges data downloaded directly from dataloggers with compiled station archives; covers gap-filling and timestamp alignment (workflow Steps 1–2).
   * - ``netrad_limits_getting_started.ipynb``
     - Applies net radiation QA/QC and physical limit checks using ``qaqc.netrad_limits`` and ``Reformatter.finalize()`` (workflow Step 3).
   * - ``Sensor Comparisons.ipynb``
     - Side-by-side instrument intercomparison using ``report.validate`` and ``report.graphs.scatterplot_instrument_comparison()`` (workflow Step 3a).
   * - ``DL_forAmeriFluxOutputOnly.ipynb``
     - Exports a quality-controlled dataset to AmeriFlux-formatted CSV: signal-strength filtering, column dropping, timestamp formatting, and −9999 fill (workflow Step 4).
   * - ``footprint_recalculation.ipynb``
     - Estimates and maps the flux footprint from EasyFlux outputs using ``report.easyflux_footprint`` (supplemental analysis).

Analysis and Utility Notebooks
--------------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Notebook
     - Description
   * - ``converter_tutorial (2).ipynb``
     - Introductory tutorial for the ``Reformatter`` class: loading raw data, running ``prepare()``, and inspecting the QC report.
   * - ``graphs_usage_example.ipynb``
     - Examples of all major plotting functions in ``report.graphs``, including energy Sankey diagrams and instrument comparison scatter plots.
   * - ``fix_headers.ipynb``
     - Demonstrates ``format.headers`` utilities for detecting and applying missing column headers across a set of ``.dat`` files.
   * - ``Pull DB Data and Impute.ipynb``
     - Retrieves gap-filled reference data from the UGS database and performs meteorological variable imputation.
   * - ``alfalfa_height.ipynb``
     - Simulates alfalfa canopy height using growing degree days via ``report.alfalfa_growth.simulate_alfalfa_height_multi_field()``, calibrated with experimental data from Wellington, UT (January 2023 – December 2024).
