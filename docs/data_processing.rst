Data Processing and Management
==============================

This section explains how Utah Flux Network data are processed and managed using Python.

For a complete end-to-end walkthrough of the processing pipeline, see the :doc:`flux_workflow_summary` and :doc:`flux_processing_workflow` pages. The :doc:`notebooks` page describes worked example notebooks that illustrate each stage.

Modules
-------

**format subpackage**

**1. ``format/reformatter.py``**

- ``AmerifluxDataProcessor``: Parses AmeriFlux TOA5 CSVs and Campbell Scientific ``.dat`` files into pandas DataFrames.
- ``Reformatter``: Cleans, standardizes, and resamples data through three pipeline stages:

  - ``preprocess()``: Strips suffixes, renames columns, validates timestamps, and subsets to target interval.
  - ``finalize()``: Applies physical limits, replaces out-of-range values, and generates a QC report.
  - ``prepare()``: Convenience wrapper combining preprocess and finalize.

**2. ``format/file_compile.py``**

- ``compile_files()``: Compiles multiple raw ``.dat`` files from a directory into a single concatenated DataFrame, handling overlapping records and header mismatches.

**3. ``format/merge.py``**

- Functions for merging eddy covariance and meteorological data streams, resolving column conflicts, and applying suffix conventions (``_EDDY`` / ``_MET``).

**4. ``format/compare.py``**

- Cross-correlation and lag-detection functions used to align sensor streams with systematic time offsets.

**5. ``format/headers.py``**

- Utilities for detecting and applying missing headers across files, particularly for Campbell Scientific TOA5 format data.

**6. ``format/transformers/``**

- ``columns``: Column renaming, suffix standardisation, and duplicate resolution.
- ``timestamps``: Timestamp parsing, correction, and format conversion.
- ``corrections``: Calibration corrections (precipitation, soil heat flux, sign inversions).
- ``validation``: Range checks and flag generation.
- ``cleanup``: Removal of all-NaN columns and artefact records.
- ``interval_updates``: Filtering data to a target measurement interval via ``subset_interval()``.

----

**qaqc subpackage**

**7. ``qaqc/variable_limits.py``**

- Dictionary defining physical and plausible ranges for every AmeriFlux variable, used by ``Reformatter.finalize()`` to flag and replace out-of-range values.

**8. ``qaqc/netrad_limits.py``**

- Net radiation QA/QC: timestamp alignment checks and signal-strength filtering for radiation sensors.

**9. ``qaqc/data_cleaning.py``**

- Applies QC flags to data, including signal-strength thresholds for IRGA variables (CO₂ and H₂O).

----

**report subpackage**

**10. ``report/graphs.py``**

- ``energy_sankey()``: Visualizes daily energy balance as Sankey diagrams.
- ``scatterplot_instrument_comparison()``: Compares instruments with regression statistics and Bland–Altman plots.

**11. ``report/tools.py``**

- ``find_irr_dates()``: Detects irrigation events from precipitation and soil moisture data.
- ``find_gaps()`` / ``plot_gaps()``: Identifies and visualizes missing data periods.

**12. ``report/validate.py``**

- ``review_lags()``: Cross-correlation lag detection between sensor columns.
- ``detect_sectional_offsets_indexed()``: Detects systematic time offsets between eddy and meteorological systems.

**13. ``report/fix_g_values.py``**

- Soil heat flux storage corrections: adjusts ``G_PLATE`` measurements for sensor burial depth differences.

**14. ``report/recalculate_albedo.py``**

- Recalculates albedo from shortwave radiation components when primary albedo values are missing or erroneous.

**15. ``report/gap_summary.py``**

- Generates tabular summaries of data gap statistics by variable and time period.

**16. ``report/eddy_plots.py``**

- Diagnostic plots for eddy covariance data using Bokeh and Plotly, including time-series and footprint overlays.

**17. ``report/easyflux_footprint.py``**

- Flux footprint estimation using EasyFlux outputs, including spatial extent and directional analysis.

**18. ``report/alfalfa_growth.py``**

- ``AlfalfaHeightParams``: Dataclass for site-specific alfalfa height model parameters.
- ``simulate_alfalfa_height_multi_field()``: Simulates alfalfa canopy height from growing degree days (GDD) across multiple cuttings and fields.

----

**station_data_pull**

**19. ``station_data_pull.py``**

- ``StationDataDownloader``: Fetches logger data over HTTP from remote Campbell Scientific stations.
- ``StationDataProcessor``: Compares and inserts downloaded data into SQL databases.
