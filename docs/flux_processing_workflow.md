# Eddy Covariance Flux Data Processing Workflow

**Utah Geological Survey -- Flux Monitoring Network**

---

## Overview

This document describes the end-to-end data processing workflow used by the Utah Geological Survey (UGS) Flux Monitoring Network to convert raw eddy covariance and meteorological data into quality-controlled, AmeriFlux-formatted output. The workflow is implemented as a series of numbered Jupyter Notebooks in the `docs/notebooks/` directory, each handling a distinct processing stage.

The pipeline progresses linearly through five major stages with review notebooks interspersed for visual quality assessment. Each notebook reads from the output of the previous stage and writes intermediate or final products as Parquet or CSV files.

### Pipeline at a Glance

| Step | Notebook | Purpose | Output |
|------|----------|---------|--------|
| 1 | `1_compile_and_preprocess` | Merge preprocessed data sources into a single raw dataset | `*_raw.parquet` |
| 2 | `2_create_raw_data` | Alternate/extended raw dataset assembly with alignment corrections | `*_raw.parquet` |
| 3 | `3_qc_data` | Apply calibrations, physical limits, flags, and manual QC | `*_qc.parquet` |
| *3a* | `3a_variable_review` | *Summary statistics and distribution analysis of QC data* | *Diagnostic output* |
| *3b* | `3b_plot_review` | *Interactive time-series plots of every variable* | *Visual review only* |
| 4 | `4_ameriflux` | Signal-strength filter, drop non-AmeriFlux columns, format and export | `*_HH_*.csv` |
| *4b* | `4b_ameriflux_plots` | *Plot every variable in the final AmeriFlux file* | *Visual review only* |
| 5 | `5_fluxqaqc` | Energy balance closure analysis and ET gap-filling | Daily ET + HTML reports |

*Italicized rows are review-only notebooks that do not modify data. Issues found during review should be corrected in the appropriate upstream notebook (primarily Notebook 3).*

---

## Prerequisites

### Software and Libraries

- Python 3.x with pandas, numpy, scipy, plotly, bokeh
- **micromet** -- core UGS processing library (`Reformatter`, `validate`, `merge`, `data_cleaning`, `columns`, `timestamps`, `fix_g_values`, `eddy_plots`, `interval_updates`)
- **soil_heat** -- SoilVue-derived ground heat flux calculations
- **fluxdataqaqc** -- energy balance ratio correction and ET gap-filling
- Supporting: `prettytable`, `requests`

### Data Sources

- Preprocessed Parquet files from the compilation stage (CSFlux, AmeriFlux eddy, MetStats, MetAF -- both web and datalogger variants)
- AmeriFlux variable naming reference CSV (`flux-met_processing_variables_*.csv`)
- UGS database API providing station metadata, visit notes, and program update history

### Directory Structure

```
M:/Shared drives/UGS_Flux/
├── Data_Downloads/compiled/
│   ├── preprocessed_site_data/   ← preprocessed parquets per source
│   └── {stationid}/              ← raw .dat files by station
└── Data_Processing/final_database_tables/
    ├── raw/          *_raw.parquet
    ├── qc/           *_qc.parquet
    └── ameriflux/    *_HH_*.csv
```

---

## Step 1 -- Compile and Preprocess

**Notebook:** `1_compile_and_preprocess.ipynb`

Assembles a single raw dataset from multiple preprocessed data sources for one station.

### Key Operations

1. **Load preprocessed parquets** for each data stream (CSFlux web/datalogger, AmeriFlux eddy web/datalogger, MetStats, MetAF)
2. **Compare and merge eddy data** -- CSFlux and AmeriFlux eddy streams are compared for differences; the AmeriFlux stream is primary, with CSFlux filling gaps and providing unique columns (e.g., `G_PLATE`, diagnostic fields)
3. **Compare and merge met data** -- MetStats and MetAF streams are compared and combined
4. **Detect and correct temporal shifts** -- SoilVue sensor data (`EC_3_*`, `K_3_*`, `SWC_3_*`, `TS_3_*`) may be offset by one time step; cross-correlation detects the lag and a frequency shift corrects it. Historical timestamp misalignments are also identified and corrected.
5. **Combine eddy and met** -- merge the two streams, resolve duplicate columns, validate 30-minute interval integrity
6. **Standardize column naming** -- apply AmeriFlux positional suffixes (`_1_1_1`, `_1_1_2`, etc.)
7. **Trim to station record** -- drop data before the station install date (retrieved from the database API)

### Configuration

- `station` -- AmeriFlux station ID
- `interval` -- measurement interval in minutes (30 or 60)
- Paths to preprocessed parquets and output directory

### Output

`{station}_{timestart}_{timeend}_raw.parquet` in `final_database_tables/raw/`

---

## Step 2 -- Create Raw Dataset

**Notebook:** `2_create_raw_data.ipynb`

An alternate or extended version of Step 1 that performs the same core task -- merging preprocessed streams into a raw dataset -- with the same general approach: compare web vs. datalogger sources, merge eddy and met data, detect SoilVue time shifts, correct historical timestamp issues, standardize columns, and export.

This notebook can serve as the primary raw-data assembly step depending on the station. The key operations and output format are the same as Step 1.

---

## Step 3 -- Quality Control

**Notebook:** `3_qc_data.ipynb`

The largest and most site-specific step. Applies corrections, physical limits, and quality flags to produce a QC-level dataset.

### Key Operations

1. **Retrieve station metadata** -- query the database API for station visit notes and program update history to inform date-gated corrections
2. **Apply calibration corrections** -- site-specific fixes applied before the program update date to avoid double-correction:
   - Soil heat flux storage (SG) thickness correction
   - Precipitation tipping bucket calibration factor
   - G_PLATE sign inversions
   - G values recalculated as `G = SG + G_PLATE`
3. **Calculate SoilVue-derived ground heat flux** -- use the `soil_heat` library (Johansen thermal properties model) to compute `G_SURFACE_3_1_1` from SoilVue temperature and moisture profiles
4. **Apply physical limits** via `Reformatter.finalize()`:
   - Converts SWC from fraction to percent
   - Applies range limits by variable type (out-of-range values set to NaN)
   - Standardizes SSITC encoding
   - Produces a limit report CSV for review
5. **Manual corrections** -- address site-specific data issues:
   - Spurious precipitation on station visit days
   - G_PLATE zeros (sensor disconnection)
   - SoilVue spikes (flagged by thermal conductivity threshold)
   - NETRAD and G_PLATE spikes
   - Wind direction offsets between instruments
   - Temperature, pressure, and other sensor-specific spikes
   - Precipitation nulling before sensor repair dates
6. **Signal-strength flagging**:
   - `H2O_SIG_FLAG` and `CO2_SIG_FLAG`: 0 = good (signal >= 0.8), 1 = marginal (< 0.8), 2 = known bad period
   - `WD_1_1_1_FLAG`: flags wind from behind the tower or obstruction sectors
7. **Gap-fill ground heat flux** -- train linear regression models between redundant G sources to impute missing values

### Output

- `{station}_{daterange}_qc.parquet` in `final_database_tables/qc/`
- `{station}_{daterange}_report.csv` -- finalization report showing flagged percentages per variable

---

## Step 3a -- Variable Review

**Notebook:** `3a_variable_review.ipynb`

Read-only review of the QC dataset. Generates summary statistics and distribution analysis for each variable to evaluate data quality. Includes counts, percentiles, data availability, and outlier detection. Any issues found should be corrected by adding blocks to Notebook 3.

---

## Step 3b -- Plot Review

**Notebook:** `3b_plot_review.ipynb`

Iterates over all columns in the QC (or raw) dataset and generates an interactive Plotly time-series plot for each. Provides a rapid visual sweep for remaining spikes, gaps, step changes, or artifacts. Can be run at either the `raw` or `qc` level by changing the `level` parameter.

---

## Step 4 -- AmeriFlux Export

**Notebook:** `4_ameriflux.ipynb`

Converts the QC dataset into an AmeriFlux-compliant half-hourly CSV file.

### Key Operations

1. **Signal-strength filtering** -- IRGA-derived variables set to NaN where signal strength < 0.8:
   - H2O signal: `H2O`, `H2O_SIGMA`, `LE`, `RH`, `VPD`, `ET`
   - CO2 signal: `CO2`, `CO2_SIGMA`, `FC`
2. **Column cleanup** -- drop all-NaN columns, remove non-AmeriFlux variables (validated against the master variable list), drop internal flags and diagnostic fields
3. **Format for submission** -- replace NaN with -9999, recalculate `TIMESTAMP_START` and `TIMESTAMP_END` in `YYYYMMDDHHmm` format

### Output

`{station}_HH_{timestamp_start}_{timestamp_end}.csv` -- ready for AmeriFlux upload

---

## Step 4b -- AmeriFlux Plot Review

**Notebook:** `4b_ameriflux_plots.ipynb`

Final visual check before AmeriFlux submission. Reads the exported CSV (converting -9999 back to NaN), then plots every variable as an interactive time series. This is the last review checkpoint before upload.

---

## Step 5 -- Flux QAQC

**Notebook:** `5_fluxqaqc.ipynb`

Runs the `fluxdataqaqc` package to perform energy balance ratio (EBR) correction, gap-fill ET, and produce diagnostic reports.

### Key Operations

1. **Gap-fill redundant sensors** -- use linear regression to cross-fill between redundant NETRAD and G sources (e.g., `NETRAD_1_1_1` / `NETRAD_1_1_2`), creating `*_FINAL` columns
2. **Run FluxDataQAQC** with `.ini` configuration files mapping columns to Rn, G, LE, H, etc.:
   - `daily_frac = 1` (require complete days)
   - `max_interp_hours = 2` (daytime) / `max_interp_hours_night = 4` (nighttime)
   - EBR correction method applied to LE
   - ET gap-filling using ETrF x gridMET reference ET
3. **Seasonal and annual subsetting** -- analyze energy balance closure and ET by year and season (growing season: Apr 1 -- Oct 31; winter: Nov 1 -- Mar 31)
4. **Sensitivity testing** -- run multiple configurations with different NETRAD and G inputs to compare results

### Outputs

- HTML diagnostic reports with interactive bokeh plots
- Monthly summaries of ET data availability (good, gap-filled, missing)
- Optional daily corrected CSV export

---

## Data Flow Diagram

```
Preprocessed Parquets (CSFlux + AmeriFlux Eddy + MetStats + MetAF)
    |
    v  [Notebook 1/2: Compile & Merge]
*_raw.parquet
    |
    v  [Notebook 3: QC -- calibrations, physical limits, flags, manual corrections]
*_qc.parquet
    |
    +---> [Notebook 3a: Variable Review -- statistics, distributions]
    +---> [Notebook 3b: Plot Review -- time-series sweep]
    |     (feedback loop: corrections go back to Notebook 3)
    |
    v  [Notebook 4: AmeriFlux Export -- signal filter, format, export]
*_HH_*.csv  (AmeriFlux submission file)
    |
    +---> [Notebook 4b: AmeriFlux Plot Review -- final visual check]
    |
    v  [Notebook 5: Flux QAQC -- EBR correction, ET gap-fill]
EBR-corrected daily ET  +  HTML diagnostic reports
```

---

## Adapting the Workflow to Other Sites

Each station needs its own copy of the notebooks with the following site-specific elements updated:

1. **Parameters** -- `station` ID, `interval`, and `date_range`
2. **Calibration corrections** (Notebook 3) -- dates, factors, and affected variables determined from station visit logs and program updates
3. **Sensor failure periods** -- date ranges and variables to null
4. **Wind direction offsets** -- instrument-specific azimuth corrections
5. **Signal-strength bad periods** -- date ranges for known IRGA contamination
6. **Column selection** (Notebooks 1/2 and 4) -- varies by station sensor array
7. **FluxDataQAQC config** (Notebook 5) -- `.ini` file with column mappings for the site's sensor configuration
