# Crop Height Estimation-- Workflow Summary

**Utah Geological Survey -- Flux Monitoring Network**

This page provides a high-level overview of the eddy covariance data processing pipeline. For full technical detail on each step, see the [complete workflow reference](flux_processing_workflow.md).

---

## Pipeline at a Glance

The pipeline converts preprocessed eddy covariance and meteorological data into quality-controlled, AmeriFlux-formatted output through five processing steps and three review checkpoints.

| Step | Notebook | Input | Key Operations | Output |
|------|----------|-------|---------------|--------|
| [**1**](flux_processing_workflow.md#step-1--compile-and-preprocess) | `1_compile_and_preprocess` | Raw data logger or EasyFlux web files | Compile datalogger files · standardize columns | `*_preprocessed.parquet` |
| [**2**](flux_processing_workflow.md#step-2--create-raw-dataset) | `2_create_raw_data` | Preprocessed parquets | Merge data from different data streams · Merge met and eddy data · fix time alignment issues  | `*_raw.parquet` |
| [**3**](flux_processing_workflow.md#step-3--quality-control) | `3_qc_data` | `*_raw.parquet` | Calibration corrections · physical limits · SoilVue G calc · manual QC · signal flags | `*_qc.parquet` |
| [**4**](flux_processing_workflow.md#step-4--ameriflux-export) | `4_ameriflux` | `*_qc.parquet` | Signal-strength filter · drop non-AmeriFlux columns · format timestamps | `*_HH_*.csv` |
| [**5**](flux_processing_workflow.md#step-5--flux-qaqc) | `5_fluxqaqc` | `*_HH_*.csv` | Gap-fill redundant sensors · EBR correction · ET gap-fill · sensitivity tests | Daily ET + HTML reports |

**Review notebooks** (read-only -- findings feed corrections back into Step 3):

| Notebook | When to run | Purpose |
|----------|-------------|---------|
| [**3a** -- Variable Review](flux_processing_workflow.md#step-3a--variable-review) | After Step 3 | Summary statistics, distributions, outlier detection |
| [**3b** -- Plot Review](flux_processing_workflow.md#step-3b--plot-review) | After Step 3 | Quick time-series sweep of every variable |
| [**4b** -- AmeriFlux Plot Review](flux_processing_workflow.md#step-4b--ameriflux-plot-review) | After Step 4 | Final visual check before AmeriFlux submission |

---

## Step-by-Step Summary

### Step 1 -- Compile & Preprocess

[-> Full details](flux_processing_workflow.md#step-1--compile-and-preprocess)

Create compiled and clean versions of data from each data stream (e.g., CSFlux web/datalogger, AmeriFlux eddy web/datalogger, MetStats, MetAF)

1. Organize datalogger files into a single directory by table name (e.g., Statistics_Ameriflux, Statistics, Flux_AmerifluxFormat, Flux_CSFormat)
2. Compile data from a single data stream into a dataframe
3. Clean data by applying renaming dictionary, setting data types, and fixing timestamp issue
4. Subset out data to only include 30 or 60 minute data, depending on user input
5. Present data for review to identify misnamed columns and missing data
6. Export data into separate parquet files for data from each data stream

**Output:** `{station}_{timestart}_{timeend}_preprocessed.parquet`

---

### Step 2 -- Create Raw Dataset

[-> Full details](flux_processing_workflow.md#step-2--create-raw-dataset)

Assemble multiple preprocessed files into final datasets and manage any datetime shifts

1. Load preprocessed parquets for each data stream (CSFlux web/datalogger, AmeriFlux eddy web/datalogger, MetStats, MetAF)
2. Compare and merge eddy data -- CSFlux and AmeriFlux eddy streams are compared for differences; the AmeriFlux stream is primary, with CSFlux filling gaps and providing unique columns (e.g., G_PLATE, diagnostic fields)
3. Compare and merge met data -- MetStats and MetAF streams are compared and combined
4. Detect and correct temporal shifts -- SoilVue sensor data (EC_3_*, K_3_*, SWC_3_*, TS_3_*) may be offset by one time step; cross-correlation detects the lag and a frequency shift corrects it. Historical timestamp misalignments are also identified and corrected.
5. Combine eddy and met -- merge the two streams, resolve duplicate columns, validate 30-minute interval integrity
6. Standardize column naming -- apply AmeriFlux positional suffixes (_1_1_1, _1_1_2, etc.)
7. Trim to station record -- drop data before the station install date (retrieved from the database API)

**Output:** `{station}_{timestart}_{timeend}_raw.parquet`

---

### Step 3 -- Quality Control

[-> Full details](flux_processing_workflow.md#step-3--quality-control)

The largest and most site-specific step:

1. **Calibration corrections** -- date-gated fixes for soil heat flux storage thickness, precipitation calibration factors, and G_PLATE sign inversions
2. **SoilVue G calculation** -- derive ground heat flux from temperature/moisture profiles using the `soil_heat` library (Johansen thermal model)
3. **Physical limits** -- `Reformatter.finalize()` applies range limits, converts SWC units, standardizes SSITC encoding, and produces a limit report
4. **Manual corrections** -- field-day precipitation, G_PLATE zeros, SoilVue spikes, wind direction offsets, sensor-specific spike removal
5. **Signal-strength flags** -- H2O/CO2 signal flags (0/1/2) and wind direction obstruction flags
6. **Gap-fill G** -- linear regression between redundant G sources to impute missing values

**Output:** `{station}_{daterange}_qc.parquet` + limit report CSV

---

### Review: 3a & 3b

[-> 3a details](flux_processing_workflow.md#step-3a--variable-review) -- [-> 3b details](flux_processing_workflow.md#step-3b--plot-review)

Run after Step 3 to evaluate data quality. Issues found here are resolved by adding correction blocks in Notebook 3 and re-running Steps 3--5.

**3a:** Summary statistics, data availability, and outlier detection for each variable.

**3b:** Interactive Plotly time-series for every column -- a rapid visual sweep for spikes, gaps, or artifacts.

---

### Step 4 -- AmeriFlux Export

[-> Full details](flux_processing_workflow.md#step-4--ameriflux-export)

Converts the QC dataset into an AmeriFlux-compliant half-hourly CSV:

- IRGA-derived variables set to NaN where signal strength < 0.8
- Non-AmeriFlux columns dropped; all-NaN columns removed
- NaN replaced with -9999; timestamps formatted as `YYYYMMDDHHmm`

**Output:** `{station}_HH_{start}_{end}.csv`

---

### Step 5 -- Flux QAQC

[-> Full details](flux_processing_workflow.md#step-5--flux-qaqc)

Runs `fluxdataqaqc` for energy balance ratio (EBR) correction and ET gap-filling:

- Gap-fill redundant NETRAD and G sensors via linear regression
- EBR correction applied to LE; ET gap-filled using ETrF x gridMET ETr
- Data subset by year and season for analysis
- Sensitivity runs with different Rn/G input combinations

**Outputs:** EBR-corrected daily ET, HTML diagnostic reports, optional daily CSV

---

## Key Libraries

| Library | Role |
|---------|------|
| [**micromet**](micromet.rst) | Core pipeline: `Reformatter`, `validate`, `merge`, `data_cleaning`, `fix_g_values`, `timestamps`, `columns`, `eddy_plots` |
| **soil_heat** | SoilVue-derived ground heat flux (Johansen model) |
| **fluxdataqaqc** | EBR correction, ET gap-fill (`Data`, `QaQc`, `Plot`) |
| **pandas / numpy** | Data wrangling and array operations |
| **scipy** | Cross-correlation and linear regression |
| **plotly / bokeh** | Interactive diagnostics and HTML reports |

---

## Directory Structure

```
M:/Shared drives/UGS_Flux/
├── Data_Downloads/compiled/
│   ├── preprocessed_site_data/   ← preprocessed parquets
│   └── {stationid}/              ← raw .dat source files
└── Data_Processing/final_database_tables/
    ├── raw/          *_raw.parquet
    ├── qc/           *_qc.parquet
    └── ameriflux/    *_HH_*.csv
```

---

## Adapting to Other Sites

Copy the notebooks and update:

1. `station`, `interval`, `date_range` -- station code, measurement interval, and date bounds
2. Calibration correction dates and factors (Notebook 3)
3. Sensor failure date ranges and affected variables
4. Wind direction offsets between instruments
5. Signal-strength bad-period date ranges
6. Column selection for eddy merging (Notebooks 1/2) and AmeriFlux export (Notebook 4)
7. `.ini` config for `fluxdataqaqc` (Notebook 5)

See the [full workflow document](flux_processing_workflow.md#adapting-the-workflow-to-other-sites) for detailed guidance.
