# Flux Data Processing -- Workflow Summary

**Utah Geological Survey -- Flux Monitoring Network**

This page provides a high-level overview of the eddy covariance data processing pipeline. For full technical detail on each step, see the [complete workflow reference](flux_processing_workflow.md).

---

## Workflow Flowchart

```{image} _static/flux_workflow_flowchart.png
:alt: Flux data processing workflow flowchart
:align: center
:width: 100%
```

---

## Pipeline at a Glance

The pipeline converts preprocessed eddy covariance and meteorological data into quality-controlled, AmeriFlux-formatted output through five processing steps and three review checkpoints.

| Step | Notebook | Input | Key Operations | Output |
|------|----------|-------|---------------|--------|
| [**1**](flux_processing_workflow.md#step-1--compile-and-preprocess) | `1_compile_and_preprocess` | Preprocessed parquets | Compare & merge eddy/met sources · fix time shifts · standardize columns | `*_raw.parquet` |
| [**2**](flux_processing_workflow.md#step-2--create-raw-dataset) | `2_create_raw_data` | Preprocessed parquets | Alternate raw assembly with alignment corrections | `*_raw.parquet` |
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

Merges multiple preprocessed data streams (CSFlux, AmeriFlux eddy, MetStats, MetAF) into a single raw dataset per station:

- Compare web vs. datalogger sources for each stream; merge with gap-filling
- Detect and correct SoilVue temporal offsets via cross-correlation
- Fix historical timestamp misalignments
- Combine eddy and met data; standardize column naming with AmeriFlux suffixes
- Drop data before station install date

**Output:** `{station}_{timestart}_{timeend}_raw.parquet`

---

### Step 2 -- Create Raw Dataset

[-> Full details](flux_processing_workflow.md#step-2--create-raw-dataset)

An alternate version of Step 1 performing the same core task -- merging preprocessed streams, correcting time shifts, and producing a raw dataset. Used interchangeably depending on station requirements.

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
