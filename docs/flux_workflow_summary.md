# Flux Data Processing – Workflow Summary

**Utah Geological Survey · Flux Monitoring Network**
Station reference implementation: **US-UTD (Dugout Ranch)**

This page provides a high-level overview of the end-to-end eddy covariance data processing pipeline. For full technical detail on each step, see the [complete workflow reference](flux_processing_workflow.md).

---

## Workflow Flowchart

```{image} _static/flux_workflow_flowchart.png
:alt: Flux data processing workflow flowchart
:align: center
:width: 100%
```

---

## Pipeline at a Glance

The pipeline converts raw Campbell Scientific `.dat` files into quality-controlled, AmeriFlux-formatted output through six processing steps and three review checkpoints.

| Step | Notebook | Input | Key Operations | Output |
|------|----------|-------|---------------|--------|
| [**1**](flux_processing_workflow.md#step-1--compile-and-preprocess) | `dugout1_compile_and_preprocess` | Raw `.dat` files | Compile files · run `Reformatter.preprocess()` · validate timestamps · subset interval | 4 × `*_preprocessed.parquet` |
| [**2**](flux_processing_workflow.md#step-2--create-raw-dataset) | `dugout2_create_raw_data` | Preprocessed parquets | Merge eddy & met · fix SoilVue time-shift · align met–eddy · standardise columns | `*_raw.parquet` |
| [**3**](flux_processing_workflow.md#step-3--quality-control) | `dugout3_qc_data` | `*_raw.parquet` | Calibration corrections · `Reformatter.finalize()` · manual QC · signal flags | `*_qc.parquet` |
| [**4**](flux_processing_workflow.md#step-4--ameriflux-export) | `dugout4_ameriflux` | `*_qc.parquet` | Signal-strength filter · drop non-AmeriFlux cols · format timestamps · fill −9999 | `*_HH_*.csv` |
| [**5**](flux_processing_workflow.md#step-5--flux-qaqc-with-energy-balance) | `dugout5_fluxqaqc` | `*_HH_*.csv` | Gap-fill NETRAD/G · EBR correction · ET gap-fill (gridMET) · sensitivity tests | Daily ET + HTML reports |

**Review notebooks** (read-only — findings feed corrections back into Step 3):

| Notebook | When to run | Purpose |
|----------|-------------|---------|
| [**3a** · Variable Review](flux_processing_workflow.md#step-3a--variable-review) | After Step 3 | Regression, wind roses, energy-balance closure, sensor intercomparison |
| [**3b** · Plot Review](flux_processing_workflow.md#step-3b--plot-review) | After Step 3 | Quick time-series sweep of every variable |
| [**4b** · AmeriFlux Plot Review](flux_processing_workflow.md#step-4b--ameriflux-plot-review) | After Step 4 | Final visual check before AmeriFlux submission |

---

## Step-by-Step Summary

### Step 1 · Compile & Preprocess

[→ Full details](flux_processing_workflow.md#step-1--compile-and-preprocess)

Four data streams are assembled from the shared drive and run through `micromet.Reformatter.preprocess()`:

- **Met Statistics** (TOA5 format) — strips `_Avg`/`_Tot` suffixes, standardises timestamps
- **Met AmeriFlux Statistics** — renames leaf-wetness columns, drops all-NA artefact columns
- **Eddy AmeriFlux Format** — validates against AmeriFlux master variable list
- **Eddy CS Format** — renames Campbell-specific columns, adds diagnostic variables absent from AmeriFlux format

All streams are filtered to the target interval (30 or 60 min) via `interval_updates.subset_interval()`.

**Outputs:** `{stationid}_{interval}_{source}_preprocessed.parquet` × 4

---

### Step 2 · Create Raw Dataset

[→ Full details](flux_processing_workflow.md#step-2--create-raw-dataset)

The four preprocessed streams are merged into one coherent dataset:

1. **Eddy merge** — AmeriFlux format is the primary stream; CS Format fills gaps and supplies unique columns (`G_PLATE`, `FC_MASS`, `TKE`, `TSTAR`, wind components).
2. **SoilVue time-shift** — SoilVue profile columns (`EC_3_*`, `K_3_*`, `SWC_3_*`, `TS_3_*`) in the AmeriFlux Statistics table are often offset by 30 min. Cross-correlation via `validate.review_lags()` detects the lag; `shift(freq='30min')` corrects it.
3. **Met–eddy alignment** — `validate.detect_sectional_offsets_indexed()` checks for systematic offsets between `NETRAD` and `WS` across the two systems; any detected shift is corrected.
4. **Column cleanup** — duplicates renamed (`FILE_NAME_EDDY`, `FILE_NAME_MET`); derived columns dropped; `_1_1_1` suffixes applied; data before install date removed.

**Output:** `{stationid}_{start}_{end}_raw.parquet`

---

### Step 3 · Quality Control

[→ Full details](flux_processing_workflow.md#step-3--quality-control)

The largest and most site-specific step. Key operations in sequence:

1. **Calibration corrections** (date-gated using program-update records)
   - Soil heat flux storage: incorrect layer thickness (0.16 m → 0.05 m) corrected by factor 0.3125
   - `G_PLATE_2` sign inversion corrected for affected period
   - Tipping bucket precipitation: calibration factor 0.1 → 0.254 (×2.54)
2. **SoilVue G calculation** — `soil_heat` library computes `SG_3_1_1` and conductive flux; `G_3_1_1 = SG_3_1_1 + G_SOILVUE`
3. **MicroMet finalize** — converts SWC fraction → percent; applies physical limits; standardises SSITC encoding; produces limit report
4. **Manual corrections** — field-day precip cleanup, G_PLATE zeros, SoilVue spikes, wind-direction offsets, pressure/temperature spikes, footprint outliers, failed sensor nulling
5. **Signal-strength flags** — `H2O_SIG_FLAG_1_1_1` and `CO2_SIG_FLAG_1_1_1` (0 = good / 1 = marginal / 2 = known bad period); `WD_1_1_1_FLAG` for tower obstruction sector

**Output:** `{stationid}_{daterange}_qc.parquet`

---

### Review: 3a & 3b

[→ 3a details](flux_processing_workflow.md#step-3a--variable-review) · [→ 3b details](flux_processing_workflow.md#step-3b--plot-review)

Run after Step 3 to evaluate data quality. Any issues found here are resolved by adding new correction blocks in Notebook 3 and re-running Steps 3–5.

**3a covers:** radiation intercomparison, albedo, wind speed/direction regression, soil heat flux plate vs. SoilVue, temperature sensor agreement, RH/VPD signal-strength stratification, energy balance closure.

**3b:** iterates over all columns and produces an interactive Plotly time-series for each.

---

### Step 4 · AmeriFlux Export

[→ Full details](flux_processing_workflow.md#step-4--ameriflux-export)

Converts the QC parquet into an AmeriFlux-ready half-hourly CSV:

- IRGA-derived variables (LE, H2O, CO2, RH, ET) set to NaN where signal strength < 0.8
- Non-AmeriFlux columns dropped (internal flags, diagnostic fields, temporal helpers)
- NaN → −9999; timestamps regenerated in `YYYYMMDDHHmm` format
- Final file retains ~80 variables across flux, radiation, temperature, humidity, soil, and wind categories

**Output:** `{stationid}_HH_{start}_{end}.csv`

---

### Step 5 · Flux QAQC

[→ Full details](flux_processing_workflow.md#step-5--flux-qaqc-with-energy-balance)

Runs `flux-data-qaqc` to perform Energy Balance Ratio (EBR) correction and ET gap-filling:

- Redundant sensors (`NETRAD_1_1_1` / `NETRAD_1_1_2`; `G_1_1_A` / `G_3_1_1`) are cross-regressed to fill gaps before passing to QAQC
- EBR correction applied to LE; corrected ET gap-filled using ETrF × gridMET ETr
- Sensitivity runs with different Rn and G input combinations produce separate HTML reports for comparison

**Outputs:** EBR-corrected daily ET, HTML diagnostic reports, optional daily CSV

---

## Key Libraries

| Library | Role |
|---------|------|
| [**micromet**](micromet.rst) | Core pipeline: `Reformatter`, `validate`, `merge`, `data_cleaning`, `fix_g_values`, `timestamps`, `columns`, `interval_updates`, `eddy_plots` |
| **soil_heat** | SoilVue-derived ground heat flux (`storage_calculations`, `soil_heat`) |
| **fluxdataqaqc** | EBR correction, ET gap-fill (`Data`, `QaQc`, `Plot`) |
| **pandas / numpy** | Data wrangling and array operations |
| **scipy** | Cross-correlation and linear regression |
| **plotly / bokeh** | Interactive diagnostics and HTML reports |

---

## Directory Structure

```
M:/Shared drives/UGS_Flux/
├── Data_Downloads/compiled/{stationid}/     ← raw .dat source files
│   ├── Statistics/
│   ├── Statistics_Ameriflux/
│   ├── AmeriFluxFormat/
│   └── Flux_CSFormat/
└── Data_Processing/final_database_tables/   ← processed outputs
    ├── raw/          *_raw.parquet
    ├── qc/           *_qc.parquet
    └── ameriflux/    *_HH_*.csv
```

---

## Adapting to Other Sites

Copy the dugout notebooks and update:

1. `stationid`, `interval` — station code and measurement interval
2. Calibration correction dates and factors in Notebook 3
3. Sensor failure date ranges and affected variable lists
4. Wind direction offset between sonic and Young anemometer
5. Signal-strength bad-period date ranges
6. `csflux_join_cols` in Notebook 2 (site-dependent sensor array)
7. `.ini` config for flux-data-qaqc (Notebook 5)

See the [full workflow document](flux_processing_workflow.md#adapting-the-workflow-to-other-sites) for detailed guidance.
