# Eddy Covariance Flux Data Processing Workflow

**Utah Geological Survey – Flux Monitoring Network**

Based on the MicroMet/dugout processing notebooks
Author: Diane Menuz | Compiled: March 2026
Station Reference Implementation: US-UTD (Dugout Ranch)

---

## Table of Contents

1. [Overview and Pipeline Summary](#overview-and-pipeline-summary)
2. [Prerequisites and Directory Structure](#prerequisites-and-directory-structure)
3. [Step 1 – Compile and Preprocess (Notebook 1)](#step-1--compile-and-preprocess)
4. [Step 2 – Create Raw Dataset (Notebook 2)](#step-2--create-raw-dataset)
5. [Step 3 – Quality Control (Notebook 3)](#step-3--quality-control)
6. [Step 3a – Variable Review (Notebook 3a)](#step-3a--variable-review)
7. [Step 3b – Plot Review (Notebook 3b)](#step-3b--plot-review)
8. [Step 4 – AmeriFlux Export (Notebook 4)](#step-4--ameriflux-export)
9. [Step 4b – AmeriFlux Plot Review (Notebook 4b)](#step-4b--ameriflux-plot-review)
10. [Step 5 – Flux QAQC with Energy Balance (Notebook 5)](#step-5--flux-qaqc-with-energy-balance)
11. [Key Libraries and Dependencies](#key-libraries-and-dependencies)
12. [Data Flow Diagram](#data-flow-diagram)
13. [Adapting the Workflow to Other Sites](#adapting-the-workflow-to-other-sites)

---

## Overview and Pipeline Summary

This document describes the end-to-end data processing workflow used by the Utah Geological Survey (UGS) Flux Monitoring Network to take raw eddy covariance and meteorological data from Campbell Scientific dataloggers and produce quality-controlled, AmeriFlux-formatted output files. The workflow is implemented as a series of numbered Jupyter Notebooks in the `MicroMet/dugout` directory, each handling a distinct stage of processing.

The pipeline follows a linear progression through six major stages, with review notebooks interspersed for interactive data exploration and visual quality assessment. Each notebook reads from the output of the previous stage and writes intermediate or final products as Parquet or CSV files to a shared Google Drive.

### Pipeline Stages at a Glance

| Step | Notebook | Purpose | Output |
|------|----------|---------|--------|
| 1 | `dugout1_compile_and_preprocess` | Compile raw .dat files; preprocess met and eddy data | `*_preprocessed.parquet` (per data source) |
| 2 | `dugout2_create_raw_data` | Merge sources into single raw dataset; fix time shifts | `*_raw.parquet` |
| 3 | `dugout3_qc_data` | Apply corrections, physical limits, flags, and manual QC | `*_qc.parquet` |
| *3a* | *`dugout3a_variable_review`* | *Interactive exploration of QC data (not modifying)* | *Diagnostic plots (PNG files)* |
| *3b* | *`dugout3b_plot_review`* | *Quick time-series plot of every variable* | *Visual review only* |
| 4 | `dugout4_ameriflux` | Drop non-AmeriFlux columns; format and export | `*_HH_*.csv` (AmeriFlux upload) |
| *4b* | *`dugout4b_ameriflux_plots`* | *Plot every variable in the final AmeriFlux file* | *Visual review only* |
| 5 | `dugout5_fluxqaqc` | Energy balance closure analysis with flux-data-qaqc | EBR-corrected daily ET; HTML reports |

*Italicized rows indicate review-only notebooks that do not modify or export data. Any issues found during review should be addressed by adding corrections back into the appropriate upstream notebook (primarily Notebook 3).*

---

## Prerequisites and Directory Structure

### Required Software and Libraries

- Python 3.x with pandas, numpy, scipy, matplotlib, plotly, geopandas
- **MicroMet** library (custom UGS package at `micromet_path`) providing: `Reformatter`, `validate`, `interval_updates`, `file_compile`, `eddy_plots`, `data_cleaning`, `merge`, `columns`, `timestamps`, `fix_g_values`, `recalculate_albedo`, `variable_limits`
- **soil_heat** library (custom UGS package) providing: `storage_calculations`, `soil_heat` modules
- **flux-data-qaqc** (`fluxdataqaqc`) – third-party package for energy balance correction
- Supporting tools: `prettytable`, `windrose`, `bokeh`

### Data Source Directory Structure

Raw data resides on a shared Google Drive under the path:

```
M:/Shared drives/UGS_Flux/Data_Downloads/compiled/{stationid}/
```

Each station folder must contain the following subdirectories and files:

- `{stationname}_Flux_AmeriFluxFormat.dat` – AmeriFlux-format eddy file from EasyFlux
- `{stationname}_Flux_CSFormat.dat` – Campbell Scientific format eddy file from EasyFlux
- `AmeriFluxFormat/` – folder of eddy data downloaded directly from the datalogger
- `Statistics_Ameriflux/` – folder of met data in AmeriFlux naming from the datalogger
- `Statistics/` – folder of met data from the datalogger (TOA5 format, may need Card Convert)
- `Flux_CSFormat/` – folder of CS-format eddy data from the datalogger

### Output Directory Structure

```
M:/Shared drives/UGS_Flux/Data_Processing/final_database_tables/
```

- `raw/` – merged raw parquet files from Notebook 2
- `qc/` – quality-controlled parquet files from Notebook 3
- `ameriflux/` – final AmeriFlux CSV files from Notebook 4
- `micromet_reports/` – physical limit reports from the `Reformatter.finalize()` step

### Supporting Data

- `flux-met_processing_variables_*.csv` – master list of AmeriFlux variable names for validation
- Database API at `ugs-koop` providing `eddy_events` (station visit notes, program updates) and `eddy_station_metadata` (install dates)

### Site Dictionary

The notebooks use a station ID to folder name mapping:

| Station ID | Folder Name | Site Name |
|------------|-------------|-----------|
| US-UTD | Dugout_Ranch | Dugout Ranch |
| US-UTB | BSF | Bonneville Salt Flats |
| US-UTJ | Bluff | Bluff |
| US-UTW | Wellington | Wellington |
| US-UTE | Escalante | Escalante |
| US-UTM | Matheson | Matheson |
| US-UTP | Phrag | Phrag |
| US-CdM | Cedar_mesa | Cedar Mesa |
| US-UTV | Desert_View_Myton | Desert View Myton |
| US-UTN | Juab | Juab |
| US-UTG | Green_River | Green River |
| US-UTL | Pelican_Lake | Pelican Lake |

---

## Step 1 – Compile and Preprocess

**Notebook:** `dugout1_compile_and_preprocess.ipynb`
**Goal:** Compile data files from multiple sources for a single station and run through the MicroMet preprocessing pipeline. Gap-fill where possible between overlapping data sources.

### Parameters

- `interval`: Measurement interval in minutes (30 or 60). Controls which records are retained via `interval_updates`.
- `stationid`: AmeriFlux station identifier (e.g., `US-UTD`).
- `micromet_path`: Path to the MicroMet library source code.

### File Compilation

The first phase copies raw `.dat` files from individual download folders into an organized structure under the `compiled/` directory. The `file_compile.compile_files()` function searches source folders by regex pattern and copies matching files into target subfolders. Data types compiled include:

- `Statistics_Ameriflux` → `Statistics_AmeriFlux/`
- `Statistics_\d+` (raw TOA5 format) → `Statistics_Raw/` (requires Card Convert before use)
- `Flux_AmeriFluxFormat` → `AmeriFluxFormat/`
- `Flux_CSFormat` → `Flux_CSFormat/`
- `Operatn_Notes`, `Config_Setting_Notes`, `Flux_Notes` → respective folders

After compilation of raw Statistics files, they must be manually converted using Campbell Scientific Card Convert, then compiled again from `Statistics_Converted/` to `Statistics/`.

### Met Data Preprocessing

Two parallel met data streams are processed through the `preprocess_data()` function:

#### Statistics Tables (TOA5 format)

- Source folder: `{stationid}/Statistics/`
- Glob pattern: `TOA5*Statistics*.dat`
- Rows 0, 2, 3 are skipped (header metadata); `-9999` and `NAN` treated as missing.
- Column suffixes `_Avg` and `_Tot` are stripped from variable names.
- Passed through `micromet.Reformatter.preprocess()` which standardizes naming and timestamps.

#### Statistics AmeriFlux Tables

- Source folder: `{stationid}/Statistics_Ameriflux/`
- Glob pattern: `*Statistics_AmeriFlux*.dat`
- Already in AmeriFlux naming; no row skipping needed.
- Leaf wetness columns are checked and renamed for consistency (e.g., `LWMWET_1_1_2` → `LEAF_WET_1_2_1`).
- Columns that are entirely NA are dropped (artifact columns from corrupted files).

### Eddy Data Preprocessing

Two parallel eddy data streams are similarly processed:

#### AmeriFlux Format (from datalogger)

- Source folder: `{stationid}/AmeriFluxFormat/`
- Glob pattern: `*Flux_AmeriFluxFormat*.dat`
- All-NA columns dropped. Variable names validated against AmeriFlux master list using `validate.compare_names_to_ameriflux()`.

#### CS Format (from datalogger)

- Source folder: `{stationid}/Flux_CSFormat/`
- Glob pattern: `*_Flux_CSFormat*.dat`
- Contains additional diagnostic variables not in AmeriFlux format (e.g., `BOWEN_RATIO`, `ENERGY_CLOSURE`, `FC_MASS`, various density and QC fields).
- Specific columns are dropped (e.g., `WS_RSLT`, `SONIC_AZIMUTH`, `SUN_ELEVATION`) and others renamed (e.g., `CS65X_EC_1_1_1` → `EC_1_1_1`, `LI7700_AMB_TMPR` → `TA_1_1_5`).

### Validation Checks

- `validate.compare_names_to_ameriflux()` – flags any variable not in the AmeriFlux master list.
- `validate.validate_timestamp_consistency()` – confirms `DATETIME_END` and `TIMESTAMP_END` agree.
- Visual review via plotly interactive time series (`ed_plot.plotlystuff`).
- Comparison plots overlay AmeriFlux and CS Format eddy data to identify coverage gaps.

### Interval Subsetting

All datasets are passed through `interval_updates.subset_interval()` which filters to the target interval (30 or 60 minutes) based on a centralized dictionary (`interval_update_dict`). This handles sites that changed their reporting interval mid-record.

### Outputs

Four Parquet files are exported per station to `preprocessed_site_data/`:

- `{stationid}_{interval}_metstats_preprocessed.parquet`
- `{stationid}_{interval}_metstatsaf_preprocessed.parquet`
- `{stationid}_{interval}_eddyaf_dl_preprocessed.parquet`
- `{stationid}_{interval}_eddycsflux_dl_preprocessed.parquet`

---

## Step 2 – Create Raw Dataset

**Notebook:** `dugout2_create_raw_data.ipynb`
**Goal:** Combine the four preprocessed data sources into a single raw dataset per station, resolving overlaps, fixing time shifts, and validating alignment.

### Database Lookups

Station metadata and event logs are fetched from the UGS database API (`ugs-koop`) to retrieve:

- **Install date** – used to drop any data preceding the station installation.
- **Station visit notes** – provide context for data anomalies (e.g., bird nesting on sensors, IRGA zeroing events).
- **Program update notes** – track changes to datalogger programs (sampling intervals, calibration factors, sonic azimuth updates).

### Eddy Data Merging

Eddy data from AmeriFlux and CS Format sources are compared and merged:

1. Read preprocessed Parquet files and pass through `data_cleaning.prep_parquet()` for index standardization.
2. Run `validate.data_diff_check()` to compare values rounded to 3 decimal places. Small differences (<0.5%) are expected due to rounding. Larger differences (e.g., `G_1_1_A`, `FETCH_90`) are noted.
3. The CS Format data contributes unique columns not found in the AmeriFlux format (`G_PLATE` values, additional diagnostic fields like `FC_MASS`, `TKE`, `TSTAR`, `UX/UY/UZ` components).
4. `merge.fillna_with_second_df()` is used to fill gaps in the AmeriFlux data using CS Format values, prioritizing the AmeriFlux stream where both have data.

### Met Data Merging

Met data from the two Statistics table sources are compared and merged with special attention to the SoilVue time-shift issue.

#### SoilVue Time-Shift Detection and Correction

A known issue exists where SoilVue sensor data in the AmeriFlux Statistics tables may be offset by one time step (30 minutes). This is detected using cross-correlation analysis (`validate.review_lags()`) and corrected by shifting the affected columns:

- **Columns affected:** all `EC_3_*`, `K_3_*`, `SWC_3_*`, and `TS_3_*` variables (SoilVue profile data).
- The shift is applied using pandas `shift(freq='30min')` on the identified columns only.
- After shifting, lags are re-verified to confirm alignment (optimal lag should be 0).
- Non-SoilVue met variables (NETRAD, WD, WS, radiation) are not shifted.

### Time-Shift Between Met and Eddy

After merging met and eddy independently, the combined streams are checked for alignment using cross-correlation of NETRAD and WS between the two systems. If a systematic offset is detected (e.g., the met data was shifted by 1 hour for a portion of the record), the older portion of the met data is shifted to align.

### Final Combination and Export

- Duplicate columns between met and eddy (`FILE_NAME`, `TIMESTAMP`, `T_NR`) are renamed with `_EDDY` or `_MET` suffixes.
- Derived columns (`G_1_1_A`, `SG_1_1_A`) are dropped since they will be recalculated later.
- Timestamp integrity is verified: all records must fall on 30-minute boundaries.
- Column suffixes are standardized using `columns.create_suffix_map()` to append `_1_1_1` suffixes to variables that lack positional indices.
- Data before the station install date is dropped.

**Output:** `{stationid}_{timestart}_{timeend}_raw.parquet` in `final_database_tables/raw/`

---

## Step 3 – Quality Control

**Notebook:** `dugout3_qc_data.ipynb`
**Goal:** Apply calibration corrections, physical limits, signal-strength flags, and manual data cleaning to produce a QC-level dataset.

### Calibration Corrections

Site-specific calibration corrections are applied before running data through the MicroMet finalize step. Corrections are applied chronologically using date masks so that already-corrected data (post-program-update) is not double-corrected.

#### Soil Heat Flux Plate Calibration

- The SG (storage) values had an incorrect soil thickness parameter (0.16 m instead of 0.05 m). Values before the program fix date are multiplied by the correction factor (0.05/0.16 = 0.3125) using `fix_g_values.correct_vars_by_factor()`.
- New G values are recalculated as `G = SG + G_PLATE` for each plate using `fix_g_values.calculate_new_g_value()`.
- `G_PLATE_2` values were inverted for a period and must be multiplied by −1 before the correction date.

#### Precipitation Calibration

- The tipping bucket calibration factor was incorrect (0.1 instead of 0.254). Values before the program fix date are multiplied by 2.54.

### SoilVue G Calculation

A third soil heat flux estimate (`G_3_1_1`) is calculated from SoilVue temperature and moisture profiles using the `soil_heat` library:

- `soil_heat.storage_calculations.compute_soil_storage_integrated()` – computes heat storage (`SG_3_1_1`) in the top 5 cm.
- `soil_heat.soil_heat.compute_heat_flux_conduction()` – computes conductive heat flux at 5 cm depth using temperatures and moisture at 5 and 10 cm.
- `G_3_1_1 = SG_3_1_1 + G_SOILVUE`

**Important:** SWC values must be in proportions (0–1), not percent, at this stage. They are converted to percent during the finalize step.

### MicroMet Finalize

The `micromet.Reformatter.finalize()` function applies the following:

- Converts SWC from fraction to percent (multiply by 100).
- Scales SSITC test values to standard 0/1/2 encoding.
- Applies physical limits based on variable type (e.g., temperature ranges, radiation bounds). Values outside limits are set to NaN/−9999.
- Reorders columns to match AmeriFlux conventions.
- Produces a report CSV listing each variable, its limits, and the count/percentage of values flagged.

The report should be reviewed for variables with high flag percentages (e.g., >5%) which may indicate sensor issues.

### Common Data Issue Corrections

After the finalize step, several categories of manual corrections are applied:

#### Precipitation on Field Days

Precipitation events coinciding with station visits are reviewed. Spurious values caused by sensor maintenance (e.g., tipping the bucket during a sensor swap) are set to zero. Genuine rain events on visit days are preserved.

#### Ground Heat Flux Plate Zeros

`G_PLATE` values of exactly 0 are set to NaN along with their corresponding G values, as zeros typically indicate sensor disconnection rather than zero flux.

#### Soil Data Spikes

SoilVue data spikes are identified where `K_3_7_1` drops below a threshold (e.g., 3.5). All `EC_3`, `K_3`, `SWC_3`, and `TS_3` columns are set to NaN for the affected timestamps.

#### Wind Direction Corrections

- Sonic azimuth was recorded as 217° but measured as 227°. The 10° difference is added to `WD_1_1_1` for all data before the program update date.
- The Young anemometer (`WD_1_1_2`) was offset from the IRGASON by approximately 80°. This offset is subtracted.

#### Miscellaneous Corrections

- Barometric pressure spikes on specific dates set to NaN.
- Albedo set to NaN where `SW_IN` or `SW_OUT` are missing (prevents misleading calculated albedo).
- Early analog data issues (first few days after install) – radiation and soil variables set to NaN.
- Leaf wetness sensor #2 failure after a known date – all `LWM`/`LEAF_WET_1_2_1` values set to NaN.
- Footprint distance outliers capped (`FP_DIST_INTRST` > 1000 m and `UPWND_DIST_INTRST` < 180 m set to NaN).
- Temperature spikes on specific dates set to NaN.

### Signal Strength Flagging

Custom quality flags are created for H2O and CO2 based on IRGA signal strength:

#### H2O Signal Flag (`H2O_SIG_FLAG_1_1_1`)

- **Flag = 0:** signal strength ≥ 0.8 (good)
- **Flag = 1:** signal strength < 0.8 (marginal)
- **Flag = 2:** within a known continuous low-signal stretch (bad). These periods are identified from site visit logs and applied using `data_cleaning.apply_internal_flags()`.

#### CO2 Signal Flag (`CO2_SIG_FLAG_1_1_1`)

Same structure as the H2O flag, using the same known bad periods (typically both gases degrade simultaneously from window contamination).

#### Wind Direction Flag (`WD_1_1_1_FLAG`)

- **Flag = 0:** wind from the expected direction (32°–212°)
- **Flag = 1:** wind from behind the tower or from known obstruction directions

### Outputs

- Exported file: `{stationid}_{daterange}_qc.parquet` in `final_database_tables/qc/`
- Includes derived columns: `day_of_year`, `time_of_day`, `days_since_20240101` (for coloring regression plots).

---

## Step 3a – Variable Review

**Notebook:** `dugout3a_variable_review.ipynb`
**Goal:** Interactive exploration of the QC dataset to evaluate sensor agreement, data quality, and identify any remaining issues. This notebook does not modify data; corrections should be added back to Notebook 3.

### Review Categories

#### Net Radiation and Radiation Components

- Comparison of NETRAD, SW_IN, SW_OUT, LW_IN, LW_OUT between instruments 1 and 2 (CNR4 on eddy system vs. NR01 on met mast).
- Daily-mean regression to detect drift over time.
- Studentized residual plots to flag outlier days.
- PPFD_IN vs. SW_IN regression to verify PAR sensor consistency.
- Check for NETRAD records where component values are missing.

#### Albedo

- `ALB_1_1_1` vs. `ALB_1_1_2` regression colored by SW_IN, day of year, and time of day.
- Check for albedo values where SW_IN or SW_OUT is missing.

#### Wind Speed and Direction

- Time-lag detection between IRGASON and Young anemometer using cross-correlation.
- Linear regression of `WS_1_1_1` vs. `WS_1_1_2` colored by time.
- Monthly wind rose plots for both instruments to verify directional consistency.

#### Soil Heat Flux

- Comparison of `G_1_1_1`, `G_2_1_1`, `G_3_1_1` (two heat flux plates plus SoilVue-derived).
- SWC, TS, G_PLATE, SG intercomparisons.
- Daily mean regression for G values to assess plate-to-plate and plate-to-SoilVue agreement.

#### Temperature

- Comparison of five temperature sources: `T_SONIC`, `TA_1_1_1` (EC100), `TA_1_2_1` (sonic-derived), `TA_1_3_1` (EE08 aspirated), `TA_1_4_1` (secondary aspirated).
- Regression colored by H2O signal strength to detect IRGA-related temperature bias.

#### Relative Humidity and VPD

- Three RH sources compared; regression colored by H2O signal strength.
- Separate analysis for low-signal vs. high-signal periods and flag=2 vs. flag<2 periods.

#### Energy Balance Closure

- Calculated as `(H + LE)` vs. `(NETRAD – G)` using `G_3_1_1`.
- Daily closure analysis with filtering by record completeness (48/48 half-hours), signal strength, and flag status.

#### CO2 and H2O Concentrations

- Time series of `CO2_1_1_1`, `FC_1_1_1` colored by signal strength.
- Review of flagged periods to confirm they capture low-quality data.

---

## Step 3b – Plot Review

**Notebook:** `dugout3b_plot_review.ipynb`
**Goal:** Generate a quick interactive time-series plot of every variable in the QC dataset for a visual sweep. This serves as a rapid check for remaining spikes, gaps, or artifacts before exporting to AmeriFlux format.

The notebook iterates over all columns (excluding `FILE_NAME`, `stationid`) and calls `ed_plot.plotlystuff()` for each, producing interactive plotly figures with range sliders. It can be run on either the `raw` or `qc` data level by changing the `level` parameter.

---

## Step 4 – AmeriFlux Export

**Notebook:** `dugout4_ameriflux.ipynb`
**Goal:** Prepare the QC dataset for submission to the AmeriFlux network by dropping non-standard variables, applying final signal-strength filters, formatting timestamps, and exporting to CSV.

### Signal Strength Filtering

Variables derived from IRGA measurements are set to NaN where signal strength is below 0.8:

- **H2O signal < 0.8:** `H2O_1_1_1`, `H2O_SIGMA`, `LE_1_1_1`, `RH_1_1_1`, `RH_1_2_1`, `VPD_1_1_1`, `ET_1_1_1` set to NaN.
- **CO2 signal < 0.8:** `CO2_1_1_1`, `CO2_SIGMA`, `FC_1_1_1` set to NaN.

### Column Cleanup

1. Drop columns that are entirely NaN.
2. Compare all remaining column names against the AmeriFlux variable list. Columns not in the list are flagged for removal.
3. Manually review the drop list to confirm no needed variables are lost. `PBLH_F` is kept despite not matching the standard list.
4. Additional columns dropped: internal flags (`WD_1_1_1_FLAG`), file names, diagnostic fields (`FETCH_MAX`, `FETCH_90`, `ZL`), `stationid`, temporal helper columns.

### Remaining Variables

The final dataset retains approximately 80 variables including: flux variables (FC, LE, H, TAU with SSITC tests), radiation (SW/LW IN/OUT for two instruments, NETRAD, ALB, PPFD_IN), temperature (4 air temp sources, T_SONIC, T_CANOPY), humidity (3 RH sources, VPD), soil (G from 3 sources, SG from 3 sources, SWC at 9 SoilVue depths plus 2 CS probes, TS at 9 depths plus 2 CS probes), wind (WS and WD from 2 instruments, WS_MAX, U/V/W sigmas, USTAR, WD_SIGMA), and other (PA, P, MO_LENGTH, CO2/H2O concentrations and sigmas, LEAF_WET, PBLH_F).

### Final Formatting

- NaN values are replaced with −9999 (AmeriFlux missing value convention).
- `TIMESTAMP_START` and `TIMESTAMP_END` are recalculated from the datetime index using `timestamps.add_ameriflux_timestamps()` in `YYYYMMDDHHmm` format.
- The start timestamp is verified against the initial AmeriFlux submission to ensure continuity.

### Output

**File:** `{stationid}_HH_{timestamp_start}_{timestamp_end}.csv`

The `HH` prefix indicates half-hourly data. This file is ready for upload to the AmeriFlux data portal.

---

## Step 4b – AmeriFlux Plot Review

**Notebook:** `dugout4b_ameriflux_plots.ipynb`
**Goal:** Final visual review of the exported AmeriFlux CSV file. Every variable is plotted as an interactive time series to verify that the exported data looks correct and complete.

This reads the AmeriFlux CSV back in (converting −9999 to NaN for plotting), sets the datetime index, and loops through all columns generating plotly figures. This is the last visual check before submission.

---

## Step 5 – Flux QAQC with Energy Balance

**Notebook:** `dugout5_fluxqaqc.ipynb` (in `fluxqaqc/` subfolder)
**Goal:** Run the AmeriFlux data through the flux-data-qaqc package to perform energy balance ratio (EBR) correction, gap-fill ET, and produce daily summaries and diagnostic reports.

### Gap-Filling Redundant Sensors

Before running the QAQC, missing values in key variables are imputed using linear regression between redundant sensors:

- `NETRAD_1_1_1` gaps filled from `NETRAD_1_1_2` (and vice versa) with R² ≈ 0.988.
- `G_1_1_A` (mean of G plates 1 and 2) is computed where both are available. Gaps are filled from `G_3_1_1` (SoilVue-derived) with R² ≈ 0.589, and vice versa.
- Lag between `G_1_1_A` and `G_3_1_1` is verified (optimal lag of 2 periods noted).

### Flux-Data-QAQC Configuration

The analysis uses `.ini` configuration files that specify:

- Which column to use for Rn (net radiation) – e.g., `NETRAD_1_1_1_FINAL`
- Which column to use for G (ground heat flux) – e.g., `G_1_1_A_FINAL`
- Column mappings for LE, H, air temperature, wind speed, VPD, SWC, etc.

### QAQC Processing

The `QaQc` class processes the data with the following settings:

- `daily_frac = 1`: days with any missing sub-daily measurements are dropped.
- `max_interp_hours = 2`: maximum gap length to interpolate during daytime (Rn ≥ 0).
- `max_interp_hours_night = 4`: maximum gap length to interpolate during nighttime (Rn < 0).
- **Method:** Energy Balance Ratio (EBR) correction applied to LE.
- **ET gap-filling:** uses filtered ETrF multiplied by gridMET reference ET (ETr).

### Seasonal Analysis

The notebook subsets data by year and season (growing season: April 1 – October 31; winter: November 1 – March 31) to examine energy balance closure and ET patterns for individual periods.

### Sensitivity Testing

Multiple runs are performed with different input combinations to evaluate sensitivity:

- `NETRAD_1_1_1_FINAL` vs. `NETRAD_1_1_2_FINAL` as the Rn input.
- `G_1_1_A_FINAL` vs. `G_3_1_1_FINAL` as the G input.

Each combination produces a separate HTML report for comparison.

### Outputs

- HTML diagnostic reports with interactive bokeh plots showing daily energy balance components, closure ratios, and ET.
- Monthly status summaries showing counts of good data, gap-filled ET, and missing ET days.
- Optional CSV export of daily corrected data.

---

## Key Libraries and Dependencies

| Library | Role in Pipeline |
|---------|-----------------|
| **micromet** | Core processing library: `Reformatter` (preprocess, finalize), `validate`, `merge`, `data_cleaning`, `file_compile`, `interval_updates`, `columns`, `timestamps`, `fix_g_values`, `recalculate_albedo`, `eddy_plots`, `variable_limits` |
| **soil_heat** | SoilVue-derived ground heat flux: `storage_calculations`, `soil_heat` modules |
| **fluxdataqaqc** | Energy balance ratio correction, ET gap-filling, daily aggregation (`Data`, `QaQc`, `Plot` classes) |
| **pandas** | DataFrame operations, time-series indexing, Parquet/CSV I/O |
| **numpy** | Numerical operations, NaN handling |
| **scipy** | Statistical analysis (cross-correlation, linear regression) |
| **plotly** | Interactive time-series and scatter plots with range sliders |
| **matplotlib** | Static regression plots, wind roses |
| **bokeh** | Interactive plots in flux-data-qaqc HTML reports |
| **windrose** | Monthly wind rose generation in variable review |
| **prettytable** | Formatted display of station visit notes and program updates |
| **requests** | REST API calls to UGS database for metadata and events |

---

## Data Flow Diagram

```
Raw .dat files (Met Statistics + Met AmeriFlux Stats + Eddy AmeriFlux + Eddy CSFormat)
    |
    v  [Notebook 1: Compile & Preprocess]
4x *_preprocessed.parquet
    |
    v  [Notebook 2: Create Raw Data - merge, fix time shifts, validate alignment]
*_raw.parquet
    |
    v  [Notebook 3: QC - calibrations, physical limits, flags, manual corrections]
*_qc.parquet
    |
    +---> [Notebook 3a: Variable Review - regression, wind roses, closure]
    +---> [Notebook 3b: Plot Review - all variables time series]
    |     (feedback loop: corrections go back to Notebook 3)
    |
    v  [Notebook 4: AmeriFlux Export - drop cols, signal filter, format timestamps]
*_HH_*.csv  (AmeriFlux submission file)
    |
    +---> [Notebook 4b: AmeriFlux Plot Review - final visual check]
    |
    v  [Notebook 5: Flux QAQC - EBR correction, ET gap-fill, sensitivity tests]
EBR-corrected daily ET  +  HTML diagnostic reports
```

---

## Adapting the Workflow to Other Sites

The dugout notebooks serve as a template that can be copied and adapted for other UGS flux sites. The key site-specific elements that must be updated for each station are:

### Parameters to Update

- `stationid` – the AmeriFlux station code (e.g., `US-UTW` for Wellington).
- `interval` – may differ between sites or change over time.
- `date_range` – reflects the start/end timestamps of the available data.

### Site-Specific Corrections (Notebook 3)

Each site will have its own set of corrections that must be determined from station visit logs, program update records, and data review. Common categories include:

- Calibration correction dates and factors (soil, precipitation, etc.).
- Sensor failure periods and the specific variables to null.
- Wind direction offset between instruments.
- Signal strength bad-period date ranges.
- Manual spike removal dates.

### Column Handling (Notebooks 2 and 4)

The specific columns selected for merging from the CS Format eddy data (`csflux_join_cols` in Notebook 2) and the columns dropped before AmeriFlux export (Notebook 4) may vary by site depending on what sensors are installed.

### Flux QAQC Configuration (Notebook 5)

Each site needs its own `.ini` configuration file specifying column mappings for the flux-data-qaqc package. The choice of which NETRAD and G columns to use as primary inputs may also differ.
