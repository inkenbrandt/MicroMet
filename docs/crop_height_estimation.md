# Crop Height Estimation-- Workflow Summary

**Utah Geological Survey -- Flux Monitoring Network**

This page provides an overview of the auxillary workflow for estimating crop height based on cut dates and periodic field measurements. The workflow optionally includes NDVI analysis from satellite imagery to help identify cut dates.

---

## Workflow at a Glance

The workflow uses a list of crop cut dates, field crop height measurements, and climate data to model crop height over the growing season. The user can set basic parameters like minimum and maximum crop heights and the model type (i.e., linear, logistic, exponential) to suit the modeled crop. The user can also adjust parameters for each site and growing season including growing season dates, k rates, and whether the crop is grazed to obtain a better fit with field observations.

Optionally, the user can analyze satellite data from Planet, Sentinel, or another source to generate true color and NDVI time series to help identify cut dates and growing season start and end dates.


| Step | Notebook | Input | Key Operations | Output |
|------|----------|-------|---------------|--------|
| **1** | `create_camera_time_lapse` | Camera imagery | Compile data by month · Create MP4 files  | Monthly time lapse files |
| **2** | `pivot_animation_executed` | Geojson of study area boundary · Planet `analytic_sr` timeseries (could be modified for other satellite data) | Compile satellite imagery · Calculate NDVI for each image · Create time series  | Animated GIF and MP4 true color and NDVI time series |
| **3** | `pivot_ndvi_analysis` | Geojson of one or more fields · Planet `analytic_sr` timeseries (could be modified for other satellite data) | NDVI calculation · Cut date estimates  | Various files including pivot_ndvi_multizone_long.csv and pivot_cut_events.csv |
| **4**| `crop_height_example_alfalfa` or `crop_height_example_corn` | Crop cut dates · Crop height measurements  ·  Daily climate data  | Crop height estimation | Crop height estimates  |

---

## Step-by-Step Summary

### Step 1 -- Create camera time lapse

Create monthly time lapse videos from station cameras and review videos for list of cut dates and approximate growing season start and end dates

1. Locate all image files with user-provided naming pattern within a user-provided folder
2. Compile images into monthly time series MP4s based on metadata timestamp
3. Manually review timeseries to identify major events, including cut dates and approximate growing season start and end dates. Note other events such as periods when fields were fallow or actively grazed. If camera view captures more than one field, separately record management events for each field.

**Output:** `{station}_{year}_{month}.mp4` and list of cuts dates

---

### Steps 2 and 3 -- Create Raw Dataset

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
