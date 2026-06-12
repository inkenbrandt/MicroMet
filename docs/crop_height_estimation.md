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

### Step 1 -- Create Camera Time Lapse

Create monthly time lapse videos from station cameras and review videos for list of cut dates and approximate growing season start and end dates

1. Locate all image files with user-provided naming pattern within a user-provided folder
2. Compile images into monthly time series MP4s based on metadata timestamp
3. Manually review timeseries to identify major events, including cut dates and approximate growing season start and end dates. Note other events such as periods when fields were fallow or actively grazed. If camera view captures more than one field, separately record management events for each field.

**Output:** `{station}_{year}_{month}.mp4` and list of cuts dates

---

### Steps 2 and 3 -- Create NDVI Time Series Data (Optional)

Compile satellite data into time series files and obtain NDVI values and cut date estimates for individual fields.

1. Download Planet or other satellite imagery or modify script to pull data directly from Google Earth Engine. Imagery should include the full growing season (approximately March 15 to November 30).
2. Run `pivot_animation_executed`

**Output:**  `{station}_pivot_timeseries.mp4`, `{station}_pivot_ndvi_multizone_long.csv`, and `{station}_pivot_cut_events.csv.parquet`

---

### Step 3 -- Quality Control


**Output:** `{station}_{daterange}.csv`

---
