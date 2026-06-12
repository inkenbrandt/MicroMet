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

1. Manually download Planet or other satellite imagery or modify script to pull data directly from Google Earth Engine. Imagery should include the full growing season (approximately March 15 to November 30).
2. Run `pivot_animation_executed`, using a geojson file of the same extent as the mask used to download the satellite data.
3. Manually review time lapse videos to identify any additional cut dates or management notes.
4. Manually estimate how many individual fields should be associated with the site. It is helpful to estimate the 80 fetch at the time when the crop was just cut ((irgason height minus about 3 cm) * 75) to determine which fields overlap the station footprint. Sometimes one larger field needs to be divided into smaller subfields if they are managed very differently (based on time lapse videos or other information). Create a geojson file with features for each field or subfield and a field called ID or Name with the unique identifer for the field.
5. Run `pivot_ndvi_analysis`, using geojson file of the different crop zones. Examine outputs and determine whether any fields have near-identical patterns, in which case you could optionally rerun as a single field.

A note about NDVI crop height estimates. NDVI-derived estimates for crop height did not capture reality well for several reasons at UFN sites. First, alfalfa fields are dense and green shortly after they are cut when heights are still low, but the NDVI model assumes they have grown quickly. Second, alfalfa fields may be green for a day or two after a cut occurs as the hay is piled on the fields, so the NDVI analysis often cannot pinpoint the exact cut date. Third, late-season alfalfa that is stunted or grazed is not well captured by the NDVI model. Use the height estimates with caution

**Output:**  `{station}_pivot_timeseries.mp4`, `{station}_pivot_ndvi_multizone_long.csv`, and `{station}_pivot_cut_events.csv.parquet`

---

### Step 3 -- Estimate Crop Height

Use the simulate_alfalfa_height_single_field function to estimate crop height. Function was designed for crops like alfalfa but will work for corn as well. Example notebooks include two years of alfalfa crop height estimates at one station using an exponential model and one year of corn crop height estimates using a logistic model.

1. Download climate data from [PRIMS](https://prism.oregonstate.edu/explorer/) for the alfalfa crop height example or any model using growing degree days.
2. Input dataframe of field crop height measurements into processing notebook
3. Finalize list of cut dates from combination of dates identified in imagery and NDVI time series analysis (if available). Cut dates will have a sharp decline in NDVI whereas a gradual decline can indicate late-season die off or grazing effects. NDVI cut date estimates will often have to be pared or corrected to match camera dates or known reality.
4. Run cut dates and climate data through the simulate_alfalfa_height_single_field function. Set initial starting parameters and adjust as needed to fit field measurements, photo observations, and likely reality (e.g., farmer would not cut alfalfa at only 20 cm). Key parameters include:
    - growing season start: Start at March 15 and adjust as needed. The model will use growing degree days to start growth when conditions are appropriate. However, sometimes growth starts earlier than expected due to a warm winter. If this conflicts with site photos or other data, you can move the start date to when green-up actually started.
   - growing season end: Start at November 30th and adjust as needed. Crop height will decline to the user-set minimum crop height by this date. The model will sometimes end all crop growth after the last cut date, but usually it is valuable to set this parameter based on camera or satellite images or other information.
   - k rate: List of one or more k rates (growth rate constants). K rates are applied in order before and between each cutting period. Program defaults to the most recently used rate if there are fewer rates than periods between cuttings. Test model with a single rate and then adjust as needed. The rate is often faste rin the first growth period than in subsequent periods, likely due to water stress. In the absence of field measurements, consider defaulting to the previous value or set rate to obtain maximum height similar to that of previous time periods.
   - grazing parameter: Optional, listed as None or a list of maximum height values to set the crop at during periods of grazing (e.g., None or [None, None, 25] if the period after the last cut should be set to a max height of 25 cm). Grazing heights in the fall can often by simulated with k rate adjustment alone, but sometimes this parameter is needed when grazing occurs over a longer period of time or during dry conditions. This parameter could also be used to simulate stunted crop height due to lack of water that peaks at a lower rate within the year
5. If there is a second field that intersects the station footprint, run the generate_field2_heights function. This only works on two fields that are managed similarly but with cut dates a few days apart.
6. Export all crop height data along with final cut dates and model parameters

**Output:** `{station}_{daterange}_crop_height.csv`, cut dates, model parameters

---
