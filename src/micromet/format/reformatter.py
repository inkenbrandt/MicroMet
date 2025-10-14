"""
This module provides the Reformatter class for cleaning and standardizing
station data for flux/met processing, with integrated timestamp alignment checks.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import numpy as np

import micromet.format.reformatter_vars as reformatter_vars
import micromet.qaqc.variable_limits as variable_limits
from micromet.utils import logger_check
from micromet.format import transformers
from micromet.qaqc.netrad_limits import analyze_timestamp_alignment, flag_issues


class Reformatter:
    """
    A class to clean and standardize station data for flux/met processing.

    This class provides a pipeline for preparing raw station data by applying
    a series of transformations, including fixing timestamps, renaming columns,
    applying physical limits, and checking timestamp alignment.

    Parameters
    ----------
    var_limits_csv : str or Path, optional
        Path to a CSV file containing variable limits. If not provided,
        default limits are used.
    drop_soil : bool, optional
        If True, extra soil-related columns are dropped. Defaults to True.
    check_timestamps : bool, optional
        If True, perform timestamp alignment analysis on radiation data.
        Defaults to False.
    site_lat : float, optional
        Latitude of the site (required if check_timestamps=True).
    site_lon : float, optional
        Longitude of the site (required if check_timestamps=True).
    site_utc_offset : float, optional
        UTC offset in hours for the site (required if check_timestamps=True).
    logger : logging.Logger, optional
        A logger for tracking the reformatting process. If not provided,
        a default logger is used.

    Attributes
    ----------
    logger : logging.Logger
        The logger used for logging messages.
    config : dict
        A dictionary of configuration parameters for the reformatting process.
    varlimits : pd.DataFrame
        A DataFrame containing the physical limits for each variable.
    drop_soil : bool
        A flag indicating whether to drop extra soil columns.
    check_timestamps : bool
        A flag indicating whether to perform timestamp alignment checks.
    site_lat : float
        The latitude of the site.
    site_lon : float
        The longitude of the site.
    site_utc_offset : float
        The UTC offset of the site in hours.
    """

    def __init__(
        self,
        var_limits_csv: str | Path | None = None,
        drop_soil: bool = True,
        check_timestamps: bool = False,
        site_lat: float | None = None,
        site_lon: float | None = None,
        site_utc_offset: int = -7,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize the Reformatter.

        Parameters
        ----------
        var_limits_csv : str or Path, optional
            Path to a CSV file containing variable limits.
        drop_soil : bool, optional
            If True, extra soil-related columns are dropped. Defaults to True.
        check_timestamps : bool, optional
            If True, perform timestamp alignment analysis. Defaults to False.
        site_lat : float, optional
            Latitude of the site (required if check_timestamps=True).
        site_lon : float, optional
            Longitude of the site (required if check_timestamps=True).
        site_utc_offset : float, optional
            UTC offset in hours (required if check_timestamps=True).
        logger : logging.Logger, optional
            A logger for tracking the reformatting process.
        """
        self.logger = logger_check(logger)
        self.config = reformatter_vars.config
        
        if var_limits_csv is None:
            self.varlimits = variable_limits.limits
        else:
            if isinstance(var_limits_csv, str):
                var_limits_csv = Path(var_limits_csv)
            self.varlimits = pd.read_csv(
                var_limits_csv, index_col=0, na_values=["-9999", "NAN", "NaN", "nan"]
            )
            self.logger.debug(f"Loaded variable limits from {var_limits_csv}")

        self.drop_soil = drop_soil
        self.check_timestamps = check_timestamps
        self.site_lat = site_lat
        self.site_lon = site_lon
        self.site_utc_offset = site_utc_offset
        
        # Validate timestamp check parameters
        if self.check_timestamps:
            if any(x is None for x in [site_lat, site_lon, site_utc_offset]):
                raise ValueError(
                    "site_lat, site_lon, and site_utc_offset are required when "
                    "check_timestamps=True"
                )

    def prepare(self, df, data_type="eddy"):
        """Current method - keep for backward compatibility"""
        return self.process(df, data_type=data_type)

    def process(
        self, df: pd.DataFrame, data_type: str = "eddy"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Dict]]:
        """
        Prepare the data by applying a series of cleaning and standardization steps.

        This method takes a DataFrame of station data and applies a pipeline of
        transformations to clean and standardize it. The steps include fixing
        timestamps, renaming columns, setting numeric types, resampling,
        applying physical limits, and optionally checking timestamp alignment.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame of station data.
        data_type : str, optional
            The type of data being processed (e.g., 'eddy', 'met'). This is
            used to determine which column renaming map to use.
            Defaults to 'eddy'.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, dict | None]
            A tuple containing:
            - The prepared DataFrame with standardized and cleaned data.
            - A report DataFrame detailing the changes made during the
              application of physical limits.
            - A dictionary with timestamp alignment results (if check_timestamps=True),
              or None otherwise. Contains keys: 'summary', 'composites', 'flags'.
        """
        self.logger.info("Starting reformat (%s rows)", len(df))

        # Standard pipeline
        df = df.pipe(transformers.fix_timestamps, logger=self.logger)
        df = df.pipe(transformers.rename_columns, data_type=data_type, 
                     config=self.config, logger=self.logger)
        df = df.pipe(transformers.make_unique_cols)
        df = df.pipe(transformers.set_number_types, logger=self.logger)
        df = df.pipe(transformers.resample_timestamps, logger=self.logger)
        df = df.pipe(transformers.timestamp_reset)
        df = df.pipe(transformers.fill_na_drop_dups)

        df = df.pipe(transformers.apply_fixes, logger=self.logger)
        # note that important to apply_fixes and rename_columns before running physical limits!
        df, mask, report = transformers.apply_physical_limits(df)

        # Timestamp alignment check (if enabled and radiation data available)
        timestamp_results = None
        if self.check_timestamps:
            timestamp_results = self._check_timestamp_alignment(df)

        if self.drop_soil:
            df = df.pipe(transformers.drop_extra_soil_columns, 
                        config=self.config, logger=self.logger)

        df = df.pipe(transformers.drop_extras, config=self.config).fillna(
            transformers.MISSING_VALUE
        )
        df = df.pipe(transformers.col_order, logger=self.logger)

        self.logger.info("Done; final shape: %s", df.shape)
        return df, report, timestamp_results

    def _check_timestamp_alignment(self, df: pd.DataFrame) -> Dict | None:
        """
        Perform timestamp alignment analysis on radiation data.

        This method analyzes SW_IN and/or PPFD_IN against theoretical 
        top-of-atmosphere radiation to detect potential timestamp issues
        such as timezone errors, DST problems, or sensor issues.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame with processed data.

        Returns
        -------
        dict | None
            A dictionary containing:
            - 'summary': DataFrame with window-by-window analysis results
            - 'composites': Dictionary of WindowComposite objects
            - 'flags': Dictionary of detected issues
            Returns None if radiation columns are not available.
        """
        # Check if we have the necessary radiation columns
        has_sw = 'SW_IN' in df.columns or 'SW_IN_1_1_1' in df.columns
        has_ppfd = 'PPFD_IN' in df.columns or 'PPFD_IN_1_1_1' in df.columns
        
        if not (has_sw or has_ppfd):
            self.logger.warning(
                "SW_IN and PPFD_IN not found in data; "
                "skipping timestamp alignment check"
            )
            return None
        
        self.logger.info("Performing timestamp alignment analysis...")
        
        try:
            # Run the analysis
            summary, composites = analyze_timestamp_alignment(
                df,
                lat=self.site_lat,
                lon=self.site_lon,
                std_utc_offset_hours=self.site_utc_offset,
                time_from='END',  # Since we use TIMESTAMP_END
                sw_col='SW_IN' if 'SW_IN' in df.columns else 'SW_IN_1_1_1',
                ppfd_col='PPFD_IN' if 'PPFD_IN' in df.columns else 'PPFD_IN_1_1_1',
                assume_naive_is_local=False,
            )
            
            # Flag potential issues
            flags = flag_issues(summary)
            
            # Log any detected issues
            if flags:
                self.logger.warning("Timestamp alignment issues detected:")
                for issue_type, message in flags.items():
                    self.logger.warning(f"  {issue_type}: {message}")
            else:
                self.logger.info("No significant timestamp alignment issues detected")
            
            return {
                'summary': summary,
                'composites': composites,
                'flags': flags
            }
            
        except Exception as e:
            self.logger.error(f"Error during timestamp alignment check: {e}")
            return None