"""
This module provides the Reformatter class for cleaning and standardizing
station data for flux/met processing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import math

import pandas as pd
import numpy as np
import yaml
from importlib.resources import files

import micromet.format.reformatter_vars as reformatter_vars
import micromet.qaqc.variable_limits as variable_limits
from micromet.utils import logger_check
from micromet.format import transformers 

class Reformatter:
    """
    A class to clean and standardize station data for flux/met processing.

    This class provides a pipeline for preparing raw station data by applying
    a series of transformations, including fixing timestamps, renaming columns,
    applying physical limits, and dropping unnecessary columns.

    Parameters
    ----------
    var_limits_csv : str or Path, optional
        Path to a CSV file containing variable limits. If not provided,
        default limits are used.
    drop_soil : bool, optional
        If True, extra soil-related columns are dropped. Defaults to True.
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
    """

    def __init__(
        self,
        var_limits_csv: str | Path | None = None,
        drop_soil: bool = True,
        logger: logging.Logger | None= None,
    ):
        """
        Initialize the Reformatter.

        Parameters
        ----------
        var_limits_csv : str or Path, optional
            Path to a CSV file containing variable limits. If not provided,
            default limits are used.
        drop_soil : bool, optional
            If True, extra soil-related columns are dropped. Defaults to True.
        logger : logging.Logger, optional
            A logger for tracking the reformatting process. If not provided,
            a default logger is used.
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

    def prepare(self, df, data_type="eddy"):
        """Current method - keep for backward compatibility"""
        return self.process(df, data_type=data_type)

    def process(
        self, df: pd.DataFrame, data_type: str = "eddy"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare the data by applying a series of cleaning and standardization steps.

        This method takes a DataFrame of station data and applies a pipeline of
        transformations to clean and standardize it. The steps include fixing
        timestamps, renaming columns, setting numeric types, resampling,
        applying physical limits, and more.

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
        tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing:
            - The prepared DataFrame with standardized and cleaned data.
            - A report DataFrame detailing the changes made during the
              application of physical limits.
        """
        self.logger.info("Starting reformat (%s rows)", len(df))

        df = df.pipe(transformers.fix_timestamps, logger=self.logger)
        df = df.pipe(transformers.rename_columns, data_type=data_type, config=self.config, logger=self.logger)
        df = df.pipe(transformers.make_unique_cols)
        df = df.pipe(transformers.set_number_types, logger=self.logger)
        df = df.pipe(transformers.resample_timestamps, logger=self.logger)
        df = df.pipe(transformers.timestamp_reset)
        df = df.pipe(transformers.fill_na_drop_dups)

        df = df.pipe(transformers.apply_fixes, logger=self.logger)
        df, mask, report = transformers.apply_physical_limits(df)

        if self.drop_soil:
            df = df.pipe(transformers.drop_extra_soil_columns, config=self.config, logger=self.logger)

        df = df.pipe(transformers.drop_extras, config=self.config).fillna(transformers.MISSING_VALUE)
        df = df.pipe(transformers.col_order, logger=self.logger)

        self.logger.info("Done; final shape: %s", df.shape)
        return df, report
