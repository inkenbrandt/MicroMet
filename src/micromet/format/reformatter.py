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
    Clean and standardize station data.
    """

    def __init__(
        self,
        var_limits_csv: str | Path | None = None,
        drop_soil: bool = True,
        logger: logging.Logger = None,
    ):
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

    def prepare(
        self, df: pd.DataFrame, data_type: str = "eddy"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.logger.info("Starting reformat (%s rows)", len(df))

        df = df.pipe(transformers.fix_timestamps, logger=self.logger)
        df = df.pipe(transformers.rename_columns, data_type=data_type, config=self.config, logger=self.logger)
        df = df.pipe(transformers.make_unique_cols)
        df = df.pipe(transformers.set_number_types, logger=self.logger)
        df = df.pipe(transformers.resample_timestamps, logger=self.logger)
        df = df.pipe(transformers.timestamp_reset)
        df = df.pipe(transformers.fill_na_drop_dups, logger=self.logger)

        df, mask, report = transformers.apply_physical_limits(df)

        df = df.pipe(transformers.apply_fixes, logger=self.logger)

        if self.drop_soil:
            df = df.pipe(transformers.drop_extra_soil_columns, config=self.config, logger=self.logger)

        df = df.pipe(transformers.drop_extras, config=self.config).fillna(transformers.MISSING_VALUE)
        df = df.pipe(transformers.col_order, logger=self.logger)

        self.logger.info("Done; final shape: %s", df.shape)
        return df, report
