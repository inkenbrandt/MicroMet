"""
This module provides the AmerifluxDataProcessor class for reading and parsing
AmeriFlux-style CSV files (TOA5 or AmeriFlux output) into a pandas DataFrame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from micromet.utils import logger_check
from micromet.station_info import site_folders, loggerids


class AmerifluxDataProcessor:
    """
    Read Campbell Scientific TOA5 or AmeriFlux output CSV into a tidy DataFrame.

    Parameters
    ----------
    path : str | Path
        File path to CSV.
    config_path : str | Path
        File path to YAML configuration file for header names. Defaults to 'reformatter_vars.yml'
    logger : logging.Logger
        Logger to use.
    """

    _TOA5_PREFIX = "TOA5"
    _HEADER_PREFIX = "TIMESTAMP_START"
    NA_VALUES = ["-9999", "NAN", "NaN", "nan", np.nan, -9999.0]

    def __init__(
        self,
        logger: logging.Logger = None,  # type: ignore
    ):
        self.logger = logger_check(logger)
        self.skip_rows = 0

    def to_dataframe(self, file: Union[str, Path]) -> pd.DataFrame:
        """Return parsed CSV as pandas DataFrame."""
        self._determine_header_rows(file)  # type: ignore
        self.logger.debug("Reading %s", file)
        df = pd.read_csv(
            file,
            skiprows=self.skip_rows,
            names=self.names,
            na_values=self.NA_VALUES,
        )
        return df

    def _determine_header_rows(self, file: Path) -> None:
        """
        Examine the first line to decide if file is TOA5 or already processed.

        TOA5 files begin with literal 'TOA5'.
        AmeriFlux standard Level‑2 output has no prefix, just column labels.
        """
        with file.open("r") as fp:
            first_line = fp.readline().strip().replace('"', "").split(",")
            second_line = fp.readline().strip().replace('"', "").split(",")
        if first_line[0] == self._HEADER_PREFIX:
            self.logger.debug(f"Header row detected: {first_line}")
            self.skip_rows = 1
            self.names = first_line
        elif first_line[0] == self._TOA5_PREFIX:
            self.logger.debug(f"TOA5 header detected: {first_line}")
            self.skip_rows = [0, 1, 2, 3]
            self.names = second_line
        else:
            raise RuntimeError(f"Header line not recognized: {first_line}")
        self.logger.debug(f"Skip rows for set to {self.skip_rows}")

    def _get_FILE_NO(self, file: Path) -> tuple[int, int]:
        basename = file.stem

        try:
            file_number = int(basename.split("_")[-1])
            datalogger_number = int(basename.split("_")[0])
        except ValueError:
            file_number = datalogger_number = -9999
        self.logger.debug(f"{file_number} -> {datalogger_number}")
        return file_number, datalogger_number

    def raw_file_compile(
        self,
        main_dir: Union[str, Path],
        station_folder_name: Union[str, Path],
        search_str: str = "*Flux_AmeriFluxFormat*.dat",
    ) -> Optional[pd.DataFrame]:
        """
        Compiles raw AmeriFlux datalogger files into a single dataframe.
        """
        compiled_data = []
        station_folder = Path(main_dir) / station_folder_name
        self.logger.info(f"Compiling data from {station_folder}")

        for file in station_folder.rglob(search_str):
            self.logger.info(f"Processing file: {file}")
            FILE_NO, datalogger_number = self._get_FILE_NO(file)
            df = self.to_dataframe(file)
            if df is not None:
                df["FILE_NO"] = FILE_NO
                df["DATALOGGER_NO"] = datalogger_number
                compiled_data.append(df)

        if compiled_data:
            compiled_df = pd.concat(compiled_data, ignore_index=True)
            return compiled_df
        else:
            self.logger.warning(f"No valid files found in {station_folder}")
            return None

    def iterate_through_stations(self):
        """Iterate through all stations."""
        data = {}
        for stationid, folder in site_folders.items():
            for datatype in ["met", "eddy"]:
                if datatype == "met":
                    station_table_str = "Statistics_Ameriflux"
                else:
                    station_table_str = "AmeriFluxFormat"
                if stationid in loggerids[datatype]:
                    for loggerid in loggerids[datatype][stationid]:
                        search_str = f"{loggerid}*{station_table_str}*.dat"
                        data[stationid] = self.raw_file_compile(
                            stationid,
                            folder,
                            search_str,
                        )
        return data
