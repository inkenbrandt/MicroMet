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
    A class for reading and parsing AmeriFlux-style CSV files.

    This class is designed to handle Campbell Scientific TOA5 files or
    standard AmeriFlux output files, parsing them into a pandas DataFrame.

    Parameters
    ----------
    logger : logging.Logger, optional
        A logger for tracking the data processing. If not provided, a
        default logger is used.

    Attributes
    ----------
    logger : logging.Logger
        The logger used for logging messages.
    skip_rows : int or list of int
        The number of rows to skip at the beginning of the file.
    names : list of str
        The column names for the DataFrame.
    """

    _TOA5_PREFIX = "TOA5"
    _HEADER_PREFIX = "TIMESTAMP_START"
    NA_VALUES = ["-9999", "NAN", "NaN", "nan", np.nan, -9999.0]

    def __init__(
        self,
        logger: logging.Logger = None,  # type: ignore
    ):
        """
        Initialize the AmerifluxDataProcessor.

        Parameters
        ----------
        logger : logging.Logger, optional
            A logger for tracking the data processing. If not provided, a
            default logger is used.
        """
        self.logger = logger_check(logger)
        self.skip_rows = 0

    def to_dataframe(self, file: Union[str, Path]) -> pd.DataFrame:
        """
        Read an AmeriFlux-style CSV file and return it as a pandas DataFrame.

        This method first determines the header structure of the file and
        then reads the data into a DataFrame, handling missing values.

        Parameters
        ----------
        file : str or Path
            The path to the CSV file to be read.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the parsed data from the file.
        """
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
        Determine the header structure of the input file.

        This method examines the first few lines of the file to determine
        if it is a TOA5 file or a standard AmeriFlux output file, and sets
        the appropriate `skip_rows` and `names` attributes.

        Parameters
        ----------
        file : Path
            The path to the file to be examined.

        Raises
        ------
        RuntimeError
            If the header format is not recognized.
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
        """
        Extract the file number and datalogger number from the filename.

        This method parses the filename to extract a file number and a
        datalogger number, which are assumed to be part of the filename.

        Parameters
        ----------
        file : Path
            The path to the file.

        Returns
        -------
        tuple[int, int]
            A tuple containing the file number and datalogger number.
            Returns (-9999, -9999) if parsing fails.
        """
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
        Compile raw AmeriFlux datalogger files into a single DataFrame.

        This method searches for files matching a given pattern within a
        station's directory, processes each file, and concatenates them
        into a single DataFrame.

        Parameters
        ----------
        main_dir : str or Path
            The main directory containing the station folders.
        station_folder_name : str or Path
            The name of the station folder.
        search_str : str, optional
            The search string (glob pattern) for finding files to compile.
            Defaults to "*Flux_AmeriFluxFormat*.dat".

        Returns
        -------
        pd.DataFrame or None
            A DataFrame containing the compiled data, or None if no valid
            files were found.
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
        """
        Iterate through all stations and compile their data.

        This method iterates through a predefined list of stations,
        compiles the data for each station, and returns a dictionary
        of DataFrames.

        Returns
        -------
        dict
            A dictionary where keys are station IDs and values are
            DataFrames of the compiled data for each station.
        """
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
