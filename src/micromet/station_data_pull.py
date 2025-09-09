import requests
import datetime
import logging
from typing import Union, Tuple, Optional

from requests.auth import HTTPBasicAuth
import pandas as pd
from io import BytesIO
import configparser
import sqlalchemy
from micromet.format.reformatter import Reformatter

micromet_version = "0.2.1"


def logger_check(logger: logging.Logger | None) -> logging.Logger:
    """
    Initialize and return a logger instance if none is provided.

    This function checks if a logger object is provided. If not, it
    creates a new logger with a default warning level and a stream
    handler that outputs to the console.

    Parameters
    ----------
    logger : logging.Logger or None
        An existing logger instance. If None, a new logger is created.

    Returns
    -------
    logging.Logger
        A configured logger instance.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.WARNING)

        # Create console handler and set level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(ch)

    return logger


class StationDataDownloader:
    """
    A class to manage station data operations including fetching, processing, and database interactions.
    """

    def __init__(
        self,
        config: Union[configparser.ConfigParser, dict],
        logger: logging.Logger = None,
    ):
        """
        Initialize the StationDataManager.

        Parameters
        ----------
        config : Union[configparser.ConfigParser, dict]
            A configuration object containing station details and
            credentials.
        logger : logging.Logger, optional
            A logger for logging messages. If None, a new logger is
            created.
        """
        self.config = config
        self.logger = logger_check(logger)

        self.logger_credentials = HTTPBasicAuth(
            config["LOGGER"]["login"], config["LOGGER"]["pw"]
        )

    def _get_port(self, station: str, loggertype: str = "eddy") -> int:
        """
        Get the port number for a given station and logger type.

        Parameters
        ----------
        station : str
            The identifier for the station.
        loggertype : str, optional
            The type of logger ('eddy' or 'met'). Defaults to 'eddy'.

        Returns
        -------
        int
            The port number for the specified station and logger type.
        """
        port_key = f"{loggertype}_port"
        return int(self.config[station].get(port_key, 80))

    def get_times(
        self, station: str, loggertype: str = "eddy"
    ) -> Tuple[Optional[str], str]:
        """
        Retrieve the current time from the logger and the system.

        This method queries a station's logger for its current time and
        also gets the current system time for comparison.

        Parameters
        ----------
        station : str
            The identifier for the station.
        loggertype : str, optional
            The type of logger ('eddy' or 'met'). Defaults to 'eddy'.

        Returns
        -------
        Tuple[Optional[str], str]
            A tuple containing the logger's current time as a string and
            the system's current time as a string.

        Notes
        -----
        For more details on the web API commands, see:
        https://help.campbellsci.com/crbasic/cr6/Content/Info/webserverapicommands1.htm
        """
        ip = self.config[station]["ip"]
        port = self._get_port(station, loggertype)
        clk_url = f"http://{ip}:{port}/?"
        clk_args = {
            "command": "ClockCheck",
            "uri": "dl",
            "format": "json",
        }

        clktimeresp = requests.get(
            clk_url, params=clk_args, auth=self.logger_credentials
        ).json()

        clktime = clktimeresp.get("time")
        comptime = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"

        return clktime, comptime

    @staticmethod
    def get_station_id(stationid: str) -> str:
        """
        Extract the station ID from a full station identifier string.

        Parameters
        ----------
        stationid : str
            The full station identifier (e.g., 'US-ABC').

        Returns
        -------
        str
            The extracted station ID (e.g., 'ABC').
        """
        return stationid.split("-")[-1]

    def download_from_station(
        self,
        station: str,
        loggertype: str = "eddy",
        mode: str = "since-time",
        p1: str = "0",
        p2: str = "0",
    ):
        """
        Download data from a station's logger.

        This method constructs a request to the station's web API to
        download data based on the specified parameters.

        Parameters
        ----------
        station : str
            The identifier for the station.
        loggertype : str, optional
            The type of logger ('eddy' or 'met'). Defaults to 'eddy'.
        mode : str, optional
            The data query mode. Can be 'since-time', 'most-recent',
            'since-record', 'date-range', or 'Backfill'. Defaults to
            'since-time'.
        p1 : str, optional
            The primary parameter for the query, such as a start time
            or record number. Defaults to "0".
        p2 : str, optional
            The secondary parameter for the query, such as an end time.
            Defaults to "0".

        Returns
        -------
        Tuple[Optional[pd.DataFrame], Optional[float], int]
            A tuple containing the downloaded data as a DataFrame, the
            size of the data packet in MB, and the HTTP status code.

        Notes
        -----
        For more details on the web API commands, see:
        https://help.campbellsci.com/crbasic/cr6/Content/Info/webserverapicommands1.htm
        """

        ip = self.config[station]["ip"]
        port = self._get_port(station, loggertype)
        tabletype = (
            "Flux_AmeriFluxFormat" if loggertype == "eddy" else "Statistics_AmeriFlux"
        )

        url = f"http://{ip}:{port}/tables.html?"
        params = {
            "command": "DataQuery",
            "mode": f"{mode}",
            "format": "toA5",
            "uri": f"dl:{tabletype}",
        }

        if p1 == "0" or p1 == 0:
            params["p1"] = "0"
        else:
            params["p1"] = p1

        if p2 == "0" or p2 == 0:
            if mode == "since-time":
                params["p1"] = (
                    f"{datetime.datetime.now() - datetime.timedelta(days=10):%Y-%m-%d}"
                )

        else:
            params["p2"] = p2

        response = requests.get(url, params=params, auth=self.logger_credentials)

        if response.status_code == 200:
            raw_data = pd.read_csv(BytesIO(response.content), skiprows=[0, 2, 3])
            pack_size = len(response.content) * 1e-6
            return raw_data, pack_size, response.status_code
        else:
            self.logger.error(f"Error downloading from station: {response.status_code}")
            return None, None, response.status_code


class StationDataProcessor(StationDataDownloader):
    def __init__(
        self,
        config: Union[configparser.ConfigParser, dict],
        engine: sqlalchemy.engine.base.Engine,
        logger: logging.Logger = None,
    ):
        """
        Initialize the StationDataProcessor.

        Parameters
        ----------
        config : Union[configparser.ConfigParser, dict]
            A configuration object with station details.
        engine : sqlalchemy.engine.base.Engine
            A SQLAlchemy engine for database connections.
        logger : logging.Logger, optional
            A logger for logging messages.
        """

        super().__init__(config, logger)
        self.config = config
        self.engine = engine
        self.logger = logger_check(logger)

    def get_station_data(
        self,
        station: str,
        reformat: bool = True,
        loggertype: str = "eddy",
        config_path: str = "./data/reformatter_vars.yml",
        var_limits_csv: str = "./data/extreme_values.csv",
        drop_soil: bool = False,
    ) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
        """
        Fetch and process data for a single station.

        This method downloads data from a station, optionally reformats
        it, and returns the processed data.

        Parameters
        ----------
        station : str
            The identifier for the station.
        reformat : bool, optional
            Whether to reformat the downloaded data. Defaults to True.
        loggertype : str, optional
            The type of logger ('eddy' or 'met'). Defaults to 'eddy'.
        config_path : str, optional
            The path to the reformatter configuration file.
        var_limits_csv : str, optional
            The path to the variable limits CSV file.
        drop_soil : bool, optional
            Whether to drop soil-related data. Defaults to False.

        Returns
        -------
        Tuple[Optional[pd.DataFrame], Optional[float]]
            A tuple containing the processed DataFrame and the size of
            the downloaded data packet in MB.
        """
        last_date = self.get_max_date(station, loggertype)
        raw_data, pack_size, status_code = self.download_from_station(
            station,
            loggertype=loggertype,
            mode="since-time",
            p1=f"{last_date:%Y-%m-%d}",
        )
        if status_code == 200:
            if raw_data is not None and reformat:
                am_data = Reformatter(
                    config_path=config_path,
                    var_limits_csv=var_limits_csv,
                    drop_soil=drop_soil,
                )
                am_df = am_data.prepare(raw_data)
                # am_data = Reformatter(raw_data)
                # am_df = am_data.et_data
            else:
                am_df = raw_data

            return am_df, pack_size

        self.logger.error(f"Error fetching station data: {status_code}")
        return None, None

    @staticmethod
    def remove_existing_records(
        df: pd.DataFrame,
        column_to_check: str,
        values_to_remove: list,
        logger: logging.Logger = None,
    ) -> pd.DataFrame:
        """
        Remove rows from a DataFrame that already exist in the database.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.
        column_to_check : str
            The name of the column to check for existing values.
        values_to_remove : list
            A list of values to be removed from the DataFrame.
        logger : logging.Logger, optional
            A logger for logging messages.

        Returns
        -------
        pd.DataFrame
            The DataFrame with existing records removed.
        """
        logger = logger_check(logger)
        column_variations = [
            column_to_check,
            column_to_check.upper(),
            column_to_check.lower(),
        ]

        for col in column_variations:
            if col in df.columns:
                logger.info(f"Column '{col}' found in DataFrame")
                remaining = df[~df[col].isin(values_to_remove)]
                logger.info(f"{len(remaining)} records remaining after filtering")
                logger.info(f"Removing {len(df) - len(remaining)} records")
                return remaining

        raise ValueError(f"Column '{column_to_check}' not found in DataFrame")

    def compare_sql_to_station(
        self,
        df: pd.DataFrame,
        station: str,
        field: str = "timestamp_end",
        loggertype: str = "eddy",
    ) -> pd.DataFrame:
        """
        Compare station data with records in the database and filter new entries.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the station data.
        station : str
            The identifier for the station.
        field : str, optional
            The field to use for comparison. Defaults to "timestamp_end".
        loggertype : str, optional
            The type of logger ('eddy' or 'met'). Defaults to 'eddy'.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing only the new records.
        """
        table = f"amflux{loggertype}"
        query = f"SELECT {field} FROM {table} WHERE stationid = '{station}';"

        exist = pd.read_sql(query, con=self.engine)
        existing = exist["timestamp_end"].values

        return self.remove_existing_records(df, field, existing, self.logger)

    def get_max_date(self, station: str, loggertype: str = "eddy") -> datetime.datetime:
        """
        Get the maximum timestamp from the station's data in the database.

        Parameters
        ----------
        station : str
            The identifier for the station.
        loggertype : str, optional
            The type of logger ('eddy' or 'met'). Defaults to 'eddy'.

        Returns
        -------
        datetime.datetime
            The latest timestamp found in the database for the station.
        """
        table = f"amflux{loggertype}"
        query = f"SELECT MAX(timestamp_end) AS max_value FROM {table} WHERE stationid = '{station}';"

        df = pd.read_sql(query, con=self.engine)
        return df["max_value"].iloc[0]

    def database_columns(self, dat: str) -> list:
        """
        Get the list of column names for a given database table.

        Parameters
        ----------
        dat : str
            The type of data ('eddy' or 'met'), which corresponds to
            the table name.

        Returns
        -------
        list
            A list of column names in the specified table.
        """
        table = f"amflux{dat}"
        query = f"SELECT * FROM {table} LIMIT 0;"
        df = pd.read_sql(query, con=self.engine)
        return df.columns.tolist()

    def process_station_data(
        self,
        site_folders: dict,
        config_path: str = "./data/reformatter_vars.yml",
        var_limits_csv: str = "./data/extreme_values.csv",
    ) -> None:
        """
        Process and upload data for all specified stations.

        This method iterates through a dictionary of site folders,
        fetches data for each station, processes it, and uploads it
        to the database.

        Parameters
        ----------
        site_folders : dict
            A dictionary mapping station IDs to folder names.
        config_path : str, optional
            The path to the reformatter configuration file.
        var_limits_csv : str, optional
            The path to the variable limits CSV file.
        """
        for stationid, name in site_folders.items():
            station = self.get_station_id(stationid)
            self.logger.info(f"Processing station: {stationid}")
            for dat in ["eddy", "met"]:
                if dat not in self.config[station]:
                    continue

                try:
                    stationtime, comptime = self.get_times(station, loggertype=dat)
                    am_df, pack_size = self.get_station_data(
                        station,
                        loggertype=dat,
                        config_path=config_path,
                        var_limits_csv=var_limits_csv,
                    )
                except Exception as e:
                    self.logger.error(f"Error fetching data for {stationid}: {e}")
                    continue

                if am_df is None:
                    self.logger.warning(f"No data for {stationid}")
                    continue

                am_cols = self.database_columns(dat)

                am_df_filt = self.compare_sql_to_station(am_df, station, loggertype=dat)
                self.logger.info(f"Filtered {len(am_df_filt)} records")
                stats = self._prepare_upload_stats(
                    am_df_filt,
                    stationid,
                    dat,
                    pack_size,
                    len(am_df),
                    len(am_df_filt),
                    stationtime,
                    comptime,
                )

                # Upload data
                am_df_filt = am_df_filt.rename(columns=str.lower)

                # Check for columns that are not in the database
                upload_cols = []

                for col in am_df_filt.columns:
                    if col in am_cols:
                        upload_cols.append(col)

                self._upload_to_database(am_df_filt[upload_cols], stats, dat)

                self._print_processing_summary(station, stats, self.logger)

    def _prepare_upload_stats(
        self,
        df: pd.DataFrame,
        stationid: str,
        tabletype: str,
        pack_size: float,
        raw_len: int,
        filtered_len: int,
        stationtime: str,
        comptime: str,
    ) -> dict:
        """
        Prepare a dictionary of statistics about the data upload.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame being uploaded.
        stationid : str
            The identifier for the station.
        tabletype : str
            The type of data table.
        pack_size : float
            The size of the data packet in MB.
        raw_len : int
            The number of rows in the raw data.
        filtered_len : int
            The number of rows after filtering.
        stationtime : str
            The timestamp from the station's logger.
        comptime : str
            The timestamp from the system running the script.

        Returns
        -------
        dict
            A dictionary of upload statistics.
        """
        return {
            "stationid": stationid,
            "talbetype": tabletype,
            "mindate": df["TIMESTAMP_START"].min(),
            "maxdate": df["TIMESTAMP_START"].max(),
            "datasize_mb": pack_size,
            "stationdf_len": raw_len,
            "uploaddf_len": filtered_len,
            "stationtime": stationtime,
            "comptime": comptime,
            "micromet_version": micromet_version,
        }

    def _upload_to_database(self, df: pd.DataFrame, stats: dict, dat: str) -> None:
        """
        Upload data and statistics to the database.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be uploaded.
        stats : dict
            A dictionary of statistics to be uploaded.
        dat : str
            The type of data ('eddy' or 'met'), used to determine the
            table name.
        """
        df.to_sql(f"amflux{dat}", con=self.engine, if_exists="append", index=False)
        pd.DataFrame([stats]).to_sql(
            "uploadstats", con=self.engine, if_exists="append", index=False
        )

    @staticmethod
    def _print_processing_summary(
        station: str, stats: dict, logger: logging.Logger = None
    ) -> None:
        """
        Print a summary of the data processing.

        Parameters
        ----------
        station : str
            The identifier for the station.
        stats : dict
            A dictionary of statistics from the processing.
        logger : logging.Logger, optional
            A logger for outputting the summary.
        """
        logger = logger_check(logger)
        logger.info(f"Station {station}")
        logger.info(f"Mindate {stats['mindate']}  Maxdate {stats['maxdate']}")
        logger.info(f"data size = {stats['datasize_mb']}")
        logger.info(f"{stats['uploaddf_len']} vs {stats['stationdf_len']} rows")
