import requests
import datetime
import logging
from typing import Union, Tuple, Optional

from requests.auth import HTTPBasicAuth
import pandas as pd
from io import BytesIO
import configparser
import sqlalchemy
from .converter import Reformatter


def logger_check(logger: logging.Logger | None) -> logging.Logger:
    """
    Check if a logger is provided, and if not, create one.

    Args:
        logger: Logger to check

    Returns:
        Logger to use
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.WARNING)

        # Create console handler and set level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        Initialize the StationDataManager with configuration and database engine.

        Args:
            config: Configuration containing station details and credentials
            logger: Logger to use for logging messages
        """
        self.config = config
        self.logger = logger_check(logger)

        self.logger_credentials = HTTPBasicAuth(
            config["LOGGER"]["login"], config["LOGGER"]["pw"]
        )

    def _get_port(self, station: str, loggertype: str = "eddy") -> int:
        """
        Get the port number for a given station and logger type.

        Args:
            station: Station identifier
            loggertype: Type of logger ('eddy' or 'met')

        Returns:
            Port number
        """
        port_key = f"{loggertype}_port"
        return int(self.config[station].get(port_key, 80))

    def get_times(
        self, station: str, loggertype: str = "eddy"
    ) -> Tuple[Optional[str], str]:
        """
        Retrieve current logger time and system time.

        Parameters
        ----------
        station : str
            Station identifier
        loggertype : str, optional
            Type of logger ('eddy' or 'met'), by default 'eddy'

        Returns
        -------
        Tuple[Optional[str], str]
            Tuple containing current logger time and system time

        Notes
        -----
        See https://help.campbellsci.com/crbasic/cr6/Content/Info/webserverapicommands1.htm
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
        """Extract station ID from full station identifier."""
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
        Download data from a station.

        Parameters
        ----------
        station : str
            Station identifier
        loggertype: str
            Type of logger ('eddy' or 'met'); default is 'eddy'
        mode: str
            Timeframe of the data to be returned ('since-time', 'most-recent', 'since-record', 'date-range', 'Backfill'); default is 'since-time'
        p1: str
            String datetime (YYYY-MM-DD T:HH:MM:SS.MS or YYYY-MM-DD) if since-time, or daterange; otherwise # of starting record or backfill interval; default is 0
        p2: str
            String datetime (YYYY-MM-DD T:HH:MM:SS.MS or YYYY-MM-DD) if daterange; default is 0

        Returns
        -------
        Tuple[Optional[pd.DataFrame], Optional[float]]
            Tuple containing downloaded data and data packet size

        Notes
        -----
        See https://help.campbellsci.com/crbasic/cr6/Content/Info/webserverapicommands1.htm
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
        Fetch and process station data.

        Args:
            station: Station identifier
            reformat: Whether to reformat the data
            loggertype: Logger type ('eddy' or 'met')
            config_path: Path to reformatter configuration
            var_limits_csv: Path to extreme values CSV
            drop_soil: Whether to drop soil data

        Returns:
            Tuple of processed DataFrame and data packet size
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
        df: pd.DataFrame, column_to_check: str, values_to_remove: list, logger: logging.Logger = None
    ) -> pd.DataFrame:
        """
        Remove existing records from DataFrame.

        Args:
            df: Input DataFrame
            column_to_check: Column name to check
            values_to_remove: Values to remove
            logger: Logger to use for logging messages

        Returns:
            Filtered DataFrame
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
        Compare station data with SQL records and filter new entries.

        Args:
            df: Station data DataFrame
            station: Station identifier
            field: Field to compare
            loggertype: Logger type

        Returns:
            Filtered DataFrame
        """
        table = f"amflux{loggertype}"
        query = f"SELECT {field} FROM {table} WHERE stationid = '{station}';"

        exist = pd.read_sql(query, con=self.engine)
        existing = exist["timestamp_end"].values

        return self.remove_existing_records(df, field, existing, self.logger)

    def get_max_date(self, station: str, loggertype: str = "eddy") -> datetime.datetime:
        """
        Get maximum timestamp from station database.

        Parameters
        ----------
        station : str
            Station identifier
        loggertype : str, optional
            Type of logger ('eddy' or 'met'), by default 'eddy'

        Returns
        -------
        int
            Latest timestamp
        """
        table = f"amflux{loggertype}"
        query = f"SELECT MAX(timestamp_end) AS max_value FROM {table} WHERE stationid = '{station}';"

        df = pd.read_sql(query, con=self.engine)
        return df["max_value"].iloc[0]

    def database_columns(self, dat: str) -> list:
        """
        Get the columns of the database table.

        Args:
            dat: Type of data ('eddy' or 'met')

        Returns:
            List of column names
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
        Process data for all stations.

        Args:
            site_folders: Dictionary mapping station IDs to names
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
        """Prepare statistics for upload."""
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
        }

    def _upload_to_database(self, df: pd.DataFrame, stats: dict, dat: str) -> None:
        """Upload data and stats to database."""
        df.to_sql(f"amflux{dat}", con=self.engine, if_exists="append", index=False)
        pd.DataFrame([stats]).to_sql(
            "uploadstats", con=self.engine, if_exists="append", index=False
        )

    @staticmethod
    def _print_processing_summary(station: str, stats: dict, logger: logging.Logger = None) -> None:
        """Print processing summary."""
        logger = logger_check(logger)
        logger.info(f"Station {station}")
        logger.info(f"Mindate {stats['mindate']}  Maxdate {stats['maxdate']}")
        logger.info(f"data size = {stats['datasize_mb']}")
        logger.info(f"{stats['uploaddf_len']} vs {stats['stationdf_len']} rows")
