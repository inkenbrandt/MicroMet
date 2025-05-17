import requests
import datetime
from typing import Union, Tuple, Optional

from requests.auth import HTTPBasicAuth
import pandas as pd
from io import BytesIO
import configparser
import sqlalchemy
from .converter import Reformatter

try:
    config = configparser.ConfigParser()
    config.read("../secrets/config.ini")
    passwrd = config["DEFAULT"]["pw"]
    ip = config["DEFAULT"]["ip"]
    login = config["DEFAULT"]["login"]
except KeyError:
    print("credentials needed")


class StationDataManager:
    """
    A class to manage station data operations including fetching, processing, and database interactions.
    """

    def __init__(
        self,
        config: Union[configparser.ConfigParser, dict],
        engine: sqlalchemy.engine.base.Engine,
    ):
        """
        Initialize the StationDataManager with configuration and database engine.

        Args:
            config: Configuration containing station details and credentials
            engine: SQLAlchemy database engine
        """
        self.config = config
        self.engine = engine
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

        Args:
            station: Station identifier
            loggertype: Logger type ('eddy' or 'met')

        Returns:
            Tuple of logger time and system time
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

    def get_station_data(
        self,
        station: str,
        reformat: bool = True,
        loggertype: str = "eddy",
        config_path: str = "data/reformatter_vars.yml",
        var_limits_csv: str = "data/extreme_values.csv",
        drop_soil: bool = False,
    ) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
        """
        Fetch and process station data.

        Args:
            station: Station identifier
            reformat: Whether to reformat the data
            loggertype: Logger type ('eddy' or 'met')

        Returns:
            Tuple of processed DataFrame and data packet size
        """
        ip = self.config[station]["ip"]
        port = self._get_port(station, loggertype)
        tabletype = (
            "Flux_AmeriFluxFormat" if loggertype == "eddy" else "Statistics_AmeriFlux"
        )

        url = f"http://{ip}:{port}/tables.html?"
        params = {
            "command": "DataQuery",
            "mode": "since-record",
            "format": "toA5",
            "uri": f"dl:{tabletype}",
            "p1": "0",
        }

        response = requests.get(url, params=params, auth=self.logger_credentials)

        if response.status_code == 200:
            raw_data = pd.read_csv(BytesIO(response.content), skiprows=[0, 2, 3])
            pack_size = len(response.content) * 1e-6

            if raw_data is not None and reformat:
                am_data = Reformatter(
                    config_path=config_path,
                    var_limits_csv=var_limits_csv,
                    drop_soil=drop_soil,
                )
                am_df = am_data.prepare(raw_data)
                am_data = Reformatter(raw_data)
                am_df = am_data.et_data
            else:
                am_df = raw_data

            return am_df, pack_size

        print(f"Error: {response.status_code}")
        return None, None

    @staticmethod
    def remove_existing_records(
        df: pd.DataFrame, column_to_check: str, values_to_remove: list
    ) -> pd.DataFrame:
        """
        Remove existing records from DataFrame.

        Args:
            df: Input DataFrame
            column_to_check: Column name to check
            values_to_remove: Values to remove

        Returns:
            Filtered DataFrame
        """
        column_variations = [
            column_to_check,
            column_to_check.upper(),
            column_to_check.lower(),
        ]

        for col in column_variations:
            if col in df.columns:
                print(f"Column '{col}' found in DataFrame")
                return df[~df[col].isin(values_to_remove)]

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

        return self.remove_existing_records(df, field, existing)

    def get_max_date(self, station: str, loggertype: str = "eddy") -> datetime.datetime:
        """
        Get maximum timestamp from station database.

        Args:
            station: Station identifier
            loggertype: Logger type

        Returns:
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

    def process_station_data(self, site_folders: dict) -> None:
        """
        Process data for all stations.

        Args:
            site_folders: Dictionary mapping station IDs to names
        """
        for stationid, name in site_folders.items():
            station = self.get_station_id(stationid)
            print(stationid)
            for dat in ["eddy", "met"]:
                if dat not in self.config[station]:
                    continue

                try:
                    stationtime, comptime = self.get_times(station, loggertype=dat)
                    am_df, pack_size = self.get_station_data(station, loggertype=dat)
                except Exception as e:
                    print(f"Error fetching data for {stationid}: {e}")
                    continue

                if am_df is None:
                    print(f"No data for {stationid}")
                    continue

                am_cols = self.database_columns(dat)

                am_df_filt = self.compare_sql_to_station(am_df, station, loggertype=dat)
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

                self._print_processing_summary(station, stats)

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
    def _print_processing_summary(station: str, stats: dict) -> None:
        """Print processing summary."""
        print(f"Station {station}")
        print(f"Mindate {stats['mindate']}  Maxdate {stats['maxdate']}")
        print(f"data size = {stats['datasize_mb']}")
        print(f"{stats['uploaddf_len']} vs {stats['stationdf_len']} rows")
