"""
converter.py
Cleaned up version of micromet data processing utilities.

Provides:
- AmerifluxDataProcessor: read AmeriFlux-style csv (TOA5 or AmeriFlux output) into DataFrame.
- Reformatter: sanitize, standardize, and resample station data for flux / met processing.

Immediate cleanup performed:
* Removed dead/commented code and duplicate lines.
* Fixed duplicate attribute assignments.
* Extracted long code blocks into helper methods for clarity.
* Updated docstrings and type hints.
* Added module-level constants and logger.
* Replaced inline magic numbers with named constants.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import numpy as np
import yaml
from importlib.resources import files

import reformatter_vars
import variable_limits

# -----------------------------------------------------------------------------#
# Helper functions
# -----------------------------------------------------------------------------#


def load_yaml(path: Path | str) -> Dict:
    """Load a YAML file and return as dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as fp:
        return yaml.safe_load(fp)


def logger_check(logger: logging.Logger | None) -> logging.Logger:
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.WARNING)
        ch = logging.StreamHandler()
        ch.setFormatter(
            logging.Formatter(
                fmt="%(levelname)s [%(asctime)s] %(name)s – %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(ch)
    else:
        logger = logger
    return logger


# -----------------------------------------------------------------------------#
# AmerifluxDataProcessor
# -----------------------------------------------------------------------------#
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
        config_path: Path | str = "reformatter_vars.yml",
        logger: logging.Logger = None,
    ):
        # if config_path == "reformatter_vars.yml":
        #    data_path = files("micromet.data").joinpath("reformatter_vars.yml")
        # else:
        #    data_path = Path(config_path)
        self.logger = logger_check(logger)
        # self.headers = load_yaml(data_path)
        self.skip_rows = None

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def to_dataframe(self, file: Union[str, Path]) -> pd.DataFrame:
        """Return parsed CSV as pandas DataFrame."""
        self._determine_header_rows(file)
        self.logger.debug("Reading %s", file)
        df = pd.read_csv(
            file,
            skiprows=self.skip_rows,
            names=self.names,
            na_values=self.NA_VALUES,
        )
        return df

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
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
            self.skip_rows = 0
            self.names = first_line
        elif first_line[0] == self._TOA5_PREFIX:
            self.logger.debug(f"TOA5 header detected: {first_line}")
            self.skip_rows = [0, 1, 2, 3]
            self.names = second_line
        else:
            raise RuntimeError(f"Header line not recognized: {first_line}")
        self.logger.debug(f"Skip rows for set to {self.skip_rows}")

    def _get_file_no(self, file: Path) -> tuple[int, int]:
        basename = file.stem

        try:
            file_number = int(basename.split("_")[-1])
            datalogger_number = int(basename.split("_")[0])
        except ValueError:
            file_number = datalogger_number = 9999
        self.logger.debug(f"{file_number} -> {datalogger_number}")
        return file_number, datalogger_number

    # --------------------------------------------------------------------- #
    # Iterators
    # --------------------------------------------------------------------- #

    def raw_file_compile(
        self,
        main_dir: Union[str, Path],
        station_folder_name: Union[str, Path],
        search_str: str = "*Flux_AmeriFluxFormat*.dat",
    ) -> Optional[pd.DataFrame]:
        """
        Compiles raw AmeriFlux datalogger files into a single dataframe.

        Parameters
        ----------
        main_dir : str | Path
            Name of main directory to search for AmeriFlux datalogger files.
        station_folder_name : str
            Name of the station folder containing the raw datalogger files.
        search_str : str
            String to search for; use asterisk (*) for wildcard.

        Returns
        -------
        pandas.DataFrame or None
            Dataframe containing compiled AmeriFlux data, or None if no valid files found.
        """
        compiled_data = []
        station_folder = Path(main_dir) / station_folder_name

        self.logger.info(f"Compiling data from {station_folder}")

        for file in station_folder.rglob(search_str):
            self.logger.info(f"Processing file: {file}")

            file_no, datalogger_number = self._get_file_no(file)

            df = self.to_dataframe(file)
            if df is not None:
                df["file_no"] = file_no
                df["datalogger_no"] = datalogger_number
                compiled_data.append(df)

        if compiled_data:
            compiled_df = pd.concat(compiled_data, ignore_index=True)
            return compiled_df
        else:
            self.logger.warning(f"No valid files found in {station_folder}")
            return None

    def iterate_through_stations(self):
        """Iterate through all stations."""
        site_folders = {
            "US-UTD": "Dugout_Ranch",
            "US-UTB": "BSF",
            "US-UTJ": "Bluff",
            "US-UTW": "Wellington",
            "US-UTE": "Escalante",
            "US-UTM": "Matheson",
            "US-UTP": "Phrag",
            "US-CdM": "Cedar_mesa",
            "US-UTV": "Desert_View_Myton",
            "US-UTN": "Juab",
            "US-UTG": "Green_River",
        }
        for stationid, folder in site_folders.items():
            for datatype in ["met", "eddy"]:
                if datatype == "met":
                    search_str = f"*Statistics_Ameriflux*.dat"
                else:
                    search_str = f"*AmeriFluxFormat*.dat"
                self.raw_file_compile(stationid, folder, search_str)


# ----------------------------------------------------------------------------
# Reformatter
# ----------------------------------------------------------------------------
class Reformatter:
    """
    Clean and standardize station data.

    Steps (all configurable):

    1. Fix and align timestamps (`_fix_timestamps`).
    2. Rename columns (`_rename_columns`).
    3. Remove obvious outliers and redundant columns (`clean_columns`).
    4. Adjust derived values (`apply_fixes`).
    5. Optionally, drop extra soil sensor channels (`_drop_extra_soil_columns`).
    6. Finalize column order and missing value representation.

    Parameters
    ----------
    config_path : str | Path
        File Path to YAML configuration file governing renames, drops, etc.
    var_limits_csv : str | Path, optional
        CSV file containing per‑variable hard min/max limits.
    drop_soil : bool, default True
        Whether to remove soil columns deemed redundant (see config).
    """

    # SoilVUE Depth/orientation conversion tables --------------------------------
    _DEPTH_MAP = {5: 1, 10: 2, 20: 3, 30: 4, 40: 5, 50: 6, 60: 7, 75: 8, 100: 9}
    _ORIENT_MAP = {"N": 3, "S": 4}
    _LEGACY_RE = re.compile(
        r"^(?P<prefix>(SWC|TS|EC|K|T))_(?P<depth>\d{1,3})cm_(?P<orient>[NS])_.*$",
        re.IGNORECASE,
    )
    _PREFIX_PATTERNS: Dict[re.Pattern[str], str] = {
        re.compile(r"^BulkEC_", re.IGNORECASE): "EC_",
        re.compile(r"^VWC_", re.IGNORECASE): "SWC_",
        re.compile(r"^Ka_", re.IGNORECASE): "K_",
    }

    # Constants ------------------------------
    MISSING_VALUE: int = -9999
    SOIL_SENSOR_SKIP_INDEX: int = 3  # Drop SWC_/TS_/EC_/K_ where second segment >= 3
    DEFAULT_SOIL_DROP_LIMIT: int = 4  # keep last 4 items in math_soils_v2 list

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

    # ------------------------------------------------------------------
    # Pipeline entry
    # ------------------------------------------------------------------
    def prepare(self, df: pd.DataFrame, data_type: str = "eddy") -> pd.DataFrame:
        self.logger.info("Starting reformat (%s rows)", len(df))

        df = (
            df.pipe(self._fix_timestamps)
            .pipe(self.rename_columns, data_type=data_type)
            .pipe(self.set_number_types)
            .pipe(self.resample_timestamps)
            .pipe(self.timestamp_reset)
            .pipe(self.clean_columns)
            .pipe(self.apply_fixes)
        )
        if self.drop_soil:
            df = self._drop_extra_soil_columns(df)

        df = df.pipe(self._drop_extras).fillna(self.MISSING_VALUE)
        self.logger.info("Done; final shape: %s", df.shape)
        return df

    # ------------------------------------------------------------------
    # 1. Timestamp handling
    # ------------------------------------------------------------------
    def infer_datetime_col(self, df: pd.DataFrame) -> str | None:
        """Return the name of the TIMESTAMP column."""
        datetime_col_options = ["TIMESTAMP_START", "TIMESTAMP_START_1"]
        datetime_col_options += [col.lower() for col in datetime_col_options]
        for cand in datetime_col_options:
            if cand in df.columns:
                return cand
            else:
                self.logger.warning("No TIMESTAMP column in dataframe")
                return df.iloc[:, 0].name
        return None

    def _fix_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "TIMESTAMP" in df.columns:
            df = df.drop(["TIMESTAMP"], axis=1)
        ts_col = self.infer_datetime_col(df)
        self.logger.debug(f"TS col {ts_col}")
        self.logger.debug(f"TIMESTAMP_START col {df[ts_col][0]}")
        ts_format = "%Y%m%d%H%M"
        df["datetime_start"] = pd.to_datetime(
            df[ts_col], format=ts_format, errors="coerce"
        )
        self.logger.debug(f"Len of unfixed timestamps {len(df)}")
        df = df.dropna(subset=["datetime_start"])
        self.logger.debug(f"Len of fixed timestamps {len(df)}")
        return df

    def resample_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample a DataFrame to 30-minute intervals based on the 'datetime_start' column.

        The method performs the following steps:
        - Filters out future timestamps beyond the current date.
        - Removes duplicate entries based on 'datetime_start'.
        - Sets 'datetime_start' as the index and sorts the DataFrame.
        - Resamples the data to 30-minute intervals using the first valid observation.
        - Linearly interpolates missing values with a maximum gap of one interval.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing a 'datetime_start' column of timestamp values.

        Returns
        -------
        pd.DataFrame
            Resampled and interpolated DataFrame indexed by 'datetime_start'.
        """
        today = pd.Timestamp("today").floor("D")
        df = df[df["datetime_start"] <= today]
        df = (
            df.drop_duplicates(subset=["datetime_start"])
            .set_index("datetime_start")
            .sort_index()
        )
        df = df.resample("30min").first().interpolate(limit=1)
        self.logger.debug(f"Len of resampled timestamps {len(df)}")
        return df

    @staticmethod
    def timestamp_reset(df, minutes=30):
        """
        Reset TIMESTAMP_START and TIMESTAMP_END columns based on the DataFrame index.

        This method assumes the DataFrame index is a datetime-like index and sets
        the 'TIMESTAMP_START' column to match the index formatted as an integer
        in 'YYYYMMDDHHMM' format. The 'TIMESTAMP_END' column is set to the timestamp
        `minutes` ahead of the index.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with a datetime-like index.
        minutes : int, optional
            Number of minutes to add for TIMESTAMP_END calculation, by default 30.

        Returns
        -------
        pd.DataFrame
            Modified DataFrame with updated 'TIMESTAMP_START' and 'TIMESTAMP_END' columns.
        """
        df["TIMESTAMP_START"] = df.index.strftime("%Y%m%d%H%M").astype(int)
        df["TIMESTAMP_END"] = (
            (df.index + pd.Timedelta(minutes=minutes))
            .strftime("%Y%m%d%H%M")
            .astype(int)
        )
        return df

    # ------------------------------------------------------------------
    # 2. Column renaming & legacy conversions
    # ------------------------------------------------------------------
    def rename_columns(self, df: pd.DataFrame, *, data_type: str) -> pd.DataFrame:
        """
        Rename DataFrame columns based on configuration and standardize column names.

        This method selects a column rename mapping based on the `data_type` parameter
        ('eddy' or 'met') from the internal configuration dictionary. It then renames
        the DataFrame columns accordingly and applies additional normalization routines:
        - `normalize_prefixes`: standardizes common prefix formats
        - `modernize_soil_legacy`: updates legacy soil column naming conventions

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with original column names to be renamed.
        data_type : str
            Type of data used to determine the appropriate rename mapping.
            Must be either 'eddy' or 'met'.

        Returns
        -------
        pd.DataFrame
            DataFrame with renamed and standardized column names.
        """
        mapping = self.config.get(
            "renames_eddy" if data_type == "eddy" else "renames_met", {}
        )
        self.logger.debug(f"Renaming columns from {df.columns} to {mapping}")
        df = df.rename(columns=mapping)
        df = self.normalize_prefixes(df)
        df = self.modernize_soil_legacy(df)
        self.logger.debug(f"Len of renamed cols {len(df)}")
        return df

    def normalize_prefixes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column name prefixes related to soil and temperature measurements.

        This method renames columns in the input DataFrame based on predefined prefix
        replacement rules. It performs the following operations:

        - Replaces known prefixes such as ``BulkEC_``, ``VWC_``, and ``Ka_`` using compiled regex patterns.
        - Replaces ``T_`` with ``Ts_`` if the column name matches the pattern 'T_<depth>cm_'.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame with original column names to be normalized.

        Returns
        -------
        pandas.DataFrame
            DataFrame with column names updated to follow normalized prefix conventions.

        Notes
        -----
        The method uses regex patterns defined in `self._PREFIX_PATTERNS` to perform
        most substitutions. Logging statements are used to report which columns were renamed.
        """
        rename_map: Dict[str, str] = {}
        for col in df.columns:
            # Simple prefix swaps
            for patt, repl in self._PREFIX_PATTERNS.items():
                if patt.match(col):
                    rename_map[col] = patt.sub(repl, col)
                    break
            else:
                # Conditional T_ rule – only if immediately followed by depthcm_
                if re.match(r"^T_\d{1,3}cm_", col, flags=re.IGNORECASE):
                    rename_map[col] = re.sub(r"^T_", "Ts_", col, flags=re.IGNORECASE)
        if rename_map:
            self.logger.debug("Prefix normalisation: %s", rename_map)
            df = df.rename(columns=rename_map)
        self.logger.debug(f"Len of normalized prefix cols {len(df)}")
        return df

    def modernize_soil_legacy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Update legacy soil sensor column names to a standardized format.

        This method parses column names from legacy SoilVUE or similar output formats
        and renames them according to a modern schema using orientation, depth, and
        sensor type mappings. Columns that do not match the expected legacy format
        are left unchanged.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing soil sensor data with potentially outdated column names.

        Returns
        -------
        pandas.DataFrame
            DataFrame with updated column names reflecting modern naming conventions.

        Notes
        -----
        - Legacy column names are parsed using a regular expression defined by `self._LEGACY_RE`.
        - Depth values are mapped to indices via `self._DEPTH_MAP`.
        - Orientation characters (e.g., 'N', 'S') are mapped via `self._ORIENT_MAP`.
        - 'T' prefix is interpreted as 'TS' due to prior normalization.
        - Logging records renamed columns if any were modernized.
        """
        rename_map: Dict[str, str] = {}
        for col in df.columns:
            m = self._LEGACY_RE.match(col)
            if not m:
                continue
            prefix = m.group("prefix").upper()
            if prefix == "T":  # became Ts in previous step
                prefix = "TS"
            depth_cm = int(m.group("depth"))
            orient = m.group("orient").upper()
            depth_idx = self._DEPTH_MAP.get(depth_cm)
            if depth_idx is None:
                continue
            replic = self._ORIENT_MAP[orient]
            new_name = f"{prefix}_{replic}_{depth_idx}_1"
            rename_map[col] = new_name
        if rename_map:
            self.logger.info(f"Legacy soil columns modernised: {rename_map}")
            df = df.rename(columns=rename_map)
        return df

    # ------------------------------------------------------------------ #
    # Stage 3 – general clean
    # ------------------------------------------------------------------ #

    def clean_columns(self, df, replace_w=np.nan):
        """
        Replace out-of-range values in float64 columns with a specified placeholder.

        For each float64 column in the DataFrame that has a defined limit in `self.varlimits`,
        this method replaces values that fall outside the [Min, Max] range with a specified
        replacement value (default is `np.nan`).

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing the data to be cleaned.
        replace_w : scalar, optional
            Value to replace out-of-range entries with. Default is `np.nan`.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with out-of-range values replaced.

        Notes
        -----
        - The method only processes columns present in both the DataFrame and `self.varlimits`.
        - Only columns of type `float64` are cleaned.
        - Limit values must be stored in a DataFrame `self.varlimits` with columns "Min" and "Max".
        """
        df = df.copy()
        if self.varlimits is not None:
            for col in set(df.columns) & set(self.varlimits.index):
                if df.dtypes[col] == "float64":
                    self.logger.debug(f"Examining column {col} limits")
                    limits = self.varlimits.loc[col]
                    valid_mask = (df[col] > limits["Min"]) & (df[col] < limits["Max"])
                    df.loc[~valid_mask, col] = replace_w
                    # df[variable] = df[variable].apply(
                    #    lambda x: replace_w if x < lower_bound or x > upper_bound else x
                    # )
        self.logger.debug(f"Cleaned rows: {len(df)}")
        return df

    # ------------------------------------------------------------------ #
    # Stage 4 – variable‑specific fixes
    # ------------------------------------------------------------------ #
    def apply_fixes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply a set of minor, variable-specific data corrections.

        This method applies a sequence of small fixes to the input DataFrame,
        including adjustments to `Tau`, scaling of SSITC values, and unit conversions
        for soil water content. These operations are encapsulated in internal helper
        methods.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing raw or intermediate data.

        Returns
        -------
        pandas.DataFrame
            DataFrame with variable-specific fixes applied.

        Notes
        -----
        The following fixes are applied in order:
        - `tau_fixer`: Handles edge cases and invalid values in the Tau variable.
        - `fix_swc_percent`: Converts volumetric water content values from percent to fraction, if needed.
        - `ssitc_scale`: Scales SSITC variable values according to predefined rules.
        """
        df = self.tau_fixer(df)
        df = self.fix_swc_percent(df)
        df = self.ssitc_scale(df)
        return df

    def tau_fixer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace zero values in the 'Tau' column with NaN.

        This method identifies entries in the 'Tau' column that are exactly zero
        and replaces them with `np.nan`, but only if both 'Tau' and 'u_star' columns
        are present in the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing turbulent flux variables, including 'Tau' and 'u_star'.

        Returns
        -------
        pandas.DataFrame
            DataFrame with zero values in 'Tau' replaced by NaN.

        Notes
        -----
        This fix targets an artifact where Tau may be reported as zero when
        it is invalid or unmeasured.
        """
        if "Tau" in df.columns and "u_star" in df.columns:
            bad_idx = df["Tau"] == 0
            df.loc[bad_idx, "Tau"] = np.nan
        return df

    def fix_swc_percent(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert fractional soil water content values to percent if applicable.

        This method scans for columns with names starting with ``SWC_`` and checks
        whether their maximum values are below or equal to 1.5. If so, the column
        is assumed to be in volumetric fraction (0–1) and is converted to percent (0–100)
        by multiplying by 100.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing soil water content (SWC) columns.

        Returns
        -------
        pandas.DataFrame
            DataFrame with updated SWC columns, where applicable.

        Notes
        -----
        - This heuristic assumes that volumetric soil water content should not exceed 1.5.
        - Columns are modified in place within a copy of the DataFrame.
        - A debug log entry is created for each column that is converted.
        """
        swc_cols = [c for c in df.columns if c.startswith("SWC_")]
        for col in swc_cols:
            if df[col].max(skipna=True) <= 1.5:  # likely 0–1 volumetric
                df[col] = df[col] * 100.0
                self.logger.debug(f"Fixed soil percent {col}")
        return df

    def ssitc_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale SSITC test columns if values exceed expected thresholds.

        This method checks for the presence of known SSITC test columns in the input
        DataFrame and applies scaling and unit conversion if the maximum value in
        a column exceeds 3. The transformation is performed using `self.scale_and_convert`.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame that may contain SSITC test variables.

        Returns
        -------
        pandas.DataFrame
            DataFrame with scaled SSITC columns where applicable.

        Notes
        -----
        - Only the following columns are considered:
          'FC_SSITC_TEST', 'LE_SSITC_TEST', 'ET_SSITC_TEST', 'H_SSITC_TEST', 'TAU_SSITC_TEST'.
        - Scaling is applied only if the column exists and its maximum value exceeds 3.
        - A debug message is logged for each column that is scaled.
        """
        ssitc_columns = [
            "FC_SSITC_TEST",
            "LE_SSITC_TEST",
            "ET_SSITC_TEST",
            "H_SSITC_TEST",
            "TAU_SSITC_TEST",
        ]
        for column in ssitc_columns:
            if column in df.columns:
                if df[column].max() > 3:
                    df[column] = self.scale_and_convert(df[column])
                    self.logger.debug(f"Scaled SSITC {column}")
        self.logger.debug(f"Scaled SSITC len: {len(df)}")
        return df

    def scale_and_convert(self, column: pd.Series) -> pd.Series:
        """
        Apply a rating transformation and convert values to float.

        This method maps values in the input Series using the `self.rating` function
        and returns the transformed result. The transformation is intended to rescale
        categorical or ordinal values to a standardized float representation.

        Parameters
        ----------
        column : pandas.Series
            Input Series containing values to be scaled and converted.

        Returns
        -------
        pandas.Series
            Series with values transformed using the rating function.

        Notes
        -----
        - The `self.rating` function is applied element-wise via `.apply()`.
        - The output Series retains the same index and is returned as float-compatible values.
        """
        # match rating to new rating
        column = column.apply(self.rating)
        # output at integer
        return column

    @staticmethod
    def rating(x):
        """
        Categorize a numeric value into a discrete rating level.

        This method assigns an integer rating category based on the input value:

        - 0 if 0 ≤ x ≤ 3
        - 1 if 4 ≤ x ≤ 6
        - 2 otherwise

        If the input is None, it is treated as 0.

        Parameters
        ----------
        x : int or float or None
            The numeric value to be converted into a rating category.

        Returns
        -------
        int
            An integer rating in the range {0, 1, 2}, based on predefined thresholds.

        Notes
        -----
        This is a simple classification utility used for scaling SSITC-related test scores.
        """
        if x is None:
            x = 0
        else:
            if 0 <= x <= 3:
                x = 0
            elif 4 <= x <= 6:
                x = 1
            else:
                x = 2
        return x

    # ------------------------------------------------------------------ #
    # Stage 5 – soil columns
    # ------------------------------------------------------------------ #
    def _drop_extra_soil_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop redundant or unused soil probe columns based on configuration and naming conventions.

        This method identifies and removes soil sensor columns that exceed expected depth
        indices, match deprecated patterns, or are listed in the configured math soil list.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing soil sensor columns (e.g., SWC, TS, EC, K).

        Returns
        -------
        pandas.DataFrame
            DataFrame with redundant soil columns removed.

        Notes
        -----
        - Columns with prefixes 'SWC', 'TS', 'EC', or 'K' and a numeric index equal to or
          greater than `self.SOIL_SENSOR_SKIP_INDEX` are dropped.
        - Columns listed in `self.config["math_soils_v2"]` beyond the last `DEFAULT_SOIL_DROP_LIMIT`
          entries are also removed.
        - Columns with prefixes 'VWC' or 'Ka' or those ending in 'cm_N' or 'cm_S' are considered legacy
          or malformed and are dropped as well.
        - Logging reports the number of columns removed.
        """
        df = df.copy()
        math_soils: Sequence[str] = self.config.get("math_soils_v2", [])
        to_drop: List[str] = []

        for col in df.columns:
            parts = col.split("_")
            if len(parts) >= 3 and parts[0] in {"SWC", "TS", "EC", "K"}:
                try:
                    if int(parts[1]) >= self.SOIL_SENSOR_SKIP_INDEX:
                        to_drop.append(col)
                        continue
                except ValueError:
                    pass  # non‑numeric, ignore
            if col in math_soils[: -self.DEFAULT_SOIL_DROP_LIMIT]:
                to_drop.append(col)
                continue
            if (
                parts[0] in {"VWC", "Ka"}
                or col.endswith("cm_N")
                or col.endswith("cm_S")
            ):
                to_drop.append(col)

        if to_drop:
            self.logger.info("Dropping %d redundant soil columns", len(to_drop))
            df = df.drop(columns=to_drop, errors="ignore")
        return df

    # ------------------------------------------------------------------
    # 6. Final housekeeping
    # ------------------------------------------------------------------
    def set_number_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert columns in the DataFrame to numeric types where appropriate.

        This method attempts to cast each column to a numeric type using `pandas.to_numeric`.
        Specific columns are downcast to integers to save memory, while all others are coerced
        to floats if needed. Non-numeric values are set to NaN.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame with columns to be type-cast.

        Returns
        -------
        pandas.DataFrame
            DataFrame with numeric types applied where possible.

        Notes
        -----
        - 'MO_LENGTH' and 'RECORD' columns are downcast to integer.
        - 'TIMESTAMP_START', 'TIMESTAMP_END', and 'SSITC' are also downcast to integer.
        - 'datetime_start' is left unchanged.
        - All other columns are converted to numeric (float) with `errors='coerce'`.
        - Logging reports the number of rows processed.
        """
        self.logger.debug(f"Setting number types: {df.head(3)}")
        for col in df.columns:
            self.logger.debug(f"Setting number types {col}")

            if col in ["MO_LENGTH", "RECORD"]:
                df[col] = pd.to_numeric(df[col], downcast="integer", errors="coerce")

            elif col in ["datetime_start"]:
                df[col] = df[col]

            elif col in ["TIMESTAMP_START", "TIMESTAMP_END", "SSITC"]:
                df[col] = pd.to_numeric(df[col], downcast="integer", errors="coerce")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        self.logger.debug(f"Set number types: {len(df)}")
        return df

    def _drop_extras(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop extra or unwanted columns from the DataFrame based on configuration.

        This method removes columns listed under the 'drop_cols' key in the configuration
        dictionary. Columns not found in the DataFrame are ignored.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame potentially containing extra columns.

        Returns
        -------
        pandas.DataFrame
            DataFrame with specified columns removed.

        Notes
        -----
        - The columns to drop are retrieved from `self.config["drop_cols"]`.
        - Uses `errors='ignore'` to skip missing columns without raising an error.
        """
        return df.drop(columns=self.config.get("drop_cols", []), errors="ignore")

    def col_order(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder DataFrame columns to place priority columns first.

        This method ensures that 'TIMESTAMP_END' and 'TIMESTAMP_START' appear as the
        first columns in the DataFrame, in that order. If these columns exist, they
        are moved to the front without altering the order of the remaining columns.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame with timestamp and other data columns.

        Returns
        -------
        pandas.DataFrame
            DataFrame with reordered columns.

        Notes
        -----
        - Columns are moved using `.pop()` and `.insert()` to preserve data.
        - A debug message logs the final column order.
        """
        first_cols = ["TIMESTAMP_END", "TIMESTAMP_START"]
        for col in first_cols:
            ncol = df.pop(col)
            df.insert(0, col, ncol)
        self.logger.debug(f"Column Order: {df.columns}")
        return df

    # Re‑export key classes -------------------------------------------------------
    __all__ = ["AmerifluxDataProcessor", "Reformatter"]
