"""
This module contains data transformation functions that are used in the
reformatting pipeline.
"""
import pandas as pd
import numpy as np
import re
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import logging

from micromet.utils import logger_check
import micromet.qaqc.variable_limits as variable_limits
import micromet.format.reformatter_vars as reformatter_vars

# Constants
MISSING_VALUE: int = -9999
SOIL_SENSOR_SKIP_INDEX: int = 3
DEFAULT_SOIL_DROP_LIMIT: int = 4

# SoilVUE Depth/orientation conversion tables
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


def infer_datetime_col(df: pd.DataFrame, logger: logging.Logger) -> str | None:
    """Return the name of the TIMESTAMP column."""
    datetime_col_options = ["TIMESTAMP_START", "TIMESTAMP_START_1"]
    datetime_col_options += [col.lower() for col in datetime_col_options]
    for cand in datetime_col_options:
        if cand in df.columns:
            return cand
    logger.warning("No TIMESTAMP column in dataframe")
    return df.iloc[:, 0].name  # type: ignore


def fix_timestamps(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    df = df.copy()
    if "TIMESTAMP" in df.columns:
        df = df.drop(["TIMESTAMP"], axis=1)
    ts_col = infer_datetime_col(df, logger)
    if ts_col is None:
        return df

    logger.debug(f"TS col {ts_col}")
    logger.debug(f"TIMESTAMP_START col {df[ts_col][0]}")
    ts_format = "%Y%m%d%H%M"
    df["DATETIME_START"] = pd.to_datetime(
        df[ts_col], format=ts_format, errors="coerce"
    )
    logger.debug(f"Len of unfixed timestamps {len(df)}")
    df = df.dropna(subset=["DATETIME_START"])
    logger.debug(f"Len of fixed timestamps {len(df)}")
    return df


def resample_timestamps(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Resample a DataFrame to 30-minute intervals based on the 'DATETIME_START' column.
    """
    today = pd.Timestamp("today").floor("D")
    df = df[df["DATETIME_START"] <= today]
    df = (
        df.drop_duplicates(subset=["DATETIME_START"])
        .set_index("DATETIME_START")
        .sort_index()
    )
    df = df.resample("30min").first().interpolate(limit=1)
    logger.debug(f"Len of resampled timestamps {len(df)}")
    return df


def timestamp_reset(df, minutes=30):
    """
    Reset TIMESTAMP_START and TIMESTAMP_END columns based on the DataFrame index.
    """
    df["TIMESTAMP_START"] = df.index.strftime("%Y%m%d%H%M").astype(int)
    df["TIMESTAMP_END"] = (
        (df.index + pd.Timedelta(minutes=minutes))
        .strftime("%Y%m%d%H%M")
        .astype(int)
    )
    return df


def rename_columns(df: pd.DataFrame, data_type: str, config: dict, logger: logging.Logger) -> pd.DataFrame:
    """
    Rename DataFrame columns based on configuration and standardize column names.
    """
    mapping = config.get(
        "renames_eddy" if data_type == "eddy" else "renames_met", {}
    )
    logger.debug(f"Renaming columns from {df.columns} to {mapping}")
    df.columns = df.columns.str.strip()
    df = df.rename(columns=mapping)
    df = normalize_prefixes(df, logger)
    df = modernize_soil_legacy(df, logger)
    df.columns = df.columns.str.upper()
    logger.debug(f"Len of renamed cols {len(df)}")
    return df


def normalize_prefixes(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Normalize column name prefixes related to soil and temperature measurements.
    """
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        for patt, repl in _PREFIX_PATTERNS.items():
            if patt.match(col):
                rename_map[col] = patt.sub(repl, col)
                break
        else:
            if re.match(r"^T_\d{1,3}cm_", col, flags=re.IGNORECASE):
                rename_map[col] = re.sub(r"^T_", "Ts_", col, flags=re.IGNORECASE)
    if rename_map:
        logger.debug("Prefix normalisation: %s", rename_map)
        df = df.rename(columns=rename_map)
    logger.debug(f"Len of normalized prefix cols {len(df)}")
    return df


def modernize_soil_legacy(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Update legacy soil sensor column names to a standardized format.
    """
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        m = _LEGACY_RE.match(col)
        if not m:
            continue
        prefix = m.group("prefix").upper()
        if prefix == "T":
            prefix = "TS"
        depth_cm = int(m.group("depth"))
        orient = m.group("orient").upper()
        depth_idx = _DEPTH_MAP.get(depth_cm)
        if depth_idx is None:
            continue
        replic = _ORIENT_MAP[orient]
        new_name = f"{prefix}_{replic}_{depth_idx}_1"
        rename_map[col] = new_name
    if rename_map:
        logger.info(f"Legacy soil columns modernised: {rename_map}")
        df = df.rename(columns=rename_map)
    return df


def apply_physical_limits(
    df: pd.DataFrame,
    how: str = "mask",
    inplace: bool = False,
    prefer_longest_key: bool = True,
    return_mask: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """
    Apply physical Min/Max bounds to columns.
    """
    if how not in {"mask", "clip"}:
        raise ValueError("how must be 'mask' or 'clip'")

    limits_dict = variable_limits.limits

    out = df if inplace else df.copy()
    keys = list(limits_dict.keys())
    if prefer_longest_key:
        keys.sort(key=len, reverse=True)

    col_map = {}
    for key in keys:
        matching_cols = [c for c in out.columns if str(c).startswith(key)]
        if not matching_cols:
            continue
        lim = limits_dict[key]
        mn = lim.get("Min", np.nan)
        mx = lim.get("Max", np.nan)
        for col in matching_cols:
            if col not in col_map or (
                prefer_longest_key and len(key) > len(col_map[col]["key"])
            ):
                col_map[col] = {"key": key, "Min": mn, "Max": mx}

    mask_df = pd.DataFrame(False, index=out.index, columns=out.columns)
    records = []

    for col, info in col_map.items():
        key = info["key"]
        mn = info["Min"]
        mx = info["Max"]
        ser = pd.to_numeric(out[col], errors="coerce")
        lower_ok = (
            ser >= mn
            if not (pd.isna(mn) or (isinstance(mn, float) and math.isnan(mn)))
            else pd.Series(True, index=ser.index)
        )
        upper_ok = (
            ser <= mx
            if not (pd.isna(mx) or (isinstance(mx, float) and math.isnan(mx)))
            else pd.Series(True, index=ser.index)
        )
        ok = lower_ok & upper_ok
        oor = ~ok
        n_below = int((~lower_ok & ser.notna()).sum())
        n_above = int((~upper_ok & ser.notna()).sum())
        n_oor = int((oor & ser.notna()).sum())
        if how == "mask":
            ser_out = ser.where(ok)
        else:
            ser_out = ser
            if not pd.isna(mn):
                ser_out = ser_out.clip(lower=mn)
            if not pd.isna(mx):
                ser_out = ser_out.clip(upper=mx)
        out[col] = ser_out.astype(float) if ser_out.isna().any() else ser_out
        mask_df[col] = oor
        records.append(
            {
                "column": col,
                "matched_key": key,
                "min": mn,
                "max": mx,
                "n_below": n_below,
                "n_above": n_above,
                "n_flagged": n_oor,
                "pct_flagged": (n_oor / len(ser) * 100.0) if len(ser) else 0.0,
            }
        )
    report = pd.DataFrame.from_records(records).sort_values(
        ["n_flagged", "column"], ascending=[False, True]
    )
    return (out, (mask_df if return_mask else None), report)


def apply_fixes(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Apply a set of minor, variable-specific data corrections.
    """
    df = tau_fixer(df)
    df = fix_swc_percent(df, logger)
    df = ssitc_scale(df, logger)
    return df


def tau_fixer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace zero values in the 'TAU' column with NaN.
    """
    if "TAU" in df.columns and "U_STAR" in df.columns:
        bad_idx = df["TAU"] == 0
        df.loc[bad_idx, "TAU"] = np.nan
    return df


def fix_swc_percent(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Convert fractional soil water content values to percent if applicable.
    """
    df = df.copy()

    def _fix_one(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        m = s.max(skipna=True)
        if pd.notna(m) and m <= 1.5:
            s = s * 100.0
            logger.debug(f"Converted {s.name} from fraction to percent")
        return s

    for name in [c for c in df.columns if str(c).startswith("SWC_")]:
        obj = df.loc[:, name]
        if isinstance(obj, pd.DataFrame):
            for sub in obj.columns:
                df[sub] = _fix_one(df[sub])
        else:
            df[name] = _fix_one(obj)
    return df


def fill_na_drop_dups(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    for col in df.columns:
        if col.endswith(".1"):
            col1 = col[:-2]
            col2 = col
            s1 = df[col1].replace(-9999, np.nan)
            s2 = df[col2].replace(-9999, np.nan)
            df[col1] = s1.combine_first(s2).fillna(-9999)
            logger.debug(f"Replaced {col1} with {col2}")
            df = df.drop([col2], axis=1)
        elif col.endswith(".2"):
            col1 = col[:-2]
            col2 = col
            s1 = df[col1].replace(-9999, np.nan)
            s2 = df[col2].replace(-9999, np.nan)
            df[col1] = s1.combine_first(s2).fillna(-9999)
            logger.debug(f"Replaced {col1} with {col2}")
            df = df.drop([col2], axis=1)
    return df


def ssitc_scale(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Scale SSITC test columns if values exceed expected thresholds.
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
                df[column] = scale_and_convert(df[column])
                logger.debug(f"Scaled SSITC {column}")
    logger.debug(f"Scaled SSITC len: {len(df)}")
    return df


def scale_and_convert(column: pd.Series) -> pd.Series:
    """
    Apply a rating transformation and convert values to float.
    """
    column = column.apply(rating)
    return column


def rating(x):
    """
    Categorize a numeric value into a discrete rating level.
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


def drop_extra_soil_columns(df: pd.DataFrame, config: dict, logger: logging.Logger) -> pd.DataFrame:
    """
    Drop redundant or unused soil probe columns.
    """
    df = df.copy()
    math_soils: Sequence[str] = config.get("math_soils_v2", [])
    to_drop: List[str] = []

    for col in df.columns:
        parts = col.split("_")
        if len(parts) >= 3 and parts[0] in {"SWC", "TS", "EC", "K"}:
            try:
                if int(parts[1]) >= SOIL_SENSOR_SKIP_INDEX:
                    to_drop.append(col)
                    continue
            except ValueError:
                pass
        if col in math_soils[: -DEFAULT_SOIL_DROP_LIMIT]:
            to_drop.append(col)
            continue
        if (
            parts[0] in {"VWC", "Ka"}
            or col.endswith("cm_N")
            or col.endswith("cm_S")
        ):
            to_drop.append(col)

    if to_drop:
        logger.info("Dropping %d redundant soil columns", len(to_drop))
        df = df.drop(columns=to_drop, errors="ignore")
    return df


def make_unique(cols):
    """
    Make column names unique by appending numeric suffixes to duplicates.
    """
    seen = {}
    out = []
    for c in cols:
        c = str(c)
        if c in seen:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out


def make_unique_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with duplicate column names suffixed .1, .2, ..."""
    df = df.copy()
    df.columns = make_unique(df.columns)
    return df


def set_number_types(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Convert columns in the DataFrame to numeric types where appropriate.
    """
    logger.debug(f"Setting number types: {df.head(3)}")
    dupes = pd.Series(df.columns).value_counts()
    logger.debug(dupes[dupes > 1])

    for col in df.columns:
        logger.debug(f"Setting number types {col}")
        pos = np.where(df.columns == col)[0]
        if len(pos) == 1:
            if col in ["MO_LENGTH", "RECORD", "FILE_NO", "DATALOGGER_NO"]:
                df[col] = pd.to_numeric(
                    df[col], downcast="integer", errors="coerce"
                )
            elif col in ["DATETIME_START"]:
                df[col] = df[col]
            elif col in ["TIMESTAMP_START", "TIMESTAMP_END", "SSITC"]:
                df[col] = pd.to_numeric(
                    df[col], downcast="integer", errors="coerce"
                )
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            logger.warning(f"Column {col} appears multiple times in DataFrame")
            for p in pos:
                s = df.iloc[:, p]
                if col in [
                    "MO_LENGTH",
                    "RECORD",
                    "FILE_NO",
                    "DATALOGGER_NO",
                    "TIMESTAMP_START",
                    "TIMESTAMP_END",
                    "SSITC",
                ]:
                    df.iloc[:, p] = pd.to_numeric(
                        s, downcast="integer", errors="coerce"
                    )
                elif col == "DATETIME_START":
                    continue
                else:
                    df.iloc[:, p] = pd.to_numeric(s, errors="coerce")
    logger.debug(f"Set number types: {len(df)}")
    return df


def drop_extras(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Drop extra or unwanted columns from the DataFrame based on configuration.
    """
    return df.drop(columns=config.get("drop_cols", []), errors="ignore")


def col_order(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Reorder DataFrame columns to place priority columns first.
    """
    first_cols = ["TIMESTAMP_END", "TIMESTAMP_START"]
    for col in first_cols:
        if col in df.columns:
            ncol = df.pop(col)
            df.insert(0, col, ncol)
    logger.debug(f"Column Order: {df.columns}")
    return df
