"""
micromet.io.readers
===================

Readers for common micrometeorological data formats used in MicroMet:
- Campbell Scientific TOA5 logger CSVs
- AmeriFlux half-hourly CSVs (upload-style)
- Generic CSVs with INI/YAML-driven column mapping
- Parquet files

Core features:
- Robust timestamp parsing and normalization to local STANDARD time (fixed UTC offset, no DST)
- Auto-detection of header rows (for logger exports with extra rows)
- Missing value handling (-9999 -> NaN by default)
- Optional standardization of column names to AmeriFlux base names
- Utilities to enforce 30-minute cadence (non-destructive)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Mapping, Iterable

import configparser
import io
import re

import numpy as np
import pandas as pd


# ----------------------------- #
# Constants / Defaults
# ----------------------------- #

DEFAULT_MISSING_SENTINEL = -9999
HALFHOUR_MINUTES = 30

# Minimal set of AmeriFlux base variable names we care about downstream
AMF_BASE_VARS = [
    "SW_IN",
    "SW_OUT",
    "LW_IN",
    "LW_OUT",
    "NETRAD",
    "H",
    "LE",
    "G",
    "TA",
]

TIMESTAMP_START = "TIMESTAMP_START"
TIMESTAMP_END = "TIMESTAMP_END"


# ----------------------------- #
# Time utilities
# ----------------------------- #


def _fixed_offset_tz_name(utc_offset_hours: int) -> str:
    """
    Build an Etc/GMT±N tz name from integer offset hours.
    Note: Etc/GMT signs are inverted (Etc/GMT+7 means UTC-7).
    """
    return f"Etc/GMT{(-utc_offset_hours):+d}"


def to_local_standard_time(series: pd.Series, utc_offset_hours: int) -> pd.Series:
    """
    Convert naive or UTC timestamps to local **standard** time using a fixed offset (no DST).

    Parameters
    ----------
    series : pd.Series (datetime64[ns] or str)
    utc_offset_hours : int
        Fixed UTC offset in hours (e.g., -7 for US Mountain Standard Time).

    Returns
    -------
    pd.Series (datetime64[ns])
    """
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    tzname = _fixed_offset_tz_name(utc_offset_hours)
    return ts.dt.tz_convert(tzname).dt.tz_localize(None)


def make_timestamp_end(start: pd.Series, minutes: int = HALFHOUR_MINUTES) -> pd.Series:
    return pd.to_datetime(start) + pd.Timedelta(minutes=minutes)


def enforce_halfhour_cadence(
    df: pd.DataFrame,
    start_col: str = TIMESTAMP_START,
    end_col: str = TIMESTAMP_END,
    repair: bool = False,
) -> pd.DataFrame:
    """
    Check (and optionally repair) 30-min cadence. Non-destructive by default.

    Parameters
    ----------
    df : pd.DataFrame
    start_col : str
    end_col : str
    repair : bool
        If True, reindex to a perfect 30-min grid from min to max TIMESTAMP_START.
        Missing rows are inserted with NaN; END is recomputed.

    Returns
    -------
    pd.DataFrame
    """
    if start_col not in df:
        return df

    if not repair:
        return df

    ts = pd.to_datetime(df[start_col])
    idx = pd.date_range(ts.min(), ts.max(), freq="30T")
    out = df.set_index(ts).reindex(idx).rename_axis(start_col).reset_index()
    out[end_col] = make_timestamp_end(out[start_col], HALFHOUR_MINUTES)
    return out


# ----------------------------- #
# General helpers
# ----------------------------- #


def coerce_missing(
    df: pd.DataFrame, missing_sentinel: float | int = DEFAULT_MISSING_SENTINEL
) -> pd.DataFrame:
    """
    Replace known missing value sentinel with NaN across numeric-like columns.
    """
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out.loc[out[c] == missing_sentinel, c] = np.nan
        else:
            # Try coercion for string columns that may contain sentinel text
            coerced = pd.to_numeric(out[c], errors="coerce")
            mask = coerced == missing_sentinel
            if mask.any():
                out.loc[mask, c] = np.nan
    return out


def _looks_like_header(cells: Iterable[str]) -> bool:
    """
    Heuristic to determine if a row looks like a header row.

    - Should contain at least one timestamp-like token (TIMESTAMP, TIMESTAMP_START, etc.)
    - Should not be mostly numeric tokens
    """
    s = [str(x).strip().upper() for x in cells]
    if any(tok in s for tok in ("TIMESTAMP", "TIMESTAMP_START", "DATETIME", "DATE")):
        return True
    numericish = sum(bool(re.fullmatch(r"[-+]?\d+(\.\d+)?", x)) for x in s)
    return numericish < max(2, len(s) // 3)


def find_header_row(
    path: str | Path, max_scan: int = 20, delimiter: Optional[str] = None
) -> int:
    """
    Scan the first `max_scan` lines and return the (0-based) line index of the header row.

    If not found, returns 0 (assume first row is header).
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i in range(max_scan):
            line = f.readline()
            if not line:
                break
            cells = line.strip().split(delimiter or ",")
            if _looks_like_header(cells):
                return i
    return 0


def try_parse_datetime(series: pd.Series, fmt: Optional[str] = None) -> pd.Series:
    """
    Try parsing datetimes with an explicit fmt, falling back to pandas inference.

    Returns a tz-aware series in UTC if possible; otherwise naive.
    """
    s = pd.to_datetime(series, format=fmt, errors="coerce")
    if s.notna().any():
        return s
    # Fallback
    return pd.to_datetime(series, errors="coerce")


def _ensure_timestamp_columns(
    df: pd.DataFrame,
    tz_offset_hours: int,
    start_col: Optional[str] = None,
    end_col: Optional[str] = None,
    infer_format: Optional[str] = "%Y%m%d%H%M",
) -> pd.DataFrame:
    """
    Make sure TIMESTAMP_START and TIMESTAMP_END exist and are localized to local standard time.
    """
    out = df.copy()

    # Heuristics for start/end column names
    candidates_start = [
        start_col,
        "TIMESTAMP_START",
        "TIMESTAMP",
        "DateTime",
        "DATE_TIME",
        "DATE",
    ]
    candidates_end = [end_col, "TIMESTAMP_END"]

    start_name = next((c for c in candidates_start if c and c in out.columns), None)
    if start_name is None:
        # fallback: first column if it looks like datetime
        first = out.columns[0]
        start_name = first

    # Parse start
    start_parsed = try_parse_datetime(out[start_name], fmt=infer_format)
    out[TIMESTAMP_START] = to_local_standard_time(start_parsed, tz_offset_hours)

    # End
    end_name = next((c for c in candidates_end if c and c in out.columns), None)
    if end_name is not None:
        end_parsed = try_parse_datetime(out[end_name], fmt=infer_format)
        out[TIMESTAMP_END] = to_local_standard_time(end_parsed, tz_offset_hours)
    else:
        out[TIMESTAMP_END] = make_timestamp_end(out[TIMESTAMP_START], HALFHOUR_MINUTES)

    return out


def apply_column_mapping(df: pd.DataFrame, mapping: Mapping[str, str]) -> pd.DataFrame:
    """
    Rename columns using a provided mapping (source -> AmeriFlux base name).

    Unmapped columns are preserved.
    """
    if not mapping:
        return df
    return df.rename(columns=mapping)


# ----------------------------- #
# INI / Config helpers
# ----------------------------- #


def load_micromet_ini(ini_path: str | Path) -> dict:
    """
    Load a MicroMet-style INI with [METADATA] and [DATA] sections.

    Returns
    -------
    dict with keys:
      site (dict), data (dict), missing_data_value (float|int), date_parser (str|None), skiprows (int)
    """
    p = Path(ini_path)
    cp = configparser.ConfigParser()
    cp.read(p)

    meta = dict(cp.items("METADATA")) if cp.has_section("METADATA") else {}
    data = dict(cp.items("DATA")) if cp.has_section("DATA") else {}

    # Normalize a few common keys
    missing_val = meta.get("missing_data_value", None)
    missing_val = (
        float(missing_val) if missing_val is not None else DEFAULT_MISSING_SENTINEL
    )
    skiprows = int(meta.get("skiprows", "0"))
    date_parser = meta.get("date_parser", None)

    # Site info
    site = {
        "id": meta.get("site_id", ""),
        "latitude": float(meta.get("station_latitude", "0") or 0),
        "longitude": float(meta.get("station_longitude", "0") or 0),
        "elevation": float(meta.get("station_elevation", "0") or 0),
        "utc_offset_hours": int(meta.get("utc_offset", "0") or 0),
    }

    return {
        "site": site,
        "data": data,
        "missing_data_value": missing_val,
        "date_parser": date_parser,
        "skiprows": skiprows,
    }


def mapping_from_ini_data_section(data_section: Mapping[str, str]) -> dict[str, str]:
    """
    Create an AmeriFlux base-name mapping from the INI [DATA] keys.

    The INI often has entries like:
      net_radiation_col = NETRAD
      latent_heat_flux_col = LE
      avg_temp_col = T_SONIC  (we map that to TA)

    Returns a mapping suitable for `apply_column_mapping`.
    """
    m: dict[str, str] = {}
    # Radiation & fluxes
    if "shortwave_in_col" in data_section:
        m[data_section["shortwave_in_col"]] = "SW_IN"
    if "shortwave_out_col" in data_section:
        m[data_section["shortwave_out_col"]] = "SW_OUT"
    if "longwave_in_col" in data_section:
        m[data_section["longwave_in_col"]] = "LW_IN"
    if "longwave_out_col" in data_section:
        m[data_section["longwave_out_col"]] = "LW_OUT"
    if "net_radiation_col" in data_section:
        m[data_section["net_radiation_col"]] = "NETRAD"
    if "sensible_heat_flux_col" in data_section:
        m[data_section["sensible_heat_flux_col"]] = "H"
    if "latent_heat_flux_col" in data_section:
        m[data_section["latent_heat_flux_col"]] = "LE"
    if "ground_flux_col" in data_section:
        m[data_section["ground_flux_col"]] = "G"

    # Temperature
    # Many stations label average sonic temp "T_SONIC"; AmeriFlux base uses TA (°C).
    if "avg_temp_col" in data_section:
        m[data_section["avg_temp_col"]] = "TA"

    # Timestamp
    if "datestring_col" in data_section:
        m[data_section["datestring_col"]] = TIMESTAMP_START

    return m


# ----------------------------- #
# Readers
# ----------------------------- #


def read_toa5(
    path: str | Path,
    tz_offset_hours: int,
    *,
    header_row: Optional[int] = None,
    delimiter: Optional[str] = None,
    missing_sentinel: float | int = DEFAULT_MISSING_SENTINEL,
    time_format: Optional[str] = "%Y%m%d%H%M",
    start_col: Optional[str] = None,
    end_col: Optional[str] = None,
    column_mapping: Optional[Mapping[str, str]] = None,
    enforce_30min: bool = True,
) -> pd.DataFrame:
    """
    Read a Campbell Scientific TOA5-style CSV and standardize timestamps/columns.

    Parameters
    ----------
    path : str|Path
    tz_offset_hours : int
        Fixed offset from UTC (e.g., -7 for MST) for local STANDARD time.
    header_row : int or None
        If None, auto-detect a header row within the first 20 lines.
    delimiter : str or None
        Delimiter override (default ',' if None).
    missing_sentinel : float|int
    time_format : str or None
        Try this format first (e.g., '%Y%m%d%H%M'), then fall back to inference.
    start_col, end_col : str or None
        Optional column names for start/end timestamp columns.
    column_mapping : Mapping[str,str] or None
        Source -> AmeriFlux base mapping. If provided, columns are renamed.
    enforce_30min : bool
        If True, reindex to perfect 30-min cadence (inserts missing rows with NaN).

    Returns
    -------
    pd.DataFrame with TIMESTAMP_START/TIMESTAMP_END in local standard time, and mapped columns.
    """
    p = Path(path)
    if header_row is None:
        header_row = find_header_row(p, delimiter=delimiter)

    df = pd.read_csv(p, header=header_row, delimiter=delimiter or ",", dtype=str)
    # Standardize timestamps
    df = _ensure_timestamp_columns(
        df,
        tz_offset_hours,
        start_col=start_col,
        end_col=end_col,
        infer_format=time_format,
    )
    # Coerce numeric columns and missing sentinel
    df = df.apply(pd.to_numeric, errors="ignore")
    df = coerce_missing(df, missing_sentinel)
    # Apply mapping if any
    if column_mapping:
        df = apply_column_mapping(df, column_mapping)
    # Optionally enforce grid
    if enforce_30min:
        df = enforce_halfhour_cadence(df, repair=True)
    return df


def read_ameriflux_hh_csv(
    path: str | Path,
    tz_offset_hours: int,
    *,
    missing_sentinel: float | int = DEFAULT_MISSING_SENTINEL,
    has_units_row: bool = True,
    column_mapping: Optional[Mapping[str, str]] = None,
    enforce_30min: bool = True,
) -> pd.DataFrame:
    """
    Read an AmeriFlux-like half-hourly CSV (TIMESTAMP_START, TIMESTAMP_END, ...).

    Parameters
    ----------
    path : str|Path
    tz_offset_hours : int
    missing_sentinel : float|int
    has_units_row : bool
        If True, drops the second header row commonly used for units.
    column_mapping : Mapping[str,str] or None
        Apply extra renames if columns differ from your internal base.
    enforce_30min : bool

    Returns
    -------
    pd.DataFrame
    """
    p = Path(path)
    # Read first two rows to decide whether to drop units row
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        head = "".join([next(f) for _ in range(2)])
    preview = pd.read_csv(io.StringIO(head))
    drop_units = (
        has_units_row
        and preview.shape[0] == 1
        and preview.columns.str.contains(
            "UNIT|W/m2|C|%|hPa", case=False, regex=True
        ).any()
    )

    df = pd.read_csv(p, skiprows=1 if drop_units else 0)
    # Ensure timestamp columns exist and localize
    if TIMESTAMP_START not in df.columns:
        # Some files might have 'TIMESTAMP'; rename it
        if "TIMESTAMP" in df.columns:
            df.rename(columns={"TIMESTAMP": TIMESTAMP_START}, inplace=True)
    df = _ensure_timestamp_columns(df, tz_offset_hours, infer_format="%Y%m%d%H%M")

    df = df.apply(pd.to_numeric, errors="ignore")
    df = coerce_missing(df, missing_sentinel)

    if column_mapping:
        df = apply_column_mapping(df, column_mapping)

    if enforce_30min:
        df = enforce_halfhour_cadence(df, repair=True)

    return df


def read_generic_csv(
    path: str | Path,
    tz_offset_hours: int,
    *,
    config_ini: Optional[str | Path] = None,
    missing_sentinel: float | int = DEFAULT_MISSING_SENTINEL,
    delimiter: Optional[str] = None,
    time_format: Optional[str] = "%Y%m%d%H%M",
    enforce_30min: bool = True,
) -> pd.DataFrame:
    """
    Read a generic CSV using an optional MicroMet-style INI to map columns.

    If `config_ini` is provided, the INI [DATA] section drives:
      - datestring_col -> TIMESTAMP_START
      - *_col -> AmeriFlux base names (SW_IN, LE, H, G, NETRAD, TA, ...)

    Parameters
    ----------
    path : str|Path
    tz_offset_hours : int
    config_ini : str|Path or None
    missing_sentinel : float|int
    delimiter : str or None
    time_format : str or None
    enforce_30min : bool

    Returns
    -------
    pd.DataFrame
    """
    column_mapping: dict[str, str] = {}
    date_fmt = time_format

    if config_ini:
        cfg = load_micromet_ini(config_ini)
        column_mapping = mapping_from_ini_data_section(cfg["data"])
        # Allow INI to override missing sentinel and date_parser
        missing_sentinel = cfg.get("missing_data_value", missing_sentinel)
        if cfg.get("date_parser"):
            date_fmt = cfg["date_parser"]

    header_row = find_header_row(path, delimiter=delimiter)
    df = pd.read_csv(path, header=header_row, delimiter=delimiter or ",", dtype=str)

    # Timestamp standardization
    start_col_guess = next(
        (k for k, v in column_mapping.items() if v == TIMESTAMP_START), None
    )
    df = _ensure_timestamp_columns(
        df, tz_offset_hours, start_col=start_col_guess, infer_format=date_fmt
    )

    # Coerce and map
    df = df.apply(pd.to_numeric, errors="ignore")
    df = coerce_missing(df, missing_sentinel)
    if column_mapping:
        df = apply_column_mapping(df, column_mapping)

    if enforce_30min:
        df = enforce_halfhour_cadence(df, repair=True)

    return df


def read_parquet(path: str | Path) -> pd.DataFrame:
    """
    Read a Parquet file into a DataFrame. Timestamps are left as-is.
    """
    return pd.read_parquet(path)


# ----------------------------- #
# Convenience loader
# ----------------------------- #


def load_dataset(
    path: str | Path,
    tz_offset_hours: int,
    *,
    flavor: str = "auto",
    config_ini: Optional[str | Path] = None,
    missing_sentinel: float | int = DEFAULT_MISSING_SENTINEL,
    enforce_30min: bool = True,
) -> pd.DataFrame:
    """
    High-level loader with simple flavor selection.

    Parameters
    ----------
    path : str|Path
    tz_offset_hours : int
    flavor : {"auto","toa5","ameriflux","generic","parquet"}
    config_ini : str|Path or None
    missing_sentinel : float|int
    enforce_30min : bool

    Returns
    -------
    pd.DataFrame
    """
    p = Path(path)
    if flavor == "parquet" or p.suffix.lower() in {".parq", ".parquet"}:
        return read_parquet(p)

    # Simple auto sniffing by header tokens
    header_idx = find_header_row(p)
    head = pd.read_csv(p, header=header_idx, nrows=1)
    cols_upper = {c.upper() for c in head.columns}

    if (
        flavor == "toa5"
        or ("RECNBR" in cols_upper)
        or ("PANEL_T" in cols_upper)
        or ("CR1000X" in " ".join(cols_upper))
    ):
        return read_toa5(
            p,
            tz_offset_hours,
            header_row=header_idx,
            missing_sentinel=missing_sentinel,
            enforce_30min=enforce_30min,
        )

    if flavor == "ameriflux" or (
        TIMESTAMP_START in cols_upper and TIMESTAMP_END in cols_upper
    ):
        return read_ameriflux_hh_csv(
            p,
            tz_offset_hours,
            missing_sentinel=missing_sentinel,
            enforce_30min=enforce_30min,
        )

    # Generic with optional INI mapping
    return read_generic_csv(
        p,
        tz_offset_hours,
        config_ini=config_ini,
        missing_sentinel=missing_sentinel,
        enforce_30min=enforce_30min,
    )


# ----------------------------- #
# INI-driven convenience loader #
# ----------------------------- #


def load_from_ini(
    ini_path: str | Path, *, enforce_30min: bool = True, flavor_hint: str = "auto"
) -> pd.DataFrame:
    """
    High-level one-liner: read a dataset using a MicroMet-style INI.

    - Parses [METADATA] for utc_offset, missing_data_value, date_parser, skiprows (optional)
    - Builds an AmeriFlux base-name column mapping from [DATA]
    - Loads the file at METADATA.climate_file_path (relative to INI unless absolute)
    - Normalizes timestamps to local STANDARD time (no DST)
    - Optionally enforces a perfect 30-min cadence (missing rows inserted as NaN)

    Parameters
    ----------
    ini_path : str|Path
    enforce_30min : bool
    flavor_hint : {"auto","toa5","ameriflux","generic"}
        If your source is known, set it to skip sniffing.

    Returns
    -------
    pd.DataFrame
    """
    ini_path = Path(ini_path)
    cfg = load_micromet_ini(ini_path)

    tz = int(cfg["site"]["utc_offset_hours"])
    missing = cfg.get("missing_data_value", DEFAULT_MISSING_SENTINEL)
    date_fmt = cfg.get("date_parser", None)

    # Resolve climate_file_path relative to INI if it's not absolute
    raw_path = Path(
        cfg["site"].get("climate_file_path", cfg.get("climate_file_path", ""))
    )
    # Fall back to [METADATA].climate_file_path if present
    if not raw_path:
        raw_path = Path(cfg.get("climate_file_path", ""))

    if not raw_path.is_absolute():
        raw_path = (ini_path.parent / raw_path).resolve()

    mapping = mapping_from_ini_data_section(cfg["data"])

    if flavor_hint == "ameriflux":
        return read_ameriflux_hh_csv(
            raw_path,
            tz,
            missing_sentinel=missing,
            column_mapping=mapping,
            enforce_30min=enforce_30min,
        )

    if flavor_hint == "toa5":
        return read_toa5(
            raw_path,
            tz,
            header_row=find_header_row(raw_path),
            missing_sentinel=missing,
            time_format=date_fmt or "%Y%m%d%H%M",
            column_mapping=mapping,
            enforce_30min=enforce_30min,
        )

    # generic (CSV) as default
    return read_generic_csv(
        raw_path,
        tz,
        config_ini=ini_path,
        missing_sentinel=missing,
        time_format=date_fmt or "%Y%m%d%H%M",
        enforce_30min=enforce_30min,
    )
