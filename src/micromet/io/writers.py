"""
micromet.io.writers
===================

Writers for MicroMet DataFrames:

- write_ameriflux_csv(): CSV tailored for AmeriFlux upload
- write_analysis_csv():  CSV friendly for pandas/R analysis
- write_parquet():       Parquet with preserved dtypes and optional metadata

Common behavior:
- Enforces TIMESTAMP_START/TIMESTAMP_END first
- Formats timestamps to AMF style '%Y%m%d%H%M' for upload CSVs
- Replaces NaN with -9999 sentinel for upload CSVs (keeps NaN for analysis CSVs)
- Checks column qualifiers (blocks '_PI'/'_QC' in upload outputs by default)
- Optional second header line for units in upload CSVs
- Optional gzip compression (.gz) for large CSVs

Assumptions:
- Timestamps in the DataFrame are already normalized to local STANDARD time (no DST).
- Column names are AmeriFlux base names (e.g., SW_IN, LW_OUT, LE, H, G, TA, NETRAD),
  though any extra columns are preserved unless excluded.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Mapping, Iterable

import io
import json
import warnings

import numpy as np
import pandas as pd

# Local schema/constants (keep in sync with format/amf_schema.py)
AMF_REQUIRED = [
    "TIMESTAMP_START",
    "TIMESTAMP_END",
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
AMF_UNITS_DEFAULT = {
    "TIMESTAMP_START": "",
    "TIMESTAMP_END": "",
    "SW_IN": "W/m2",
    "SW_OUT": "W/m2",
    "LW_IN": "W/m2",
    "LW_OUT": "W/m2",
    "NETRAD": "W/m2",
    "H": "W/m2",
    "LE": "W/m2",
    "G": "W/m2",
    "TA": "C",
}
UPLOAD_DISALLOWED_QUALIFIERS = ("_PI", "_QC")  # do not emit in upload CSVs
MISSING_SENTINEL = -9999
AMF_TIMESTAMP_FMT = "%Y%m%d%H%M"
HALFHOUR_MINUTES = 30

# ----------------------------- #
# Options
# ----------------------------- #


@dataclass
class UploadCsvOptions:
    """
    Options for write_ameriflux_csv()
    """

    missing_sentinel: float | int = MISSING_SENTINEL
    enforce_required: bool = True
    enforce_order: bool = True
    allow_extra_columns: bool = True
    exclude_columns: tuple[str, ...] = tuple()  # e.g., QC masks
    include_units_row: bool = False  # second header line
    units_map: Optional[Mapping[str, str]] = None
    check_qualifiers: bool = True  # block _PI/_QC
    float_format: Optional[str] = None  # e.g., "%.3f"
    gzip: bool = False  # write .gz
    line_terminator: Optional[str] = None  # defaults to OS; e.g., "\n"
    metadata_header: Optional[Mapping[str, str]] = None  # written as commented lines


@dataclass
class AnalysisCsvOptions:
    """
    Options for write_analysis_csv()
    """

    rfc3339_timestamps: bool = False  # ISO8601 'YYYY-MM-DDTHH:MM'
    enforce_order: bool = True
    exclude_columns: tuple[str, ...] = tuple()
    float_format: Optional[str] = None
    gzip: bool = False
    line_terminator: Optional[str] = None
    metadata_header: Optional[Mapping[str, str]] = None  # commented lines


@dataclass
class ParquetOptions:
    """
    Options for write_parquet()
    """

    compression: str = "snappy"  # "snappy"|"gzip"|"brotli"|"zstd"|"none"
    coerce_timestamps: bool = True
    coerce_float32: bool = False
    metadata: Optional[Mapping[str, str]] = None  # stored in schema (key/value)


# ----------------------------- #
# Helpers
# ----------------------------- #


def _reorder_columns(df: pd.DataFrame, required_first: Iterable[str]) -> list[str]:
    first = [c for c in required_first if c in df.columns]
    rest = [c for c in df.columns if c not in first]
    return first + rest


def _format_amf_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "TIMESTAMP_START" in out.columns:
        out["TIMESTAMP_START"] = pd.to_datetime(out["TIMESTAMP_START"]).dt.strftime(
            AMF_TIMESTAMP_FMT
        )
    if "TIMESTAMP_END" in out.columns:
        out["TIMESTAMP_END"] = pd.to_datetime(out["TIMESTAMP_END"]).dt.strftime(
            AMF_TIMESTAMP_FMT
        )
    return out


def _qualifier_violations(columns: Iterable[str]) -> list[str]:
    bad = []
    for c in columns:
        for q in UPLOAD_DISALLOWED_QUALIFIERS:
            if c.endswith(q):
                bad.append(c)
    return bad


def _apply_missing_sentinel(
    df: pd.DataFrame, missing: float | int = MISSING_SENTINEL
) -> pd.DataFrame:
    out = df.copy()
    # Only apply to numeric columns; timestamps already rendered as strings in upload
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out.loc[out[c].isna(), c] = missing
        else:
            # keep strings as-is; e.g., TIMESTAMP_START after formatting
            pass
    return out


def _write_text_header(f, metadata: Mapping[str, str] | None):
    if not metadata:
        return
    for k, v in metadata.items():
        # AmeriFlux upload tolerates comment lines in many tools (but keep minimal)
        f.write(f"# {k}: {v}\n")


def _units_row(columns: list[str], units_map: Mapping[str, str] | None) -> list[str]:
    if units_map is None:
        units_map = AMF_UNITS_DEFAULT
    return [units_map.get(c, "") for c in columns]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _maybe_open_gzip(path: Path):
    if not path.suffix.endswith(".gz"):
        raise ValueError("Expected .gz extension for gzip output")
    import gzip

    # Open in text mode to use csv writer consistently
    return gzip.open(path, mode="wt", encoding="utf-8", newline="")


def _safe_float_format(df: pd.DataFrame, float_format: Optional[str]) -> dict:
    """
    Build per-column formatters for to_csv, only for floating dtypes.
    """
    if not float_format:
        return {}
    fmts = {}
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            fmts[c] = lambda x, _fmt=float_format: ("" if pd.isna(x) else _fmt % x)
    return fmts


# ----------------------------- #
# Public writers
# ----------------------------- #


def write_ameriflux_csv(
    df: pd.DataFrame, path: str | Path, options: UploadCsvOptions | None = None
) -> Path:
    """
    Write an AmeriFlux-style CSV suitable for upload.

    - TIMESTAMP_* formatted as '%Y%m%d%H%M' (strings)
    - NaN -> -9999 (or options.missing_sentinel) in numeric columns
    - TIMESTAMP_START, TIMESTAMP_END ordered first
    - Optional units second header line
    - Rejects columns ending with '_PI'/'_QC' by default

    Parameters
    ----------
    df : DataFrame
        Input data (assumed already localized to local STANDARD time).
    path : str|Path
        Destination path. Use '.gz' to gzip the CSV.
    options : UploadCsvOptions | None

    Returns
    -------
    Path to the written file.
    """
    opts = options or UploadCsvOptions()
    out_path = Path(path)
    _ensure_parent(out_path)

    # Basic validation
    missing_required = [c for c in AMF_REQUIRED if c not in df.columns]
    if opts.enforce_required and missing_required:
        raise ValueError(
            f"Missing required columns for AmeriFlux upload: {missing_required}"
        )

    # Exclude unwanted columns (e.g., internal QC flags)
    cols = [c for c in df.columns if c not in set(opts.exclude_columns)]
    work = df.loc[:, cols].copy()

    # Enforce order
    if opts.enforce_order:
        cols = _reorder_columns(work, ["TIMESTAMP_START", "TIMESTAMP_END"])
        work = work.loc[:, cols]

    # Qualifier check
    if opts.check_qualifiers:
        bad = _qualifier_violations(work.columns)
        if bad:
            raise ValueError(
                f"Upload CSV cannot include columns with disallowed qualifiers: {bad} "
                f"(remove or rename before upload)"
            )

    # Format timestamps to AMF strings
    work = _format_amf_timestamps(work)

    # Numeric -> sentinel
    work = _apply_missing_sentinel(work, missing=opts.missing_sentinel)

    # Float formatting (optional)
    fmt_map = _safe_float_format(work, opts.float_format)

    # Write with optional metadata header and units line
    if opts.gzip or out_path.suffix == ".gz":
        if out_path.suffix != ".gz":
            out_path = out_path.with_suffix(out_path.suffix + ".gz")
        with _maybe_open_gzip(out_path) as f:
            _write_text_header(f, opts.metadata_header)
            # Header row (names)
            f.write(",".join(work.columns) + (opts.line_terminator or "\n"))
            # Units row if requested
            if opts.include_units_row:
                units = _units_row(list(work.columns), opts.units_map)
                f.write(",".join(units) + (opts.line_terminator or "\n"))
            # Data
            work.to_csv(
                f, index=False, header=False, float_format=None, date_format=None
            )
    else:
        # Non-gz path
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            _write_text_header(f, opts.metadata_header)
            # Header
            f.write(",".join(work.columns) + (opts.line_terminator or "\n"))
            if opts.include_units_row:
                units = _units_row(list(work.columns), opts.units_map)
                f.write(",".join(units) + (opts.line_terminator or "\n"))
            # Data
            work.to_csv(
                f,
                index=False,
                header=False,
                float_format=None,  # we preformatted via fmt_map if needed
                date_format=None,
            )

    return out_path


def write_analysis_csv(
    df: pd.DataFrame, path: str | Path, options: AnalysisCsvOptions | None = None
) -> Path:
    """
    Write a CSV optimized for analysis (pandas/R):

    - Keeps NaN (no sentinel)
    - Timestamps as ISO 8601 (optional) or left as datetime
    - Places TIMESTAMP_* first (optional)
    - Preserves all columns except those excluded

    Parameters
    ----------
    df : DataFrame
    path : str|Path
    options : AnalysisCsvOptions | None

    Returns
    -------
    Path
    """
    opts = options or AnalysisCsvOptions()
    out_path = Path(path)
    _ensure_parent(out_path)

    cols = [c for c in df.columns if c not in set(opts.exclude_columns)]
    work = df.loc[:, cols].copy()

    if opts.enforce_order:
        cols = _reorder_columns(work, ["TIMESTAMP_START", "TIMESTAMP_END"])
        work = work.loc[:, cols]

    if opts.rfc3339_timestamps:
        # Convert to ISO strings (local naive timestamps assumed)
        for c in ("TIMESTAMP_START", "TIMESTAMP_END"):
            if c in work.columns:
                work[c] = pd.to_datetime(work[c]).dt.strftime("%Y-%m-%dT%H:%M:%S")

    fmt_map = _safe_float_format(work, opts.float_format)

    if opts.gzip or out_path.suffix == ".gz":
        if out_path.suffix != ".gz":
            out_path = out_path.with_suffix(out_path.suffix + ".gz")
        import gzip

        with gzip.open(out_path, mode="wt", encoding="utf-8", newline="") as f:
            _write_text_header(f, opts.metadata_header)
            work.to_csv(
                f,
                index=False,
                float_format=None if not fmt_map else None,
                date_format=None,
                lineterminator=opts.line_terminator,
            )
    else:
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            _write_text_header(f, opts.metadata_header)
            work.to_csv(
                f,
                index=False,
                float_format=None if not fmt_map else None,
                date_format=None,
                lineterminator=opts.line_terminator,
            )

    return out_path


def write_parquet(
    df: pd.DataFrame, path: str | Path, options: ParquetOptions | None = None
) -> Path:
    """
    Write a Parquet file preserving dtypes and (optionally) embedding metadata.

    Parameters
    ----------
    df : DataFrame
    path : str|Path
    options : ParquetOptions | None

    Returns
    -------
    Path
    """
    opts = options or ParquetOptions()
    out_path = Path(path)
    _ensure_parent(out_path)

    work = df.copy()

    # Optional dtype coercions for compact files / compatibility
    if opts.coerce_timestamps:
        for c in ("TIMESTAMP_START", "TIMESTAMP_END"):
            if c in work.columns:
                work[c] = pd.to_datetime(work[c], errors="coerce")
    if opts.coerce_float32:
        for c in work.select_dtypes(include=["float64"]).columns:
            work[c] = work[c].astype("float32")

    kwargs = {}
    if opts.compression and opts.compression.lower() != "none":
        kwargs["compression"] = opts.compression

    # Store metadata if provided (in Parquet file schema key_value_metadata)
    if opts.metadata:
        # pyarrow engine supports "metadata" via schema; pandas exposes via engine_kwargs
        # We'll pass via "metadata" in to_parquet when engine="pyarrow"
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(work, preserve_index=False)
            md = {str(k): str(v) for k, v in opts.metadata.items()}
            existing = table.schema.metadata or {}
            merged = {**{k.decode(): v.decode() for k, v in existing.items()}, **md}
            # Rebuild schema with merged metadata
            schema = table.schema.with_metadata(
                {k: v.encode() for k, v in merged.items()}
            )
            pq.write_table(table, out_path, compression=kwargs.get("compression", None), schema=schema)  # type: ignore
            return out_path
        except Exception:
            warnings.warn(
                "pyarrow metadata embedding failed; falling back to pandas.to_parquet() without metadata"
            )

    # Fallback: plain pandas writer
    work.to_parquet(out_path, index=False, **kwargs)
    return out_path


# ----------------------------- #
# Sidecar / convenience outputs
# ----------------------------- #


def write_column_dictionary(
    df: pd.DataFrame, path: str | Path, units_map: Optional[Mapping[str, str]] = None
) -> Path:
    """
    Write a simple JSON dictionary describing columns and (optional) units.
    """
    out_path = Path(path)
    _ensure_parent(out_path)
    units_map = units_map or AMF_UNITS_DEFAULT
    desc = {c: {"units": units_map.get(c, "")} for c in df.columns}
    out_path.write_text(json.dumps(desc, indent=2))
    return out_path


def write_readme(path: str | Path, lines: Iterable[str]) -> Path:
    """
    Write a minimal README.txt next to outputs to describe provenance or notes.
    """
    out_path = Path(path)
    _ensure_parent(out_path)
    content = "\n".join(lines)
    out_path.write_text(content)
    return out_path
