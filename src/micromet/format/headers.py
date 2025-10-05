#!/usr/bin/env python3
"""
Header detection and repair utilities for delimited text files.

This module provides functions to detect missing headers in data files and
repair them by borrowing headers from peer files. It supports both single-file
processing and batch operations across directories.

Key Features
------------
- Automatic delimiter detection using csv.Sniffer with fallback heuristics
- Header presence detection with multiple strategies
- Peer file matching based on filename similarity and column count
- Directory-based batch processing for duplicate files
- Support for UTF-8, UTF-8-sig, and Latin-1 encodings
"""
from __future__ import annotations

import csv
import io
import re
import shutil
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

COMMON_DELIMITERS = [",", "\t", ";", "|", " "]  # space last (least likely)
DEFAULT_ENCODINGS = ["utf-8-sig", "utf-8", "latin-1"]
DEFAULT_SAMPLE_SIZE = 64_000


# ──────────────────────────────────────────────────────────────────────────────
# CORE FILE I/O AND ENCODING
# ──────────────────────────────────────────────────────────────────────────────

def open_text(path: Path, encodings: list[str] | None = None) -> io.TextIOWrapper:
    """
    Open a text file, trying a list of encodings until one succeeds.

    Parameters
    ----------
    path : Path
        The path to the text file.
    encodings : list[str], optional
        A list of character encodings to try, in order.
        Defaults to ["utf-8-sig", "utf-8", "latin-1"].

    Returns
    -------
    io.TextIOWrapper
        An open file object.

    Raises
    ------
    Exception
        If all attempted encodings fail, the last exception is re-raised.
    """
    if encodings is None:
        encodings = DEFAULT_ENCODINGS
    last_err = None
    for enc in encodings:
        try:
            return open(path, "r", encoding=enc, newline="")
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise last_err  # type: ignore[misc]


def get_first_line_raw(path: Path) -> str:
    """
    Return the first line of a file as raw text, without trailing newlines.

    Parameters
    ----------
    path : Path
        The path to the file.

    Returns
    -------
    str
        The content of the first line.
    """
    with open_text(path) as f:
        first = f.readline()
    return first.rstrip("\r\n")


# ──────────────────────────────────────────────────────────────────────────────
# DELIMITER AND HEADER DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def sniff_delimiter(
    path: Path, sample_bytes: int = 2048, default: str = ","
) -> str:
    """
    Infer the most likely delimiter used in a text file.

    This function reads a sample from the beginning of a file and uses
    `csv.Sniffer` to detect the delimiter.

    Parameters
    ----------
    path : Path
        The path to the file.
    sample_bytes : int, optional
        The number of bytes to read for the sample. Defaults to 2048.
    default : str, optional
        The delimiter to return if detection fails. Defaults to ",".

    Returns
    -------
    str
        The detected or default delimiter.
    """
    with open_text(path) as fh:
        sample = fh.read(sample_bytes)
    try:
        return csv.Sniffer().sniff(sample).delimiter
    except csv.Error:
        # Fallback: most frequent delimiter among common ones
        counts = {d: sample.count(d) for d in COMMON_DELIMITERS}
        return max(counts, key=counts.get) if any(counts.values()) else default  # type: ignore


def looks_like_header(line: str, alpha_thresh: float = 0.2) -> bool:
    """
    Heuristically determine if a line appears to be a header.

    This function checks if a line from a text file is likely to be a
    header row by checking for the presence of alphabetic characters.

    Parameters
    ----------
    line : str
        A single line of text from a file.
    alpha_thresh : float, optional
        The minimum fraction of fields that must contain alphabetic
        characters to be considered a header. Defaults to 0.2 (20%).

    Returns
    -------
    bool
        True if the line is likely a header, False otherwise.
    """
    # Ignore empty/whitespace lines
    if not line.strip():
        return False
    
    # Remove quotes and split on comma (basic delimiter assumption)
    sample = line.replace('"', "")
    tokens = sample.split(",")
    
    # Check first 5 tokens for alphabetic content
    check_tokens = tokens[:5]
    if not check_tokens:
        return False
    
    n_alpha = sum(bool(re.search("[A-Za-z]", t)) for t in check_tokens)
    if n_alpha / len(check_tokens) >= alpha_thresh:
        return True
    
    # Let csv.Sniffer decide on tougher cases
    try:
        return csv.Sniffer().has_header(sample)
    except csv.Error:
        return False


def _fallback_has_header(sample: str, delimiter: str) -> bool:
    """
    Apply a fallback heuristic to guess if a sample of text has a header.

    This is used when `csv.Sniffer.has_header` fails. The heuristic is:
    - If the first line contains alphabetic characters and the second line is
      mostly numeric, assume a header exists.
    - If the first line is mostly numeric, assume no header.

    Parameters
    ----------
    sample : str
        A string sample from the beginning of the file.
    delimiter : str
        The delimiter used to separate fields.

    Returns
    -------
    bool
        True if the sample is likely to have a header, False otherwise.
    """
    lines = [ln for ln in sample.splitlines() if ln.strip() != ""]
    if len(lines) < 2:
        return False
    
    first = lines[0].split(delimiter)
    second = lines[1].split(delimiter)

    def _frac_numeric(fields: list[str]) -> float:
        n = 0
        for x in fields:
            x = x.strip().strip('"').strip("'")
            try:
                float(x)
                n += 1
            except Exception:
                pass
        return n / max(1, len(fields))

    frac1 = _frac_numeric(first)
    frac2 = _frac_numeric(second)
    has_alpha_first = any(re.search(r"[A-Za-z]", c or "") for c in first)
    
    if has_alpha_first and (frac2 > 0.6):
        return True
    if frac1 > 0.6:
        return False
    return False


def detect_delimiter_and_header(
    path: Path, sample_size: int = DEFAULT_SAMPLE_SIZE
) -> Tuple[str, bool]:
    """
    Detect the delimiter and presence of a header in a text file.

    Uses `csv.Sniffer` to determine the delimiter and whether a header
    row exists. Includes fallbacks for both detection steps if the sniffer fails.

    Parameters
    ----------
    path : Path
        The path to the file to inspect.
    sample_size : int, optional
        The number of bytes to read from the beginning of the file to use for
        detection. Defaults to 64,000.

    Returns
    -------
    Tuple[str, bool]
        A tuple containing:
        - The detected delimiter character (e.g., ',').
        - A boolean that is True if a header is detected, False otherwise.
    """
    with open_text(path) as f:
        sample = f.read(sample_size)
    
    # Default delimiter guess: comma
    delimiter = ","
    has_header = False
    sniffer = csv.Sniffer()
    
    try:
        dialect = sniffer.sniff(sample, delimiters="".join(COMMON_DELIMITERS))
        delimiter = dialect.delimiter
    except Exception:
        # Try a simple fallback: guess by most frequent among COMMON_DELIMITERS
        counts = {d: sample.count(d) for d in COMMON_DELIMITERS}
        delimiter = max(counts, key=counts.get) if any(counts.values()) else ","  # type: ignore

    # Header detection with a fallback heuristic
    try:
        has_header = sniffer.has_header(sample)
    except Exception:
        has_header = _fallback_has_header(sample, delimiter)

    # If the very first line is empty/whitespace, treat as no header
    first_line = sample.splitlines()[0] if sample.splitlines() else ""
    if first_line.strip() == "":
        has_header = False
    
    return delimiter, has_header


# ──────────────────────────────────────────────────────────────────────────────
# COLUMN AND HEADER UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def count_columns(path: Path, delimiter: str) -> int:
    """
    Count the number of columns in the first non-empty row of a file.

    Parameters
    ----------
    path : Path
        The path to the file.
    delimiter : str
        The delimiter character to use for splitting rows into columns.

    Returns
    -------
    int
        The number of columns detected in the first non-empty row. Returns 0
        if the file is empty or contains only empty rows.
    """
    with open_text(path) as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if row and any(cell.strip() != "" for cell in row):
                return len(row)
    return 0


def read_colnames(path: Path) -> list[str]:
    """
    Read column names from the first line of a file.

    This function infers the delimiter, reads the first line of the
    file, and returns the column names.

    Parameters
    ----------
    path : Path
        The path to the file.

    Returns
    -------
    list[str]
        A list of column names.
    """
    delimiter = sniff_delimiter(path)
    
    # Read first line, handling BOM
    with path.open("rb") as fh:
        first = fh.readline().lstrip(b"\xef\xbb\xbf").decode()
    
    tokens = first.rstrip("\r\n").split(delimiter)
    return [t.strip('"') for t in tokens]


def header_line_is_valid(header_line: str, delimiter: str, expected_cols: int) -> bool:
    """
    Check if a header line has the expected number of columns.

    This function properly handles quoted fields.

    Parameters
    ----------
    header_line : str
        The raw header line text.
    delimiter : str
        The delimiter character.
    expected_cols : int
        The number of columns the header should have.

    Returns
    -------
    bool
        True if the parsed header has the correct number of columns,
        False otherwise.
    """
    reader = csv.reader([header_line], delimiter=delimiter)
    try:
        fields = next(reader)
        return len(fields) == expected_cols
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# PEER FILE MATCHING AND HEADER BORROWING
# ──────────────────────────────────────────────────────────────────────────────

def name_similarity(a: str, b: str) -> float:
    """
    Calculate the similarity ratio between two strings.

    Uses `difflib.SequenceMatcher` for the comparison.

    Parameters
    ----------
    a : str
        The first string.
    b : str
        The second string.

    Returns
    -------
    float
        A similarity score between 0.0 and 1.0.
    """
    return SequenceMatcher(None, a, b).ratio()


def find_header_donor(
    target: Path,
    delimiter: str,
    expected_cols: int,
    min_name_sim: float = 0.4,
) -> Optional[Tuple[Path, str]]:
    """
    Find a peer file to serve as a header "donor".

    Searches the same directory as the target file for a suitable file to
    borrow a header from. A donor is considered suitable if it:
    - Is a file with a common text extension.
    - Has a detectable header and the same delimiter.
    - Has the same number of columns as the target.
    - Has a filename similarity above `min_name_sim`.

    Among candidates, the one with the closest modification time to the target
    is chosen. Ties are broken by selecting the one with the highest name
    similarity.

    Parameters
    ----------
    target : Path
        The path to the file that needs a header.
    delimiter : str
        The delimiter used in the target file.
    expected_cols : int
        The number of columns in the target file.
    min_name_sim : float, optional
        The minimum name similarity ratio (0.0 to 1.0) required for a file
        to be considered a potential donor. Defaults to 0.4.

    Returns
    -------
    Optional[Tuple[Path, str]]
        A tuple containing the path to the donor file and its raw header line,
        or None if no suitable donor is found.
    """
    folder = target.parent
    t_mtime = target.stat().st_mtime
    t_stem = target.stem
    best: Optional[Tuple[float, float, Path, str]] = None  # (time_diff, -name_sim, path, header_line)

    for p in folder.iterdir():
        if p == target or not p.is_file():
            continue
        try:
            # Only consider text-like files by extension
            if p.suffix.lower() not in {".csv", ".dat", ".txt", ".tsv"}:
                continue

            d_delim, d_has_header = detect_delimiter_and_header(p)
            if d_delim != delimiter:
                # Different delimiter—skip to avoid mismatched header
                continue
            if not d_has_header:
                continue
            
            cols = count_columns(p, d_delim)
            if cols != expected_cols:
                continue
            
            hdr = get_first_line_raw(p)
            if not header_line_is_valid(hdr, d_delim, expected_cols):
                continue
            
            sim = name_similarity(t_stem, p.stem)
            if sim < min_name_sim:
                continue
            
            diff = abs(p.stat().st_mtime - t_mtime)
            key = (diff, -sim, p, hdr)
            if best is None or key < best:
                best = key
        except Exception:
            continue
    
    if best is None:
        return None
    return best[2], best[3]  # type: ignore[return-value]


# ──────────────────────────────────────────────────────────────────────────────
# HEADER APPLICATION AND FILE MODIFICATION
# ──────────────────────────────────────────────────────────────────────────────

def prepend_header_in_place(path: Path, header_line: str) -> None:
    """
    Insert a header line at the top of a file.

    This function reads the entire file, then writes it back with the
    provided header line at the beginning. It attempts to preserve the
    original newline style.

    Parameters
    ----------
    path : Path
        The path to the file to be modified.
    header_line : str
        The header line to prepend to the file.

    Returns
    -------
    None
    """
    # Read original content
    with open_text(path) as f:
        original = f.read()
    
    newline = "\n"
    if "\r\n" in original and "\n" in original:
        # mixed newlines; default to '\n'
        newline = "\n"
    elif "\r\n" in original and "\n" not in original:
        newline = "\r\n"
    
    # Write back with header
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(header_line.rstrip("\r\n") + newline + original.lstrip("\r\n"))


def apply_header(
    header_file: Path,
    target_file: Path,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Apply a header from a reference file to a data file and return a DataFrame.

    This function reads column names from `header_file` and applies them to
    `target_file`, which is assumed to lack a header row. The result is returned
    as a pandas DataFrame. Optionally, the function can overwrite `target_file`
    with the updated version, keeping a backup as `*.bak`.

    Parameters
    ----------
    header_file : Path
        Path to the file containing the correct column headers.
    target_file : Path
        Path to the file that is missing column headers.
    inplace : bool, optional
        If True, the modified DataFrame is written back to `target_file`,
        and a backup of the original file is saved with a `.bak` extension.
        Default is False.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the contents of `target_file` with headers applied
        from `header_file`.

    Notes
    -----
    The delimiter is inferred using a sniffing function to ensure consistent parsing
    between the header and target files.
    """
    delimiter = sniff_delimiter(header_file)
    cols = read_colnames(header_file)
    
    df = pd.read_csv(target_file, header=None, names=cols, delimiter=delimiter)
    
    if inplace:
        backup = target_file.with_suffix(target_file.suffix + ".bak")
        target_file.replace(backup)  # keep a backup
        df.to_csv(target_file, index=False, sep=delimiter)
    
    return df


def patch_file(donor: Path, target: Path) -> pd.DataFrame:
    """
    Apply a header from a donor file to a target file.

    This function reads the header from a `donor` file and applies it
    to a `target` file that is assumed to be missing a header. The
    modified data is returned as a DataFrame and written back to the
    target file.

    Parameters
    ----------
    donor : Path
        The path to the file with the correct header.
    target : Path
        The path to the file that needs a header.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the data from the target file with the
        new header.
    """
    cols = read_colnames(donor)
    delimiter = sniff_delimiter(donor)
    
    df = pd.read_csv(target, header=None, names=cols, delimiter=delimiter)
    df.to_csv(target, index=False, sep=delimiter, quoting=csv.QUOTE_NONE, escapechar="\\")
    
    return df


# ──────────────────────────────────────────────────────────────────────────────
# SINGLE FILE AND DIRECTORY PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def process_file(path: Path, min_sim: float, make_backup: bool) -> None:
    """
    Detect and repair a headerless delimited text file in place.

    The function inspects `path` to determine its delimiter and whether the file
    already contains a header row. If a header is missing, it searches for a
    "donor" file in the same directory with a compatible delimiter and
    column count, and with column-name similarity above `min_sim`. When a donor
    is found, its header is prepended to `path` (optionally creating a ``.bak``
    backup first). Progress is reported via ``print`` messages.

    Parameters
    ----------
    path : pathlib.Path
        Path to the target text file to check and possibly fix.
    min_sim : float
        Minimum similarity threshold (0–1) for column-name matching when
        selecting a donor header. Higher values are stricter.
    make_backup : bool
        If True, write a bytes-for-bytes backup alongside the file at
        ``path.with_suffix(path.suffix + ".bak")`` before modifying the file.

    Returns
    -------
    None
        The file at `path` may be modified in place as a side effect.

    Raises
    ------
    OSError
        If reading or writing the file fails.
    Exception
        Any error originating from helper functions may propagate.
    """
    delim, has_hdr = detect_delimiter_and_header(path)
    if has_hdr:
        return  # nothing to do

    cols = count_columns(path, delim)
    donor = find_header_donor(path, delimiter=delim, expected_cols=cols, min_name_sim=min_sim)
    
    if donor is None:
        print(f"[SKIP] {path.name}: no donor found")
        return

    dpath, header = donor
    if make_backup:
        bkp = path.with_suffix(path.suffix + ".bak")
        bkp.write_bytes(path.read_bytes())
    
    prepend_header_in_place(path, header)
    print(f"[FIXED] {path.stem}  ← header from {dpath.name}")


def scan(root: Path, min_sim: float = 0.5, backup: bool = False) -> None:
    """
    Recursively scan a directory tree and fix headerless text files.

    Walks `root` with ``Path.rglob("*")`` and applies :func:`process_file` to
    every file whose extension is in ``{".dat"}``. Exceptions raised by
    :func:`process_file` are caught and reported, allowing the scan to continue.

    Parameters
    ----------
    root : pathlib.Path
        Directory to search recursively for candidate text files.
    min_sim : float, default=0.5
        Minimum column-name similarity (0–1) when selecting a donor header;
        passed through to :func:`process_file`.
    backup : bool, default=False
        If True, create a ``.bak`` file for each modified file; passed through
        to :func:`process_file` as ``make_backup``.

    Returns
    -------
    None

    Side Effects
    ------------
    - May modify files in place by inserting a header line.
    - May create ``.bak`` files adjacent to modified files when `backup=True`.
    - Prints progress, skip, and error messages to standard output.
    """
    TEXT_EXT = {".dat"}

    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in TEXT_EXT:
            try:
                process_file(p, min_sim=min_sim, make_backup=backup)
            except Exception as exc:
                print(f"[ERROR] {p.name}: {exc}")


def fix_all_in_parent(parent: Path, searchstr: str = "*_AmeriFluxFormat_*.dat") -> dict:
    """
    Recursively scan a parent directory for files with duplicate names and fix missing headers.

    This function searches `parent` for files matching a given pattern. If duplicate
    filenames are found such that one version has a header and another does not,
    the header is copied from the former to the latter. The target files are
    overwritten in-place, and a `.bak` backup is created for each.

    Parameters
    ----------
    parent : Path
        Root directory to scan for matching files. All subdirectories are included recursively.
    searchstr : str, optional
        Glob-style pattern to match filenames (default is "*_AmeriFluxFormat_*.dat").

    Returns
    -------
    dict
        A dictionary mapping filenames to lists of paths where they were found.

    Notes
    -----
    - Files are grouped by basename and inspected line-by-line to determine whether
      they contain a header.
    - If multiple files have headers, only the first one is used as the donor.
    - Files with no header and no matching header source are skipped.
    """
    # Collect every file path, grouped by basename
    paths_by_name: dict[str, list[Path]] = defaultdict(list)
    
    for p in parent.rglob(searchstr):
        if p.is_file():
            paths_by_name[p.name].append(p)

    # Examine each group of duplicates
    for fname, paths in paths_by_name.items():
        if len(paths) < 2:
            continue  # no duplicates → nothing to do

        # Classify each copy
        header_files, noheader_files = [], []
        for p in paths:
            first = p.open("r", encoding="utf-8").readline()
            if looks_like_header(first):
                header_files.append(p)
            else:
                noheader_files.append(p)

        if not header_files or not noheader_files:
            # Either (a) every copy already has a header, or (b) none do
            continue

        # Use the first header-bearing file as the "donor" for all others
        donor = header_files[0]
        for tgt in noheader_files:
            df_fixed = patch_file(donor, tgt)
            print(
                f"[INFO]  Patched  {tgt.relative_to(parent)}   "
                f"({len(df_fixed):,d} rows)"
            )

    print("\n✔ All possible files have been checked.")
    return dict(paths_by_name)


def fix_directory_pairs(dir_with_headers: Path, dir_without_headers: Path) -> None:
    """
    Apply headers from a directory of correctly formatted files to a directory
    of files missing headers.

    This function loops through all files in `dir_without_headers`. For each file
    that lacks a header, it attempts to find a matching file by name in
    `dir_with_headers` and uses it to patch the missing header. The original file
    is overwritten, and a `.bak` backup is created.

    Parameters
    ----------
    dir_with_headers : Path
        Directory containing files with valid headers.
    dir_without_headers : Path
        Directory containing files that may be missing headers.

    Returns
    -------
    None

    Notes
    -----
    This function assumes that files in both directories are named identically,
    and that headers can be determined by inspecting the first line of each file.
    """
    # Index the header-bearing directory for O(1) lookup
    header_index = {p.name: p for p in dir_with_headers.iterdir() if p.is_file()}

    for f in dir_without_headers.iterdir():
        if not f.is_file():
            continue

        # Fast header check: read only the first line
        first_line = f.open("r", encoding="utf-8").readline()
        if looks_like_header(first_line):
            continue  # nothing to do

        if f.name not in header_index:
            print(f"[WARN] No header twin found for {f}")
            continue

        df_fixed = apply_header(header_index[f.name], f, inplace=True)
        print(f"[INFO] Patched header on {f} ({len(df_fixed)} rows)")


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    # File I/O
    "open_text",
    "get_first_line_raw",
    # Detection
    "sniff_delimiter",
    "looks_like_header",
    "detect_delimiter_and_header",
    # Column utilities
    "count_columns",
    "read_colnames",
    "header_line_is_valid",
    # Peer matching
    "name_similarity",
    "find_header_donor",
    # Header application
    "prepend_header_in_place",
    "apply_header",
    "patch_file",
    # Processing
    "process_file",
    "scan",
    "fix_all_in_parent",
    "fix_directory_pairs",
]