#!/usr/bin/env python3
"""
add_header_from_peer.py

Detects a missing header row in a delimited text file. If missing, finds a
similarly named file in the same directory with the same number of columns and
closest modified date, copies that header, and inserts it at the top of the
target file.

Usage:
    python add_header_from_peer.py /path/to/file.dat [--dry-run] [--encoding utf-8] [--min-sim 0.4] [--backup]

Notes:
- Delimiter is auto-detected (csv.Sniffer) among common delimiters.
- "Header present" is inferred via csv.Sniffer.has_header with a fallback heuristic.
- We only borrow a header from a peer file that appears to HAVE a header and
  matches the target file's column count.
- Among candidates with the same column count, we pick the closest modified date.
- A basic name "similarity" filter is applied with difflib; adjust with --min-sim.

This script is conservative and will not modify the file unless a viable
header-donor is found. Use --dry-run to preview actions.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import shutil
from pathlib import Path
from difflib import SequenceMatcher
from datetime import datetime
from typing import Optional, Tuple, List


COMMON_DELIMS = [",", "\t", ";", "|", " "]  # space last (least likely)


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
        encodings = ["utf-8-sig", "utf-8", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            return open(path, "r", encoding=enc, newline="")
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise last_err  # type: ignore[misc]


def detect_delimiter_and_header(
    path: Path, sample_size: int = 64_000
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
        dialect = sniffer.sniff(sample, delimiters="".join(COMMON_DELIMS))
        delimiter = dialect.delimiter
    except Exception:
        # Try a simple fallback: guess by most frequent among COMMON_DELIMS
        counts = {d: sample.count(d) for d in COMMON_DELIMS}
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
    best: Optional[Tuple[float, float, Path, str]] = (
        None  # (time_diff, -name_sim, path, header_line)
    )

    for p in folder.iterdir():
        if p == target or not p.is_file():
            continue
        try:
            # Only consider text-like files by extension; you can relax this if needed
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


def process_file(path: Path, min_sim: float, make_backup: bool) -> None:
    """
    Detect and repair a headerless delimited text file in place.

    The function inspects `path` to determine its delimiter and whether the file
    already contains a header row. If a header is missing, it searches for a
    “donor” file in the same directory tree with a compatible delimiter and
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
        Any error originating from helper functions in ``ahp`` may propagate.

    Notes
    -----
    - Uses the following helper functions from ``ahp``:
      ``detect_delimiter_and_header``, ``count_columns``,
      ``find_header_donor``, and ``prepend_header_in_place``.
    - If the file already has a header, the function returns immediately and
      makes no changes.
    - If no suitable donor is found, the function prints a ``[SKIP]`` message
      and returns without modification.

    Examples
    --------
    >>> from pathlib import Path
    >>> process_file(Path("data/sample.txt"), min_sim=0.6, make_backup=True)
    [FIXED] sample  ← header from donor_sample.txt
    """
    delim, has_hdr = detect_delimiter_and_header(path)
    if has_hdr:
        return  # nothing to do

    cols = count_columns(path, delim)
    donor = find_header_donor(
        path, delimiter=delim, expected_cols=cols, min_name_sim=min_sim
    )
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
    every file whose extension is in ``TEXT_EXT``. Exceptions raised by
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

    Notes
    -----
    - Candidate files are filtered by ``p.suffix.lower() in TEXT_EXT``.
    - Errors from individual files are printed as ``[ERROR]`` lines and do not
      halt the overall scan.

    See Also
    --------
    process_file : Repair a single file in place.

    Examples
    --------
    >>> from pathlib import Path
    >>> scan(Path("data/"), min_sim=0.65, backup=True)
    [FIXED] station_log  ← header from station_log_2024.txt
    [SKIP] notes.md: no donor found
    """
    TEXT_EXT = {".dat"}

    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in TEXT_EXT:
            try:
                process_file(p, min_sim=min_sim, make_backup=backup)
            except Exception as exc:
                print(f"[ERROR] {p.name}: {exc}")
