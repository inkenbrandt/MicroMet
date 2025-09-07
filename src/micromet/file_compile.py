#!/usr/bin/env python3
"""
Compile files by substring into a single directory.

Key logic:
- Group by exact filename (case-sensitive match on the filename itself).
- Within each group, deduplicate items that have the *same* (creation_time, size).
- If >1 unique items remain *and* both creation_time and size differ across them,
  copy all, labeled sequentially: name_1.ext, name_2.ext, ...
- Else (effectively duplicates), copy only one.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import shutil
import sys
import time


@dataclass(frozen=True)
class FileInfo:
    """
    A container for file metadata.

    Attributes
    ----------
    path : Path
        The full path to the file.
    size : int
        The size of the file in bytes.
    create_ts : float
        The creation timestamp of the file. This may be platform-dependent.
    mtime_ts : float
        The modification timestamp of the file.
    """

    path: Path
    size: int
    create_ts: float  # "creation" time (platform-dependent, see _get_creation_time)
    mtime_ts: float


def _get_creation_time(p: Path) -> float:
    """
    Get the file creation time in a cross-platform manner.

    This function attempts to get the most accurate creation time
    available on the current operating system.

    Parameters
    ----------
    p : Path
        The path to the file.

    Returns
    -------
    float
        The creation timestamp of the file.
    """
    st = p.stat()
    if hasattr(st, "st_birthtime"):  # macOS, some BSDs
        return st.st_birthtime
    return st.st_ctime  # Windows: creation, Linux: change time


def _gather_files(root: Path, contains: str, case_sensitive: bool) -> List[Path]:
    """
    Gather all files in a directory tree that contain a specific substring.

    Parameters
    ----------
    root : Path
        The root directory to start the search from.
    contains : str
        The substring to search for in filenames.
    case_sensitive : bool
        Whether the search should be case-sensitive.

    Returns
    -------
    List[Path]
        A list of paths to the files that match the criteria.
    """
    files: List[Path] = []
    if not case_sensitive:
        contains_low = contains.lower()

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            hay = fn if case_sensitive else fn.lower()
            needle = contains if case_sensitive else contains_low  # type: ignore
            if needle in hay:
                files.append(Path(dirpath) / fn)
    return files


def _to_fileinfo(paths: List[Path], use_mtime: bool) -> List[FileInfo]:
    """
    Convert a list of file paths to a list of FileInfo objects.

    Parameters
    ----------
    paths : List[Path]
        A list of paths to files.
    use_mtime : bool
        If True, use the modification time as the creation time.

    Returns
    -------
    List[FileInfo]
        A list of FileInfo objects corresponding to the input paths.
    """
    out: List[FileInfo] = []
    for p in paths:
        try:
            st = p.stat()
            fi = FileInfo(
                path=p,
                size=st.st_size,
                create_ts=st.st_mtime if use_mtime else _get_creation_time(p),
                mtime_ts=st.st_mtime,
            )
            out.append(fi)
        except FileNotFoundError:
            # Skip files that disappear between walk and stat
            continue
    return out


def _group_by_filename(infos: List[FileInfo]) -> Dict[str, List[FileInfo]]:
    """
    Group a list of FileInfo objects by their filename.

    Parameters
    ----------
    infos : List[FileInfo]
        A list of FileInfo objects.

    Returns
    -------
    Dict[str, List[FileInfo]]
        A dictionary mapping each filename to a list of FileInfo objects
        that share that name.
    """
    byname: Dict[str, List[FileInfo]] = {}
    for fi in infos:
        byname.setdefault(fi.path.name, []).append(fi)
    return byname


def _unique_by_ctime_size(items: List[FileInfo]) -> List[FileInfo]:
    """
    Filter a list of FileInfo objects to find unique items by creation time and size.

    Parameters
    ----------
    items : List[FileInfo]
        A list of FileInfo objects to be filtered.

    Returns
    -------
    List[FileInfo]
        A list containing only the unique FileInfo objects.
    """
    seen: set[Tuple[int, int]] = set()
    unique: List[FileInfo] = []
    # Round timestamps to integer seconds for dedup; adjust if you need finer resolution
    for fi in items:
        key = (int(fi.create_ts), fi.size)
        if key not in seen:
            seen.add(key)
            unique.append(fi)
    return unique


def _all_differ_in_both_ctime_and_size(items: List[FileInfo]) -> bool:
    """
    Check if all items in a list differ in both creation time and size.

    Returns True if every pair of items in the list has a different
    creation time and a different size.

    Parameters
    ----------
    items : List[FileInfo]
        A list of FileInfo objects to compare.

    Returns
    -------
    bool
        True if all items are unique in both creation time and size,
        False otherwise.
    """
    n = len(items)
    if n <= 1:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            same_ctime = int(items[i].create_ts) == int(items[j].create_ts)
            same_size = items[i].size == items[j].size
            if same_ctime or same_size:
                return False
    return True


def _ensure_outdir(p: Path):
    """
    Ensure that a directory exists, creating it if necessary.

    Parameters
    ----------
    p : Path
        The path to the directory.
    """
    p.mkdir(parents=True, exist_ok=True)


def _format_time(ts: float) -> str:
    """
    Format a timestamp into a string.

    Parameters
    ----------
    ts : float
        The timestamp to format.

    Returns
    -------
    str
        The formatted time string in 'YYYY-MM-DD_HH-MM-SS' format.
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(ts))


def compile_files(
    root: Path,
    outdir: Path,
    contains: str,
    case_sensitive: bool = False,
    dry_run: bool = False,
    use_mtime: bool = False,
    sequential_zero_pad: int = 1,
) -> None:
    """
    Compile files from a source directory to a destination, handling duplicates.

    This function scans a directory tree for files containing a specific
    substring in their names, groups them by filename, and then copies
    them to an output directory. It includes logic to handle duplicate
    files based on their creation time and size.

    Parameters
    ----------
    root : Path
        The root directory to search for files.
    outdir : Path
        The directory where the compiled files will be saved.
    contains : str
        The substring that filenames must contain to be included.
    case_sensitive : bool, optional
        If True, the search for `contains` is case-sensitive.
        Defaults to False.
    dry_run : bool, optional
        If True, the function will only print the actions it would take
        without actually copying any files. Defaults to False.
    use_mtime : bool, optional
        If True, use the file's modification time instead of its creation
        time for comparisons. Defaults to False.
    sequential_zero_pad : int, optional
        The number of digits to use for zero-padding when creating
        sequential filenames for duplicates. Defaults to 1.
    """
    _ensure_outdir(outdir)

    paths = _gather_files(root, contains, case_sensitive)
    infos = _to_fileinfo(paths, use_mtime=use_mtime)
    groups = _group_by_filename(infos)

    copied = 0
    skipped_dup = 0
    made_sequential = 0

    for filename, items in sorted(groups.items()):
        uniques = _unique_by_ctime_size(items)

        # Only one unique (ctime,size): copy just one (the earliest by ctime)
        if len(uniques) == 1:
            src = uniques[0].path
            dst = outdir / filename
            if dst.exists():
                # If same filename already placed (from another pass), skip if same size,
                # else append a suffix to avoid overwrite.
                if dst.stat().st_size == uniques[0].size:
                    skipped_dup += 1
                    continue
                # Different size: avoid overwrite by adding suffix
                stem, ext = Path(filename).stem, Path(filename).suffix
                dst = outdir / f"{stem}_1{ext}"
            if dry_run:
                print(f"[DRY-RUN] COPY {src} -> {dst}")
            else:
                shutil.copy2(src, dst)
            copied += 1

        else:
            # Multiple unique versions for the same filename
            # If they differ in BOTH creation time and size, label sequentially.
            # Otherwise treat as duplicates and copy only one.
            if _all_differ_in_both_ctime_and_size(uniques):
                # Sort by creation time (oldest first)
                uniques_sorted = sorted(uniques, key=lambda fi: fi.create_ts)
                stem, ext = Path(filename).stem, Path(filename).suffix
                for idx, fi in enumerate(uniques_sorted, start=1):
                    suffix = f"_{str(idx).zfill(sequential_zero_pad)}"
                    dst = outdir / f"{stem}{suffix}{ext}"
                    if dst.exists():
                        # Find the next available suffix to avoid accidental overwrite
                        k = idx
                        while dst.exists():
                            k += 1
                            suffix = f"_{str(k).zfill(sequential_zero_pad)}"
                            dst = outdir / f"{stem}{suffix}{ext}"
                    if dry_run:
                        print(f"[DRY-RUN] COPY {fi.path} -> {dst}")
                    else:
                        shutil.copy2(fi.path, dst)
                    made_sequential += 1
            else:
                # Treat as duplicates: pick the earliest by creation time and copy once
                choice = min(uniques, key=lambda fi: fi.create_ts)
                dst = outdir / filename
                if dst.exists():
                    if dst.stat().st_size == choice.size:
                        skipped_dup += 1
                        continue
                    # Different size but not both differing -> keep just one, but avoid overwrite
                    stem, ext = Path(filename).stem, Path(filename).suffix
                    dst = outdir / f"{stem}_1{ext}"
                if dry_run:
                    print(f"[DRY-RUN] COPY {choice.path} -> {dst}")
                else:
                    shutil.copy2(choice.path, dst)
                copied += 1

    print(
        f"Done. Copied: {copied}, Sequentially labeled: {made_sequential}, Skipped duplicates: {skipped_dup}"
    )
