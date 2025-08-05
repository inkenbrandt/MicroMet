#!/usr/bin/env python3
"""
Compile files by substring into a single directory.

Usage:
  python compile_files.py --root /path/to/root --out /path/to/eddy --contains "21020_Flux_AmeriFluxFormat_"

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
    path: Path
    size: int
    create_ts: float  # "creation" time (platform-dependent, see _get_creation_time)
    mtime_ts: float


def _get_creation_time(p: Path) -> float:
    """
    Best-effort file creation time:
      - Windows: st_ctime is creation time.
      - macOS: st_birthtime if available.
      - Linux: falls back to st_ctime (metadata change time).
    """
    st = p.stat()
    if hasattr(st, "st_birthtime"):  # macOS, some BSDs
        return st.st_birthtime
    return st.st_ctime  # Windows: creation, Linux: change time


def _gather_files(root: Path, contains: str, case_sensitive: bool) -> List[Path]:
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
    byname: Dict[str, List[FileInfo]] = {}
    for fi in infos:
        byname.setdefault(fi.path.name, []).append(fi)
    return byname


def _unique_by_ctime_size(items: List[FileInfo]) -> List[FileInfo]:
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
    Returns True if, across the set, every pair differs in BOTH creation time and size.
    If any pair shares the same creation time or the same size, returns False.
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
    p.mkdir(parents=True, exist_ok=True)


def _format_time(ts: float) -> str:
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


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Copy & compile files by substring, handling same-name duplicates."
    )
    p.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Root directory to search (recursively).",
    )
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory to place compiled files.",
    )
    p.add_argument(
        "--contains", required=True, type=str, help="Substring to match in filenames."
    )
    p.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Make substring matching case-sensitive.",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print actions without copying."
    )
    p.add_argument(
        "--use-mtime",
        action="store_true",
        help="Use modification time instead of creation time.",
    )
    p.add_argument(
        "--pad",
        type=int,
        default=1,
        help="Zero padding width for sequential suffix (default: 1).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    compile_files(
        root=args.root,
        outdir=args.out,
        contains=args.contains,
        case_sensitive=args.case_sensitive,
        dry_run=args.dry_run,
        use_mtime=args.use_mtime,
        sequential_zero_pad=args.pad,
    )
