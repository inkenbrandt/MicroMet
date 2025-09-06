import os
import time
from pathlib import Path
import pytest
from src.micromet.file_compile import compile_files, _gather_files, _to_fileinfo, _group_by_filename, _unique_by_ctime_size, _all_differ_in_both_ctime_and_size, FileInfo

@pytest.fixture
def temp_dir_structure(tmp_path):
    """Create a temporary directory structure for testing."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "subdir1").mkdir()
    (source_dir / "subdir2").mkdir()

    # Create some files
    (source_dir / "file1.txt").write_text("content1")
    (source_dir / "subdir1" / "file2.txt").write_text("content2")
    (source_dir / "subdir2" / "file1.txt").write_text("content1_dup")
    (source_dir / "UPPERCASE.txt").write_text("uppercase")

    # for sequential naming test
    file3_1 = source_dir / "file3.txt"
    file3_2 = source_dir / "subdir1" / "file3.txt"
    file3_1.write_text("content3_1")
    time.sleep(0.1) # ensure different creation times
    file3_2.write_text("content3_2_longer")


    return source_dir

def test_gather_files(temp_dir_structure):
    """Test the _gather_files function."""
    # case_sensitive=True should not match UPPERCASE.txt if we search for 'file'
    files = _gather_files(temp_dir_structure, "file", case_sensitive=True)
    assert len(files) == 5
    filenames = {p.name for p in files}
    assert "file1.txt" in filenames
    assert "file2.txt" in filenames
    assert "file3.txt" in filenames
    assert "UPPERCASE.txt" not in filenames

    # case_sensitive=False should match UPPERCASE.txt if we search for 'uppercase'
    files_case_insensitive = _gather_files(temp_dir_structure, "uppercase", case_sensitive=False)
    assert len(files_case_insensitive) == 1
    assert "UPPERCASE.txt" in {p.name for p in files_case_insensitive}

    # Test with .txt
    files_txt = _gather_files(temp_dir_structure, ".txt", case_sensitive=True)
    assert len(files_txt) == 6

def test_compile_files_basic(temp_dir_structure, tmp_path):
    """Test basic file compilation."""
    outdir = tmp_path / "output"
    compile_files(temp_dir_structure, outdir, ".txt", case_sensitive=False)

    assert outdir.exists()
    assert (outdir / "file1.txt").exists()
    assert (outdir / "file2.txt").exists()
    assert (outdir / "UPPERCASE.txt").exists()

def test_compile_files_dry_run(temp_dir_structure, tmp_path, capsys):
    """Test dry run mode."""
    outdir = tmp_path / "output"
    compile_files(temp_dir_structure, outdir, ".txt", dry_run=True, case_sensitive=False)

    assert not (outdir / "file1.txt").exists()
    captured = capsys.readouterr()
    assert "[DRY-RUN]" in captured.out

def test_all_differ_in_both_ctime_and_size():
    """Test the logic for identifying files that need sequential naming."""
    now = time.time()
    # Same time, different size -> False
    items1 = [
        FileInfo(Path("a"), size=100, create_ts=now, mtime_ts=now),
        FileInfo(Path("b"), size=200, create_ts=now, mtime_ts=now),
    ]
    assert not _all_differ_in_both_ctime_and_size(items1)

    # Different time, same size -> False
    items2 = [
        FileInfo(Path("a"), size=100, create_ts=now, mtime_ts=now),
        FileInfo(Path("b"), size=100, create_ts=now + 1, mtime_ts=now + 1),
    ]
    assert not _all_differ_in_both_ctime_and_size(items2)

    # Different time, different size -> True
    items3 = [
        FileInfo(Path("a"), size=100, create_ts=now, mtime_ts=now),
        FileInfo(Path("b"), size=200, create_ts=now + 1, mtime_ts=now + 1),
    ]
    assert _all_differ_in_both_ctime_and_size(items3)
