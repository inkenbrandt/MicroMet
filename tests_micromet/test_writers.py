import pandas as pd
from pathlib import Path
from micromet.io.writers import (
    write_ameriflux_csv,
    UploadCsvOptions,
    write_analysis_csv,
    write_parquet,
)


def _toy_df():
    idx = pd.date_range("2024-06-01 00:00", periods=3, freq="30T")
    df = pd.DataFrame(
        {
            "TIMESTAMP_START": idx,
            "TIMESTAMP_END": idx + pd.Timedelta(minutes=30),
            "SW_IN": [0.0, 10.0, float("nan")],
            "SW_OUT": [0.0, 1.0, 2.0],
            "LW_IN": [350.0, 351.0, 352.0],
            "LW_OUT": [400.0, 401.0, 402.0],
            "NETRAD": [-50.0, -41.0, -30.0],
            "H": [5.0, float("nan"), 5.0],
            "LE": [10.0, 10.0, 10.0],
            "G": [0.0, 0.0, 0.0],
            "TA": [20.0, 20.0, 20.0],
        }
    )
    return df


def test_write_ameriflux_csv(tmp_path: Path):
    df = _toy_df()
    out = tmp_path / "amf.csv"
    write_ameriflux_csv(df, out, UploadCsvOptions(include_units_row=True))
    text = out.read_text()
    lines = text.strip().splitlines()
    # Header + units + 3 data rows = 5 lines
    assert len(lines) == 5
    # Check timestamp formatting
    assert lines[1].split(",")[0] == "TIMESTAMP_START"
    assert lines[2].split(",")[0] == ""  # units row first col blank
    # NaN -> -9999 applied (H has a NaN at row 2)
    assert "-9999" in lines[-2]  # second data row


def test_write_analysis_csv(tmp_path: Path):
    df = _toy_df()
    out = tmp_path / "analysis.csv"
    write_analysis_csv(df, out)
    text = out.read_text()
    assert "TIMESTAMP_START" in text
    assert "-9999" not in text  # NaN preserved


def test_write_parquet(tmp_path: Path):
    df = _toy_df()
    out = tmp_path / "data.parquet"
    write_parquet(df, out)
    assert out.exists()
    back = pd.read_parquet(out)
    assert set(df.columns) == set(back.columns)
