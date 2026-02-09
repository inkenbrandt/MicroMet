import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from micromet.reader import AmerifluxDataProcessor
from micromet.format.reformatter import Reformatter

@pytest.fixture
def sample_toa5_file(tmp_path):
    content = """"TOA5","CR6","CR6","1056","CR6.Std.09.02","CPU:MicroMet.CR6","50525","Flux_AmeriFluxFormat"
"TIMESTAMP","RECORD","STAT_1","STAT_2"
"TS","RN","",""
"","","Smp","Smp"
"2024-06-19 12:00:00",1,10.1,20.2
"2024-06-19 12:30:00",2,11.1,21.2
"2024-06-19 13:00:00",3,-9999,22.2
"""
    file_path = tmp_path / "sample_toa5.dat"
    file_path.write_text(content)
    return file_path

@pytest.fixture
def sample_ameriflux_file(tmp_path):
    content = """"TIMESTAMP_START","TIMESTAMP_END","VAR_1","VAR_2"
"202401010000","202401010030",1.1,2.2
"202401010030","202401010100",1.2,2.3
"202401010100","202401010130",-9999,2.4
"""
    file_path = tmp_path / "sample_ameriflux.dat"
    file_path.write_text(content)
    return file_path

@pytest.fixture
def malformed_file(tmp_path):
    content = "this is not a valid file format"
    file_path = tmp_path / "malformed.dat"
    file_path.write_text(content)
    return file_path

@pytest.fixture
def station_data_factory(tmp_path):
    def _create_station_data(station_name, file_contents):
        station_dir = tmp_path / station_name
        station_dir.mkdir()
        for i, content in enumerate(file_contents):
            file_path = station_dir / f"21314_Flux_AmeriFluxFormat_{i}.dat"
            file_path.write_text(content)
    return _create_station_data

@pytest.fixture
def sample_reformatter_df():
    data = {
        'TIMESTAMP_START': ['202401010000', '202401010030', '202401010100'],
        'TIMESTAMP_END': ['202401010030', '202401010100', '202401010130'],
        'Ta': [25.5, 26.0, 24.9],
        'Tau': [0.1, 0.0, 0.2],
        'u_star': [0.5, 0.6, 0.7],
        'SWC_1_1_1': [0.5, 0.6, 0.7],
        'some_other_var': [1, 2, 3]
    }
    return pd.DataFrame(data)

def test_read_toa5_file(sample_toa5_file):
    processor = AmerifluxDataProcessor()
    df = processor.to_dataframe(sample_toa5_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 4)
    assert list(df.columns) == ["TIMESTAMP", "RECORD", "STAT_1", "STAT_2"]
    assert pd.isna(df.iloc[2, 2])

def test_read_ameriflux_file(sample_ameriflux_file):
    processor = AmerifluxDataProcessor()
    df = processor.to_dataframe(sample_ameriflux_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 4)
    assert list(df.columns) == ["TIMESTAMP_START", "TIMESTAMP_END", "VAR_1", "VAR_2"]
    assert pd.isna(df.iloc[2, 2])

def test_read_malformed_file(malformed_file):
    processor = AmerifluxDataProcessor()
    with pytest.raises(RuntimeError):
        processor.to_dataframe(malformed_file)

def test_raw_file_compile(station_data_factory, tmp_path):
    station_name = "US-UTD"
    toa5_content = """"TOA5","CR6","CR6","1056","CR6.Std.09.02","CPU:MicroMet.CR6","50525","Flux_AmeriFluxFormat"
"TIMESTAMP","RECORD","STAT_1","STAT_2"
"TS","RN","",""
"","","Smp","Smp"
"2024-06-19 12:00:00",1,10.1,20.2
"""
    station_data_factory(station_name, [toa5_content, toa5_content])

    processor = AmerifluxDataProcessor()
    compiled_df = processor.raw_file_compile(tmp_path, station_name)

    assert isinstance(compiled_df, pd.DataFrame)
    assert compiled_df.shape == (2, 6)
    assert "FILE_NO" in compiled_df.columns
    assert "DATALOGGER_NO" in compiled_df.columns

def test_reformatter_prepare(sample_reformatter_df):
    reformatter = Reformatter()

    # In the default variable_limits, Ta should be between -50 and 50
    # Let's add a value outside this range to the input
    sample_reformatter_df.loc[0, 'Ta'] = 100

    df, report = reformatter.prepare(sample_reformatter_df, data_type='eddy')

    assert isinstance(df, pd.DataFrame)
    assert 'DATETIME_START' not in df.columns
    assert 'TIMESTAMP_START' in df.columns
    assert 'TIMESTAMP_END' in df.columns

    # Test column renaming and case conversion
    assert 'TA' in df.columns
    assert 'TAU' in df.columns

    # Test tau_fixer (value should be -9999 after fillna)
    assert df.loc[df['TIMESTAMP_START'] == 202401010030, 'TAU'].iloc[0] == -9999

    # Test fix_swc_percent
    assert df['SWC_1_1_1'].max() > 1.5

    # Test physical limits (value should be -9999 after fillna)
    assert df.loc[df['TIMESTAMP_START'] == 202401010000, 'TA'].iloc[0] == -9999
