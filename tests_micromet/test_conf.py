# tests/conftest.py
import io
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import sqlalchemy
from sqlalchemy import text
import plotly.graph_objects as go
import os
import sys

sys.path.append("../src")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
# tests/test_station_data_pull.py

from micromet.station_data_pull import StationDataManager
from micromet import graphs

from micromet import tools
# ------------------------------------------------------------------ #
#  Generic fixtures
# ------------------------------------------------------------------ #
@pytest.fixture(scope="session")
def sample_df():
    idx = pd.date_range(pd.to_datetime("2025‑01‑01"), periods=48, freq="30min")
    df = pd.DataFrame(
        {
            "SW_IN": 600 + np.sin(np.linspace(0, 2 * np.pi, len(idx))) * 50,
            "LW_IN": 420 + np.cos(np.linspace(0, 2 * np.pi, len(idx))) * 20,
            "SW_OUT": 80,
            "LW_OUT": 50,
            "NETRAD": 300,
            "G": 25,
            "LE": 120,
            "H": 145,
        },
        index=idx,
    )
    return df


@pytest.fixture(scope="session")
def toy_engine(tmp_path_factory):
    """
    Returns an in‑memory SQLite engine pre‑loaded with the minimal
    tables used by StationDataManager.
    """
    engine = sqlalchemy.create_engine("sqlite:///:memory:", future=True)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            """
            CREATE TABLE amfluxeddy (
                stationid TEXT,
                timestamp_end INTEGER
            );
            """
        )
        # insert one record for filtering tests
        conn.execute(
            text(
                "INSERT INTO amfluxeddy (stationid, timestamp_end) VALUES (:sid, :ts)"
            ),
            [{"sid": "UTD", "ts": 202501010000}],
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE uploadstats (
                stationid TEXT,
                talbetype TEXT,
                mindate INTEGER,
                maxdate INTEGER,
                datasize_mb REAL,
                stationdf_len INTEGER,
                uploaddf_len INTEGER,
                stationtime TEXT,
                comptime TEXT
            );
            """
        )
    return engine


@pytest.fixture(scope="session")
def fake_config():
    return {
        "LOGGER": {"login": "u", "pw": "p"},
        # Only the fields actually referenced by the code under test
        "UTD": {"ip": "0.0.0.0", "eddy_port": "80"},
    }


@pytest.fixture
def response_json(monkeypatch):
    """Force requests.get(...).json() to return a fixed clock time."""

    class FakeResponse:
        status_code = 200

        def __init__(self, content=b"TIMESTAMP_START,VAL\n", j=None):
            self.content = content
            self._j = j or {"time": "2025‑01‑01 00:00:00"}

        def json(self):
            return self._j

    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResponse())



def test_get_station_id():
    assert StationDataManager.get_station_id("US-UTD") == "UTD"


def test__get_port(fake_config, toy_engine):
    sdm = StationDataManager(fake_config, toy_engine)
    assert sdm._get_port("UTD", "eddy") == 80


def test_get_times(fake_config, toy_engine, response_json):
    sdm = StationDataManager(fake_config, toy_engine)
    logger_time, comp_time = sdm.get_times("UTD", loggertype="eddy")
    assert logger_time == "2025‑01‑01 00:00:00"
    # comp_time should parse into a datetime without error
    pd.to_datetime(comp_time)


def test_remove_existing_records():
    df = pd.DataFrame({"id": [1, 2, 3, 4]})
    filtered = StationDataManager.remove_existing_records(df, "id", [2, 3])
    assert list(filtered["id"]) == [1, 4]

def test_mean_squared_error_simple():
    a = pd.Series([1, 2, 3])
    b = pd.Series([1, 2, 4])
    assert graphs.mean_squared_error(a, b) == 1 / 3




def test_find_irr_dates_detects_peak():
    idx = pd.date_range(pd.to_datetime("2024-05-01",format='%Y-%m-%d'), periods=100, freq="h")
    swc = pd.Series(20, index=idx)
    swc.iloc[10] = 60  # spike
    df = pd.DataFrame({"SWC_1_1_1": swc})
    dates, _ = tools.find_irr_dates(df, height=30, prom=0.5)
    # should detect exactly that spike
    assert len(dates) == 1
    assert dates[0] == idx[10]


def test_polar_to_cartesian_dataframe():
    df = pd.DataFrame({"WD": [0, 90, 180], "Dist": [1, 1, 1]})
    out = tools.polar_to_cartesian_dataframe(df)
    # north (0°) –> (0,1)
    assert np.isclose(out.loc[0, "X_Dist"], 0, atol=1e-7)
    assert np.isclose(out.loc[0, "Y_Dist"], 1, atol=1e-7)


def test_filter_near_neutral():
    """Test the filter_near_neutral function with various inputs."""
    # Test with default bounds (-0.1 < z/L < 0)
    z_over_L = np.array([-0.2, -0.05, 0.05, 0.2])
    mask = tools.filter_near_neutral(z_over_L)
    assert np.array_equal(mask, np.array([False, True, False, False]))

    # Test with custom bounds (-0.2 < z/L < 0.1)
    mask = tools.filter_near_neutral(z_over_L, lower=-0.2, upper=0.1)
    assert np.array_equal(mask, np.array([True, True, True, False]))

    # Test with list input
    z_over_L_list = [-0.2, -0.05, 0.05, 0.2]
    mask = tools.filter_near_neutral(z_over_L_list)
    assert np.array_equal(mask, np.array([False, True, False, False]))
