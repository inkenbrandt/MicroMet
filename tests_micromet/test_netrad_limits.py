import numpy as np
import pandas as pd
import pytest
from datetime import datetime
import pytz

from micromet.qaqc.netrad_limits import (
    solar_declination,
    hour_angle,
    solar_elevation,
    clear_sky_radiation,
    _to_local_standard_time,
    sw_in_pot_noaa,
    _max_diurnal_composite,
    _xcorr_best_lag,
    analyze_timestamp_alignment,
    flag_issues,
    _infer_freq_minutes,
    _fifteen_day_window_id
)

def test_solar_declination():
    """Test solar declination calculation."""
    # Near summer solstice
    assert np.isclose(solar_declination(172), np.radians(23.44), atol=0.1)
    # Near winter solstice
    assert np.isclose(solar_declination(355), np.radians(-23.44), atol=0.1)
    # Near equinox
    assert np.isclose(solar_declination(81), 0, atol=0.1)

def test_hour_angle():
    """Test hour angle calculation."""
    assert np.isclose(hour_angle(12), 0)
    assert np.isclose(hour_angle(6), -np.pi / 2)
    assert np.isclose(hour_angle(18), np.pi / 2)

def test_to_local_standard_time():
    """Test timezone conversion to local standard time."""
    # Naive timestamp
    naive_dt = pd.to_datetime(["2024-07-01 12:00:00"])
    utc_offset = -7
    # Assuming naive is UTC
    localized = _to_local_standard_time(naive_dt, utc_offset, assume_naive_is_local=False)
    assert localized.tz.utcoffset(None).total_seconds() == utc_offset * 3600
    assert localized[0].hour == 5 # 12:00 UTC is 5:00 in UTC-7

    # Assuming naive is already local
    localized_local = _to_local_standard_time(naive_dt, utc_offset, assume_naive_is_local=True)
    assert localized_local.tz.utcoffset(None).total_seconds() == utc_offset * 3600
    assert localized_local[0].hour == 12

    # Already localized timestamp (e.g., US/Mountain which has DST)
    aware_dt = pd.to_datetime(["2024-07-01 12:00:00"]).tz_localize("US/Mountain")
    converted = _to_local_standard_time(aware_dt, utc_offset)
    assert converted.tz.utcoffset(None).total_seconds() == utc_offset * 3600
    # In July, US/Mountain is UTC-6 (MDT), so converting to UTC-7 should shift by -1 hour
    assert converted[0].hour == 11

def test_infer_freq_minutes():
    """Test frequency inference."""
    dt = pd.to_datetime(pd.date_range("2024-01-01", periods=3, freq="30min"))
    assert _infer_freq_minutes(dt) == 30

    with pytest.raises(ValueError):
        _infer_freq_minutes(pd.to_datetime(["2024-01-01"]))

def test_fifteen_day_window_id():
    """Test window ID calculation."""
    assert _fifteen_day_window_id(1) == 1
    assert _fifteen_day_window_id(15) == 1
    assert _fifteen_day_window_id(16) == 2
    assert _fifteen_day_window_id(365) == 25

def test_xcorr_best_lag():
    """Test cross-correlation lag finding."""
    a = np.array([0, 0, 1, 2, 1, 0, 0, 0, 0, 0])
    b = np.array([0, 1, 2, 1, 0, 0, 0, 0, 0, 0]) # b is a shifted version of a
    lag, corr = _xcorr_best_lag(a, b, max_lag=3)
    assert lag == -1
    assert np.isclose(corr, 1.0)

    # Test with NaNs
    b_nan = np.array([0, 1, 2, 1, 0, 0, 0, np.nan, np.nan, np.nan])
    lag_nan, corr_nan = _xcorr_best_lag(a, b_nan, max_lag=3)
    assert lag_nan == -1
    assert np.isclose(corr_nan, 1.0)

@pytest.fixture
def sample_netrad_df():
    """Create a sample DataFrame for testing timestamp alignment."""
    # 20 days of 30-min data
    start_time = "2024-06-01 00:00:00"
    periods = 20 * 48
    dt_index = pd.to_datetime(pd.date_range(start_time, periods=periods, freq="30min"))

    # Create realistic SW_IN data (sunny days)
    # This is a simplified model for test purposes
    hours = dt_index.hour + dt_index.minute / 60
    sw_in = 1000 * np.sin(np.pi * hours / 24)**2 * (np.sin(np.pi * (dt_index.dayofyear - 80) / 365))
    sw_in = sw_in.to_numpy() # Convert to numpy array to make it mutable
    sw_in[sw_in < 0] = 0

    # Introduce a 1-hour lag
    sw_in_lagged = np.roll(sw_in, 2)

    df = pd.DataFrame({
        "TIMESTAMP_START": dt_index.strftime("%Y%m%d%H%M"),
        "TIMESTAMP_END": (dt_index + pd.Timedelta(minutes=30)).strftime("%Y%m%d%H%M"),
        "SW_IN": sw_in_lagged,
        "PPFD_IN": sw_in_lagged * 2.1 # approx conversion
    })
    return df

def test_analyze_timestamp_alignment_and_flags(sample_netrad_df):
    """Test the main analysis function and the flagging logic."""
    summary, composites = analyze_timestamp_alignment(
        sample_netrad_df,
        lat=40.0,
        lon=-111.0,
        std_utc_offset_hours=-7,
        time_from="CENTER"
    )

    assert not summary.empty
    assert "lag_steps_sw" in summary.columns
    # With a 1-hour lag (2 steps for 30-min data), we expect the median lag to be 2
    assert summary["lag_steps_sw"].median() == 2
    assert summary["lag_steps_ppfd"].median() == 2

    # Test flagging
    issues = flag_issues(summary)
    assert "timezone_or_dst" in issues
    assert "start_vs_end" not in issues # lag is 2, not 1
