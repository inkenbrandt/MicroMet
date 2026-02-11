import unittest
import pandas as pd
import numpy as np
from micromet.format.transformers.timestamp_update import (
    resample_alternating_frequency_with_other,
    resample_single_frequency_switch
)

class TestTimestampUpdate(unittest.TestCase):
    def test_resample_single_frequency_switch_30min(self):
        # Create 30 min data
        times = pd.date_range('2024-01-01', periods=10, freq='30min')
        df = pd.DataFrame({'val': range(10)}, index=times)

        result = resample_single_frequency_switch(df)
        self.assertEqual(result['timestep'].iloc[0], 30)
        self.assertEqual(len(result), 10)

    def test_resample_single_frequency_switch_60min(self):
        # Create 60 min data
        times = pd.date_range('2024-01-01', periods=10, freq='60min')
        df = pd.DataFrame({'val': range(10)}, index=times)

        result = resample_single_frequency_switch(df)
        self.assertEqual(result['timestep'].iloc[0], 60)
        self.assertEqual(len(result), 10)

    def test_resample_single_frequency_switch_mixed(self):
        # Create mixed data: 20 records of 30min, then 5 records of 60min
        times30 = pd.date_range('2024-01-01', periods=20, freq='30min')
        times60 = pd.date_range(times30[-1] + pd.Timedelta(minutes=60), periods=5, freq='60min')
        times = times30.append(times60)
        df = pd.DataFrame({'val': range(25)}, index=times)

        result = resample_single_frequency_switch(df)
        # It should detect the switch
        # Check first record (should be 30)
        self.assertEqual(result['timestep'].iloc[0], 30)
        # Check last record (should be 60)
        self.assertEqual(result['timestep'].iloc[-1], 60)

    def test_resample_alternating_frequency_with_other(self):
        # Create alternating data
        times30 = pd.date_range('2024-01-01', periods=48, freq='30min')
        times60 = pd.date_range(times30[-1] + pd.Timedelta(minutes=60), periods=24, freq='60min')
        times = times30.append(times60)
        df = pd.DataFrame({'val': range(72)}, index=times)

        result = resample_alternating_frequency_with_other(df, min_records_threshold=10)
        self.assertIn('timestep', result.columns)
        self.assertEqual(result['timestep'].iloc[0], 30)
        self.assertEqual(result['timestep'].iloc[-1], 60)

if __name__ == '__main__':
    unittest.main()
