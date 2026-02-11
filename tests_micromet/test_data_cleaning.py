import unittest
import pandas as pd
import numpy as np
from micromet.qaqc.data_cleaning import (
    set_range_to_nan,
    find_optimal_shift,
    apply_lag_shift,
    mask_wind_direction,
    mask_by_rolling_window_combined,
    despike_data_nan_aware
)

class TestDataCleaning(unittest.TestCase):
    def test_set_range_to_nan(self):
        times = pd.date_range('2024-01-01', periods=10, freq='D')
        df = pd.DataFrame({'VAR': range(10)}, index=times)
        df_new = set_range_to_nan(df, 'VAR', '2024-01-03', '2024-01-05')
        self.assertTrue(np.isnan(df_new.loc['2024-01-03', 'VAR']))
        self.assertTrue(np.isnan(df_new.loc['2024-01-04', 'VAR']))
        self.assertTrue(np.isnan(df_new.loc['2024-01-05', 'VAR']))
        self.assertFalse(np.isnan(df_new.loc['2024-01-02', 'VAR']))

    def test_find_optimal_shift(self):
        times = pd.date_range('2024-01-01', periods=1000, freq='h')
        # Reference series: a sine wave
        t = np.linspace(0, 10*np.pi, 1000)
        s1 = np.sin(t)
        # Series 2: shifted by 50 units
        s2 = np.roll(s1, 50)

        df1 = pd.DataFrame({'val': s1}, index=times)
        df2 = pd.DataFrame({'val': s2}, index=times)

        best_lag, corr = find_optimal_shift(df1, df2, 'val', 'val', min_lag_units=10, max_lag_units=100)
        self.assertEqual(abs(best_lag), 50)

    def test_apply_lag_shift(self):
        times = pd.date_range('2024-01-01', periods=10, freq='D')
        df = pd.DataFrame({'val': range(10)}, index=times)
        df_shifted = apply_lag_shift(df, 1, 'D')
        self.assertEqual(df_shifted.index[0], times[0] - pd.Timedelta(days=1))

    def test_mask_wind_direction(self):
        df = pd.DataFrame({'WD': [0, 90, 180, 270, 350]})
        mask = mask_wind_direction(df, 'WD', 170, 190)
        self.assertTrue(mask[2]) # 180 is bad
        self.assertFalse(mask[0])

        # Wrap around
        mask_wrap = mask_wind_direction(df, 'WD', 350, 10)
        self.assertTrue(mask_wrap[0]) # 0 is bad
        self.assertTrue(mask_wrap[4]) # 350 is bad
        self.assertFalse(mask_wrap[1])

    def test_mask_by_rolling_window_combined(self):
        df = pd.DataFrame({'SIG': [1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0]})
        mask = mask_by_rolling_window_combined(df, 'SIG', rolling_window=3, threshold_value=0.8)
        self.assertFalse(mask[2])
        self.assertFalse(mask[3])
        self.assertFalse(mask[4])
        self.assertTrue(mask[0])
        self.assertTrue(mask[1])
        self.assertTrue(mask[5])
        self.assertTrue(mask[6])

    def test_despike_data_nan_aware(self):
        data = np.array([10.0, 11.0, 100.0, 12.0, 11.0])
        # Use a very small threshold factor to ensure detection
        clean, mask = despike_data_nan_aware(data, filter_size=3, threshold_factor=0.1)
        self.assertTrue(mask[2])
        self.assertAlmostEqual(clean[2], 12.0) # Median of [11, 100, 12] is 12.0

if __name__ == '__main__':
    unittest.main()
