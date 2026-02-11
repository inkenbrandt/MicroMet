import unittest
import pandas as pd
import numpy as np
from micromet.report.tools import (
    find_gaps,
    detect_extreme_variations,
    clean_extreme_variations,
    aggregate_to_daily_centroid,
    compute_Cw
)

class TestTools(unittest.TestCase):
    def test_find_gaps(self):
        times = pd.date_range('2024-01-01', periods=10, freq='30min')
        df = pd.DataFrame({'VAR': [1, 2, np.nan, np.nan, 5, 6, -9999, 8, 9, 10]}, index=times)

        # With min_gap_periods=0, it should find both gaps
        gaps = find_gaps(df, 'VAR', missing_value=-9999, min_gap_periods=0)
        self.assertEqual(len(gaps), 2)
        self.assertEqual(gaps.iloc[0]["missing_records"], 2)
        self.assertEqual(gaps.iloc[1]["missing_records"], 1)

    def test_detect_extreme_variations(self):
        times = pd.date_range('2024-01-01', periods=100, freq='h')
        data = np.random.normal(0, 1, 100)
        data[50] = 100.0 # Much more extreme variation
        df = pd.DataFrame({'VAR': data}, index=times)

        results = detect_extreme_variations(df, 'VAR', frequency='D', variation_threshold=3.0)
        self.assertTrue(results['extreme_points']['VAR_extreme'].iloc[50])

    def test_clean_extreme_variations(self):
        times = pd.date_range('2024-01-01', periods=100, freq='h')
        data = np.random.normal(0, 1, 100)
        data[50] = 100.0 # Much more extreme variation
        df = pd.DataFrame({'VAR': data}, index=times)

        results = clean_extreme_variations(df, 'VAR', replacement_method='nan', variation_threshold=3.0)
        self.assertTrue(np.isnan(results['cleaned_data'].iloc[50]['VAR']))

    def test_aggregate_to_daily_centroid(self):
        df = pd.DataFrame({
            'Timestamp': pd.to_datetime(['2024-01-01 00:00', '2024-01-01 01:00']),
            'X': [1.0, 3.0],
            'Y': [2.0, 4.0],
            'ET': [1.0, 1.0]
        })
        centroid = aggregate_to_daily_centroid(df, weighted=True)
        self.assertEqual(centroid.iloc[0]['X'], 2.0)
        self.assertEqual(centroid.iloc[0]['Y'], 3.0)

    def test_compute_Cw(self):
        cw = compute_Cw(1.0, 1.0, target=1.25)
        self.assertAlmostEqual(cw, (1.25/1.0)**2)

        cw_no_corr = compute_Cw(2.0, 1.0, target=1.25)
        self.assertEqual(cw_no_corr, 1.0)

if __name__ == '__main__':
    unittest.main()
