import unittest
import pandas as pd
import numpy as np
from micromet.report.gap_summary import summarize_gaps, compare_gap_summaries

class TestGapSummary(unittest.TestCase):
    def setUp(self):
        # Create a MultiIndex DataFrame
        times = pd.date_range('2024-01-01', periods=20, freq='30min')
        stns = ['STN1']
        idx = pd.MultiIndex.from_product([stns, times], names=['STATIONID', 'DATETIME_END'])
        self.df = pd.DataFrame({'VAR1': [1.0]*20}, index=idx)
        # Introduce some gaps
        self.df.iloc[2, 0] = np.nan # 3rd record is NaN

    def test_summarize_gaps(self):
        gaps = summarize_gaps(self.df, expected_freq='30min')
        self.assertIsInstance(gaps, pd.DataFrame)
        self.assertFalse(gaps.empty)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps.iloc[0]['N_STEPS_MISSING'], 1)

    def test_compare_gap_summaries(self):
        # Dataset A has a gap at index 2
        gaps_a = summarize_gaps(self.df, expected_freq='30min')

        # Dataset B has a gap at index 5
        df_b = self.df.copy()
        df_b.iloc[2, 0] = 1.0 # Fill gap at 2
        df_b.iloc[5, 0] = np.nan # New gap at 5
        gaps_b = summarize_gaps(df_b, expected_freq='30min')

        comparison = compare_gap_summaries(gaps_a, gaps_b, expected_freq='30min')
        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertFalse(comparison.empty)
        # Check that B can fill A's gap at 2
        fill_b_to_a = comparison[comparison['TARGET_DATASET'] == 'A']
        self.assertFalse(fill_b_to_a.empty)

if __name__ == '__main__':
    unittest.main()
