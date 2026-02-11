import unittest
import pandas as pd
import numpy as np
from micromet.report.validate import (
    validate_flags,
    validate_timestamp_consistency,
    find_zero_chunks,
    compare_names_to_ameriflux,
    data_diff_check
)

class TestValidate(unittest.TestCase):
    def test_validate_flags(self):
        df = pd.DataFrame({
            'FC_SSITC_TEST': [0, 1, 2, 3, np.nan],
            'LE_SSITC_TEST': [0, 1, 0, 1, 0]
        })
        invalid = validate_flags(df)
        self.assertIn('FC_SSITC_TEST', invalid)
        self.assertIn(3, invalid['FC_SSITC_TEST'])
        self.assertIn('NaN', invalid['FC_SSITC_TEST'])
        self.assertNotIn('LE_SSITC_TEST', invalid)

    def test_validate_timestamp_consistency(self):
        df = pd.DataFrame({
            'DATETIME_END': pd.to_datetime(['2024-01-01 00:30', '2024-01-01 01:00']),
            'TIMESTAMP_END': [202401010030, 202401010115] # 2nd is inconsistent
        })
        mismatches = validate_timestamp_consistency(df)
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches.iloc[0]['TIMESTAMP_END'], 202401010115)

    def test_find_zero_chunks(self):
        times = pd.date_range('2024-01-01', periods=100, freq='D')
        data = np.random.randn(100)
        data[10:20] = 0 # 10 days of zeros
        df = pd.DataFrame({'precip': data}, index=times)

        chunks = find_zero_chunks(df, 'precip', days_threshold=5)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks.iloc[0]['Duration (Days)'], 10)

    def test_compare_names_to_ameriflux(self):
        df = pd.DataFrame(columns=['FC_1_1_1', 'LE_1', 'H_2_1_1', 'UNKNOWN'])
        amflux = pd.Series(['FC', 'LE', 'H'])
        results = compare_names_to_ameriflux(df, amflux)
        self.assertTrue(results.loc[results['all_columns'] == 'FC_1_1_1', 'is_in_amflux'].iloc[0])
        self.assertFalse(results.loc[results['all_columns'] == 'UNKNOWN', 'is_in_amflux'].iloc[0])

    def test_data_diff_check(self):
        times = pd.date_range('2024-01-01', periods=10, freq='30min')
        df1 = pd.DataFrame({'VAR': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, index=times)
        df2 = pd.DataFrame({'VAR': [1, 2, 3, 4, 10, 6, 7, 8, 9, 10]}, index=times) # one diff

        diff = data_diff_check(df1, df2)
        self.assertEqual(diff.loc['VAR', 'percent_different'], 10.0)

if __name__ == '__main__':
    unittest.main()
