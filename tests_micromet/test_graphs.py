import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from micromet.report.graphs import (
    energy_sankey,
    mean_squared_error,
    mean_diff_plot,
    scatterplot_instrument_comparison,
    bland_alt_plot
)

class TestGraphs(unittest.TestCase):
    def test_energy_sankey(self):
        idx = pd.date_range("2024-06-19 12:00", periods=1, freq="30min")
        df = pd.DataFrame({
            "SW_IN": [600],
            "LW_IN": [400],
            "SW_OUT": [100],
            "LW_OUT": [350],
            "NETRAD": [550],
            "G": [50],
            "LE": [200],
            "H": [150]
        }, index=idx)

        fig = energy_sankey(df, date_text="2024-06-19 12:00")
        self.assertIsInstance(fig, go.Figure)

    def test_mean_squared_error(self):
        s1 = pd.Series([1, 2, 3])
        s2 = pd.Series([1, 2, 4])
        self.assertEqual(mean_squared_error(s1, s2), 1/3)

        with self.assertRaises(ValueError):
            mean_squared_error(pd.Series([1]), pd.Series([1, 2]))

    def test_mean_diff_plot(self):
        m1 = np.array([1, 2, 3, 4, 5])
        m2 = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        fig = mean_diff_plot(m1, m2)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_scatterplot_instrument_comparison(self):
        times = pd.date_range('2024-01-01', periods=24, freq='h')
        df = pd.DataFrame({
            'INST1': np.random.randn(24),
            'INST2': np.random.randn(24)
        }, index=times)
        compare_dict = {
            'INST1': ['Inst 1', 'Var', 'unit'],
            'INST2': ['Inst 2', 'Var', 'unit']
        }
        # Mock plt.show to avoid blocking
        with patch('matplotlib.pyplot.show'):
            slope, intercept, r_squared, p_value, std_err, fig, ax = scatterplot_instrument_comparison(
                df, compare_dict, 'StationA'
            )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_bland_alt_plot(self):
        times = pd.date_range('2024-01-01', periods=24, freq='h')
        df = pd.DataFrame({
            'INST1': np.random.randn(24),
            'INST2': np.random.randn(24)
        }, index=times)
        compare_dict = {
            'INST1': ['Inst 1', 'Var', 'unit'],
            'INST2': ['Inst 2', 'Var', 'unit']
        }
        with patch('matplotlib.pyplot.show'):
            fig, ax = bland_alt_plot(df, compare_dict, 'StationA')
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

if __name__ == '__main__':
    unittest.main()
