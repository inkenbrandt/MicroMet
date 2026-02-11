import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from micromet.report.eddy_plots import (
    create_grouped_boxplot,
    ols_plot,
    comparison_plot,
    plot_wind_rose_from_df,
    plot_linear_regression_with_color,
    plot_flux_vs_ustar
)

class TestEddyPlots(unittest.TestCase):
    def test_create_grouped_boxplot(self):
        df = pd.DataFrame({
            'val': [1, 2, 3, 4],
            'cat': ['A', 'A', 'B', 'B']
        })
        fig = create_grouped_boxplot(df, 'val', 'cat')
        self.assertIsInstance(fig, go.Figure)

    @patch('matplotlib.pyplot.show')
    def test_ols_plot(self, mock_show):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        ols_plot(x, y, 'X', 'Y', 'Title')
        plt.close()

    @patch('matplotlib.pyplot.show')
    def test_comparison_plot(self, mock_show):
        df = pd.DataFrame({
            'V1': [1, 2, 3],
            'V2': [1.1, 1.9, 3.1]
        })
        comparison_plot(df, 'V1', 'V2', 'Title', 'X', 'Y', 'out.png', print_plot=False)
        plt.close()

    @patch('matplotlib.pyplot.show')
    def test_plot_wind_rose_from_df(self, mock_show):
        df = pd.DataFrame({
            'WD': [0, 90, 180, 270],
            'WS': [1, 2, 3, 4]
        })
        plot_wind_rose_from_df(df, 'WD', 'WS', title='Wind Rose')
        plt.close()

    @patch('matplotlib.pyplot.show')
    def test_plot_linear_regression_with_color(self, mock_show):
        df = pd.DataFrame({
            'X': [1, 2, 3],
            'Y': [1.1, 1.9, 3.1],
            'C': [10, 20, 30]
        })
        plot_linear_regression_with_color(df, 'X', 'Y', 'C')
        plt.close()

    @patch('matplotlib.pyplot.show')
    def test_plot_flux_vs_ustar(self, mock_show):
        times = pd.date_range('2024-01-01', periods=100, freq='30min')
        df = pd.DataFrame({
            'USTAR': np.random.rand(100),
            'LE_1_1_1': np.random.randn(100),
            'H_1_1_1': np.random.randn(100),
            'NETRAD_1_1_2': np.random.randn(100) * 100
        }, index=times)
        plot_flux_vs_ustar(df)
        plt.close()

if __name__ == '__main__':
    unittest.main()
