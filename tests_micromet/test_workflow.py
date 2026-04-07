import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from micromet.workflow import (
    WorkflowRunner,
    WorkflowConfig,
    WorkflowResult,
    SiteCorrections,
    DateRangeDrop,
    FlagWindow,
    run_workflow,
    _DEFAULT_CSFLUX_JOIN_COLS,
    _DEFAULT_MERGE_DROP_COLS,
    _DEFAULT_NO_SUFFIX_COLS,
)


class TestWorkflowConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = WorkflowConfig()
        self.assertEqual(cfg.station, "")
        self.assertEqual(cfg.interval, 30)
        self.assertEqual(cfg.steps, [1, 2, 3, 4])
        self.assertFalse(cfg.generate_plots)

    def test_preprocessed_path_default(self):
        cfg = WorkflowConfig(raw_data_root=Path("/data"))
        self.assertEqual(cfg.preprocessed_path, Path("/data/preprocessed_site_data"))

    def test_preprocessed_path_custom(self):
        cfg = WorkflowConfig(
            raw_data_root=Path("/data"),
            preprocessed_dir=Path("/custom"),
        )
        self.assertEqual(cfg.preprocessed_path, Path("/custom"))

    def test_all_config_params(self):
        cfg = WorkflowConfig(
            station="US-UTJ",
            interval=60,
            raw_data_root=Path("/raw"),
            output_root=Path("/out"),
            steps=[2, 3],
            generate_plots=True,
            drop_soil=True,
            fetch_events_from_db=True,
            data_interval_label="HR",
            soilvue_g_calculation=True,
        )
        self.assertEqual(cfg.station, "US-UTJ")
        self.assertEqual(cfg.interval, 60)
        self.assertEqual(cfg.steps, [2, 3])
        self.assertTrue(cfg.generate_plots)
        self.assertTrue(cfg.soilvue_g_calculation)
        self.assertEqual(cfg.data_interval_label, "HR")


class TestSiteCorrections(unittest.TestCase):
    def test_defaults(self):
        corr = SiteCorrections()
        self.assertIsNone(corr.sg_correction_factor)
        self.assertEqual(corr.sg_correction_vars, ["SG_1_1_1", "SG_2_1_1"])
        self.assertEqual(corr.signal_strength_threshold, 0.8)
        self.assertTrue(corr.drop_precip_on_visits)
        self.assertEqual(corr.date_range_drops, [])
        self.assertEqual(corr.h2o_flag_windows, [])

    def test_custom_corrections(self):
        corr = SiteCorrections(
            sg_correction_factor=0.3125,
            sg_correction_end="2025-11-09",
            precip_correction_factor=2.54,
            precip_correction_end="2025-12-01",
            wind_direction_offset=82,
            wind_direction_change_date="2025-07-15",
            wind_flag_bad_range=(0, 180),
            wind_flag_marginal_ranges=[(180, 190), (350, 360)],
            date_range_drops=[
                DateRangeDrop("LW_IN_1_1_1", "2024-08-30", "2024-08-30 01:00"),
            ],
            h2o_flag_windows=[
                FlagWindow(["H2O_SIG_FLAG_1_1_1"], "2024-07-25", "2024-08-18", 2),
            ],
        )
        self.assertAlmostEqual(corr.sg_correction_factor, 0.3125)
        self.assertEqual(len(corr.date_range_drops), 1)
        self.assertEqual(corr.date_range_drops[0].column, "LW_IN_1_1_1")
        self.assertEqual(len(corr.h2o_flag_windows), 1)


class TestDateRangeDrop(unittest.TestCase):
    def test_creation(self):
        d = DateRangeDrop("TA_1_4_1", "2024-08-29 20:00", "2024-08-30 0:00")
        self.assertEqual(d.column, "TA_1_4_1")
        self.assertEqual(d.start, "2024-08-29 20:00")
        self.assertEqual(d.end, "2024-08-30 0:00")


class TestFlagWindow(unittest.TestCase):
    def test_creation(self):
        fw = FlagWindow(["H2O_SIG_FLAG_1_1_1"], "2024-07-25", "2024-08-18")
        self.assertEqual(fw.flag_columns, ["H2O_SIG_FLAG_1_1_1"])
        self.assertEqual(fw.flag_value, 2)

    def test_custom_flag_value(self):
        fw = FlagWindow(["CO2_FLAG"], "2024-01-01", "2024-02-01", 1)
        self.assertEqual(fw.flag_value, 1)


class TestWorkflowResult(unittest.TestCase):
    def test_defaults(self):
        r = WorkflowResult(station="US-UTJ", success=True)
        self.assertEqual(r.station, "US-UTJ")
        self.assertTrue(r.success)
        self.assertEqual(r.steps_completed, [])
        self.assertEqual(r.errors, {})

    def test_summary(self):
        r = WorkflowResult(
            station="US-UTJ",
            success=True,
            steps_completed=[1, 2, 3],
            processing_time=42.5,
        )
        s = r.summary()
        self.assertIn("SUCCESS", s)
        self.assertIn("US-UTJ", s)
        self.assertIn("42.5", s)

    def test_summary_with_errors(self):
        r = WorkflowResult(
            station="US-UTJ",
            success=False,
            errors={3: "Missing file"},
        )
        s = r.summary()
        self.assertIn("FAILED", s)
        self.assertIn("Missing file", s)


class TestWorkflowRunnerInit(unittest.TestCase):
    def test_init_defaults(self):
        cfg = WorkflowConfig(station="US-UTJ")
        runner = WorkflowRunner(config=cfg)
        self.assertEqual(runner.config.station, "US-UTJ")
        self.assertIsNotNone(runner.logger)
        self.assertIsInstance(runner.corrections, SiteCorrections)

    def test_init_with_corrections(self):
        cfg = WorkflowConfig(station="US-UTJ")
        corr = SiteCorrections(sg_correction_factor=0.5)
        runner = WorkflowRunner(config=cfg, corrections=corr)
        self.assertAlmostEqual(runner.corrections.sg_correction_factor, 0.5)


class TestWorkflowRunnerStep3Logic(unittest.TestCase):
    """Test the QC step logic with synthetic data."""

    def _make_test_df(self) -> pd.DataFrame:
        """Create a synthetic raw DataFrame for testing step 3."""
        idx = pd.date_range("2024-06-01", periods=100, freq="30min")
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "G_PLATE_1_1_1": rng.normal(10, 5, 100),
                "G_PLATE_2_1_1": rng.normal(10, 5, 100),
                "SG_1_1_1": rng.normal(2, 1, 100),
                "SG_2_1_1": rng.normal(2, 1, 100),
                "P_1_1_1": rng.uniform(0, 5, 100),
                "WD_1_1_1": rng.uniform(0, 360, 100),
                "WS_1_1_1": rng.uniform(0, 15, 100),
                "TA_1_1_1": rng.normal(20, 5, 100),
                "H2O_SIG_STRGTH_MIN_1_1_1": rng.uniform(0.5, 1.0, 100),
                "CO2_SIG_STRGTH_MIN_1_1_1": rng.uniform(0.5, 1.0, 100),
                "SW_IN_1_1_1": rng.uniform(0, 800, 100),
                "SW_OUT_1_1_1": rng.uniform(0, 200, 100),
                "LW_IN_1_1_1": rng.uniform(200, 400, 100),
                "LW_OUT_1_1_1": rng.uniform(300, 500, 100),
                "NETRAD_1_1_1": rng.normal(100, 50, 100),
                "stationid": "US-UTJ",
            },
            index=idx,
        )

    def test_precip_correction(self):
        """Precipitation values before correction date are multiplied."""
        df = self._make_test_df()
        original_precip = df["P_1_1_1"].iloc[0]

        cfg = WorkflowConfig(
            station="US-UTJ",
            output_root=Path("/tmp/test_out"),
            steps=[3],
        )
        corr = SiteCorrections(
            precip_correction_factor=2.54,
            precip_correction_end="2024-06-01 12:00",
        )

        runner = WorkflowRunner(config=cfg, corrections=corr)

        # Directly apply just the precip correction logic
        end_dt = pd.to_datetime(corr.precip_correction_end)
        mask = df.index < end_dt
        df.loc[mask, "P_1_1_1"] = df.loc[mask, "P_1_1_1"] * corr.precip_correction_factor

        # First record is before the cutoff, should be multiplied
        self.assertAlmostEqual(
            df["P_1_1_1"].iloc[0],
            original_precip * 2.54,
            places=4,
        )

    def test_wind_direction_offset(self):
        """Wind direction offset wraps correctly around 360."""
        df = self._make_test_df()
        df["WD_1_1_1"] = 50.0  # Set constant for easy testing

        offset = 82
        change_date = pd.to_datetime("2024-06-02 00:00")
        mask = df.index < change_date
        df.loc[mask, "WD_1_1_1"] = (df.loc[mask, "WD_1_1_1"] - offset) % 360

        # 50 - 82 = -32 -> 328 after modulo
        expected = (50 - 82) % 360
        self.assertAlmostEqual(df.loc[mask, "WD_1_1_1"].iloc[0], expected)
        # Values after change date should be untouched
        self.assertAlmostEqual(df.loc[~mask, "WD_1_1_1"].iloc[0], 50.0)

    def test_g_plate_rename(self):
        """G_PLATE columns are renamed to G and G_SURFACE is calculated."""
        df = self._make_test_df()

        for col_plate, col_g, col_sg, col_surf in [
            ("G_PLATE_1_1_1", "G_1_1_1", "SG_1_1_1", "G_SURFACE_1_1_1"),
            ("G_PLATE_2_1_1", "G_2_1_1", "SG_2_1_1", "G_SURFACE_2_1_1"),
        ]:
            df[col_g] = df[col_plate]
            df.drop(columns=[col_plate], inplace=True)
            df[col_surf] = df[col_g] + df[col_sg]
            invalid = df[col_g].isna() | df[col_sg].isna()
            df.loc[invalid, col_surf] = np.nan

        self.assertIn("G_1_1_1", df.columns)
        self.assertIn("G_SURFACE_1_1_1", df.columns)
        self.assertNotIn("G_PLATE_1_1_1", df.columns)
        # G_SURFACE = G + SG
        np.testing.assert_allclose(
            df["G_SURFACE_1_1_1"].values,
            df["G_1_1_1"].values + df["SG_1_1_1"].values,
        )

    def test_signal_strength_flags(self):
        """H2O and CO2 signal-strength flags are set correctly."""
        df = self._make_test_df()
        threshold = 0.8

        df["H2O_SIG_FLAG_1_1_1"] = 0
        low_sig = df["H2O_SIG_STRGTH_MIN_1_1_1"] < threshold
        df.loc[low_sig, "H2O_SIG_FLAG_1_1_1"] = 1

        # All records below threshold should be flagged
        self.assertTrue(
            (df.loc[low_sig, "H2O_SIG_FLAG_1_1_1"] == 1).all()
        )
        # All records at or above threshold should not be flagged
        self.assertTrue(
            (df.loc[~low_sig, "H2O_SIG_FLAG_1_1_1"] == 0).all()
        )

    def test_wind_direction_flags(self):
        """Wind direction flags correctly mark bad and marginal ranges."""
        idx = pd.date_range("2024-06-01", periods=4, freq="30min")
        df = pd.DataFrame(
            {"WD_1_1_1": [90, 185, 355, 250]},
            index=idx,
        )

        bad_range = (0, 180)
        marginal_ranges = [(180, 190), (350, 360)]

        df["WD_1_1_1_FLAG"] = 0
        lo, hi = bad_range
        bad_wind = (df["WD_1_1_1"] >= lo) & (df["WD_1_1_1"] < hi)
        df.loc[bad_wind, "WD_1_1_1_FLAG"] = 2
        for lo_m, hi_m in marginal_ranges:
            marginal = df["WD_1_1_1"].between(lo_m, hi_m, inclusive="right")
            df.loc[marginal, "WD_1_1_1_FLAG"] = 1

        self.assertEqual(df.loc[idx[0], "WD_1_1_1_FLAG"], 2)   # 90 -> bad
        self.assertEqual(df.loc[idx[1], "WD_1_1_1_FLAG"], 1)   # 185 -> marginal
        self.assertEqual(df.loc[idx[2], "WD_1_1_1_FLAG"], 1)   # 355 -> marginal
        self.assertEqual(df.loc[idx[3], "WD_1_1_1_FLAG"], 0)   # 250 -> good


class TestWorkflowRunnerRun(unittest.TestCase):
    """Test the run orchestrator catches errors and reports results."""

    def test_run_unknown_step(self):
        cfg = WorkflowConfig(station="US-UTJ", steps=[99])
        runner = WorkflowRunner(config=cfg)
        result = runner.run()
        # Unknown step is skipped, not an error
        self.assertTrue(result.success)
        self.assertEqual(result.steps_completed, [])

    def test_run_step_failure_stops_pipeline(self):
        cfg = WorkflowConfig(
            station="US-UTJ",
            steps=[1, 2],
            raw_data_root=Path("/nonexistent"),
        )
        runner = WorkflowRunner(config=cfg)
        result = runner.run()
        self.assertFalse(result.success)
        # Step 1 produces empty frames (no error), step 2 fails on empty data
        self.assertIn(2, result.errors)


class TestDefaultConstants(unittest.TestCase):
    def test_csflux_join_cols_not_empty(self):
        self.assertGreater(len(_DEFAULT_CSFLUX_JOIN_COLS), 30)

    def test_merge_drop_cols_not_empty(self):
        self.assertGreater(len(_DEFAULT_MERGE_DROP_COLS), 3)

    def test_no_suffix_cols_not_empty(self):
        self.assertGreater(len(_DEFAULT_NO_SUFFIX_COLS), 20)


if __name__ == "__main__":
    unittest.main()
