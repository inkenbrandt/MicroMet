"""
Automated workflow for micrometeorological data processing.

This module automates the numbered notebook workflow (notebooks 1-4b)
into a single programmable pipeline. Each step corresponds to a notebook:

    Step 1  (notebook 1) - Compile raw files and preprocess
    Step 2  (notebook 2) - Merge data sources and create raw dataset
    Step 3  (notebook 3) - Apply corrections, QC, and flagging
    Step 4  (notebook 4) - Export AmeriFlux-formatted file
    Plots   (notebooks 3b/4b) - Generate review plots

Site-specific corrections (calibration fixes, date-range drops, wind offsets,
flag windows) are specified declaratively via :class:`SiteCorrections` so each
station can be processed without editing code.

Examples
--------
Minimal usage::

    from micromet.workflow import WorkflowRunner, WorkflowConfig

    runner = WorkflowRunner(
        config=WorkflowConfig(
            station="US-UTJ",
            interval=30,
            raw_data_root=Path("M:/Shared drives/UGS_Flux/Data_Downloads/compiled"),
            output_root=Path("M:/Shared drives/UGS_Flux/Data_Processing/final_database_tables"),
        ),
    )
    result = runner.run()

With site corrections::

    from micromet.workflow import SiteCorrections, CorrectionEntry, FlagWindow

    corrections = SiteCorrections(
        sg_correction_factor=0.05 / 0.16,
        sg_correction_end="2025-11-09 16:30",
        precip_correction_factor=2.54,
        precip_correction_end="2025-12-01 14:49:00",
        wind_direction_offset=82,
        wind_direction_change_date="2025-07-15 12:30",
    )
    runner = WorkflowRunner(
        config=WorkflowConfig(station="US-UTJ", ...),
        corrections=corrections,
    )
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from micromet.format import file_compile, merge
from micromet.format.reformatter import Reformatter
from micromet.format.transformers import (
    columns,
    interval_updates,
    timestamps,
)
from micromet.qaqc import data_cleaning
from micromet.report import eddy_plots, fix_g_values, validate
from micromet.station_info import site_folders
from micromet.utils import logger_check

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DateRangeDrop:
    """A date range within which a column's values should be set to NaN."""

    column: str
    start: str
    end: str


@dataclass
class FlagWindow:
    """A time window for applying a quality flag value to one or more columns."""

    flag_columns: List[str]
    start: str
    end: str
    flag_value: int = 2


@dataclass
class SiteCorrections:
    """
    Declarative specification of site-specific corrections applied during QC.

    All fields are optional; only the corrections relevant to a given station
    need to be populated.

    Parameters
    ----------
    sg_correction_factor : float or None
        Multiplicative factor for soil-heat-flux storage (SG) sensors.
    sg_correction_vars : list of str
        Columns to which ``sg_correction_factor`` applies.
    sg_correction_end : str or None
        Datetime string; correction is applied to data **before** this date.
    precip_correction_factor : float or None
        Multiplicative factor for precipitation before a program fix date.
    precip_correction_end : str or None
        Datetime string; precip correction is applied **before** this date.
    precip_bad_before : str or None
        Drop all precip data before this date (e.g. broken bucket).
    wind_direction_offset : float or None
        Degrees to subtract from WD_1_1_1 before the change date.
    wind_direction_change_date : str or None
        Datetime string when the IRGASON orientation changed.
    date_range_drops : list of DateRangeDrop
        Specific column/date-range pairs to null out (spikes, sensor issues).
    h2o_flag_windows : list of FlagWindow
        Windows to flag H2O signal-strength issues.
    co2_flag_windows : list of FlagWindow
        Windows to flag CO2 signal-strength issues.
    wind_flag_bad_range : tuple of float or None
        (start_deg, end_deg) range of wind directions flagged as 2 (bad).
    wind_flag_marginal_ranges : list of tuple
        List of (start_deg, end_deg) ranges flagged as 1 (marginal).
    signal_strength_threshold : float
        Threshold below which signal-strength data is flagged.
    drop_precip_on_visits : bool
        Whether to zero-out precipitation on station-visit days.
    csflux_join_cols : list of str or None
        Subset of CSFlux columns to merge into the final eddy dataset.
        If None, a default set is used.
    columns_to_drop_from_merge : list of str or None
        Columns to drop after the eddy/met merge (e.g. RECORD, G_1_1_A).
    soilvue_bad_ec_threshold : float or None
        Minimum EC_3_7_1 value; rows below are dropped for SoilVue columns.
    extra_drops : list of DateRangeDrop
        Additional ad-hoc date/column drops.
    """

    sg_correction_factor: Optional[float] = None
    sg_correction_vars: List[str] = field(
        default_factory=lambda: ["SG_1_1_1", "SG_2_1_1"]
    )
    sg_correction_end: Optional[str] = None

    precip_correction_factor: Optional[float] = None
    precip_correction_end: Optional[str] = None
    precip_bad_before: Optional[str] = None

    wind_direction_offset: Optional[float] = None
    wind_direction_change_date: Optional[str] = None

    date_range_drops: List[DateRangeDrop] = field(default_factory=list)
    h2o_flag_windows: List[FlagWindow] = field(default_factory=list)
    co2_flag_windows: List[FlagWindow] = field(default_factory=list)

    wind_flag_bad_range: Optional[Tuple[float, float]] = None
    wind_flag_marginal_ranges: List[Tuple[float, float]] = field(
        default_factory=list
    )

    signal_strength_threshold: float = 0.8
    drop_precip_on_visits: bool = True

    csflux_join_cols: Optional[List[str]] = None
    columns_to_drop_from_merge: Optional[List[str]] = None
    soilvue_bad_ec_threshold: Optional[float] = None
    extra_drops: List[DateRangeDrop] = field(default_factory=list)


@dataclass
class WorkflowConfig:
    """
    Top-level configuration for the automated workflow.

    Parameters
    ----------
    station : str
        Station identifier (e.g. ``'US-UTJ'``).
    interval : int
        Data interval in minutes (30 or 60).
    raw_data_root : Path
        Root folder containing compiled station data.
    output_root : Path
        Root folder for processed outputs (raw/, qc/, ameriflux/ sub-dirs).
    amflux_var_file : Path or None
        Path to the AmeriFlux variable-name CSV. Used for column validation.
    preprocessed_dir : Path or None
        Directory for preprocessed parquet files. Defaults to
        ``raw_data_root / 'preprocessed_site_data'``.
    steps : list of int
        Which workflow steps to run (1-4). Default is all.
    generate_plots : bool
        Whether to generate review plots (notebooks 3b/4b).
    drop_soil : bool
        Whether to drop extra soil columns during reformatter finalize.
    fetch_events_from_db : bool
        Whether to pull station events from the UGS API.
    events_api_url : str
        Base URL for the station events API.
    data_interval_label : str
        AmeriFlux interval label (``'HH'`` for half-hourly).
    soilvue_g_calculation : bool
        Whether to calculate SoilVue G values using gradient+storage.
    soilvue_depths_cm : list of float
        SoilVue sensor depths in centimeters.
    """

    station: str = ""
    interval: int = 30
    raw_data_root: Path = Path(".")
    output_root: Path = Path(".")
    amflux_var_file: Optional[Path] = None
    preprocessed_dir: Optional[Path] = None

    steps: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    generate_plots: bool = False
    drop_soil: bool = False

    fetch_events_from_db: bool = False
    events_api_url: str = "https://ugs-koop-umfdxaxiyq-wm.a.run.app"

    data_interval_label: str = "HH"

    soilvue_g_calculation: bool = False
    soilvue_depths_cm: List[float] = field(
        default_factory=lambda: [5, 10, 20, 30, 40, 50, 60]
    )

    @property
    def preprocessed_path(self) -> Path:
        if self.preprocessed_dir is not None:
            return self.preprocessed_dir
        return self.raw_data_root / "preprocessed_site_data"


@dataclass
class WorkflowResult:
    """Container for results of a workflow run."""

    station: str
    success: bool
    steps_completed: List[int] = field(default_factory=list)
    output_files: Dict[str, Path] = field(default_factory=dict)
    reports: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[int, str] = field(default_factory=dict)
    processing_time: float = 0.0

    def summary(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"Workflow Result: {status}",
            f"Station: {self.station}",
            f"Steps completed: {self.steps_completed}",
            f"Time: {self.processing_time:.1f}s",
        ]
        for step, path in self.output_files.items():
            lines.append(f"  {step}: {path}")
        for step, err in self.errors.items():
            lines.append(f"  Step {step} error: {err}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default CSFlux join columns (from notebook 2)
# ---------------------------------------------------------------------------

_DEFAULT_CSFLUX_JOIN_COLS = [
    "LE_1_1_1", "H_1_1_1", "NETRAD_1_1_1", "G_1_1_A", "G_2_1_1",
    "G_1_1_1", "SG_2_1_1", "SG_1_1_1", "G_PLATE_2_1_1", "G_PLATE_1_1_1",
    "SG_1_1_A", "TAU", "USTAR", "TA_1_1_1", "RH_1_1_1", "T_DP_1_1_1",
    "TA_1_2_1", "RH_1_2_1", "T_DP_1_2_1", "TA_1_3_1", "RH_1_3_1",
    "T_DP_1_3_1", "PA_1_1_1", "VPD_1_1_1", "TA_1_4_1", "T_SONIC_SIGMA",
    "WS_1_1_1", "WD_1_1_1", "WS_MAX_1_1_1", "CO2_SIG_STRGTH_MIN",
    "H2O_SIG_STRGTH_MIN", "P_1_1_1", "ALB_1_1_1", "SW_IN_1_1_1",
    "SW_OUT_1_1_1", "LW_IN_1_1_1", "LW_OUT_1_1_1", "T_SONIC_1_1_1",
    "TS_1_1_1", "TS_2_1_1", "SWC_1_1_1", "SWC_2_1_1", "FETCH_MAX",
    "FETCH_90", "FETCH_55", "FETCH_40", "R_LW_IN_MEAS", "R_LW_OUT_MEAS",
    "FILE_NAME", "CO2_DENSITY", "CO2_DENSITY_SIGMA", "FC_MASS", "FC_QC",
    "FC_SAMPLES", "FP_DIST_INTRST", "H2O_DENSITY", "H2O_DENSITY_SIGMA",
    "H_QC", "H_SAMPLES", "LE_QC", "LE_SAMPLES", "TAU_QC", "TKE", "TSTAR",
    "T_NR", "UPWND_DIST_INTRST", "UX", "UX_SIGMA", "UY", "UY_SIGMA", "UZ",
    "UZ_SIGMA", "WD_SIGMA", "WD_SONIC",
]

_DEFAULT_MERGE_DROP_COLS = [
    "RECORD_x", "RECORD_y", "G_1_1_A", "G_3_1_1", "SG_1_1_A",
    "TIMESTAMP_START_x", "TIMESTAMP_END_x",
    "TIMESTAMP_START_y", "TIMESTAMP_END_y",
]

_DEFAULT_NO_SUFFIX_COLS = [
    "CO2_SIGMA", "H2O_SIGMA", "FC_SSITC_TEST", "LE_SSITC_TEST",
    "ET_SSITC_TEST", "H_SSITC_TEST", "USTAR", "ZL", "TAU",
    "TAU_SSITC_TEST", "MO_LENGTH", "U", "U_SIGMA", "V", "V_SIGMA",
    "W", "W_SIGMA", "T_SONIC_SIGMA", "CO2_SIG_STRGTH_MIN",
    "H2O_SIG_STRGTH_MIN", "R_LW_IN_MEAS", "R_LW_OUT_MEAS",
    "T_NR", "T_NR_OUT", "T_CANOPY", "T_SI111_BODY", "PPFD_IN",
    "WND_DIR_SD1_WVT", "D_SNOW", "CO2_DENSITY", "CO2_DENSITY_SIGMA",
    "FC_MASS", "FC_QC", "FC_SAMPLES", "H2O_DENSITY", "H2O_DENSITY_SIGMA",
    "H_QC", "H_SAMPLES", "LE_QC", "LE_SAMPLES", "TAU_QC", "TKE", "TSTAR",
    "UX", "UX_SIGMA", "UY", "UY_SIGMA", "UZ", "UZ_SIGMA",
    "WD_SIGMA", "WD_SONIC",
]


# ---------------------------------------------------------------------------
# Notebook 1 helper: preprocess_data (adapted from the notebook function)
# ---------------------------------------------------------------------------


def _preprocess_data(
    station: str,
    parent_fold: Path,
    glob_name: str,
    interval: int,
    skip_rows: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Read, preprocess, and concatenate station data files.

    Mirrors the ``preprocess_data`` function defined in notebook 1.
    """
    from micromet.reader import AmerifluxDataProcessor

    reader = AmerifluxDataProcessor(logger=logger)

    files = sorted(parent_fold.glob(glob_name))
    if not files:
        logger.warning("No files matching %s in %s", glob_name, parent_fold)
        return pd.DataFrame()

    data_type = "met" if "statistics" in glob_name.lower() else "eddy"
    logger.info(
        "Preprocessing %d %s files from %s", len(files), data_type, parent_fold
    )

    frames: list[pd.DataFrame] = []
    reformatter = Reformatter(
        check_timestamps=False,
        drop_soil=False,
        logger=logger,
    )

    for f in files:
        try:
            df = reader.to_dataframe(f)
            df, _report, _ts = reformatter.process(
                df, interval=interval, data_type=data_type
            )
            frames.append(df)
        except Exception as exc:
            logger.warning("Skipping %s: %s", f.name, exc)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()
    return combined


# ---------------------------------------------------------------------------
# WorkflowRunner
# ---------------------------------------------------------------------------


class WorkflowRunner:
    """
    Orchestrates the full numbered-notebook workflow for a single station.

    Parameters
    ----------
    config : WorkflowConfig
        Workflow configuration.
    corrections : SiteCorrections or None
        Site-specific corrections to apply during the QC step.
    logger : logging.Logger or None
        Logger instance.
    """

    def __init__(
        self,
        config: WorkflowConfig,
        corrections: Optional[SiteCorrections] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.corrections = corrections or SiteCorrections()
        self.logger = logger_check(logger)
        self._events_df: Optional[pd.DataFrame] = None
        self._metadata_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> WorkflowResult:
        """
        Execute the configured workflow steps in sequence.

        Returns
        -------
        WorkflowResult
        """
        start = datetime.now()
        result = WorkflowResult(station=self.config.station, success=False)

        step_methods = {
            1: self.step1_compile_and_preprocess,
            2: self.step2_create_raw_data,
            3: self.step3_qc_data,
            4: self.step4_export_ameriflux,
        }

        # Intermediate data passed between steps
        context: Dict[str, Any] = {}

        for step_num in sorted(self.config.steps):
            method = step_methods.get(step_num)
            if method is None:
                self.logger.warning("Unknown step %d, skipping", step_num)
                continue

            self.logger.info(
                "\n%s Step %d: %s %s",
                "=" * 20, step_num, method.__doc__.strip().split("\n")[0],
                "=" * 20,
            )
            try:
                context = method(context)
                result.steps_completed.append(step_num)

                # Capture any output files produced
                for key in ("preprocessed_files", "raw_file", "qc_file", "ameriflux_file"):
                    if key in context and context[key] is not None:
                        if isinstance(context[key], dict):
                            result.output_files.update(context[key])
                        else:
                            result.output_files[key] = context[key]

            except Exception as exc:
                self.logger.error("Step %d failed: %s", step_num, exc, exc_info=True)
                result.errors[step_num] = str(exc)
                break

        # Optional plot generation
        if self.config.generate_plots and "qc_df" in context:
            try:
                self.logger.info("\n%s Generating review plots %s", "=" * 20, "=" * 20)
                self.generate_review_plots(context)
            except Exception as exc:
                self.logger.warning("Plot generation failed: %s", exc)

        result.processing_time = (datetime.now() - start).total_seconds()
        result.success = len(result.errors) == 0
        self.logger.info("\n%s", result.summary())
        return result

    # ------------------------------------------------------------------
    # Step 1: Compile and Preprocess  (Notebook 1)
    # ------------------------------------------------------------------

    def step1_compile_and_preprocess(
        self, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile raw files and preprocess into parquet datasets."""
        cfg = self.config
        station = cfg.station
        interval = cfg.interval
        raw_root = cfg.raw_data_root
        out_dir = cfg.preprocessed_path
        out_dir.mkdir(parents=True, exist_ok=True)

        station_dir = raw_root / station
        preprocessed_files: Dict[str, Path] = {}

        # --- Met Statistics tables ---
        met_stats = _preprocess_data(
            station,
            station_dir / "Statistics",
            "TOA5*Statistics*.dat",
            interval,
            skip_rows=True,
            logger=self.logger,
        )
        if not met_stats.empty:
            met_stats = interval_updates.subset_interval(
                met_stats,
                interval_updates.interval_update_dict,
                interval,
                data_type="met",
            )
            p = out_dir / f"{station}_{interval}_metstats_preprocessed.parquet"
            met_stats.to_parquet(p)
            preprocessed_files["metstats"] = p
            self.logger.info("  Exported %s (%d rows)", p.name, len(met_stats))

        # --- Met Statistics AmeriFlux tables ---
        met_af = _preprocess_data(
            station,
            station_dir / "Statistics_Ameriflux",
            "*Statistics_AmeriFlux*.dat",
            interval,
            skip_rows=False,
            logger=self.logger,
        )
        if not met_af.empty:
            # Drop all-NA columns
            met_af = met_af.dropna(axis=1, how="all")
            met_af = interval_updates.subset_interval(
                met_af,
                interval_updates.interval_update_dict,
                interval,
                data_type="met",
            )
            p = out_dir / f"{station}_{interval}_metstatsaf_preprocessed.parquet"
            met_af.to_parquet(p)
            preprocessed_files["metstatsaf"] = p
            self.logger.info("  Exported %s (%d rows)", p.name, len(met_af))

        # --- Eddy AmeriFlux Format from web ---
        eddy_af_web = _preprocess_data(
            station,
            station_dir,
            "*_Flux_AmeriFluxFormat.dat",
            interval,
            skip_rows=True,
            logger=self.logger,
        )
        if not eddy_af_web.empty:
            eddy_af_web = interval_updates.subset_interval(
                eddy_af_web,
                interval_updates.interval_update_dict,
                interval,
                data_type="eddy",
            )
            p = out_dir / f"{station}_{interval}_eddyaf_web_preprocessed.parquet"
            eddy_af_web.to_parquet(p)
            preprocessed_files["eddyaf_web"] = p
            self.logger.info("  Exported %s (%d rows)", p.name, len(eddy_af_web))

        # --- Eddy CSFlux Format from web ---
        eddy_cs_web = _preprocess_data(
            station,
            station_dir,
            "*_Flux_CSFormat*.dat",
            interval,
            skip_rows=True,
            logger=self.logger,
        )
        if not eddy_cs_web.empty:
            eddy_cs_web = interval_updates.subset_interval(
                eddy_cs_web,
                interval_updates.interval_update_dict,
                interval,
                data_type="eddy",
            )
            p = out_dir / f"{station}_{interval}_eddycsflux_web_preprocessed.parquet"
            eddy_cs_web.to_parquet(p)
            preprocessed_files["eddycsflux_web"] = p
            self.logger.info("  Exported %s (%d rows)", p.name, len(eddy_cs_web))

        # --- Eddy AmeriFlux Format from datalogger ---
        eddy_af_dl = _preprocess_data(
            station,
            station_dir / "AmeriFluxFormat",
            "*Flux_AmeriFluxFormat*.dat",
            interval,
            skip_rows=False,
            logger=self.logger,
        )
        if not eddy_af_dl.empty:
            eddy_af_dl = eddy_af_dl.dropna(axis=1, how="all")
            eddy_af_dl = interval_updates.subset_interval(
                eddy_af_dl,
                interval_updates.interval_update_dict,
                interval,
                data_type="eddy",
            )
            p = out_dir / f"{station}_{interval}_eddyaf_dl_preprocessed.parquet"
            eddy_af_dl.to_parquet(p)
            preprocessed_files["eddyaf_dl"] = p
            self.logger.info("  Exported %s (%d rows)", p.name, len(eddy_af_dl))

        # --- Eddy CSFlux Format from datalogger ---
        eddy_cs_dl = _preprocess_data(
            station,
            station_dir / "Flux_CSFormat",
            "*_Flux_CSFormat*.dat",
            interval,
            skip_rows=True,
            logger=self.logger,
        )
        if not eddy_cs_dl.empty:
            eddy_cs_dl = interval_updates.subset_interval(
                eddy_cs_dl,
                interval_updates.interval_update_dict,
                interval,
                data_type="eddy",
            )
            p = out_dir / f"{station}_{interval}_eddycsflux_dl_preprocessed.parquet"
            eddy_cs_dl.to_parquet(p)
            preprocessed_files["eddycsflux_dl"] = p
            self.logger.info("  Exported %s (%d rows)", p.name, len(eddy_cs_dl))

        context["preprocessed_files"] = preprocessed_files
        return context

    # ------------------------------------------------------------------
    # Step 2: Create Raw Data  (Notebook 2)
    # ------------------------------------------------------------------

    def step2_create_raw_data(
        self, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge data sources and create the combined raw dataset."""
        cfg = self.config
        station = cfg.station
        interval = cfg.interval
        pp = cfg.preprocessed_path

        # ---- helper to safely read a preprocessed parquet ----
        def _read(name: str) -> pd.DataFrame:
            p = pp / f"{station}_{interval}_{name}_preprocessed.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                return data_cleaning.prep_parquet(station, df)
            self.logger.warning("  Missing preprocessed file: %s", p.name)
            return pd.DataFrame()

        # ---- Eddy: merge CSFlux web + dl ----
        csflux_web = _read("eddycsflux_web")
        csflux_dl = _read("eddycsflux_dl")
        if not csflux_web.empty and not csflux_dl.empty:
            csflux = merge.fillna_with_second_df(csflux_web, csflux_dl)
        elif not csflux_web.empty:
            csflux = csflux_web
        elif not csflux_dl.empty:
            csflux = csflux_dl
        else:
            csflux = pd.DataFrame()
        self.logger.info("  CSFlux merged: %d rows", len(csflux))

        # ---- Eddy: merge AmeriFlux web + dl ----
        af_web = _read("eddyaf_web")
        af_dl = _read("eddyaf_dl")

        # Rename T_SONIC to match csflux naming convention
        for df in (af_web, af_dl):
            if not df.empty and "T_SONIC" in df.columns:
                df.rename(columns={"T_SONIC": "T_SONIC_1_1_1"}, inplace=True)

        if not af_web.empty and not af_dl.empty:
            amflux = merge.fillna_with_second_df(af_web, af_dl)
        elif not af_web.empty:
            amflux = af_web
        elif not af_dl.empty:
            amflux = af_dl
        else:
            amflux = pd.DataFrame()
        self.logger.info("  AmeriFlux eddy merged: %d rows", len(amflux))

        # ---- Combine AmeriFlux + CSFlux eddy data ----
        if not amflux.empty and not csflux.empty:
            join_cols = self.corrections.csflux_join_cols or _DEFAULT_CSFLUX_JOIN_COLS
            available_cols = [c for c in join_cols if c in csflux.columns]
            eddy = merge.fillna_with_second_df(amflux, csflux[available_cols])
        elif not amflux.empty:
            eddy = amflux
        else:
            eddy = csflux
        self.logger.info("  Final eddy: %d rows", len(eddy))

        # ---- Met: merge met stats + met AF stats ----
        met_stat = _read("metstats")
        met_af = _read("metstatsaf")

        if not met_stat.empty and not met_af.empty:
            met = merge.fillna_with_second_df(met_stat, met_af)
        elif not met_stat.empty:
            met = met_stat
        elif not met_af.empty:
            met = met_af
        else:
            met = pd.DataFrame()

        # Drop SAMPLING_INTERVAL if present to avoid duplication in merge
        if "SAMPLING_INTERVAL" in met.columns:
            met = met.drop(columns=["SAMPLING_INTERVAL"])
        self.logger.info("  Final met: %d rows", len(met))

        # ---- Combine eddy + met ----
        if not eddy.empty:
            eddy = eddy.rename(columns={
                "FILE_NAME": "FILE_NAME_EDDY",
                "T_NR": "T_NR_1_1_1",
            })
        if not met.empty:
            met = met.rename(columns={
                "FILE_NAME": "FILE_NAME_MET",
                "T_NR": "T_NR_1_1_2",
            })

        if not eddy.empty and not met.empty:
            alldat = pd.merge(
                eddy, met, left_index=True, right_index=True, how="outer"
            )
        elif not eddy.empty:
            alldat = eddy
        else:
            alldat = met

        # Drop known duplicate/derived columns safely
        drop_cols = self.corrections.columns_to_drop_from_merge or _DEFAULT_MERGE_DROP_COLS
        existing_drop = [c for c in drop_cols if c in alldat.columns]
        if existing_drop:
            alldat = alldat.drop(columns=existing_drop)

        # Add _1_1_1 suffix to un-suffixed columns
        rename_dict = columns.create_suffix_map(
            alldat, _DEFAULT_NO_SUFFIX_COLS, suffix="_1_1_1"
        )
        alldat = alldat.rename(columns=rename_dict)

        alldat["stationid"] = station

        # ---- Export raw parquet ----
        if alldat.empty:
            raise ValueError(
                f"No data found for {station}. Check that preprocessed files "
                "exist or that raw data is available."
            )

        raw_dir = cfg.output_root / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        dt_end = alldat.index.max()
        dt_start = alldat.index.min() - pd.Timedelta(minutes=interval)
        ts_start = dt_start.strftime("%Y%m%d%H%M")
        ts_end = dt_end.strftime("%Y%m%d%H%M")
        raw_file = raw_dir / f"{station}_{ts_start}_{ts_end}_raw.parquet"
        alldat.to_parquet(raw_file)
        self.logger.info("  Exported raw data to %s (%d rows)", raw_file.name, len(alldat))

        context["raw_file"] = raw_file
        context["raw_df"] = alldat
        context["date_range"] = f"{ts_start}_{ts_end}"
        return context

    # ------------------------------------------------------------------
    # Step 3: QC Data  (Notebook 3)
    # ------------------------------------------------------------------

    def step3_qc_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply corrections, physical limits, QC, and flagging."""
        cfg = self.config
        station = cfg.station
        corr = self.corrections

        # Load raw data (from context or disk)
        if "raw_df" in context:
            raw_dat = context["raw_df"]
        else:
            raw_dir = cfg.output_root / "raw"
            raw_files = sorted(raw_dir.glob(f"{station}_*_raw.parquet"))
            if not raw_files:
                raise FileNotFoundError(f"No raw parquet file found for {station}")
            raw_dat = pd.read_parquet(raw_files[-1])
            self.logger.info("  Loaded raw data from %s", raw_files[-1].name)

        df = raw_dat.copy()

        # ---- SG calibration correction ----
        if corr.sg_correction_factor is not None:
            self.logger.info("  Applying SG calibration correction (factor=%.4f)", corr.sg_correction_factor)
            df = fix_g_values.correct_vars_by_factor(
                df,
                correction_factor=corr.sg_correction_factor,
                vars_to_correct=corr.sg_correction_vars,
                min_correction_date="2010-01-01",
                max_correction_date=corr.sg_correction_end or "2030-01-01",
            )
            df = fix_g_values.calculate_new_g_value(df, "1")
            df = fix_g_values.calculate_new_g_value(df, "2")

        # ---- Precipitation correction ----
        if corr.precip_correction_factor is not None and corr.precip_correction_end is not None:
            self.logger.info("  Applying precipitation correction (factor=%.2f)", corr.precip_correction_factor)
            end_dt = pd.to_datetime(corr.precip_correction_end)
            mask = df.index < end_dt
            if "P_1_1_1" in df.columns:
                df.loc[mask, "P_1_1_1"] = df.loc[mask, "P_1_1_1"] * corr.precip_correction_factor

        # ---- Rename G plates and calculate G surface ----
        for col_plate, col_g, col_sg, col_surf in [
            ("G_PLATE_1_1_1", "G_1_1_1", "SG_1_1_1", "G_SURFACE_1_1_1"),
            ("G_PLATE_2_1_1", "G_2_1_1", "SG_2_1_1", "G_SURFACE_2_1_1"),
        ]:
            if col_plate in df.columns:
                df[col_g] = df[col_plate]
                df.drop(columns=[col_plate], inplace=True)
            if col_g in df.columns and col_sg in df.columns:
                df[col_surf] = df[col_g] + df[col_sg]
                invalid = df[col_g].isna() | df[col_sg].isna()
                df.loc[invalid, col_surf] = np.nan

        # ---- SoilVue G calculation (optional) ----
        if cfg.soilvue_g_calculation:
            self._calculate_soilvue_g(df)

        # ---- Run Reformatter.finalize for physical limits ----
        self.logger.info("  Running Reformatter.finalize (physical limits)")
        reformatter = Reformatter(
            drop_soil=cfg.drop_soil,
            check_timestamps=False,
            logger=self.logger,
        )
        df, report, ts_results = reformatter.finalize(df)
        df = df.replace(-9999, np.nan)
        context["qc_report"] = report

        # Save report
        report_dir = cfg.output_root / "micromet_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        date_range = context.get("date_range", "unknown")
        report_path = report_dir / f"{station}_{date_range}_report.csv"
        report.to_csv(report_path)
        self.logger.info("  Report exported to %s", report_path.name)

        # ---- Drop precip on station visit days ----
        if corr.drop_precip_on_visits and "P_1_1_1" in df.columns:
            events = self._get_events()
            if events is not None:
                visits = events.loc[
                    (events.stationid == station) & (events.event_type == "station visit")
                ]
                visit_dates = visits.event_date.dt.floor("D")
                precip_mask = (
                    df.index.floor("D").isin(visit_dates) & (df["P_1_1_1"] > 0)
                )
                n_zeroed = precip_mask.sum()
                df.loc[precip_mask, "P_1_1_1"] = 0
                self.logger.info("  Zeroed %d precip records on visit days", n_zeroed)

        # ---- Drop zero G plate values ----
        for g_col, surf_col in [
            ("G_1_1_1", "G_SURFACE_1_1_1"),
            ("G_2_1_1", "G_SURFACE_2_1_1"),
        ]:
            if g_col in df.columns:
                zero_mask = df[g_col] == 0
                df.loc[zero_mask, [g_col]] = np.nan
                if surf_col in df.columns:
                    df.loc[zero_mask, [surf_col]] = np.nan

        # ---- Date-range drops ----
        for drop in corr.date_range_drops + corr.extra_drops:
            if drop.column in df.columns:
                df = data_cleaning.set_range_to_nan(
                    df, drop.column, drop.start, drop.end
                )
                self.logger.info("  Nulled %s from %s to %s", drop.column, drop.start, drop.end)

        # ---- Precipitation: drop before bad date ----
        if corr.precip_bad_before is not None and "P_1_1_1" in df.columns:
            bad_dt = pd.to_datetime(corr.precip_bad_before)
            df.loc[df.index <= bad_dt, "P_1_1_1"] = np.nan
            self.logger.info("  Dropped precip before %s", corr.precip_bad_before)

        # ---- Wind direction offset ----
        if corr.wind_direction_offset is not None and "WD_1_1_1" in df.columns:
            change_dt = pd.to_datetime(corr.wind_direction_change_date or "2099-01-01")
            mask = df.index < change_dt
            df.loc[mask, "WD_1_1_1"] = (
                df.loc[mask, "WD_1_1_1"] - corr.wind_direction_offset
            ) % 360
            self.logger.info(
                "  Applied wind direction offset of %.1f° before %s",
                corr.wind_direction_offset,
                change_dt,
            )

        # ---- SoilVue bad-data filter ----
        if corr.soilvue_bad_ec_threshold is not None and "EC_3_7_1" in df.columns:
            bad_mask = df["EC_3_7_1"] < corr.soilvue_bad_ec_threshold
            soilvue_cols = [
                c for c in df.columns
                if any(c.startswith(p) for p in ("EC_3", "K_3", "SWC_3", "TS_3"))
            ]
            df.loc[bad_mask, soilvue_cols] = np.nan
            self.logger.info(
                "  Dropped %d SoilVue records (EC < %.2f)",
                bad_mask.sum(),
                corr.soilvue_bad_ec_threshold,
            )

        # ---- H2O signal-strength flags ----
        if "H2O_SIG_STRGTH_MIN_1_1_1" in df.columns:
            df["H2O_SIG_FLAG_1_1_1"] = 0
            low_sig = df["H2O_SIG_STRGTH_MIN_1_1_1"] < corr.signal_strength_threshold
            df.loc[low_sig, "H2O_SIG_FLAG_1_1_1"] = 1
            for win in corr.h2o_flag_windows:
                df = data_cleaning.apply_internal_flags(
                    df, win.flag_columns, win.start, win.end, win.flag_value
                )

        # ---- CO2 signal-strength flags ----
        if "CO2_SIG_STRGTH_MIN_1_1_1" in df.columns:
            df["CO2_SIG_FLAG_1_1_1"] = 0
            low_sig = df["CO2_SIG_STRGTH_MIN_1_1_1"] < corr.signal_strength_threshold
            df.loc[low_sig, "CO2_SIG_FLAG_1_1_1"] = 1
            for win in corr.co2_flag_windows:
                df = data_cleaning.apply_internal_flags(
                    df, win.flag_columns, win.start, win.end, win.flag_value
                )

        # ---- Wind direction flags ----
        if corr.wind_flag_bad_range is not None and "WD_1_1_1" in df.columns:
            df["WD_1_1_1_FLAG"] = 0
            lo, hi = corr.wind_flag_bad_range
            bad_wind = (df["WD_1_1_1"] >= lo) & (df["WD_1_1_1"] < hi)
            df.loc[bad_wind, "WD_1_1_1_FLAG"] = 2
            for lo_m, hi_m in corr.wind_flag_marginal_ranges:
                marginal = df["WD_1_1_1"].between(lo_m, hi_m, inclusive="right")
                df.loc[marginal, "WD_1_1_1_FLAG"] = 1

        # ---- Create composite G_1 via regression imputation ----
        g_surface_cols = [
            c for c in ("G_SURFACE_1_1_1", "G_SURFACE_2_1_1", "G_SURFACE_3_1_1")
            if c in df.columns
        ]
        if len(g_surface_cols) >= 2:
            self.logger.info("  Creating composite G_1 from %s", g_surface_cols)
            df["G_1"] = df[g_surface_cols[:2]].mean(axis=1, skipna=False)
            for pred_col in g_surface_cols:
                model, results = data_cleaning.train_linear_regression_model(
                    df, target_col="G_1", predictor_col=pred_col
                )
                if model is not None:
                    df["G_1"] = data_cleaning.impute_missing_values(
                        df, model, target_col="G_1", predictor_col=pred_col
                    )
                    self.logger.info(
                        "    Imputed G_1 from %s (R²=%.3f, n=%d)",
                        pred_col,
                        results.get("r_squared", 0),
                        results.get("training_n_samples", 0),
                    )

        # ---- Add time-of-day / day-of-year columns ----
        df["day_of_year"] = df.index.dayofyear
        df["time_of_day"] = df.index.hour + df.index.minute / 60
        df["days_since_20240101"] = (df.index - pd.Timestamp("2024-01-01")).days

        # ---- Export QC parquet ----
        qc_dir = cfg.output_root / "qc"
        qc_dir.mkdir(parents=True, exist_ok=True)
        date_range = context.get("date_range", "unknown")
        qc_file = qc_dir / f"{station}_{date_range}_qc.parquet"
        df.to_parquet(qc_file)
        self.logger.info("  Exported QC data to %s (%d rows)", qc_file.name, len(df))

        context["qc_file"] = qc_file
        context["qc_df"] = df
        return context

    # ------------------------------------------------------------------
    # Step 4: Export AmeriFlux  (Notebook 4)
    # ------------------------------------------------------------------

    def step4_export_ameriflux(
        self, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Export AmeriFlux-formatted CSV from QC data."""
        cfg = self.config
        station = cfg.station
        corr = self.corrections

        # Load QC data
        if "qc_df" in context:
            qc_dat = context["qc_df"].copy()
        else:
            qc_dir = cfg.output_root / "qc"
            qc_files = sorted(qc_dir.glob(f"{station}_*_qc.parquet"))
            if not qc_files:
                raise FileNotFoundError(f"No QC parquet file found for {station}")
            qc_dat = pd.read_parquet(qc_files[-1])

        # ---- Drop data based on signal strength ----
        threshold = corr.signal_strength_threshold
        if "H2O_SIG_STRGTH_MIN_1_1_1" in qc_dat.columns:
            mask = qc_dat["H2O_SIG_STRGTH_MIN_1_1_1"] < threshold
            h2o_drops = [
                c for c in [
                    "H2O_1_1_1", "H2O_SIGMA_1_1_1", "LE_1_1_1",
                    "RH_1_1_1", "RH_1_2_1", "VPD_1_1_1", "ET_1_1_1",
                ] if c in qc_dat.columns
            ]
            qc_dat.loc[mask, h2o_drops] = np.nan
            self.logger.info("  Dropped %d records for H2O signal strength", mask.sum())

        if "CO2_SIG_STRGTH_MIN_1_1_1" in qc_dat.columns:
            mask = qc_dat["CO2_SIG_STRGTH_MIN_1_1_1"] < threshold
            co2_drops = [
                c for c in ["CO2_1_1_1", "CO2_SIGMA_1_1_1", "FC_1_1_1"]
                if c in qc_dat.columns
            ]
            qc_dat.loc[mask, co2_drops] = np.nan
            self.logger.info("  Dropped %d records for CO2 signal strength", mask.sum())

        # ---- Drop all-null columns ----
        all_null = qc_dat.isnull().all()
        if all_null.any():
            self.logger.info("  Dropping %d all-null columns", all_null.sum())
            qc_dat = qc_dat.drop(columns=qc_dat.columns[all_null])

        # ---- Drop non-AmeriFlux columns ----
        if cfg.amflux_var_file is not None and cfg.amflux_var_file.exists():
            amflux_vars = pd.read_csv(cfg.amflux_var_file)
            comparison = validate.compare_names_to_ameriflux(qc_dat, amflux_vars)
            non_af = comparison.loc[~comparison.is_in_amflux, "all_columns"]
            # Always keep PBLH_F if present
            keep = {"PBLH_F"}
            non_af = non_af[~non_af.isin(keep)]
            if not non_af.empty:
                existing = [c for c in non_af if c in qc_dat.columns]
                qc_dat = qc_dat.drop(columns=existing)
                self.logger.info("  Dropped %d non-AmeriFlux columns", len(existing))
        else:
            # Drop known non-AF columns by pattern
            drop_patterns = [
                "FILE_NAME", "stationid", "day_of_year", "time_of_day",
                "days_since", "_FLAG", "RECORD",
            ]
            cols_to_drop = [
                c for c in qc_dat.columns
                if any(p in c for p in drop_patterns)
            ]
            if cols_to_drop:
                qc_dat = qc_dat.drop(columns=cols_to_drop)

        # ---- Fill NaN with -9999 ----
        qc_dat = qc_dat.fillna(-9999)

        # ---- Add AmeriFlux timestamps ----
        qc_dat = timestamps.add_ameriflux_timestamps(qc_dat, interval_minutes=cfg.interval)

        # ---- Export ----
        af_dir = cfg.output_root / "ameriflux"
        af_dir.mkdir(parents=True, exist_ok=True)
        ts_start = qc_dat["TIMESTAMP_START"].min()
        ts_end = qc_dat["TIMESTAMP_END"].max()
        af_file = af_dir / f"{station}_{cfg.data_interval_label}_{ts_start}_{ts_end}.csv"
        qc_dat.to_csv(af_file, index=False)
        self.logger.info("  Exported AmeriFlux data to %s (%d rows)", af_file.name, len(qc_dat))

        context["ameriflux_file"] = af_file
        return context

    # ------------------------------------------------------------------
    # Review plots  (Notebooks 3b / 4b)
    # ------------------------------------------------------------------

    def generate_review_plots(self, context: Dict[str, Any]) -> None:
        """Generate time-series plots for all variables in the QC dataset."""
        df = context.get("qc_df")
        if df is None:
            self.logger.warning("No QC data available for plotting")
            return

        skip_cols = {"FILE_NAME_MET", "FILE_NAME_EDDY", "stationid"}
        plot_cols = [c for c in df.columns.sort_values() if c not in skip_cols]
        self.logger.info("  Generating plots for %d variables", len(plot_cols))

        for var in plot_cols:
            try:
                eddy_plots.plotlystuff([df], [var], chrttitle=var)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_events(self) -> Optional[pd.DataFrame]:
        """Fetch station events from the UGS API (cached)."""
        if self._events_df is not None:
            return self._events_df

        if not self.config.fetch_events_from_db:
            return None

        try:
            import requests

            headers = {
                "Accept-Profile": "groundwater",
                "Content-Type": "application/json",
            }
            resp = requests.get(
                f"{self.config.events_api_url}/eddy_events",
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            self._events_df = pd.DataFrame(resp.json())
            self._events_df["event_date"] = pd.to_datetime(
                self._events_df["event_date"]
            )
            return self._events_df
        except Exception as exc:
            self.logger.warning("Failed to fetch events: %s", exc)
            return None

    def _calculate_soilvue_g(self, df: pd.DataFrame) -> None:
        """Calculate SoilVue G values using gradient+storage method."""
        try:
            from soil_heat import johansen
        except ImportError:
            self.logger.warning(
                "soil_heat package not installed; skipping SoilVue G calculation"
            )
            return

        depths_cm = np.array(self.config.soilvue_depths_cm, dtype=float)
        suffixes = [f"3_{i}_1" for i in range(1, len(depths_cm) + 1)]
        ts_cols = [f"TS_{s}" for s in suffixes]
        swc_cols = [f"SWC_{s}" for s in suffixes]

        missing = [c for c in ts_cols + swc_cols if c not in df.columns]
        if missing:
            self.logger.warning("Missing SoilVue columns: %s", missing)
            return

        dt = self.config.interval * 60
        ts = df[ts_cols].copy()
        ts.columns = depths_cm
        swc = df[swc_cols].copy()
        swc.columns = depths_cm

        new_vals = johansen.gradient_plus_storage(
            ts, swc, ref_depth_idx=2, dt=dt, lam_model="johansen"
        )
        new_vals.rename(
            columns={
                "G_ref": "G_3_1_1",
                "G_surface": "G_SURFACE_3_1_1",
                "Storage": "SG_3_1_1",
            },
            inplace=True,
        )
        for col in new_vals.columns:
            df[col] = new_vals[col]
        self.logger.info("  Calculated SoilVue G values")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def run_workflow(
    station: str,
    raw_data_root: Union[str, Path],
    output_root: Union[str, Path],
    corrections: Optional[SiteCorrections] = None,
    **kwargs,
) -> WorkflowResult:
    """
    Convenience function to run the complete workflow for a station.

    Parameters
    ----------
    station : str
        Station identifier (e.g. ``'US-UTJ'``).
    raw_data_root : str or Path
        Root folder containing compiled station data.
    output_root : str or Path
        Root folder for processed outputs.
    corrections : SiteCorrections, optional
        Site-specific corrections.
    **kwargs
        Additional arguments passed to :class:`WorkflowConfig`.

    Returns
    -------
    WorkflowResult
    """
    config = WorkflowConfig(
        station=station,
        raw_data_root=Path(raw_data_root),
        output_root=Path(output_root),
        **kwargs,
    )
    runner = WorkflowRunner(config=config, corrections=corrections)
    return runner.run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """Command-line interface for the workflow."""
    parser = argparse.ArgumentParser(
        description="Run the MicroMet data processing workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all steps for a station
  python -m micromet.workflow --station US-UTJ \\
      --raw-data-root M:/Shared/Data_Downloads/compiled \\
      --output-root M:/Shared/Data_Processing/final_database_tables

  # Run only steps 2 and 3 (data already preprocessed)
  python -m micromet.workflow --station US-UTJ --steps 2 3 \\
      --raw-data-root ... --output-root ...

  # Run with plots
  python -m micromet.workflow --station US-UTJ --plots \\
      --raw-data-root ... --output-root ...
        """,
    )

    parser.add_argument(
        "--station", "-s", required=True, help="Station ID (e.g. US-UTJ)"
    )
    parser.add_argument(
        "--raw-data-root", "-r", required=True,
        help="Root directory with compiled raw data",
    )
    parser.add_argument(
        "--output-root", "-o", required=True,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--interval", type=int, default=30,
        help="Data interval in minutes (default: 30)",
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, default=[1, 2, 3, 4],
        help="Workflow steps to run (default: 1 2 3 4)",
    )
    parser.add_argument(
        "--plots", action="store_true",
        help="Generate review plots after processing",
    )
    parser.add_argument(
        "--amflux-var-file", type=str, default=None,
        help="Path to AmeriFlux variable-name CSV",
    )
    parser.add_argument(
        "--fetch-events", action="store_true",
        help="Fetch station events from UGS API",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s [%(asctime)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = WorkflowConfig(
        station=args.station,
        interval=args.interval,
        raw_data_root=Path(args.raw_data_root),
        output_root=Path(args.output_root),
        amflux_var_file=Path(args.amflux_var_file) if args.amflux_var_file else None,
        steps=args.steps,
        generate_plots=args.plots,
        fetch_events_from_db=args.fetch_events,
    )

    runner = WorkflowRunner(config=config)
    result = runner.run()

    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
