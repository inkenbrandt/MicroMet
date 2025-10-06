"""
Complete pipeline for processing micrometeorological data with Micromet.

This module provides high-level orchestration for the complete data processing
workflow, from raw data files to cleaned, validated, and analyzed datasets.

Classes
-------
Pipeline : Main orchestration class for data processing
PipelineConfig : Configuration container for pipeline settings
ProcessingResult : Container for processing results and metadata

Functions
---------
run_pipeline : Convenience function to run complete pipeline
process_station : Process a single station's data
batch_process : Process multiple stations

Examples
--------
Basic usage:

    >>> from micromet.pipeline import Pipeline
    >>> 
    >>> # Process a single file
    >>> pipeline = Pipeline()
    >>> result = pipeline.process_file(
    ...     'data/US-UTW_Flux.dat',
    ...     site_id='US-UTW'
    ... )
    >>> 
    >>> # Batch process all stations
    >>> results = pipeline.batch_process(
    ...     input_dir='./raw_data',
    ...     output_dir='./processed_data'
    ... )

Command-line usage:

    $ python -m micromet.pipeline --site US-UTW --input data/ --output results/
    $ python -m micromet.pipeline --batch --input data/ --output results/
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from micromet.reader import AmerifluxDataProcessor
from micromet.format.reformatter import Reformatter
from micromet.qaqc import variable_limits
from micromet.report import gap_summary, validate, graphs
from micromet.utils import (
    logger_check,
    read_site_config,
    get_all_site_configs,
    create_reformatter_from_site
)


# =============================================================================
# Configuration and Result Containers
# =============================================================================

@dataclass
class PipelineConfig:
    """
    Configuration settings for the data processing pipeline.
    
    Attributes
    ----------
    check_timestamps : bool
        Whether to perform timestamp alignment analysis (slower but thorough).
    drop_soil : bool
        Whether to drop extra soil sensor columns.
    generate_reports : bool
        Whether to generate validation and gap reports.
    generate_plots : bool
        Whether to generate diagnostic plots.
    save_intermediate : bool
        Whether to save intermediate processing steps.
    var_limits_csv : Path or None
        Path to custom variable limits CSV file.
    expected_freq : str
        Expected data frequency (e.g., '30min').
    output_format : str
        Output file format ('csv', 'parquet', 'feather').
    """
    
    check_timestamps: bool = True
    drop_soil: bool = True
    generate_reports: bool = True
    generate_plots: bool = False
    save_intermediate: bool = False
    var_limits_csv: Optional[Path] = None
    expected_freq: str = '30min'
    output_format: str = 'csv'
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        d = asdict(self)
        # Convert Path objects to strings
        if d['var_limits_csv'] is not None:
            d['var_limits_csv'] = str(d['var_limits_csv'])
        return d


@dataclass
class ProcessingResult:
    """
    Container for processing results and metadata.
    
    Attributes
    ----------
    site_id : str
        Station identifier.
    success : bool
        Whether processing completed successfully.
    input_file : Path
        Path to input file.
    output_file : Path or None
        Path to output file (if saved).
    n_records_input : int
        Number of records in input data.
    n_records_output : int
        Number of records in output data.
    n_flagged : int
        Number of records flagged during QA/QC.
    processing_time : float
        Processing time in seconds.
    timestamp_issues : dict or None
        Detected timestamp alignment issues.
    error_message : str or None
        Error message if processing failed.
    reports : dict
        Dictionary of generated reports.
    """
    
    site_id: str
    success: bool
    input_file: Path
    output_file: Optional[Path] = None
    n_records_input: int = 0
    n_records_output: int = 0
    n_flagged: int = 0
    processing_time: float = 0.0
    timestamp_issues: Optional[Dict] = None
    error_message: Optional[str] = None
    reports: Dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        d = asdict(self)
        # Convert Path objects to strings
        d['input_file'] = str(d['input_file'])
        if d['output_file'] is not None:
            d['output_file'] = str(d['output_file'])
        return d
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"Processing Result: {status}",
            f"Site: {self.site_id}",
            f"Input: {self.input_file}",
            f"Records: {self.n_records_input} → {self.n_records_output}",
            f"Flagged: {self.n_flagged}",
            f"Time: {self.processing_time:.2f}s",
        ]
        
        if self.timestamp_issues:
            lines.append(f"Timestamp Issues: {len(self.timestamp_issues)}")
        
        if self.error_message:
            lines.append(f"Error: {self.error_message}")
        
        return "\n".join(lines)


# =============================================================================
# Main Pipeline Class
# =============================================================================

class Pipeline:
    """
    Main orchestration class for micrometeorological data processing.
    
    This class coordinates the complete workflow from raw data files to
    cleaned, validated, and analyzed datasets.
    
    Parameters
    ----------
    config : PipelineConfig, optional
        Configuration settings for the pipeline.
    logger : logging.Logger, optional
        Logger instance for tracking progress.
    
    Attributes
    ----------
    config : PipelineConfig
        Pipeline configuration.
    logger : logging.Logger
        Logger instance.
    reader : AmerifluxDataProcessor
        Data reader instance.
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the Pipeline."""
        self.config = config or PipelineConfig()
        self.logger = logger_check(logger)
        self.reader = AmerifluxDataProcessor(logger=self.logger)
        
        self.logger.info("Pipeline initialized")
        self.logger.debug(f"Configuration: {self.config.to_dict()}")
    
    def process_file(
        self,
        input_file: Union[str, Path],
        site_id: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        data_type: str = 'eddy',
    ) -> ProcessingResult:
        """
        Process a single data file through the complete pipeline.
        
        Parameters
        ----------
        input_file : str or Path
            Path to input data file.
        site_id : str, optional
            Station identifier. If None, attempts to extract from filename.
        output_dir : str or Path, optional
            Directory for output files. If None, uses input file directory.
        data_type : str, optional
            Type of data ('eddy' or 'met'). Defaults to 'eddy'.
        
        Returns
        -------
        ProcessingResult
            Container with processing results and metadata.
        """
        start_time = datetime.now()
        input_file = Path(input_file)
        
        # Extract site_id if not provided
        if site_id is None:
            site_id = self._extract_site_id(input_file)
        
        self.logger.info(f"Processing {site_id}: {input_file.name}")
        
        try:
            # Step 1: Read raw data
            self.logger.info("Step 1/5: Reading raw data...")
            df_raw = self.reader.to_dataframe(input_file)
            n_input = len(df_raw)
            self.logger.info(f"  Read {n_input:,} records")
            
            # Step 2: Reformat and clean
            self.logger.info("Step 2/5: Reformatting and cleaning...")
            df_clean, limits_report, ts_results = self._reformat_data(
                df_raw, site_id, data_type
            )
            n_output = len(df_clean)
            self.logger.info(f"  Output: {n_output:,} records")
            
            # Step 3: Generate reports
            reports = {}
            if self.config.generate_reports:
                self.logger.info("Step 3/5: Generating QA/QC reports...")
                reports = self._generate_reports(
                    df_clean, limits_report, ts_results, site_id
                )
            else:
                self.logger.info("Step 3/5: Skipping reports (disabled)")
            
            # Step 4: Generate plots
            if self.config.generate_plots and output_dir:
                self.logger.info("Step 4/5: Generating diagnostic plots...")
                self._generate_plots(df_clean, site_id, Path(output_dir))
            else:
                self.logger.info("Step 4/5: Skipping plots")
            
            # Step 5: Save output
            output_file = None
            if output_dir:
                self.logger.info("Step 5/5: Saving output...")
                output_file = self._save_output(
                    df_clean, site_id, Path(output_dir), data_type
                )
                
                # Save reports
                if reports and self.config.generate_reports:
                    self._save_reports(reports, site_id, Path(output_dir))
            else:
                self.logger.info("Step 5/5: Skipping save (no output_dir)")
            
            # Calculate metrics
            n_flagged = limits_report['n_flagged'].sum() if 'n_flagged' in limits_report else 0
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Extract timestamp issues
            ts_issues = None
            if ts_results and 'flags' in ts_results:
                ts_issues = ts_results['flags']
            
            result = ProcessingResult(
                site_id=site_id,
                success=True,
                input_file=input_file,
                output_file=output_file,
                n_records_input=n_input,
                n_records_output=n_output,
                n_flagged=int(n_flagged),
                processing_time=processing_time,
                timestamp_issues=ts_issues,
                reports=reports
            )
            
            self.logger.info(f"✓ Processing complete: {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"✗ Processing failed: {e}", exc_info=True)
            
            return ProcessingResult(
                site_id=site_id or "UNKNOWN",
                success=False,
                input_file=input_file,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def batch_process(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        pattern: str = "*Flux*.dat",
        data_type: str = 'eddy',
    ) -> List[ProcessingResult]:
        """
        Process multiple files in a directory.
        
        Parameters
        ----------
        input_dir : str or Path
            Directory containing input files.
        output_dir : str or Path
            Directory for output files.
        pattern : str, optional
            Glob pattern for finding input files. Defaults to "*Flux*.dat".
        data_type : str, optional
            Type of data ('eddy' or 'met'). Defaults to 'eddy'.
        
        Returns
        -------
        list of ProcessingResult
            Results for all processed files.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all matching files
        files = sorted(input_dir.rglob(pattern))
        
        if not files:
            self.logger.warning(f"No files found matching '{pattern}' in {input_dir}")
            return []
        
        self.logger.info(f"Found {len(files)} files to process")
        
        results = []
        for i, file in enumerate(files, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"File {i}/{len(files)}: {file.name}")
            self.logger.info(f"{'='*60}")
            
            result = self.process_file(
                input_file=file,
                output_dir=output_dir,
                data_type=data_type
            )
            results.append(result)
        
        # Generate batch summary
        self._log_batch_summary(results)
        self._save_batch_summary(results, output_dir)
        
        return results
    
    def process_station(
        self,
        site_id: str,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        data_types: List[str] = ['eddy', 'met'],
    ) -> Dict[str, ProcessingResult]:
        """
        Process all data types for a single station.
        
        Parameters
        ----------
        site_id : str
            Station identifier (e.g., 'US-UTW').
        input_dir : str or Path
            Directory containing input files.
        output_dir : str or Path
            Directory for output files.
        data_types : list of str, optional
            Data types to process. Defaults to ['eddy', 'met'].
        
        Returns
        -------
        dict
            Dictionary mapping data_type to ProcessingResult.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        self.logger.info(f"\n{'#'*60}")
        self.logger.info(f"Processing Station: {site_id}")
        self.logger.info(f"{'#'*60}")
        
        results = {}
        
        for data_type in data_types:
            # Find files for this data type
            if data_type == 'eddy':
                pattern = f"*{site_id}*AmeriFlux*.dat"
            else:
                pattern = f"*{site_id}*Statistics*.dat"
            
            files = list(input_dir.rglob(pattern))
            
            if not files:
                self.logger.warning(f"No {data_type} files found for {site_id}")
                continue
            
            # Process the most recent file (or combine if needed)
            file = max(files, key=lambda p: p.stat().st_mtime)
            
            result = self.process_file(
                input_file=file,
                site_id=site_id,
                output_dir=output_dir,
                data_type=data_type
            )
            
            results[data_type] = result
        
        return results
    
    # -------------------------------------------------------------------------
    # Private Helper Methods
    # -------------------------------------------------------------------------
    
    def _reformat_data(
        self,
        df: pd.DataFrame,
        site_id: str,
        data_type: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Dict]]:
        """Reformat and clean data using Reformatter."""
        # Try to load site configuration
        try:
            reformatter = create_reformatter_from_site(
                site_id=site_id,
                check_timestamps=self.config.check_timestamps,
                drop_soil=self.config.drop_soil,
                var_limits_csv=self.config.var_limits_csv
            )
        except (FileNotFoundError, KeyError):
            self.logger.warning(
                f"Could not load config for {site_id}, using defaults"
            )
            reformatter = Reformatter(
                check_timestamps=False,  # Can't check without site config
                drop_soil=self.config.drop_soil,
                var_limits_csv=self.config.var_limits_csv
            )
        
        # Process the data
        df_clean, limits_report, ts_results = reformatter.process(
            df, data_type=data_type
        )
        
        return df_clean, limits_report, ts_results
    
    def _generate_reports(
        self,
        df: pd.DataFrame,
        limits_report: pd.DataFrame,
        ts_results: Optional[Dict],
        site_id: str
    ) -> Dict[str, pd.DataFrame]:
        """Generate QA/QC and validation reports."""
        reports = {}
        
        # Limits report (already generated)
        reports['limits'] = limits_report
        
        # Gap summary
        if isinstance(df.index, pd.DatetimeIndex):
            try:
                # Create temporary multi-index for gap_summary
                df_temp = df.copy()
                df_temp['STATIONID'] = site_id
                df_temp['DATETIME_END'] = df_temp.index
                df_temp = df_temp.set_index(['STATIONID', 'DATETIME_END'])
                
                gaps = gap_summary.summarize_gaps(
                    df_temp,
                    expected_freq=self.config.expected_freq
                )
                reports['gaps'] = gaps
                self.logger.info(f"  Found {len(gaps)} gap periods")
            except Exception as e:
                self.logger.warning(f"  Gap analysis failed: {e}")
        
        # Flag validation
        try:
            flag_cols = [c for c in df.columns if c.endswith('_SSITC_TEST')]
            if flag_cols:
                invalid_flags = validate.validate_flags(df, flag_cols)
                if invalid_flags:
                    reports['invalid_flags'] = pd.DataFrame([
                        {'column': k, 'invalid_values': str(v)}
                        for k, v in invalid_flags.items()
                    ])
                    self.logger.warning(f"  Found invalid flags in {len(invalid_flags)} columns")
        except Exception as e:
            self.logger.warning(f"  Flag validation failed: {e}")
        
        # Timestamp consistency
        try:
            ts_inconsistencies = validate.validate_timestamp_consistency(df)
            if not ts_inconsistencies.empty:
                reports['timestamp_inconsistencies'] = ts_inconsistencies
                self.logger.warning(f"  Found {len(ts_inconsistencies)} timestamp inconsistencies")
        except Exception as e:
            self.logger.warning(f"  Timestamp validation failed: {e}")
        
        # Timestamp alignment (if checked)
        if ts_results:
            if 'summary' in ts_results:
                reports['timestamp_alignment'] = ts_results['summary']
            if 'flags' in ts_results and ts_results['flags']:
                self.logger.warning(f"  Timestamp alignment issues detected: {list(ts_results['flags'].keys())}")
        
        return reports
    
    def _generate_plots(
        self,
        df: pd.DataFrame,
        site_id: str,
        output_dir: Path
    ) -> None:
        """Generate diagnostic plots."""
        plot_dir = output_dir / 'plots' / site_id
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Add plot generation here as needed
        # Example: energy balance, time series, etc.
        self.logger.debug(f"  Plots would be saved to {plot_dir}")
    
    def _save_output(
        self,
        df: pd.DataFrame,
        site_id: str,
        output_dir: Path,
        data_type: str
    ) -> Path:
        """Save processed data to file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{site_id}_{data_type}_processed_{timestamp}"
        
        if self.config.output_format == 'csv':
            output_file = output_dir / f"{filename}.csv"
            df.to_csv(output_file, index=True)
        elif self.config.output_format == 'parquet':
            output_file = output_dir / f"{filename}.parquet"
            df.to_parquet(output_file, index=True)
        elif self.config.output_format == 'feather':
            output_file = output_dir / f"{filename}.feather"
            df.reset_index().to_feather(output_file)
        else:
            raise ValueError(f"Unknown output format: {self.config.output_format}")
        
        self.logger.info(f"  Saved to {output_file}")
        return output_file
    
    def _save_reports(
        self,
        reports: Dict[str, pd.DataFrame],
        site_id: str,
        output_dir: Path
    ) -> None:
        """Save QA/QC reports to files."""
        report_dir = output_dir / 'reports' / site_id
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for name, report_df in reports.items():
            if isinstance(report_df, pd.DataFrame) and not report_df.empty:
                filename = report_dir / f"{name}_{timestamp}.csv"
                report_df.to_csv(filename, index=False)
                self.logger.debug(f"  Saved {name} report to {filename}")
    
    def _extract_site_id(self, filepath: Path) -> str:
        """Extract site ID from filename."""
        # Look for US-XXX pattern
        name = filepath.stem
        for part in name.split('_'):
            if part.startswith('US-'):
                return part
        
        # Fallback
        self.logger.warning(f"Could not extract site_id from {filepath.name}, using 'UNKNOWN'")
        return 'UNKNOWN'
    
    def _log_batch_summary(self, results: List[ProcessingResult]) -> None:
        """Log summary statistics for batch processing."""
        n_total = len(results)
        n_success = sum(1 for r in results if r.success)
        n_failed = n_total - n_success
        
        total_input = sum(r.n_records_input for r in results)
        total_output = sum(r.n_records_output for r in results)
        total_time = sum(r.processing_time for r in results)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("BATCH PROCESSING SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total files:    {n_total}")
        self.logger.info(f"Successful:     {n_success} ({n_success/n_total*100:.1f}%)")
        self.logger.info(f"Failed:         {n_failed}")
        self.logger.info(f"Total records:  {total_input:,} → {total_output:,}")
        self.logger.info(f"Total time:     {total_time:.2f}s")
        self.logger.info(f"{'='*60}")
        
        if n_failed > 0:
            self.logger.warning("\nFailed files:")
            for r in results:
                if not r.success:
                    self.logger.warning(f"  {r.site_id}: {r.error_message}")
    
    def _save_batch_summary(
        self,
        results: List[ProcessingResult],
        output_dir: Path
    ) -> None:
        """Save batch processing summary to JSON file."""
        summary_file = output_dir / 'batch_summary.json'
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'n_total': len(results),
            'n_success': sum(1 for r in results if r.success),
            'n_failed': sum(1 for r in results if not r.success),
            'results': [r.to_dict() for r in results]
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"\nBatch summary saved to {summary_file}")


# =============================================================================
# Convenience Functions
# =============================================================================

def process_station(
    site_id: str,
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    **kwargs
) -> Dict[str, ProcessingResult]:
    """
    Convenience function to process a single station.
    
    Parameters
    ----------
    site_id : str
        Station identifier.
    input_dir : str or Path
        Input directory.
    output_dir : str or Path
        Output directory.
    **kwargs
        Additional arguments passed to Pipeline constructor.
    
    Returns
    -------
    dict
        Processing results for each data type.
    """
    pipeline = Pipeline(**kwargs)
    return pipeline.process_station(site_id, input_dir, output_dir)


def batch_process(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    **kwargs
) -> List[ProcessingResult]:
    """
    Convenience function for batch processing.
    
    Parameters
    ----------
    input_dir : str or Path
        Input directory.
    output_dir : str or Path
        Output directory.
    **kwargs
        Additional arguments passed to Pipeline constructor.
    
    Returns
    -------
    list of ProcessingResult
        Results for all processed files.
    """
    pipeline = Pipeline(**kwargs)
    return pipeline.batch_process(input_dir, output_dir)


# =============================================================================
# Command-Line Interface
# =============================================================================

def main():
    """Command-line interface for the pipeline."""
    parser = argparse.ArgumentParser(
        description='Process micrometeorological data with Micromet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  python -m micromet.pipeline --input data/US-UTW_Flux.dat --output results/
  
  # Process a single station (all data types)
  python -m micromet.pipeline --site US-UTW --input data/ --output results/
  
  # Batch process all files
  python -m micromet.pipeline --batch --input data/ --output results/
  
  # With custom settings
  python -m micromet.pipeline --site US-UTW --input data/ --output results/ \\
      --no-timestamp-check --keep-soil --format parquet
        """
    )
    
    # Input/output
    parser.add_argument('--input', '-i', required=True,
                       help='Input file or directory')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory')
    
    # Processing mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--site', '-s',
                           help='Process specific station (e.g., US-UTW)')
    mode_group.add_argument('--batch', '-b', action='store_true',
                           help='Batch process all files in directory')
    
    # Data type
    parser.add_argument('--data-type', '-t', choices=['eddy', 'met'], default='eddy',
                       help='Data type to process (default: eddy)')
    
    # Configuration options
    parser.add_argument('--no-timestamp-check', action='store_true',
                       help='Disable timestamp alignment analysis (faster)')
    parser.add_argument('--keep-soil', action='store_true',
                       help='Keep all soil sensor columns')
    parser.add_argument('--no-reports', action='store_true',
                       help='Skip generating QA/QC reports')
    parser.add_argument('--plots', action='store_true',
                       help='Generate diagnostic plots')
    parser.add_argument('--format', choices=['csv', 'parquet', 'feather'],
                       default='csv', help='Output file format (default: csv)')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output (DEBUG level)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet output (WARNING level only)')
    
    args = parser.parse_args()
    
    # Set up logging
    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    
    logging.basicConfig(
        level=level,
        format='%(levelname)s [%(asctime)s] %(name)s – %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create pipeline configuration
    config = PipelineConfig(
        check_timestamps=not args.no_timestamp_check,
        drop_soil=not args.keep_soil,
        generate_reports=not args.no_reports,
        generate_plots=args.plots,
        output_format=args.format
    )
    
    # Create pipeline
    pipeline = Pipeline(config=config)
    
    # Run appropriate processing mode
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if args.site:
        # Process single station
        results = pipeline.process_station(
            site_id=args.site,
            input_dir=input_path,
            output_dir=output_path
        )
        
    elif args.batch or input_path.is_dir():
        # Batch process
        results = pipeline.batch_process(
            input_dir=input_path,
            output_dir=output_path,
            data_type=args.data_type
        )
        
    else:
        # Process single file
        result = pipeline.process_file(
            input_file=input_path,
            output_dir=output_path,
            data_type=args.data_type
        )
        print("\n" + result.summary())


if __name__ == '__main__':
    main()