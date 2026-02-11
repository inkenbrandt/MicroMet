import unittest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import pandas as pd
import numpy as np
from micromet.pipeline import Pipeline, PipelineConfig, ProcessingResult

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.config = PipelineConfig(
            check_timestamps=False,
            generate_plots=False,
            generate_reports=False
        )

    def test_pipeline_init(self):
        pipeline = Pipeline(config=self.config)
        self.assertEqual(pipeline.config.check_timestamps, False)
        self.assertIsNotNone(pipeline.logger)

    @patch('micromet.pipeline.AmerifluxDataProcessor')
    @patch('micromet.pipeline.create_reformatter_from_site')
    def test_process_file_success(self, mock_create_reformatter, mock_reader_class):
        # Setup mocks
        mock_reader = mock_reader_class.return_value
        mock_df = pd.DataFrame({
            'TIMESTAMP': pd.to_datetime(['2024-01-01 00:00', '2024-01-01 00:30']),
            'DATA': [1, 2]
        })
        mock_reader.to_dataframe.return_value = mock_df

        pipeline = Pipeline(config=self.config)

        mock_reformatter = MagicMock()
        mock_df_clean = pd.DataFrame({
            'DATA': [1, 2]
        }, index=pd.to_datetime(['2024-01-01 00:00', '2024-01-01 00:30']))
        mock_limits_report = pd.DataFrame({'n_flagged': [0]})
        mock_reformatter.process.return_value = (mock_df_clean, mock_limits_report, None)
        mock_create_reformatter.return_value = mock_reformatter

        # Run process_file
        with patch.object(Path, 'exists', return_value=True):
            result = pipeline.process_file('test_US-UTW_Flux.dat', site_id='US-UTW')

        # Assertions
        self.assertTrue(result.success, msg=result.error_message)
        self.assertEqual(result.site_id, 'US-UTW')
        self.assertEqual(result.n_records_input, 2)
        self.assertEqual(result.n_records_output, 2)
        mock_reader.to_dataframe.assert_called_once()
        mock_reformatter.process.assert_called_once()

    def test_extract_site_id(self):
        pipeline = Pipeline(config=self.config)
        self.assertEqual(pipeline._extract_site_id(Path('US-UTW_Flux.dat')), 'US-UTW')
        self.assertEqual(pipeline._extract_site_id(Path('Something_US-ABC_Else.dat')), 'US-ABC')
        self.assertEqual(pipeline._extract_site_id(Path('NoSiteID.dat')), 'UNKNOWN')

    @patch('micromet.pipeline.Pipeline.process_file')
    @patch('builtins.open', new_callable=mock_open)
    def test_batch_process(self, mock_file, mock_process_file):
        # Setup mock results
        mock_result = ProcessingResult(
            site_id='US-UTW',
            success=True,
            input_file=Path('test.dat')
        )
        mock_process_file.return_value = mock_result

        pipeline = Pipeline(config=self.config)

        # Mock directory structure
        with patch.object(Path, 'rglob', return_value=[Path('file1_Flux.dat'), Path('file2_Flux.dat')]):
            with patch.object(Path, 'mkdir'):
                results = pipeline.batch_process('input_dir', 'output_dir')

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].site_id, 'US-UTW')
        mock_file.assert_called()

if __name__ == '__main__':
    unittest.main()
