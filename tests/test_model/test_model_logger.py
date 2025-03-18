import pytest
import os
import logging
import tempfile
import shutil
from unittest import mock
from baseballcv.model.utils.model_logger import ModelLogger

class TestModelLogger:
    """
    Test suite for the ModelLogger class.
    
    This class contains tests to verify the functionality of the ModelLogger,
    including initialization, log directory creation, and logging behavior.
    """

    @pytest.fixture
    def setup_model_logger(self) -> dict:
        """
        Set up a minimal environment for testing the ModelLogger.
        
        Creates a temporary directory structure and returns a dictionary
        with test parameters for ModelLogger initialization.
        
        Returns:
            dict: Configuration parameters for ModelLogger testing including
                 model_name, model_run_path, model_id, batch_size, device, and temp_dir.
        """
        temp_dir = tempfile.mkdtemp()
        model_run_path = os.path.join(temp_dir, "model_run")
        os.makedirs(model_run_path, exist_ok=True)
        
        return {
            'model_name': "TestModel",
            'model_run_path': model_run_path,
            'model_id': "test/model-id",
            'batch_size': 4,
            'device': "cpu",
            'temp_dir': temp_dir
        }
        
    def teardown_method(self, method) -> None:
        """
        Clean up temporary files after each test method execution.
        
        Removes any temporary directories created during testing to ensure
        a clean state for subsequent tests.
        
        Args:
            method: The test method that was executed.
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, dict) and 'temp_dir' in attr:
                if os.path.exists(attr['temp_dir']):
                    shutil.rmtree(attr['temp_dir'])

    def test_logger_functionality(self, setup_model_logger, monkeypatch) -> None:
        """
        Test the core functionality of the ModelLogger.
        
        Verifies that the logger initializes correctly, creates the necessary
        log directories, and logs the expected initialization messages.
        
        Args:
            setup_model_logger: Fixture providing test configuration.
            monkeypatch: Pytest fixture for patching objects during testing.
        """
        mock_logger = mock.MagicMock()
        
        def get_logger_mock(*args, **kwargs):
            return mock_logger
            
        monkeypatch.setattr(logging, 'getLogger', get_logger_mock)
        
        logger = ModelLogger(
            model_name=setup_model_logger['model_name'],
            model_run_path=setup_model_logger['model_run_path'],
            model_id=setup_model_logger['model_id'],
            batch_size=setup_model_logger['batch_size'],
            device=setup_model_logger['device']
        )
        
        result = logger.orig_logging()
        
        assert result == mock_logger, "Should return the logger instance"
        
        log_dir = os.path.join(setup_model_logger['model_run_path'], "logs")
        assert os.path.exists(log_dir), "Log directory should be created"
        
        mock_logger.info.assert_any_call(f"Initializing {setup_model_logger['model_name']} model with Batch Size: {setup_model_logger['batch_size']}")
        mock_logger.info.assert_any_call(f"Device: {setup_model_logger['device']}")