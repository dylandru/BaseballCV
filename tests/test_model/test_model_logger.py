import pytest
import os
import logging
import tempfile
import shutil
from unittest import mock
from baseballcv.model.utils.model_logger import ModelLogger

class TestModelLogger:
    """Test only essential functionality of ModelLogger."""

    @pytest.fixture
    def setup(self):
        """Minimal setup for testing."""
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
        
    def teardown_method(self, method):
        """Clean up temporary files."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, dict) and 'temp_dir' in attr:
                if os.path.exists(attr['temp_dir']):
                    shutil.rmtree(attr['temp_dir'])

    def test_logger_functionality(self, setup, monkeypatch):
        """Test that logger initializes and creates log directories properly."""
        mock_logger = mock.MagicMock()
        monkeypatch.setattr(logging, 'getLogger', lambda name: mock_logger)
        
        logger = ModelLogger(
            model_name=setup['model_name'],
            model_run_path=setup['model_run_path'],
            model_id=setup['model_id'],
            batch_size=setup['batch_size'],
            device=setup['device']
        )
        
        result = logger.orig_logging()
        
        assert result == mock_logger, "Should return the logger instance"
        
        log_dir = os.path.join(setup['model_run_path'], "logs")
        assert os.path.exists(log_dir), "Log directory should be created"
        
        mock_logger.info.assert_any_call(f"Initializing {setup['model_name']} model with Batch Size: {setup['batch_size']}")
        mock_logger.info.assert_any_call(f"Device: {setup['device']}")