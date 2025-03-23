import pytest
import logging
import os
import time
from unittest.mock import patch
from baseballcv.utilities import BaseballCVLogger

class TestBaseballCVLogger:
    """
    Test class for the BaseballCVLogger utility.
    """
    def test_logger_initialization(self, reset_logger_registry):
        """
        Test the basic initialization and properties of the BaseballCVLogger.
        
        Verifies that a new logger is properly initialized with the correct name,
        logger instance, default log level, and appropriate handlers.
        
        Args:
            reset_logger_registry: Fixture to reset the logger registry
        """
        logger = BaseballCVLogger("TestLogger")
        
        assert logger.name == "TestLogger"
        assert isinstance(logger.logger, logging.Logger)
        assert logger.logger.level == logging.INFO
        assert len(logger.logger.handlers) > 0
        assert isinstance(logger.logger.handlers[0], logging.StreamHandler)
        
    def test_get_logger_singleton(self, reset_logger_registry):
        """
        Test the singleton behavior of the get_logger method to ensure that calling it with the same name returns the same instance, while calling
        it with different names returns different instances.
        
        Args:
            reset_logger_registry: Fixture to reset the logger registry
        """
        logger1 = BaseballCVLogger.get_logger("TestSingleton")
        logger2 = BaseballCVLogger.get_logger("TestSingleton")
        logger3 = BaseballCVLogger.get_logger("DifferentLogger")
        
        assert logger1 is logger2
        assert logger1 is not logger3
        
    def test_log_levels(self, reset_logger_registry):
        """
        Test that all log level methods work as expected.
        
        Verifies that each log level method (debug, info, warning, error, critical)
        correctly calls the corresponding method on the underlying logger.
        
        Args:
            reset_logger_registry: Fixture to reset the logger registry
        """
        logger = BaseballCVLogger("TestLevels", level="DEBUG")
        
        with patch.object(logger.logger, 'debug') as mock_debug, \
             patch.object(logger.logger, 'info') as mock_info, \
             patch.object(logger.logger, 'warning') as mock_warning, \
             patch.object(logger.logger, 'error') as mock_error, \
             patch.object(logger.logger, 'critical') as mock_critical:
            
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
            
            mock_debug.assert_called_once_with("Debug message")
            mock_info.assert_called_once_with("Info message")
            mock_warning.assert_called_once_with("Warning message")
            mock_error.assert_called_once_with("Error message")
            mock_critical.assert_called_once_with("Critical message")
    
    def test_set_level(self, reset_logger_registry):
        """
        Test the set_level method for changing log levels.
        
        Verifies that the set_level method correctly changes the log level
        of the underlying logger.
        
        Args:
            reset_logger_registry: Fixture to reset the logger registry
        """
        logger = BaseballCVLogger("TestSetLevel", level="INFO")
        assert logger.logger.level == logging.INFO
        
        logger.set_level("DEBUG")
        assert logger.logger.level == logging.DEBUG
        
        logger.set_level("WARNING")
        assert logger.logger.level == logging.WARNING
    
    def test_timer_functionality(self, reset_logger_registry):
        """
        Test the timer functionality of the logger.
        
        Verifies that the timer method correctly starts and stops timers,
        and returns the elapsed time.
        
        Args:
            reset_logger_registry: Fixture to reset the logger registry
        """
        logger = BaseballCVLogger("TimerTest")
        
        start_result = logger.timer("test_timer")
        assert start_result == 0
        assert "test_timer" in logger.timers
        
        time.sleep(0.1)
        
        with patch.object(logger.logger, 'debug') as mock_debug:
            elapsed = logger.timer("test_timer", start=False)
            assert elapsed > 0
            assert "test_timer" not in logger.timers
            assert mock_debug.call_count == 1
    
    def test_nonexistent_timer(self, reset_logger_registry):
        """
        Test the behavior when stopping a timer that was never started.
        
        Verifies that attempting to stop a nonexistent timer returns 0
        and logs a warning.
        
        Args:
            reset_logger_registry: Fixture to reset the logger registry
        """
        logger = BaseballCVLogger("TimerTestNonexistent")
        
        with patch.object(logger.logger, 'warning') as mock_warning:
            result = logger.timer("nonexistent_timer", start=False)
            assert result == 0
            mock_warning.assert_called_once()
    
    def test_log_exception(self, reset_logger_registry):
        """
        Test the log_exception method for logging exceptions with traceback.
        
        Verifies that the log_exception method correctly logs the exception
        message and traceback.
        
        Args:
            reset_logger_registry: Fixture to reset the logger registry
        """
        logger = BaseballCVLogger("ExceptionTest")
        
        with patch.object(logger.logger, 'error') as mock_error:
            try:
                1 / 0
            except Exception as e:
                logger.log_exception(e, "Division error")
                
            mock_error.assert_called_once()
            assert "Division error: division by zero" in mock_error.call_args[0][0]
            assert "Traceback" in mock_error.call_args[0][0]
    
    @pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_logger_respects_levels(self, reset_logger_registry, log_level):
        """
        Test that the logger respects its level setting.
        
        Verifies that messages are only logged if their level is at or above
        the logger's configured level.
        
        Args:
            reset_logger_registry: Fixture to reset the logger registry
            log_level: The log level to test
        """
        numeric_level = BaseballCVLogger.LEVELS[log_level]
        logger = BaseballCVLogger("LevelTest", level=log_level)
        
        with patch.object(logger.logger, 'log') as mock_log:
            for level_name, level_value in BaseballCVLogger.LEVELS.items():
                if hasattr(logger, level_name.lower()):
                    getattr(logger, level_name.lower())("Test message")
            
            for call in mock_log.call_args_list:
                call_level = call[0][0]
                assert call_level >= numeric_level
    
    def test_file_logging(self, reset_logger_registry):
        """
        Test file logging functionality.
        
        Verifies that when log_to_file is enabled, the logger correctly creates
        a file handler and writes log messages to the specified file.
        
        Args:
            reset_logger_registry: Fixture to reset the logger registry
        """
        log_file = os.path.join(os.getcwd(), "test.log")
        logger = BaseballCVLogger(
            "FileLogger", 
            log_to_file=True, 
            log_file=str(log_file)
        )
        
        assert len(logger.logger.handlers) >= 2
        
        file_handlers = [h for h in logger.logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        
        assert file_handlers[0].baseFilename == str(log_file)
        
        logger.info("File log test")
        
        assert os.path.exists(log_file)
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        assert "File log test" in log_content
