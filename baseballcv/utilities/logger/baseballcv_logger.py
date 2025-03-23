import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

class BaseballCVLogger:
    """
    Comprehensive logger for BaseballCV applications.
    
    Features:
    - Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Console and file logging
    - Module-specific logging
    - Contextual information (timestamps, module names, line numbers)
    - Exception logging with stack traces
    - Performance timing
    - Log rotation
    """
    
    _loggers: Dict[str, 'BaseballCVLogger'] = {}
    
    DEFAULT_LOG_DIR = os.path.join(os.path.expanduser("~"), ".baseballcv", "logs")
    
    LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    def __init__(
        self, 
        name: str = "BaseballCV",
        level: str = "INFO",
        log_to_console: bool = True,
        log_to_file: bool = False,
        log_dir: str = None,
        log_file: str = None,
        max_file_size: int = 10,
        backup_count: int = 5,
    ):
        """
        Initialize a new BaseballCVLogger.
        
        Args:
            name: Logger name, typically the module name (e.g., 'baseballcv.model.detector')
            level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            log_to_console: Whether to log to console
            log_to_file: Whether to log to file
            log_dir: Directory to store log files
            log_file: Specific log file name
            max_file_size: Maximum size of log file before rotation (in MB)
            backup_count: Number of backup log files to keep
            format_string: Custom log format string
        """
        
        self.name = name
        
        if name in BaseballCVLogger._loggers:
            self.logger = BaseballCVLogger._loggers[name].logger
            return
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.LEVELS.get(level.upper(), logging.INFO))
        
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(format)
        
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        if log_to_file:
            log_dir = self.DEFAULT_LOG_DIR
            os.makedirs(log_dir, exist_ok=True)
            
            if log_file is None:
                timestamp = datetime.now().strftime("%Y%m%d")
                module_name = name.split('.')[-1]
                log_file = f"{module_name}_{timestamp}.log"
            
            log_path = os.path.join(log_dir, log_file)
            
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=max_file_size * 1024 * 1024,
                backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        BaseballCVLogger._loggers[name] = self
        
        self.timers = {}
    
    @classmethod
    def get_logger(cls, name: str = None, **kwargs) -> 'BaseballCVLogger':
        """
        Get or create a logger instance.
        
        Args:
            name: Logger name
            **kwargs: Additional arguments to pass to the constructor
            
        Returns:
            BaseballCVLogger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
        return cls(name=name, **kwargs)
    
    def debug(self, message: Any, *args, **kwargs):
        """Log a debug message."""
        return self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: Any, *args, **kwargs):
        """Log an info message."""
        return self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: Any, *args, **kwargs):
        """Log a warning message."""
        return self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: Any, *args, **kwargs):
        """Log an error message."""
        return self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: Any, *args, **kwargs):
        """Log a critical message."""
        return self.logger.critical(message, *args, **kwargs)
    
    def exception(self, message: Any, *args, exc_info=True, **kwargs):
        """Log an exception with traceback."""
        return self.logger.exception(message, *args, exc_info=exc_info, **kwargs)
    
    def log_exception(self, e: Exception, message: str = None):
        """
        Log an exception with detailed traceback.
        
        Args:
            e: The exception to log
            message: Optional message to include
        """
        if message:
            error_message = f"{message}: {str(e)}"
        else:
            error_message = str(e)
        
        return self.error(f"{error_message}\n{traceback.format_exc()}")
    
    def timer(self, name: str, start: bool = True, log_level: str = "DEBUG") -> float:
        """
        Start or stop a timer for performance measurement.
        
        Args:
            name: Timer name
            start: True to start timer, False to stop and return elapsed time
            log_level: Log level to use for the message when stopping
            
        Returns:
            Elapsed time in seconds when stopping, 0 when starting
        """
        if start:
            self.timers[name] = time.time()
            self.debug(f"Timer '{name}' started")
            return 0
        else:
            if name not in self.timers:
                self.warning(f"Timer '{name}' was never started")
                return 0
            
            elapsed = time.time() - self.timers[name]
            log_method = getattr(self, log_level.lower(), self.debug)
            log_method(f"Timer '{name}' stopped after {elapsed:.4f} seconds")
            
            del self.timers[name]
            return elapsed
    
    def log(self, message: Any, level: str = "INFO", *args, **kwargs):
        """
        Log a message with the specified level.
        
        Args:
            message: Message to log
            level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        log_method = getattr(self, level.lower(), self.info)
        log_method(message, *args, **kwargs)
    
    def set_level(self, level: str):
        """
        Set the logger level.
        
        Args:
            level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        self.logger.setLevel(self.LEVELS.get(level.upper(), logging.INFO))
    
    def __call__(self, message: Any, level: str = "INFO", *args, **kwargs):
        """
        Call the logger directly with a specified level.
        
        Args:
            message: Message to log
            level: Log level
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        self.log(message, level, *args, **kwargs)
