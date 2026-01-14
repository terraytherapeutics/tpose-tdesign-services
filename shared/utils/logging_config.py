"""
Logging configuration for TPose services.
"""

import logging
import logging.config
import sys
from typing import Dict, Any, Optional


def setup_logging(level: str = "INFO",
                 log_format: Optional[str] = None,
                 include_timestamp: bool = True,
                 log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration for TPose services.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string
        include_timestamp: Whether to include timestamp in logs
        log_file: Optional file to write logs to
    """

    if log_format is None:
        if include_timestamp:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            log_format = "%(name)s - %(levelname)s - %(message)s"

    # Base configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': log_format,
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
        },
        'handlers': {
            'console': {
                'level': level,
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': sys.stdout
            },
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console'],
                'level': level,
                'propagate': False
            },
            'boto3': {
                'level': 'WARNING',
                'propagate': True
            },
            'botocore': {
                'level': 'WARNING',
                'propagate': True
            },
            'urllib3': {
                'level': 'WARNING',
                'propagate': True
            }
        }
    }

    # Add file handler if specified
    if log_file:
        config['handlers']['file'] = {
            'level': level,
            'class': 'logging.FileHandler',
            'formatter': 'standard',
            'filename': log_file,
            'mode': 'a'
        }
        config['loggers']['']['handlers'].append('file')

    logging.config.dictConfig(config)
    logging.info(f"Logging configured with level: {level}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggingContext:
    """Context manager for temporary logging configuration."""

    def __init__(self, level: str):
        self.level = level
        self.original_level = None

    def __enter__(self):
        root_logger = logging.getLogger()
        self.original_level = root_logger.level
        root_logger.setLevel(getattr(logging, self.level.upper()))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_level is not None:
            logging.getLogger().setLevel(self.original_level)
