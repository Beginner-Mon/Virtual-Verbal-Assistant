"""Logging configuration for Agentic RAG system."""

import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger as loguru_logger

from config import get_config


def setup_logging(
    log_level: str = None,
    log_file: str = None,
    enable_json: bool = False
) -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, uses config default.
        enable_json: Whether to use JSON formatting
    """
    config = get_config().logging
    
    # Use provided values or defaults from config
    level = log_level or config.level
    output_file = log_file or config.output_file
    
    # Remove default logger
    loguru_logger.remove()
    
    # Setup console logging
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    if enable_json or config.format == "json":
        # JSON format for production
        loguru_logger.add(
            sys.stderr,
            format="{time} {level} {name} {function} {line} {message}",
            level=level,
            serialize=True
        )
    else:
        # Human-readable format for development
        loguru_logger.add(
            sys.stderr,
            format=log_format,
            level=level,
            colorize=True
        )
    
    # Setup file logging
    if output_file:
        # Ensure log directory exists
        log_path = Path(output_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add file handler
        loguru_logger.add(
            output_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation="10 MB",  # Rotate when file reaches 10MB
            retention="30 days",  # Keep logs for 30 days
            compression="zip",  # Compress rotated logs
            serialize=enable_json or config.format == "json"
        )
    
    loguru_logger.info(f"Logging initialized: level={level}, file={output_file}")


def get_logger(name: str) -> loguru_logger:
    """Get a logger instance for a module.
    
    Args:
        name: Logger name (typically __name__ of the module)
        
    Returns:
        Logger instance
    """
    return loguru_logger.bind(name=name)


class LogContext:
    """Context manager for logging with additional context."""
    
    def __init__(self, logger, **context):
        """Initialize log context.
        
        Args:
            logger: Logger instance
            **context: Context key-value pairs
        """
        self.logger = logger
        self.context = context
    
    def __enter__(self):
        """Enter context."""
        # Bind context to logger
        self.bound_logger = self.logger.bind(**self.context)
        return self.bound_logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if exc_type is not None:
            self.bound_logger.error(
                f"Exception in context: {exc_type.__name__}: {exc_val}"
            )
        return False


# Initialize logging on module import
try:
    setup_logging()
except Exception as e:
    # Fallback to basic logging if config fails
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.warning(f"Failed to setup loguru, using basic logging: {str(e)}")


if __name__ == "__main__":
    # Test logging
    logger = get_logger(__name__)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test context logging
    with LogContext(logger, user_id="test_123", session_id="session_456") as ctx_logger:
        ctx_logger.info("Processing query with context")
    
    print("\nLogging test complete. Check logs/ directory for output file.")
