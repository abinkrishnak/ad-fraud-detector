"""
Logging Configuration
Sets up structured logging for the entire project
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


class LoggerSetup:
    """
    Configure logging with different levels and output formats
    
    Example usage:
        logger_setup = LoggerSetup(level="INFO", log_file="logs/training.log")
        logger_setup.get_logger()
        
        # Then anywhere in your code:
        from loguru import logger
        logger.info("Training started")
        logger.warning("Low fraud samples detected")
        logger.error("Model failed to converge")
    """
    
    def __init__(
        self,
        level: str = "INFO",
        log_file: Optional[str] = None,
        rotation: str = "10 MB",
        retention: str = "1 week"
    ):
        """
        Initialize logger configuration
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (optional, logs to console if None)
            rotation: When to rotate log files (e.g., "10 MB", "1 day")
            retention: How long to keep old logs (e.g., "1 week", "30 days")
        """
        self.level = level.upper()
        self.log_file = log_file
        self.rotation = rotation
        self.retention = retention
        
        # Remove default logger
        logger.remove()
        
        # Setup logger
        self._setup_logger()
    
    def _setup_logger(self):
        """Configure logger with console and file handlers"""
        
        # Console handler (colored, human-readable)
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=self.level,
            colorize=True
        )
        
        # File handler (if specified)
        if self.log_file:
            # Create logs directory if it doesn't exist
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                self.log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                level=self.level,
                rotation=self.rotation,
                retention=self.retention,
                compression="zip"  # Compress old logs
            )
            
            logger.info(f"Logging to file: {self.log_file}")
    
    @staticmethod
    def get_logger():
        """
        Get logger instance
        
        Returns:
            loguru.logger instance
        """
        return logger


# Convenience function for quick setup
def setup_logger(
    level: str = "INFO",
    log_file: Optional[str] = None
) -> logger: #type: ignore
    """
    Quick logger setup
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
        
    Example:
        >>> from src.utils.logger import setup_logger
        >>> logger = setup_logger(level="DEBUG", log_file="logs/training.log")
        >>> logger.info("Starting training...")
    """
    LoggerSetup(level=level, log_file=log_file)
    return logger


# Example usage patterns for different scenarios
def log_data_loading(n_samples: int, n_features: int):
    """Example: Log data loading"""
    logger.info(f"Data loaded: {n_samples:,} samples, {n_features} features")


def log_model_training(model_name: str, n_estimators: int):
    """Example: Log model training start"""
    logger.info(f"Training {model_name} with {n_estimators} estimators...")


def log_metrics(metric_name: str, value: float):
    """Example: Log model metrics"""
    logger.success(f"{metric_name}: {value:.4f}")


def log_error(error: Exception, context: str = ""):
    """Example: Log errors with context"""
    logger.error(f"Error in {context}: {str(error)}")
    logger.exception(error)  # Includes full traceback