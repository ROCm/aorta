"""
Logging configuration for Weekly CI Kickoff.

This module provides:
- Colored console output with ANSI codes
- Dual logging to console and file
- Timestamp-based log file naming
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter with ANSI color codes for console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[31;1m",  # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors.

        Args:
            record: The log record to format.

        Returns:
            Formatted string with ANSI color codes.
        """
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


class StageFormatter(logging.Formatter):
    """Formatter that adds stage context to log messages."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with stage prefix if available.

        Args:
            record: The log record to format.

        Returns:
            Formatted string with stage prefix.
        """
        if hasattr(record, "stage"):
            record.msg = f"[{record.stage}] {record.msg}"
        return super().format(record)


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> logging.Logger:
    """Configure logging to console and file.

    Sets up:
    - Console handler with colored output
    - File handler with plain text (includes DEBUG level)
    - Timestamp-based log filename

    Args:
        log_dir: Directory to store log files.
        log_level: Minimum log level for console output.

    Returns:
        Configured logger instance.
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"weekly_ci_{timestamp}.log"

    # Create logger
    logger = logging.getLogger("weekly_ci")
    logger.setLevel(logging.DEBUG)  # Capture all levels, filter at handler

    # Remove any existing handlers
    logger.handlers.clear()

    # Console handler (colored, respects log_level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_format = "%(asctime)s - %(levelname)s - %(message)s"
    console_handler.setFormatter(ColoredFormatter(console_format))
    logger.addHandler(console_handler)

    # File handler (plain, DEBUG level for full trace)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    file_handler.setFormatter(logging.Formatter(file_format))
    logger.addHandler(file_handler)

    logger.info(f"Log file: {log_file}")

    return logger


def log_stage_start(logger: logging.Logger, stage_name: str) -> None:
    """Log the start of a pipeline stage.

    Args:
        logger: Logger instance.
        stage_name: Name of the stage being started.
    """
    logger.info("=" * 60)
    logger.info(f"STAGE: {stage_name}")
    logger.info("=" * 60)


def log_stage_skip(logger: logging.Logger, stage_name: str) -> None:
    """Log that a stage is being skipped.

    Args:
        logger: Logger instance.
        stage_name: Name of the stage being skipped.
    """
    logger.info(f"⏭️  Skipping stage: {stage_name}")


def log_stage_complete(logger: logging.Logger, stage_name: str) -> None:
    """Log the successful completion of a stage.

    Args:
        logger: Logger instance.
        stage_name: Name of the completed stage.
    """
    logger.info(f"✅ Stage complete: {stage_name}")


def log_stage_error(logger: logging.Logger, stage_name: str, error: str) -> None:
    """Log an error in a pipeline stage.

    Args:
        logger: Logger instance.
        stage_name: Name of the stage where error occurred.
        error: Error message or description.
    """
    logger.error(f"❌ Stage failed: {stage_name}")
    logger.error(f"   Error: {error}")

