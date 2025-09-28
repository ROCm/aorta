"""Logging utilities for the AORTA toolkit."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None, *, rank: int = 0) -> None:
    """Configure structured logging for the current process.

    Args:
        level: Log level name.
        log_file: Optional path to a file where logs should be appended.
        rank: Distributed process rank. Rank>0 suppresses stdout handler.
    """

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers to avoid duplicate logs when reinitializing.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT)

    if rank == 0:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


__all__ = ["setup_logging"]
