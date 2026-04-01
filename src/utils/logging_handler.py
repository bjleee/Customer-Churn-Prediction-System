"""Utilities for configuring project logging."""

import logging
from datetime import datetime
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_LOG_LEVEL = logging.INFO


def _build_log_file_path() -> Path:
    """Builds a timestamped log file path under the project log directory.

    Returns:
        Absolute path to a new log file.
    """
    project_root = Path(__file__).resolve().parents[2]
    logs_dir = project_root / "log"
    logs_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    return logs_dir / file_name


def configure_logger(name: str = "customer_churn") -> logging.Logger:
    """Creates and configures a logger with file output.

    Args:
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(_LOG_LEVEL)

    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(_build_log_file_path(), encoding="utf-8")
    file_handler.setLevel(_LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


LOGGER = configure_logger()
