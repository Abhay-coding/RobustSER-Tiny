"""
Structured JSON logging with correlation IDs for RobustSER-Tiny.

All log entries are JSON lines with standard fields:
- timestamp: ISO 8601 UTC
- level: DEBUG, INFO, WARNING, ERROR, CRITICAL
- logger: logger name
- message: human-readable message
- correlation_ids: dict of trace identifiers (experiment_id, run_id, epoch, seed, split, etc.)
- extra: additional structured fields
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Context variable for correlation IDs (thread-safe and async-safe)
_correlation_ids: ContextVar[dict[str, Any]] = ContextVar("correlation_ids", default={})

# Thread-local storage for logger configuration
_thread_local = threading.local()


@dataclass(frozen=True)
class LogContext:
    """Immutable correlation context for structured logging."""
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    epoch: Optional[int] = None
    seed: Optional[int] = None
    split: Optional[str] = None  # train, validation, test, noisy_test, external_test
    snr_db: Optional[float] = None
    device: Optional[str] = None
    request_id: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, omitting None values."""
        return {k: v for k, v in {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "epoch": self.epoch,
            "seed": self.seed,
            "split": self.split,
            "snr_db": self.snr_db,
            "device": self.device,
            "request_id": self.request_id,
            **self.extra,
        }.items() if v is not None}


class CorrelationFilter(logging.Filter):
    """Logging filter that injects correlation IDs into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Get correlation IDs from context variable
        correlation = _correlation_ids.get({})
        if correlation:
            record.correlation_ids = correlation
        else:
            record.correlation_ids = {}
        return True


class JSONFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        # Base log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add correlation IDs
        correlation = getattr(record, "correlation_ids", {})
        if correlation:
            log_entry["correlation_ids"] = correlation

        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "message", "msg", "pathname", "process", "processName",
                "relativeCreated", "thread", "threadName", "exc_info",
                "exc_text", "stack_info", "correlation_ids"
            }:
                extra_fields[key] = value

        if extra_fields:
            log_entry["extra"] = extra_fields

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_structured_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    json_output: bool = True,
) -> logging.Logger:
    """
    Configure structured logging for the application.

    Args:
        level: Logging level
        log_file: Optional file path for JSON log output
        json_output: If True, use JSON formatting; otherwise use human-readable format

    Returns:
        Root logger configured with structured logging
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Add correlation filter
    correlation_filter = CorrelationFilter()

    # Console handler (human-readable by default)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    if json_output:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
    console_handler.addFilter(correlation_filter)
    root_logger.addHandler(console_handler)

    # File handler (always JSON)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(JSONFormatter())
        file_handler.addFilter(correlation_filter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


def set_correlation_ids(correlation_ids: dict[str, Any]) -> None:
    """Set correlation IDs for the current context."""
    _correlation_ids.set(correlation_ids)


def get_correlation_ids() -> dict[str, Any]:
    """Get current correlation IDs."""
    return _correlation_ids.get({}).copy()


def clear_correlation_ids() -> None:
    """Clear correlation IDs for the current context."""
    _correlation_ids.set({})


class CorrelationContext:
    """Context manager for setting correlation IDs."""

    def __init__(self, context: LogContext):
        self.context = context
        self.previous: dict[str, Any] = {}

    def __enter__(self) -> LogContext:
        self.previous = get_correlation_ids()
        set_correlation_ids(self.context.to_dict())
        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        set_correlation_ids(self.previous)


def with_correlation(context: LogContext):
    """Decorator to run a function with correlation context."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with CorrelationContext(context):
                return func(*args, **kwargs)
        return wrapper
    return decorator