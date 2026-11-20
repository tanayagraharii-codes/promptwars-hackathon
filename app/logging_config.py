"""Structured JSON logging that Google Cloud Logging parses natively on Cloud Run.

Every line is one JSON object using Cloud Logging's special fields
(``severity``, ``logging.googleapis.com/trace``, ...). When a request carries
Cloud Run's ``X-Cloud-Trace-Context`` header, its trace ID is attached to every
log line written while handling it, so Cloud Logging groups a request's logs
together and links them to Cloud Trace.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

TRACE_HEADER = "x-cloud-trace-context"
NOISY_LOGGERS = ("httpx", "httpx2", "httpcore", "google_genai")

_STANDARD_ATTRS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


@dataclass(frozen=True)
class TraceContext:
    """A Cloud Trace reference parsed from ``TRACE_ID/SPAN_ID;o=OPTIONS``."""

    trace_id: str
    span_id: str | None = None
    sampled: bool = False

    @classmethod
    def parse(cls, header: str | None) -> TraceContext | None:
        """Parse the header value; malformed or missing headers yield ``None``."""
        if not header:
            return None
        trace_part, _, options = header.partition(";")
        trace_id, _, span_id = trace_part.partition("/")
        if not trace_id.isalnum() or len(trace_id) != 32:
            return None
        return cls(
            trace_id=trace_id.lower(),
            span_id=span_id if span_id.isdigit() else None,
            sampled=options.strip() == "o=1",
        )

    def log_fields(self, project: str) -> dict[str, Any]:
        """Cloud Logging fields that link a log line to this trace."""
        fields: dict[str, Any] = {
            "logging.googleapis.com/trace": f"projects/{project}/traces/{self.trace_id}",
            "logging.googleapis.com/trace_sampled": self.sampled,
        }
        if self.span_id:
            fields["logging.googleapis.com/spanId"] = self.span_id
        return fields


current_trace: ContextVar[TraceContext | None] = ContextVar("current_trace", default=None)


class CloudLoggingFormatter(logging.Formatter):
    """Format records as single-line JSON for Cloud Logging."""

    def __init__(self, project: str = "") -> None:
        super().__init__()
        self._project = project

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "logger": record.name,
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS})
        trace = current_trace.get()
        if trace is not None and self._project:
            payload.update(trace.log_fields(self._project))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(project: str = "", level: int = logging.INFO) -> None:
    """Route all logging to stdout as Cloud Logging JSON and quiet chatty libraries."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CloudLoggingFormatter(project))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
