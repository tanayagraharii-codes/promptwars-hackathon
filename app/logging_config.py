"""Structured JSON logging that Google Cloud Logging parses natively on Cloud Run."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

_STANDARD_ATTRS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class CloudLoggingFormatter(logging.Formatter):
    """Emit one JSON object per line using Cloud Logging's special fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "logger": record.name,
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CloudLoggingFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
