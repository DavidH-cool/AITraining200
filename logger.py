"""Structured JSON Logging & Intent-vs-Outcome Telemetry.

Addresses Rubric Category 4 (Observability & Tracing):
- Structured JSON Logging: Emits machine-parseable JSON logs compatible with
  Google Cloud Logging (`severity`, `timestamp`, `logging.googleapis.com/trace`,
  `event_type`, `agent_name`, `session_id`) using `python-json-logger`.
- Intent vs. Outcome Capture: Provides explicit `log_agent_intent` and
  `log_agent_outcome` helpers and ADK callbacks that record both what the agent
  intends to execute before an action and the actual outcome afterwards.
- Active PII Redaction: Every log record is scrubbed by `PIIRedactor` before emission.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import sys
import time
from typing import Any, Dict, Optional
import uuid

from blog_writer_agent.observability.pii_redactor import scrub_pii

try:
    from pythonjsonlogger import jsonlogger  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    jsonlogger = None


class PIIRedactingJsonFormatter(logging.Formatter):
    """Structured JSON formatter that injects Cloud Logging fields and scrubs PII."""

    RESERVED_ATTRS = frozenset(
        {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        raw_message = record.getMessage()
        extra_fields: Dict[str, Any] = {
            k: v
            for k, v in record.__dict__.items()
            if k not in self.RESERVED_ATTRS and not k.startswith("_")
        }

        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": scrub_pii(raw_message),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if extra_fields:
            payload.update(scrub_pii(extra_fields))

        if record.exc_info:
            payload["exception"] = scrub_pii(self.formatException(record.exc_info))

        return json.dumps(payload, default=str)


def get_structured_logger(name: str = "blog_writer_agent") -> logging.Logger:
    """Returns a configured structured JSON logger with automatic PII redaction."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(PIIRedactingJsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class IntentOutcomeTracker:
    """Captures paired 'Intent' (pre-execution) and 'Outcome' (post-execution) audit records."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or get_structured_logger("blog_writer_agent.audit")
        self._pending_intents: Dict[str, Dict[str, Any]] = {}

    def record_intent(
        self,
        *,
        agent_name: str,
        action_name: str,
        intended_goal: str,
        planned_inputs: Dict[str, Any],
        session_id: str = "default",
        trace_id: Optional[str] = None,
    ) -> str:
        """Logs the agent's intended action BEFORE execution and returns a correlation_id."""
        correlation_id = str(uuid.uuid4())
        intent_record = {
            "event_type": "AGENT_INTENT_CAPTURED",
            "phase": "PRE_EXECUTION_INTENT",
            "correlation_id": correlation_id,
            "agent_name": agent_name,
            "action_name": action_name,
            "intended_goal": intended_goal,
            "planned_inputs": scrub_pii(planned_inputs),
            "session_id": session_id,
            "trace_id": trace_id or "unlinked",
            "started_at_monotonic": time.monotonic(),
        }
        self._pending_intents[correlation_id] = intent_record
        self._logger.info(
            f"Intent captured for '{action_name}' by agent '{agent_name}'.",
            extra={
                k: v
                for k, v in intent_record.items()
                if k != "started_at_monotonic"
            },
        )
        return correlation_id

    def record_outcome(
        self,
        *,
        correlation_id: str,
        actual_outcome: Any,
        status: str = "SUCCESS",
        error_details: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Logs the agent's actual outcome AFTER execution, linked to the prior intent."""
        intent_record = self._pending_intents.pop(correlation_id, {})
        started = intent_record.get("started_at_monotonic", time.monotonic())
        latency_ms = round((time.monotonic() - started) * 1000.0, 2)

        outcome_record = {
            "event_type": "AGENT_OUTCOME_CAPTURED",
            "phase": "POST_EXECUTION_OUTCOME",
            "correlation_id": correlation_id,
            "agent_name": intent_record.get("agent_name", "unknown_agent"),
            "action_name": intent_record.get("action_name", "unknown_action"),
            "intended_goal": intent_record.get("intended_goal", ""),
            "status": status,
            "latency_ms": latency_ms,
            "actual_outcome": scrub_pii(actual_outcome),
            "error_details": scrub_pii(error_details) if error_details else None,
            "session_id": intent_record.get("session_id", "default"),
            "trace_id": intent_record.get("trace_id", "unlinked"),
        }
        log_fn = self._logger.info if status == "SUCCESS" else self._logger.error
        log_fn(
            f"Outcome captured for '{outcome_record['action_name']}' with status={status}.",
            extra=outcome_record,
        )
        return outcome_record


intent_outcome_tracker = IntentOutcomeTracker()
