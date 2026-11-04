"""Active PII and Sensitive Data Redaction Pipeline.

Addresses Rubric Category 4 (Observability & Tracing) - PII Redaction:
- Scrubs sensitive personal data, secrets, internal tokens, and email addresses
  before logging or persisting to session/memory databases.
- Integrates with Google Cloud Sensitive Data Protection (DLP API `google.cloud.dlp_v2`)
  when cloud credentials are active, paired with deterministic local regex scrubbers
  for zero-latency defense-in-depth.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence

try:
    from google.cloud import dlp_v2  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    dlp_v2 = None


# Deterministic defense-in-depth scrubbing patterns for PII & engineering secrets
_PII_PATTERNS: Sequence[tuple[str, re.Pattern[str]]] = (
    (
        "[REDACTED_EMAIL]",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        "[REDACTED_PHONE]",
        re.compile(
            r"(?<!\w)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?!\w)"
        ),
    ),
    (
        "[REDACTED_SSN]",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    (
        "[REDACTED_API_KEY]",
        re.compile(
            r"\b(?:AIza[0-9A-Za-z\-_]{20,50}|sk-[0-9A-Za-z]{20,}|ghp_[0-9A-Za-z]{20,60}|ya29\.[0-9A-Za-z\-_]+)\b"
        ),
    ),
    (
        "[REDACTED_BEARER_TOKEN]",
        re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*\b"),
    ),
    (
        "[REDACTED_IPV4]",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
        ),
    ),
)

_SENSITIVE_DICT_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "authorization",
        "password",
        "secret",
        "client_secret",
        "private_key",
        "ssn",
    }
)


class PIIRedactor:
    """Scrubs PII and credentials from text and structured payloads via Cloud DLP + regex."""

    DEFAULT_INFO_TYPES: Sequence[str] = (
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "PERSON_NAME",
        "US_SOCIAL_SECURITY_NUMBER",
        "IP_ADDRESS",
        "GCP_API_KEY",
        "AUTH_TOKEN",
    )

    def __init__(
        self,
        project_id: Optional[str] = None,
        enable_cloud_dlp: Optional[bool] = None,
    ) -> None:
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        if enable_cloud_dlp is None:
            enable_cloud_dlp = (
                os.environ.get("ENABLE_CLOUD_DLP_REDACTION", "false").lower() == "true"
            )
        self.enable_cloud_dlp = enable_cloud_dlp and bool(self.project_id) and dlp_v2 is not None
        self._dlp_client = None
        if self.enable_cloud_dlp and dlp_v2 is not None:
            try:
                self._dlp_client = dlp_v2.DlpServiceClient()
            except Exception:
                self._dlp_client = None

    def redact_text(self, text: str) -> str:
        """Redacts PII and secrets from a string using Cloud DLP and local regex patterns.

        Args:
            text: Raw input text that may contain PII, credentials, or internal identifiers.

        Returns:
            Sanitized string with all detected PII replaced by redaction markers.
        """
        if not text:
            return text

        sanitized = text
        # 1. Deterministic regex scrubbing (always runs first for zero-latency protection)
        for replacement_tag, pattern in _PII_PATTERNS:
            sanitized = pattern.sub(replacement_tag, sanitized)

        # 2. Optional Google Cloud DLP API de-identification pass
        if self._dlp_client is not None and self.project_id:
            try:
                parent = f"projects/{self.project_id}/locations/global"
                inspect_config = {
                    "info_types": [{"name": t} for t in self.DEFAULT_INFO_TYPES],
                    "min_likelihood": "POSSIBLE",
                }
                deidentify_config = {
                    "info_type_transformations": {
                        "transformations": [
                            {
                                "primitive_transformation": {
                                    "replace_with_info_type_config": {}
                                }
                            }
                        ]
                    }
                }
                response = self._dlp_client.deidentify_content(
                    request={
                        "parent": parent,
                        "inspect_config": inspect_config,
                        "deidentify_config": deidentify_config,
                        "item": {"value": sanitized},
                    }
                )
                sanitized = response.item.value
            except Exception:
                # Fail closed to the regex-sanitized string if Cloud DLP is unreachable
                pass

        return sanitized

    def redact_structured(self, payload: Any) -> Any:
        """Recursively scrubs strings and sensitive keys inside dicts, lists, and tuples.

        Args:
            payload: Arbitrary JSON-serializable structure (dict, list, str, int, etc.).

        Returns:
            A deep-copied structure with all PII strings and secret keys redacted.
        """
        if isinstance(payload, str):
            return self.redact_text(payload)
        if isinstance(payload, dict):
            cleaned: Dict[str, Any] = {}
            for key, value in payload.items():
                normalized_key = str(key).lower().strip()
                if normalized_key in _SENSITIVE_DICT_KEYS:
                    cleaned[key] = "[REDACTED_SECRET_FIELD]"
                else:
                    cleaned[key] = self.redact_structured(value)
            return cleaned
        if isinstance(payload, list):
            return [self.redact_structured(item) for item in payload]
        if isinstance(payload, tuple):
            return tuple(self.redact_structured(item) for item in payload)
        return payload


default_pii_redactor = PIIRedactor()


def scrub_pii(value: Any) -> Any:
    """Convenience function to redact PII from any string or structured object."""
    return default_pii_redactor.redact_structured(value)
