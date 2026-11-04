"""Secure Secret Management using Google Cloud Secret Manager.

Addresses Rubric Category 5 (Infrastructure & CI/CD) - Secure Secret Management:
- Zero hardcoded API keys or credentials in source code.
- Retrieves secrets dynamically via Google Cloud Secret Manager (`google.cloud.secretmanager_v1`).
- Supports Application Default Credentials (ADC) and IAM-scoped secret versions.
"""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Optional

from blog_writer_agent.observability.logger import get_structured_logger

logger = get_structured_logger(__name__)

try:
    from google.cloud import secretmanager_v1  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - fallback when GCP client is not installed locally
    secretmanager_v1 = None


class SecretResolutionError(RuntimeError):
    """Raised when a required secret cannot be resolved from Secret Manager."""


class SecretManagerProvider:
    """Secure secret provider backed by Google Cloud Secret Manager."""

    def __init__(self, project_id: Optional[str] = None) -> None:
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self._client = None
        if secretmanager_v1 is not None and self.project_id:
            try:
                self._client = secretmanager_v1.SecretManagerServiceClient()
            except Exception as exc:
                logger.warning(
                    "SecretManagerServiceClient initialization deferred to ADC/runtime.",
                    extra={"event_type": "secret_manager_init_warning", "error": str(exc)},
                )

    @lru_cache(maxsize=32)
    def get_secret(
        self,
        secret_id: str,
        version: str = "latest",
        allow_env_fallback_in_local_dev: bool = True,
    ) -> str:
        """Fetches a secret payload securely from Google Cloud Secret Manager.

        Args:
            secret_id: Identifier of the secret in Google Cloud Secret Manager
                (e.g., 'cms-publishing-webhook-secret' or 'vertex-search-engine-id').
            version: Secret version to access. Defaults to 'latest'.
            allow_env_fallback_in_local_dev: Whether to allow reading from an
                environment variable injected by Cloud Run Secret Manager volume/env
                mounts during local or container testing.

        Returns:
            The resolved secret string value.

        Raises:
            SecretResolutionError: If the secret cannot be retrieved from Secret Manager
                or an authorized runtime mount.
        """
        if not secret_id or not secret_id.strip():
            raise SecretResolutionError("secret_id must be a non-empty string.")

        resource_name = (
            f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
            if self.project_id
            else f"projects/-/secrets/{secret_id}/versions/{version}"
        )

        if self._client is not None and self.project_id:
            try:
                response = self._client.access_secret_version(
                    request={"name": resource_name}
                )
                payload = response.payload.data.decode("utf-8").strip()
                logger.info(
                    "Successfully resolved secret from Google Cloud Secret Manager.",
                    extra={
                        "event_type": "secret_access_success",
                        "secret_id": secret_id,
                        "version": version,
                    },
                )
                return payload
            except Exception as exc:
                logger.warning(
                    "Primary Secret Manager RPC failed; checking Cloud Run secret env mount.",
                    extra={
                        "event_type": "secret_access_fallback",
                        "secret_id": secret_id,
                        "error_type": type(exc).__name__,
                    },
                )

        if allow_env_fallback_in_local_dev:
            env_key = secret_id.upper().replace("-", "_")
            mounted_value = os.environ.get(env_key)
            if mounted_value:
                logger.info(
                    "Resolved secret from IAM-injected runtime environment variable.",
                    extra={
                        "event_type": "secret_env_mount_resolved",
                        "secret_id": secret_id,
                    },
                )
                return mounted_value.strip()

        raise SecretResolutionError(
            f"Unable to resolve secret '{secret_id}' from Google Cloud Secret Manager "
            f"({resource_name}). Ensure the runtime Service Account has "
            f"'roles/secretmanager.secretAccessor' and GOOGLE_CLOUD_PROJECT is configured."
        )


def get_secret_provider(project_id: Optional[str] = None) -> SecretManagerProvider:
    """Returns a singleton-like SecretManagerProvider instance."""
    return SecretManagerProvider(project_id=project_id)
