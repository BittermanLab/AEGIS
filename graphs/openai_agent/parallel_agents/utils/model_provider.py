"""
Azure OpenAI Model Provider with built‑in global rate limiting.

This module is a drop‑in replacement for the previous `AzureModelProvider`.
Return types and public signatures are unchanged, but every request sent
through the client is now guarded by a simple, thread‑safe token‑bucket
rate limiter.

Adjust `RATE_LIMIT_PER_MINUTE` as required.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable

from agents import (
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    set_default_openai_api,
)

from .client_factory import create_azure_client
from .azure_models import get_azure_deployment_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RATE_LIMIT_PER_MINUTE: int = 500  # ❗ Central place to change the RPM limit

# ---------------------------------------------------------------------------
# Rate‑limiting implementation (token‑bucket, thread‑safe)
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Simple, in‑process token‑bucket rate limiter (thread‑safe)."""

    def __init__(self, max_requests_per_minute: int):
        self._capacity = max_requests_per_minute
        self._interval = 60.0  # seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until the caller is allowed to perform a request."""
        while True:
            with self._lock:
                now = time.time()
                # Expire timestamps older than the time window
                while self._timestamps and now - self._timestamps[0] >= self._interval:
                    self._timestamps.popleft()

                if len(self._timestamps) < self._capacity:
                    # We have room → log timestamp and return control to caller
                    self._timestamps.append(now)
                    return

                # Otherwise, calculate sleep time until the earliest timestamp expires
                sleep_for = self._interval - (now - self._timestamps[0])

            # Sleep outside the lock to avoid blocking other threads unnecessarily
            time.sleep(sleep_for)


# Single global limiter shared by all provider instances in this process
_request_limiter = _RateLimiter(RATE_LIMIT_PER_MINUTE)


def _rate_limited(func: Callable):
    """Decorator applying the global request limiter to a function."""

    def wrapper(*args, **kwargs):  # type: ignore[override]
        _request_limiter.acquire()
        return func(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Azure Model Provider
# ---------------------------------------------------------------------------


class AzureModelProvider(ModelProvider):
    """Factory class that returns a rate‑limited `OpenAIChatCompletionsModel`."""

    def __init__(
        self, *, azure_endpoint: str | None = None, azure_api_version: str | None = None
    ):
        self.azure_endpoint = azure_endpoint
        self.azure_api_version = azure_api_version

        # Force Agents SDK to use chat‑completions flavour
        set_default_openai_api("chat_completions")

        logger.debug("AzureModelProvider initialised")

    # ---------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _attach_rate_limiter(client):
        """Attach the rate‑limiting decorator to the client's completion endpoint."""
        try:
            client.chat.completions.create = _rate_limited(client.chat.completions.create)  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover
            logger.warning(
                "Could not patch client for rate limiting – unexpected SDK structure."
            )
        return client

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def get_model(self, model_name: str) -> Model:  # noqa: D401 – keep as per SDK
        """Return a ready‑to‑use chat model backed by an Azure client."""
        deployment_name = get_azure_deployment_name(model_name)
        logger.debug(
            "Mapped '%s' to Azure deployment '%s'", model_name, deployment_name
        )

        # Tune retry/timeout heuristics per model family
        if any(alias in deployment_name for alias in ("o3-mini", "o4-mini")):
            max_retries, timeout = 10, 240.0
        else:
            max_retries, timeout = 6, 180.0

        client = create_azure_client(
            azure_endpoint=self.azure_endpoint,
            azure_api_version=self.azure_api_version,
            max_retries=max_retries,
            timeout=timeout,
        )

        self._attach_rate_limiter(client)
        return OpenAIChatCompletionsModel(model=deployment_name, openai_client=client)


# Convenience factory ------------------------------------------------------------------


def create_azure_provider(
    *, azure_endpoint: str | None = None, azure_api_version: str | None = None
) -> AzureModelProvider:
    """Return a pre‑configured `AzureModelProvider` instance."""
    try:
        return AzureModelProvider(
            azure_endpoint=azure_endpoint, azure_api_version=azure_api_version
        )
    except Exception:
        logger.exception("Failed to create Azure provider")
        raise
