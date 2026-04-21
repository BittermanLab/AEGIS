"""
High‑performance Azure OpenAI client factory.

* Uses **AzureCliCredential** (fast, no IMDS calls) with an in‑process token
  cache so we don't shell‑out on every request.
* Signature **unchanged** – callers keep passing (endpoint, api_version, …)
  and receive an **AsyncAzureOpenAI** instance.
* No client‑side rate limiting here; that remains in the provider layer.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Final, Optional

import httpx
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
    DefaultAzureCredential,
)
from openai import AsyncAzureOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults pulled from environment for flexibility
# ---------------------------------------------------------------------------

DEFAULT_AZURE_ENDPOINT: Final[str] = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "https://bwh-bittermanlab-nonprod-openai-service.openai.azure.com/",
)
DEFAULT_AZURE_API_VERSION: Final[str] = os.getenv(
    "AZURE_OPENAI_API_VERSION",
    "2025-04-01-preview",
)

# Retry/timeout tuned for typical 500‑RPM quota
DEFAULT_MAX_RETRIES: Final[int] = 4
DEFAULT_TIMEOUT: Final[float] = 120.0

# ---------------------------------------------------------------------------
# Token provider with caching ------------------------------------------------
# ---------------------------------------------------------------------------


def _build_token_provider() -> "callable[[], str]":
    """Return a `azure_ad_token_provider` callable with local cache."""

    # Use DefaultAzureCredential which has better error handling and fallback logic
    # Or create custom chain with more robust options, preferring Managed Identity first
    # to avoid the CLI occasional crashes
    try:
        # First attempt with DefaultAzureCredential which has better error handling
        credential_chain = AzureCliCredential()
        logger.debug("Using DefaultAzureCredential for token authentication")
    except Exception as e:
        logger.warning(
            f"DefaultAzureCredential initialization failed: {e}. Falling back to custom chain."
        )
        # Fallback to custom chain with ManagedIdentity as the first option
        # since CLI occasionally has issues with "Illegal instruction"
        credential_chain = ChainedTokenCredential(
            AzureCliCredential(),  # Fall back to CLI if needed
            ManagedIdentityCredential(),  # Try Managed Identity first
        )
        logger.debug("Using custom ChainedTokenCredential for token authentication")

    _cached_token: dict[str, Optional[str | int]] = {"token": None, "exp": 0}
    _scope = "https://cognitiveservices.azure.com/.default"

    def _provider() -> str:  # noqa: D401 – simple provider fn
        # Refresh token if we're within 60 s of expiry (or missing)
        if _cached_token["token"] is None or (_cached_token["exp"] - 60) < time.time():
            try:
                token = credential_chain.get_token(_scope)
                _cached_token["token"] = token.token
                _cached_token["exp"] = token.expires_on
                logger.debug(
                    "Fetched fresh AAD token – expires at %s", token.expires_on
                )
            except Exception as e:
                logger.error(f"Failed to get token: {e}")
                # If we already have a token, keep using it rather than failing
                if _cached_token["token"] is not None:
                    logger.warning(
                        "Using previously cached token despite refresh failure"
                    )
                else:
                    # Reraise if we have no token at all
                    raise
        return str(_cached_token["token"])

    return _provider


# ---------------------------------------------------------------------------
# Public factory -------------------------------------------------------------
# ---------------------------------------------------------------------------


def create_azure_client(
    azure_endpoint: str | None = None,
    azure_api_version: str | None = None,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
) -> AsyncAzureOpenAI:
    """Return a ready‑to‑use **AsyncAzureOpenAI** client.

    The caller is responsible for scheduling the event loop / awaiting calls.
    Rate‑limiting (if desired) should be applied one layer up (see provider).
    """

    endpoint = azure_endpoint or DEFAULT_AZURE_ENDPOINT
    api_version = azure_api_version or DEFAULT_AZURE_API_VERSION

    logger.debug(
        "Creating Azure OpenAI client for %s (api‑version=%s)", endpoint, api_version
    )

    token_provider = _build_token_provider()

    transport = httpx.AsyncHTTPTransport(
        retries=max_retries,
        limits=httpx.Limits(
            max_connections=50, max_keepalive_connections=15, keepalive_expiry=30.0
        ),
    )

    http_client = httpx.AsyncClient(transport=transport, timeout=timeout)

    client = AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_ad_token_provider=token_provider,
        http_client=http_client,
        max_retries=max_retries,
        timeout=timeout,
    )

    logger.debug(
        "Azure OpenAI client initialised (retries=%d, timeout=%ss)",
        max_retries,
        timeout,
    )
    return client
