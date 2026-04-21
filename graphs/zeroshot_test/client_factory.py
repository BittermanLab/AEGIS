"""
Azure OpenAI client factory with support for local development and production environments.
"""

import os
import logging
import time
import json
from pathlib import Path
from typing import Optional, Dict, Callable, Any

from openai import AsyncAzureOpenAI
from azure.identity import (
    DefaultAzureCredential,
    ManagedIdentityCredential,
    AzureCliCredential,
    ChainedTokenCredential,
)

logger = logging.getLogger(__name__)

# Default configuration from environment
DEFAULT_AZURE_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "https://bwh-bittermanlab-nonprod-openai-service.openai.azure.com/",
)
DEFAULT_AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
AUTH_METHOD = os.getenv("AZURE_AUTH_METHOD", "auto").lower()

# Token cache configuration
TOKEN_CACHE_PATH = "/tmp/azure_token_cache.json"
TOKEN_EXPIRY_BUFFER = 300  # 5 minutes buffer before token expiry


class TokenCache:
    """Simple token cache to minimize authentication calls"""

    def __init__(self, cache_path: str = TOKEN_CACHE_PATH):
        self.cache_path = Path(cache_path)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self):
        """Load token cache from disk if it exists"""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r") as f:
                    self._cache = json.load(f)
                logger.debug(f"Loaded token cache from {self.cache_path}")
            except Exception as e:
                logger.warning(f"Failed to load token cache: {str(e)}")
                self._cache = {}

    def _save_cache(self):
        """Save token cache to disk"""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f)
            logger.debug(f"Saved token cache to {self.cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save token cache: {str(e)}")

    def get_token(self, scope: str) -> Optional[Dict[str, Any]]:
        """Get token from cache if it exists and is not expired"""
        if scope in self._cache:
            token_data = self._cache[scope]
            # Check if token is still valid with buffer time
            if token_data["expires_on"] > time.time() + TOKEN_EXPIRY_BUFFER:
                logger.debug(
                    f"Using cached token for {scope}, expires in {token_data['expires_on'] - time.time():.0f} seconds"
                )
                return token_data
        return None

    def set_token(self, scope: str, token: str, expires_on: float):
        """Save token to cache"""
        self._cache[scope] = {"token": token, "expires_on": expires_on}
        self._save_cache()


def create_enhanced_token_provider(
    scope: str = "https://cognitiveservices.azure.com/.default",
    token_cache_path: str = TOKEN_CACHE_PATH,
) -> Callable[[], str]:
    """
    Creates a token provider that handles caching and credential selection based on environment

    This is an enhanced version that supports development (Azure CLI) and production (Managed Identity)
    environments, with token caching to minimize authentication calls.

    Args:
        scope: The scope for the token request
        token_cache_path: Path to token cache file
    """
    token_cache = TokenCache(cache_path=token_cache_path)

    # Choose the right credential chain based on environment and auth method
    if AUTH_METHOD == "cli" or (AUTH_METHOD == "auto" and ENVIRONMENT == "development"):
        logger.debug(
            "Setting up credential chain for development environment with Azure CLI"
        )
        # For development, try Azure CLI first, then fall back to default credential
        credential = ChainedTokenCredential(
            AzureCliCredential(),
            DefaultAzureCredential(exclude_managed_identity_credential=True),
        )
    elif AUTH_METHOD == "managed_identity" or (
        AUTH_METHOD == "auto" and ENVIRONMENT != "development"
    ):
        logger.debug(
            "Setting up credential chain for production environment with Managed Identity"
        )
        # For production, try Managed Identity first, then fall back to default
        credential = ChainedTokenCredential(
            DefaultAzureCredential(), ManagedIdentityCredential()
        )
    else:
        logger.debug("Using DefaultAzureCredential for authentication")
        credential = DefaultAzureCredential()

    def get_token() -> str:
        # Check cache first
        cached_token = token_cache.get_token(scope)
        if cached_token:
            return cached_token["token"]

        # If not in cache, acquire new token
        try:
            logger.debug(f"Acquiring new token for scope {scope}")
            token_response = credential.get_token(scope)
            logger.debug(
                f"Token acquired, expires in {token_response.expires_on - time.time():.0f} seconds"
            )

            # Cache the token
            token_cache.set_token(
                scope=scope,
                token=token_response.token,
                expires_on=token_response.expires_on,
            )

            return token_response.token
        except Exception as e:
            logger.error(f"Failed to get token: {str(e)}")
            raise

    return get_token


# Legacy function for backward compatibility
def get_bearer_token_provider(
    credential, scope="https://cognitiveservices.azure.com/.default"
):
    """Creates a callable token provider function from a credential (legacy)."""
    logger.warning(
        "Using legacy get_bearer_token_provider - consider upgrading to create_enhanced_token_provider"
    )

    def get_token():
        try:
            token = credential.get_token(scope)
            return token.token
        except Exception as e:
            logger.error(f"Error getting token: {str(e)}")
            raise

    return get_token


def create_azure_client(
    azure_endpoint: Optional[str] = None,
    azure_api_version: Optional[str] = None,
    token_cache_path: str = TOKEN_CACHE_PATH,
    api_key: Optional[str] = None,
) -> AsyncAzureOpenAI:
    """
    Create an AsyncAzureOpenAI client with Azure AD authentication.

    This enhanced function supports both development and production environments,
    with token caching to minimize authentication calls.

    Args:
        azure_endpoint: Azure OpenAI endpoint URL
        azure_api_version: Azure OpenAI API version
        token_cache_path: Path to token cache file

    Returns:
        AsyncAzureOpenAI: The Azure OpenAI client
    """
    # Use provided parameters or defaults
    azure_endpoint = azure_endpoint or DEFAULT_AZURE_ENDPOINT
    azure_api_version = azure_api_version or DEFAULT_AZURE_API_VERSION

    try:
        logger.debug(f"Creating Azure client with endpoint: {azure_endpoint}")
        logger.debug(f"Environment: {ENVIRONMENT}, Auth Method: {AUTH_METHOD}")
        logger.debug(f"Using token cache at: {token_cache_path}")

        # Create token provider with environment awareness and caching
        token_provider = create_enhanced_token_provider(
            token_cache_path=token_cache_path
        )

        # Create and return the Azure client
        client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
            azure_ad_token_provider=token_provider,
            api_key=api_key,
        )

        logger.debug("Azure OpenAI client created successfully with token cache")
        return client

    except Exception as e:
        logger.error(f"Error creating Azure OpenAI client: {str(e)}")
        raise
