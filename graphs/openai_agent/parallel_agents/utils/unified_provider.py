"""
Unified model provider factory that supports both Azure OpenAI and local Ollama models.
This module selects the appropriate provider based on model name and configuration.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Union

from agents import Model, OpenAIChatCompletionsModel

from .model_config import is_ollama_model, is_vllm_model
from .ollama_provider import get_ollama_provider
from .model_provider import create_azure_provider
from .vllm_provider import VLLMModelProvider

logger = logging.getLogger(__name__)


def get_model_for_name(
    model_name: str, provider_config: Optional[Dict[str, Any]] = None
) -> Model:
    """
    Get a model instance for the specified model name using the appropriate provider.

    Automatically determines whether to use Azure OpenAI, local Ollama, or vLLM based on the model name.

    Args:
        model_name: Name of the model to use
        provider_config: Optional provider-specific configuration
            For Azure: {'azure_endpoint': '...', 'azure_api_version': '...'}
            For Ollama: {'ollama_endpoint': '...'}
            For vLLM: {'vllm_endpoint': '...'}

    Returns:
        Model: Appropriate model instance for the specified model
    """
    provider_config = provider_config or {}

    # Check if it's a vLLM model
    if is_vllm_model(model_name):
        logger.info(f"Using vLLM provider for model: {model_name}")
        vllm_endpoint = provider_config.get("vllm_endpoint")
        provider = VLLMModelProvider(base_url=vllm_endpoint)
        return provider.get_model(model_name)

    # Check if it's an Ollama model
    elif is_ollama_model(model_name):
        logger.info(f"Using Ollama provider for model: {model_name}")
        ollama_endpoint = provider_config.get("ollama_endpoint")
        provider = get_ollama_provider(endpoint=ollama_endpoint)
        return provider.get_model(model_name)

    # Otherwise, use Azure OpenAI provider
    else:
        logger.info(f"Using Azure OpenAI provider for model: {model_name}")
        azure_endpoint = provider_config.get("azure_endpoint")
        azure_api_version = provider_config.get("azure_api_version")
        provider = create_azure_provider(
            azure_endpoint=azure_endpoint, azure_api_version=azure_api_version
        )
        return provider.get_model(model_name)


class UnifiedModelFactory:
    """
    Unified factory for creating model instances based on model name.

    This class maintains configuration state and provider instances
    across multiple model requests.
    """

    def __init__(self, provider_config: Optional[Dict[str, Any]] = None):
        """
        Initialize with optional provider configuration.

        Args:
            provider_config: Optional provider-specific configuration
        """
        self.provider_config = provider_config or {}
        self.azure_provider = None
        self.ollama_provider = None
        self.vllm_provider = None

    def get_model(self, model_name: str) -> Model:
        """
        Get an appropriate model instance based on the model name.

        Args:
            model_name: Name of the model to use

        Returns:
            Model: Appropriate model instance
        """
        # Determine which provider to use
        if is_vllm_model(model_name):
            # Initialize vLLM provider if not already done
            if not self.vllm_provider:
                vllm_endpoint = self.provider_config.get("vllm_endpoint")
                self.vllm_provider = VLLMModelProvider(base_url=vllm_endpoint)

            # Get and return the model
            # Note: vLLM provider's get_model is async, but we need to return a sync model
            # The VLLMModelProvider should handle this internally by returning a sync wrapper
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context, so we can't use run_until_complete
                # This should be handled by the calling code
                raise RuntimeError(
                    "Cannot call get_model for vLLM models from within an async context. "
                    "Use get_model_async instead or ensure the model is created before entering async code."
                )
            else:
                # We can run the async method synchronously
                return loop.run_until_complete(self.vllm_provider.get_model(model_name))
        elif is_ollama_model(model_name):
            # Initialize Ollama provider if not already done
            if not self.ollama_provider:
                ollama_endpoint = self.provider_config.get("ollama_endpoint")
                self.ollama_provider = get_ollama_provider(endpoint=ollama_endpoint)

            # Get and return the model
            return self.ollama_provider.get_model(model_name)
        else:
            # Initialize Azure provider if not already done
            if not self.azure_provider:
                azure_endpoint = self.provider_config.get("azure_endpoint")
                azure_api_version = self.provider_config.get("azure_api_version")
                self.azure_provider = create_azure_provider(
                    azure_endpoint=azure_endpoint, azure_api_version=azure_api_version
                )

            # Get and return the model
            return self.azure_provider.get_model(model_name)

    async def get_model_async(self, model_name: str) -> Model:
        """
        Get an appropriate model instance based on the model name (async version).

        Args:
            model_name: Name of the model to use

        Returns:
            Model: Appropriate model instance
        """
        # Determine which provider to use
        if is_vllm_model(model_name):
            # Initialize vLLM provider if not already done
            if not self.vllm_provider:
                vllm_endpoint = self.provider_config.get("vllm_endpoint")
                self.vllm_provider = VLLMModelProvider(base_url=vllm_endpoint)

            # Get and return the model asynchronously
            return await self.vllm_provider.get_model(model_name)
        else:
            # For non-async providers, just call the sync method
            return self.get_model(model_name)


# Global factory instance for convenient access
DEFAULT_FACTORY = None


def get_model_factory(
    provider_config: Optional[Dict[str, Any]] = None,
) -> UnifiedModelFactory:
    """
    Get or create the global model factory instance.

    Args:
        provider_config: Optional provider configuration to use if creating a new factory

    Returns:
        UnifiedModelFactory: The configured factory instance
    """
    global DEFAULT_FACTORY

    if DEFAULT_FACTORY is None:
        DEFAULT_FACTORY = UnifiedModelFactory(provider_config)
        logger.debug("Created new UnifiedModelFactory instance")

    return DEFAULT_FACTORY
