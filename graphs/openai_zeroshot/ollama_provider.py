"""
Ollama model provider for integration with Agents SDK.
This module provides functions to create and configure an Ollama client for local LLM inference.
"""

import logging
import os
from typing import Optional

from agents import (
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    set_default_openai_api,
)

logger = logging.getLogger(__name__)

# Default Ollama endpoint from environment
DEFAULT_OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT")


class OllamaModelProvider(ModelProvider):
    """
    Model provider for Ollama local models using the OpenAI-compatible API.

    Uses AsyncOpenAI client with a custom base URL pointing to the Ollama server.
    """

    def __init__(self, ollama_endpoint: Optional[str] = None):
        """
        Initialize the Ollama model provider.

        Args:
            ollama_endpoint: URL of the Ollama API endpoint (or OLLAMA_ENDPOINT env var)
        """
        self.ollama_endpoint = ollama_endpoint or DEFAULT_OLLAMA_ENDPOINT
        if not self.ollama_endpoint:
            raise ValueError(
                "Ollama endpoint is required. Pass --ollama-endpoint (propagates to parameters) or set OLLAMA_ENDPOINT."
            )

        # Ensure endpoint ends with /v1
        if not self.ollama_endpoint.endswith("/v1"):
            if self.ollama_endpoint.endswith("/"):
                self.ollama_endpoint += "v1"
            else:
                self.ollama_endpoint += "/v1"
        
        logger.debug("Configured Ollama endpoint: %s", self.ollama_endpoint)

        # Force chat completions API since Ollama uses OpenAI compatible format
        self.logger = logger.getChild(self.__class__.__name__)
        self.logger.info(
            "OllamaModelProvider initialized with endpoint: %s", self.ollama_endpoint
        )

        set_default_openai_api("chat_completions")

    def get_client(self):
        """Get the OpenAI client for Ollama"""
        from agents import AsyncOpenAI
        self.logger.debug("Creating AsyncOpenAI client for Ollama with base_url=%s", self.ollama_endpoint)
        return AsyncOpenAI(base_url=self.ollama_endpoint, api_key="ollama")

    def get_model(self, model_name: str) -> Model:
        """
        Get a model instance that uses Ollama local inference.

        Args:
            model_name: Name of the model to use, e.g. "llama2", "deepseek-coder-1.3b-instruct", etc.

        Returns:
            Model: Configured model instance
        """
        self.logger.debug("Creating Ollama model for %s", model_name)
        
        # Create a client for this model
        client = self.get_client()
        
        try:
            # Create and return the model with its dedicated client
            # For Ollama models, we only pass the minimal required parameters
            # Do NOT pass temperature, top_p, or other parameters that are unsupported
            model = OpenAIChatCompletionsModel(
                model=model_name,
                openai_client=client,
            )
            self.logger.debug("Created Ollama model for %s (without temperature)", model_name)
            return model
        except Exception as e:
            self.logger.error("Error creating Ollama model: %s", str(e))
            self.logger.exception(e)
            raise


# Singleton instance for convenient access
OLLAMA_PROVIDER = None


def get_ollama_provider(endpoint: Optional[str] = None) -> OllamaModelProvider:
    """
    Get or create the singleton Ollama provider instance.
    If an explicit endpoint is provided, creates a new provider with that endpoint.

    Args:
        endpoint: Optional Ollama API endpoint URL

    Returns:
        OllamaModelProvider: Configured provider instance
    """
    global OLLAMA_PROVIDER

    # Force creation of a new provider if a specific endpoint is provided
    if endpoint is not None:
        try:
            logger.info("Creating new OllamaModelProvider with explicit endpoint: %s", endpoint)
            return OllamaModelProvider(endpoint)  # Return non-singleton instance with specified endpoint
        except Exception as e:
            logger.error("Failed to create OllamaModelProvider with endpoint %s: %s", endpoint, str(e))
            raise
    
    # Otherwise use/create singleton
    if OLLAMA_PROVIDER is None:
        try:
            OLLAMA_PROVIDER = OllamaModelProvider(endpoint)
            logger.debug("Created new OllamaModelProvider singleton with default endpoint")
        except Exception as e:
            logger.error("Failed to create OllamaModelProvider: %s", str(e))
            raise

    return OLLAMA_PROVIDER
