"""
Azure model provider for the Agents SDK with support for local development and production environments.
"""

import os
import logging

from agents import (
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    set_default_openai_api,
)

from .client_factory import create_azure_client, TOKEN_CACHE_PATH
from graphs.openai_agent.parallel_agents.utils.azure_models import (
    get_azure_deployment_name,
)

logger = logging.getLogger(__name__)

# Environment configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
AUTH_METHOD = os.getenv("AZURE_AUTH_METHOD", "auto").lower()


class AzureModelProvider(ModelProvider):
    """
    Model provider that uses Azure OpenAI with environment-aware authentication.

    This provider supports both development (Azure CLI) and production (Managed Identity)
    environments, with token caching to minimize authentication calls.
    """

    def __init__(
        self,
        azure_endpoint=None,
        azure_api_version=None,
        token_cache_path=TOKEN_CACHE_PATH,
    ):
        """
        Initialize the model provider with an Azure OpenAI client.

        Args:
            azure_endpoint: Azure OpenAI endpoint URL (optional)
            azure_api_version: Azure OpenAI API version (optional)
            token_cache_path: Path to token cache file (optional)
        """
        try:
            # Log the environment and authentication method
            logger.debug(
                f"Initializing AzureModelProvider in {ENVIRONMENT} environment"
            )
            logger.debug(f"Using authentication method: {AUTH_METHOD}")
            logger.debug(f"Using token cache at: {token_cache_path}")

            # Create the Azure client with environment-aware authentication and token caching
            self.client = create_azure_client(
                azure_endpoint=azure_endpoint,
                azure_api_version=azure_api_version,
                token_cache_path=token_cache_path,
            )

            # Azure OpenAI uses the chat completions API
            set_default_openai_api("chat_completions")

            logger.debug(
                "AzureModelProvider initialized successfully with token caching"
            )

        except Exception as e:
            logger.error(f"Failed to initialize AzureModelProvider: {str(e)}")
            raise

    def get_model(self, model_name: str) -> Model:
        """
        Get a model instance with the Azure client.

        Args:
            model_name: Model name to use (will be mapped to Azure deployment)

        Returns:
            Model: OpenAI model instance configured for Azure
        """
        # Map the model name to an Azure deployment name
        deployment_name = get_azure_deployment_name(model_name)
        logger.debug(f"Mapped '{model_name}' to Azure deployment '{deployment_name}'")

        # Create and return the model
        return OpenAIChatCompletionsModel(
            model=deployment_name, openai_client=self.client
        )


# Function to create a provider instance
def create_azure_provider(
    azure_endpoint=None, azure_api_version=None, token_cache_path=TOKEN_CACHE_PATH
):
    """
    Create an Azure model provider instance with environment-aware authentication.

    This is a factory function instead of using a singleton to avoid
    initialization issues when used with different parameters.

    Args:
        azure_endpoint: Azure OpenAI endpoint URL (optional)
        azure_api_version: Azure OpenAI API version (optional)
        token_cache_path: Path to token cache file (optional)

    Returns:
        AzureModelProvider: The configured provider
    """
    try:
        provider = AzureModelProvider(
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
            token_cache_path=token_cache_path,
        )
        return provider
    except Exception as e:
        logger.error(f"Failed to create Azure provider: {str(e)}")
        raise
