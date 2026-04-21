"""
Centralized model mapping between different model names and deployment identifiers.
This module provides functions to map between standard OpenAI model names
and Azure OpenAI deployment names.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Default mapping between standard OpenAI model names and Azure deployment names
# This should be customized to match your actual Azure OpenAI deployments
DEFAULT_AZURE_MODEL_MAPPING = {
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini-jg",
    "gpt-4": "gpt-4",
    "o1": "o1",
    "o3-mini": "o3-mini-jg",
    "o4-mini": "o4-mini",
    # Map any aliases or shorthand names to their actual deployment names
    "4o": "gpt-4o",
    "4o-mini": "gpt-4o-mini-jg",
    "gpt-4o-mini-jg": "gpt-4o-mini-jg",
    "o3-mini-jg": "o3-mini-jg",
    "gpt-4.1-nano": "gpt-4.1-nano",
    "gpt-4.1-mini": "gpt-4.1-mini",
}


def get_model_mapping():
    """
    Get the model mapping from environment variables or use defaults.

    The mapping can be customized by setting environment variables in the format:
    AZURE_MODEL_<openai_model_name>=<azure_deployment_name>

    For example:
    AZURE_MODEL_gpt_4o=my-gpt4-deployment

    Returns:
        dict: Mapping from OpenAI model names to Azure deployment names
    """
    mapping = DEFAULT_AZURE_MODEL_MAPPING.copy()

    # Look for environment variables that define model mappings
    for var_name, var_value in os.environ.items():
        if var_name.startswith("AZURE_MODEL_"):
            openai_model = var_name[12:].lower().replace("_", "-")
            azure_deployment = var_value
            mapping[openai_model] = azure_deployment
            logger.debug(f"Found model mapping: {openai_model} -> {azure_deployment}")

    return mapping


def get_azure_deployment_name(model_name):
    """
    Convert standard OpenAI model name to Azure deployment name.

    Args:
        model_name: Standard OpenAI model name (e.g., "gpt-4o")

    Returns:
        str: Azure deployment name or original name if no mapping exists
    """
    mapping = get_model_mapping()

    # Handle None model name
    if not model_name:
        default_deployment = os.getenv("AZURE_DEFAULT_DEPLOYMENT", "gpt-4o-mini-jg")
        logger.warning(f"No model name provided, using default: {default_deployment}")
        return default_deployment

    # Convert the model name to lowercase to make it case-insensitive
    model_name_lower = model_name.lower()

    # Get the Azure deployment name or use the original if not found
    deployment_name = mapping.get(model_name_lower, model_name)

    if model_name_lower in mapping:
        logger.debug(
            f"Mapped model '{model_name}' to Azure deployment '{deployment_name}'"
        )
    else:
        logger.warning(f"No mapping found for model '{model_name}', using as-is")

    return deployment_name
