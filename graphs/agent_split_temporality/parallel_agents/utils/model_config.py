"""
Model configuration for the OpenAI Agent SDK integration.

This module handles model selection for different agent roles and provides
optimized configurations for various quality/cost tradeoffs.
"""

import logging
import os
from typing import Dict, Union

from agents import OpenAIChatCompletionsModel
from agents.model_settings import ModelSettings

logger = logging.getLogger(__name__)

# Default model from environment with JG deployment suffix
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gpt-4o-mini-jg")

# Deployment name mapping to use custom deployments with higher rate limits
DEPLOYMENT_MAPPING = {
    "gpt-4o-mini": "gpt-4o-mini-jg",
    "o3-mini": "o3-mini-jg",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "gpt-4.1-nano": "gpt-4.1-nano",
    "o4-mini": "o4-mini",
}

# List of Ollama model names to identify which models should use Ollama provider
OLLAMA_MODELS = [
    "deepseek-r1:1.5b",
    "deepseek-r1:8b",
    "deepseek-r1:14b",
    "llama3.2:1b",
    "qwen3:14b",
]

# List of vLLM model names to identify which models should use vLLM provider
VLLM_MODELS = [
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "Qwen/Qwen3-8B-FP8",
    "Qwen/Qwen3-14B-FP8",
    "Qwen/Qwen3-32B-FP8",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "google/medgemma-27b-text-it",
]

# Role-based model configurations
MODEL_CONFIGS = {
    # Default configuration (balanced)
    "default": {
        "extractor": "gpt-4o-mini-jg",
        "parallel_agent": "gpt-4o-mini-jg",
        "judge_agent": "gpt-4o-mini-jg",
        "general": DEFAULT_MODEL,
    },
    # Economy configuration (cost-optimized)
    "4o-mini": {
        "extractor": "gpt-4o-mini-jg",
        "parallel_agent": "gpt-4o-mini-jg",
        "judge_agent": "gpt-4o-mini-jg",
        "general": "gpt-4o-mini-jg",
    },
    # O3-mini configuration
    "o3-mini": {
        "extractor": "o3-mini-jg",
        "parallel_agent": "o3-mini-jg",
        "judge_agent": "o3-mini-jg",
        "general": "o3-mini-jg",
    },
    # O4-mini configuration
    "o4-mini": {
        "extractor": "o4-mini",
        "parallel_agent": "o4-mini",
        "judge_agent": "o4-mini",
        "general": "o4-mini",
    },
    # Premium configuration (4o-mini)
    "o3-judge": {
        "extractor": "gpt-4o-mini-jg",
        "parallel_agent": "gpt-4o-mini-jg",
        "judge_agent": "o3-mini-jg",
        "general": "gpt-4o-mini-jg",
    },
    "o3-middle": {
        "extractor": "gpt-4o-mini-jg",
        "parallel_agent": "o3-mini-jg",
        "judge_agent": "o3-mini-jg",
        "general": "gpt-4o-mini-jg",
    },
    "4.1-mini": {
        "extractor": "gpt-4.1-mini",
        "parallel_agent": "gpt-4.1-mini",
        "judge_agent": "gpt-4.1-mini",
        "general": "gpt-4.1-mini",
    },
    "4.1-nano": {
        "extractor": "gpt-4.1-nano",
        "parallel_agent": "gpt-4.1-nano",
        "judge_agent": "gpt-4.1-nano",
        "general": "gpt-4.1-nano",
    },
    "gpt-4.1-nano": {
        "extractor": "gpt-4.1-nano",
        "parallel_agent": "gpt-4.1-nano",
        "judge_agent": "gpt-4.1-nano",
        "general": "gpt-4.1-nano",
    },
    "gpt-4.1-mini": {
        "extractor": "gpt-4.1-mini",
        "parallel_agent": "gpt-4.1-mini",
        "judge_agent": "gpt-4.1-mini",
        "general": "gpt-4.1-mini",
    },
    # Ollama model configurations with specific names
    "ollama-deepseek-1b": {
        "extractor": "deepseek-r1:1.5b",
        "parallel_agent": "deepseek-r1:1.5b",
        "judge_agent": "deepseek-r1:1.5b",
        "general": "deepseek-r1:1.5b",
    },
    "ollama-deepseek-8b": {
        "extractor": "deepseek-r1:8b",
        "parallel_agent": "deepseek-r1:8b",
        "judge_agent": "deepseek-r1:8b",
        "general": "deepseek-r1:8b",
    },
    "ollama-deepseek-14b": {
        "extractor": "deepseek-r1:14b",
        "parallel_agent": "deepseek-r1:14b",
        "judge_agent": "deepseek-r1:14b",
        "general": "deepseek-r1:14b",
    },
    "ollama-llama-1b": {
        "extractor": "llama3.2:1b",
        "parallel_agent": "llama3.2:1b",
        "judge_agent": "llama3.2:1b",
        "general": "llama3.2:1b",
    },
    "ollama-qwen3-14b": {
        "extractor": "qwen3:14b",
        "parallel_agent": "qwen3:14b",
        "judge_agent": "qwen3:14b",
        "general": "qwen3:14b",
    },
    # vLLM model configurations
    "vllm-deepseek-r1-8b": {
        "extractor": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "parallel_agent": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "judge_agent": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "general": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    },
    "vllm-deepseek-r1-distill-qwen-32b": {
        "extractor": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "parallel_agent": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "judge_agent": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "general": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    },
    "vllm-qwen3-8b": {
        "extractor": "Qwen/Qwen3-8B-FP8",
        "parallel_agent": "Qwen/Qwen3-8B-FP8",
        "judge_agent": "Qwen/Qwen3-8B-FP8",
        "general": "Qwen/Qwen3-8B-FP8",
    },
    "vllm-qwen3-14b": {
        "extractor": "Qwen/Qwen3-14B-FP8",
        "parallel_agent": "Qwen/Qwen3-14B-FP8",
        "judge_agent": "Qwen/Qwen3-14B-FP8",
        "general": "Qwen/Qwen3-14B-FP8",
    },
    "vllm-qwen3-32b": {
        "extractor": "Qwen/Qwen3-32B-FP8",
        "parallel_agent": "Qwen/Qwen3-32B-FP8",
        "judge_agent": "Qwen/Qwen3-32B-FP8",
        "general": "Qwen/Qwen3-32B-FP8",
    },
    "vllm-gemma3-4b": {
        "extractor": "google/gemma-3-4b-it",
        "parallel_agent": "google/gemma-3-4b-it",
        "judge_agent": "google/gemma-3-4b-it",
        "general": "google/gemma-3-4b-it",
    },
    "vllm-gemma3-12b": {
        "extractor": "google/gemma-3-12b-it",
        "parallel_agent": "google/gemma-3-12b-it",
        "judge_agent": "google/gemma-3-12b-it",
        "general": "google/gemma-3-12b-it",
    },
    "vllm-gemma3-27b": {
        "extractor": "google/gemma-3-27b-it",
        "parallel_agent": "google/gemma-3-27b-it",
        "judge_agent": "google/gemma-3-27b-it",
        "general": "google/gemma-3-27b-it",
    },
    "vllm-medgemma-27b": {
        "extractor": "google/medgemma-27b-text-it",
        "parallel_agent": "google/medgemma-27b-text-it",
        "judge_agent": "google/medgemma-27b-text-it",
        "general": "google/medgemma-27b-text-it",
    },
}

# Model-specific parameter settings
MODEL_SETTINGS = {
    # GPT-4o-mini - Use temperature 0
    "gpt-4o-mini-jg": ModelSettings(temperature=0.0, top_p=1.0),
    # o3-mini - Cannot have temperature values
    "o3-mini-jg": ModelSettings(),
    "o4-mini": ModelSettings(),
    "gpt-4.1-mini": ModelSettings(temperature=0.0, top_p=1.0),
    "gpt-4.1-nano": ModelSettings(temperature=0.0, top_p=1.0),
    # Ollama models settings
    "deepseek-r1:1.5b": ModelSettings(temperature=0.0, top_p=1.0),
    "deepseek-r1:8b": ModelSettings(temperature=0.0, top_p=1.0),
    "deepseek-r1:14b": ModelSettings(temperature=0.0, top_p=1.0),
    "llama3.2:1b": ModelSettings(temperature=0.0, top_p=1.0),
    "qwen3:14b": ModelSettings(temperature=0.0, top_p=1.0),
    # vLLM models settings
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": ModelSettings(temperature=0.0, top_p=1.0, max_tokens=4096),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": ModelSettings(temperature=0.0, top_p=1.0, max_tokens=4096),
    "Qwen/Qwen3-8B-FP8": ModelSettings(temperature=0.0, top_p=1.0, max_tokens=4096),
    "Qwen/Qwen3-14B-FP8": ModelSettings(temperature=0.0, top_p=1.0, max_tokens=4096),
    "Qwen/Qwen3-32B-FP8": ModelSettings(temperature=0.0, top_p=1.0, max_tokens=4096),
    "google/gemma-3-4b-it": ModelSettings(temperature=0.0, top_p=1.0, max_tokens=4096),
    "google/gemma-3-12b-it": ModelSettings(temperature=0.0, top_p=1.0, max_tokens=4096),
    "google/gemma-3-27b-it": ModelSettings(temperature=0.0, top_p=1.0, max_tokens=4096),
    "google/medgemma-27b-text-it": ModelSettings(temperature=0.0, top_p=1.0, max_tokens=4096),
    # Default fallback settings
    "default": ModelSettings(temperature=0.0, top_p=1.0),
}


def is_ollama_model(model_name: str) -> bool:
    """
    Check if the given model name is an Ollama model.

    Args:
        model_name: Model name to check

    Returns:
        bool: True if it's an Ollama model, False otherwise
    """
    if not model_name:
        return False

    # Heuristic for local Ollama shorthand names only (exclude HuggingFace-style names with '/').
    if (
        "/" not in model_name
        and ("deepseek" in model_name.lower() or "llama" in model_name.lower())
    ):
        logger.info("Detected Ollama model by name pattern: %s", model_name)
        return True

    # Config key indicates Ollama model
    if model_name.lower().startswith("ollama-"):
        logger.info("Detected Ollama model by config prefix: %s", model_name)
        return True

    # Check against our list of Ollama models
    is_ollama = any(
        model_name.lower() == ollama_model.lower()
        or ollama_model.lower() in model_name.lower()
        for ollama_model in OLLAMA_MODELS
    )

    if is_ollama:
        logger.info("Detected Ollama model in model list: %s", model_name)

    return is_ollama


def is_vllm_model(model_name: str) -> bool:
    """
    Check if the given model name is a vLLM model.

    Args:
        model_name: Model name to check

    Returns:
        bool: True if it's a vLLM model, False otherwise
    """
    if not model_name:
        return False

    # Config key indicates vLLM model
    if model_name.lower().startswith("vllm-"):
        logger.info("Detected vLLM model by config prefix: %s", model_name)
        return True

    # Check if model name contains organization/model pattern (common for HuggingFace models served by vLLM)
    if "/" in model_name and any(org in model_name.lower() for org in ["meta-llama", "mistralai", "deepseek-ai", "qwen", "google"]):
        logger.info("Detected vLLM model by HuggingFace naming pattern: %s", model_name)
        return True

    # Check against our list of vLLM models
    is_vllm = any(
        model_name.lower() == vllm_model.lower()
        or vllm_model.lower() in model_name.lower()
        for vllm_model in VLLM_MODELS
    )

    if is_vllm:
        logger.info("Detected vLLM model in model list: %s", model_name)

    return is_vllm


def get_model_for_role(
    config_name: str, role: str, openai_client=None
) -> Union[str, OpenAIChatCompletionsModel]:
    """
    Get the appropriate model for a specific agent role.

    Args:
        config_name: Configuration name (default, economy, high_quality, premium, vllm-*, ollama-*)
        role: Agent role (extractor, parallel_agent, judge_agent, general)
        openai_client: Optional OpenAI client to use with the model

    Returns:
        Either a model name string or an OpenAIChatCompletionsModel instance
    """
    # Check if we're dealing with a vLLM model configuration
    if config_name.lower().startswith("vllm-") or is_vllm_model(config_name):
        logger.info("Direct vLLM detection in get_model_for_role: %s", config_name)
        # Try to find the vLLM config
        config = MODEL_CONFIGS.get(config_name, None)
        if config is None:
            # If not found, use the direct model name
            model_name = config_name
            logger.info("Using direct vLLM model name: %s", model_name)
        else:
            model_name = config.get(role, config.get("general", config_name))
            logger.info("Using vLLM model %s for role %s from config", model_name, role)
    # First check if we're dealing with an Ollama model by name pattern
    elif (
        "deepseek" in config_name.lower()
        or "llama" in config_name.lower()
        or "qwen" in config_name.lower()
    ) and not "/" in config_name:  # Exclude HuggingFace format models (vLLM)
        logger.info("Direct Ollama detection in get_model_for_role: %s", config_name)
        # Handle specific Ollama model sizes
        if "deepseek" in config_name.lower():
            if "1.5b" in config_name.lower() or "1b" in config_name.lower():
                config = MODEL_CONFIGS.get("ollama-deepseek-1b")
                model_name = config.get(role, config.get("general", "deepseek-r1:1.5b"))
            elif "8b" in config_name.lower():
                config = MODEL_CONFIGS.get("ollama-deepseek-8b")
                model_name = config.get(role, config.get("general", "deepseek-r1:8b"))
            elif "14b" in config_name.lower():
                config = MODEL_CONFIGS.get("ollama-deepseek-14b")
                model_name = config.get(role, config.get("general", "deepseek-r1:14b"))
            else:
                config = MODEL_CONFIGS.get("ollama-deepseek-1b")  # Default to 1b
                model_name = config.get(role, config.get("general", "deepseek-r1:1.5b"))
        elif "llama" in config_name.lower():
            config = MODEL_CONFIGS.get("ollama-llama-1b")
            model_name = config.get(role, config.get("general", "llama3.2:1b"))
        elif "qwen3" in config_name.lower():
            config = MODEL_CONFIGS.get("ollama-qwen3-14b")
            model_name = config.get(role, config.get("general", "qwen3:14b"))
        logger.info(
            "Using Ollama model %s for role %s from direct detection", model_name, role
        )
    else:
        # Standard lookup for non-Ollama/non-vLLM models
        config = MODEL_CONFIGS.get(config_name, None)

        # If the config isn't found directly, check if it's an Ollama config by prefix
        if config is None and config_name.lower().startswith("ollama-"):
            logger.info("Detected Ollama config by prefix: %s", config_name)
            config = MODEL_CONFIGS.get(config_name, MODEL_CONFIGS["default"])
        # Fall back to default if still not found
        elif config is None:
            logger.info("Config %s not found, using default", config_name)
            config = MODEL_CONFIGS["default"]

        # Get model for role or fall back to general model
        model_name = config.get(role, config.get("general", DEFAULT_MODEL))

    # Special case for direct matching of deployment names if model_name is already defined
    # (This only applies to the non-Ollama/non-vLLM path since those paths set model_name directly)
    if "model_name" in locals() and config_name == model_name:
        logger.info("Direct model name match: %s = %s", config_name, model_name)
        # This is likely a deployment name being passed directly
        # Check if it's an Ollama or vLLM model
        if is_ollama_model(config_name):
            model_name = config_name  # Use the actual deployment name
            logger.info("Using direct Ollama deployment name: %s", model_name)
        elif is_vllm_model(config_name):
            model_name = config_name  # Use the actual deployment name
            logger.info("Using direct vLLM deployment name: %s", model_name)

    # Return OpenAIChatCompletionsModel if client is provided
    if openai_client:
        return OpenAIChatCompletionsModel(model=model_name, openai_client=openai_client)

    # Otherwise return model name string
    return model_name


def get_model_settings(model_name: str = None) -> ModelSettings:
    """
    Get optimized model settings based on model name or default settings.

    Args:
        model_name: Optional model name to determine settings

    Returns:
        ModelSettings object with appropriate parameters
    """
    if not model_name:
        return MODEL_SETTINGS["default"]

    # Check if we have specific settings for this model
    if model_name in MODEL_SETTINGS:
        return MODEL_SETTINGS[model_name]

    # Special handling for models in our mapping
    for base_model, mapped_model in DEPLOYMENT_MAPPING.items():
        if base_model in model_name:
            model_key = next(
                (k for k in MODEL_SETTINGS.keys() if k.startswith(mapped_model)), None
            )
            if model_key:
                return MODEL_SETTINGS[model_key]

    # Log that we're using default settings for an unknown model
    logger.debug("No specific settings found for model %s, using defaults", model_name)
    return MODEL_SETTINGS["default"]


def get_available_configs() -> Dict[str, Dict[str, str]]:
    """
    Get all available model configurations.

    Returns:
        Dictionary of all model configurations
    """
    return MODEL_CONFIGS


def load_model_config(config_key: str) -> Dict[str, str]:
    """
    Load a specific model configuration by key.

    Args:
        config_key: The configuration key to load (e.g., 'default', 'o3-mini', 'vllm-llama-70b')

    Returns:
        Dictionary containing the model configuration or default if not found
    """
    # Check if it's a vLLM model configuration
    if config_key.lower().startswith("vllm-") or is_vllm_model(config_key):
        # Try to find a matching vLLM config
        if config_key in MODEL_CONFIGS:
            logger.info("Using vLLM config '%s'", config_key)
            return MODEL_CONFIGS[config_key]
        else:
            # Create a default vLLM config using the model name directly
            logger.info("Creating default vLLM config for model %s", config_key)
            return {
                "extractor": config_key,
                "parallel_agent": config_key,
                "judge_agent": config_key,
                "general": config_key,
            }
    
    # Special case for deployment name passed as config key
    # If the model name contains 'deepseek' or 'llama', use appropriate Ollama config
    if "deepseek" in config_key.lower() and "/" not in config_key:
        if "1.5b" in config_key.lower() or "1b" in config_key.lower():
            logger.info("Using 'ollama-deepseek-1b' config for %s", config_key)
            return MODEL_CONFIGS.get("ollama-deepseek-1b")
        elif "8b" in config_key.lower():
            logger.info("Using 'ollama-deepseek-8b' config for %s", config_key)
            return MODEL_CONFIGS.get("ollama-deepseek-8b")
        elif "14b" in config_key.lower():
            logger.info("Using 'ollama-deepseek-14b' config for %s", config_key)
            return MODEL_CONFIGS.get("ollama-deepseek-14b")

    if "llama" in config_key.lower() and "/" not in config_key:
        logger.info("Using 'ollama-llama-1b' config for %s", config_key)
        return MODEL_CONFIGS.get("ollama-llama-1b")

    # Check if the config key exists, otherwise use default
    if config_key not in MODEL_CONFIGS:
        logger.warning("Model configuration '%s' not found, using default", config_key)

        # If no specific Ollama config was matched above but it looks like an Ollama model,
        # default to ollama-deepseek-1b instead of the Azure default
        if is_ollama_model(config_key):
            logger.info(
                "Falling back to 'ollama-deepseek-1b' for Ollama model %s", config_key
            )
            config_key = "ollama-deepseek-1b"
        else:
            config_key = "default"

    # Return the configuration
    return MODEL_CONFIGS[config_key]
