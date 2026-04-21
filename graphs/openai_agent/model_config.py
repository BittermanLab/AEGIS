"""
Model configurations for the OpenAI Agent SDK implementation.

This module defines model configurations with optimized settings for different
use cases, cost profiles, and quality requirements.
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Literal, Union

from agents.model_settings import ModelSettings

logger = logging.getLogger(__name__)

# Define provider types
ProviderType = Literal["azure", "ollama", "vllm"]


@dataclass
class OpenAIModelConfig:
    """
    Configuration for an OpenAI model with Agent SDK settings.

    Attributes:
        model_name: Name of the model to use
        temperature: Sampling temperature (0.0 = deterministic)
        top_p: Nucleus sampling parameter
        presence_penalty: Repetition penalty for token presence
        frequency_penalty: Repetition penalty for token frequency
    """

    model_name: str
    temperature: float = 0.0
    top_p: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    def to_model_settings(self) -> ModelSettings:
        """Convert to ModelSettings for OpenAI Agent SDK"""
        return ModelSettings(
            temperature=self.temperature,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
        )


# Environment-based default model
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gpt-4o-mini-jg")

# List of Ollama model names to identify which models should use Ollama provider
OLLAMA_MODELS = [
    "deepseek-r1:1.5b",
    "deepseek-r1:8b",
    "deepseek-r1:14b",
    "deepseek-1b",
    "deepseek-8b",
    "deepseek-14b",
    "llama3.2:1b",
    "qwen2.5:7b",
    "mistral:7b",
    "llama3:8b",
    "llama3:70b",
    "qwen3:14b",
]

# List of vLLM model names to identify which models should use vLLM provider
VLLM_MODELS = [
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "Qwen/Qwen3-8B-FP8",
    "Qwen/Qwen3-14B-FP8",
    "Qwen/Qwen3-32B-FP8",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "google/medgemma-27b-text-it",
]

# Standard configurations
DEFAULT_CONFIG = OpenAIModelConfig(model_name=DEFAULT_MODEL)
O1_CONFIG = OpenAIModelConfig(model_name="o1")
O3_MINI_CONFIG = OpenAIModelConfig(model_name="o3-mini")
O4_MINI_CONFIG = OpenAIModelConfig(model_name="o4-mini")
GPT4O_CONFIG = OpenAIModelConfig(model_name="gpt-4o")
GPT4O_MINI_CONFIG = OpenAIModelConfig(model_name="gpt-4o-mini")
GPT41_MINI_CONFIG = OpenAIModelConfig(model_name="gpt-4.1-mini")
GPT41_NANO_CONFIG = OpenAIModelConfig(model_name="gpt-4.1-nano")

# Ollama model configurations
DEEPSEEK_SMALL_CONFIG = OpenAIModelConfig(model_name="deepseek-r1:1.5b")
DEEPSEEK_MEDIUM_CONFIG = OpenAIModelConfig(model_name="deepseek-r1:8b")
DEEPSEEK_LARGE_CONFIG = OpenAIModelConfig(model_name="deepseek-r1:14b")
LLAMA_SMALL_CONFIG = OpenAIModelConfig(model_name="llama3.2:1b")
QWEN_MED_CONFIG = OpenAIModelConfig(model_name="qwen3:14b")

# vLLM model configurations
VLLM_DEEPSEEK_R1_8B_CONFIG = OpenAIModelConfig(model_name="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
VLLM_QWEN3_8B_CONFIG = OpenAIModelConfig(model_name="Qwen/Qwen3-8B-FP8")
VLLM_QWEN3_8B_LOCAL_CONFIG = OpenAIModelConfig(model_name="Qwen/Qwen3-8B")
VLLM_QWEN3_14B_CONFIG = OpenAIModelConfig(model_name="Qwen/Qwen3-14B-FP8")
VLLM_QWEN3_32B_CONFIG = OpenAIModelConfig(model_name="Qwen/Qwen3-32B-FP8")
VLLM_GEMMA3_4B_CONFIG = OpenAIModelConfig(model_name="google/gemma-3-4b-it")
VLLM_GEMMA3_12B_CONFIG = OpenAIModelConfig(model_name="google/gemma-3-12b-it")
VLLM_GEMMA3_27B_CONFIG = OpenAIModelConfig(model_name="google/gemma-3-27b-it")
VLLM_MEDGEMMA_27B_CONFIG = OpenAIModelConfig(model_name="google/medgemma-27b-text-it")

# Role-specific configurations
EXTRACTION_CONFIG = OpenAIModelConfig(
    model_name="gpt-4.1-mini", temperature=0.0  # More deterministic for extraction
)

JUDGE_CONFIG = OpenAIModelConfig(
    model_name="gpt-4.1-mini", temperature=0.0  # More deterministic for judgments
)

GRADING_CONFIG = OpenAIModelConfig(
    model_name="gpt-4.1-mini", temperature=0.0  # More deterministic for judgments
)

# Additional model configurations seen in base_config.yaml
O3_MINI_HYPHENATED_CONFIG = OpenAIModelConfig(model_name="o3-mini")
O3_JUDGE_CONFIG = OpenAIModelConfig(model_name="o3-judge")
O3_MIDDLE_CONFIG = OpenAIModelConfig(model_name="o3-middle")
GPT4O_MINI_HYPHENATED_CONFIG = OpenAIModelConfig(model_name="gpt-4o-mini")
GPT41_MINI_HYPHENATED_CONFIG = OpenAIModelConfig(model_name="gpt-4.1-mini")
GPT41_NANO_HYPHENATED_CONFIG = OpenAIModelConfig(model_name="gpt-4.1-nano")

# Export all model configurations
MODEL_CONFIGS = {
    "default": DEFAULT_CONFIG,
    "o1": O1_CONFIG,
    "o3_mini": O3_MINI_CONFIG,
    "o4_mini": O4_MINI_CONFIG,
    "gpt4o": GPT4O_CONFIG,
    "gpt4o_mini": GPT4O_MINI_CONFIG,
    "gpt41_mini": GPT41_MINI_CONFIG,
    "gpt41_nano": GPT41_NANO_CONFIG,
    "gpt-4.1-mini": GPT41_MINI_CONFIG,
    "gpt-4.1-nano": GPT41_NANO_CONFIG,
    "4.1-mini": GPT41_MINI_CONFIG,
    "4.1-nano": GPT41_NANO_CONFIG,
    "extraction": EXTRACTION_CONFIG,
    "judge": JUDGE_CONFIG,
    "grading": GRADING_CONFIG,
    # Add new configurations
    "4o-mini": GPT4O_MINI_HYPHENATED_CONFIG,
    "o3-mini": O3_MINI_HYPHENATED_CONFIG,
    "o4-mini": O4_MINI_CONFIG,
    "o3-judge": O3_JUDGE_CONFIG,
    "o3-middle": O3_MIDDLE_CONFIG,
    "4.1-mini": GPT41_MINI_HYPHENATED_CONFIG,
    "4.1-nano": GPT41_NANO_HYPHENATED_CONFIG,
    # Ollama configurations with more specific names and aliases
    "ollama-deepseek-1b": DEEPSEEK_SMALL_CONFIG,
    "ollama-deepseek-8b": DEEPSEEK_MEDIUM_CONFIG,
    "ollama-deepseek-14b": DEEPSEEK_LARGE_CONFIG,
    "ollama-llama-1b": LLAMA_SMALL_CONFIG,
    "ollama-qwen3-14b": QWEN_MED_CONFIG,
    # Canonical Ollama names
    "deepseek-r1:1.5b": DEEPSEEK_SMALL_CONFIG,
    "deepseek-r1:8b": DEEPSEEK_MEDIUM_CONFIG,
    "deepseek-r1:14b": DEEPSEEK_LARGE_CONFIG,
    "deepseek-1b": DEEPSEEK_SMALL_CONFIG,
    "deepseek-8b": DEEPSEEK_MEDIUM_CONFIG,
    "deepseek-14b": DEEPSEEK_LARGE_CONFIG,
    "llama3.2:1b": LLAMA_SMALL_CONFIG,
    # vLLM configurations
    "vllm-deepseek-r1-8b": VLLM_DEEPSEEK_R1_8B_CONFIG,
    "vllm-qwen3-8b": VLLM_QWEN3_8B_CONFIG,
    "vllm-qwen3-8b-local": VLLM_QWEN3_8B_LOCAL_CONFIG,
    "vllm-qwen3-14b": VLLM_QWEN3_14B_CONFIG,
    "vllm-qwen3-32b": VLLM_QWEN3_32B_CONFIG,
    "vllm-gemma3-4b": VLLM_GEMMA3_4B_CONFIG,
    "vllm-gemma3-12b": VLLM_GEMMA3_12B_CONFIG,
    "vllm-gemma3-27b": VLLM_GEMMA3_27B_CONFIG,
    "vllm-medgemma-27b": VLLM_MEDGEMMA_27B_CONFIG,
    # Canonical vLLM names
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": VLLM_DEEPSEEK_R1_8B_CONFIG,
    "Qwen/Qwen3-8B-FP8": VLLM_QWEN3_8B_CONFIG,
    "Qwen/Qwen3-8B": VLLM_QWEN3_8B_LOCAL_CONFIG,
    "Qwen/Qwen3-14B-FP8": VLLM_QWEN3_14B_CONFIG,
    "Qwen/Qwen3-32B-FP8": VLLM_QWEN3_32B_CONFIG,
    "google/gemma-3-4b-it": VLLM_GEMMA3_4B_CONFIG,
    "google/gemma-3-12b-it": VLLM_GEMMA3_12B_CONFIG,
    "google/gemma-3-27b-it": VLLM_GEMMA3_27B_CONFIG,
    "google/medgemma-27b-text-it": VLLM_MEDGEMMA_27B_CONFIG,
}

# Model-specific parameter settings for direct lookup
MODEL_SETTINGS = {
    "gpt-4o-mini-jg": ModelSettings(temperature=0.0, top_p=1.0),
    "o3-mini-jg": ModelSettings(),
    "o4-mini": ModelSettings(),
    "gpt-4.1-mini": ModelSettings(temperature=0.0, top_p=1.0),
    "gpt-4.1-nano": ModelSettings(temperature=0.0, top_p=1.0),
    "deepseek-r1:1.5b": ModelSettings(temperature=0.0, top_p=1.0),
    "deepseek-r1:8b": ModelSettings(temperature=0.0, top_p=1.0),
    "deepseek-r1:14b": ModelSettings(temperature=0.0, top_p=1.0),
    "deepseek-1b": ModelSettings(),
    "deepseek-8b": ModelSettings(),
    "deepseek-14b": ModelSettings(),
    "llama3.2:1b": ModelSettings(temperature=0.0, top_p=1.0),
    # Add aliases for sweep configs
    "ollama-deepseek-1b": ModelSettings(),
    "ollama-deepseek-8b": ModelSettings(),
    "ollama-deepseek-14b": ModelSettings(),
    "ollama-llama-1b": ModelSettings(),
    "default": ModelSettings(temperature=0.0, top_p=1.0),
    # vLLM model settings
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": ModelSettings(temperature=0.0, top_p=1.0),
    "Qwen/Qwen3-8B-FP8": ModelSettings(temperature=0.0, top_p=1.0),
    "Qwen/Qwen3-14B-FP8": ModelSettings(temperature=0.0, top_p=1.0),
    "Qwen/Qwen3-32B-FP8": ModelSettings(temperature=0.0, top_p=1.0),
    "google/gemma-3-4b-it": ModelSettings(temperature=0.0, top_p=1.0),
    "google/gemma-3-12b-it": ModelSettings(temperature=0.0, top_p=1.0),
    "google/gemma-3-27b-it": ModelSettings(temperature=0.0, top_p=1.0),
    "google/medgemma-27b-text-it": ModelSettings(temperature=0.0, top_p=1.0),
    "vllm-deepseek-r1-8b": ModelSettings(temperature=0.0, top_p=1.0),
    "vllm-qwen3-8b": ModelSettings(temperature=0.0, top_p=1.0),
    "vllm-qwen3-14b": ModelSettings(temperature=0.0, top_p=1.0),
    "vllm-qwen3-32b": ModelSettings(temperature=0.0, top_p=1.0),
    "vllm-gemma3-4b": ModelSettings(temperature=0.0, top_p=1.0),
    "vllm-gemma3-12b": ModelSettings(temperature=0.0, top_p=1.0),
    "vllm-gemma3-27b": ModelSettings(temperature=0.0, top_p=1.0),
    "vllm-medgemma-27b": ModelSettings(temperature=0.0, top_p=1.0),
}


# Normalize model config key (handles aliases)
def normalize_model_key(model_name: str) -> str:
    name = model_name.lower().replace("_", "-")
    if name.startswith("ollama-"):
        name = name[len("ollama-") :]
    # Map ollama-deepseek-1b and deepseek-1b to canonical
    alias_map = {
        "deepseek-1b": "deepseek-r1:1.5b",
        "deepseek-8b": "deepseek-r1:8b",
        "deepseek-14b": "deepseek-r1:14b",
    }
    return alias_map.get(name, name)


# Lookup with normalization and error on missing
def get_model_config(key: str) -> OpenAIModelConfig:
    """
    Get a model configuration by key.

    Args:
        key: Configuration key

    Returns:
        OpenAIModelConfig: The model configuration
    """
    norm_key = normalize_model_key(key)
    if norm_key in MODEL_CONFIGS:
        return MODEL_CONFIGS[norm_key]
    else:
        raise ValueError(
            f"Model configuration '{key}' (normalized: '{norm_key}') not found. Please add it to MODEL_CONFIGS."
        )


def is_ollama_model(model_name: str) -> bool:
    """
    Check if a model name corresponds to an Ollama model.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        bool: True if the model is an Ollama model, False otherwise
    """
    return model_name in OLLAMA_MODELS


def is_vllm_model(model_name: str) -> bool:
    """
    Check if a model name corresponds to a vLLM model.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        bool: True if the model is a vLLM model, False otherwise
    """
    return model_name in VLLM_MODELS


def get_provider_type(model_name: str) -> ProviderType:
    """
    Determine which provider type to use for a given model name.

    Args:
        model_name: Name of the model

    Returns:
        ProviderType: Either 'azure' or 'ollama'
    """
    if is_ollama_model(model_name):
        return "ollama"
    else:
        return "azure"


# Function to add a new Ollama model to the registry
def register_ollama_model(model_name: str) -> None:
    """
    Register a new Ollama model in the OLLAMA_MODELS list.

    Args:
        model_name: Name of the Ollama model to register
    """
    if model_name not in OLLAMA_MODELS:
        OLLAMA_MODELS.append(model_name)
        logger.info(f"Registered new Ollama model: {model_name}")


# Export symbols
__all__ = [
    "OpenAIModelConfig",
    "MODEL_CONFIGS",
    "get_model_config",
    "is_ollama_model",
    "is_vllm_model",
    "get_provider_type",
    "register_ollama_model",
    "OLLAMA_MODELS",
]

# Log available configurations on module load
logger.info(f"Model configurations available: {list(MODEL_CONFIGS.keys())}")
