"""
Model configurations for the OpenAI zeroshot agent.

This module defines the model configurations for the OpenAI zeroshot implementation.
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, Literal

logger = logging.getLogger(__name__)

# Define provider types
ProviderType = Literal["azure", "ollama", "vllm"]


@dataclass
class OpenAIModelConfig:
    """Configuration for an OpenAI model."""

    model_name: str  # Name of the model to use
    temperature: float = 0.0  # Temperature for generation
    top_p: float = 1.0  # Top-p for generation
    presence_penalty: float = 0.0  # Presence penalty
    frequency_penalty: float = 0.0  # Frequency penalty


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

# Default model configuration
default_config = OpenAIModelConfig(
    model_name="gpt-4o-mini-jg",
)

# Claude-3 Opus model configuration
o1_config = OpenAIModelConfig(model_name="o1")

# Claude-3 Sonnet model configuration
o3_mini_config = OpenAIModelConfig(model_name="o3-mini-jg")

# GPT-3.5 Turbo model configuration
turbo_config = OpenAIModelConfig(model_name="gpt-3.5-turbo")

# Ollama model configurations
ollama_deepseek_small_config = OpenAIModelConfig(model_name="deepseek-r1:1.5b")
ollama_deepseek_medium_config = OpenAIModelConfig(model_name="deepseek-r1:8b")
ollama_deepseek_large_config = OpenAIModelConfig(model_name="deepseek-r1:14b")
ollama_llama_small_config = OpenAIModelConfig(model_name="llama3.2:1b")
ollama_qwen_config = OpenAIModelConfig(model_name="qwen3:14b")

# vLLM model configurations
vllm_deepseek_r1_8b_config = OpenAIModelConfig(
    model_name="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
)
vllm_qwen3_8b_config = OpenAIModelConfig(model_name="Qwen/Qwen3-8B-FP8")
vllm_qwen3_14b_config = OpenAIModelConfig(model_name="Qwen/Qwen3-14B-FP8")
vllm_qwen3_32b_config = OpenAIModelConfig(model_name="Qwen/Qwen3-32B-FP8")
vllm_gemma3_4b_config = OpenAIModelConfig(model_name="google/gemma-3-4b-it")
vllm_gemma3_12b_config = OpenAIModelConfig(model_name="google/gemma-3-12b-it")
vllm_gemma3_27b_config = OpenAIModelConfig(model_name="google/gemma-3-27b-it")
vllm_medgemma_27b_config = OpenAIModelConfig(model_name="google/medgemma-27b-text-it")


# Export model configurations
MODEL_CONFIGS = {
    "default": default_config,
    "4.1-mini": OpenAIModelConfig(model_name="gpt-4.1-mini"),
    "4.1-nano": OpenAIModelConfig(model_name="gpt-4.1-nano"),
    "o4-mini": OpenAIModelConfig(model_name="o4-mini"),
    "o1": o1_config,
    "o3_mini": o3_mini_config,
    "turbo": turbo_config,
    # Ollama configurations
    "ollama-deepseek-1b": ollama_deepseek_small_config,
    "ollama-deepseek-8b": ollama_deepseek_medium_config,
    "ollama-deepseek-14b": ollama_deepseek_large_config,
    "ollama-llama-1b": ollama_llama_small_config,
    "ollama-qwen3-14b": ollama_qwen_config,
    # Canonical Ollama names
    "deepseek-r1:1.5b": ollama_deepseek_small_config,
    "deepseek-r1:8b": ollama_deepseek_medium_config,
    "deepseek-r1:14b": ollama_deepseek_large_config,
    "deepseek-1b": ollama_deepseek_small_config,
    "deepseek-8b": ollama_deepseek_medium_config,
    "deepseek-14b": ollama_deepseek_large_config,
    "llama3.2:1b": ollama_llama_small_config,
    # vLLM configurations
    "vllm-deepseek-r1-8b": vllm_deepseek_r1_8b_config,
    "vllm-qwen3-8b": vllm_qwen3_8b_config,
    "vllm-qwen3-14b": vllm_qwen3_14b_config,
    "vllm-qwen3-32b": vllm_qwen3_32b_config,
    "vllm-gemma3-4b": vllm_gemma3_4b_config,
    "vllm-gemma3-12b": vllm_gemma3_12b_config,
    "vllm-gemma3-27b": vllm_gemma3_27b_config,
    "vllm-medgemma-27b": vllm_medgemma_27b_config,
    # Canonical vLLM names
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": vllm_deepseek_r1_8b_config,
    "Qwen/Qwen3-8B-FP8": vllm_qwen3_8b_config,
    "Qwen/Qwen3-14B-FP8": vllm_qwen3_14b_config,
    "Qwen/Qwen3-32B-FP8": vllm_qwen3_32b_config,
    "google/gemma-3-4b-it": vllm_gemma3_4b_config,
    "google/gemma-3-12b-it": vllm_gemma3_12b_config,
    "google/gemma-3-27b-it": vllm_gemma3_27b_config,
    "google/medgemma-27b-text-it": vllm_medgemma_27b_config,
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

    # Check if it's prefixed with "ollama-"
    if model_name.lower().startswith("ollama-"):
        return True

    # Check against our list of Ollama models
    return any(
        model_name.lower() == ollama_model.lower() for ollama_model in OLLAMA_MODELS
    )


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
        return True

    # Check if model name contains organization/model pattern (common for HuggingFace models served by vLLM)
    if "/" in model_name and any(
        org in model_name.lower()
        for org in ["meta-llama", "mistralai", "deepseek-ai", "qwen", "google"]
    ):
        return True

    # Check against our list of vLLM models
    return any(model_name.lower() == vllm_model.lower() for vllm_model in VLLM_MODELS)


def get_provider_type(model_name: str) -> ProviderType:
    """
    Determine which provider type to use for a given model name.

    Args:
        model_name: Name of the model

    Returns:
        ProviderType: Either 'azure', 'ollama', or 'vllm'
    """
    if is_vllm_model(model_name):
        return "vllm"
    elif is_ollama_model(model_name):
        return "ollama"
    else:
        return "azure"


# Log available configurations
logger.debug(
    f"OpenAI zeroshot model configurations loaded: {list(MODEL_CONFIGS.keys())}"
)
