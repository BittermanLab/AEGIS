"""Configuration file for model costs and other settings."""

# Model-specific costs per million tokens
MODEL_COSTS = {
    "default": {"prompt": 0.15, "completion": 0.60},
    "o1": {"prompt": 15.0, "completion": 60.0},
    "o3_mini": {"prompt": 1.10, "completion": 4.40},
    "o3_early": {"prompt": 3.0, "completion": 15.0},
    "o3_late": {"prompt": 3.0, "completion": 15.0},
    "hybrid": {"prompt": 3.0, "completion": 15.0},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o-mini-jg": {"prompt": 0.15, "completion": 0.60},
    "o3_mini-jg": {"prompt": 1.10, "completion": 4.40},
    # o4
    "o4-mini": {"prompt": 1.10, "completion": 4.40},
    "o4-mini-jg": {"prompt": 1.10, "completion": 4.40},
    # Mini and nano models from GPT-4.1
    "gpt-4.1-mini": {"prompt": 0.4, "completion": 1.60},
    "gpt-4.1-nano": {"prompt": 0.1, "completion": 0.40},
    "4.1-mini": {"prompt": 0.4, "completion": 1.60},
    "4.1-nano": {"prompt": 0.1, "completion": 0.40},
    "gpt-4.1-mini-jg": {"prompt": 0.4, "completion": 1.60},
    "4.1-nano": {"prompt": 0.1, "completion": 0.40},
    "gpt-4.1-nano-jg": {"prompt": 0.1, "completion": 0.40},
    # Local Ollama models (no cost)
    "deepseek-r1:1.5b": {"prompt": 0.0, "completion": 0.0},
    "deepseek-r1:8b": {"prompt": 0.0, "completion": 0.0},
    "deepseek-r1:14b": {"prompt": 0.0, "completion": 0.0},
    "llama3.2:1b": {"prompt": 0.0, "completion": 0.0},
    "qwen3:14b": {"prompt": 0.0, "completion": 0.0},
    # vLLM models (local serving, no API cost)
    "Qwen/Qwen3-8B-FP8": {"prompt": 0.0, "completion": 0.0},
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": {"prompt": 0.0, "completion": 0.0},
    "Qwen/Qwen3-32B-FP8": {"prompt": 0.0, "completion": 0.0},
    "google/gemma-3-27b-it": {"prompt": 0.0, "completion": 0.0},
    "google/medgemma-27b-text-it": {"prompt": 0.0, "completion": 0.0},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {"prompt": 0.0, "completion": 0.0},
}


def get_model_cost_rates(model_name: str) -> dict:
    """Get cost rates for a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary containing prompt and completion costs
    """
    return MODEL_COSTS.get(model_name, MODEL_COSTS["default"])
