"""vLLM Model Provider for serving models via vLLM's OpenAI-compatible API."""

import os
from typing import Optional, Dict, Any
from openai import AsyncOpenAI

from agents import Model, ModelSettings, ModelProvider, OpenAIChatCompletionsModel


class VLLMModelProvider(ModelProvider):
    """Provider for vLLM-served models using OpenAI-compatible API."""
    
    def __init__(self, base_url: Optional[str] = None):
        """Initialize vLLM provider with endpoint configuration.
        
        Args:
            base_url: vLLM server URL. If not provided, uses VLLM_BASE_URL.
        """
        self.base_url = base_url or os.getenv("VLLM_BASE_URL")
        if not self.base_url:
            raise ValueError(
                "vLLM endpoint is required. Pass --vllm-endpoint (propagates to parameters) or set VLLM_BASE_URL."
            )
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key="dummy-key-for-vllm"  # vLLM requires this but doesn't use it
        )
        print(f"Initialized vLLM provider with base URL: {self.base_url}")
    
    async def get_model(self, model_name: Optional[str] = None) -> Model:
        """Get a model instance for the specified model name.
        
        Args:
            model_name: Name of the model to use (e.g., "meta-llama/Llama-3.1-70B-Instruct")
        
        Returns:
            OpenAIChatCompletionsModel configured for vLLM
        """
        if not model_name:
            raise ValueError("Model name is required for vLLM provider")
        
        # Create model settings optimized for vLLM
        model_settings = self._create_model_settings(model_name)
        
        # Return OpenAI-compatible model wrapper
        return OpenAIChatCompletionsModel(
            openai_client=self.client,
            model=model_name,
        )
    
    def _create_model_settings(self, model_name: str) -> ModelSettings:
        """Create model settings optimized for vLLM.
        
        vLLM supports most OpenAI parameters but has some specific optimizations.
        """
        # Get any custom settings from environment or use defaults
        max_tokens = int(os.getenv("VLLM_MAX_TOKENS", "4096"))
        temperature = float(os.getenv("VLLM_TEMPERATURE", "0.0"))
        
        # vLLM-specific settings can be passed through extra_body
        vllm_settings = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            # vLLM supports these additional parameters:
            # "best_of": 1,  # Number of sequences to generate
            # "use_beam_search": False,
            # "stop_token_ids": [],
            # "skip_special_tokens": True,
        }
        
        return ModelSettings(**vllm_settings)
    
    def supports_structured_output(self) -> bool:
        """Check if vLLM supports structured output.
        
        vLLM supports JSON mode for models that have been trained with it.
        """
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return {
            "provider": "vLLM",
            "base_url": self.base_url,
            "supports_streaming": True,
            "supports_function_calling": False,  # vLLM doesn't support function calling yet
            "supports_json_mode": True,
        }