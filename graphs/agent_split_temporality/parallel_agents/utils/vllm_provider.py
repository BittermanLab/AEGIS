"""vLLM Model Provider for serving models via vLLM's OpenAI-compatible API."""

import os
from typing import Optional, Dict, Any
from openai import AsyncOpenAI

from agents import Model, ModelProvider, OpenAIChatCompletionsModel, set_default_openai_api


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
        
        # Force Agents SDK to use chat-completions flavour (following Azure pattern)
        set_default_openai_api("chat_completions")
        
        print(f"Initialized vLLM provider with base URL: {self.base_url}")
    
    def get_model(self, model_name: Optional[str] = None) -> Model:
        """Get a model instance for the specified model name.
        
        Args:
            model_name: Name of the model to use (e.g., "meta-llama/Llama-3.1-70B-Instruct")
        
        Returns:
            OpenAIChatCompletionsModel configured for vLLM
        """
        if not model_name:
            raise ValueError("Model name is required for vLLM provider")
        
        # Create client for this model (following Azure pattern)
        client = AsyncOpenAI(
            base_url=self.base_url,
            api_key="dummy-key-for-vllm"  # vLLM requires this but doesn't use it
        )
        
        # Return OpenAI-compatible model wrapper (following Azure pattern)
        return OpenAIChatCompletionsModel(
            model=model_name,
            openai_client=client,
        )
    
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