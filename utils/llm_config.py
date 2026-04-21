"""
Simplified LLM configuration for the OpenAI agent.
No Langchain dependencies required.
"""

import os
import logging
from typing import Dict


# Create a simple token handler dictionary for tracking usage
class SimpleTokenHandler:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.reset_usage()

    def reset_usage(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.prompt_cost = 0.0
        self.completion_cost = 0.0
        self.total_cost = 0.0

    def get_overall_costs(self) -> Dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "prompt_cost": self.prompt_cost,
            "completion_cost": self.completion_cost,
            "total_cost": self.total_cost,
        }

    def get_model_costs(self) -> Dict:
        return {
            self.model_name: {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "prompt_cost": self.prompt_cost,
                "completion_cost": self.completion_cost,
                "total_cost": self.total_cost,
            }
        }


# Create handlers for different models
main_token_handler = SimpleTokenHandler("default")
mini_token_handler = SimpleTokenHandler("gpt-4o-mini-jg")
o1_token_handler = SimpleTokenHandler("o1")
o3_mini_token_handler = SimpleTokenHandler("o3-mini-jg")
o4_mini_token_handler = SimpleTokenHandler("o4-mini")

# Placeholder for LLM - this won't be used directly
llm = "This is a placeholder. For OpenAI agent, we don't use Langchain LLMs"

# Export all handlers for usage tracking
token_handlers = {
    "main": main_token_handler,
    "mini": mini_token_handler,
    "o1": o1_token_handler,
    "o3_mini": o3_mini_token_handler,
    "o4_mini": o4_mini_token_handler,
}
