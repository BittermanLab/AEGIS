"""
Model definitions for the OpenAI agent integration with the experimental framework.
Implements the required data structures for the parallel agent workflow.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import json
from .model_config import is_ollama_model, is_vllm_model
from utils.config import get_model_cost_rates

# We don't need any complex models, just a dictionary of expected output fields
EXPECTED_OUTPUT_FIELDS = [
    "messages",
    "final_output",
    "token_usage",
    "processing_time",
    "error",
]


@dataclass
class TokenUsageMetadata:
    """Metadata about token usage for a model run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    total_cost: float = 0.0
    model_costs: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "prompt_cost": self.prompt_cost,
            "completion_cost": self.completion_cost,
            "total_cost": self.total_cost,
            "model_costs": self.model_costs or {},
        }


@dataclass
class NoteInput:
    """Input model for a clinical note."""

    pmrn: str
    note_id: str
    note_type: str
    type_name: str
    loc_name: str
    date: str
    prov_name: str
    prov_type: str
    line: int
    note_text: str
    labels: Optional[Dict[str, Any]] = None
    timepoint: Optional[str] = None


@dataclass
class NotePrediction:
    """Prediction output for a clinical note."""

    PMRN: str
    NOTE_ID: str
    NOTE_TEXT: str
    SHORTENED_NOTE_TEXT: str
    MESSAGES: List[Any]
    PREDICTION: Dict[str, Any]
    PROCESSED_AT: datetime
    PROCESSING_TIME: float
    MODEL_NAME: str
    TOKEN_USAGE: Optional[TokenUsageMetadata] = None
    ERROR: Optional[str] = None
    COST_REPORT: Optional[Dict[str, Any]] = None
    TOKEN_TRACKER: Optional[Any] = None


class TokenTracker:
    """
    Tracks token usage from agent completions.
    Extracts actual token counts rather than estimating.
    """

    def __init__(self):
        self.usage_by_model = {}
        self.usage_by_agent = {}
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.logger = logging.getLogger(__name__)

    def _is_local_model(self, model_name: str) -> bool:
        """
        Check if a model is a local model (Ollama or vLLM).
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            bool: True if the model is local, False otherwise
        """
        return is_ollama_model(model_name) or is_vllm_model(model_name)

    def track_usage(
        self, agent_name: str, result: Any, model_name: Optional[str] = None
    ):
        """
        Extract token usage from a completion result and track it

        Args:
            agent_name: Name of the agent for tracking
            result: The completion result containing token usage
            model_name: The model used for this completion
        """
        # Default model name if not provided
        if model_name is None:
            model_name = "unknown"

        # Initialize model in tracking dict if not exists
        if model_name not in self.usage_by_model:
            self.usage_by_model[model_name] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "count": 0,
            }

        # Initialize agent in tracking dict if not exists
        if agent_name not in self.usage_by_agent:
            self.usage_by_agent[agent_name] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "model": model_name,
                "count": 0,
            }

        # Try to extract token usage from different possible locations
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        try:
            # Check if token usage is directly in result
            if hasattr(result, "token_usage"):
                token_usage = result.token_usage
                if token_usage:
                    prompt_tokens = token_usage.get("prompt_tokens", 0)
                    completion_tokens = token_usage.get("completion_tokens", 0)
                    total_tokens = token_usage.get("total_tokens", 0) or (
                        prompt_tokens + completion_tokens
                    )

            # Try agents sdk format where usage might be in usage attribute
            elif hasattr(result, "usage"):
                usage = result.usage
                if usage:
                    prompt_tokens = getattr(usage, "prompt_tokens", 0)
                    completion_tokens = getattr(usage, "completion_tokens", 0)
                    total_tokens = getattr(usage, "total_tokens", 0) or (
                        prompt_tokens + completion_tokens
                    )

            # Check if usage is in response.usage format
            elif hasattr(result, "response") and hasattr(result.response, "usage"):
                usage = result.response.usage
                if usage:
                    prompt_tokens = getattr(usage, "prompt_tokens", 0)
                    completion_tokens = getattr(usage, "completion_tokens", 0)
                    total_tokens = getattr(usage, "total_tokens", 0) or (
                        prompt_tokens + completion_tokens
                    )

            # Check if usage is in messages format (older OpenAI SDK)
            elif hasattr(result, "messages"):
                for message in result.messages:
                    if hasattr(message, "usage") and message.usage:
                        prompt_tokens += getattr(message.usage, "prompt_tokens", 0)
                        completion_tokens += getattr(
                            message.usage, "completion_tokens", 0
                        )
                        message_total = getattr(message.usage, "total_tokens", 0) or (
                            getattr(message.usage, "prompt_tokens", 0)
                            + getattr(message.usage, "completion_tokens", 0)
                        )
                        total_tokens += message_total

            # Get cost rates based on whether it's a local model
            if self._is_local_model(model_name):
                cost_rates = {"prompt": 0.0, "completion": 0.0}
            else:
                cost_rates = get_model_cost_rates(model_name)

            # Calculate costs
            prompt_cost = (prompt_tokens / 1000000) * cost_rates["prompt"]
            completion_cost = (completion_tokens / 1000000) * cost_rates["completion"]
            total_cost = prompt_cost + completion_cost

            # Update tracking dictionaries with costs
            self.usage_by_model[model_name].update({
                "prompt_tokens": self.usage_by_model[model_name]["prompt_tokens"] + prompt_tokens,
                "completion_tokens": self.usage_by_model[model_name]["completion_tokens"] + completion_tokens,
                "total_tokens": self.usage_by_model[model_name]["total_tokens"] + total_tokens,
                "count": self.usage_by_model[model_name]["count"] + 1,
                "prompt_cost": self.usage_by_model[model_name].get("prompt_cost", 0.0) + prompt_cost,
                "completion_cost": self.usage_by_model[model_name].get("completion_cost", 0.0) + completion_cost,
                "total_cost": self.usage_by_model[model_name].get("total_cost", 0.0) + total_cost
            })

            self.usage_by_agent[agent_name].update({
                "prompt_tokens": self.usage_by_agent[agent_name]["prompt_tokens"] + prompt_tokens,
                "completion_tokens": self.usage_by_agent[agent_name]["completion_tokens"] + completion_tokens,
                "total_tokens": self.usage_by_agent[agent_name]["total_tokens"] + total_tokens,
                "count": self.usage_by_agent[agent_name]["count"] + 1,
                "prompt_cost": self.usage_by_agent[agent_name].get("prompt_cost", 0.0) + prompt_cost,
                "completion_cost": self.usage_by_agent[agent_name].get("completion_cost", 0.0) + completion_cost,
                "total_cost": self.usage_by_agent[agent_name].get("total_cost", 0.0) + total_cost
            })

            # Update totals
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total_tokens

        except Exception as e:
            self.logger.error(
                f"Error extracting token usage for {agent_name}: {str(e)}"
            )

    def debug_token_extraction(self, result: Any):
        """
        Debug helper to print the structure of a result object to help with token extraction

        Args:
            result: The completion result to debug
        """
        self.logger.debug("===== TOKEN EXTRACTION DEBUG =====")

        # Try to identify where token information might be
        if hasattr(result, "token_usage"):
            self.logger.debug(f"result.token_usage: {result.token_usage}")

        if hasattr(result, "usage"):
            self.logger.debug(f"result.usage: {result.usage}")

        if hasattr(result, "response") and hasattr(result.response, "usage"):
            self.logger.debug(f"result.response.usage: {result.response.usage}")

        if hasattr(result, "messages"):
            self.logger.debug(
                f"result.messages[0].usage: {getattr(result.messages[0], 'usage', None) if result.messages else None}"
            )

        # Log object structure
        try:
            # Try to log vars or __dict__ to see structure
            if hasattr(result, "__dict__"):
                self.logger.debug(f"Result structure: {list(result.__dict__.keys())}")
            else:
                self.logger.debug(f"Result has no __dict__, type: {type(result)}")
        except Exception as e:
            self.logger.debug(f"Error examining result structure: {str(e)}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of token usage

        Returns:
            Dictionary with usage statistics
        """
        total_prompt_cost = sum(
            usage.get("prompt_cost", 0.0) for usage in self.usage_by_model.values()
        )
        total_completion_cost = sum(
            usage.get("completion_cost", 0.0) for usage in self.usage_by_model.values()
        )
        total_cost = sum(
            usage.get("total_cost", 0.0) for usage in self.usage_by_model.values()
        )

        return {
            "by_model": self.usage_by_model,
            "by_agent": self.usage_by_agent,
            "total": {
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
                "prompt_cost": total_prompt_cost,
                "completion_cost": total_completion_cost,
                "total_cost": total_cost,
            },
            "timestamp": datetime.now().isoformat(),
        }

    def save_summary(self, path: str):
        """
        Save token usage summary to a JSON file

        Args:
            path: Path to save the JSON file
        """
        try:
            with open(path, "w") as f:
                json.dump(self.get_summary(), f, indent=2)
            self.logger.info(f"Token usage summary saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving token usage summary: {str(e)}")
