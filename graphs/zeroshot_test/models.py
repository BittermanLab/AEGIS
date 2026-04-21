"""
Model definitions for the OpenAI agent integration with the experimental framework.
Implements the required data structures for the parallel agent workflow.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging
import json
import os
from utils.config import MODEL_COSTS, get_model_cost_rates

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
        self.total_prompt_cost = 0.0
        self.total_completion_cost = 0.0
        self.total_cost = 0.0
        self.logger = logging.getLogger(__name__)

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

        # Get cost rates for this model
        cost_rates = self._get_model_cost_rates(model_name)
        prompt_rate = cost_rates["prompt"]
        completion_rate = cost_rates["completion"]

        # Initialize model in tracking dict if not exists
        if model_name not in self.usage_by_model:
            self.usage_by_model[model_name] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_cost": 0.0,
                "completion_cost": 0.0,
                "total_cost": 0.0,
                "count": 0,
                "rates": {
                    "prompt": prompt_rate,
                    "completion": completion_rate,
                    "unit": "USD per 1M tokens",
                },
            }

        # Initialize agent in tracking dict if not exists
        if agent_name not in self.usage_by_agent:
            self.usage_by_agent[agent_name] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_cost": 0.0,
                "completion_cost": 0.0,
                "total_cost": 0.0,
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

            # Calculate costs
            prompt_cost = (prompt_tokens / 1000000.0) * prompt_rate
            completion_cost = (completion_tokens / 1000000.0) * completion_rate
            total_cost = prompt_cost + completion_cost

            # Update tracking dictionaries with tokens and costs
            self.usage_by_model[model_name]["prompt_tokens"] += prompt_tokens
            self.usage_by_model[model_name]["completion_tokens"] += completion_tokens
            self.usage_by_model[model_name]["total_tokens"] += total_tokens
            self.usage_by_model[model_name]["prompt_cost"] += prompt_cost
            self.usage_by_model[model_name]["completion_cost"] += completion_cost
            self.usage_by_model[model_name]["total_cost"] += total_cost
            self.usage_by_model[model_name]["count"] += 1

            self.usage_by_agent[agent_name]["prompt_tokens"] += prompt_tokens
            self.usage_by_agent[agent_name]["completion_tokens"] += completion_tokens
            self.usage_by_agent[agent_name]["total_tokens"] += total_tokens
            self.usage_by_agent[agent_name]["prompt_cost"] += prompt_cost
            self.usage_by_agent[agent_name]["completion_cost"] += completion_cost
            self.usage_by_agent[agent_name]["total_cost"] += total_cost
            self.usage_by_agent[agent_name]["count"] += 1

            # Update totals
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            self.total_prompt_cost += prompt_cost
            self.total_completion_cost += completion_cost
            self.total_cost += total_cost

            # Log the cost information
            self.logger.info(
                f"Model: {model_name}, Agent: {agent_name}, "
                f"Tokens: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total, "
                f"Cost: ${prompt_cost:.6f} + ${completion_cost:.6f} = ${total_cost:.6f}"
            )

        except Exception as e:
            self.logger.error(
                f"Error extracting token usage for {agent_name}: {str(e)}"
            )

    def _get_model_cost_rates(self, model_name: str) -> Dict[str, float]:
        """
        Get the cost rates for a specific model

        Args:
            model_name: The name of the model

        Returns:
            Dictionary with prompt and completion rates
        """
        # Use the centralized get_model_cost_rates function
        return get_model_cost_rates(model_name)

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
        return {
            "by_model": self.usage_by_model,
            "by_agent": self.usage_by_agent,
            "total": {
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
                "prompt_cost": self.total_prompt_cost,
                "completion_cost": self.total_completion_cost,
                "total_cost": self.total_cost,
                "cost_unit": "USD",
            },
            "timestamp": datetime.now().isoformat(),
        }

    def save_summary(self, path: str, model_name: str = "unknown"):
        """
        Save token usage summary to a JSON file

        Args:
            path: Path to save the JSON file
            model_name: Name of the model for directory organization
        """
        try:
            # Create a timestamped directory with the model name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = os.path.dirname(path)

            # Create base token_usage dir if it doesn't exist
            if not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=True)

            # Create timestamped model subdirectory
            model_dir = os.path.join(base_dir, f"{model_name}_{timestamp}")
            os.makedirs(model_dir, exist_ok=True)

            # Create final output path
            filename = os.path.basename(path)
            final_path = os.path.join(model_dir, filename)

            # Save the summary to the file
            with open(final_path, "w") as f:
                json.dump(self.get_summary(), f, indent=2)

            # Also save a copy of the summary with a standard name in the model directory
            summary_path = os.path.join(model_dir, "token_usage_summary.json")
            with open(summary_path, "w") as f:
                json.dump(self.get_summary(), f, indent=2)

            self.logger.info(f"Token usage summary saved to {final_path}")
            return final_path
        except Exception as e:
            self.logger.error(f"Error saving token usage summary: {str(e)}")
            # Fallback to original path
            try:
                with open(path, "w") as f:
                    json.dump(self.get_summary(), f, indent=2)
                self.logger.info(f"Token usage summary saved to fallback path {path}")
                return path
            except Exception as e2:
                self.logger.error(f"Error saving to fallback path: {str(e2)}")
                return None
