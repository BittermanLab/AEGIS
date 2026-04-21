"""
Model definitions for the OpenAI agent integration with the experimental framework.
Implements the required data structures for the parallel agent workflow.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import json

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
                    # Check for Azure format with input_tokens/output_tokens
                    if hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
                        prompt_tokens = getattr(usage, "input_tokens", 0)
                        completion_tokens = getattr(usage, "output_tokens", 0)
                        total_tokens = getattr(usage, "total_tokens", 0) or (
                            prompt_tokens + completion_tokens
                        )
                    else:
                        # Standard OpenAI format
                        prompt_tokens = getattr(usage, "prompt_tokens", 0)
                        completion_tokens = getattr(usage, "completion_tokens", 0)
                        total_tokens = getattr(usage, "total_tokens", 0) or (
                            prompt_tokens + completion_tokens
                        )

            # Check if usage is in response.usage format
            elif hasattr(result, "response") and hasattr(result.response, "usage"):
                usage = result.response.usage
                if usage:
                    # Check for Azure format with input_tokens/output_tokens
                    if hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
                        prompt_tokens = getattr(usage, "input_tokens", 0)
                        completion_tokens = getattr(usage, "output_tokens", 0)
                        total_tokens = getattr(usage, "total_tokens", 0) or (
                            prompt_tokens + completion_tokens
                        )
                    else:
                        # Standard OpenAI format
                        prompt_tokens = getattr(usage, "prompt_tokens", 0)
                        completion_tokens = getattr(usage, "completion_tokens", 0)
                        total_tokens = getattr(usage, "total_tokens", 0) or (
                            prompt_tokens + completion_tokens
                        )

            # Try to look for raw_responses in OpenAI Agents SDK format
            elif hasattr(result, "raw_responses") and result.raw_responses:
                self.logger.debug("===== TOKEN EXTRACTION DEBUG =====")
                self.logger.debug(f"Result structure: {list(vars(result).keys()) if hasattr(result, '__dict__') else 'No __dict__'}")
                
                # Extract from raw_responses
                for response in result.raw_responses:
                    if hasattr(response, "usage"):
                        usage = response.usage
                        # Check for Azure format with input_tokens/output_tokens
                        if hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
                            prompt_tokens += getattr(usage, "input_tokens", 0)
                            completion_tokens += getattr(usage, "output_tokens", 0)
                            total_tokens += getattr(usage, "total_tokens", 0) or (
                                getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)
                            )
                            self.logger.debug(f"Extracted Azure usage from raw_response: {prompt_tokens}p + {completion_tokens}c = {total_tokens}t")
                        else:
                            # Standard OpenAI format
                            prompt_tokens += getattr(usage, "prompt_tokens", 0)
                            completion_tokens += getattr(usage, "completion_tokens", 0)
                            total_tokens += getattr(usage, "total_tokens", 0)
                            self.logger.debug(f"Extracted usage from raw_response: {prompt_tokens}p + {completion_tokens}c = {total_tokens}t")
                    elif hasattr(response, "response") and hasattr(response.response, "usage"):
                        usage = response.response.usage
                        # Check for Azure format with input_tokens/output_tokens
                        if hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
                            prompt_tokens += getattr(usage, "input_tokens", 0)
                            completion_tokens += getattr(usage, "output_tokens", 0)
                            total_tokens += getattr(usage, "total_tokens", 0) or (
                                getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)
                            )
                            self.logger.debug(f"Extracted Azure usage from response.response: {prompt_tokens}p + {completion_tokens}c = {total_tokens}t")
                        else:
                            # Standard OpenAI format
                            prompt_tokens += getattr(usage, "prompt_tokens", 0)
                            completion_tokens += getattr(usage, "completion_tokens", 0)
                            total_tokens += getattr(usage, "total_tokens", 0)
                            self.logger.debug(f"Extracted usage from response.response: {prompt_tokens}p + {completion_tokens}c = {total_tokens}t")

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

            # Update tracking dictionaries
            self.usage_by_model[model_name]["prompt_tokens"] += prompt_tokens
            self.usage_by_model[model_name]["completion_tokens"] += completion_tokens
            self.usage_by_model[model_name]["total_tokens"] += total_tokens
            self.usage_by_model[model_name]["count"] += 1

            self.usage_by_agent[agent_name]["prompt_tokens"] += prompt_tokens
            self.usage_by_agent[agent_name]["completion_tokens"] += completion_tokens
            self.usage_by_agent[agent_name]["total_tokens"] += total_tokens
            self.usage_by_agent[agent_name]["count"] += 1

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
            usage = result.usage
            if usage:
                # Check both formats
                if hasattr(usage, "prompt_tokens"):
                    self.logger.debug(f"  - prompt_tokens: {getattr(usage, 'prompt_tokens', 'N/A')}")
                if hasattr(usage, "completion_tokens"):
                    self.logger.debug(f"  - completion_tokens: {getattr(usage, 'completion_tokens', 'N/A')}")
                if hasattr(usage, "input_tokens"):
                    self.logger.debug(f"  - input_tokens: {getattr(usage, 'input_tokens', 'N/A')}")
                if hasattr(usage, "output_tokens"):
                    self.logger.debug(f"  - output_tokens: {getattr(usage, 'output_tokens', 'N/A')}")
                if hasattr(usage, "total_tokens"):
                    self.logger.debug(f"  - total_tokens: {getattr(usage, 'total_tokens', 'N/A')}")

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
        # Calculate costs for totals
        from utils.config import get_model_cost_rates
        
        # Get the primary model from the most used model
        primary_model = None
        max_tokens = 0
        for model, usage in self.usage_by_model.items():
            if usage["total_tokens"] > max_tokens:
                max_tokens = usage["total_tokens"]
                primary_model = model
        
        # Get cost rates for the primary model
        if primary_model:
            cost_rates = get_model_cost_rates(primary_model)
            prompt_cost = (self.total_prompt_tokens / 1_000_000) * cost_rates["prompt"]
            completion_cost = (self.total_completion_tokens / 1_000_000) * cost_rates["completion"]
            total_cost = prompt_cost + completion_cost
        else:
            prompt_cost = 0.0
            completion_cost = 0.0
            total_cost = 0.0
            
        return {
            "by_model": self.usage_by_model,
            "by_agent": self.usage_by_agent,
            "total": {
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
                "prompt_cost": prompt_cost,
                "completion_cost": completion_cost,
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
