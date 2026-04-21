"""
Utility for tracking token usage and costs across the workflow.
"""

import logging
import os
from typing import Dict, Any, Optional, List, Sequence
import time

logger = logging.getLogger(__name__)
# Allow setting token tracker log level through environment variable
log_level = os.environ.get("TOKEN_TRACKER_LOG_LEVEL", "INFO")
logger.setLevel(logging.getLevelName(log_level))

# Set httpx logger to WARNING level to completely suppress HTTP request logs
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

# Define cost per 1M
MODEL_COSTS = {
    # O1 model
    "o1-jg": {"prompt": 15.0, "completion": 60.0, "per_unit": "1M"},
    # O3 models
    "o3-mini-jg": {"prompt": 1.10, "completion": 4.40, "per_unit": "1M"},
    "o4-mini": {"prompt": 1.10, "completion": 4.40, "per_unit": "1M"},
    "o3-jg": {"prompt": 3.0, "completion": 15.0, "per_unit": "1M"},
    # GPT-4 models
    "gpt-4.5-jg": {"prompt": 75.0, "completion": 150.0, "per_unit": "1M"},
    "gpt-4o-jg": {"prompt": 2.50, "completion": 10.0, "per_unit": "1M"},
    "gpt-4o-mini-jg": {"prompt": 0.15, "completion": 0.60, "per_unit": "1M"},
    # Default fallback
    "gpt-4.1-mini": {"prompt": 0.15, "completion": 0.60, "per_unit": "1M"},
    "gpt-4.1-nano": {"prompt": 0.03, "completion": 0.12, "per_unit": "1M"},
    "default": {"prompt": 0.15, "completion": 0.60, "per_unit": "1M"},
}


class TokenUsageTracker:
    """
    Class for tracking token usage and costs across the workflow.
    """

    def __init__(self):
        """Initialize the token tracker with empty usage data."""
        self.usage_data = {}
        self.model_usage = {}  # Track usage by model

    def track_usage(
        self, agent_name: str, result: Any, model_name: Optional[str] = None
    ) -> None:
        """
        Track token usage for a specific agent from a result object.

        Args:
            agent_name: Name of the agent (e.g., "extractor", "retriever")
            result: The result object from Runner.run(), containing token usage info
            model_name: The specific model used by this agent (for cost calculation)
        """
        try:
            # Convert model_name to string if it's an object
            if model_name is not None and not isinstance(model_name, str):
                model_name = str(model_name)

            # Extract token usage information from the result
            logger.debug(
                f"Attempting to extract token usage for agent: {agent_name}, model: {model_name or 'unknown'}"
            )
            token_info = self._extract_token_info(result)

            # If we couldn't extract the token info, try using synthetic usage as fallback
            if token_info is None:
                logger.warning(
                    f"No token usage information found for {agent_name} with model {model_name or 'unknown'}, using synthetic fallback"
                )
                # Try using the debug method to get more information
                if hasattr(self, "debug_token_extraction"):
                    logger.debug(f"Attempting detailed debugging for {agent_name}")
                    self.debug_token_extraction(result)

                # Generate synthetic token usage
                token_info = self.create_synthetic_usage(agent_name, result, model_name)
                logger.debug(
                    f"Created synthetic token usage for {agent_name}: {token_info}"
                )

            if token_info:
                # Add cost information to token info based on model
                if model_name:
                    token_info["model"] = model_name
                    token_info = self._add_cost_to_token_info(token_info, model_name)

                # Store the usage data for this agent
                self.usage_data[agent_name] = token_info

                # Track by model as well
                if model_name:
                    if model_name not in self.model_usage:
                        self.model_usage[model_name] = {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "prompt_cost": 0.0,
                            "completion_cost": 0.0,
                            "total_cost": 0.0,
                        }

                    # Add to model tracking
                    model_stats = self.model_usage[model_name]
                    prompt_tokens = token_info.get("prompt_tokens", 0)
                    completion_tokens = token_info.get("completion_tokens", 0)
                    total_tokens = token_info.get("total_tokens", 0)

                    model_stats["prompt_tokens"] += prompt_tokens
                    model_stats["completion_tokens"] += completion_tokens
                    model_stats["total_tokens"] += total_tokens
                    model_stats["prompt_cost"] += token_info.get("prompt_cost", 0.0)
                    model_stats["completion_cost"] += token_info.get(
                        "completion_cost", 0.0
                    )
                    model_stats["total_cost"] += token_info.get("total_cost", 0.0)

                    logger.debug(
                        f"Tracked tokens for {agent_name} (model: {model_name}): "
                        f"prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}"
                    )
                else:
                    prompt_tokens = token_info.get("prompt_tokens", 0)
                    completion_tokens = token_info.get("completion_tokens", 0)
                    total_tokens = token_info.get("total_tokens", 0)
                    is_synthetic = token_info.get("synthetic", False)
                    synthetic_flag = " (SYNTHETIC)" if is_synthetic else ""

                    logger.debug(
                        f"Tracked tokens for {agent_name}{synthetic_flag}: "
                        f"prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}"
                    )
            else:
                logger.error(
                    f"Failed to track token usage for {agent_name} even with synthetic fallback"
                )
        except Exception as e:
            logger.error(f"Error tracking token usage for {agent_name}: {str(e)}")
            import traceback

            logger.debug(f"Token tracking error details: {traceback.format_exc()}")

    def _extract_token_info(self, result: Any) -> Optional[Dict[str, Any]]:
        """
        Extract token usage information from a result object.

        Args:
            result: The result object from Runner.run()

        Returns:
            Dictionary with token usage information or None if not available
        """
        try:
            # Direct access to usage attribute (common in Azure OpenAI responses)
            if hasattr(result, "usage") and result.usage is not None:
                logger.debug(f"Found direct usage attribute: {result.usage}")
                # Convert to dict if it's not already
                if not isinstance(result.usage, dict):
                    try:
                        if hasattr(result.usage, "dict") and callable(
                            result.usage.dict
                        ):
                            return result.usage.dict()
                        if hasattr(result.usage, "model_dump") and callable(
                            result.usage.model_dump
                        ):
                            return result.usage.model_dump()
                        # Try direct attribute access
                        return {
                            "prompt_tokens": getattr(result.usage, "prompt_tokens", 0),
                            "completion_tokens": getattr(
                                result.usage, "completion_tokens", 0
                            ),
                            "total_tokens": getattr(result.usage, "total_tokens", 0),
                        }
                    except Exception as usage_err:
                        logger.debug(
                            f"Error converting usage to dict: {str(usage_err)}"
                        )

                # If it's already a dict, return it
                if isinstance(result.usage, dict):
                    return result.usage

            # Special case for the raw_responses array containing ModelResponse objects (Azure OpenAI SDK)
            azure_usage = self._track_usage_from_raw_responses(result)
            if azure_usage:
                return azure_usage

            # Try to extract token usage from the result object
            if hasattr(result, "token_usage") and result.token_usage:
                return result.token_usage

            # Look for JSON serialization methods
            if hasattr(result, "json") and callable(result.json):
                try:
                    json_data = result.json()
                    if isinstance(json_data, dict) and "usage" in json_data:
                        return json_data["usage"]
                except Exception as json_err:
                    logger.debug(f"Error extracting from json(): {str(json_err)}")

            # Look for model_dump method (newer OpenAI Python SDK)
            if hasattr(result, "model_dump") and callable(result.model_dump):
                try:
                    model_dump = result.model_dump()
                    if isinstance(model_dump, dict) and "usage" in model_dump:
                        return model_dump["usage"]
                except Exception as dump_err:
                    logger.debug(f"Error extracting from model_dump(): {str(dump_err)}")

            # Access dict attributes directly
            if hasattr(result, "__dict__"):
                try:
                    if "usage" in result.__dict__:
                        usage = result.__dict__["usage"]
                        if isinstance(usage, dict):
                            return usage
                        elif hasattr(usage, "__dict__"):
                            return usage.__dict__
                except Exception as dict_err:
                    logger.debug(f"Error accessing __dict__: {str(dict_err)}")

            # For Azure OpenAI Service, check for the response structure
            if hasattr(result, "raw") and result.raw is not None:
                if isinstance(result.raw, dict) and "usage" in result.raw:
                    return result.raw["usage"]
                # Sometimes nested in 'choices'
                if isinstance(result.raw, dict) and "choices" in result.raw:
                    choices = result.raw["choices"]
                    if (
                        choices
                        and isinstance(choices[0], dict)
                        and "usage" in choices[0]
                    ):
                        return choices[0]["usage"]

            # Check specific Azure OpenAI response object structure
            # Direct presence of "choices" and "id" attributes indicate Azure OpenAI response
            if hasattr(result, "choices") and hasattr(result, "id"):
                # For newer Azure SDK, usage might be directly accessible
                if hasattr(result, "usage"):
                    usage = result.usage
                    # If usage is an object with prompt_tokens attribute
                    if hasattr(usage, "prompt_tokens"):
                        return {
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": getattr(usage, "completion_tokens", 0),
                            "total_tokens": usage.total_tokens,
                        }

            # For OpenAI Agents SDK, check common locations
            for attr in [
                "raw_responses",
                "token_usage",
                "metadata",
                "raw",
                "model_output",
            ]:
                if hasattr(result, attr):
                    attr_value = getattr(result, attr)
                    if isinstance(attr_value, dict):
                        # Check for direct usage
                        if "usage" in attr_value:
                            return attr_value["usage"]
                        # Check for usage in Azure response structure
                        if "choices" in attr_value and attr_value["choices"]:
                            choices = attr_value["choices"]
                            if isinstance(choices[0], dict) and "usage" in choices[0]:
                                return choices[0]["usage"]
                        # Check for completion tokens in a different format
                        if (
                            "completion_tokens" in attr_value
                            and "prompt_tokens" in attr_value
                        ):
                            return {
                                "prompt_tokens": attr_value.get("prompt_tokens", 0),
                                "completion_tokens": attr_value.get(
                                    "completion_tokens", 0
                                ),
                                "total_tokens": attr_value.get("prompt_tokens", 0)
                                + attr_value.get("completion_tokens", 0),
                            }

                    # Handle list of responses
                    if isinstance(attr_value, list) and attr_value:
                        for item in attr_value:
                            if isinstance(item, dict) and "usage" in item:
                                return item["usage"]
                            if (
                                isinstance(item, dict)
                                and "completion_tokens" in item
                                and "prompt_tokens" in item
                            ):
                                return {
                                    "prompt_tokens": item.get("prompt_tokens", 0),
                                    "completion_tokens": item.get(
                                        "completion_tokens", 0
                                    ),
                                    "total_tokens": item.get("prompt_tokens", 0)
                                    + item.get("completion_tokens", 0),
                                }

            # Create a minimal token usage if model has the info but not in standard format
            if hasattr(result, "completion_tokens") and hasattr(
                result, "prompt_tokens"
            ):
                prompt_tokens = getattr(result, "prompt_tokens", 0)
                completion_tokens = getattr(result, "completion_tokens", 0)
                if prompt_tokens or completion_tokens:
                    return {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }

            # If no token info found but we have a model response, try to extract from Azure-specific format
            if hasattr(result, "response") and result.response is not None:
                if hasattr(result.response, "usage"):
                    # If it's an object with attributes
                    if hasattr(result.response.usage, "prompt_tokens"):
                        return {
                            "prompt_tokens": result.response.usage.prompt_tokens,
                            "completion_tokens": getattr(
                                result.response.usage, "completion_tokens", 0
                            ),
                            "total_tokens": result.response.usage.total_tokens,
                        }
                    return result.response.usage
                if isinstance(result.response, dict) and "usage" in result.response:
                    return result.response["usage"]

            # If all else fails, attempt to estimate token usage based on response content
            if hasattr(result, "final_output") and result.final_output is not None:
                try:
                    import json

                    # Get content as string
                    content = ""
                    if hasattr(result.final_output, "__str__"):
                        content = str(result.final_output)
                    elif hasattr(result.final_output, "model_dump"):
                        content = json.dumps(result.final_output.model_dump())
                    else:
                        # Try a simple JSON dump
                        content = json.dumps(result.final_output)

                    if content:
                        # Rough estimate: 1 token ≈ 4 characters
                        completion_tokens = len(content) // 4
                        logger.debug(
                            f"Estimated token usage from content: {completion_tokens} tokens"
                        )
                        return {
                            "prompt_tokens": 0,  # Unknown input tokens
                            "completion_tokens": completion_tokens,
                            "total_tokens": completion_tokens,
                            "estimated": True,  # Flag that this is an estimate
                        }
                except Exception as est_err:
                    logger.debug(f"Error estimating tokens: {str(est_err)}")

            logger.debug(
                f"All token extraction methods failed for result type: {type(result)}"
            )
            return None
        except Exception as e:
            logger.error(f"Error extracting token info: {str(e)}")
            import traceback

            logger.debug(f"Token extraction error details: {traceback.format_exc()}")
            return None

    def _add_cost_to_token_info(
        self, token_info: Dict[str, Any], model_name: str
    ) -> Dict[str, Any]:
        """
        Add cost information to token usage data based on model pricing.

        Args:
            token_info: Dictionary with token usage information
            model_name: Name of the model to calculate costs for

        Returns:
            Dictionary with token usage information and cost data added
        """
        # Get cost information for this model, or use default if not available
        cost_info = MODEL_COSTS.get(model_name, MODEL_COSTS["default"])

        # Extract pricing unit (per 1K or per 1M)
        per_unit = cost_info.get("per_unit", "1M")

        # Determine the divisor based on the unit
        divisor = 1000 if per_unit == "1K" else 1000000

        # Calculate costs
        prompt_tokens = token_info.get("prompt_tokens", 0)
        completion_tokens = token_info.get("completion_tokens", 0)

        prompt_cost = (prompt_tokens / divisor) * cost_info["prompt"]
        completion_cost = (completion_tokens / divisor) * cost_info["completion"]
        total_cost = prompt_cost + completion_cost

        # Add cost information to token info
        token_info["prompt_cost"] = prompt_cost
        token_info["completion_cost"] = completion_cost
        token_info["total_cost"] = total_cost
        token_info["cost_calculation"] = (
            f"Based on {model_name} pricing: ${cost_info['prompt']} per {per_unit} prompt tokens, ${cost_info['completion']} per {per_unit} completion tokens"
        )

        return token_info

    def get_agent_usage(self, agent_name: str) -> Dict[str, Any]:
        """
        Get token usage for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Dictionary with token usage information
        """
        return self.usage_data.get(agent_name, {})

    def get_total_usage(self) -> Dict[str, Any]:
        """
        Get total token usage across all agents and models.

        Returns:
            Dictionary with aggregated token usage information
        """
        total: Dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_cost": 0.0,
            "completion_cost": 0.0,
            "total_cost": 0.0,
            "model_breakdown": {},
            "agent_breakdown": {},  # Use empty dict literal with proper typing
        }

        # First gather all agent usage
        for agent_name, usage in self.usage_data.items():
            # Extract token counts
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = prompt_tokens + completion_tokens

            # Add to totals
            total["prompt_tokens"] += prompt_tokens
            total["completion_tokens"] += completion_tokens
            total["total_tokens"] += total_tokens

            # Add cost information (if available)
            prompt_cost = usage.get("prompt_cost", 0.0)
            completion_cost = usage.get("completion_cost", 0.0)
            total_cost = prompt_cost + completion_cost

            total["prompt_cost"] += prompt_cost
            total["completion_cost"] += completion_cost
            total["total_cost"] += total_cost

            # Get model - ensure it's a string
            model = usage.get("model", "unknown")
            model_str = str(model) if not isinstance(model, str) else model

            # Add to agent breakdown - use dict assignment with proper typing
            total["agent_breakdown"][agent_name] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "prompt_cost": prompt_cost,
                "completion_cost": completion_cost,
                "total_cost": total_cost,
                "model": model_str,
            }

        # Add model breakdown - ensure all keys are strings
        model_breakdown = {}
        for model, usage in self.model_usage.items():
            # Convert model name to string if needed
            model_key = str(model) if not isinstance(model, str) else model
            model_breakdown[model_key] = usage

        total["model_breakdown"] = model_breakdown

        return total

    def estimate_cost(self, model_name: str = "default") -> Dict[str, Any]:
        """
        Estimate the cost based on token usage.

        Args:
            model_name: Name of the model to use for cost estimation

        Returns:
            Dictionary with estimated costs
        """
        # Get total token usage
        total = self.get_total_usage()

        # Get cost per 1K tokens for the specified model
        cost_per_1k = MODEL_COSTS.get(model_name, MODEL_COSTS["default"])

        # Determine if this is per 1K or per 1M pricing
        per_unit = cost_per_1k.get("per_unit", "1K")
        divisor = 1000 if per_unit == "1K" else 1000000

        # Calculate costs
        prompt_cost = (total["prompt_tokens"] / divisor) * cost_per_1k["prompt"]
        completion_cost = (total["completion_tokens"] / divisor) * cost_per_1k[
            "completion"
        ]
        total_cost = prompt_cost + completion_cost

        return {
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": total_cost,
            "model": model_name,
        }

    def generate_cost_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive cost report for all usage.

        Returns:
            Dictionary with detailed cost breakdown by model and agent
        """
        # Get the complete usage data
        total_usage = self.get_total_usage()

        # Generate a more structured report
        report = {
            "summary": {
                "total_tokens": total_usage["total_tokens"],
                "total_cost": total_usage["total_cost"],
                "prompt_tokens": total_usage["prompt_tokens"],
                "completion_tokens": total_usage["completion_tokens"],
                "prompt_cost": total_usage["prompt_cost"],
                "completion_cost": total_usage["completion_cost"],
            },
            "by_model": {},
            "by_agent": {},
        }

        # Add model breakdown - ensure model names are strings
        for model_name, usage in total_usage.get("model_breakdown", {}).items():
            # Convert model object to string if needed
            model_key = (
                str(model_name) if not isinstance(model_name, str) else model_name
            )
            report["by_model"][model_key] = {
                "total_tokens": usage.get("total_tokens", 0),
                "total_cost": usage.get("total_cost", 0.0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "prompt_cost": usage.get("prompt_cost", 0.0),
                "completion_cost": usage.get("completion_cost", 0.0),
            }

        # Add agent breakdown
        for agent_name, usage in total_usage.get("agent_breakdown", {}).items():
            model = usage.get("model", "unknown")
            # Convert model object to string if needed
            model_str = str(model) if not isinstance(model, str) else model

            report["by_agent"][agent_name] = {
                "model": model_str,
                "total_tokens": usage.get("total_tokens", 0),
                "total_cost": usage.get("total_cost", 0.0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "prompt_cost": usage.get("prompt_cost", 0.0),
                "completion_cost": usage.get("completion_cost", 0.0),
            }

        return report

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a clean summary suitable for predictions output.

        Returns:
            Dictionary with clean token usage summary
        """
        total_usage = self.get_total_usage()

        # Create a clean model costs dictionary with proper model names
        model_costs = {}
        for model_name, usage in self.model_usage.items():
            # Ensure model name is a string
            clean_model_name = (
                str(model_name) if not isinstance(model_name, str) else model_name
            )

            # Get cost rates from utils/config.py
            from ....utils.config import get_model_cost_rates

            cost_rates = get_model_cost_rates(clean_model_name)

            model_costs[clean_model_name] = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "prompt_cost": usage.get("prompt_cost", 0.0),
                "completion_cost": usage.get("completion_cost", 0.0),
                "total_cost": usage.get("total_cost", 0.0),
            }

        return {
            "total": {
                "prompt_tokens": total_usage["prompt_tokens"],
                "completion_tokens": total_usage["completion_tokens"],
                "total_tokens": total_usage["total_tokens"],
                "prompt_cost": total_usage["prompt_cost"],
                "completion_cost": total_usage["completion_cost"],
                "total_cost": total_usage["total_cost"],
            },
            "by_model": model_costs,
        }

    def debug_token_extraction(self, result: Any) -> None:
        """
        Debug token extraction by logging the structure of the result object.

        Args:
            result: The result object from Runner.run()
        """
        try:
            import json
            import pprint

            logger.debug(f"Token extraction DEBUG for result type: {type(result)}")

            # Log the complete result object with a max depth limit to avoid excessive output
            try:
                logger.debug(f"Full result object: {pprint.pformat(result, depth=2)}")
            except Exception as fmt_err:
                logger.debug(f"Cannot format full result: {str(fmt_err)}")

            # Check all common attributes
            debug_info: Dict[str, Any] = {}

            # Check direct attributes first for Azure OpenAI response structure
            if hasattr(result, "usage") and result.usage is not None:
                logger.debug(f"Found direct usage attribute: {result.usage}")
                return result.usage

            # Azure OpenAI specific response structure checking
            for field in [
                "headers",
                "http_request",
                "model",
                "id",
                "choices",
                "finish_reason",
                "usage",
            ]:
                if hasattr(result, field):
                    debug_info[f"direct_{field}"] = f"Present: {getattr(result, field)}"

            # Look for json representation
            if hasattr(result, "json"):
                try:
                    if callable(result.json):
                        json_data = result.json()
                        logger.debug(f"JSON representation: {json_data}")
                        if isinstance(json_data, dict) and "usage" in json_data:
                            logger.debug(f"Found usage in JSON: {json_data['usage']}")
                    else:
                        logger.debug(f"JSON attribute (not callable): {result.json}")
                except Exception as json_err:
                    logger.debug(f"Error getting JSON: {str(json_err)}")

            # Look for model output
            if hasattr(result, "model_output") and result.model_output is not None:
                logger.debug(f"Model output: {result.model_output}")
                if (
                    isinstance(result.model_output, dict)
                    and "usage" in result.model_output
                ):
                    logger.debug(
                        f"Found usage in model_output: {result.model_output['usage']}"
                    )

            # Try to access dict attributes
            if hasattr(result, "__dict__"):
                try:
                    logger.debug(f"Object __dict__: {result.__dict__}")
                    if "usage" in result.__dict__:
                        logger.debug(
                            f"Found usage in __dict__: {result.__dict__['usage']}"
                        )
                except Exception as dict_err:
                    logger.debug(f"Error accessing __dict__: {str(dict_err)}")

            # Try direct attributes for standard OpenAI response structure
            for attr_name in [
                "token_usage",
                "raw_responses",
                "metadata",
                "raw",
                "response",
            ]:
                if hasattr(result, attr_name):
                    attr_value = getattr(result, attr_name)
                    attr_type = type(attr_value)
                    if attr_value is not None:
                        logger.debug(f"Attribute {attr_name} = {attr_value}")
                        if isinstance(attr_value, dict):
                            debug_info[attr_name] = {
                                "type": str(attr_type),
                                "keys": list(attr_value.keys()),
                            }
                            # Look for usage
                            if "usage" in attr_value:
                                debug_info[f"{attr_name}.usage"] = attr_value["usage"]
                                logger.debug(
                                    f"Found usage in {attr_name}: {attr_value['usage']}"
                                )
                            # Look for choices
                            if "choices" in attr_value and attr_value["choices"]:
                                debug_info[f"{attr_name}.choices_count"] = len(
                                    attr_value["choices"]
                                )
                                if isinstance(attr_value["choices"][0], dict):
                                    debug_info[f"{attr_name}.choices[0].keys"] = list(
                                        attr_value["choices"][0].keys()
                                    )
                        elif isinstance(attr_value, list):
                            debug_info[attr_name] = {
                                "type": str(attr_type),
                                "length": len(attr_value),
                            }
                            if attr_value and isinstance(attr_value[0], dict):
                                debug_info[f"{attr_name}[0].keys"] = list(
                                    attr_value[0].keys()
                                )
                        else:
                            debug_info[attr_name] = {"type": str(attr_type)}

            # Try direct token attributes
            for attr_name in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                if hasattr(result, attr_name):
                    attr_value = getattr(result, attr_name)
                    debug_info[attr_name] = attr_value

            # Check response object if it exists
            if hasattr(result, "response"):
                response = result.response
                logger.debug(f"Response object: {response}")
                if hasattr(response, "usage"):
                    debug_info["response.usage"] = response.usage
                    logger.debug(f"Found usage in response object: {response.usage}")
                elif isinstance(response, dict) and "usage" in response:
                    debug_info["response.usage"] = response["usage"]
                    logger.debug(f"Found usage in response dict: {response['usage']}")

            # Add specific Azure OpenAI SDK response structure checks
            # For azure-openai or openai packages
            if hasattr(result, "model_dump"):
                try:
                    model_dump = result.model_dump()
                    logger.debug(f"Model dump: {model_dump}")
                    if isinstance(model_dump, dict) and "usage" in model_dump:
                        debug_info["model_dump.usage"] = model_dump["usage"]
                        logger.debug(
                            f"Found usage in model_dump: {model_dump['usage']}"
                        )
                except Exception as dump_err:
                    logger.debug(f"Error in model_dump: {str(dump_err)}")

            logger.debug(f"Token extraction debug info: {debug_info}")

            # Try the extraction method
            token_info = self._extract_token_info(result)
            if token_info:
                logger.debug(f"Successfully extracted token info: {token_info}")
            else:
                logger.warning("Token extraction failed despite debugging attempt")

        except Exception as e:
            logger.error(f"Error during token extraction debugging: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())

    def create_synthetic_usage(
        self, agent_name: str, result: Any, model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create synthetic token usage information when no real token data is available.
        This is a fallback solution for testing or when token usage data is not provided
        by the API.

        Args:
            agent_name: Name of the agent
            result: The result object
            model_name: Optional model name

        Returns:
            Dict with synthetic token usage information
        """
        try:
            logger.debug(f"Creating synthetic token usage for {agent_name}")

            # Get the final output if available
            output_text = ""
            if hasattr(result, "final_output"):
                try:
                    import json

                    output_obj = result.final_output
                    if hasattr(output_obj, "__str__"):
                        output_text = str(output_obj)
                    elif hasattr(output_obj, "model_dump"):
                        content = json.dumps(output_obj.model_dump())
                        output_text = (
                            content if isinstance(content, str) else str(content)
                        )
                    else:
                        # Try a simple JSON dump
                        content = json.dumps(output_obj)
                        output_text = (
                            content if isinstance(content, str) else str(content)
                        )
                except:
                    output_text = str(result.final_output)

            # Estimate completion tokens (rough calculation: 1 token ≈ 4 characters)
            completion_tokens = max(1, len(output_text) // 4)

            # Estimate prompt tokens based on agent type
            prompt_tokens = 0
            if "grader" in agent_name:
                prompt_tokens = 500  # Typical grader prompt tokens
            elif "judge" in agent_name:
                prompt_tokens = 800  # Judges get more context
            elif "identifier" in agent_name:
                prompt_tokens = 400  # Identifiers typically have shorter prompts
            elif "attribution" in agent_name or "certainty" in agent_name:
                prompt_tokens = 600  # Attribution/certainty assessors
            else:
                prompt_tokens = 500  # Default fallback

            # Total tokens
            total_tokens = prompt_tokens + completion_tokens

            # Create synthetic token usage with proper typing
            synthetic_usage: Dict[str, Any] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "synthetic": True,  # Flag that this is not real data
            }

            # Add to our tracking
            if model_name:
                synthetic_usage["model"] = model_name  # Now properly typed as Any
                synthetic_usage = self._add_cost_to_token_info(
                    synthetic_usage, model_name
                )

            logger.debug(f"Created synthetic usage for {agent_name}: {synthetic_usage}")
            return synthetic_usage

        except Exception as e:
            logger.error(f"Error creating synthetic usage: {str(e)}")
            # Return minimal dictionary with some tokens
            return {
                "prompt_tokens": 500,
                "completion_tokens": 100,
                "total_tokens": 600,
                "synthetic": True,
                "error": str(e),
            }

    def _track_usage_from_raw_responses(self, result: Any) -> Optional[Dict[str, Any]]:
        """
        Extract token usage specifically from the raw_responses field in Azure OpenAI API responses.

        The expected structure from Azure OpenAI:

        result.raw_responses = [
            ModelResponse(
                output=[...],
                usage=Usage(requests=1, input_tokens=987, output_tokens=16, total_tokens=1003),
                referenceable_id=None
            )
        ]

        Args:
            result: The result object from Runner.run()

        Returns:
            Dictionary with token usage information or None if not available
        """
        if (
            not hasattr(result, "raw_responses")
            or not isinstance(result.raw_responses, list)
            or not result.raw_responses
        ):
            return None

        # Look through the raw_responses for token usage
        for response_obj in result.raw_responses:
            # Check if this is a ModelResponse with usage information
            if hasattr(response_obj, "usage"):
                usage = response_obj.usage
                logger.debug(f"Found usage in raw_responses item: {usage}")

                # Convert the usage object to a dictionary
                if hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
                    # This appears to be the Azure OpenAI SDK Usage object format
                    return {
                        "prompt_tokens": getattr(usage, "input_tokens", 0),
                        "completion_tokens": getattr(usage, "output_tokens", 0),
                        "total_tokens": getattr(usage, "total_tokens", 0)
                        or (
                            getattr(usage, "input_tokens", 0)
                            + getattr(usage, "output_tokens", 0)
                        ),
                    }

                # Try parsing from string representation as a fallback
                try:
                    usage_str = str(usage)
                    if "input_tokens" in usage_str and "output_tokens" in usage_str:
                        # Example: Usage(requests=1, input_tokens=987, output_tokens=16, total_tokens=1003)
                        import re

                        input_match = re.search(r"input_tokens=(\d+)", usage_str)
                        output_match = re.search(r"output_tokens=(\d+)", usage_str)
                        total_match = re.search(r"total_tokens=(\d+)", usage_str)

                        if input_match and output_match:
                            input_tokens = int(input_match.group(1))
                            output_tokens = int(output_match.group(1))
                            total_tokens = (
                                int(total_match.group(1))
                                if total_match
                                else input_tokens + output_tokens
                            )

                            return {
                                "prompt_tokens": input_tokens,
                                "completion_tokens": output_tokens,
                                "total_tokens": total_tokens,
                            }
                except Exception as str_err:
                    logger.debug(f"Error extracting from usage string: {str_err}")

        return None
