"""
Token tracking utilities for converting between different token tracking formats.
"""

import os
import sys
from typing import Dict, Any, Optional


from ..models import TokenUsageMetadata


def convert_to_token_usage_metadata(token_tracker: Any) -> Optional[TokenUsageMetadata]:
    """
    Convert token tracking results from OpenAI Agents SDK to the TokenUsageMetadata format
    expected by the application.

    Args:
        token_tracker: The token tracker from the OpenAI Agents SDK

    Returns:
        TokenUsageMetadata: The converted token usage metadata, or None if conversion fails
    """
    if not token_tracker:
        return None

    try:
        # Get total usage statistics
        total_usage = token_tracker.get_total_usage()

        # Extract model costs if available
        model_costs = {}
        if "model_breakdown" in total_usage:
            for model_name, model_stats in total_usage["model_breakdown"].items():
                model_costs[model_name] = {
                    "prompt_tokens": model_stats.get("prompt_tokens", 0),
                    "completion_tokens": model_stats.get("completion_tokens", 0),
                    "total_tokens": model_stats.get("total_tokens", 0),
                    "prompt_cost": model_stats.get("prompt_cost", 0.0),
                    "completion_cost": model_stats.get("completion_cost", 0.0),
                    "total_cost": model_stats.get("total_cost", 0.0),
                }

        # Create TokenUsageMetadata object with proper type conversions
        return TokenUsageMetadata(
            prompt_tokens=int(total_usage.get("prompt_tokens", 0)),
            completion_tokens=int(total_usage.get("completion_tokens", 0)),
            total_tokens=int(total_usage.get("total_tokens", 0)),
            prompt_cost=float(total_usage.get("prompt_cost", 0.0)),
            completion_cost=float(total_usage.get("completion_cost", 0.0)),
            total_cost=float(total_usage.get("total_cost", 0.0)),
            model_costs=model_costs,
        )
    except Exception as e:
        # Log error but don't halt processing
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Error converting token usage: {str(e)}")
        return None
