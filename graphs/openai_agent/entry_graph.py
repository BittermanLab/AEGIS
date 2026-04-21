"""
OpenAI Agent SDK workflow entry point.

This module provides the main integration with the experimental framework,
enabling processing of clinical notes using the OpenAI Agent SDK.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Union
import json
import os
from datetime import datetime

from agents import set_tracing_disabled

# Import adapter utilities for data conversion
from .adapter import convert_note_to_input_format, convert_prediction_to_expected_format

# Import the processor
from .processor import LLMProcessor

# Import token tracker
from .models import TokenTracker

# Set up logging
logger = logging.getLogger(__name__)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

# Disable tracing for all OpenAI agent executions
set_tracing_disabled(True)


def configure_logging(logging_config: Dict[str, Any]) -> None:
    """Configure logging based on the provided configuration."""
    if not logging_config:
        return

    # Map string log levels to logging module constants
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    # Get the log level, defaulting to INFO
    log_level_str = logging_config.get("level", "INFO").upper()
    log_level = log_level_map.get(log_level_str, logging.INFO)

    # Configure the logger for this module
    logger.setLevel(log_level)

    # Configure other loggers in the graph package
    for module_name in ["processor", "adapter", "models"]:
        module_logger = logging.getLogger(f"graphs.openai_agent.{module_name}")
        module_logger.setLevel(log_level)

    # Set httpx logger to WARNING to suppress HTTP request logs
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)

    logger.debug(f"OpenAI Agent logging configured with level: {log_level_str}")


class OpenAIAgentWorkflow:
    """
    Workflow for processing clinical notes using the OpenAI Agent SDK.
    """

    def __init__(
        self,
        model_config: str = "default",
        prompt_variant: str = "default",
        max_concurrent_runs: int = 5,
        logging_config: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the workflow with the specified model configuration.

        Args:
            model_config: The model configuration key
            prompt_variant: The prompt variant to use
            max_concurrent_runs: Maximum number of concurrent runs to allow
            logging_config: Optional logging configuration dictionary
            parameters: Optional parameters dictionary with additional settings
        """
        # Configure logging if provided
        if logging_config:
            configure_logging(logging_config)

        self.model_config = model_config
        self.prompt_variant = prompt_variant
        self.parameters = parameters or {}

        # Initialize the processor with concurrency control
        self.processor = LLMProcessor(
            model_config_key=model_config,
            prompt_variant=prompt_variant,
            max_concurrent_runs=max_concurrent_runs,
            logging_config=logging_config,
            parameters=parameters,
        )
        
        # NOTE: The processor now automatically configures either Azure or Ollama
        # based on the model type in __init__, so we don't need to explicitly call
        # configure_azure() or configure_ollama() here.
        
        # Override the default service configuration only if explicitly needed
        if parameters and parameters.get("force_azure_config", False):
            logger.info("Explicitly configuring Azure due to force_azure_config parameter")
            self.processor.configure_azure()

        logger.info(
            f"Initialized workflow with model={model_config}, variant={prompt_variant}, concurrency={max_concurrent_runs}"
        )

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single note and return the result.

        Args:
            state: Input state dictionary with patient_note and metadata

        Returns:
            Dictionary with prediction results
        """
        try:
            # Extract note from state
            patient_note = state.get("patient_note", "")
            if not patient_note:
                logger.error("No patient note provided in state")
                return self._create_error_result("No patient note provided")

            # Convert to input format
            note_input = convert_note_to_input_format(
                {
                    "note": patient_note,
                    "patient_id": state.get("patient_id", "UNKNOWN"),
                    "timepoint": state.get("timepoint", "t1"),
                }
            )

            # Process the note
            prediction = self.processor.process_single_note_sync(note_input)

            # Convert result to expected format
            result = convert_prediction_to_expected_format(prediction)

            # Save token usage to file for analysis
            if hasattr(self.processor, "token_tracker"):
                try:
                    # Create output directory if it doesn't exist
                    output_dir = os.path.join("output", "token_usage")
                    os.makedirs(output_dir, exist_ok=True)

                    # Save token usage summary to file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    token_file = os.path.join(
                        output_dir, f"{note_input.note_id}_{timestamp}_tokens.json"
                    )

                    # Get the summary and save it
                    token_summary = self.processor.token_tracker.get_summary()
                    with open(token_file, "w") as f:
                        json.dump(token_summary, f, indent=2)

                    logger.info(f"Token usage saved to {token_file}")

                    # Add the summary to the result for reference
                    if "token_usage" in result:
                        result["token_usage"]["by_model"] = token_summary["by_model"]
                        result["token_usage"]["by_agent"] = token_summary["by_agent"]
                except Exception as e:
                    logger.error(f"Error saving token usage: {e}")

            # Log token usage for debugging
            if "token_usage" in result and result["token_usage"]:
                logger.info(
                    f"Token usage for note {note_input.note_id}: "
                    f"prompt={result['token_usage'].get('prompt_tokens', 0)}, "
                    f"completion={result['token_usage'].get('completion_tokens', 0)}, "
                    f"total={result['token_usage'].get('total_tokens', 0)}"
                )

            return result

        except Exception as e:
            logger.exception(f"Error processing note: {str(e)}")
            return self._create_error_result(f"Processing error: {str(e)}")

    def process_batch(self, states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple notes in a batch.

        Args:
            states: List of state dictionaries, each containing a patient_note

        Returns:
            List of prediction results
        """
        # Convert states to note inputs
        note_inputs = []
        for state in states:
            try:
                note_input = convert_note_to_input_format(state)
                note_inputs.append(note_input)
            except Exception as e:
                logger.error(f"Error converting state to input: {str(e)}")
                # Will be handled in results conversion

        # Process the batch
        if not note_inputs:
            logger.warning("No valid note inputs to process")
            return []

        try:
            # Process notes
            predictions = self.processor.process_batch(note_inputs)

            # Convert to expected format
            results = [
                convert_prediction_to_expected_format(pred) for pred in predictions
            ]

            # Log token usage for debugging
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_tokens = 0

            for idx, result in enumerate(results):
                if "token_usage" in result and result["token_usage"]:
                    prompt_tokens = result["token_usage"].get("prompt_tokens", 0)
                    completion_tokens = result["token_usage"].get(
                        "completion_tokens", 0
                    )
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    total_tokens += result["token_usage"].get("total_tokens", 0)

            # Log the aggregated token usage data without saving to file
            if hasattr(self.processor, "token_tracker"):
                try:
                    # Get token usage summary and log it
                    token_summary = self.processor.token_tracker.get_summary()
                    logger.info(
                        f"Total batch token usage from tracker: "
                        f"{token_summary['total']['prompt_tokens']} prompt tokens, "
                        f"{token_summary['total']['completion_tokens']} completion tokens, "
                        f"{token_summary['total']['total_tokens']} total tokens, "
                        f"${token_summary['total']['total_cost']:.4f}"
                    )
                except Exception as e:
                    logger.error(f"Error logging batch token usage: {e}")

            logger.info(
                f"Batch token usage: prompt={total_prompt_tokens}, "
                f"completion={total_completion_tokens}, total={total_tokens}"
            )

            return results

        except Exception as e:
            logger.exception(f"Error in batch processing: {str(e)}")
            return [self._create_error_result(f"Batch error: {str(e)}") for _ in states]

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create a standardized error result dictionary"""
        return {
            "messages": [],
            "final_output": {},
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "error": True,
            },
            "processing_time": 0.0,
            "error": error_message,
        }


def create_workflow(
    config: Optional[Any] = None,
    prompt_variant: str = "default",
    parameters: Optional[Dict[str, Any]] = None,
) -> OpenAIAgentWorkflow:
    """
    Create a workflow instance with the given configuration.

    Args:
        config: Configuration object that may contain model_name
        prompt_variant: The prompt variant to use
        parameters: Optional parameters dictionary with additional settings

    Returns:
        Initialized OpenAIAgentWorkflow
    """
    model_config = "default"
    if config is not None:
        model_config = getattr(config, "model_name", "default")

    # Extract max_concurrent_runs from parameters if available
    max_concurrent_runs = 3  # Default reasonable value
    logging_config = None

    if parameters is not None:
        # Check for concurrency parameter with multiple possible keys
        for key in ["max_concurrent_runs", "max_concurrency", "concurrency"]:
            if key in parameters:
                max_concurrent_runs = int(parameters[key])
                logger.info(
                    f"Setting max_concurrent_runs={max_concurrent_runs} from parameters['{key}']"
                )
                break

        # Extract logging configuration if available
        if "logging" in parameters:
            logging_config = parameters["logging"]
            logger.debug(f"Found logging configuration in parameters: {logging_config}")

    return OpenAIAgentWorkflow(
        model_config=model_config,
        prompt_variant=prompt_variant,
        max_concurrent_runs=max_concurrent_runs,
        logging_config=logging_config,
        parameters=parameters,  # Pass parameters to workflow
    )


# Create default workflow instance for main.py
graph = create_workflow()
