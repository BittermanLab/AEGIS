"""
Test zeroshotentry graph - Integration with main.py.

This file provides integration with the main experimental framework,
connecting it with the test zeroshot processing approach.
"""

import logging
import os
import sys
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

# Set up logging - this will be configured properly by the environment/sweep settings
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Explicitly use console logging only
)

# Add parent directory to sys.path to fix imports
parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(parent_dir)
logger = logging.getLogger(__name__)


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
    for module_name in ["zeroshot_processor", "openai_handler"]:
        module_logger = logging.getLogger(f"graphs.openai_zeroshot.{module_name}")
        module_logger.setLevel(log_level)

    logger.debug(f"Test Zeroshot logging configured with level: {log_level_str}")


# Define our own adapter functions instead of importing from openai_agent
def convert_note_to_input_format(note_text: str):
    """Convert a note text to NoteInput format."""
    from graphs.openai_zeroshot.zeroshot_processor import NoteInput
    import uuid
    from datetime import datetime

    note_id = f"note_{uuid.uuid4().hex[:8]}"
    current_time = datetime.now().strftime("%Y-%m-%d")

    return NoteInput(
        pmrn="SYNTHETIC",
        note_id=note_id,
        note_type="CLINICAL",
        type_name="Clinical Note",
        loc_name="SYNTHETIC",
        date=current_time,
        prov_name="SYNTHETIC",
        prov_type="SYNTHETIC",
        line=1,
        note_text=note_text,
    )


def convert_prediction_to_expected_format(prediction):
    """Convert prediction to expected format."""
    # Initialize the output
    token_usage = {}

    # Extract token usage information
    if hasattr(prediction, "TOKEN_USAGE") and prediction.TOKEN_USAGE:
        if hasattr(prediction.TOKEN_USAGE, "to_dict"):
            token_usage = prediction.TOKEN_USAGE.to_dict()
        elif isinstance(prediction.TOKEN_USAGE, dict):
            token_usage = prediction.TOKEN_USAGE
        else:
            # Convert TokenUsageMetadata object to dict manually
            token_usage = {
                "prompt_tokens": getattr(prediction.TOKEN_USAGE, "prompt_tokens", 0),
                "completion_tokens": getattr(
                    prediction.TOKEN_USAGE, "completion_tokens", 0
                ),
                "total_tokens": getattr(prediction.TOKEN_USAGE, "total_tokens", 0),
                "prompt_cost": getattr(prediction.TOKEN_USAGE, "prompt_cost", 0.0),
                "completion_cost": getattr(
                    prediction.TOKEN_USAGE, "completion_cost", 0.0
                ),
                "total_cost": getattr(prediction.TOKEN_USAGE, "total_cost", 0.0),
            }

        # Add model name and cost rates
        if hasattr(prediction, "MODEL_NAME"):
            token_usage["model"] = prediction.MODEL_NAME

        # Add cost rates if available
        if (
            hasattr(prediction.TOKEN_USAGE, "model_costs")
            and prediction.TOKEN_USAGE.model_costs
        ):
            token_usage["cost_rates"] = prediction.TOKEN_USAGE.model_costs

    # Add token tracker data if available
    if hasattr(prediction, "TOKEN_TRACKER") and prediction.TOKEN_TRACKER:
        token_tracker_summary = prediction.TOKEN_TRACKER.get_summary()
        token_usage["by_model"] = token_tracker_summary["by_model"]
        token_usage["by_agent"] = token_tracker_summary["by_agent"]

    return {
        "messages": [],
        "final_output": (
            prediction.PREDICTION if hasattr(prediction, "PREDICTION") else {}
        ),
        "token_usage": token_usage,
        "processing_time": (
            prediction.PROCESSING_TIME
            if hasattr(prediction, "PROCESSING_TIME")
            else 0.0
        ),
        "error": prediction.ERROR if hasattr(prediction, "ERROR") else None,
    }


# Import the OpenAI zeroshot processor
try:
    # First try absolute import
    from graphs.zeroshot_test.zeroshot_processor import ZeroshotProcessor
except ImportError:
    # If absolute fails, perhaps we are running from a different context.
    # Let's log an error and re-raise, as relative import seems problematic.
    logger.error(
        "Failed to import ZeroshotProcessor using absolute path. "
        "Relative import is commented out due to potential issues."
    )
    raise
    # Fall back to relative import (commented out)
    # from .zeroshot_processor import ZeroshotProcessor


class OpenAIZeroshotWorkflow:
    """
    Adapter class that processes notes using the OpenAI zeroshot approach.
    """

    def __init__(
        self,
        model_config: str = "default",
        prompt_variant: str = "default",
        logging_config: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the OpenAI zeroshot workflow with model configuration.

        Args:
            model_config: The model configuration to use
            prompt_variant: The prompt variant to use
            logging_config: Optional logging configuration settings
            parameters: Optional parameters for provider configuration
        """
        # Configure logging if provided
        if logging_config:
            configure_logging(logging_config)

        self.model_config = model_config
        self.prompt_variant = prompt_variant
        self.parameters = parameters or {}

        # Create the processor instance
        try:
            self.processor = ZeroshotProcessor(
                model_config_key=model_config,
                prompt_variant=prompt_variant,
                logging_config=logging_config,
                parameters=parameters,
            )
            # Note: The processor now handles provider configuration internally
            # based on the model type, so we don't need to call configure_azure here
        except Exception as e:
            logger.error(f"Failed to initialize processor: {str(e)}", exc_info=True)
            raise

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input state and return the result.
        This mimics the invoke functionality of the old graph system.
        """
        # Extract the patient note from the state
        patient_note = state.get("patient_note", "")

        # Create a note input for the OpenAI agent
        note_input = convert_note_to_input_format(patient_note)

        # Process the note using the zeroshot processor
        try:
            # Process the note
            # Wrap the single note input into a list for batch processing
            batch_input = [note_input]
            batch_results = self.processor.process_notes_batch(batch_input)

            # Check if the results list is not empty and extract the first result
            if batch_results:
                prediction = batch_results[0]
            else:
                # Handle case where batch processing returned an empty list (should not happen for non-empty input)
                logger.error("Batch processing returned no results for a valid input.")
                raise RuntimeError("Batch processing failed to return a result.")

            # Attach token tracker to the prediction for reporting
            if hasattr(self.processor, "token_tracker"):
                prediction.TOKEN_TRACKER = self.processor.token_tracker

                # Log token usage statistics without saving to file
                try:
                    # Get the model name based on provider type
                    model_name = self.processor.model_config_key
                    if self.processor.provider_type == "azure" and hasattr(
                        self.processor, "deployment_mapping"
                    ):
                        deployment_name = self.processor.deployment_mapping.get(
                            model_name,
                            self.processor.deployment_mapping.get(
                                "default", model_name
                            ),
                        )
                    else:
                        # For Ollama and vLLM, use the model name directly
                        deployment_name = self.processor.model_name

                    # Log token usage statistics
                    token_summary = self.processor.token_tracker.get_summary()
                    logger.info(
                        f"Token usage for note {note_input.note_id} with model {deployment_name}: "
                        f"{token_summary['total']['prompt_tokens']} prompt tokens (${token_summary['total']['prompt_cost']:.4f}), "
                        f"{token_summary['total']['completion_tokens']} completion tokens (${token_summary['total']['completion_cost']:.4f}), "
                        f"Total: {token_summary['total']['total_tokens']} tokens, ${token_summary['total']['total_cost']:.4f}"
                    )
                except Exception as e:
                    logger.error(f"Error logging token usage: {e}")

            # Convert the prediction to the expected format
            result = convert_prediction_to_expected_format(prediction)

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
            logger.error(f"Error in processing: {str(e)}", exc_info=True)
            # Return a minimal error result
            return {
                "messages": [],
                "final_output": {},
                "token_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "processing_time": 0.0,
                "error": f"Processing error: {str(e)}",
            }

    def process_batch(self, states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of note states concurrently using the zeroshot workflow.

        Args:
            states: A list of state dictionaries, each containing at least a 'patient_note' key.

        Returns:
            A list of results in the expected format.
        """
        note_inputs = []
        original_indices = []  # Keep track of original order
        for i, state in enumerate(states):
            try:
                patient_note = state.get("patient_note", "")
                note_input = convert_note_to_input_format(patient_note)
                note_inputs.append(note_input)
                original_indices.append(i)
            except Exception as e:
                logger.error(f"Error converting state {i} to NoteInput: {e}")
                # Optionally add an error result for this state

        if not note_inputs:
            return []

        # Process the batch using the processor's batch method
        try:
            batch_predictions = self.processor.process_notes_batch(note_inputs)

            # Log aggregated token usage for the batch without saving to file
            if hasattr(self.processor, "token_tracker"):
                try:
                    # Get the model name based on provider type
                    model_name = self.processor.model_config_key
                    if self.processor.provider_type == "azure" and hasattr(
                        self.processor, "deployment_mapping"
                    ):
                        deployment_name = self.processor.deployment_mapping.get(
                            model_name,
                            self.processor.deployment_mapping.get(
                                "default", model_name
                            ),
                        )
                    else:
                        # For Ollama and vLLM, use the model name directly
                        deployment_name = self.processor.model_name

                    # Log token usage statistics
                    token_summary = self.processor.token_tracker.get_summary()
                    logger.info(
                        f"Total token usage for batch with model {deployment_name}: "
                        f"{token_summary['total']['prompt_tokens']} prompt tokens (${token_summary['total']['prompt_cost']:.4f}), "
                        f"{token_summary['total']['completion_tokens']} completion tokens (${token_summary['total']['completion_cost']:.4f}), "
                        f"Total: {token_summary['total']['total_tokens']} tokens, ${token_summary['total']['total_cost']:.4f}"
                    )
                except Exception as e:
                    logger.error(f"Error logging batch token usage: {e}")
        except Exception as e:
            logger.error(f"Error during batch processing: {e}", exc_info=True)
            # Return error results for all inputs in the batch
            error_result = {
                "messages": [],
                "final_output": {},
                "token_usage": {},
                "processing_time": 0.0,
                "error": f"Batch processing error: {str(e)}",
            }
            return [error_result] * len(note_inputs)

        results = []
        # Ensure the number of predictions matches the number of inputs
        if len(batch_predictions) != len(note_inputs):
            logger.error(
                f"Mismatch between input batch size ({len(note_inputs)}) and prediction count ({len(batch_predictions)})"
            )
            # Handle mismatch, e.g., return error results
            error_result = {
                "messages": [],
                "final_output": {},
                "token_usage": {},
                "processing_time": 0.0,
                "error": "Batch processing size mismatch",
            }
            results = [error_result] * len(note_inputs)
        else:
            for pred in batch_predictions:
                try:
                    # Attach token tracker to each prediction for reporting
                    if hasattr(self.processor, "token_tracker"):
                        pred.TOKEN_TRACKER = self.processor.token_tracker

                    result = convert_prediction_to_expected_format(pred)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error converting prediction to expected format: {e}")
                    # Add an error placeholder for this specific prediction
                    results.append(
                        {
                            "messages": [],
                            "final_output": {},
                            "token_usage": {},
                            "processing_time": 0.0,
                            "error": f"Result conversion error: {str(e)}",
                        }
                    )

        # Note: If inputs were skipped due to conversion errors, the results list
        # might be shorter than the original states list. This implementation assumes
        # all states convert successfully or returns errors for the whole batch.
        # A more robust implementation might map results back using original_indices.
        return results


def create_workflow(
    config: Optional[Any] = None,
    prompt_variant: str = "default",
    parameters: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Create a workflow instance that uses the zeroshot approach.

    Args:
        config: Configuration object that may contain model_name
        prompt_variant: Which prompt variant to use
        parameters: Optional parameters dictionary with additional settings

    Returns:
        An initialized OpenAIZeroshotWorkflow
    """
    # Import MODEL_CONFIGS to find the config key
    from graphs.openai_zeroshot.model_config import MODEL_CONFIGS

    # Get model config name from config object if available
    model_config = "default"
    if config is not None:
        # If config is a string, use it directly
        if isinstance(config, str):
            model_config = config
        else:
            # If config is an object, find the matching key in MODEL_CONFIGS
            # by comparing the model_name attribute
            config_model_name = getattr(config, "model_name", None)
            if config_model_name:
                # Find the key that has this model_name
                for key, cfg in MODEL_CONFIGS.items():
                    if (
                        hasattr(cfg, "model_name")
                        and cfg.model_name == config_model_name
                    ):
                        model_config = key
                        break
                else:
                    # If no matching key found, log a warning and use the model name directly
                    logger.warning(
                        f"No config key found for model_name '{config_model_name}', using 'default'"
                    )
                    model_config = "default"

    # Extract logging configuration if available
    logging_config = None
    if parameters is not None and "logging" in parameters:
        logging_config = parameters["logging"]
        logger.debug(f"Found logging configuration in parameters: {logging_config}")

    logger.debug(f"Creating OpenAIZeroshotWorkflow with model_config='{model_config}'")

    return OpenAIZeroshotWorkflow(
        model_config=model_config,
        prompt_variant=prompt_variant,
        logging_config=logging_config,
        parameters=parameters,
    )


# Create default instance for main.py to access
graph = create_workflow()
