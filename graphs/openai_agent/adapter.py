"""
Data conversion utilities for the OpenAI Agent SDK implementation.
"""

import logging
import uuid
from typing import Dict, Any, Union, List, Optional
from datetime import datetime

from .parallel_agents.models.input_models import NoteInput, TokenUsageMetadata
from .parallel_agents.models.output_models import NotePrediction

logger = logging.getLogger(__name__)


def convert_note_to_input_format(note_data: Union[str, Dict[str, Any]]) -> NoteInput:
    """
    Convert a note to the input format expected by the OpenAI agent processor.

    Args:
        note_data: Either a string containing the note text or a dictionary with note data

    Returns:
        NoteInput object ready for processing
    """
    current_time = datetime.now().strftime("%Y-%m-%d")

    # Handle string input (raw note text)
    if isinstance(note_data, str):
        note_text = note_data
        patient_id = "SYNTHETIC"
        timepoint = "t1"
        note_id = f"note_{uuid.uuid4().hex[:8]}"
    else:
        # Handle dictionary input
        note_text = note_data.get("patient_note", note_data.get("note", ""))
        if not note_text:
            logger.critical(
                "EMPTY NOTE TEXT DETECTED - This will cause fabrication issues!"
            )

        # Extract metadata
        patient_id = note_data.get("patient_id", "SYNTHETIC")
        timepoint = note_data.get("timepoint", "t1")

        # Create a unique note ID
        if patient_id not in ("SYNTHETIC", "UNKNOWN"):
            note_id = f"{patient_id}_{timepoint}"
        else:
            note_id = f"note_{uuid.uuid4().hex[:8]}"

    # Create the input object
    note_input = NoteInput(
        pmrn=patient_id,
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

    logger.info(f"Created note input with ID: {note_id}")
    return note_input


def convert_prediction_to_expected_format(prediction: NotePrediction) -> Dict[str, Any]:
    """
    Convert an agent prediction to the format expected by the experimental framework.

    Args:
        prediction: NotePrediction object from the agent processor

    Returns:
        Dictionary with standardized prediction data
    """
    # Initialize result structure
    result = {
        "messages": [],
        "final_output": {},
        "token_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_cost": 0.0,
            "completion_cost": 0.0,
            "total_cost": 0.0,
            "model_costs": {},
        },
        "processing_time": 0.0,
        "error": None,
    }

    # Extract messages
    if hasattr(prediction, "MESSAGES") and isinstance(prediction.MESSAGES, list):
        result["messages"] = [str(msg) for msg in prediction.MESSAGES]

    # Extract prediction data
    if hasattr(prediction, "PREDICTION") and isinstance(prediction.PREDICTION, dict):
        # Check if we need to flatten the dictionary structure
        nested_dicts = False
        for k, v in prediction.PREDICTION.items():
            if k in [
                "past_grades",
                "current_grades",
                "past_attributions",
                "current_attributions",
                "past_certainties",
                "current_certainties",
                "evidences",
                "user_overviews",
            ]:
                nested_dicts = True
                break

        if nested_dicts:
            # Need to flatten the nested structure
            for k, v in prediction.PREDICTION.items():
                if k == "past_grades" and isinstance(v, dict):
                    for event_type, grade in v.items():
                        result["final_output"][f"{event_type}_past_grade"] = grade

                elif k == "current_grades" and isinstance(v, dict):
                    for event_type, grade in v.items():
                        result["final_output"][f"{event_type}_current_grade"] = grade

                elif k == "past_attributions" and isinstance(v, dict):
                    for event_type, attr in v.items():
                        result["final_output"][f"{event_type}_past_attribution"] = attr

                elif k == "current_attributions" and isinstance(v, dict):
                    for event_type, attr in v.items():
                        result["final_output"][
                            f"{event_type}_current_attribution"
                        ] = attr

                elif k == "past_certainties" and isinstance(v, dict):
                    for event_type, cert in v.items():
                        result["final_output"][f"{event_type}_past_certainty"] = cert

                elif k == "current_certainties" and isinstance(v, dict):
                    for event_type, cert in v.items():
                        result["final_output"][f"{event_type}_current_certainty"] = cert

                elif k == "evidences" and isinstance(v, dict):
                    for event_type, evidence in v.items():
                        result["final_output"][f"{event_type}_evidence"] = evidence

                elif k == "user_overviews" and isinstance(v, dict):
                    for event_type, overview in v.items():
                        result["final_output"][f"{event_type}_user_overview"] = overview

                else:
                    # Other fields like evidence, reasoning, user_summary
                    if isinstance(v, (str, int, float, bool, type(None))):
                        result["final_output"][str(k)] = v
                    else:
                        result["final_output"][str(k)] = str(v)
        else:
            # Regular flat structure, just copy values
            for k, v in prediction.PREDICTION.items():
                # Ensure serializable values
                if isinstance(v, (str, int, float, bool, type(None))):
                    result["final_output"][str(k)] = v
                else:
                    result["final_output"][str(k)] = str(v)

    # Extract token usage - ENHANCED to handle different formats
    if hasattr(prediction, "TOKEN_USAGE") and prediction.TOKEN_USAGE is not None:
        if hasattr(prediction.TOKEN_USAGE, "to_dict") and callable(
            getattr(prediction.TOKEN_USAGE, "to_dict")
        ):
            token_usage_dict = prediction.TOKEN_USAGE.to_dict()
            result["token_usage"].update(token_usage_dict)
        elif hasattr(prediction.TOKEN_USAGE, "model_dump") and callable(
            getattr(prediction.TOKEN_USAGE, "model_dump")
        ):
            token_usage_dict = prediction.TOKEN_USAGE.model_dump()
            result["token_usage"].update(token_usage_dict)
        elif hasattr(prediction.TOKEN_USAGE, "prompt_tokens") and hasattr(
            prediction.TOKEN_USAGE, "completion_tokens"
        ):
            # Direct attribute access for TokenUsageMetadata objects
            result["token_usage"][
                "prompt_tokens"
            ] = prediction.TOKEN_USAGE.prompt_tokens
            result["token_usage"][
                "completion_tokens"
            ] = prediction.TOKEN_USAGE.completion_tokens
            result["token_usage"]["total_tokens"] = prediction.TOKEN_USAGE.total_tokens

            # Calculate costs based on standard model costs
            model_name = getattr(prediction, "MODEL_NAME", "default")
            cost_per_million = {
                "default": {"prompt": 0.15, "completion": 0.60},
                "o1": {"prompt": 15.0, "completion": 60.0},
                "o3_mini": {"prompt": 1.10, "completion": 4.40},
                "o3_early": {"prompt": 3.0, "completion": 15.0},
                "o3_late": {"prompt": 3.0, "completion": 15.0},
                "hybrid": {"prompt": 3.0, "completion": 15.0},
            }.get(model_name, {"prompt": 0.15, "completion": 0.60})

            prompt_cost = (
                result["token_usage"]["prompt_tokens"] / 1000000
            ) * cost_per_million["prompt"]
            completion_cost = (
                result["token_usage"]["completion_tokens"] / 1000000
            ) * cost_per_million["completion"]
            total_cost = prompt_cost + completion_cost

            result["token_usage"]["prompt_cost"] = prompt_cost
            result["token_usage"]["completion_cost"] = completion_cost
            result["token_usage"]["total_cost"] = total_cost
            result["token_usage"]["model_costs"] = {
                model_name: {
                    "prompt_tokens": result["token_usage"]["prompt_tokens"],
                    "completion_tokens": result["token_usage"]["completion_tokens"],
                    "total_tokens": result["token_usage"]["total_tokens"],
                    "prompt_cost": prompt_cost,
                    "completion_cost": completion_cost,
                    "total_cost": total_cost,
                }
            }

        elif isinstance(prediction.TOKEN_USAGE, dict):
            # It's already a dictionary
            token_usage_dict = {str(k): v for k, v in prediction.TOKEN_USAGE.items()}
            result["token_usage"].update(token_usage_dict)

            # Add costs if not already included
            if "prompt_cost" not in token_usage_dict:
                # Calculate costs based on standard model costs
                model_name = getattr(prediction, "MODEL_NAME", "default")
                cost_per_million = {
                    "default": {"prompt": 0.15, "completion": 0.60},
                    "o1": {"prompt": 15.0, "completion": 60.0},
                    "o3_mini": {"prompt": 1.10, "completion": 4.40},
                    "o3_early": {"prompt": 3.0, "completion": 15.0},
                    "o3_late": {"prompt": 3.0, "completion": 15.0},
                    "hybrid": {"prompt": 3.0, "completion": 15.0},
                }.get(model_name, {"prompt": 0.15, "completion": 0.60})

                prompt_tokens = token_usage_dict.get("prompt_tokens", 0)
                completion_tokens = token_usage_dict.get("completion_tokens", 0)

                prompt_cost = (prompt_tokens / 1000000) * cost_per_million["prompt"]
                completion_cost = (completion_tokens / 1000000) * cost_per_million[
                    "completion"
                ]
                total_cost = prompt_cost + completion_cost

                result["token_usage"]["prompt_cost"] = prompt_cost
                result["token_usage"]["completion_cost"] = completion_cost
                result["token_usage"]["total_cost"] = total_cost
                result["token_usage"]["model_costs"] = {
                    model_name: {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "prompt_cost": prompt_cost,
                        "completion_cost": completion_cost,
                        "total_cost": total_cost,
                    }
                }

    # Extract processing time
    if hasattr(prediction, "PROCESSING_TIME"):
        try:
            result["processing_time"] = float(prediction.PROCESSING_TIME)
        except (ValueError, TypeError):
            pass

    # Extract error
    if hasattr(prediction, "ERROR") and prediction.ERROR:
        result["error"] = str(prediction.ERROR)

    # Extract the extracted note (shortened note text)
    if hasattr(prediction, "SHORTENED_NOTE_TEXT"):
        result["extracted_note"] = prediction.SHORTENED_NOTE_TEXT

    return result


def create_note_prediction(
    note_id: str,
    patient_id: str,
    timepoint: str,
    model: str,
    events: List[Any],
    processing_time: float,
    token_usage: Optional[Dict[str, Any]] = None,
) -> NotePrediction:
    """
    Create a NotePrediction object from processed events.

    Args:
        note_id: The ID of the note
        patient_id: The patient ID
        timepoint: The timepoint
        model: The model name
        events: List of event results
        processing_time: Time taken to process
        token_usage: Optional token usage information

    Returns:
        A NotePrediction object
    """
    # Convert events to a dictionary format
    prediction_dict = {}
    for event in events:
        prediction_dict[event.event_type] = {
            "grade": event.grade,
            "attribution": event.attribution,
            "certainty": event.certainty,
            "reasoning": event.reasoning,
            "past_grade": getattr(event, "past_grade", 0),
            "current_grade": getattr(event, "current_grade", 0),
        }

        # Add evidence if available
        if hasattr(event, "attribution_evidence") and event.attribution_evidence:
            prediction_dict[event.event_type][
                "attribution_evidence"
            ] = event.attribution_evidence

        if hasattr(event, "certainty_evidence") and event.certainty_evidence:
            prediction_dict[event.event_type][
                "certainty_evidence"
            ] = event.certainty_evidence

        if hasattr(event, "identification_evidence") and event.identification_evidence:
            prediction_dict[event.event_type][
                "identification_evidence"
            ] = event.identification_evidence

        if hasattr(event, "past_grading_evidence") and event.past_grading_evidence:
            prediction_dict[event.event_type][
                "past_grading_evidence"
            ] = event.past_grading_evidence

        if (
            hasattr(event, "current_grading_evidence")
            and event.current_grading_evidence
        ):
            prediction_dict[event.event_type][
                "current_grading_evidence"
            ] = event.current_grading_evidence

    # Create the default token usage dictionary if not provided
    if token_usage is None:
        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    # Create the prediction
    return NotePrediction(
        PMRN=patient_id,
        NOTE_ID=note_id,
        NOTE_TEXT="",  # Empty to save space, note is already in database
        SHORTENED_NOTE_TEXT="",
        MESSAGES=[],  # Add messages if relevant
        PREDICTION=prediction_dict,
        PROCESSED_AT=datetime.now(),
        PROCESSING_TIME=processing_time,
        MODEL_NAME=model,
        TOKEN_USAGE=token_usage,
        ERROR=None,
    )


def create_error_prediction(
    note_id: str,
    patient_id: str,
    timepoint: str,
    error_message: str,
    processing_time: float,
) -> NotePrediction:
    """
    Create an error prediction.

    Args:
        note_id: The ID of the note
        patient_id: The patient ID
        timepoint: The timepoint
        error_message: The error message
        processing_time: Time taken to process

    Returns:
        A NotePrediction object with error information
    """
    return NotePrediction(
        PMRN=patient_id,
        NOTE_ID=note_id,
        NOTE_TEXT="",  # Empty to save space
        SHORTENED_NOTE_TEXT="",
        MESSAGES=[],
        PREDICTION={},
        PROCESSED_AT=datetime.now(),
        PROCESSING_TIME=processing_time,
        MODEL_NAME="error",
        TOKEN_USAGE={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "error": True,
        },
        ERROR=error_message,
    )
