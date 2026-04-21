"""
Regex entry graph - Integration with main.py.

This file provides integration with the main experimental framework,
connecting it with the regex pattern matching approach.
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

# Define adapter functions locally since ray_regex doesn't exist
class NoteInput:
    """Simple NoteInput class for regex processing."""
    def __init__(
        self,
        pmrn,
        note_id,
        note_type,
        type_name,
        loc_name,
        date,
        prov_name,
        prov_type,
        line,
        note_text,
    ):
        self.pmrn = pmrn
        self.note_id = note_id
        self.note_type = note_type
        self.type_name = type_name
        self.loc_name = loc_name
        self.date = date
        self.prov_name = prov_name
        self.prov_type = prov_type
        self.line = line
        self.note_text = note_text


def convert_note_to_input_format(note_text: str):
    """Convert note text to NoteInput format."""
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
    """Convert regex prediction to expected format."""
    # Initialize the output
    return {
        "messages": [],
        "final_output": (
            prediction.PREDICTION if hasattr(prediction, "PREDICTION") else {}
        ),
        "token_usage": None,  # No token usage in regex processing
        "processing_time": (
            prediction.PROCESSING_TIME
            if hasattr(prediction, "PROCESSING_TIME")
            else 0.0
        ),
        "error": prediction.ERROR if hasattr(prediction, "ERROR") else None,
    }


# Import the Regex processor
from .processor import RegexProcessor


class RegexWorkflow:
    """
    Adapter class that processes notes using the Regex pattern matching approach.
    """

    def __init__(self):
        """Initialize the Regex workflow."""
        # Create the processor instance
        try:
            self.processor = RegexProcessor()
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

        # Create a note input for the processor
        note_input = convert_note_to_input_format(patient_note)

        # Process the note using the processor
        try:
            # Process the note - use async method
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            prediction = loop.run_until_complete(
                self.processor._process_single_note_async(note_input)
            )
            loop.close()

            # Convert the prediction to the expected format
            result = convert_prediction_to_expected_format(prediction)

            return result
        except Exception as e:
            logger.error(f"Error in processing: {str(e)}", exc_info=True)
            # Return a minimal error result
            return {
                "messages": [],
                "final_output": {},
                "token_usage": None,  # No token usage in regex processing
                "processing_time": 0.0,
                "error": f"Processing error: {str(e)}",
            }


def create_workflow(
    config: Optional[Any] = None,
    prompt_variant: str = "default", 
    parameters: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Create a workflow instance that uses the regex approach.

    Args:
        config: Configuration object (not used in regex processor)
        prompt_variant: Which prompt variant to use (not used in regex)
        parameters: Optional parameters dictionary (not used in regex processor)

    Returns:
        An initialized RegexWorkflow
    """
    # Log if parameters are passed (for consistency with other graph types)
    if parameters:
        logger.debug(f"Parameters passed to regex workflow (not used): {parameters}")
    
    return RegexWorkflow()


# Create default instance for main.py to access
graph = create_workflow()
