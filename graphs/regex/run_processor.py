#!/usr/bin/env python
"""
Run script for Regex processor.
Processes clinical notes using regex pattern matching instead of LLM calls.
"""

import os
import sys
import json
import argparse
import logging
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

# Set up logging - this will be configured properly by the environment/sweep settings
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Explicitly use console logging only
)
logger = logging.getLogger(__name__)

# Add parent directory to sys.path to fix imports
parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(parent_dir)
logger.info(f"Added parent directory to sys.path: {parent_dir}")

# Try to import data loading utilities
try:
    from data.synthetic_notes.load_data import load_patient_data

    logger.info("Successfully imported data loading utilities")
except ImportError as e:
    logger.error(f"Failed to import data loading utilities: {e}")
    sys.exit(1)

# Import the processor and models directly
try:
    from graphs.regex.processor import RegexProcessor
    from regex.app.models import NoteInput, NotePrediction

    logger.info("Successfully imported regex processor and models")
except ImportError as e:
    logger.error(f"Failed to import regex processor or models: {e}")
    # Try alternative import path
    sys.path.append(os.path.join(parent_dir, "regex"))
    try:
        from app.models import NoteInput, NotePrediction

        logger.info("Successfully imported models using alternative path")
    except ImportError as e2:
        logger.error(f"All import attempts failed: {e2}")

        # Define fallback models
        class NoteInput:
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

        class NotePrediction:
            def __init__(
                self,
                pmrn,
                note_id,
                note_text,
                shortened_note_text,
                messages,
                prediction,
                processed_at,
                processing_time,
                model_name,
                token_usage,
                error=None,
            ):
                self.PMRN = pmrn
                self.NOTE_ID = note_id
                self.NOTE_TEXT = note_text
                self.SHORTENED_NOTE_TEXT = shortened_note_text
                self.MESSAGES = messages
                self.PREDICTION = prediction
                self.PROCESSED_AT = processed_at
                self.PROCESSING_TIME = processing_time
                self.MODEL_NAME = model_name
                self.TOKEN_USAGE = token_usage
                self.ERROR = error


def convert_note_to_input_format(note_data: Dict[str, Any]) -> NoteInput:
    """
    Convert a note dictionary to the input format.
    """
    # Get the note text and other metadata
    note_text = note_data.get("note", "")
    patient_id = note_data.get("patient_id", "UNKNOWN")
    timepoint = note_data.get("timepoint", "t1")

    # Create a unique note ID
    note_id = f"{patient_id}_{timepoint}"

    logger.info(f"Converting note to input format with ID: {note_id}")
    logger.debug(f"Note length: {len(note_text)} characters")

    # Create a minimal valid note input
    current_time = datetime.now().strftime("%Y-%m-%d")

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


def save_results(
    results: List[Dict[str, Any]],
    output_dir: str,
):
    """
    Save the processing results to a JSON file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        output_dir,
        f"regex_results_{timestamp}.json",
    )

    # Save the results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {output_file}")
    return output_file


async def process_notes(
    processor: RegexProcessor, notes_data: List[Dict[str, Any]], batch_size: int = 10
) -> List[Dict[str, Any]]:
    """
    Process a list of notes using the regex processor.

    Args:
        processor: RegexProcessor instance
        notes_data: List of note dictionaries
        batch_size: Number of notes to process in each batch (processed in parallel)

    Returns:
        List of processed results
    """
    all_results = []
    total_notes = len(notes_data)

    logger.info(f"Processing {total_notes} notes with batch size {batch_size}")

    # Process notes in batches (parallel processing within each batch)
    for i in range(0, total_notes, batch_size):
        batch = notes_data[i : i + batch_size]
        batch_size_actual = len(batch)
        logger.info(
            f"Processing batch {i//batch_size + 1} with {batch_size_actual} notes"
        )

        # Convert notes to input format
        note_inputs = []
        for note_data in batch:
            try:
                note_input = convert_note_to_input_format(note_data)
                note_inputs.append(note_input)
            except Exception as e:
                logger.error(f"Error converting note to input format: {e}")
                # Add error result
                all_results.append(
                    {
                        "patient_id": note_data.get("patient_id", "UNKNOWN"),
                        "timepoint": note_data.get("timepoint", "UNKNOWN"),
                        "file_id": note_data.get("file_id", "UNKNOWN"),
                        "error": f"Error converting note to input format: {str(e)}",
                        "processed": False,
                    }
                )

        # Process the batch
        batch_start_time = time.time()
        try:
            # Process notes in parallel using asyncio.gather
            batch_results = await processor.process_all_notes(note_inputs)
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time

            logger.info(f"Batch processed in {batch_duration:.2f} seconds")

            # Convert results to a more JSON-friendly format
            for i, result in enumerate(batch_results):
                if isinstance(result, NotePrediction):
                    # Create a dictionary with all attributes from NotePrediction
                    result_dict = {
                        "patient_id": note_inputs[i].pmrn,
                        "note_id": result.NOTE_ID,
                        "processed_at": (
                            result.PROCESSED_AT.isoformat()
                            if hasattr(result, "PROCESSED_AT")
                            else None
                        ),
                        "processing_time": result.PROCESSING_TIME,
                        "prediction": result.PREDICTION,
                        "shortened_note_text": (
                            result.SHORTENED_NOTE_TEXT
                            if hasattr(result, "SHORTENED_NOTE_TEXT")
                            else ""
                        ),
                        "model_name": (
                            result.MODEL_NAME if hasattr(result, "MODEL_NAME") else ""
                        ),
                        "error": result.ERROR if hasattr(result, "ERROR") else None,
                        "processed": True,
                    }
                    all_results.append(result_dict)
                else:
                    # Handle the case where result is already a dictionary
                    all_results.append(
                        {
                            **result,
                            "processed": True,
                        }
                    )
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Add error results for all notes in the batch
            for note_input in note_inputs:
                all_results.append(
                    {
                        "patient_id": note_input.pmrn,
                        "note_id": note_input.note_id,
                        "error": f"Error processing batch: {str(e)}",
                        "processed": False,
                    }
                )

    logger.info(f"Processed {len(all_results)} notes total")
    return all_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Regex processor for clinical notes."
    )

    parser.add_argument(
        "--data_dir",
        default="data/synthetic_notes/",
        help="Directory containing patient data",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/regex/",
        help="Directory to save processed results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of notes to process in each batch",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with reduced dataset size",
    )

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    args = parse_args()

    logger.info("Starting Regex processor run script")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Debug mode: {args.debug}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the processor
    try:
        logger.info("Initializing RegexProcessor")
        processor = RegexProcessor()
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        sys.exit(1)

    # Load patient data
    try:
        logger.info(f"Loading patient data from {args.data_dir}")
        patient_data = load_patient_data(args.data_dir)
        logger.info(f"Loaded {len(patient_data)} patient records")

        # Limit dataset size in debug mode
        if args.debug:
            debug_size = min(2, len(patient_data))
            logger.info(f"Debug mode: limiting to {debug_size} records")
            patient_data = patient_data[:debug_size]
    except Exception as e:
        logger.error(f"Failed to load patient data: {e}")
        sys.exit(1)

    # Process the notes
    try:
        logger.info("Processing notes")
        start_time = time.time()

        # Create event loop and run async process_notes
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(
            process_notes(
                processor=processor, notes_data=patient_data, batch_size=args.batch_size
            )
        )
        end_time = time.time()

        # Calculate statistics
        total_time = end_time - start_time
        total_notes = len(results)
        avg_time_per_note = total_time / total_notes if total_notes > 0 else 0

        logger.info(f"Processing completed in {total_time:.2f} seconds")
        logger.info(f"Average time per note: {avg_time_per_note:.2f} seconds")

        # Save the results
        output_file = save_results(
            results=results,
            output_dir=args.output_dir,
        )

        logger.info(f"Results saved to {output_file}")
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
