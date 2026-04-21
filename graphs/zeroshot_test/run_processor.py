#!/usr/bin/env python
"""
Run script for OpenAI zeroshot processor.
Processes clinical notes using a single comprehensive prompt.
"""

import os
import sys
import json
import argparse
import logging
import time
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
logger.debug(f"Added parent directory to sys.path: {parent_dir}")

# Try to import data loading utilities
from data.synthetic_notes.load_data import load_patient_data

from graphs.openai_zeroshot.zeroshot_processor import ZeroshotProcessor
from graphs.openai_zeroshot.models import (
    NoteInput,
    NotePrediction,
)


def convert_note_to_input_format(note_data: Dict[str, Any]) -> NoteInput:
    """
    Convert a note dictionary to the zeroshot input format.
    """
    # Get the note text and other metadata
    note_text = note_data.get("note", "")
    patient_id = note_data.get("patient_id", "UNKNOWN")
    timepoint = note_data.get("timepoint", "t1")

    # Create a unique note ID
    note_id = f"{patient_id}_{timepoint}"

    logger.debug(f"Converting note to zeroshot input format with ID: {note_id}")
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

    logger.debug(f"Created note input with ID: {note_id}")
    return note_input


def save_results(
    results: List[Dict[str, Any]],
    output_dir: str,
    model_config: str,
    prompt_variant: str,
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
        f"openai_zeroshot_results_{model_config}_{prompt_variant}_{timestamp}.json",
    )

    # Save the results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.debug(f"Saved results to {output_file}")
    return output_file


def process_notes(
    processor: ZeroshotProcessor, notes_data: List[Dict[str, Any]], batch_size: int = 5
) -> List[Dict[str, Any]]:
    """
    Process a list of notes using the zeroshot processor.

    Args:
        processor: ZeroshotProcessor instance
        notes_data: List of note dictionaries
        batch_size: Number of notes to process in each batch (processed sequentially)

    Returns:
        List of processed results
    """
    all_results = []
    total_notes = len(notes_data)

    logger.info(f"Processing {total_notes} notes with batch size {batch_size}")

    # Process notes in batches (sequential processing within each batch)
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
            # Process each note individually
            batch_results = []
            for note_input in note_inputs:
                logger.debug(f"Processing note {note_input.note_id}")
                result = processor.process_single_note(note_input)
                batch_results.append(result)
                logger.debug(f"Completed processing note {note_input.note_id}")

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time

            logger.debug(f"Batch processed in {batch_duration:.2f} seconds")

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
                        "token_usage": (
                            result.TOKEN_USAGE.to_dict()
                            if hasattr(result.TOKEN_USAGE, "to_dict")
                            else {}
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
        description="Run OpenAI zeroshot processor for clinical notes."
    )

    parser.add_argument(
        "--data_dir",
        default="data/synthetic_notes/",
        help="Directory containing patient data",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/openai_zeroshot/",
        help="Directory to save processed results",
    )
    parser.add_argument(
        "--model_config",
        default="default",
        choices=["default", "o1", "o3_mini", "turbo"],
        help="Model configuration to use",
    )
    parser.add_argument(
        "--prompt_variant",
        default="default",
        choices=["default"],
        help="Prompt variant to use (only default is currently supported)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Number of notes to process in each batch",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with reduced dataset size",
    )
    parser.add_argument(
        "--configure_azure", action="store_true", help="Configure Azure authentication"
    )

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    args = parse_args()

    logger.debug("Starting OpenAI zeroshot processor run script")
    logger.debug(
        f"Configuration: model_config={args.model_config}, prompt_variant={args.prompt_variant}"
    )
    logger.debug(f"Data directory: {args.data_dir}")
    logger.debug(f"Output directory: {args.output_dir}")
    logger.debug(f"Batch size: {args.batch_size}")
    logger.debug(f"Debug mode: {args.debug}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the processor with the specified configuration
    try:
        logger.debug(
            f"Initializing ZeroshotProcessor with model_config={args.model_config}, prompt_variant={args.prompt_variant}"
        )
        processor = ZeroshotProcessor(
            model_config_key=args.model_config, prompt_variant=args.prompt_variant
        )

        # Configure Azure if requested
        if args.configure_azure:
            logger.debug("Configuring Azure authentication")
            processor.configure_azure()
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        sys.exit(1)

    # Load patient data
    try:
        logger.debug(f"Loading patient data from {args.data_dir}")
        patient_data = load_patient_data(args.data_dir)
        logger.debug(f"Loaded {len(patient_data)} patient records")

        # Limit dataset size in debug mode
        if args.debug:
            debug_size = min(2, len(patient_data))
            logger.debug(f"Debug mode: limiting to {debug_size} records")
            patient_data = patient_data[:debug_size]
    except Exception as e:
        logger.error(f"Failed to load patient data: {e}")
        sys.exit(1)

    # Process the notes
    try:
        logger.debug("Processing notes")
        start_time = time.time()
        results = process_notes(
            processor=processor, notes_data=patient_data, batch_size=args.batch_size
        )
        end_time = time.time()

        # Calculate statistics
        total_time = end_time - start_time
        total_notes = len(results)
        avg_time_per_note = total_time / total_notes if total_notes > 0 else 0

        logger.debug(f"Processing completed in {total_time:.2f} seconds")
        logger.debug(f"Average time per note: {avg_time_per_note:.2f} seconds")

        # Save the results
        output_file = save_results(
            results=results,
            output_dir=args.output_dir,
            model_config=args.model_config,
            prompt_variant=args.prompt_variant,
        )

        logger.debug(f"Results saved to {output_file}")
        logger.debug("Processing completed successfully")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
