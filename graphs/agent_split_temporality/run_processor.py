#!/usr/bin/env python
"""
OpenAI Agent SDK clinical note processor.

This script processes clinical notes using the OpenAI Agent SDK and outputs results.
"""

import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to sys.path
parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(parent_dir)

# Import data loading utilities
from data.synthetic_notes.load_data import load_patient_data

# Import from our package
from graphs.agent_split_temporality.processor import LLMProcessor
from graphs.agent_split_temporality.adapter import convert_note_to_input_format
from graphs.agent_split_temporality.parallel_agents.models.input_models import NoteInput


def save_results(
    results: List[Dict[str, Any]],
    output_dir: str,
    model_config: str,
    prompt_variant: str,
) -> str:
    """
    Save processing results to a JSON file.

    Args:
        results: List of processing results
        output_dir: Directory to save results
        model_config: Model configuration used
        prompt_variant: Prompt variant used

    Returns:
        Path to the output file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        output_dir,
        f"openai_agent_results_{model_config}_{prompt_variant}_{timestamp}.json",
    )

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    return output_file


def process_notes(
    processor: LLMProcessor, notes_data: List[Dict[str, Any]], batch_size: int = 50
) -> List[Dict[str, Any]]:
    """
    Process clinical notes in batches.

    Args:
        processor: Initialized LLMProcessor
        notes_data: List of note dictionaries
        batch_size: Number of notes to process in each batch

    Returns:
        List of processed results
    """
    all_results = []
    total_notes = len(notes_data)
    logger.info(f"Processing {total_notes} notes with batch size {batch_size}")

    # Process notes in batches
    for i in range(0, total_notes, batch_size):
        batch = notes_data[i : i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} with {len(batch)} notes")

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
                        "error": f"Input format error: {str(e)}",
                        "processed": False,
                    }
                )

        if not note_inputs:
            logger.warning("No valid notes in this batch, skipping")
            continue

        # Process the batch
        batch_start_time = time.time()
        try:
            batch_results = processor.process_batch(note_inputs)

            # Convert to JSON-serializable format
            for idx, result in enumerate(batch_results):
                # Skip already processed error results
                if not hasattr(result, "NOTE_ID"):
                    all_results.append(result)
                    continue

                # Create serializable result
                result_dict = {
                    "patient_id": note_inputs[idx].pmrn,
                    "note_id": result.NOTE_ID,
                    "processed_at": (
                        result.PROCESSED_AT.isoformat()
                        if hasattr(result, "PROCESSED_AT")
                        else None
                    ),
                    "processing_time": result.PROCESSING_TIME,
                    "prediction": result.PREDICTION,
                    "model_name": (
                        result.MODEL_NAME if hasattr(result, "MODEL_NAME") else ""
                    ),
                    "token_usage": (
                        result.TOKEN_USAGE.to_dict()
                        if hasattr(result, "TOKEN_USAGE")
                        else {}
                    ),
                    "error": result.ERROR if hasattr(result, "ERROR") else None,
                    "processed": True,
                }
                all_results.append(result_dict)

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            # Add error results for all notes in the batch
            for note_input in note_inputs:
                all_results.append(
                    {
                        "patient_id": note_input.pmrn,
                        "note_id": note_input.note_id,
                        "error": f"Batch processing error: {str(e)}",
                        "processed": False,
                    }
                )

        batch_duration = time.time() - batch_start_time
        logger.info(f"Batch processed in {batch_duration:.2f} seconds")

    logger.info(f"Processed {len(all_results)} notes total")
    return all_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process clinical notes using the OpenAI Agent SDK."
    )
    parser.add_argument(
        "--data_dir",
        default="data/synthetic_notes/",
        help="Directory containing patient data",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/openai_agent/",
        help="Directory to save results",
    )
    parser.add_argument(
        "--model_config",
        default="default",
        choices=["default", "o1", "o3_mini", "o3_early", "o3_late", "hybrid"],
        help="Model configuration to use",
    )
    parser.add_argument(
        "--prompt_variant",
        default="default",
        choices=["default", "detailed", "concise"],
        help="Prompt variant to use",
    )
    parser.add_argument(
        "--batch_size", type=int, default=5, help="Number of notes per batch"
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=int(os.getenv("MAX_CONCURRENCY", "3")),
        help="Maximum concurrent API calls (can also set via MAX_CONCURRENCY env var)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode with reduced dataset"
    )
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    # Parse arguments
    args = parse_args()

    logger.info(
        f"OpenAI Agent processor starting with model={args.model_config}, variant={args.prompt_variant}"
    )

    # Initialize processor
    try:
        processor = LLMProcessor(
            model_config_key=args.model_config,
            prompt_variant=args.prompt_variant,
            max_concurrent_runs=args.max_concurrency,
        )

        # Configure Azure authentication
        processor.configure_azure()
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        sys.exit(1)

    # Load data
    try:
        logger.info(f"Loading patient data from {args.data_dir}")
        patient_data = load_patient_data(args.data_dir)
        logger.info(f"Loaded {len(patient_data)} patient records")

        # Limit dataset in debug mode
        if args.debug:
            debug_size = min(2, len(patient_data))
            logger.info(f"Debug mode: limiting to {debug_size} records")
            patient_data = patient_data[:debug_size]
    except Exception as e:
        logger.error(f"Failed to load patient data: {e}")
        sys.exit(1)

    # Process notes
    try:
        logger.info("Processing notes")
        start_time = time.time()

        results = process_notes(
            processor=processor, notes_data=patient_data, batch_size=args.batch_size
        )

        total_time = time.time() - start_time
        total_notes = len(results)
        avg_time = total_time / total_notes if total_notes > 0 else 0

        logger.info(f"Processing completed in {total_time:.2f} seconds")
        logger.info(f"Average time per note: {avg_time:.2f} seconds")

        # Save results
        output_file = save_results(
            results=results,
            output_dir=args.output_dir,
            model_config=args.model_config,
            prompt_variant=args.prompt_variant,
        )

        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)
    finally:
        # Clean up agent logger
        try:
            processor.cleanup()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    logger.info("Processing completed successfully")


if __name__ == "__main__":
    main()
