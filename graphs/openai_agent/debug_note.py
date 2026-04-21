"""
Debug script for manually testing note processing with detailed agent output.

This script allows you to paste a clinical note and see the detailed input/output
of every agent in the workflow for debugging purposes.

Usage:
  python -m graphs.openai_agent.debug_note --note-file path/to/note.txt

  Or you can use stdin to pass a note:
  cat note.txt | python -m graphs.openai_agent.debug_note
"""

import os
import sys
import json
import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

# Configure specific loggers to DEBUG level rather than using root.manager.loggerDict
for logger_name in ["graphs.openai_agent", "graphs.openai_agent.parallel_agents"]:
    logging.getLogger(logger_name).setLevel(logging.DEBUG)

# Import agent models - we'll use relative imports here
from .parallel_agents.models.input_models import NoteInput
from .processor import LLMProcessor
from .adapter import convert_note_to_input_format, convert_prediction_to_expected_format

# Import necessary components for initializing agents
from .parallel_agents.utils.ctcae_utils import get_ctcae_data
from .parallel_agents.agent_factory import create_enhanced_event_agents_with_judges

# Import our agent I/O logger
from .agent_io_logger import init_io_logger, restore_runner

# Create a custom logger for this script
logger = logging.getLogger("debug_note")
logger.setLevel(logging.DEBUG)


class DebugNoteProcessor:
    """Debug processor for manual testing of note processing."""

    def __init__(
        self,
        model_config: str = "default",
        prompt_variant: str = "default",
        io_log_dir: str = "agent_logs",
    ):
        """Initialize debug processor with specified model config."""
        self.model_config = model_config
        self.prompt_variant = prompt_variant
        self.io_log_dir = io_log_dir

        logger.info(
            f"Initializing with model_config={model_config}, prompt_variant={prompt_variant}"
        )

        # Initialize agent I/O logger
        self.io_logger = init_io_logger(log_dir=io_log_dir)
        logger.info(f"Agent I/O logging enabled. Logs will be saved to {io_log_dir}")

        # Create the LLM Processor
        self.processor = LLMProcessor(
            model_config_key=model_config,
            prompt_variant=prompt_variant,
            max_concurrent_runs=1,  # Single note processing
        )

        # Configure Azure authentication
        self.processor.configure_azure()
        logger.info("Azure authentication configured")

        # Initialize CTCAE data and event agents
        logger.info("Loading CTCAE data and initializing event agents")
        self._initialize_processor_data()

    def _initialize_processor_data(self):
        """Initialize the CTCAE data and event agents for the processor."""
        # Load CTCAE data
        ctcae_data = get_ctcae_data()
        logger.info(f"CTCAE data loaded with {len(ctcae_data)} categories")

        # Create event agents with the loaded CTCAE data
        request_id = f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        event_agents = create_enhanced_event_agents_with_judges(
            ctcae_data=ctcae_data,
            model_config=self.model_config,
            azure_provider=self.processor.azure_provider,
            request_id=request_id,
        )
        logger.info(f"Event agents created for {len(event_agents)} event types")

        # Set these on the processor
        self.processor.ctcae_data = ctcae_data
        self.processor.event_agents = event_agents

    async def process_note_text(
        self, note_text: str, patient_id: str = "DEBUG", timepoint: str = "t1"
    ) -> Dict[str, Any]:
        """
        Process a single note text and return detailed results.

        Args:
            note_text: The clinical note text to process
            patient_id: Optional patient identifier
            timepoint: Optional timepoint

        Returns:
            Dictionary containing prediction results and processing details
        """
        logger.info(
            f"Processing note for patient_id={patient_id}, timepoint={timepoint}"
        )
        logger.info(f"Note text length: {len(note_text)} characters")

        # Convert to input format
        note_input = convert_note_to_input_format(
            {
                "note": note_text,
                "patient_id": patient_id,
                "timepoint": timepoint,
            }
        )

        # Process asynchronously (we use async method directly for more control)
        start_time = datetime.now()
        prediction = await self.processor.process_single_note(note_input)
        elapsed_time = (datetime.now() - start_time).total_seconds()

        # Convert prediction to standard format
        result = convert_prediction_to_expected_format(prediction)

        # Add processing metadata
        result["processing_time"] = elapsed_time
        result["processed_at"] = datetime.now().isoformat()

        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")

        # Save agent I/O logs summary
        if hasattr(self, "io_logger") and self.io_logger:
            summary_file = self.io_logger.save_entries()
            logger.info(f"Agent I/O summary saved to {summary_file}")

        return result


async def main():
    """Main entry point for debugging note processing."""
    parser = argparse.ArgumentParser(description="Debug clinical note processing")
    parser.add_argument("--note-file", help="Path to file containing clinical note")
    parser.add_argument("--model-config", default="default", help="Model configuration")
    parser.add_argument("--prompt-variant", default="default", help="Prompt variant")
    parser.add_argument(
        "--output-file", default="debug_output.json", help="Output file for results"
    )
    parser.add_argument("--patient-id", default="DEBUG", help="Patient ID for note")
    parser.add_argument("--timepoint", default="t1", help="Timepoint for note")
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Enable highly detailed debugging output",
    )
    parser.add_argument(
        "--event-types",
        default="pneumonitis,myocarditis,colitis,thyroiditis,hepatitis,dermatitis",
        help="Comma-separated list of event types to process",
    )
    parser.add_argument(
        "--io-log-dir",
        default="agent_logs",
        help="Directory to store agent I/O logs",
    )
    args = parser.parse_args()

    # Set environment variable for detailed token tracking if requested
    if args.detailed:
        os.environ["TOKEN_TRACKER_LOG_LEVEL"] = "DEBUG"
        # Set specific loggers to DEBUG level
        logging.getLogger().setLevel(logging.DEBUG)
        for logger_name in [
            "graphs.openai_agent",
            "graphs.openai_agent.parallel_agents",
        ]:
            logging.getLogger(logger_name).setLevel(logging.DEBUG)

    # Get note text from file or stdin
    if args.note_file:
        with open(args.note_file, "r") as f:
            note_text = f.read()
    elif not sys.stdin.isatty():
        note_text = sys.stdin.read()
    else:
        print("Enter note text (press Ctrl+D when finished):")
        note_text = sys.stdin.read()

    if not note_text.strip():
        print("Error: Note text is empty")
        return

    # Process the note
    print(
        f"Processing note with model={args.model_config}, variant={args.prompt_variant}"
    )
    processor = DebugNoteProcessor(
        args.model_config, args.prompt_variant, args.io_log_dir
    )

    # Process and time the execution
    start_time = datetime.now()
    result = await processor.process_note_text(
        note_text, args.patient_id, args.timepoint
    )
    elapsed_time = (datetime.now() - start_time).total_seconds()

    # Save output to file
    with open(args.output_file, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Results saved to {args.output_file}")

    # Print key predictions
    print("\nPrediction Summary:")
    for event_type in [
        "pneumonitis",
        "myocarditis",
        "colitis",
        "thyroiditis",
        "hepatitis",
        "dermatitis",
    ]:
        current_grade = result["final_output"].get(f"{event_type}_current_grade", 0)
        current_attr = result["final_output"].get(
            f"{event_type}_current_attribution", 0
        )
        current_cert = result["final_output"].get(f"{event_type}_current_certainty", 0)

        if current_grade > 0:
            print(
                f"  * {event_type.capitalize()}: Grade {current_grade}, Attribution {current_attr}, Certainty {current_cert}"
            )

    print("=" * 80)

    # Print information about detailed output
    print(
        f"\nDetailed agent input/output is available in the {args.io_log_dir} directory"
    )
    print("Each agent's IO is captured in JSONL format for analysis")

    # Clean up and restore original Runner
    restore_runner()


if __name__ == "__main__":
    asyncio.run(main())
