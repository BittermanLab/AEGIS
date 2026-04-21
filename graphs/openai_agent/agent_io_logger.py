"""
Agent I/O Logger for capturing inputs and outputs from all agents.

This module provides a class for logging all agent prompts and responses to a file.
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class AgentIOLogger:
    """
    Logger for capturing agent prompts and responses.

    This class intercepts agent calls to record their prompts and responses,
    storing them in a structured log file for analysis.
    """

    def __init__(self, log_dir: str = "agent_logs"):
        """
        Initialize the AgentIOLogger with a directory for log files.

        Args:
            log_dir: Directory to store log files (default: "agent_logs")
        """
        self.log_dir = log_dir
        self.entries: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"Agent IO Logger initialized with log directory: {log_dir}")

        # Generate timestamp for filename
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"agent_io_{self.timestamp}.jsonl")

    def log_agent_io(
        self,
        agent_name: str,
        prompt: str,
        response: Any,
        event_type: Optional[str] = None,
        agent_type: Optional[str] = None,
        request_id: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an agent's input and output.

        Args:
            agent_name: The name of the agent
            prompt: The prompt sent to the agent
            response: The response received from the agent
            event_type: Optional event type (e.g., "Pneumonitis")
            agent_type: Optional agent type (e.g., "identifier", "grader")
            request_id: Optional request ID for correlation
            additional_info: Any additional information to log
        """
        # Create log entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "event_type": event_type,
            "agent_type": agent_type,
            "request_id": request_id,
            "prompt": prompt,
        }

        # Extract response - handle different response formats
        if hasattr(response, "final_output"):
            # Handle OpenAI Agents SDK format
            entry["response"] = self._extract_response_data(response.final_output)
        elif hasattr(response, "choices") and len(response.choices) > 0:
            # Handle raw OpenAI API response format
            message = response.choices[0].message
            entry["response"] = {
                "content": (
                    message.content if hasattr(message, "content") else str(message)
                ),
                "role": message.role if hasattr(message, "role") else "assistant",
            }
        else:
            # Handle other formats by converting to string
            entry["response"] = str(response)

        # Add any additional info
        if additional_info:
            entry["additional_info"] = additional_info

        # Add to entries list and write to file (thread-safe)
        with self._lock:
            self.entries.append(entry)
            self._write_entry_to_file(entry)

    def _extract_response_data(self, response_obj: Any) -> Union[Dict[str, Any], str]:
        """
        Extract relevant data from a response object.

        Args:
            response_obj: The response object to extract data from

        Returns:
            Dictionary of extracted data or string representation
        """
        try:
            # If it's a simple object like string, int, etc.
            if isinstance(response_obj, (str, int, float, bool)):
                return response_obj

            # If it's a dictionary
            if isinstance(response_obj, dict):
                return response_obj

            # If it has a to_dict or dict method
            if hasattr(response_obj, "to_dict"):
                return response_obj.to_dict()

            if hasattr(response_obj, "dict"):
                return response_obj.dict()

            # If it has __dict__ attribute (most objects)
            if hasattr(response_obj, "__dict__"):
                return response_obj.__dict__

            # Convert to string as last resort
            return str(response_obj)
        except Exception as e:
            logger.warning(f"Error extracting response data: {e}")
            return str(response_obj)

    def _write_entry_to_file(self, entry: Dict[str, Any]) -> None:
        """
        Write a single entry to the log file in JSONL format.

        Args:
            entry: The log entry to write
        """
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Error writing to log file: {e}")

    def get_entries(self) -> List[Dict[str, Any]]:
        """
        Get all logged entries.

        Returns:
            List of all log entries
        """
        with self._lock:
            return self.entries.copy()

    def save_entries(self, output_file: Optional[str] = None) -> str:
        """
        Save all entries to a JSON file.

        Args:
            output_file: Optional file path to save to

        Returns:
            Path to the saved file
        """
        if output_file is None:
            output_file = os.path.join(
                self.log_dir, f"agent_io_summary_{self.timestamp}.json"
            )

        with self._lock:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, indent=2)

        logger.info(f"Saved {len(self.entries)} log entries to {output_file}")
        return output_file

    def clear_entries(self) -> None:
        """Clear all stored entries."""
        with self._lock:
            self.entries.clear()


# Monkey patch for the Runner.run method to capture inputs/outputs
original_runner_run = None
io_logger = None


def init_io_logger(log_dir: str = "agent_logs") -> AgentIOLogger:
    """
    Initialize the agent IO logger and set up the monkey patching.

    Args:
        log_dir: Directory to store log files

    Returns:
        The initialized AgentIOLogger instance
    """
    global io_logger, original_runner_run

    try:
        from agents import Runner

        # Store the original method if we haven't already
        if original_runner_run is None:
            original_runner_run = Runner.run

        # Create the IO logger
        io_logger = AgentIOLogger(log_dir)

        # Define the patched run method
        async def patched_run(agent, prompt, *, context=None, run_config=None):
            # Call the original method
            response = await original_runner_run(
                agent, prompt, context=context, run_config=run_config
            )

            # Log the input/output
            agent_name = agent.name if hasattr(agent, "name") else "unknown_agent"
            event_type = None
            if context and hasattr(context, "event_type"):
                event_type = context.event_type

            request_id = None
            if context and hasattr(context, "request_id"):
                request_id = context.request_id

            io_logger.log_agent_io(
                agent_name=agent_name,
                prompt=prompt,
                response=response,
                event_type=event_type,
                request_id=request_id,
            )

            return response

        # Monkey patch the Runner.run method
        Runner.run = patched_run
        logger.info("Monkey patched Runner.run to capture agent I/O")

        return io_logger

    except ImportError:
        logger.error("Failed to import agents package. IO logging unavailable.")
        io_logger = AgentIOLogger(log_dir)  # Create a dummy logger
        return io_logger


def restore_runner():
    """Restore the original Runner.run method."""
    global original_runner_run
    if original_runner_run:
        try:
            from agents import Runner

            Runner.run = original_runner_run
            logger.info("Restored original Runner.run method")
        except ImportError:
            logger.error("Failed to import agents package. Cannot restore Runner.run.")
