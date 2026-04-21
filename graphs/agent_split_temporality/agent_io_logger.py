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
        system_prompt: Optional[str] = None,
        token_usage: Optional[Dict[str, Any]] = None,
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
            system_prompt: The system prompt/instructions for the agent
            token_usage: Token usage information including prompt/completion/total tokens and costs
        """
        # Create log entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "event_type": event_type,
            "agent_type": agent_type,
            "request_id": request_id,
            "system_prompt": system_prompt,
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

        # Add token usage information
        if token_usage:
            entry["token_usage"] = token_usage

        # Add any additional info
        if additional_info:
            entry["additional_info"] = additional_info

        # Log to console in real-time
        self._log_to_console(entry)

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

    def _log_to_console(self, entry: Dict[str, Any]) -> None:
        """
        Log agent activity to console in real-time.

        Args:
            entry: The log entry to display
        """
        timestamp = entry.get("timestamp", "")
        agent_name = entry.get("agent_name", "Unknown")
        event_type = entry.get("event_type", "")
        request_id = entry.get("request_id", "")
        
        # Format the console output
        header = f"\n{'='*80}\n[{timestamp}] AGENT: {agent_name}"
        if event_type:
            header += f" | EVENT: {event_type}"
        if request_id:
            header += f" | REQUEST: {request_id[:8]}..."
        
        logger.info(header)
        logger.info("="*80)
        
        # Log system prompt (truncated)
        system_prompt = entry.get("system_prompt", "")
        if system_prompt:
            system_preview = system_prompt[:300] + "..." if len(system_prompt) > 300 else system_prompt
            logger.info(f"SYSTEM PROMPT:\n{system_preview}")
        
        # Log prompt (truncated)
        prompt = entry.get("prompt", "")
        prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
        logger.info(f"\nUSER PROMPT:\n{prompt_preview}")
        
        # Log token usage if available
        token_usage = entry.get("token_usage", {})
        if token_usage:
            logger.info(f"\nTOKEN USAGE:")
            logger.info(f"  Prompt tokens: {token_usage.get('prompt_tokens', 'N/A')}")
            logger.info(f"  Completion tokens: {token_usage.get('completion_tokens', 'N/A')}")  
            logger.info(f"  Total tokens: {token_usage.get('total_tokens', 'N/A')}")
            logger.info(f"  Cost: ${token_usage.get('total_cost', 0.0):.4f}")
        
        # Log response (formatted)
        response = entry.get("response", {})
        if isinstance(response, dict):
            # Pretty print dict responses
            response_str = json.dumps(response, indent=2)
            response_preview = response_str[:1000] + "..." if len(response_str) > 1000 else response_str
        else:
            response_preview = str(response)[:1000] + "..." if len(str(response)) > 1000 else str(response)
        
        logger.info(f"\nRESPONSE:\n{response_preview}")
        logger.info("="*80)

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

    def get_total_token_usage(self) -> Dict[str, Any]:
        """
        Calculate total token usage across all logged entries.
        
        Returns:
            Dictionary containing aggregated token usage statistics
        """
        with self._lock:
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_cost = 0.0
            agent_breakdown = {}
            event_breakdown = {}
            
            for entry in self.entries:
                token_usage = entry.get("token_usage", {})
                agent_name = entry.get("agent_name", "unknown")
                event_type = entry.get("event_type", "unknown")
                
                if token_usage:
                    prompt_tokens = token_usage.get("prompt_tokens", 0)
                    completion_tokens = token_usage.get("completion_tokens", 0)
                    cost = token_usage.get("total_cost", 0.0)
                    
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    total_cost += cost
                    
                    # Agent breakdown
                    if agent_name not in agent_breakdown:
                        agent_breakdown[agent_name] = {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_cost": 0.0,
                            "count": 0
                        }
                    agent_breakdown[agent_name]["prompt_tokens"] += prompt_tokens
                    agent_breakdown[agent_name]["completion_tokens"] += completion_tokens
                    agent_breakdown[agent_name]["total_cost"] += cost
                    agent_breakdown[agent_name]["count"] += 1
                    
                    # Event breakdown
                    if event_type not in event_breakdown:
                        event_breakdown[event_type] = {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_cost": 0.0,
                            "count": 0
                        }
                    event_breakdown[event_type]["prompt_tokens"] += prompt_tokens
                    event_breakdown[event_type]["completion_tokens"] += completion_tokens
                    event_breakdown[event_type]["total_cost"] += cost
                    event_breakdown[event_type]["count"] += 1
            
            return {
                "summary": {
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_completion_tokens": total_completion_tokens,
                    "total_tokens": total_prompt_tokens + total_completion_tokens,
                    "total_cost": total_cost,
                    "total_entries": len(self.entries),
                    "entries_with_tokens": sum(1 for e in self.entries if e.get("token_usage"))
                },
                "by_agent": agent_breakdown,
                "by_event": event_breakdown
            }


def _extract_token_usage(response: Any) -> Optional[Dict[str, Any]]:
    """
    Extract token usage information from a response object.
    
    Args:
        response: The response object to extract token usage from
        
    Returns:
        Dictionary containing token usage information or None
    """
    try:
        # Try different ways to access usage information
        usage = None
        
        # Direct usage attribute (Azure OpenAI)
        if hasattr(response, "usage") and response.usage is not None:
            usage = response.usage
        # Raw responses array (OpenAI Agents SDK)
        elif hasattr(response, "raw_responses") and response.raw_responses:
            for resp in response.raw_responses:
                if hasattr(resp, "usage") and resp.usage is not None:
                    usage = resp.usage
                    break
        # Response.usage format
        elif hasattr(response, "response") and hasattr(response.response, "usage"):
            usage = response.response.usage
        
        if not usage:
            return None
            
        # Convert usage to dictionary format
        token_data = {}
        
        # Handle different token field names
        if hasattr(usage, "prompt_tokens"):
            token_data["prompt_tokens"] = usage.prompt_tokens
        elif hasattr(usage, "input_tokens"):
            token_data["prompt_tokens"] = usage.input_tokens
            
        if hasattr(usage, "completion_tokens"):
            token_data["completion_tokens"] = usage.completion_tokens
        elif hasattr(usage, "output_tokens"):
            token_data["completion_tokens"] = usage.output_tokens
            
        if hasattr(usage, "total_tokens"):
            token_data["total_tokens"] = usage.total_tokens
        else:
            # Calculate total if not provided
            prompt_tokens = token_data.get("prompt_tokens", 0)
            completion_tokens = token_data.get("completion_tokens", 0)
            token_data["total_tokens"] = prompt_tokens + completion_tokens
            
        # Calculate costs if possible (use same rates as TokenUsageTracker)
        # Note: This should match the rates used in the main token tracking system
        prompt_tokens = token_data.get("prompt_tokens", 0)
        completion_tokens = token_data.get("completion_tokens", 0)
        
        # Use rates that match the main TokenUsageTracker (gpt-4.1-nano rates)
        default_prompt_cost = 0.1  # $0.10 per 1M prompt tokens (matches token_tracker.py)
        default_completion_cost = 0.4  # $0.40 per 1M completion tokens (matches token_tracker.py)
        
        if prompt_tokens > 0 or completion_tokens > 0:
            prompt_cost = (prompt_tokens / 1_000_000) * default_prompt_cost
            completion_cost = (completion_tokens / 1_000_000) * default_completion_cost
            total_cost = prompt_cost + completion_cost
            
            token_data["prompt_cost"] = prompt_cost
            token_data["completion_cost"] = completion_cost
            token_data["total_cost"] = total_cost
            
        return token_data
        
    except Exception as e:
        logger.debug(f"Error extracting token usage: {e}")
        return None


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

            # Extract agent information
            agent_name = agent.name if hasattr(agent, "name") else "unknown_agent"
            event_type = None
            if context and hasattr(context, "event_type"):
                event_type = context.event_type

            request_id = None
            if context and hasattr(context, "request_id"):
                request_id = context.request_id

            # Extract system prompt from agent
            system_prompt = None
            if hasattr(agent, "instructions"):
                if callable(agent.instructions):
                    # Dynamic instructions - call the function if we have context
                    try:
                        if context:
                            from agents import RunContextWrapper
                            wrapped_context = RunContextWrapper(context)
                            system_prompt = agent.instructions(wrapped_context, agent)
                        else:
                            system_prompt = "Dynamic instructions (no context available)"
                    except Exception as e:
                        system_prompt = f"Dynamic instructions (error: {str(e)})"
                else:
                    # Static instructions
                    system_prompt = str(agent.instructions)

            # Extract token usage from response
            token_usage = None
            try:
                token_usage = _extract_token_usage(response)
            except Exception as e:
                logger.debug(f"Could not extract token usage: {e}")

            io_logger.log_agent_io(
                agent_name=agent_name,
                prompt=prompt,
                response=response,
                event_type=event_type,
                request_id=request_id,
                system_prompt=system_prompt,
                token_usage=token_usage,
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


def print_token_usage_summary(logger_instance: AgentIOLogger) -> None:
    """
    Print a summary of total token usage from the logger.
    
    Args:
        logger_instance: The AgentIOLogger instance to get usage from
    """
    usage_summary = logger_instance.get_total_token_usage()
    summary = usage_summary["summary"]
    
    print("\n" + "="*60)
    print("TOTAL TOKEN USAGE SUMMARY")
    print("="*60)
    print(f"Total Entries Logged: {summary['total_entries']}")
    print(f"Entries with Token Data: {summary['entries_with_tokens']}")
    print(f"Total Prompt Tokens: {summary['total_prompt_tokens']:,}")
    print(f"Total Completion Tokens: {summary['total_completion_tokens']:,}")
    print(f"Total Tokens: {summary['total_tokens']:,}")
    print(f"Total Cost: ${summary['total_cost']:.4f}")
    
    print("\nBY AGENT:")
    print("-" * 40)
    for agent_name, agent_data in usage_summary["by_agent"].items():
        print(f"{agent_name}:")
        print(f"  Calls: {agent_data['count']}")
        print(f"  Prompt tokens: {agent_data['prompt_tokens']:,}")
        print(f"  Completion tokens: {agent_data['completion_tokens']:,}")
        print(f"  Cost: ${agent_data['total_cost']:.4f}")
    
    print("\nBY EVENT TYPE:")
    print("-" * 40)
    for event_type, event_data in usage_summary["by_event"].items():
        print(f"{event_type}:")
        print(f"  Calls: {event_data['count']}")
        print(f"  Prompt tokens: {event_data['prompt_tokens']:,}")
        print(f"  Completion tokens: {event_data['completion_tokens']:,}")
        print(f"  Cost: ${event_data['total_cost']:.4f}")
    
    print("="*60)
