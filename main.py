import json
import os
import argparse
import logging
import time  # for tracking the run time
from typing import Optional, Any, Dict, List, Union
import subprocess
from datetime import datetime
import sys
import asyncio
import uuid  # Import uuid
from data.synthetic_notes.load_data import load_patient_data
from utils.llm_config import token_handlers
from utils.path_utils import GRAPH_TYPE_MAP
from utils.config import get_model_cost_rates
from pathlib import Path

# Setup basic logging - will be overridden by environment settings
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)  # Initialize the logger for the main module

# Set httpx logger to WARNING level to completely suppress HTTP request logs
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

# Disable OpenAI Agents tracing at the entry point
try:
    from agents import set_tracing_disabled

    # Disable tracing for all OpenAI agent executions
    set_tracing_disabled(True)
    # Also set environment variable to ensure child processes disable tracing
    os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"
    logger.info("OpenAI Agents tracing disabled globally")
except ImportError:
    logger.warning(
        "Failed to import set_tracing_disabled, setting environment variable only"
    )
    # Set environment variable to disable tracing
    os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"

# Load environment variables (optional)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    logging.warning("dotenv not installed, skipping .env loading")


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

    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # Override any existing configurations
        handlers=[logging.StreamHandler()],  # Explicitly use console logging only
    )

    # Set the log level for the current logger
    logger.setLevel(log_level)

    # Set httpx logger to WARNING to suppress HTTP request logs
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)

    # Configure openai_agent module loggers
    openai_agent_logger = logging.getLogger("graphs.openai_agent")
    if openai_agent_logger:
        openai_agent_logger.setLevel(log_level)
        # Ensure openai_agent module only logs to console
        openai_agent_logger.handlers = []
        openai_agent_logger.addHandler(logging.StreamHandler())
        openai_agent_logger.propagate = True

    # Set environment variable for token tracking in the openai_agent module
    if log_level == logging.DEBUG:
        os.environ["TOKEN_TRACKER_LOG_LEVEL"] = "DEBUG"
    else:
        os.environ["TOKEN_TRACKER_LOG_LEVEL"] = log_level_str

    # Additional logger configuration
    save_intermediate = logging_config.get("save_intermediate", True)
    token_tracking = logging_config.get("token_tracking", True)

    # Configure other loggers in the project
    for module in ["utils", "graphs", "data"]:
        module_logger = logging.getLogger(module)
        module_logger.setLevel(log_level)

    logger.info(f"Logging configured with level: {log_level_str}")
    logger.debug(
        f"Additional logging settings - save_intermediate: {save_intermediate}, token_tracking: {token_tracking}"
    )


# Load base configuration
def load_base_config():
    base_config_path = "config/base_config.yaml"
    if not os.path.exists(base_config_path):
        raise ValueError(f"Base configuration file not found: {base_config_path}")

    try:
        import yaml

        with open(base_config_path, "r") as f:
            return yaml.safe_load(f)
    except ImportError:
        logging.warning("PyYAML not installed, using simplified config loading")
        # Fallback to a minimal configuration
        return {
            "graph_types": {
                "openai_agent": {
                    "parameters": {},
                    "valid_models": [
                        "default",
                        "4.1-mini",
                        "4.1-nano",
                        "o4_mini",
                        "ollama-deepseek-1b",
                        "ollama-llama-1b",
                        "ollama-qwen3-14b",
                    ],
                    "prompt_variants": {
                        "default": {
                            "name": "default",
                            "description": "Standard instructions",
                            "enabled": True,
                        },
                        "ablation_single": {
                            "name": "ablation_single",
                            "description": "Ablation with 1 identifier agent per event type",
                            "enabled": True,
                        },
                        "ablation_no_judge": {
                            "name": "ablation_no_judge",
                            "description": "Ablation with 3 identifiers but no judge selection",
                            "enabled": True,
                        },
                    },
                },
                "openai_zeroshot": {
                    "parameters": {},
                    "valid_models": [
                        "default",
                        "4.1-mini",
                        "4.1-nano",
                        "o4_mini",
                        "ollama-deepseek-1b",
                        "ollama-llama-1b",
                        "ollama-qwen3-14b",
                    ],
                    "prompt_variants": {
                        "default": {
                            "name": "default",
                            "description": "Standard instructions",
                            "enabled": True,
                        },
                    },
                },
                "agent_split_temporality": {
                    "parameters": {},
                    "valid_models": [
                        "default",
                        "4.1-mini",
                        "4.1-nano",
                        "o4_mini",
                        "ollama-deepseek-1b",
                        "ollama-llama-1b",
                        "ollama-qwen3-14b",
                    ],
                    "prompt_variants": {
                        "default": {
                            "name": "default",
                            "description": "Standard instructions",
                            "enabled": True,
                        },
                        "ablation_single": {
                            "name": "ablation_single",
                            "description": "Ablation with 1 identifier agent per event type",
                            "enabled": True,
                        },
                        "ablation_no_judge": {
                            "name": "ablation_no_judge",
                            "description": "Ablation with 3 identifiers but no judge selection",
                            "enabled": True,
                        },
                    },
                },
                "regex": {
                    "parameters": {},
                    "valid_models": [
                        "default",
                    ],
                    "prompt_variants": {
                        "default": {
                            "name": "default",
                            "description": "Standard instructions",
                            "enabled": True,
                        },
                    },
                },
            }
        }


# Simple message conversion function
def convert_to_dict(message):
    """Converts a message to a dictionary."""
    return {"type": "message", "content": str(message)}


def load_graph(graph_type):
    """
    Load the specified graph type.
    """
    # Extract base graph type from dataset-specific graph types
    base_graph_type = graph_type.split("_")[-1] if "_" in graph_type else graph_type

    # Import the correct module based on the graph type
    if graph_type in GRAPH_TYPE_MAP:
        module_path = GRAPH_TYPE_MAP[graph_type]
        try:
            module = __import__(module_path, fromlist=["graph"])
            return module.graph
        except ImportError as e:
            raise ValueError(f"Failed to import graph module for {graph_type}: {e}")
    else:
        raise ValueError(
            f"Graph type '{graph_type}' not supported. Available types: {list(GRAPH_TYPE_MAP.keys())}"
        )


def run_experiment(
    graph_type,
    data_dir,
    output_base_dir,
    model_config: str = "default",
    prompt_variant: str = "default",
    debug=False,
    environment: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
):
    """Run an experiment with the specified configuration."""
    # Configure logging based on environment settings
    if environment and "logging" in environment:
        configure_logging(environment["logging"])

    # Add logging banner and parameter details at start
    print("\n" + "=" * 80)
    print(f"Starting experiment with configuration:")
    print("=" * 80)
    print(f"Graph Type:      {graph_type}")
    print(f"Model Config:    {model_config}")
    print(f"Prompt Variant:  {prompt_variant}")
    print(f"Debug Mode:      {debug}")
    print(f"Data Directory:  {data_dir}")
    print(f"Output Dir:      {output_base_dir}")
    if parameters:
        print("\nParameters:")
        for key, value in parameters.items():
            print(f"  {key}: {value}")
    if environment:
        print("\nEnvironment Settings:")
        for key, value in environment.items():
            if key != "logging":  # Skip logging details in printout
                print(f"  {key}: {value}")
        # Log the logging config separately
        if "logging" in environment:
            logging_config = environment["logging"]
            print("\nLogging Configuration:")
            for key, value in logging_config.items():
                print(f"  {key}: {value}")
    print("=" * 80 + "\n")

    # Log the configuration with logging module as well
    logger.info(
        f"Starting experiment with graph_type={graph_type}, model_config={model_config}"
    )
    logger.debug(
        f"Full experiment configuration: debug={debug}, prompt_variant={prompt_variant}"
    )

    # Load base configuration
    base_config = load_base_config()

    # Apply environment variables if provided
    if environment:
        for key, value in environment.items():
            if isinstance(value, (str, int, float, bool)):
                os.environ[key] = str(value)

    # Apply parameters to the graph configuration if provided
    if parameters:
        # Store original parameters for reference
        original_parameters = parameters.copy()

        # Update graph-specific parameters
        if graph_type in base_config["graph_types"]:
            graph_params = base_config["graph_types"][graph_type].get("parameters", {})
            graph_params.update(parameters)
            parameters = graph_params

    # Reset all handlers
    for handler in token_handlers.values():
        handler.reset_usage()

    # Get sweep name from parameters first, then environment, then default
    sweep_name = "default_sweep"
    if parameters and "sweep_name" in parameters:
        sweep_name = parameters["sweep_name"]
    elif environment and "sweep_name" in environment:
        sweep_name = environment["sweep_name"]

    # Build the output directory path
    output_dir = os.path.join(
        output_base_dir,
        sweep_name,  # Add sweep name to path
        graph_type,  # Use the actual graph_type
        model_config,
        f"variant_{prompt_variant}",  # Always include prompt variant
    )

    os.makedirs(output_dir, exist_ok=True)

    patient_data = load_patient_data(data_dir)

    # Limit dataset size based on debug mode and max_samples setting
    if debug:
        # Get max_samples from environment runtime settings, default to 2 if not specified
        max_samples = environment.get("max_samples", 2) if environment else 2
        patient_data = patient_data[:max_samples]
        logging.info(f"Debug mode: Using {max_samples} samples")
        logging.info(f"New dataset size: {len(patient_data)}")

    # Get the module path from GRAPH_TYPE_MAP
    if graph_type not in GRAPH_TYPE_MAP:
        raise ValueError(f"Graph type '{graph_type}' not found in GRAPH_TYPE_MAP")

    module_path = GRAPH_TYPE_MAP[graph_type]
    # Split string by '.' to get package components
    if isinstance(module_path, str):
        base_path = ".".join(module_path.split(".")[:-1])
    else:
        raise TypeError(f"Expected string for module_path, got {type(module_path)}")

    # Import model configs from the appropriate module
    try:
        model_config_module = __import__(
            f"{base_path}.model_config", fromlist=["MODEL_CONFIGS"]
        )
        MODEL_CONFIGS = model_config_module.MODEL_CONFIGS

        if model_config not in MODEL_CONFIGS:
            raise ValueError(
                f"Model configuration '{model_config}' not found in {base_path}.model_config!"
            )

        # Import create_workflow from the appropriate module
        entry_graph_module = __import__(
            f"{base_path}.entry_graph", fromlist=["create_workflow"]
        )
        create_workflow = entry_graph_module.create_workflow

        # Determine if this graph type uses prompt variants
        base_graph_type = graph_type.split("_")[-1] if "_" in graph_type else graph_type

        # Agent and zeroshot types use prompt variants, regex doesn't
        if base_graph_type in ["agent", "zeroshot"]:
            # Pass logging settings to the workflow
            if environment and "logging" in environment:
                if not parameters:
                    parameters = {}
                parameters["logging"] = environment["logging"]

            graph = create_workflow(
                MODEL_CONFIGS[model_config],
                prompt_variant=prompt_variant,
                parameters=parameters,  # Pass parameters to workflow creation
            )
        else:
            # For regex, still pass logging settings if available
            if environment and "logging" in environment:
                if not parameters:
                    parameters = {}
                parameters["logging"] = environment["logging"]
            graph = create_workflow(MODEL_CONFIGS[model_config], parameters=parameters)

    except ImportError as e:
        raise ValueError(f"Failed to import modules for graph type '{graph_type}': {e}")
    except AttributeError as e:
        raise ValueError(
            f"Missing required attribute in modules for graph type '{graph_type}': {e}"
        )

    # Create a run ID for the entire experiment
    experiment_run_id = str(uuid.uuid4())
    experiment_run_name = (
        f"{graph_type}_{model_config}_variant_{prompt_variant}_{experiment_run_id}"
    )

    # Start timing
    total_patients = len(patient_data)
    start_time = time.perf_counter()

    # SIMPLIFIED APPROACH: Process all notes at once
    logger.info(
        f"Processing all {total_patients} notes at once - processor semaphore will handle concurrency"
    )

    # Prepare all input states
    all_input_states = []
    for data in patient_data:
        input_state = {
            "patient_note": data["note"],
            "patient_id": data["patient_id"],
            "timepoint": data["timepoint"],
            "file_id": data["file_id"],
            "filename": data["filename"],
            "context": [],
            "messages": [data["note"]],  # Initial message for processing
            "parameters": parameters,  # Pass parameters if needed by workflow
        }
        all_input_states.append(input_state)

    # Process all notes at once - the processor will handle concurrency with its semaphore
    try:
        # For OpenAI agent-based graph types, use the optimized batch processor
        if graph_type == "openai_agent":
            logger.info("Using optimized OpenAI Agent SDK batch processing")
            # Check if max_concurrent_runs is specified in parameters
            max_concurrent_runs = None

            # Look for concurrency parameters with different possible key names
            if parameters:
                for key in ["max_concurrent_runs", "max_concurrency", "concurrency"]:
                    if key in parameters:
                        max_concurrent_runs = parameters[key]
                        logger.info(
                            f"Using {key}={max_concurrent_runs} from parameters"
                        )
                        break

            # If not found in parameters, use a reasonable default
            if max_concurrent_runs is None:
                # Check environment settings
                if environment and "max_concurrent_runs" in environment:
                    max_concurrent_runs = environment["max_concurrent_runs"]
                    logger.info(
                        f"Using max_concurrent_runs={max_concurrent_runs} from environment settings"
                    )
                else:
                    # Default value
                    max_concurrent_runs = 3
                    logger.info(
                        f"Using default max_concurrent_runs={max_concurrent_runs}"
                    )

            logger.info(
                f"Using max_concurrent_runs={max_concurrent_runs} for batch processing"
            )

            # For large batches, split into smaller batches to avoid memory issues
            batch_size = parameters.get("batch_size", 5) if parameters else 5
            if len(all_input_states) <= batch_size:
                # Small enough batch, process all at once
                all_results_raw = graph.process_batch(all_input_states)

                # Log the token usage for debugging
                logger.info(
                    "Processing complete, examining results for token usage data"
                )
                total_tokens = 0
                for idx, result in enumerate(all_results_raw):
                    if "token_usage" in result and result["token_usage"]:
                        tokens = result["token_usage"].get("total_tokens", 0)
                        total_tokens += tokens
                        logger.info(f"Result {idx}: found {tokens} tokens")
                    else:
                        logger.warning(f"Result {idx}: no token usage data found")

                logger.info(f"Total tokens found in all results: {total_tokens}")
            else:
                # Large batch, split into smaller batches
                all_results_raw = []
                total_tokens = 0
                for i in range(0, len(all_input_states), batch_size):
                    batch = all_input_states[i : i + batch_size]
                    logger.info(
                        f"Processing batch {i//batch_size + 1}/{(len(all_input_states)-1)//batch_size + 1} with {len(batch)} notes"
                    )
                    batch_results = graph.process_batch(batch)

                    # Log and track token usage for this batch
                    batch_tokens = 0
                    for idx, result in enumerate(batch_results):
                        if "token_usage" in result and result["token_usage"]:
                            tokens = result["token_usage"].get("total_tokens", 0)
                            batch_tokens += tokens
                            logger.info(f"Batch result {idx}: found {tokens} tokens")

                    logger.info(
                        f"Batch {i//batch_size + 1} total tokens: {batch_tokens}"
                    )
                    total_tokens += batch_tokens
                    all_results_raw.extend(batch_results)

                logger.info(f"All batches total tokens: {total_tokens}")
        # Check if other graph types have a process_batch method
        elif hasattr(graph, "process_batch") and callable(
            getattr(graph, "process_batch")
        ):
            logger.info("Using graph.process_batch for all notes")
            all_results_raw = graph.process_batch(all_input_states)

            # Log the token usage for this graph type too
            total_tokens = 0
            for idx, result in enumerate(all_results_raw):
                if "token_usage" in result and result["token_usage"]:
                    tokens = result["token_usage"].get("total_tokens", 0)
                    total_tokens += tokens
                    logger.info(f"Result {idx}: found {tokens} tokens")

            logger.info(f"Total tokens from graph.process_batch: {total_tokens}")
        else:
            # Fallback to sequential processing if no batch method is available
            logger.warning(
                f"Graph type {graph_type} does not have process_batch, processing sequentially."
            )
            all_results_raw = []
            for i, state in enumerate(all_input_states):
                result = graph(state)
                all_results_raw.append(result)
                # Log token usage for sequential processing too
                if "token_usage" in result and result["token_usage"]:
                    tokens = result["token_usage"].get("total_tokens", 0)
                    logger.info(f"Sequential result {i}: found {tokens} tokens")
    except Exception as e:
        logger.error(f"Error processing notes: {e}", exc_info=True)
        # Create error results
        all_results_raw = [
            {
                "messages": [],
                "final_output": {},
                "error": f"Processing error: {str(e)}",
            }
        ] * len(patient_data)

    # Check for size mismatch
    if len(all_results_raw) != len(patient_data):
        logger.error(
            f"Mismatch in results size. Expected {len(patient_data)}, got {len(all_results_raw)}"
        )
        all_results_raw = [
            {
                "messages": [],
                "final_output": {},
                "error": "Result size mismatch",
            }
        ] * len(patient_data)

    # Combine results with original data
    results = []
    for idx, raw_result in enumerate(all_results_raw):
        original_data = patient_data[idx]
        run_id = str(uuid.uuid4())
        run_name = f"{graph_type}_{original_data['patient_id']}_{original_data['timepoint']}_{model_config}_variant_{prompt_variant}_{run_id}"

        # Extract relevant parts from raw_result, handling potential errors
        prediction = raw_result.get("final_output", {})
        converted_messages = raw_result.get("messages", [])
        error = raw_result.get("error")
        
        # Extract the extracted note if available
        extracted_note = raw_result.get("extracted_note", None)

        # Extract token usage if available
        token_usage = raw_result.get("token_usage")
        if token_usage is None:
            token_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "no_data": True,
            }

        # Get processing time if available
        processing_time = raw_result.get("processing_time", 0.0)

        # Log if there was an error in the result
        if error:
            logger.warning(
                f"Error in result for patient {original_data['patient_id']} (file: {original_data['filename']}): {error}"
            )
        
        # Log extraction info
        if extracted_note is not None:
            original_length = len(original_data["note"])
            extracted_length = len(extracted_note)
            reduction_percent = ((original_length - extracted_length) / original_length * 100) if original_length > 0 else 0
            logger.debug(
                f"Note extraction for patient {original_data['patient_id']}: "
                f"Original={original_length} chars, Extracted={extracted_length} chars, "
                f"Reduction={reduction_percent:.1f}%"
            )

        results.append(
            {
                "patient_id": original_data["patient_id"],
                "timepoint": original_data["timepoint"],
                "note": extracted_note if extracted_note and extracted_note.strip() else original_data["note"],  # Use extracted note if available and not empty
                "raw_note": original_data["note"],  # Always keep the original note
                "extracted_note": extracted_note,  # Also save extracted note separately for comparison
                "true_labels": original_data["labels"],
                "predicted_labels": prediction,
                "messages": converted_messages,
                "file_id": original_data["file_id"],
                "filename": original_data["filename"],
                "run_name": run_name,
                "run_id": run_id,
                "model_config": model_config,
                "model": token_usage.get("model", model_config),
                "error": error,
                "token_usage": token_usage,
                "processing_time": processing_time,
            }
        )

    # After the combined results are produced, calculate costs for each individual prediction
    def update_token_costs_in_results(
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Update token costs in results if they're not already present."""
        for result in results:
            if not isinstance(result, dict) or "token_usage" not in result:
                continue

            token_usage = result["token_usage"]
            if not isinstance(token_usage, dict):
                continue

            # Check if model is already in the token_usage, otherwise use result's model field or fall back to default
            if "model" not in token_usage:
                token_usage["model"] = result.get(
                    "model", result.get("model_config", "default")
                )

            model_name = token_usage.get(
                "model", result.get("model", result.get("model_config", "default"))
            )
            cost_rates = get_model_cost_rates(model_name)

            # Store cost rates in token usage for future reference
            token_usage["cost_rates"] = cost_rates

            # Calculate costs if not present
            if "prompt_cost" not in token_usage:
                prompt_tokens = token_usage.get("prompt_tokens", 0)
                completion_tokens = token_usage.get("completion_tokens", 0)

                token_usage["prompt_cost"] = (prompt_tokens / 1000000) * cost_rates[
                    "prompt"
                ]
                token_usage["completion_cost"] = (
                    completion_tokens / 1000000
                ) * cost_rates["completion"]
                token_usage["total_cost"] = (
                    token_usage["prompt_cost"] + token_usage["completion_cost"]
                )

        return results

    results = update_token_costs_in_results(results)

    # --- End overall time tracking ---
    total_run_time = time.perf_counter() - start_time
    avg_time_per_patient = total_run_time / total_patients

    # Store the results to a JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_filepath = os.path.join(output_dir, f"predictions_{timestamp}.json")
    with open(predictions_filepath, "w") as f:
        json.dump(results, f, indent=4)

    # log the total number of predictions
    logger.info(f"Total number of predictions: {len(results)}")
    logger.info(f"Saved predictions to {predictions_filepath}")
    
    # Log extraction statistics
    extraction_count = sum(1 for r in results if r.get("extracted_note") and r["extracted_note"].strip())
    if extraction_count > 0:
        logger.info(f"Note extraction performed on {extraction_count}/{len(results)} notes ({extraction_count/len(results)*100:.1f}%)")
        
        # Calculate average reduction
        total_reduction = 0
        for r in results:
            if r.get("extracted_note") and r["extracted_note"].strip():
                original_len = len(r["raw_note"])
                extracted_len = len(r["extracted_note"])
                if original_len > 0:
                    total_reduction += (original_len - extracted_len) / original_len * 100
        avg_reduction = total_reduction / extraction_count if extraction_count > 0 else 0
        logger.info(f"Average note length reduction: {avg_reduction:.1f}%")

    # Aggregate token usage metrics across all predictions
    def aggregate_token_usage_from_results(
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate token usage from all results, properly handling multi-model setups."""
        total_stats: Dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_cost": 0.0,
            "completion_cost": 0.0,
            "total_cost": 0.0,
            "processing_time": 0.0,
            "model_stats": {},
            "models_used": set(),  # Track all unique models used
            "avg_total_tokens": 0.0,
            "avg_prompt_tokens": 0.0,
            "avg_completion_tokens": 0.0,
        }
        records_with_usage = 0

        for result in results:
            if not isinstance(result, dict) or "token_usage" not in result:
                continue

            token_usage = result["token_usage"]
            if not isinstance(token_usage, dict):
                continue

            records_with_usage += 1

            # First check token_usage for model, then the result, then use model_config
            model_name = token_usage.get(
                "model", result.get("model", result.get("model_config", "default"))
            )
            total_stats["models_used"].add(model_name)

            # Update total stats
            total_stats["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
            total_stats["completion_tokens"] += token_usage.get("completion_tokens", 0)
            total_stats["total_tokens"] += token_usage.get("total_tokens", 0)
            total_stats["prompt_cost"] += token_usage.get("prompt_cost", 0.0)
            total_stats["completion_cost"] += token_usage.get("completion_cost", 0.0)
            total_stats["total_cost"] += token_usage.get("total_cost", 0.0)
            total_stats["processing_time"] += result.get("processing_time", 0.0)

            # Update per-model stats
            if model_name not in total_stats["model_stats"]:
                # Get cost rates either from token_usage or directly from config
                cost_rates = token_usage.get(
                    "cost_rates", get_model_cost_rates(model_name)
                )

                total_stats["model_stats"][model_name] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "prompt_cost": 0.0,
                    "completion_cost": 0.0,
                    "total_cost": 0.0,
                    "cost_rates": cost_rates,
                    "count": 0,
                }

            model_stats = total_stats["model_stats"][model_name]
            model_stats["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
            model_stats["completion_tokens"] += token_usage.get("completion_tokens", 0)
            model_stats["total_tokens"] += token_usage.get("total_tokens", 0)
            model_stats["prompt_cost"] += token_usage.get("prompt_cost", 0.0)
            model_stats["completion_cost"] += token_usage.get("completion_cost", 0.0)
            model_stats["total_cost"] += token_usage.get("total_cost", 0.0)
            model_stats["count"] += 1

        # Calculate averages
        total_stats["records_with_usage"] = records_with_usage
        if records_with_usage > 0:
            total_stats["avg_total_tokens"] = (
                float(total_stats["total_tokens"]) / records_with_usage
            )
            total_stats["avg_prompt_tokens"] = (
                float(total_stats["prompt_tokens"]) / records_with_usage
            )
            total_stats["avg_completion_tokens"] = (
                float(total_stats["completion_tokens"]) / records_with_usage
            )

        # Convert models_used set to list for JSON serialization
        total_stats["models_used"] = sorted(list(total_stats["models_used"]))

        return total_stats

    token_usage_data = aggregate_token_usage_from_results(results)

    # Write token usage to file
    token_usage_output_path = os.path.join(output_dir, f"token_usage_{timestamp}.json")
    with open(token_usage_output_path, "w") as f:
        json.dump(token_usage_data, f, indent=2)

    logger.info(f"Wrote token usage data to {token_usage_output_path}")
    logger.info(f"Total tokens used: {token_usage_data['total_tokens']:,}")
    logger.info(f"Total cost: ${token_usage_data['total_cost']:.4f}")
    logger.info("Per-model breakdown:")
    for model_name, stats in token_usage_data["model_stats"].items():
        logger.info(f"  {model_name}:")
        logger.info(f"    Tokens: {stats['total_tokens']:,}")
        logger.info(f"    Cost: ${stats['total_cost']:.4f}")
    logger.info(f"Total processing time: {token_usage_data['processing_time']:.2f}s")

    # After saving predictions and usage info, run gold error evaluation if configured
    if environment and environment.get("run_gold_error_eval", False):
        logger.info("Running evaluation with gold error analysis...")
        try:
            # Get Azure settings from environment
            use_azure = environment.get("use_azure", False)
            azure_endpoint = environment.get("azure_endpoint")
            deployment_name = environment.get("deployment_name")

            # Build command for evaluation with gold error analysis
            cmd = [
                "python",
                "scripts/evaluate_run_output.py",
                "--input",
                predictions_filepath,
                "--output_dir",
                os.path.join(output_dir, "evaluation"),
                "--run-gold-error-eval",
                "--data-dir",
                data_dir,
                "--graph_type",
                graph_type,
                "--model_config",
                model_config,
                "--prompt_variant",
                prompt_variant,
                "--sweep_name",
                sweep_name,
            ]

            # Add Azure-specific arguments if configured
            if use_azure:
                cmd.append("--use-azure")
                if azure_endpoint:
                    cmd.extend(["--azure-endpoint", azure_endpoint])
                if deployment_name:
                    cmd.extend(["--deployment-name", deployment_name])

            # Add logging level if available - now supports proper --log-level parameter
            if (
                environment
                and "logging" in environment
                and "level" in environment["logging"]
            ):
                log_level = environment["logging"]["level"]
                if log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                    cmd.extend(["--log-level", log_level])

            logger.info(f"Running evaluation with command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            logger.info("Evaluation with gold error analysis completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running evaluation: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during evaluation: {e}")

    print(f"\nExperiment for {graph_type} ({model_config}) completed successfully!")
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenAI agent workflow.")
    parser.add_argument(
        "graph_type",
        help="Type of graph to run (supported types include openai_agent, openai_zeroshot, regex, and dataset variants)",
    )
    parser.add_argument(
        "--data_dir",
        default="data/synthetic_notes/",
        help="Directory containing patient data",
    )
    parser.add_argument(
        "--output_base_dir",
        default="outputs/synthetic",
        help="Base directory to store outputs",
    )
    parser.add_argument(
        "--model_config",
        default="default",
        choices=[
            "default",
            "4.1-mini",
            "4.1-nano",
            "o4-mini",
            "ollama-deepseek-1b",
            "ollama-llama-1b",
            "ollama-qwen3-14b",
            "vllm-deepseek-r1-8b",
            "vllm-qwen3-8b",
            "vllm-qwen3-8b-local",
            "vllm-qwen3-14b",
            "vllm-qwen3-32b",
            "vllm-gemma3-4b",
            "vllm-gemma3-12b",
            "vllm-gemma3-27b",
            "vllm-medgemma-27b",
        ],
        help="Model configuration for the workflow",
    )
    parser.add_argument(
        "--prompt_variant",
        help="Specify which prompt variant to use for the graph",
        default="default",
        choices=[
            "default",
            "ablation_single",
            "ablation_no_judge",
        ],
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (first 2 notes only)",
    )
    parser.add_argument(
        "--parameters",
        help="Additional parameters as comma-separated key=value pairs (e.g., batch_size=5,configure_azure=true)",
        default="",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "azure", "vllm", "ollama"],
        default="auto",
        help="Force provider routing. Use 'auto' to infer from model config.",
    )
    parser.add_argument(
        "--vllm-endpoint",
        help="vLLM OpenAI-compatible base URL (e.g., http://localhost:8001/v1)",
    )
    parser.add_argument(
        "--ollama-endpoint",
        help="Ollama OpenAI-compatible base URL (e.g., http://localhost:11434/v1)",
    )
    # Add arguments for gold error evaluation
    parser.add_argument(
        "--run-gold-error-eval",
        action="store_true",
        help="Run gold error evaluation after experiment",
    )
    parser.add_argument(
        "--evaluate-most-recent",
        action="store_true",
        help="Evaluate most recent results in output directory instead of running a new experiment",
    )
    parser.add_argument(
        "--eval-output-dir",
        help="Specific output directory to evaluate most recent results from",
    )
    parser.add_argument(
        "--use-azure",
        action="store_true",
        help="Use Azure OpenAI for gold error evaluation",
    )
    parser.add_argument(
        "--azure-endpoint",
        help="Azure OpenAI endpoint for gold error evaluation",
    )
    parser.add_argument(
        "--deployment-name",
        help="Azure OpenAI deployment name for gold error evaluation",
    )
    # Add patient-level evaluation argument
    parser.add_argument(
        "--patient-level-eval",
        action="store_true",
        help="Perform patient-level aggregation before evaluation",
    )
    # Add logging level argument
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Set logging level (overrides configuration file)",
    )

    args = parser.parse_args()

    # Set log level from command line if provided (overrides configuration)
    if args.log_level:
        log_level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        log_level = log_level_map.get(args.log_level.upper(), logging.INFO)
        logging.basicConfig(level=log_level, force=True)
        logger.setLevel(log_level)
        logger.info(f"Log level set to {args.log_level} from command line")

    # Handle evaluation-only mode if requested
    if args.evaluate_most_recent:
        # Determine the directory to evaluate
        eval_dir = args.eval_output_dir
        if not eval_dir:
            # Build default output directory based on provided arguments
            graph_type_folder = "openai_agent"  # Default to openai_agent
            reasoning_type_folder = "base"  # Default to base

            # Use sweep_name from parameters if available, otherwise default
            sweep_name = "default_sweep"
            if args.parameters:
                for param in args.parameters.split(","):
                    if param.startswith("sweep_name="):
                        sweep_name = param.split("=")[1]

            eval_dir = os.path.join(
                args.output_base_dir,
                sweep_name,
                graph_type_folder,
                reasoning_type_folder,
                args.model_config,
                f"variant_{args.prompt_variant}",
            )
        else:
            # Try to extract sweep_name from the directory path if provided
            # Pattern: data/rwd/ray_dev_outputs/dev_ray_agent_sweep/ray_agent/default/variant_default
            # Extract the sweep name (dev_ray_agent_sweep) from the path
            path_parts = Path(eval_dir).parts
            sweep_name = "default_sweep"
            # Look for a part that ends with "_sweep"
            for part in path_parts:
                if part.endswith("_sweep"):
                    sweep_name = part
                    break
            logger.info(f"Extracted sweep name from path: {sweep_name}")

        # Check if directory exists
        if not os.path.exists(eval_dir):
            logger.error(f"Evaluation directory does not exist: {eval_dir}")
            sys.exit(1)

        logger.info(f"Running evaluation on most recent file in {eval_dir}")

        # Find the most recent predictions file
        prediction_files = [
            f
            for f in os.listdir(eval_dir)
            if f.startswith("predictions_") and f.endswith(".json")
        ]
        if not prediction_files:
            logger.error(f"No prediction files found in {eval_dir}")
            sys.exit(1)

        prediction_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(eval_dir, f)), reverse=True
        )
        input_file = os.path.join(eval_dir, prediction_files[0])
        logger.info(f"Found most recent prediction file: {input_file}")

        # If patient-level evaluation is requested, create patient-level aggregation first
        if args.patient_level_eval:
            logger.info("Creating patient-level aggregation before evaluation")
            os.makedirs(os.path.join(eval_dir, "patient_level"), exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            patient_level_file = os.path.join(
                eval_dir, "patient_level", f"patient_level_predictions_{timestamp}.json"
            )

            # Run the patient-level aggregation script
            try:
                cmd = [
                    "python",
                    "scripts/create_patient_level_aggregation.py",
                    "--input",
                    input_file,
                    "--output",
                    patient_level_file,
                ]
                logger.info(
                    f"Running patient-level aggregation with command: {' '.join(cmd)}"
                )
                subprocess.run(cmd, check=True)
                logger.info(
                    f"Patient-level aggregation created at {patient_level_file}"
                )

                # Use the patient-level file for evaluation
                input_file = patient_level_file

            except subprocess.CalledProcessError as e:
                logger.error(f"Error running patient-level aggregation: {e}")
                sys.exit(1)

        # Build command for evaluation
        cmd = [
            "python",
            "scripts/evaluate_run_output.py",
            "--input",
            input_file,  # Use either the original or patient-level file
            "--output_dir",
            os.path.join(eval_dir, "evaluation"),
            "--graph_type",
            args.graph_type,
            "--model_config",
            args.model_config,
            "--prompt_variant",
            args.prompt_variant,
            "--data-dir",
            args.data_dir,
            "--sweep_name",
            sweep_name,  # Add sweep name to evaluation
        ]

        # Add gold error evaluation if requested
        if args.run_gold_error_eval:
            cmd.append("--run-gold-error-eval")

        # Add Azure-specific arguments if configured
        if args.use_azure:
            cmd.append("--use-azure")
            if args.azure_endpoint:
                cmd.extend(["--azure-endpoint", args.azure_endpoint])
            if args.deployment_name:
                cmd.extend(["--deployment-name", args.deployment_name])

        # Add logging level if provided
        if args.log_level:
            if args.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                cmd.extend(["--log-level", args.log_level])

        logger.info(f"Running evaluation with command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            logger.info("Evaluation completed successfully")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            sys.exit(1)

    # Parse any parameters from the --parameters argument
    parameters = {}
    if args.parameters:
        # First try to parse as JSON (from run_sweep.py)
        try:
            import json

            parameters = json.loads(args.parameters)
        except (json.JSONDecodeError, ValueError):
            # Fall back to comma-separated format for backward compatibility
            for param in args.parameters.split(","):
                if "=" in param:
                    key, value = param.split("=", 1)
                    # Convert value types if possible
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    else:
                        try:
                            value = int(value)
                        except ValueError:
                            try:
                                value = float(value)
                            except ValueError:
                                pass  # Keep as string
                    parameters[key] = value

    # Explicit provider/runtime arguments override inferred defaults.
    if args.provider and args.provider != "auto":
        parameters["provider"] = args.provider
    if args.vllm_endpoint:
        parameters["vllm_endpoint"] = args.vllm_endpoint
    if args.ollama_endpoint:
        parameters["ollama_endpoint"] = args.ollama_endpoint

    # Create environment dictionary with gold error evaluation settings
    environment = {
        "run_gold_error_eval": args.run_gold_error_eval,
        "use_azure": args.use_azure,
        "azure_endpoint": args.azure_endpoint,
        "deployment_name": args.deployment_name,
    }

    # Add logging configuration if log level is provided
    if args.log_level:
        environment["logging"] = {"level": args.log_level}

    run_experiment(
        args.graph_type,
        args.data_dir,
        args.output_base_dir,
        model_config=args.model_config,
        prompt_variant=args.prompt_variant,
        debug=args.debug,
        environment=environment,
        parameters=parameters,
    )
