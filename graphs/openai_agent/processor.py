"""
OpenAI Agent Processor for clinical note analysis
Refactored for direct OpenAI Agent SDK integration
"""

import asyncio
import logging
import os
import json
import sys
import platform
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
import uuid

# Import the models from parallel agents
from .parallel_agents.models.input_models import NoteInput, TokenUsageMetadata
from .parallel_agents.models.output_models import NotePrediction
from .parallel_agents.models.enhanced_output_models import EventResult

# Import the parallel agent workflow
from .parallel_agents.workflow.note_processor import (
    process_note_with_judge,
    process_all_notes_with_judge,
)
from .parallel_agents.utils.model_config import get_available_configs, load_model_config
from .parallel_agents.utils.client_factory import create_azure_client
from .parallel_agents.utils.model_provider import create_azure_provider

# For OpenAI client setup
from agents import (
    set_default_openai_key,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
)

# Import our TokenTracker
from .models import TokenTracker

# Configure thread-specific event loops
# This ensures each thread created by ThreadPoolExecutor gets its own event loop
if platform.system() == "Windows":
    # On Windows, ensure we use a selector event loop policy
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
else:
    # On Unix systems, use the default policy but ensure it creates new event loops for each thread
    # This ensures thread safety when multiple threads are used with asyncio
    class ThreadSafeEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
        """Event loop policy that creates a new event loop for each thread."""

        def __init__(self):
            super().__init__()
            self._loop_map = {}

        def get_event_loop(self):
            """Get the event loop for the current thread, creating one if it doesn't exist."""
            thread_id = threading.get_ident()
            if thread_id not in self._loop_map or self._loop_map[thread_id].is_closed():
                self._loop_map[thread_id] = self.new_event_loop()
            return self._loop_map[thread_id]

        def set_event_loop(self, loop):
            """Set the event loop for the current thread."""
            thread_id = threading.get_ident()
            self._loop_map[thread_id] = loop

    # Only set this policy if we're running in a multi-threaded environment
    # like ThreadPoolExecutor, which is detected by checking the thread module
    import threading

    if (
        hasattr(threading, "main_thread")
        and threading.current_thread() is threading.main_thread()
    ):
        asyncio.set_event_loop_policy(ThreadSafeEventLoopPolicy())

logger = logging.getLogger(__name__)


# Disable tracing at module level
set_tracing_disabled(True)


class LLMProcessor:
    """
    Process clinical notes using LLM-based agents.
    """

    def __init__(
        self,
        model_config_key: str = "default",
        prompt_variant: str = "default",
        max_concurrent_runs: int = 8,
        logging_config: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the LLM processor.

        Args:
            model_config_key: Configuration key for model settings
            prompt_variant: Which prompt variant to use
            max_concurrent_runs: Maximum number of concurrent processing runs (default 8)
            logging_config: Optional logging configuration for controlling log levels
            parameters: Optional parameters for customizing processor behavior
        """
        # Configure logging if provided
        if logging_config:
            self._configure_logging(logging_config)

        # Initialize main components
        self.model_config_key = model_config_key
        self.prompt_variant = prompt_variant
        self.max_concurrent_runs = max_concurrent_runs
        self.semaphore = asyncio.Semaphore(max_concurrent_runs)
        self.parameters = parameters or {}

        # Initialize token tracker
        self.token_tracker = TokenTracker()

        # Load model config
        self.model_config = load_model_config(model_config_key)
        logger.info("Using model config: %s", model_config_key)

        # Determine if we are using an Ollama or vLLM model
        from .parallel_agents.utils.model_config import (
            is_ollama_model,
            is_vllm_model,
            get_model_for_role,
        )

        # Get the actual model name from the loaded configuration
        extractor_role = "extractor"
        if extractor_role in self.model_config:
            extractor_model_name = self.model_config[extractor_role]
            logger.info(
                "Using model %s for extractor role from loaded config",
                extractor_model_name,
            )
        else:
            # Fallback to get_model_for_role if not found directly in config
            extractor_model_name = get_model_for_role(model_config_key, extractor_role)
            logger.info(
                "Using model %s for extractor role from get_model_for_role",
                extractor_model_name,
            )

        # Check what type of model this is (or use explicit provider override)
        provider_override = str(self.parameters.get("provider", "auto")).lower()
        if provider_override not in {"auto", "azure", "vllm", "ollama"}:
            raise ValueError(
                f"Invalid provider override '{provider_override}'. Use one of: auto, azure, vllm, ollama."
            )

        if provider_override == "vllm":
            self.is_vllm = True
            self.is_ollama = False
        elif provider_override == "ollama":
            self.is_vllm = False
            self.is_ollama = True
        elif provider_override == "azure":
            self.is_vllm = False
            self.is_ollama = False
        else:
            self.is_ollama = is_ollama_model(extractor_model_name)
            self.is_vllm = is_vllm_model(extractor_model_name)
        
        # Determine provider type
        if self.is_vllm:
            provider_type = "vLLM"
        elif self.is_ollama:
            provider_type = "Ollama"
        else:
            provider_type = "Azure"
            
        logger.info(
            "Model provider routing: %s for model %s",
            provider_type,
            extractor_model_name,
        )

        # Provider configuration based on model type
        self.azure_provider = None
        self.azure_client = None
        self.azure_endpoint = None
        self.azure_api_version = None
        self.ollama_provider = None
        self.vllm_provider = None

        # Configure the appropriate service
        if self.is_vllm:
            logger.info(
                "Initializing vLLM provider for model %s", extractor_model_name
            )
            self.configure_vllm()
            # Explicitly prevent other providers
            self.azure_provider = None
            self.azure_client = None
            self.ollama_provider = None
        elif self.is_ollama:
            logger.info(
                "Initializing Ollama provider for model %s", extractor_model_name
            )
            self.configure_ollama()
            # Explicitly prevent other providers
            self.azure_provider = None
            self.azure_client = None
            self.vllm_provider = None
        else:
            logger.info(
                "Initializing Azure provider for model %s", extractor_model_name
            )
            self.configure_azure()
            # Explicitly prevent other providers
            self.ollama_provider = None
            self.vllm_provider = None

        # Initialize event processors
        self._init_event_processor_components()

    def _init_event_processor_components(self):
        """Initialize components for event processing"""
        # Max iterations for event processing
        self.max_iterations = 1  # Default to single iteration

        default_event_types = [
            "Pneumonitis",
            "Myocarditis",
            "Colitis",
            "Thyroiditis",
            "Hepatitis",
            "Dermatitis",
        ]

        self.event_types = self.parameters.get("event_types", default_event_types)
        logger.info("Configured event types: %s", self.event_types)

        # Agents and data for event processing
        # These will be populated when needed
        self.event_agents = {}
        self.ctcae_data = {}

    def configure_azure(
        self,
        azure_endpoint: Optional[str] = None,
        azure_api_version: Optional[str] = None,
    ):
        """
        Configure Azure authentication for the OpenAI Agent SDK

        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            azure_api_version: Azure OpenAI API version
        """
        # Get Azure configuration from explicit args, runtime parameters, or environment
        param_endpoint = (self.parameters or {}).get("azure_endpoint")
        param_api_version = (
            (self.parameters or {}).get("azure_api_version")
            or (self.parameters or {}).get("api_version")
        )

        self.azure_endpoint = (
            azure_endpoint
            or param_endpoint
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("ENDPOINT_URL")
        )
        self.azure_api_version = (
            azure_api_version
            or param_api_version
            or os.getenv("AZURE_OPENAI_API_VERSION")
            or os.getenv("API_VERSION")
            or "2024-12-01-preview"
        )

        if not self.azure_endpoint or not self.azure_api_version:
            logger.warning("Azure endpoint or API version not provided")
            # Try to use OpenAI API key instead
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                set_default_openai_key(api_key)
                return
            else:
                logger.warning("OPENAI_API_KEY not set. SDK may not work properly.")
                return

        # Set up Azure provider
        try:
            # Ensure downstream model factories see resolved Azure config
            self.parameters = self.parameters or {}
            provider_cfg = self.parameters.get("provider_config", {})
            provider_cfg["azure_endpoint"] = self.azure_endpoint
            provider_cfg["azure_api_version"] = self.azure_api_version
            self.parameters["provider_config"] = provider_cfg

            # Create the Azure provider for model authentication
            self.azure_provider = create_azure_provider(
                azure_endpoint=self.azure_endpoint,
                azure_api_version=self.azure_api_version,
            )

            # Create Azure client for the OpenAI SDK
            self.azure_client = create_azure_client(
                azure_endpoint=self.azure_endpoint,
                azure_api_version=self.azure_api_version,
            )

            # Configure the OpenAI Agent SDK to use this client
            set_default_openai_client(client=self.azure_client, use_for_tracing=False)
            set_default_openai_api("chat_completions")
            logger.info("Azure OpenAI client configured successfully")
        except Exception as e:
            logger.error("Failed to configure Azure: %s", str(e))
            # Try to use OpenAI API key as fallback
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                set_default_openai_key(api_key)
                logger.info("Configured OpenAI API key as fallback")

    async def _run_with_semaphore(self, coro):
        """
        Run a coroutine with a semaphore to limit concurrency.

        Args:
            coro: The coroutine to run

        Returns:
            The result of the coroutine
        """
        async with self.semaphore:
            return await coro

    def configure_ollama(self):
        """
        Configure Ollama for local LLM inference with the OpenAI Agent SDK
        """
        # Import the Ollama provider
        from .parallel_agents.utils.ollama_provider import get_ollama_provider

        try:
            # Get the Ollama endpoint from arguments/parameters or environment
            ollama_endpoint = None
            if self.parameters and "ollama_endpoint" in self.parameters:
                ollama_endpoint = self.parameters.get("ollama_endpoint")
                logger.info(
                    "Using Ollama endpoint from parameters: %s", ollama_endpoint
                )
            elif os.getenv("OLLAMA_ENDPOINT"):
                ollama_endpoint = os.getenv("OLLAMA_ENDPOINT")
                logger.info(
                    "Using Ollama endpoint from environment: %s", ollama_endpoint
                )
            else:
                raise ValueError(
                    "Ollama provider selected but no endpoint configured. Pass --ollama-endpoint or set OLLAMA_ENDPOINT."
                )

            # Get the Ollama provider
            self.ollama_provider = get_ollama_provider(ollama_endpoint)

            # Make sure we're not using Azure for Ollama models
            self.azure_provider = None
            self.azure_client = None
            self.azure_endpoint = None
            self.azure_api_version = None

            # We don't set a default client here because Ollama models will use
            # their own clients created by the provider when requested
            logger.info("Ollama provider configured successfully")
        except Exception as e:
            logger.error("Failed to configure Ollama: %s", str(e))
            raise RuntimeError(f"Ollama configuration failed: {e}") from e

    def configure_vllm(self):
        """
        Configure vLLM for local LLM inference with OpenAI-compatible API
        """
        # Import the vLLM provider
        from .parallel_agents.utils.vllm_provider import VLLMModelProvider

        try:
            # Get the vLLM endpoint from arguments/parameters or environment
            vllm_endpoint = None
            if self.parameters and "vllm_endpoint" in self.parameters:
                vllm_endpoint = self.parameters.get("vllm_endpoint")
                logger.info(
                    "Using vLLM endpoint from parameters: %s", vllm_endpoint
                )
            elif os.getenv("VLLM_BASE_URL"):
                vllm_endpoint = os.getenv("VLLM_BASE_URL")
                logger.info(
                    "Using vLLM endpoint from environment: %s", vllm_endpoint
                )
            else:
                raise ValueError(
                    "vLLM provider selected but no endpoint configured. Pass --vllm-endpoint or set VLLM_BASE_URL."
                )

            # Initialize the vLLM provider
            self.vllm_provider = VLLMModelProvider(base_url=vllm_endpoint)

            # Make sure we're not using Azure or Ollama for vLLM models
            self.azure_provider = None
            self.azure_client = None
            self.azure_endpoint = None
            self.azure_api_version = None
            self.ollama_provider = None

            logger.info("vLLM provider configured successfully")
        except Exception as e:
            logger.error("Failed to configure vLLM: %s", str(e))
            raise RuntimeError(f"vLLM configuration failed: {e}") from e

    async def process_single_note(self, note_input: NoteInput) -> NotePrediction:
        """
        Process a single note with the processor.

        Args:
            note_input: The note input to process

        Returns:
            The prediction result
        """
        processing_start = time.time()

        try:
            from .parallel_agents.workflow.note_processor import process_note_with_judge

            # Use the appropriate provider based on model type
            if self.is_vllm:
                provider = self.vllm_provider
                client = None  # vLLM provider manages its own client
                # Ensure provider_config is set for vLLM
                if 'provider_config' not in self.parameters:
                    self.parameters['provider_config'] = {}
                self.parameters['provider_config']['vllm_endpoint'] = self.vllm_provider.base_url
            elif self.is_ollama:
                provider = self.ollama_provider
                client = None  # Ollama provider manages its own client
            else:
                provider = self.azure_provider
                client = self.azure_client

            result = await process_note_with_judge(
                note_input,
                self.model_config_key,
                self.prompt_variant,
                provider,
                client,
                self.parameters,
            )

            # Calculate processing time for logging and attribution
            prediction = result
            prediction.processing_time = time.time() - processing_start

            # Add token tracking data to the prediction
            token_usage_summary = self.token_tracker.get_summary()
            
            # Get the primary model cost rates from utils/config.py
            from utils.config import get_model_cost_rates
            primary_model = self.model_config_key  # This is the model config like "default", "o3-mini", etc.
            cost_rates = get_model_cost_rates(primary_model)
            
            # Create clean token usage data
            prediction.token_usage = {
                "prompt_tokens": token_usage_summary["total"]["prompt_tokens"],
                "completion_tokens": token_usage_summary["total"]["completion_tokens"],
                "total_tokens": token_usage_summary["total"]["total_tokens"],
                "prompt_cost": token_usage_summary["total"]["prompt_cost"],
                "completion_cost": token_usage_summary["total"]["completion_cost"],
                "total_cost": token_usage_summary["total"]["total_cost"],
                "model_costs": token_usage_summary["by_model"],
                "model": primary_model,
                "cost_rates": cost_rates,
            }

            # Log completion statistics
            logger.info(
                f"Completed note {note_input.note_id} processing in {prediction.processing_time:.2f} seconds"
            )
            logger.info(
                f"Token usage: {prediction.token_usage['prompt_tokens']} prompt tokens, "
                f"{prediction.token_usage['completion_tokens']} completion tokens, "
                f"{prediction.token_usage['total_tokens']} total tokens"
            )

            # Token usage is logged above and included in the prediction object

            return prediction

        except Exception as e:
            logger.exception(f"Error in process_single_note: {str(e)}")
            # Create an error prediction
            return ""

    def process_single_note_sync(self, note: NoteInput) -> NotePrediction:
        """
        Process a single note synchronously (wrapper for the async method)

        Args:
            note: The clinical note to process

        Returns:
            The prediction result
        """
        try:
            # Get the current event loop or create a new one if none exists in this thread
            try:
                # Try to get the current event loop first
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # If there is no event loop in this thread, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                logger.debug("Created new event loop for this thread")

            # Check if the loop is running (in case we're in a nested call)
            if loop.is_running():
                logger.warning(
                    "Event loop is already running. Using create_task to avoid blocking."
                )
                # If loop is already running, use create_task and asyncio.Future to get the result
                future = asyncio.Future()

                async def run_process():
                    try:
                        result = await self.process_single_note(note)
                        future.set_result(result)
                    except Exception as e:
                        future.set_exception(e)

                loop.create_task(run_process())
                # Wait for the result with a timeout
                try:
                    result = loop.run_until_complete(
                        asyncio.wait_for(future, timeout=3000)
                    )
                except asyncio.TimeoutError:
                    logger.error("Single note processing timed out after 5 minutes")
                    raise TimeoutError(
                        "Single note processing timed out after 5 minutes"
                    )
            else:
                # The loop is not running, use run_until_complete
                result = loop.run_until_complete(self.process_single_note(note))

            return result
        except Exception as e:
            logger.exception(f"Error in synchronous note processing: {str(e)}")
            # Return error prediction
            return NotePrediction(
                PMRN=note.pmrn if note.pmrn else "UNKNOWN",
                NOTE_ID=note.note_id,
                NOTE_TEXT=note.note_text,
                SHORTENED_NOTE_TEXT="None",
                MESSAGES=[],
                PREDICTION={},
                PROCESSED_AT=datetime.now(),
                PROCESSING_TIME=0.0,
                MODEL_NAME=self.model_config_key,
                TOKEN_USAGE=None,
                ERROR=f"Synchronous processing error: {str(e)}",
            )

    async def process_all_notes(
        self,
        notes: List[NoteInput],
        openai_client=None,
    ) -> List[NotePrediction]:
        """
        Process multiple notes concurrently with controlled parallelism

        Args:
            notes: List of clinical notes to process
            openai_client: Optional OpenAI client for thread safety

        Returns:
            List of prediction results
        """
        total_notes = len(notes)
        processing_start = time.time()

        try:
            logger.info(
                f"Processing {total_notes} notes with {self.max_concurrent_runs} concurrent workers"
            )

            # Create a single client for this batch for thread safety, if needed
            if not openai_client and not self.is_ollama and not self.is_vllm:
                openai_client = self._create_batch_client()

            # Use the appropriate provider based on model type
            if self.is_vllm:
                provider = self.vllm_provider
                client = None  # vLLM provider manages its own client
                # Ensure provider_config is set for vLLM
                if 'provider_config' not in self.parameters:
                    self.parameters['provider_config'] = {}
                self.parameters['provider_config']['vllm_endpoint'] = self.vllm_provider.base_url
            elif self.is_ollama:
                provider = self.ollama_provider
                client = None  # Ollama provider manages its own client
            else:
                provider = self.azure_provider
                client = openai_client

            # Create tasks for each note, using semaphore to control concurrency
            tasks = []
            for note in notes:
                # Create a task for processing this note with the semaphore
                task = asyncio.create_task(
                    self._run_with_semaphore(
                        process_note_with_judge(
                            note,
                            self.model_config_key,
                            self.prompt_variant,
                            provider,
                            client,
                            self.parameters,
                        )
                    )
                )
                tasks.append(task)

            # Process and collect results in order of completion
            results = await asyncio.gather(*tasks)

            # Log total processing time
            total_time = time.time() - processing_start
            logger.info(f"Processed {len(results)} notes in {total_time:.2f} seconds")

            return results
        except Exception as e:
            logger.exception(f"Error in batch processing: {str(e)}")
            # Return error predictions
            return [
                NotePrediction(
                    PMRN=note.pmrn if note.pmrn else "UNKNOWN",
                    NOTE_ID=note.note_id,
                    NOTE_TEXT=note.note_text,
                    SHORTENED_NOTE_TEXT="None",
                    MESSAGES=[],
                    PREDICTION={},
                    PROCESSED_AT=datetime.now(),
                    PROCESSING_TIME=0.0,
                    MODEL_NAME=self.model_config_key,
                    TOKEN_USAGE=None,
                    ERROR=f"Batch processing error: {str(e)}",
                )
                for note in notes
            ]

    def process_batch(self, notes: List[NoteInput]) -> List[NotePrediction]:
        """
        Process a batch of notes synchronously (wrapper for the async method)

        Args:
            notes: List of clinical notes to process

        Returns:
            List of prediction results
        """
        try:
            # Get the current event loop or create a new one if none exists in this thread
            try:
                # Try to get the current event loop first
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # If there is no event loop in this thread, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                logger.debug("Created new event loop for this thread")

            # Create a batch-specific client only for Azure models
            batch_client = self._create_batch_client()

            # Check if the loop is running (in case we're in a nested call)
            if loop.is_running():
                logger.warning(
                    "Event loop is already running. Using create_task to avoid blocking."
                )
                # If loop is already running, use create_task and asyncio.Future to get the result
                future = asyncio.Future()

                async def run_process():
                    try:
                        result = await self.process_all_notes(
                            notes, openai_client=batch_client
                        )
                        future.set_result(result)
                    except Exception as e:
                        future.set_exception(e)

                loop.create_task(run_process())
                # Wait for the result with a timeout
                try:
                    results = loop.run_until_complete(
                        asyncio.wait_for(future, timeout=600)
                    )
                except asyncio.TimeoutError:
                    logger.error("Batch processing timed out after 10 minutes")
                    raise TimeoutError("Batch processing timed out after 10 minutes")
            else:
                # The loop is not running, use run_until_complete
                results = loop.run_until_complete(
                    self.process_all_notes(notes, openai_client=batch_client)
                )

            return results
        except Exception as e:
            logger.exception(f"Error in batch processing: {str(e)}")
            # Return error predictions
            return [
                NotePrediction(
                    PMRN=note.pmrn if note.pmrn else "UNKNOWN",
                    NOTE_ID=note.note_id,
                    NOTE_TEXT=note.note_text,
                    SHORTENED_NOTE_TEXT="None",
                    MESSAGES=[],
                    PREDICTION={},
                    PROCESSED_AT=datetime.now(),
                    PROCESSING_TIME=0.0,
                    MODEL_NAME=self.model_config_key,
                    TOKEN_USAGE=None,
                    ERROR=f"Batch processing error: {str(e)}",
                )
                for note in notes
            ]

    def _create_batch_client(self):
        """
        Create a new Azure client for batch processing thread safety.
        Returns None for Ollama or vLLM providers.
        """
        # Only create Azure client for Azure models
        if self.is_vllm or self.is_ollama:
            return None
            
        try:
            # Create Azure client for batch processing
            batch_client = create_azure_client(
                azure_endpoint=self.azure_endpoint,
                azure_api_version=self.azure_api_version,
            )
            logger.debug("Created new Azure client for batch processing")
            return batch_client
        except Exception as e:
            logger.error("Failed to create batch Azure client: %s", str(e))
            return None

    def _cleanup_resources(self, client=None):
        """Safely clean up resources like custom clients"""
        if not client:
            return

        try:
            # Close the client if it's different from the instance client
            if client != self.azure_client:
                # Get the current event loop
                try:
                    loop = asyncio.get_event_loop()
                    # Only try to close the client if we have a loop
                    if not loop.is_running():
                        loop.run_until_complete(client.aclose())
                    else:
                        logger.debug("Loop is running, skipping client closure")
                except RuntimeError:
                    logger.debug("No event loop available, skipping client closure")
        except Exception as e:
            logger.warning(f"Error during resource cleanup: {str(e)}")

    def set_concurrency_limit(self, max_concurrent_runs: int) -> None:
        """
        Update the concurrency limit at runtime.

        Args:
            max_concurrent_runs: New concurrency limit
        """
        if max_concurrent_runs < 1:
            logger.warning(f"Invalid concurrency limit {max_concurrent_runs}, using 1")
            max_concurrent_runs = 1

        self.semaphore = asyncio.Semaphore(max_concurrent_runs)
        logger.info(f"Updated concurrency limit to {max_concurrent_runs}")

    async def _process_all_events(self, note_text: str) -> List[EventResult]:
        """
        Process all event types in parallel.

        Args:
            note_text: The text of the note to process

        Returns:
            List of event results
        """
        # Create a task for each event type
        tasks = []
        for event_type in self.event_types:
            task = self._process_single_event(
                event_type,
                note_text,
                self.event_agents,
                self.ctcae_data,
                token_tracker=self.token_tracker,
            )
            tasks.append(task)

        # Run all event types in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, including any errors
        event_results = []
        for i, result in enumerate(results):
            event_type = self.event_types[i]
            if isinstance(result, Exception):
                logger.error(f"Error processing event {event_type}: {str(result)}")
                # Create an error result for this event
                event_results.append(
                    EventResult(
                        event_type=event_type,
                        grade=0,
                        attribution=0,
                        certainty=0,
                        reasoning=f"Error occurred: {str(result)}",
                    )
                )
            else:
                # Add the successful result
                event_results.append(result)

        return event_results

    async def _process_single_event(
        self,
        event_type: str,
        note_text: str,
        event_agents: Dict[str, Dict[str, Any]],
        ctcae_data: Dict[str, Any],
        token_tracker: Optional[TokenTracker] = None,
    ) -> EventResult:
        """
        Process a single event type with judge-enhanced workflow.

        Args:
            event_type: The type of event to process
            note_text: The text of the note
            event_agents: Dictionary of agents for each event type
            ctcae_data: CTCAE data for reference
            token_tracker: Optional token tracker for tracking token usage

        Returns:
            Event result with grade, attribution, and certainty
        """
        # Use the ProcessEvent function from parallel_agents module
        from .parallel_agents.workflow.event_processor import process_event_with_judge

        async with self.semaphore:
            try:
                # Generate a unique request ID for tracing
                request_id = f"{event_type}_{uuid.uuid4().hex[:8]}"

                # Process the event
                event_result = await process_event_with_judge(
                    event_type=event_type,
                    event_agents=event_agents,
                    note_text=note_text,
                    ctcae_data=ctcae_data,
                    max_iterations=self.max_iterations,
                    azure_provider=self.azure_provider,
                    token_tracker=token_tracker,
                    request_id=request_id,
                    prompt_variant=self.prompt_variant,
                )

                return event_result
            except Exception as e:
                logger.exception(
                    f"Error in _process_single_event for {event_type}: {str(e)}"
                )
                raise

    def _configure_logging(self, logging_config: Dict[str, Any]) -> None:
        """Configure logging settings for the processor.

        Args:
            logging_config: Dictionary with logging configuration settings
        """
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

        # Set the level for this module's logger
        logger.setLevel(log_level)

        # Configure other related loggers
        for module_name in ["parallel_agents", "workflow", "models", "utils"]:
            module_logger = logging.getLogger(
                f"graphs.openai_agent.parallel_agents.{module_name}"
            )
            module_logger.setLevel(log_level)

        logger.debug(f"Processor logging configured with level: {log_level_str}")
