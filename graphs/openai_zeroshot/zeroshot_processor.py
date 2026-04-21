"""
Zeroshot processor for clinical note analysis.

This module provides a simpler, single-prompt processor that replaces the multi-agent approach.
"""

import logging
import os
import sys
import time
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to sys.path to fix imports
parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(parent_dir)

# Import Azure authentication components
from graphs.openai_zeroshot.client_factory import create_azure_client
from graphs.openai_zeroshot.model_provider import (
    create_azure_provider,
    AzureModelProvider,
)

# Import model configuration and detection functions
from graphs.openai_zeroshot.model_config import (
    MODEL_CONFIGS,
    is_ollama_model,
    is_vllm_model,
    get_provider_type,
)

# Import the agents package components
from agents import (
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    set_default_openai_api,
)

# Import token tracker from models
from graphs.openai_zeroshot.models import TokenTracker, TokenUsageMetadata

# Import the zeroshot prompt using absolute import to avoid relative import errors
try:
    from graphs.openai_zeroshot.zeroshot_prompt import ZEROSHOT_PROMPT
except ImportError:
    # Try direct import if we're in the module directory
    try:
        from zeroshot_prompt import ZEROSHOT_PROMPT
    except ImportError as e:
        logger.error(f"Failed to import zeroshot prompt: {e}")
        # As a fallback, try loading the prompt file directly
        try:
            prompt_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "zeroshot_prompt.py"
            )
            with open(prompt_path, "r") as f:
                prompt_content = f.read()
                # Extract the ZEROSHOT_PROMPT variable
                import re

                prompt_match = re.search(
                    r'ZEROSHOT_PROMPT\s*=\s*"""(.*?)"""', prompt_content, re.DOTALL
                )
                if prompt_match:
                    ZEROSHOT_PROMPT = prompt_match.group(1)
                else:
                    raise ImportError("Could not extract ZEROSHOT_PROMPT from file")
        except Exception as e3:
            logger.error(f"Failed to load prompt directly: {e3}")
            raise ImportError(f"Could not load zeroshot prompt: {e3}")


# Define model classes locally to avoid import issues
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
        PMRN,
        NOTE_ID,
        NOTE_TEXT,
        SHORTENED_NOTE_TEXT,
        MESSAGES,
        PREDICTION,
        PROCESSED_AT,
        PROCESSING_TIME,
        MODEL_NAME,
        TOKEN_USAGE,
        ERROR=None,
    ):
        self.PMRN = PMRN
        self.NOTE_ID = NOTE_ID
        self.NOTE_TEXT = NOTE_TEXT
        self.SHORTENED_NOTE_TEXT = SHORTENED_NOTE_TEXT
        self.MESSAGES = MESSAGES
        self.PREDICTION = PREDICTION
        self.PROCESSED_AT = PROCESSED_AT
        self.PROCESSING_TIME = PROCESSING_TIME
        self.MODEL_NAME = MODEL_NAME
        self.TOKEN_USAGE = TOKEN_USAGE
        self.ERROR = ERROR


class ZeroshotProcessor:
    """
    A simplified processor that uses a single comprehensive prompt to analyze clinical notes.
    This replaces the multi-agent approach with a single end-to-end workflow.
    """

    def __init__(
        self,
        model_config_key: str = "default",
        prompt_variant: str = "default",
        max_concurrent_runs: int = 5,  # Add parameter for max concurrent runs
        logging_config: Optional[
            Dict[str, Any]
        ] = None,  # Add logging configuration parameter
        parameters: Optional[Dict[str, Any]] = None,  # Add parameters for provider configuration
    ):
        """
        Initialize the zeroshot processor.

        Args:
            model_config_key: Model configuration to use
            prompt_variant: Prompt variant to use (currently not used in zeroshot)
            max_concurrent_runs: Maximum number of concurrent LLM calls (default: 10)
            logging_config: Optional logging configuration dictionary
            parameters: Optional parameters for provider configuration
        """
        # Configure logging if provided
        if logging_config:
            self._configure_logging(logging_config)

        self.model_config_key = model_config_key
        self.prompt_variant = prompt_variant
        self.parameters = parameters or {}
        
        # Initialize providers to None
        self.azure_provider = None
        self.azure_client = None
        self.ollama_provider = None
        self.vllm_provider = None

        # Initialize token tracker
        self.token_tracker = TokenTracker()

        # Store max_concurrent_runs for later use when creating semaphore
        self.max_concurrent_runs = max_concurrent_runs
        # We'll initialize the semaphore lazily when needed, not in __init__
        self.semaphore = None

        logger.info(
            f"Configured ZeroshotProcessor with concurrency limit of {max_concurrent_runs}"
        )

        # Get the model configuration
        if model_config_key in MODEL_CONFIGS:
            model_config = MODEL_CONFIGS[model_config_key]
            self.model_name = model_config.model_name
        else:
            # Try to use the key as a model name directly
            self.model_name = model_config_key
            logger.warning(f"Model config '{model_config_key}' not found, using as model name directly")

        # Determine provider type (allow explicit override via CLI/parameters)
        provider_override = str(self.parameters.get("provider", "auto")).lower()
        if provider_override not in {"auto", "azure", "vllm", "ollama"}:
            raise ValueError(
                f"Invalid provider override '{provider_override}'. Use one of: auto, azure, vllm, ollama."
            )
        self.provider_type = (
            provider_override
            if provider_override != "auto"
            else get_provider_type(self.model_name)
        )
        logger.info(f"Provider type for model '{self.model_name}': {self.provider_type}")

        # Initialize provider-specific configuration
        if self.provider_type == "azure":
            # Initialize Azure configuration with better defaults
            self.azure_endpoint = os.environ.get(
                "AZURE_OPENAI_ENDPOINT",
                "https://bwh-bittermanlab-nonprod-openai-service.openai.azure.com/",
            )
            self.api_version = os.environ.get(
                "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
            )
            self.azure_deployment = os.environ.get("DEPLOYMENT_NAME", model_config_key)

            # Set token cache path
            self.token_cache_path = os.environ.get(
                "AZURE_TOKEN_CACHE_PATH", "/tmp/azure_token_cache.json"
            )

            # Map deployment names (this matches the behavior in openai_agent)
            self.deployment_mapping = {
                "default": "gpt-4o-mini-jg",
                "o1": "o1",
                "o3_mini": "o3-mini-jg",
                "turbo": "gpt-3.5-turbo",
                "4.1-mini": "gpt-4.1-mini",
                "4.1-nano": "gpt-4.1-nano",
                "o4-mini": "o4-mini",
            }

            # Configure Azure during initialization
            self.provider_configured = self.configure_azure()
            if not self.provider_configured:
                raise RuntimeError("Azure configuration failed during initialization.")
                
        elif self.provider_type == "ollama":
            # Configure Ollama
            self.provider_configured = self.configure_ollama()
            if not self.provider_configured:
                raise RuntimeError("Ollama configuration failed during initialization.")
                
        elif self.provider_type == "vllm":
            # Configure vLLM
            self.provider_configured = self.configure_vllm()
            if not self.provider_configured:
                raise RuntimeError("vLLM configuration failed during initialization.")
        else:
            raise ValueError(f"Unknown provider type: {self.provider_type}")

        # Try to import OpenAI library
        try:
            import openai

            self.openai = openai
        except ImportError:
            logger.error("Failed to import OpenAI library. Make sure it's installed.")
            raise ImportError(
                "OpenAI library not found. Please install it with 'pip install openai'"
            )

        # Try to import Azure utilities from the parallel agents module
        try:
            from graphs.openai_zeroshot.model_provider import (
                AzureModelProvider,
            )
            from graphs.openai_zeroshot.client_factory import (
                create_azure_client,
            )

            self.AzureModelProvider = AzureModelProvider
            self.create_azure_client = create_azure_client
        except ImportError as e:
            logger.warning(f"Could not import Azure utilities: {e}")
            self.AzureModelProvider = None
            self.create_azure_client = None

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

        # Configure related loggers
        other_loggers = ["model_provider", "client_factory"]
        for module_name in other_loggers:
            module_logger = logging.getLogger(f"graphs.openai_zeroshot.{module_name}")
            module_logger.setLevel(log_level)

        logger.debug(
            f"Zeroshot processor logging configured with level: {log_level_str}"
        )

    def configure_azure(self):
        """
        Configure Azure OpenAI for the processor.
        This implementation creates an AzureModelProvider for token acquisition.
        """
        # Get environment
        environment = os.getenv("ENVIRONMENT", "development").lower()

        # Check if we have Azure endpoint
        if not self.azure_endpoint:
            logger.error(
                "Azure endpoint not set, cannot proceed without Azure authentication"
            )
            return False

        try:
            # Create Azure provider instance with correct parameters
            self.azure_provider = create_azure_provider(
                azure_endpoint=self.azure_endpoint,
                azure_api_version=self.api_version,
                token_cache_path=self.token_cache_path,
            )
            return True
        except Exception as e:
            logger.error(f"Could not create Azure provider: {e}")
            # Store the provider as None to indicate failure
            self.azure_provider = None
            return False

    def configure_ollama(self):
        """
        Configure Ollama for local LLM inference.
        """
        try:
            from graphs.openai_zeroshot.ollama_provider import get_ollama_provider
            
            # Get Ollama endpoint from parameters or environment
            ollama_endpoint = self.parameters.get("ollama_endpoint")
            if not ollama_endpoint:
                ollama_endpoint = os.getenv("OLLAMA_ENDPOINT")
            if not ollama_endpoint:
                raise ValueError(
                    "Ollama provider selected but no endpoint configured. Pass --ollama-endpoint or set OLLAMA_ENDPOINT."
                )
            
            logger.info(f"Configuring Ollama provider with endpoint: {ollama_endpoint}")
            self.ollama_provider = get_ollama_provider(endpoint=ollama_endpoint)
            return True
        except Exception as e:
            logger.error(f"Could not create Ollama provider: {e}")
            self.ollama_provider = None
            return False

    def configure_vllm(self):
        """
        Configure vLLM for local LLM inference with OpenAI-compatible API.
        """
        try:
            from graphs.openai_zeroshot.vllm_provider import VLLMModelProvider
            
            # Get vLLM endpoint from parameters or environment
            vllm_endpoint = self.parameters.get("vllm_endpoint")
            if not vllm_endpoint:
                vllm_endpoint = os.getenv("VLLM_BASE_URL")
            if not vllm_endpoint:
                raise ValueError(
                    "vLLM provider selected but no endpoint configured. Pass --vllm-endpoint or set VLLM_BASE_URL."
                )
            
            logger.info(f"Configuring vLLM provider with endpoint: {vllm_endpoint}")
            self.vllm_provider = VLLMModelProvider(base_url=vllm_endpoint)
            return True
        except Exception as e:
            logger.error(f"Could not create vLLM provider: {e}")
            self.vllm_provider = None
            return False

    async def _run_with_semaphore(self, func, *args, **kwargs):
        """
        Run a function with a semaphore to limit concurrency.
        This wraps any async function call.

        Args:
            func: The async function to run
            *args, **kwargs: Arguments to pass to the function

        Returns:
            The result of the function
        """
        # Lazy initialize semaphore if needed
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.max_concurrent_runs)

        async with self.semaphore:
            logger.debug(
                f"Acquired semaphore. Available permits: {self.semaphore._value}"
            )
            try:
                return await func(*args, **kwargs)
            finally:
                logger.debug(
                    f"Released semaphore. Available permits: {self.semaphore._value + 1}"
                )

    async def process_note_async(self, note_input: NoteInput) -> NotePrediction:
        """
        Process a clinical note asynchronously using a single comprehensive prompt.
        Uses semaphore to limit concurrent LLM calls.

        Args:
            note_input: The clinical note to process

        Returns:
            NotePrediction containing the extracted information
        """
        start_time = time.time()

        # Ensure provider was configured successfully during init
        if not self.provider_configured:
            raise RuntimeError(
                f"{self.provider_type} provider not available. Configuration might have failed during initialization."
            )

        # Get the deployment/model name to use based on provider type
        if self.provider_type == "azure":
            deployment_name = self.deployment_mapping.get(
                self.model_config_key, self.deployment_mapping.get("default", self.model_name)
            )
        else:
            # For Ollama and vLLM, use the model name directly
            deployment_name = self.model_name

        # Get the maximum token limit based on model
        max_tokens = 8000 if "gpt-4" in deployment_name else 4000

        # Create messages array with system prompt and user note
        messages = [
            {"role": "system", "content": ZEROSHOT_PROMPT},
            {"role": "user", "content": note_input.note_text},
        ]

        # Initialize token usage tracking
        token_usage = TokenUsageMetadata()
        error_message = None
        prediction_data = {"event_analyses": []}

        try:
            # Get the model and client from the appropriate provider
            if self.provider_type == "azure":
                model = self.azure_provider.get_model(deployment_name)
                client = self.azure_provider.client
            elif self.provider_type == "ollama":
                model = self.ollama_provider.get_model(deployment_name)
                # Ollama provider creates its own client internally
                from openai import OpenAI
                client = OpenAI(
                    base_url=self.ollama_provider.ollama_endpoint,
                    api_key="ollama",  # Ollama doesn't need an API key
                )
            elif self.provider_type == "vllm":
                model = await self.vllm_provider.get_model(deployment_name)
                # vLLM uses the client from the provider
                client = self.vllm_provider.client
            else:
                raise ValueError(f"Unknown provider type: {self.provider_type}")

            # Define a function schema for structured output that matches our evaluation mapping
            functions = [
                {
                    "name": "clinical_note_analysis",
                    "description": (
                        "Return structured predicted_labels results in a standardized format. "
                        "Only grade, attribution, and certainty predictions are included, where "
                        "grade is between 0 and 5, attribution between 0 and 1, and certainty between 0 and 4."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "predicted_labels": {
                                "type": "object",
                                "properties": {
                                    # Pneumonitis
                                    "pneumonitis_current_grade": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 5,
                                        "description": "Current grade for pneumonitis (0-5)",
                                    },
                                    "pneumonitis_current_attribution": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 1,
                                        "description": "Current attribution for pneumonitis (0-1)",
                                    },
                                    "pneumonitis_current_certainty": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 4,
                                        "description": "Current certainty for pneumonitis (0-4)",
                                    },
                                    "pneumonitis_past_grade": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 5,
                                        "description": "Past grade for pneumonitis (0-5)",
                                    },
                                    "pneumonitis_past_attribution": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 1,
                                        "description": "Past attribution for pneumonitis (0-1)",
                                    },
                                    "pneumonitis_past_certainty": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 4,
                                        "description": "Past certainty for pneumonitis (0-4)",
                                    },
                                    # Colitis
                                    "colitis_current_grade": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 5,
                                        "description": "Current grade for colitis (0-5)",
                                    },
                                    "colitis_current_attribution": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 1,
                                        "description": "Current attribution for colitis (0-1)",
                                    },
                                    "colitis_current_certainty": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 4,
                                        "description": "Current certainty for colitis (0-4)",
                                    },
                                    "colitis_past_grade": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 5,
                                        "description": "Past grade for colitis (0-5)",
                                    },
                                    "colitis_past_attribution": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 1,
                                        "description": "Past attribution for colitis (0-1)",
                                    },
                                    "colitis_past_certainty": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 4,
                                        "description": "Past certainty for colitis (0-4)",
                                    },
                                    # Hepatitis
                                    "hepatitis_current_grade": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 5,
                                        "description": "Current grade for hepatitis (0-5)",
                                    },
                                    "hepatitis_current_attribution": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 1,
                                        "description": "Current attribution for hepatitis (0-1)",
                                    },
                                    "hepatitis_current_certainty": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 4,
                                        "description": "Current certainty for hepatitis (0-4)",
                                    },
                                    "hepatitis_past_grade": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 5,
                                        "description": "Past grade for hepatitis (0-5)",
                                    },
                                    "hepatitis_past_attribution": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 1,
                                        "description": "Past attribution for hepatitis (0-1)",
                                    },
                                    "hepatitis_past_certainty": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 4,
                                        "description": "Past certainty for hepatitis (0-4)",
                                    },
                                    # Dermatitis
                                    "dermatitis_current_grade": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 5,
                                        "description": "Current grade for dermatitis (0-5)",
                                    },
                                    "dermatitis_current_attribution": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 1,
                                        "description": "Current attribution for dermatitis (0-1)",
                                    },
                                    "dermatitis_current_certainty": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 4,
                                        "description": "Current certainty for dermatitis (0-4)",
                                    },
                                    "dermatitis_past_grade": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 5,
                                        "description": "Past grade for dermatitis (0-5)",
                                    },
                                    "dermatitis_past_attribution": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 1,
                                        "description": "Past attribution for dermatitis (0-1)",
                                    },
                                    "dermatitis_past_certainty": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 4,
                                        "description": "Past certainty for dermatitis (0-4)",
                                    },
                                    # Thyroiditis
                                    "thyroiditis_current_grade": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 5,
                                        "description": "Current grade for thyroiditis (0-5)",
                                    },
                                    "thyroiditis_current_attribution": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 1,
                                        "description": "Current attribution for thyroiditis (0-1)",
                                    },
                                    "thyroiditis_current_certainty": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 4,
                                        "description": "Current certainty for thyroiditis (0-4)",
                                    },
                                    "thyroiditis_past_grade": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 5,
                                        "description": "Past grade for thyroiditis (0-5)",
                                    },
                                    "thyroiditis_past_attribution": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 1,
                                        "description": "Past attribution for thyroiditis (0-1)",
                                    },
                                    "thyroiditis_past_certainty": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 4,
                                        "description": "Past certainty for thyroiditis (0-4)",
                                    },
                                    # Myocarditis
                                    "myocarditis_current_grade": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 5,
                                        "description": "Current grade for myocarditis (0-5)",
                                    },
                                    "myocarditis_current_attribution": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 1,
                                        "description": "Current attribution for myocarditis (0-1)",
                                    },
                                    "myocarditis_current_certainty": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 4,
                                        "description": "Current certainty for myocarditis (0-4)",
                                    },
                                    "myocarditis_past_grade": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 5,
                                        "description": "Past grade for myocarditis (0-5)",
                                    },
                                    "myocarditis_past_attribution": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 1,
                                        "description": "Past attribution for myocarditis (0-1)",
                                    },
                                    "myocarditis_past_certainty": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 4,
                                        "description": "Past certainty for myocarditis (0-4)",
                                    },
                                },
                                "required": [
                                    "pneumonitis_current_grade",
                                    "pneumonitis_current_attribution",
                                    "pneumonitis_current_certainty",
                                    "pneumonitis_past_grade",
                                    "pneumonitis_past_attribution",
                                    "pneumonitis_past_certainty",
                                    "colitis_current_grade",
                                    "colitis_current_attribution",
                                    "colitis_current_certainty",
                                    "colitis_past_grade",
                                    "colitis_past_attribution",
                                    "colitis_past_certainty",
                                    "hepatitis_current_grade",
                                    "hepatitis_current_attribution",
                                    "hepatitis_current_certainty",
                                    "hepatitis_past_grade",
                                    "hepatitis_past_attribution",
                                    "hepatitis_past_certainty",
                                    "dermatitis_current_grade",
                                    "dermatitis_current_attribution",
                                    "dermatitis_current_certainty",
                                    "dermatitis_past_grade",
                                    "dermatitis_past_attribution",
                                    "dermatitis_past_certainty",
                                    "thyroiditis_current_grade",
                                    "thyroiditis_current_attribution",
                                    "thyroiditis_current_certainty",
                                    "thyroiditis_past_grade",
                                    "thyroiditis_past_attribution",
                                    "thyroiditis_past_certainty",
                                    "myocarditis_current_grade",
                                    "myocarditis_current_attribution",
                                    "myocarditis_current_certainty",
                                    "myocarditis_past_grade",
                                    "myocarditis_past_attribution",
                                    "myocarditis_past_certainty",
                                ],
                            }
                        },
                        "required": ["predicted_labels"],
                    },
                }
            ]

            # Make the API call within a function to handle retry logic cleanly
            async def _make_api_call():
                response = await client.chat.completions.create(
                    model=deployment_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=max_tokens,
                    function_call={"name": "clinical_note_analysis"},
                    functions=functions,
                    timeout=600,  # 10 minute timeout
                )
                # Extract token usage and track it
                if hasattr(response, "usage"):
                    token_usage.prompt_tokens = response.usage.prompt_tokens
                    token_usage.completion_tokens = response.usage.completion_tokens
                    token_usage.total_tokens = response.usage.total_tokens

                    # Track token usage with our TokenTracker
                    self.token_tracker.track_usage(
                        agent_name="zeroshot",
                        result=response,
                        model_name=deployment_name,
                    )

                    logger.info(
                        f"Token usage: {token_usage.prompt_tokens} prompt, "
                        f"{token_usage.completion_tokens} completion, "
                        f"{token_usage.total_tokens} total"
                    )
                else:
                    logger.warning("No token usage information in response")

                # Rest of existing code
                return response

            # Wrap the API call with our semaphore - FIXED: Use the correct pattern
            # vLLM doesn't support function calling, so we need to handle it differently
            if self.provider_type == "vllm":
                # For vLLM, just use regular chat completion without functions
                response = await self._run_with_semaphore(
                    client.chat.completions.create,
                    model=deployment_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
            else:
                # For Azure/Ollama, use function calling
                response = await self._run_with_semaphore(
                    client.chat.completions.create,
                    model=deployment_name,
                    messages=messages,
                    functions=functions,
                    function_call={"name": "clinical_note_analysis"},
                    temperature=0.0,
                    max_tokens=max_tokens,
                )

            logger.debug(f"Response: {response}")
            # Extract the content from the response
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                message = choice.message

                # Check for function call first
                if hasattr(message, "function_call") and message.function_call:
                    content = message.function_call.arguments
                # Fall back to message content
                elif hasattr(message, "content") and message.content:
                    content = message.content
                else:
                    content = str(response)
                    logger.warning(
                        "No content found in response, using string representation"
                    )

                # Update token usage if available
                if hasattr(response, "usage"):
                    usage = response.usage
                    prompt_tokens = (
                        usage.prompt_tokens
                        if hasattr(usage, "prompt_tokens")
                        else 0
                    )
                    completion_tokens = (
                        usage.completion_tokens
                        if hasattr(usage, "completion_tokens")
                        else 0
                    )
                    total_tokens = (
                        usage.total_tokens if hasattr(usage, "total_tokens") else 0
                    )
                    
                    # Get cost rates for this model from utils/config.py
                    from utils.config import get_model_cost_rates
                    cost_rates = get_model_cost_rates(deployment_name)
                    
                    # Calculate costs
                    prompt_cost = (prompt_tokens / 1000000.0) * cost_rates["prompt"]
                    completion_cost = (completion_tokens / 1000000.0) * cost_rates["completion"]
                    total_cost = prompt_cost + completion_cost
                    
                    token_usage = TokenUsageMetadata(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        prompt_cost=prompt_cost,
                        completion_cost=completion_cost,
                        total_cost=total_cost,
                        model_costs=cost_rates
                    )
                    logger.debug(f"Token usage: {total_tokens} tokens, cost: ${total_cost:.6f}")
                    
                    # Track usage in the token tracker
                    self.token_tracker.track_usage(
                        agent_name="zeroshot",
                        result=response,
                        model_name=deployment_name
                    )
            else:
                content = str(response)
                logger.warning("Unexpected response format, using as string")

            # Parse the response content
            try:
                # First try to parse the JSON directly
                if isinstance(content, str):
                    if content.strip().startswith("{") and content.strip().endswith(
                        "}"
                    ):
                        prediction_data = json.loads(content)
                    else:
                        # Extract JSON from markdown code blocks if present
                        import re

                        json_matches = re.findall(
                            r"```(?:json)?\n([\s\S]+?)\n```", content
                        )
                        if json_matches:
                            prediction_data = json.loads(json_matches[0])
                        else:
                            # No JSON found, use the raw content
                            prediction_data = {"raw_response": content}
                            logger.warning(
                                "No JSON found in response, using raw content"
                            )

                else:
                    # If content is already a dict, use it directly
                    prediction_data = content
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {str(e)}")
                prediction_data = {
                    "error": "Failed to parse JSON response",
                    "raw_response": content,
                    "event_analyses": [],
                }
                error_message = f"JSON parse error: {str(e)}"

        except Exception as e:
            logger.error(f"Error processing note: {str(e)}")
            error_message = f"Processing error: {str(e)}"

        # Convert event_analyses format to predicted_labels format if needed
        if isinstance(prediction_data, dict) and "event_analyses" in prediction_data and "predicted_labels" not in prediction_data:
            prediction_data["predicted_labels"] = self._convert_event_analyses_to_predicted_labels(prediction_data["event_analyses"])

        # Safely access 'predicted_labels', default to empty dict if not found
        final_prediction = (
            prediction_data.get("predicted_labels", {})
            if isinstance(prediction_data, dict)
            else {}
        )
        if not final_prediction and not error_message:
            # If prediction_data was not a dict or didn't have 'predicted_labels'
            logger.warning(
                f"Prediction structure missing 'predicted_labels' for note {note_input.note_id}. Raw prediction: {prediction_data}"
            )
            if error_message is None:  # Avoid overwriting existing errors
                error_message = "Prediction structure missing 'predicted_labels'."

        # Calculate processing time
        processing_time = time.time() - start_time
        logger.debug(f"Processed note in {processing_time:.2f} seconds")

        # Return the prediction
        return NotePrediction(
            PMRN=note_input.pmrn,
            NOTE_ID=note_input.note_id,
            NOTE_TEXT=note_input.note_text,
            SHORTENED_NOTE_TEXT=note_input.note_text[:500] + "...",
            MESSAGES=[{"role": "system", "content": "Processing complete"}],
            PREDICTION=final_prediction,  # Use the safely accessed dictionary
            PROCESSED_AT=datetime.now(),
            PROCESSING_TIME=processing_time,
            MODEL_NAME=deployment_name,
            TOKEN_USAGE=token_usage,
            ERROR=error_message,
        )

    def set_concurrency_limit(self, max_concurrent_runs: int) -> None:
        """
        Update the concurrency limit at runtime.

        Args:
            max_concurrent_runs: New maximum number of concurrent LLM calls
        """
        if max_concurrent_runs < 1:
            logger.warning(
                f"Invalid concurrency limit {max_concurrent_runs}, using 1 instead"
            )
            max_concurrent_runs = 1

        # Update max_concurrent_runs and reset semaphore
        self.max_concurrent_runs = max_concurrent_runs
        self.semaphore = None  # Will be recreated on next use
        logger.info(f"Updated concurrency limit to {max_concurrent_runs}")

    def _convert_event_analyses_to_predicted_labels(self, event_analyses):
        """
        Convert event_analyses format from VLLM to predicted_labels format expected by the system.
        
        Args:
            event_analyses: List of event analysis dictionaries from VLLM response
            
        Returns:
            Dictionary in predicted_labels format
        """
        # Initialize with all required keys set to 0 (default)
        predicted_labels = {}
        
        # Event type mapping from analysis to label format
        event_mapping = {
            "Pneumonitis": "pneumonitis",
            "Myocarditis": "myocarditis", 
            "Colitis": "colitis",
            "Thyroiditis": "thyroiditis",
            "Hepatitis": "hepatitis",
            "Dermatitis": "dermatitis"
        }
        
        # Initialize all keys with default value of 0
        for event_name in event_mapping.values():
            for temporal in ["current", "past"]:
                predicted_labels[f"{event_name}_{temporal}_grade"] = 0
                predicted_labels[f"{event_name}_{temporal}_attribution"] = 0
                predicted_labels[f"{event_name}_{temporal}_certainty"] = 0
        
        # Now update with actual values from the response
        for analysis in event_analyses:
            event_type = analysis.get("event_type", "")
            if event_type not in event_mapping:
                continue
                
            event_prefix = event_mapping[event_type]
            
            # Extract grading information
            grading = analysis.get("grading", {})
            predicted_labels[f"{event_prefix}_current_grade"] = grading.get("current_grade", 0)
            predicted_labels[f"{event_prefix}_past_grade"] = grading.get("past_grade", 0)
            
            # Extract attribution information
            attribution = analysis.get("attribution", {})
            predicted_labels[f"{event_prefix}_current_attribution"] = attribution.get("current_attribution", 0)
            predicted_labels[f"{event_prefix}_past_attribution"] = attribution.get("past_attribution", 0)
            
            # Extract certainty information
            certainty = analysis.get("certainty", {})
            current_certainty = certainty.get("current_certainty", 0)
            past_certainty = certainty.get("past_certainty", 0)
            
            # Validation: If grade is 0, certainty should also be 0
            if predicted_labels[f"{event_prefix}_current_grade"] == 0 and current_certainty != 0:
                logger.debug(f"Correcting {event_prefix} current certainty from {current_certainty} to 0 (grade=0)")
                current_certainty = 0
            if predicted_labels[f"{event_prefix}_past_grade"] == 0 and past_certainty != 0:
                logger.debug(f"Correcting {event_prefix} past certainty from {past_certainty} to 0 (grade=0)")
                past_certainty = 0
                
            predicted_labels[f"{event_prefix}_current_certainty"] = current_certainty
            predicted_labels[f"{event_prefix}_past_certainty"] = past_certainty
        
        return predicted_labels

    async def process_notes(self, notes: List[NoteInput]) -> List[NotePrediction]:
        """
        Process multiple notes concurrently with semaphore control.

        Args:
            notes: List of notes to process

        Returns:
            List of note predictions
        """
        tasks = []

        # Create a task for each note, the semaphore built into process_note_async
        # will limit concurrent API calls
        for note in notes:
            # No need to use _run_with_semaphore here as it's already used inside process_note_async
            task = asyncio.create_task(self.process_note_async(note))
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        predictions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing note {notes[i].note_id}: {str(result)}")
                # Create error prediction
                predictions.append(
                    NotePrediction(
                        PMRN=notes[i].pmrn if notes[i].pmrn else "UNKNOWN",
                        NOTE_ID=notes[i].note_id,
                        NOTE_TEXT=notes[i].note_text,
                        SHORTENED_NOTE_TEXT="None",
                        MESSAGES=[],
                        PREDICTION={},
                        PROCESSED_AT=datetime.now(),
                        PROCESSING_TIME=0.0,
                        MODEL_NAME=self.model_config_key,
                        TOKEN_USAGE=None,
                        ERROR=f"Processing error: {str(result)}",
                    )
                )
            else:
                predictions.append(result)

        return predictions

    def process_notes_batch(self, notes_input: List[NoteInput]) -> List[NotePrediction]:
        """
        Process a batch of notes synchronously.
        This is a wrapper around the async method mainly for use with run_sweep.py

        Args:
            notes_input: List of clinical notes to process

        Returns:
            List of prediction results
        """
        # Get or create the event loop with better handling
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                logger.debug("Event loop is closed, creating a new one")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            logger.debug("No event loop in current thread, creating a new one")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Process the batch using the async method
        try:
            # Check if the loop is already running (e.g., in Jupyter or nested async)
            if loop.is_running():
                logger.warning("Event loop is already running. Using create_task to avoid blocking.")
                # Use asyncio.Future to get result from running loop
                future = asyncio.Future()
                
                async def run_process():
                    try:
                        result = await self.process_notes(notes_input)
                        future.set_result(result)
                    except Exception as e:
                        future.set_exception(e)
                
                loop.create_task(run_process())
                # Wait for the future to complete
                results = asyncio.run_coroutine_threadsafe(
                    asyncio.wait_for(future, timeout=3000), loop
                ).result()
            else:
                # The loop is not running, use run_until_complete
                results = loop.run_until_complete(self.process_notes(notes_input))

            # Log token usage summary without saving to file
            try:
                # Get the deployment name used based on provider type
                if self.provider_type == "azure":
                    deployment_name = self.deployment_mapping.get(
                        self.model_config_key, self.deployment_mapping.get("default", self.model_name)
                    )
                else:
                    # For Ollama and vLLM, use the model name directly
                    deployment_name = self.model_name

                # Log token usage statistics
                token_summary = self.token_tracker.get_summary()
                logger.info(
                    f"Total token usage for {deployment_name}: "
                    f"{token_summary['total']['prompt_tokens']} prompt tokens (${token_summary['total']['prompt_cost']:.4f}), "
                    f"{token_summary['total']['completion_tokens']} completion tokens (${token_summary['total']['completion_cost']:.4f}), "
                    f"Total: {token_summary['total']['total_tokens']} tokens, ${token_summary['total']['total_cost']:.4f}"
                )
            except Exception as e:
                logger.error(f"Error logging batch token usage: {e}")

            return results
        except Exception as e:
            logger.exception(f"Error in batch processing: {str(e)}")
            # Create error predictions
            error_predictions = []
            for note in notes_input:
                error_predictions.append(
                    NotePrediction(
                        PMRN=note.pmrn,
                        NOTE_ID=note.note_id,
                        NOTE_TEXT=note.note_text,
                        SHORTENED_NOTE_TEXT="None",
                        MESSAGES=[],
                        PREDICTION={},
                        PROCESSED_AT=datetime.now(),
                        PROCESSING_TIME=0.0,
                        MODEL_NAME=self.model_config_key,
                        TOKEN_USAGE=TokenUsageMetadata(),
                        ERROR=f"Batch processing error: {str(e)}",
                    )
                )
            return error_predictions
