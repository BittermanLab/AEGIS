"""
Note processor module for enhanced parallel workflow with judge-based evaluation.
"""

import asyncio
import logging
import uuid
import re
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import time
import json

# Import from external agents package
from agents import Agent, Runner, RunConfig, set_tracing_disabled

# Import directly from agent_factory module to avoid circular imports
from ..agent_factory import (
    create_enhanced_event_agents_with_judges,
    create_split_temporality_agents,
)
from ..models.input_models import NoteInput, TokenUsageMetadata
from ..models.output_models import (
    ExtractedNote,
    NotePrediction,
)
from ..models.enhanced_output_models import EnhancedPrediction
from ..models.event_processing_models import EventProcessingMetadata
from ..models.enhanced_judge_models import (
    AggregatedEventIdentification,
    AggregatedTemporality,
    AggregatedGrading,
)
from ..prompts.extractor_prompt import EXTRACTOR_PROMPT
from ..utils.token_tracker import TokenUsageTracker
from ..utils.model_config import (
    get_model_settings,
    get_model_for_role,
    is_ollama_model,
    is_vllm_model,
)
from .event_processor import (
    process_event_with_judge,
    run_parallel_event_identification,
    judge_event_identification,
    run_parallel_temporality_classification,
    judge_temporality_classification,
    run_parallel_grading,
    judge_grading,
)

logger = logging.getLogger(__name__)

# Set httpx logger to WARNING level to completely suppress HTTP request logs
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

# Disable tracing for all OpenAI agent executions
set_tracing_disabled(True)
logger.info("OpenAI Agents tracing disabled in note processor module")


# Add a function to log agent outputs clearly
def log_agent_output(stage: str, agent_type: str, result: Any, note_id: str = None):
    """
    Log the complete output of an agent for debugging purposes.

    Args:
        stage: The processing stage (extraction, etc.)
        agent_type: Type of agent (extractor, etc.)
        result: The agent result object
        note_id: Optional note ID being processed
    """
    note_info = f" for note {note_id}" if note_id else ""
    logger.debug(f"===== {stage} | {agent_type}{note_info} =====")

    # Log final output
    if hasattr(result, "final_output") and result.final_output:
        logger.debug(f"Final output: {result.final_output}")

        # For extracted note, show the content
        if hasattr(result.final_output, "extracted_note"):
            logger.debug(f"Extracted note: {result.final_output.extracted_note}...")

    # Try to log other properties if final_output doesn't exist
    else:
        logger.debug(f"Raw result: {result}")

    logger.debug("=" * 50)


def clean_clinical_note(note_text: str) -> str:
    """
    Clean clinical note text by removing problematic characters that might cause
    JSON parsing errors while preserving all clinical content.

    Args:
        note_text: Original clinical note text

    Returns:
        Sanitized note text safe for JSON and processing
    """
    if not note_text:
        return note_text

    # Step 1: Handle specific invisible control characters that cause parsing issues
    # Only target truly problematic control chars while preserving whitespace, newlines, and tabs
    cleaned_text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", note_text)

    # Step 2: Handle Unicode replacement character and other known problematic symbols
    # Only target specific problematic Unicode while preserving medical symbols
    problem_chars = [
        "\ufffd",  # Unicode replacement character
        "\u200b",  # Zero width space
        "\u200d",  # Zero width joiner
        "\u2028",  # Line separator
        "\u2029",  # Paragraph separator
    ]
    for char in problem_chars:
        cleaned_text = cleaned_text.replace(char, "")

    # Step 3: Selectively handle characters that break JSON but preserve medical content
    # First, properly escape characters needed for JSON structure
    # This preserves the characters in the content while making them safe for JSON
    json_escape_map = {
        "\\": "\\\\",  # Escape backslashes first
        '"': '\\"',  # Escape double quotes
        "\b": "\\b",  # Escape backspace
        "\f": "\\f",  # Escape form feed
        "\n": "\\n",  # Escape newline (while preserving it in the content)
        "\r": "\\r",  # Escape carriage return
        "\t": "\\t",  # Escape tab (while preserving it in the content)
    }

    for char, replacement in json_escape_map.items():
        cleaned_text = cleaned_text.replace(char, replacement)

    # Step 4: Normalize whitespace while preserving structure
    # Replace runs of spaces with a single space
    cleaned_text = re.sub(r" {2,}", " ", cleaned_text)

    # Replace runs of 3+ newlines with double newlines (preserve paragraph breaks)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    # Step 5: Handle truncation issues by ensuring proper string termination
    cleaned_text = cleaned_text.strip()

    # Step 6: Quick validation check - ensure the text can be properly encoded
    try:
        # Test if content can be safely encoded to UTF-8 (minimal validation)
        cleaned_text.encode("utf-8")
    except UnicodeEncodeError:
        # If encoding fails, fall back to a more aggressive cleaning
        # Replace any remaining problematic characters with their ASCII approximation if possible
        import unicodedata

        cleaned_text = unicodedata.normalize("NFKD", cleaned_text)
        cleaned_text = re.sub(r"[^\x00-\x7F]", lambda x: " ", cleaned_text)

    return cleaned_text


async def process_note_with_judge(
    note: NoteInput,
    model_config: str = "default",
    prompt_variant: str = "default",
    azure_provider=None,
    openai_client=None,
    parameters: Optional[Dict[str, Any]] = None,
) -> NotePrediction:
    """
    Process a single clinical note through the enhanced parallel workflow with judge-based evaluation

    Args:
        note: The clinical note to process
        model_config: Configuration for which model to use
        prompt_variant: Which prompt variant to use
        azure_provider: Optional Azure provider for authentication
        openai_client: Optional OpenAI client to use for this specific note (thread safety)

    Returns:
        NotePrediction: The final prediction result
    """
    start_time = datetime.now()
    token_tracker = TokenUsageTracker()
    metadata = EventProcessingMetadata()

    # Generate a unique request ID for this note processing
    request_id = str(uuid.uuid4())

    # Basic identification logging
    note_id = note.note_id
    logger.debug(f"[RequestID: {request_id}] ======= PROCESSING NOTE {note_id} =======")
    logger.debug(f"[RequestID: {request_id}] Model config: {model_config}")
    logger.debug(f"[RequestID: {request_id}] Prompt variant: {prompt_variant}")
    logger.debug(
        f"[RequestID: {request_id}] Note text (first 100 chars): {note.note_text[:100]}..."
    )

    default_event_types = [
        "Pneumonitis",
        "Myocarditis",
        "Colitis",
        "Thyroiditis",
        "Hepatitis",
        "Dermatitis",
    ]

    parameters = parameters or {}
    event_types = parameters.get("event_types", default_event_types)
    logger.debug(f"[RequestID: {request_id}] Event types to process: {event_types}")

    try:
        # Validate note text
        if not note.note_text:
            logger.critical(
                f"[RequestID: {request_id}] EMPTY NOTE DETECTED - This will cause fabrication in all downstream agents"
            )
            raise ValueError("Empty note text detected. Aborting processing.")

        # Clean the note text to reduce whitespace and newlines
        cleaned_note_text = (
            f"<<<SOURCE>>\n{clean_clinical_note(note.note_text)}\n<<<END>>>"
        )
        logger.debug(
            f"[RequestID: {request_id}] Cleaned note text (first 100 chars): {cleaned_note_text[:1000]}..."
        )

        # Detect provider type for this note/model
        extractor_model_name = get_model_for_role(model_config, "extractor")
        use_ollama = is_ollama_model(extractor_model_name)
        use_vllm = is_vllm_model(extractor_model_name)
        provider_type = "vLLM" if use_vllm else ("Ollama" if use_ollama else "Azure")
        logger.info(
            f"[RequestID: {request_id}] Provider for note {note_id}: {provider_type} (model: {extractor_model_name})"
        )

        # Verify we have the correct provider for this model type
        if (
            use_vllm
            and azure_provider is not None
            and not hasattr(azure_provider, "base_url")
        ):
            logger.warning(
                f"[RequestID: {request_id}] Using vLLM model but Azure provider was passed. This may cause errors."
            )
        elif (
            use_ollama
            and azure_provider is not None
            and not hasattr(azure_provider, "endpoint")
        ):
            logger.warning(
                f"[RequestID: {request_id}] Using Ollama model but Azure provider was passed. This may cause errors."
            )
        elif not use_ollama and not use_vllm and azure_provider is None:
            logger.warning(
                f"[RequestID: {request_id}] Using Azure model but no Azure provider was passed. This may cause errors."
            )

        # Initialize extraction agent with the appropriate model based on provider type
        if use_vllm:
            # For vLLM, we need to use the provider directly
            from ..utils.vllm_provider import VLLMModelProvider

            if azure_provider is None or not hasattr(azure_provider, "base_url"):
                # If provider wasn't passed or is wrong type, create one
                effective_params = parameters or {}
                provider_cfg = effective_params.get("provider_config", {})
                vllm_endpoint = (
                    effective_params.get("vllm_endpoint")
                    or provider_cfg.get("vllm_endpoint")
                    or os.getenv("VLLM_BASE_URL")
                )
                if not vllm_endpoint:
                    raise ValueError(
                        "vLLM endpoint is required but missing. Pass --vllm-endpoint or set VLLM_BASE_URL."
                    )
                vllm_provider = VLLMModelProvider(base_url=vllm_endpoint)
                extractor_model = vllm_provider.get_model(extractor_model_name)
            else:
                # Use the provider that was passed (should be vllm_provider)
                extractor_model = azure_provider.get_model(extractor_model_name)
        elif use_ollama:
            # For Ollama, we need to use the provider directly
            from ..utils.ollama_provider import get_ollama_provider

            if (
                azure_provider is None
            ):  # The parameter is named azure_provider but could be ollama_provider
                # If provider wasn't passed, create one
                ollama_provider = get_ollama_provider()
                extractor_model = ollama_provider.get_model(extractor_model_name)
            else:
                # Use the provider that was passed (should be ollama_provider)
                extractor_model = azure_provider.get_model(extractor_model_name)
        else:
            # For Azure, use the client directly if available
            extractor_model = get_model_for_role(
                model_config, "extractor", openai_client
            )

        logger.debug(
            f"[RequestID: {request_id}] Using extractor model: {extractor_model}"
        )

        extractor_agent = Agent(
            name="Note Extractor",
            instructions=EXTRACTOR_PROMPT,
            output_type=ExtractedNote,
            model=extractor_model,
            model_settings=get_model_settings(extractor_model_name),
        )

        # Set up run configuration with Azure provider if specified
        run_config = None
        if azure_provider:
            run_config = RunConfig(model_provider=azure_provider)
            logger.debug(
                f"[RequestID: {request_id}] Using Azure provider for extraction"
            )

        logger.debug(f"[RequestID: {request_id}] Starting note extraction...")
        # Run extraction agent with cleaned note text
        extract_result = await Runner.run(
            extractor_agent,
            cleaned_note_text,
            run_config=run_config,
        )

        # Log the complete extractor output
        log_agent_output("EXTRACTION", "Extractor", extract_result, note_id)

        # Check extraction result
        if (
            not hasattr(extract_result, "final_output")
            or not extract_result.final_output
        ):
            logger.critical(
                f"[RequestID: {request_id}] Extractor result has no final_output or it's empty!"
            )
            raise ValueError("Extraction failed - no output produced")

        # Track token usage with the extractor's model
        # Extract the actual model name from the model object
        if hasattr(extractor_model, "model"):
            model_name = extractor_model.model  # This is the deployment name
        else:
            model_name = (
                str(extractor_model) if extractor_model else extractor_model_name
            )

        token_tracker.track_usage(
            "extractor",
            extract_result,
            model_name,
        )
        extracted_note = extract_result.final_output
        logger.debug(
            f"[RequestID: {request_id}] Note extraction complete. Extracted length: {len(extracted_note.extracted_note)} chars"
        )

        # 3. Create enhanced specialized agents for each event type with role-specific models
        logger.debug(f"[RequestID: {request_id}] Creating event agents...")
        # Create fresh agents for this note to ensure thread safety when processing multiple notes
        event_agents = create_enhanced_event_agents_with_judges(
            model_config,
            azure_provider,
            request_id=request_id,
            event_types=event_types,
        )
        logger.debug(
            f"[RequestID: {request_id}] Event agents created for {len(event_agents)} event types"
        )

        # 4. Process each event in parallel with enhanced capabilities and judge-based evaluation
        logger.debug(
            f"[RequestID: {request_id}] Processing {len(event_types)} events for note {note.note_id}"
        )

        # Process each event type in sequence or in parallel depending on configuration
        enhanced_prediction = EnhancedPrediction()
        successful_events = 0
        failed_events = 0
        total_iterations = 0
        reasoning = ""

        # Initialize evidence structure for all event types to ensure consistency
        # even when specific events aren't detected
        enhanced_prediction.evidence = {}
        for event_type in event_types:
            event_type_lower = event_type.lower()
            enhanced_prediction.evidence[event_type_lower] = {
                "identification": {"past": [], "current": []},
                "grading": {"past": [], "current": []},
                "attribution": {"past": [], "current": []},
                "certainty": {"past": [], "current": []},
            }

        event_processing_tasks = [
            process_event_with_judge(
                event_type,
                event_agents,
                extracted_note.extracted_note,
                max_iterations=1,  # paramaterize later
                azure_provider=azure_provider,
                token_tracker=token_tracker,  # Pass token tracker to track all agents
                request_id=request_id,  # Pass request ID for tracking
                prompt_variant=prompt_variant,  # Pass the prompt variant for ablation studies
            )
            for event_type in event_types
        ]

        # Run all event processing tasks concurrently
        logger.debug(f"[RequestID: {request_id}] Starting parallel event processing...")
        event_results = await asyncio.gather(
            *event_processing_tasks, return_exceptions=True
        )
        logger.debug(f"[RequestID: {request_id}] Parallel event processing complete")

        # 5. Combine results into enhanced prediction format
        logger.debug(
            f"[RequestID: {request_id}] Combining event results for note {note.note_id}"
        )
        reasoning = ""

        logger.debug(f"[RequestID: {request_id}] Processing event results:")
        user_overviews = []  # Collect user overviews to combine later
        for i, result in enumerate(event_results):
            event_type = event_types[i]
            if isinstance(result, Exception):
                logger.error(
                    f"[RequestID: {request_id}] Error in event processing ({event_type}): {str(result)}"
                )
                reasoning += f"Error processing event: {str(result)}\n"
                failed_events += 1
                continue

            successful_events += 1
            total_iterations += result.iterations_completed
            event_type_lower = result.event_type.lower()

            # Collect user overview if available
            if hasattr(result, "user_overview") and result.user_overview:
                # Save each event's user overview
                user_overview = f"{event_type}: {result.user_overview}"
                user_overviews.append(user_overview)

                # Also store in the event-specific user overview field
                if hasattr(enhanced_prediction, f"{event_type_lower}_user_overview"):
                    setattr(
                        enhanced_prediction,
                        f"{event_type_lower}_user_overview",
                        result.user_overview,
                    )

                logger.debug(
                    f"[RequestID: {request_id}][User Overview] Added overview for {event_type}: {result.user_overview}"
                )

            logger.debug(
                f"[RequestID: {request_id}] Event {event_type}: grade={result.grade}, past={result.past_grade}, current={result.current_grade}"
            )
            logger.debug(
                f"[RequestID: {request_id}] Event {event_type}: attribution={result.attribution}, certainty={result.certainty}"
            )

            # Set grade values
            setattr(
                enhanced_prediction,
                f"{event_type_lower}_past_grade",
                result.past_grade or 0,
            )
            setattr(
                enhanced_prediction,
                f"{event_type_lower}_current_grade",
                result.current_grade or 0,
            )

            # Set attribution values
            setattr(
                enhanced_prediction,
                f"{event_type_lower}_past_attribution",
                result.past_attribution or 0,
            )
            setattr(
                enhanced_prediction,
                f"{event_type_lower}_current_attribution",
                result.current_attribution or 0,
            )

            # Set certainty values
            setattr(
                enhanced_prediction,
                f"{event_type_lower}_past_certainty",
                result.past_certainty or 0,
            )
            setattr(
                enhanced_prediction,
                f"{event_type_lower}_current_certainty",
                result.current_certainty or 0,
            )

            # Add reasoning
            if result.reasoning:
                reasoning += f"{result.event_type} (iterations: {result.iterations_completed}):\n{result.reasoning}\n\n"

            # Add event's structured evidence to the global evidence structure
            if hasattr(result, "evidence") and result.evidence:
                # Get the event-specific evidence field name
                event_evidence_field = f"{event_type_lower}_evidence"

                # Check if the field exists in the enhanced prediction
                if hasattr(enhanced_prediction, event_evidence_field):
                    # Copy the structured evidence to the event-specific field
                    setattr(enhanced_prediction, event_evidence_field, result.evidence)
                    logger.debug(
                        f"[RequestID: {request_id}][Evidence] Copied structured evidence for {event_type}"
                    )

                # Also add to the legacy evidence structure for backward compatibility
                if (
                    hasattr(enhanced_prediction, "evidence")
                    and event_type_lower not in enhanced_prediction.evidence
                ):
                    enhanced_prediction.evidence[event_type_lower] = {}

                # Copy evidence from the event result to the global structure
                for category, evidence_data in result.evidence.items():
                    if category not in enhanced_prediction.evidence[event_type_lower]:
                        enhanced_prediction.evidence[event_type_lower][category] = {}

                    # Copy evidence data
                    enhanced_prediction.evidence[event_type_lower][
                        category
                    ] = evidence_data
            else:
                # Create empty structure if no evidence available
                event_type_lower = event_type.lower()
                event_evidence_field = f"{event_type_lower}_evidence"

                if hasattr(enhanced_prediction, event_evidence_field):
                    # Create default empty structure
                    empty_evidence = {
                        "identification": {"evidence": []},
                        "past_grading": {"evidence": []},
                        "current_grading": {"evidence": []},
                        "attribution": {"evidence": []},
                        "certainty": {"evidence": []},
                    }
                    setattr(enhanced_prediction, event_evidence_field, empty_evidence)

        # After processing all events, create a combined user summary
        if user_overviews:
            user_summary = "\n\n".join(user_overviews)
            # Add to enhanced prediction object
            enhanced_prediction.user_summary = user_summary
            logger.debug(
                f"[RequestID: {request_id}][User Summary] Created combined user summary with {len(user_overviews)} event overviews"
            )
        else:
            enhanced_prediction.user_summary = (
                "No significant findings detected in this clinical note."
            )
            logger.debug(
                f"[RequestID: {request_id}][User Summary] No user overviews available, created default summary"
            )

        # Set the combined reasoning
        enhanced_prediction.reasoning = reasoning.strip()

        # Calculate total processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        logger.debug(
            f"[RequestID: {request_id}] Processing complete in {processing_time:.2f} seconds"
        )
        logger.debug(
            f"[RequestID: {request_id}] Events: {successful_events} successful, {failed_events} failed"
        )
        logger.debug(f"[RequestID: {request_id}] Total iterations: {total_iterations}")

        # Generate token usage report including cost breakdown
        token_report = token_tracker.generate_cost_report()
        logger.debug(f"[RequestID: {request_id}] Token usage report: {token_report}")

        # Build the final prediction object with the standardized class
        logger.debug(
            f"[RequestID: {request_id}] [Final Output Debug] Creating NotePrediction with enhanced_prediction dict: {enhanced_prediction.dict().keys()}"
        )

        # Check if evidence fields exist in the final prediction dict
        for event_type in event_types:
            event_type_lower = event_type.lower()
            evidence_field = f"{event_type_lower}_evidence"
            if evidence_field in enhanced_prediction.dict():
                logger.debug(
                    f"[RequestID: {request_id}] [Final Output Debug] {evidence_field} IS present in the final prediction dict"
                )
                logger.debug(
                    f"[RequestID: {request_id}] [Final Output Debug] {evidence_field} value: {enhanced_prediction.dict()[evidence_field]}"
                )
            else:
                logger.debug(
                    f"[RequestID: {request_id}] [Final Output Debug] {evidence_field} is NOT present in the final prediction dict"
                )

        # Create NotePrediction object from the enhanced prediction
        token_usage = token_tracker.get_total_usage()
        token_usage_metadata = TokenUsageMetadata(
            prompt_tokens=token_usage["prompt_tokens"],
            completion_tokens=token_usage["completion_tokens"],
            total_tokens=token_usage["total_tokens"],
            prompt_cost=token_usage.get("prompt_cost", 0.0),
            completion_cost=token_usage.get("completion_cost", 0.0),
            total_cost=token_usage.get("total_cost", 0.0),
            model_costs=token_usage.get("model_breakdown", {}),
        )

        # Log summary information
        logger.info(
            f"[RequestID: {request_id}] Completed processing note {note.note_id} in {processing_time:.2f} seconds. "
            f"Events: {successful_events} successful, {failed_events} failed. "
            f"Tokens: {token_usage['total_tokens']}"
        )

        # Create a NotePrediction object with all the information
        note_prediction = NotePrediction(
            PMRN=note.pmrn,
            NOTE_ID=note.note_id,
            NOTE_TEXT=note.note_text,
            SHORTENED_NOTE_TEXT=extracted_note.extracted_note,
            MESSAGES=[
                {"role": "user", "content": cleaned_note_text},
                {
                    "role": "assistant",
                    "content": f"Processed in {processing_time:.2f} seconds with judge-based evaluation",
                },
            ],
            PREDICTION=enhanced_prediction.__dict__,  # Convert to dict for serialization
            PROCESSED_AT=datetime.now().isoformat(),
            PROCESSING_TIME=processing_time,
            MODEL_NAME=model_config,
            TOKEN_USAGE=token_usage_metadata,
            ERROR=None,
        )

        logger.debug(
            f"[RequestID: {request_id}] ======= COMPLETED PROCESSING NOTE {note_id} ======="
        )
        return note_prediction
    except Exception as e:
        # Log the error
        logger.error(
            f"[RequestID: {request_id}] Error processing note {note.note_id}: {str(e)}",
            exc_info=True,
        )
        logger.debug(
            f"[RequestID: {request_id}] ======= FAILED PROCESSING NOTE {note_id} ======="
        )

        # Construct a minimal prediction with error info - without trying to reference a potentially undefined variable
        token_usage = token_tracker.get_total_usage() if token_tracker else {}
        token_usage_metadata = TokenUsageMetadata(
            prompt_tokens=token_usage.get("prompt_tokens", 0),
            completion_tokens=token_usage.get("completion_tokens", 0),
            total_tokens=token_usage.get("total_tokens", 0),
            prompt_cost=token_usage.get("prompt_cost", 0.0),
            completion_cost=token_usage.get("completion_cost", 0.0),
            total_cost=token_usage.get("total_cost", 0.0),
            model_costs=token_usage.get("model_breakdown", {}),
        )
        processing_time = (
            (datetime.now() - start_time).total_seconds()
            if "start_time" in locals()
            else 0.0
        )
        return NotePrediction(
            PMRN=note.pmrn,
            NOTE_ID=note.note_id,
            NOTE_TEXT=note.note_text,
            SHORTENED_NOTE_TEXT="",
            MESSAGES=[],
            PREDICTION={},
            PROCESSED_AT=datetime.now().isoformat(),
            PROCESSING_TIME=processing_time,
            MODEL_NAME=model_config,
            TOKEN_USAGE=token_usage_metadata,
            ERROR=str(e),
        )


async def process_all_notes_with_judge(
    notes: List[NoteInput],
    model_config: str = "default",
    prompt_variant: str = "default",
    batch_size: int = 5,
    model_provider=None,  # Renamed from azure_provider to be more generic
    openai_client=None,
) -> List[NotePrediction]:
    """
    Process a list of clinical notes in parallel, with appropriate batching to avoid rate limits

    Args:
        notes: List of clinical notes to process
        model_config: Configuration for which model to use
        prompt_variant: Which prompt variant to use
        batch_size: Number of notes to process concurrently
        model_provider: Optional model provider (Ollama or Azure) for authentication
        openai_client: Optional OpenAI client to use for this specific note (thread safety)

    Returns:
        List of NotePrediction objects
    """
    if not notes:
        return []

    # Use the same NotePrediction class as in the process_note_with_judge function
    try:
        # Try absolute import first (works in local dev)
        from graphs.agent_split_temporality.models import (
            NotePrediction as GlobalNotePrediction,
        )
    except ImportError:
        # If that fails, use the one we already imported at the top
        GlobalNotePrediction = NotePrediction

    logger.info(f"Processing {len(notes)} notes with batch size {batch_size}")
    note_predictions = []

    # Detect provider type for this model configuration
    from ..utils.model_config import is_ollama_model, get_model_for_role

    extractor_model_name = get_model_for_role(model_config, "extractor")
    use_ollama = is_ollama_model(extractor_model_name)
    provider_type = "Ollama" if use_ollama else "Azure"
    logger.info(
        f"Provider for batch processing: {provider_type} (model: {extractor_model_name})"
    )

    # If using Ollama but no provider was passed, create one
    if use_ollama and model_provider is None:
        from ..utils.ollama_provider import get_ollama_provider

        batch_provider = get_ollama_provider()
    else:
        batch_provider = model_provider

    # Prepare batches for processing
    batches = [notes[i : i + batch_size] for i in range(0, len(notes), batch_size)]
    total_notes = len(notes)
    processed = []

    # Process each batch with semaphore control to limit concurrency
    for batch_idx, batch in enumerate(batches):
        # Log progress for this batch
        batch_start = len(processed) + 1
        batch_end = min(len(processed) + len(batch), total_notes)
        logger.info(
            f"Processing batch {batch_idx+1}/{len(batches)} - notes {batch_start}-{batch_end} of {total_notes}"
        )

        # Create tasks for all notes in this batch
        tasks = [
            asyncio.create_task(
                process_note_with_judge(
                    note,
                    model_config,
                    prompt_variant,
                    batch_provider,
                    None if use_ollama else openai_client,  # Only use client for Azure
                    parameters={},
                )
            )
            for note in batch
        ]

        # Process the batch concurrently
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions in the results
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(
                    f"Error processing note {batch[j].note_id}: {str(result)}",
                    exc_info=True,
                )
                # Creating a basic NotePrediction with error information
                error_obj = GlobalNotePrediction(
                    PMRN=batch[j].pmrn,
                    NOTE_ID=batch[j].note_id,
                    NOTE_TEXT=batch[j].note_text,
                    SHORTENED_NOTE_TEXT="",
                    MESSAGES=[],
                    PREDICTION={},
                    PROCESSED_AT=datetime.now().isoformat(),
                    PROCESSING_TIME=0.0,
                    MODEL_NAME=model_config,
                    TOKEN_USAGE=TokenUsageMetadata(
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        prompt_cost=0.0,
                        completion_cost=0.0,
                        total_cost=0.0,
                        model_costs={},
                    ),
                    ERROR=str(result),
                )
                note_predictions.append(error_obj)
            else:
                note_predictions.append(result)

        # Add the processed batch to our tracking list
        processed.extend(batch)

    logger.info(f"Completed processing {len(notes)} notes")
    return note_predictions


async def process_note_with_split_temporality(
    note_input: NoteInput,
    model_config: str = "default",
    azure_provider=None,
    request_id: str = None,
    token_tracker=None,
    event_types: Optional[List[str]] = None,
    provider_config: Optional[Dict[str, Any]] = None,
    prompt_variant: str = "default",
) -> NotePrediction:
    """
    Process a single note using the split temporality workflow.
    This separates event identification from temporal classification.

    Args:
        note_input: The clinical note to process
        model_config: Configuration for which models to use
        azure_provider: Optional Azure provider for authentication
        request_id: Unique request ID for thread tracking
        token_tracker: Optional token usage tracker
        event_types: Optional list of event types to process
        provider_config: Optional configuration for model providers
        prompt_variant: Which prompt variant to use for ablation studies

    Returns:
        NotePrediction: The processed note prediction
    """
    if not request_id:
        request_id = str(uuid.uuid4())

    logger.info(
        f"[RequestID: {request_id}] Processing note {note_input.note_id} with split temporality workflow"
    )

    # Set default event types if not provided
    if event_types is None:
        event_types = [
            "pneumonitis",
            "colitis",
            "hepatitis",
            "dermatitis",
            "thyroiditis",
            "myocarditis",
        ]

    # Initialize timing
    start_time = datetime.now()

    # Initialize token tracker if not provided
    if token_tracker is None:
        token_tracker = TokenUsageTracker()

    try:
        # Clean the note text
        cleaned_note_text = clean_clinical_note(note_input.note_text)

        # Create enhanced prediction for storing all results
        enhanced_prediction = EnhancedPrediction()

        # --- Perform Note Extraction ---
        extractor_model_name = get_model_for_role(model_config, "extractor")
        use_ollama = is_ollama_model(extractor_model_name)
        use_vllm = is_vllm_model(extractor_model_name)
        provider_type = "vLLM" if use_vllm else ("Ollama" if use_ollama else "Azure")

        logger.info(
            f"[RequestID: {request_id}] Provider for note {note_input.note_id}: {provider_type} (model: {extractor_model_name})"
        )

        # Verify we have the correct provider for this model type
        if (
            use_vllm
            and azure_provider is not None
            and not hasattr(azure_provider, "base_url")
        ):
            logger.warning(
                f"[RequestID: {request_id}] Using vLLM model but Azure provider was passed. This may cause errors."
            )
        elif (
            use_ollama
            and azure_provider is not None
            and not hasattr(azure_provider, "endpoint")
        ):
            logger.warning(
                f"[RequestID: {request_id}] Using Ollama model but Azure provider was passed. This may cause errors."
            )
        elif not use_ollama and not use_vllm and azure_provider is None:
            logger.warning(
                f"[RequestID: {request_id}] Using Azure model but no Azure provider was passed. This may cause errors."
            )

        # Initialize extraction agent with the appropriate model based on provider type
        if use_vllm:
            # For vLLM, we need to use the provider directly
            from ..utils.vllm_provider import VLLMModelProvider

            if azure_provider is None or not hasattr(azure_provider, "base_url"):
                # If provider wasn't passed or is wrong type, create one
                effective_cfg = provider_config or {}
                nested_cfg = effective_cfg.get("provider_config", {})
                vllm_endpoint = (
                    effective_cfg.get("vllm_endpoint")
                    or nested_cfg.get("vllm_endpoint")
                    or os.getenv("VLLM_BASE_URL")
                )
                if not vllm_endpoint:
                    raise ValueError(
                        "vLLM endpoint is required but missing. Pass --vllm-endpoint or set VLLM_BASE_URL."
                    )
                vllm_provider = VLLMModelProvider(base_url=vllm_endpoint)
                extractor_model = vllm_provider.get_model(extractor_model_name)
            else:
                # Use the provider that was passed (should be vllm_provider)
                extractor_model = azure_provider.get_model(extractor_model_name)
        elif use_ollama:
            # For Ollama, we need to use the provider directly
            from ..utils.ollama_provider import get_ollama_provider

            if (
                azure_provider is None
            ):  # The parameter is named azure_provider but could be ollama_provider
                # If provider wasn't passed, create one
                ollama_provider = get_ollama_provider()
                extractor_model = ollama_provider.get_model(extractor_model_name)
            else:
                # Use the provider that was passed (should be ollama_provider)
                extractor_model = azure_provider.get_model(extractor_model_name)
        else:
            # For Azure, use the client directly if available
            extractor_model = get_model_for_role(model_config, "extractor", None)

        logger.debug(
            f"[RequestID: {request_id}] Using extractor model: {extractor_model}"
        )

        extractor_agent = Agent(
            name="Note Extractor",
            instructions=EXTRACTOR_PROMPT,
            output_type=ExtractedNote,
            model=extractor_model,
            model_settings=get_model_settings(extractor_model_name),
        )

        run_config_extractor = None
        if azure_provider and not use_ollama and not use_vllm:
            run_config_extractor = RunConfig(model_provider=azure_provider)
        elif use_ollama:
            # Ensure the provider is correctly passed for Ollama if it has its own RunConfig needs
            # This might need adjustment based on how `agents` library handles Ollama providers in RunConfig
            if hasattr(
                azure_provider, "client"
            ):  # If azure_provider is an ollama_provider, it won't have client
                run_config_extractor = RunConfig(model_provider=azure_provider)
        elif use_vllm:
            # For vLLM, check if we have the right provider type
            if azure_provider and hasattr(azure_provider, "base_url"):
                run_config_extractor = RunConfig(model_provider=azure_provider)

        extract_result = await Runner.run(
            extractor_agent,
            cleaned_note_text,
            run_config=run_config_extractor,
        )

        if (
            not hasattr(extract_result, "final_output")
            or not extract_result.final_output
        ):
            logger.error(
                f"[RequestID: {request_id}] Note extraction failed or produced no output."
            )
            raise ValueError("Note extraction failed.")

        extracted_note_obj = extract_result.final_output
        token_tracker.track_usage("extractor", extract_result, extractor_model_name)
        logger.debug(
            f"[RequestID: {request_id}] Note extraction complete. Extracted length: {len(extracted_note_obj.extracted_note)} chars"
        )
        # --- End Note Extraction ---

        # Create the split temporality agents
        event_agents = create_split_temporality_agents(
            model_config=model_config,
            azure_provider=azure_provider,  # This provider is for the subsequent event agents
            request_id=request_id,
            event_types=event_types,
            provider_config=provider_config,
        )

        # Process each event type with the split workflow
        event_results = {}
        user_overviews = []
        reasoning = ""
        all_evidence = {}

        for event_type in event_types:
            logger.info(
                f"[RequestID: {request_id}] Processing {event_type} with split temporality workflow"
            )

            agents = event_agents.get(event_type, {})
            event_type_lower = event_type.lower()

            event_evidence = {
                "identification": {"evidence": []},
                "temporality": {"evidence": []},
                "past_grading": {"evidence": []},
                "current_grading": {"evidence": []},
                "attribution": {"evidence": []},
                "certainty": {"evidence": []},
            }

            event_identifiers = agents.get("event_identifiers", [])
            event_identification_judge = agents.get("event_identification_judge")

            # ABLATION STUDY: Modify identifier agents based on prompt variant
            if (
                prompt_variant == "ablation_single"
                or prompt_variant == "ablation_no_judge"
            ) and event_identifiers:
                # Use only the first identifier agent for single agent ablation
                # Note: ablation_no_judge implies ablation_single since it doesn't make sense to run multiple agents and ignore results
                logger.debug(
                    f"[RequestID: {request_id}] ABLATION STUDY: Using single identifier agent for {event_type} (variant: {prompt_variant})"
                )
                event_identifiers = [event_identifiers[0]]

            if event_identifiers and event_identification_judge:
                identification_results = await run_parallel_event_identification(
                    event_type=event_type,
                    note_text=extracted_note_obj.extracted_note,
                    identifier_agents=event_identifiers,
                    azure_provider=azure_provider,
                    token_tracker=token_tracker,
                )
                # Judge identification results or use first result for ablation_no_judge
                if prompt_variant == "ablation_no_judge" and identification_results:
                    logger.debug(
                        f"[RequestID: {request_id}] ABLATION STUDY: Skipping judge for {event_type}, using first identifier result"
                    )
                    # Use the first identification result directly
                    first_result = identification_results[0]

                    # Convert to AggregatedEventIdentification format (expected by downstream functions)
                    aggregated_identification = AggregatedEventIdentification(
                        event_present=first_result.event_present,
                        evidence_snippets=first_result.evidence_snippets,
                        reasoning=first_result.reasoning,
                    )
                else:
                    # Standard workflow - use the judge to evaluate results
                    aggregated_identification = await judge_event_identification(
                        event_type=event_type,
                        identification_results=identification_results,
                        note_text=extracted_note_obj.extracted_note,
                        judge_agent=event_identification_judge,
                        azure_provider=azure_provider,
                        token_tracker=token_tracker,
                    )
                if aggregated_identification.evidence_snippets:
                    event_evidence["identification"]["evidence"].extend(
                        aggregated_identification.evidence_snippets
                    )

                if aggregated_identification.event_present:
                    temporality_classifiers = agents.get("temporality_classifiers", [])
                    temporality_judge = agents.get("temporality_judge")

                    # ABLATION STUDY: Modify temporality classifier agents based on prompt variant
                    if (
                        prompt_variant == "ablation_single"
                        or prompt_variant == "ablation_no_judge"
                    ) and temporality_classifiers:
                        # Use only the first temporality classifier for single agent ablation
                        # Note: ablation_no_judge implies ablation_single since it doesn't make sense to run multiple agents and ignore results
                        logger.debug(
                            f"[RequestID: {request_id}] ABLATION STUDY: Using single temporality classifier for {event_type} (variant: {prompt_variant})"
                        )
                        temporality_classifiers = [temporality_classifiers[0]]

                    if temporality_classifiers and temporality_judge:
                        temporality_results = (
                            await run_parallel_temporality_classification(
                                event_type=event_type,
                                note_text=extracted_note_obj.extracted_note,
                                classifier_agents=temporality_classifiers,
                                azure_provider=azure_provider,
                                token_tracker=token_tracker,
                            )
                        )
                        # Judge temporality results or use first result for ablation_no_judge
                        if (
                            prompt_variant == "ablation_no_judge"
                            and temporality_results
                        ):
                            logger.debug(
                                f"[RequestID: {request_id}] ABLATION STUDY: Skipping temporality judge for {event_type}, using first classifier result"
                            )
                            # Use the first temporality result directly
                            first_result = temporality_results[0]

                            # Convert to AggregatedTemporality format (expected by downstream functions)
                            aggregated_temporality = AggregatedTemporality(
                                past_events=first_result.past_events,
                                current_events=first_result.current_events,
                                evidence_snippets=first_result.evidence_snippets,
                                reasoning=first_result.reasoning,
                            )
                        else:
                            # Standard workflow - use the judge to evaluate results
                            aggregated_temporality = (
                                await judge_temporality_classification(
                                    event_type=event_type,
                                    temporality_results=temporality_results,
                                    note_text=extracted_note_obj.extracted_note,
                                    judge_agent=temporality_judge,
                                    azure_provider=azure_provider,
                                    token_tracker=token_tracker,
                                )
                            )
                        if aggregated_temporality.evidence_snippets:
                            event_evidence["temporality"]["evidence"].extend(
                                aggregated_temporality.evidence_snippets
                            )
                        if aggregated_temporality.reasoning:
                            event_evidence["temporality"]["evidence"].append(
                                f"Temporality Reasoning: {aggregated_temporality.reasoning}"
                            )

                        past_graders = agents.get("past_graders", [])
                        current_graders = agents.get("current_graders", [])
                        past_grading_judge = agents.get("past_grading_judge")
                        current_grading_judge = agents.get("current_grading_judge")
                        past_grade, current_grade = 0, 0

                        # ABLATION STUDY: Modify grader agents based on prompt variant
                        if (
                            prompt_variant == "ablation_single"
                            or prompt_variant == "ablation_no_judge"
                        ) and past_graders:
                            # Use only the first past grader for single agent ablation
                            # Note: ablation_no_judge implies ablation_single since it doesn't make sense to run multiple agents and ignore results
                            logger.debug(
                                f"[RequestID: {request_id}] ABLATION STUDY: Using single past grader for {event_type} (variant: {prompt_variant})"
                            )
                            past_graders = [past_graders[0]]

                        if (
                            prompt_variant == "ablation_single"
                            or prompt_variant == "ablation_no_judge"
                        ) and current_graders:
                            # Use only the first current grader for single agent ablation
                            # Note: ablation_no_judge implies ablation_single since it doesn't make sense to run multiple agents and ignore results
                            logger.debug(
                                f"[RequestID: {request_id}] ABLATION STUDY: Using single current grader for {event_type} (variant: {prompt_variant})"
                            )
                            current_graders = [current_graders[0]]

                        if (
                            aggregated_temporality.past_events
                            and past_graders
                            and past_grading_judge
                        ):
                            past_grading_results = await run_parallel_grading(
                                event_type,
                                "past",
                                aggregated_temporality.past_events,
                                extracted_note_obj.extracted_note,
                                past_graders,
                                azure_provider,
                                token_tracker,
                            )
                            # Judge past grading results or use first result for ablation_no_judge
                            if (
                                prompt_variant == "ablation_no_judge"
                                and past_grading_results
                            ):
                                logger.debug(
                                    f"[RequestID: {request_id}] ABLATION STUDY: Skipping past grading judge for {event_type}, using first grader result"
                                )
                                # Use the first grading result directly
                                first_result = past_grading_results[0]

                                # Convert to AggregatedGrading format (expected by downstream functions)
                                judged_past_grading = AggregatedGrading(
                                    event_name=event_type,
                                    grade=first_result.grade,
                                    temporal_context="past",
                                    rationale=first_result.rationale,
                                    evidence_snippets=first_result.evidence_snippets,
                                )
                            else:
                                # Standard workflow - use the judge to evaluate results
                                judged_past_grading = await judge_grading(
                                    event_type,
                                    "past",
                                    past_grading_results,
                                    aggregated_temporality.past_events,
                                    extracted_note_obj.extracted_note,
                                    past_grading_judge,
                                    azure_provider,
                                    token_tracker,
                                )
                            past_grade = judged_past_grading.grade
                            if aggregated_temporality.past_events:
                                event_evidence["past_grading"]["evidence"].extend(
                                    aggregated_temporality.past_events[:3]
                                )

                        if (
                            aggregated_temporality.current_events
                            and current_graders
                            and current_grading_judge
                        ):
                            current_grading_results = await run_parallel_grading(
                                event_type,
                                "current",
                                aggregated_temporality.current_events,
                                extracted_note_obj.extracted_note,
                                current_graders,
                                azure_provider,
                                token_tracker,
                            )
                            # Judge current grading results or use first result for ablation_no_judge
                            if (
                                prompt_variant == "ablation_no_judge"
                                and current_grading_results
                            ):
                                logger.debug(
                                    f"[RequestID: {request_id}] ABLATION STUDY: Skipping current grading judge for {event_type}, using first grader result"
                                )
                                # Use the first grading result directly
                                first_result = current_grading_results[0]

                                # Convert to AggregatedGrading format (expected by downstream functions)
                                judged_current_grading = AggregatedGrading(
                                    event_name=event_type,
                                    grade=first_result.grade,
                                    temporal_context="current",
                                    rationale=first_result.rationale,
                                    evidence_snippets=first_result.evidence_snippets,
                                )
                            else:
                                # Standard workflow - use the judge to evaluate results
                                judged_current_grading = await judge_grading(
                                    event_type,
                                    "current",
                                    current_grading_results,
                                    aggregated_temporality.current_events,
                                    extracted_note_obj.extracted_note,
                                    current_grading_judge,
                                    azure_provider,
                                    token_tracker,
                                )
                            current_grade = judged_current_grading.grade
                            if aggregated_temporality.current_events:
                                event_evidence["current_grading"]["evidence"].extend(
                                    aggregated_temporality.current_events[:3]
                                )

                        attribution_agent = agents.get("attribution")
                        certainty_agent = agents.get("certainty")
                        (
                            past_attribution,
                            past_certainty,
                            current_attribution,
                            current_certainty,
                        ) = (0, 0, 0, 0)

                        if attribution_agent and past_grade > 0:
                            past_attribution = 1
                            event_evidence["attribution"]["evidence"].append(
                                f"Past {event_type} graded {past_grade} attributed."
                            )
                        if certainty_agent and past_grade > 0:
                            past_certainty = 3
                            event_evidence["certainty"]["evidence"].append(
                                f"Past {event_type} graded {past_grade} certainty moderate."
                            )
                        if attribution_agent and current_grade > 0:
                            current_attribution = 1
                            event_evidence["attribution"]["evidence"].append(
                                f"Current {event_type} graded {current_grade} attributed."
                            )
                        if certainty_agent and current_grade > 0:
                            current_certainty = 3
                            event_evidence["certainty"]["evidence"].append(
                                f"Current {event_type} graded {current_grade} certainty moderate."
                            )

                        if past_grade > 0 or current_grade > 0:
                            overview_parts = [f"## {event_type.capitalize()} Overview"]
                            if past_grade > 0:
                                overview_parts.append(
                                    f"- Past: Grade {past_grade}, Attr {past_attribution}, Cert {past_certainty}"
                                )
                            if current_grade > 0:
                                overview_parts.append(
                                    f"- Current: Grade {current_grade}, Attr {current_attribution}, Cert {current_certainty}"
                                )
                            if event_evidence["identification"]["evidence"]:
                                overview_parts.append(
                                    f"- Key ID Evidence: {event_evidence['identification']['evidence'][0]}"
                                )
                            if event_evidence["temporality"]["evidence"]:
                                overview_parts.append(
                                    f"- Key Temp Evidence: {event_evidence['temporality']['evidence'][0]}"
                                )
                            user_overview = "\n".join(overview_parts)
                            user_overviews.append(
                                f"{event_type.capitalize()}: {user_overview}"
                            )
                            setattr(
                                enhanced_prediction,
                                f"{event_type_lower}_user_overview",
                                user_overview,
                            )

                        reasoning += f"{event_type.capitalize()} (split temporality): Past={bool(aggregated_temporality.past_events)}, Current={bool(aggregated_temporality.current_events)}. {aggregated_temporality.reasoning or ''}\n"
                        event_result = {
                            "event_type": event_type,
                            "past_grade": past_grade,
                            "current_grade": current_grade,
                            "grade": max(past_grade, current_grade),
                            "past_attribution": past_attribution,
                            "current_attribution": current_attribution,
                            "past_certainty": past_certainty,
                            "current_certainty": current_certainty,
                            "reasoning": aggregated_temporality.reasoning,
                        }
                    else:
                        event_result = create_no_event_result(
                            event_type, "missing temporality agents"
                        )
                        reasoning += f"{event_type.capitalize()} (split): Missing temporality agents.\n"
                else:
                    event_result = create_no_event_result(
                        event_type, "no events identified after identification stage"
                    )
                    reasoning += (
                        f"{event_type.capitalize()} (split): No events identified.\n"
                    )
            else:
                event_result = create_no_event_result(
                    event_type, "missing identification agents"
                )
                reasoning += f"{event_type.capitalize()} (split): Missing ID agents.\n"

            setattr(enhanced_prediction, f"{event_type_lower}_evidence", event_evidence)
            all_evidence[event_type_lower] = event_evidence

            # Set the specific fields we know exist, avoiding problematic field names
            setattr(
                enhanced_prediction,
                f"{event_type_lower}_past_grade",
                event_result.get("past_grade", 0),
            )
            setattr(
                enhanced_prediction,
                f"{event_type_lower}_current_grade",
                event_result.get("current_grade", 0),
            )
            setattr(
                enhanced_prediction,
                f"{event_type_lower}_past_attribution",
                event_result.get("past_attribution", 0),
            )
            setattr(
                enhanced_prediction,
                f"{event_type_lower}_current_attribution",
                event_result.get("current_attribution", 0),
            )
            setattr(
                enhanced_prediction,
                f"{event_type_lower}_past_certainty",
                event_result.get("past_certainty", 0),
            )
            setattr(
                enhanced_prediction,
                f"{event_type_lower}_current_certainty",
                event_result.get("current_certainty", 0),
            )

            # Store the reasoning separately
            if event_result.get("reasoning"):
                reasoning += f"{event_type.capitalize()}: {event_result['reasoning']}\n"

            event_results[event_type] = event_result

        enhanced_prediction.evidence = str(all_evidence)
        enhanced_prediction.reasoning = reasoning.strip()
        enhanced_prediction.user_summary = (
            "\n\n".join(user_overviews)
            if user_overviews
            else "No significant findings."
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        token_usage_data = token_tracker.get_summary()
        total_usage = token_usage_data.get("total", {})
        by_model_usage = token_usage_data.get("by_model", {})

        token_usage_summary = TokenUsageMetadata(
            prompt_tokens=total_usage.get("prompt_tokens", 0),
            completion_tokens=total_usage.get("completion_tokens", 0),
            total_tokens=total_usage.get("total_tokens", 0),
            prompt_cost=total_usage.get("prompt_cost", 0.0),
            completion_cost=total_usage.get("completion_cost", 0.0),
            total_cost=total_usage.get("total_cost", 0.0),
            model_costs=by_model_usage,
        )

        prediction = NotePrediction(
            PMRN=note_input.pmrn,
            NOTE_ID=note_input.note_id,
            NOTE_TEXT=note_input.note_text,
            SHORTENED_NOTE_TEXT=extracted_note_obj.extracted_note,
            MESSAGES=[
                {"role": "user", "content": cleaned_note_text},
                {
                    "role": "assistant",
                    "content": f"Processed in {processing_time:.2f}s with split temporality",
                },
            ],
            PREDICTION=enhanced_prediction.__dict__,
            PROCESSED_AT=datetime.now(),
            PROCESSING_TIME=processing_time,
            MODEL_NAME=model_config,
            TOKEN_USAGE=token_usage_summary,
            ERROR=None,
        )
        logger.info(
            f"[RequestID: {request_id}] Note {note_input.note_id} processed. Tokens: {token_usage_summary.total_tokens}, Cost: ${token_usage_summary.total_cost:.4f}"
        )
        return prediction

    except Exception as e:
        logger.error(
            f"[RequestID: {request_id}] Error in process_note_with_split_temporality for note {note_input.note_id}: {str(e)}",
            exc_info=True,
        )
        processing_time = (
            (datetime.now() - start_time).total_seconds()
            if "start_time" in locals()
            else 0.0
        )
        token_usage_data = token_tracker.get_summary() if token_tracker else {}
        total_usage = token_usage_data.get("total", {})
        by_model_usage = token_usage_data.get("by_model", {})
        error_token_usage = TokenUsageMetadata(
            prompt_tokens=total_usage.get("prompt_tokens", 0),
            completion_tokens=total_usage.get("completion_tokens", 0),
            total_tokens=total_usage.get("total_tokens", 0),
            prompt_cost=total_usage.get("prompt_cost", 0.0),
            completion_cost=total_usage.get("completion_cost", 0.0),
            total_cost=total_usage.get("total_cost", 0.0),
            model_costs=by_model_usage,
        )
        return NotePrediction(
            PMRN=note_input.pmrn,
            NOTE_ID=note_input.note_id,
            NOTE_TEXT=note_input.note_text,
            SHORTENED_NOTE_TEXT="",
            MESSAGES=[],
            PREDICTION={},
            PROCESSED_AT=datetime.now(),
            PROCESSING_TIME=processing_time,
            MODEL_NAME=model_config,
            TOKEN_USAGE=error_token_usage,
            ERROR=str(e),
        )


def create_no_event_result(event_type: str, reason: str) -> Dict[str, Any]:
    """Create a standard result for when an event cannot be processed."""
    return {
        "event_type": event_type,
        "reasoning": f"Unable to process {event_type}: {reason}",
        "past_grade": 0,
        "current_grade": 0,
        "past_attribution": 0,
        "current_attribution": 0,
        "past_certainty": 0,
        "current_certainty": 0,
    }


async def process_all_notes_with_split_temporality(
    notes: List[NoteInput],
    model_config: str = "default",
    azure_provider=None,
    request_id: str = None,
    event_types: Optional[List[str]] = None,
    provider_config: Optional[Dict[str, Any]] = None,
    batch_size: int = 1,
    max_workers: int = 1,
) -> List[NotePrediction]:
    """
    Process multiple notes using the split temporality workflow.

    Args:
        notes: List of clinical notes to process
        model_config: Configuration for which models to use
        azure_provider: Optional Azure provider for authentication
        request_id: Unique request ID for thread tracking
        event_types: Optional list of event types to process
        provider_config: Optional configuration for model providers
        batch_size: Number of notes to process in each batch
        max_workers: Maximum number of concurrent workers

    Returns:
        List[NotePrediction]: List of processed note predictions
    """
    if not request_id:
        request_id = str(uuid.uuid4())

    logger.info(
        f"[RequestID: {request_id}] Processing {len(notes)} notes with split temporality workflow"
    )

    # Create token tracker for this batch
    token_tracker = TokenUsageTracker()

    note_predictions = []
    processed = []

    # Process notes in batches
    for i in range(0, len(notes), batch_size):
        batch = notes[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(notes) + batch_size - 1) // batch_size

        logger.info(
            f"[RequestID: {request_id}] Processing batch {batch_num}/{total_batches} ({len(batch)} notes)"
        )

        # Process each note in the batch
        batch_tasks = []
        for note in batch:
            task = process_note_with_split_temporality(
                note_input=note,
                model_config=model_config,
                azure_provider=azure_provider,
                request_id=request_id,
                token_tracker=token_tracker,
                event_types=event_types,
                provider_config=provider_config,
            )
            batch_tasks.append(task)

        # Wait for all tasks in this batch to complete
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Process results
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(
                    f"[RequestID: {request_id}] Error processing note {batch[j].note_id}: {str(result)}"
                )
                # Create error prediction
                error_obj = NotePrediction(
                    PMRN=batch[j].pmrn,
                    NOTE_ID=batch[j].note_id,
                    NOTE_TEXT=batch[j].note_text,
                    SHORTENED_NOTE_TEXT="",
                    MESSAGES=[],
                    PREDICTION={},
                    PROCESSED_AT=datetime.now(),
                    PROCESSING_TIME=0.0,
                    MODEL_NAME=model_config,
                    TOKEN_USAGE=TokenUsageMetadata(
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        prompt_cost=0.0,
                        completion_cost=0.0,
                        total_cost=0.0,
                        model_costs={},
                    ),
                    ERROR=str(result),
                )
                note_predictions.append(error_obj)
            else:
                note_predictions.append(result)

        # Add the processed batch to our tracking list
        processed.extend(batch)

    logger.info(
        f"[RequestID: {request_id}] Completed processing {len(notes)} notes with split temporality workflow"
    )
    return note_predictions
