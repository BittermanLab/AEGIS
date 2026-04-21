"""
Event processor module for enhanced parallel workflow with judge-based evaluation.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, cast
import os
import json
import re

# Import from openai-agents package
from agents import Agent, Runner, RunConfig, set_tracing_disabled

# Get the event context from our local module
from ..event_context import EventContext

# Import models
from ..models.enhanced_output_models import (
    EventIdentification,
    TemporalityClassification,
    TemporalIdentification,
    AttributionDetection,
    CertaintyAssessment,
    EventGrading,
    EventResult,
)

from ..models.enhanced_judge_models import (
    EventIdentificationResult,
    TemporalityResult,
    AggregatedEventIdentification,
    AggregatedTemporality,
    IdentificationResult,
    AggregatedIdentification,
    GradingResult,
    AggregatedGrading,
    MetaJudgeFeedback,
    EnhancedEventProcessingResult,
    EnhancedEventResult,
)

# Import utility functions
from ..utils.ctcae_utils import (
    get_ctcae_subset,
    format_ctcae_grades_for_prompt,
    get_terms_definitions_and_grades,
    get_terms_only,
    get_terms_and_grades_only,
)

logger = logging.getLogger(__name__)

# Set httpx logger to WARNING level to completely suppress HTTP request logs
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

# Disable tracing for all OpenAI agent executions
set_tracing_disabled(True)
logger.info("OpenAI Agents tracing disabled in event processor module")


# Add a utility function for sanitizing strings for JSON serialization
def sanitize_for_json(text: Any) -> Any:
    """Sanitize a value to ensure it's safe for JSON serialization.

    Args:
        text: The text or value to sanitize

    Returns:
        The sanitized value that's safe for JSON serialization
    """
    if text is None:
        return None

    if isinstance(text, (int, float, bool)):
        return text

    if isinstance(text, str):
        # Replace carriage returns, tabs, and other control characters
        # that can cause JSON parsing issues
        cleaned_text = text

        # Replace all control characters with spaces
        cleaned_text = re.sub(r"[\x00-\x1F\x7F]", " ", cleaned_text)

        # Replace multiple spaces with a single space
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)

        # Trim leading/trailing spaces
        cleaned_text = cleaned_text.strip()

        return cleaned_text

    if isinstance(text, list):
        return [sanitize_for_json(item) for item in text]

    if isinstance(text, dict):
        return {k: sanitize_for_json(v) for k, v in text.items()}

    # For other types, convert to string and sanitize
    return sanitize_for_json(str(text))


def flatten_evidence_list(evidence: Any) -> List[str]:
    """Flatten and sanitize an evidence list to ensure it contains only strings.

    This function handles nested lists, dictionaries, and other complex evidence structures,
    converting them to a flat list of sanitized string evidence items.

    Args:
        evidence: The evidence data which could be a string, list, dict, or other type

    Returns:
        A flat list of sanitized string evidence items
    """
    # Handle None case
    if evidence is None:
        return []

    # If it's already a string, sanitize it and return as a single-item list
    if isinstance(evidence, str):
        return [sanitize_for_json(evidence)]

    # If it's a list, process each item
    if isinstance(evidence, list):
        result = []
        for item in evidence:
            # If item is a string, add it directly
            if isinstance(item, str):
                result.append(sanitize_for_json(item))
            # If item is another list, flatten it recursively
            elif isinstance(item, list):
                result.extend(flatten_evidence_list(item))
            # If item is a dict, convert to string
            elif isinstance(item, dict):
                result.append(sanitize_for_json(str(item)))
            # For other types, convert to string
            else:
                result.append(sanitize_for_json(str(item)))
        return result

    # If it's a dict, convert to string and return as a single-item list
    if isinstance(evidence, dict):
        return [sanitize_for_json(str(evidence))]

    # For any other type, convert to string and return as a single-item list
    return [sanitize_for_json(str(evidence))]


def sanitize_json_evidence(evidence_text):
    """Sanitize evidence text to ensure valid JSON and prevent parsing errors."""
    if not evidence_text:
        return []

    # If it's already a list, flatten and sanitize it
    if isinstance(evidence_text, list):
        return flatten_evidence_list(evidence_text)

    # If it's a string, clean it up
    if isinstance(evidence_text, str):
        # Replace problematic characters
        sanitized = evidence_text.replace("\r", " ").replace("\n", " ")

        # Handle common malformed patterns
        if sanitized.startswith('":['):
            sanitized = "[" + sanitized[3:]

        # Try to parse as JSON if it looks like JSON
        try:
            if sanitized.startswith("[") or sanitized.startswith("{"):
                parsed = json.loads(sanitized)
                return flatten_evidence_list(parsed)
        except:
            pass

        # Return as a single item list
        return [sanitized]

    # For any other type, convert to string and return as list
    return [str(evidence_text)]


def join_safe(items, sep=", "):
    """Join any iterable into a string after str() + sanitize."""
    return sep.join(str(sanitize_for_json(x)) for x in (items or []))


# Add a function to log agent outputs clearly
def log_agent_output(
    stage: str, agent_type: str, agent_idx: int, result: Any, event_type: str = None
):
    """
    Log the complete output of an agent for debugging purposes.

    Args:
        stage: The processing stage (identification, grading, etc.)
        agent_type: Type of agent (identifier, grader, judge, etc.)
        agent_idx: Index of the agent if there are multiple
        result: The agent result object
        event_type: Optional event type being processed
    """
    event_info = f" for {event_type}" if event_type else ""
    logger.debug(f"===== {stage} | {agent_type} {agent_idx}{event_info} =====")

    # Log final output
    if hasattr(result, "final_output") and result.final_output:
        logger.debug(f"Final output: {result.final_output}")

        # Log specific fields based on output type
        if hasattr(result.final_output, "past_events"):
            logger.debug(f"Past events: {result.final_output.past_events}")
        if hasattr(result.final_output, "current_events"):
            logger.debug(f"Current events: {result.final_output.current_events}")
        if hasattr(result.final_output, "grade"):
            logger.debug(f"Grade: {result.final_output.grade}")
        if hasattr(result.final_output, "attribution"):
            logger.debug(f"Attribution: {result.final_output.attribution}")
        if hasattr(result.final_output, "certainty"):
            logger.debug(f"Certainty: {result.final_output.certainty}")
        if hasattr(result.final_output, "rationale") or hasattr(
            result.final_output, "reasoning"
        ):
            reasoning = getattr(
                result.final_output,
                "rationale",
                getattr(result.final_output, "reasoning", "No reasoning provided"),
            )
            logger.debug(f"Reasoning: {reasoning}")
        if hasattr(result.final_output, "evidence_snippets"):
            logger.debug(f"Evidence snippets: {result.final_output.evidence_snippets}")
        if hasattr(result.final_output, "evidence"):
            logger.debug(f"Evidence: {result.final_output.evidence}")

    # Try to log other properties if final_output doesn't exist
    else:
        logger.debug(f"Raw result: {result}")

    logger.debug("=" * 50)


# Custom debug logging function for evidence tracking
def log_evidence(
    stage: str, event_type: str, evidence_data: Any, note_text: str = None
):
    """
    Log evidence data at various stages of processing to track potential issues.

    Args:
        stage: Processing stage name
        event_type: Type of event being processed
        evidence_data: Evidence data to log
        note_text: Optional note text to check evidence against
    """
    try:
        # Create a consistent log prefix
        prefix = f"[EVIDENCE_DEBUG][{event_type}][{stage}]"

        # Log evidence data
        if isinstance(evidence_data, list):
            logger.debug(f"{prefix} Evidence list with {len(evidence_data)} items")
            for i, item in enumerate(evidence_data):
                logger.debug(f"{prefix} Item {i}: {item}")

                # If note text is provided, check if evidence is actually in the note
                if note_text and isinstance(item, str):
                    if item in note_text:
                        logger.debug(f"{prefix} ✓ Item {i} FOUND in note text")
                    else:
                        logger.debug(
                            f"{prefix} ✗ Item {i} NOT FOUND in note text - POTENTIAL FABRICATION"
                        )
        elif isinstance(evidence_data, dict):
            logger.debug(
                f"{prefix} Evidence dict with keys: {list(evidence_data.keys())}"
            )
            for key, values in evidence_data.items():
                logger.debug(
                    f"{prefix} Category '{key}' with {len(values) if isinstance(values, list) else 1} items"
                )

                if isinstance(values, list):
                    for i, item in enumerate(values):
                        logger.debug(f"{prefix} [{key}] Item {i}: {item}")

                        # Check if evidence is in note text
                        if note_text and isinstance(item, str):
                            if item in note_text:
                                logger.debug(
                                    f"{prefix} ✓ [{key}] Item {i} FOUND in note text"
                                )
                            else:
                                logger.debug(
                                    f"{prefix} ✗ [{key}] Item {i} NOT FOUND in note text - POTENTIAL FABRICATION"
                                )
                else:
                    logger.debug(f"{prefix} [{key}] Value: {values}")

                    # Check if evidence is in note text
                    if note_text and isinstance(values, str):
                        if values in note_text:
                            logger.debug(f"{prefix} ✓ [{key}] Value FOUND in note text")
                        else:
                            logger.debug(
                                f"{prefix} ✗ [{key}] Value NOT FOUND in note text - POTENTIAL FABRICATION"
                            )
        else:
            logger.debug(f"{prefix} Evidence (non-list/dict): {evidence_data}")

            # Check if evidence is in note text
            if note_text and isinstance(evidence_data, str):
                if evidence_data in note_text:
                    logger.debug(f"{prefix} ✓ Evidence FOUND in note text")
                else:
                    logger.debug(
                        f"{prefix} ✗ Evidence NOT FOUND in note text - POTENTIAL FABRICATION"
                    )
    except Exception as e:
        logger.error(f"Error in log_evidence function: {str(e)}")


async def format_identification_results(
    identification_results: List[IdentificationResult],
) -> str:
    """Formats identification results clearly for the judge agent."""
    try:
        formatted_results = []
        for result in identification_results:
            past_events = (
                "\n".join([f"- {event}" for event in result.past_events]) or "- None"
            )
            current_events = (
                "\n".join([f"- {event}" for event in result.current_events]) or "- None"
            )
            snippets = result.evidence_snippets or []
            evidence_snippets = (
                "\n".join([f"- {snippet}" for snippet in snippets]) or "- None"
            )

            formatted_result = (
                f"""**Identifier ID**: {result.identifier_id}

"""
                f"**Past Events**:\n{past_events}\n\n"
                f"**Current Events**:\n{current_events}\n\n"
                f"**Evidence Snippets**:\n{evidence_snippets}\n\n"
                f"**Reasoning**:\n{result.reasoning}\n"
            )

            formatted_results.append(formatted_result)

        results_text = "\n\n---\n\n".join(formatted_results)

    except Exception as e:
        logging.error(f"Error formatting identification results: {str(e)}")
        results_text = "Error formatting results"

    return results_text


async def judge_identification(
    event_type: str,
    identification_results: List[IdentificationResult],
    note_text: str,
    judge_agent: Agent,
    azure_provider=None,
    token_tracker=None,
) -> AggregatedIdentification:
    """
    Judge multiple identification results to select the best identification

    Args:
        event_type: The type of event being identified
        identification_results: List of identification results from different agents
        note_text: The extracted note text
        judge_agent: The judge agent to evaluate the results
        azure_provider: Optional Azure provider for authentication
        token_tracker: Optional token usage tracker

    Returns:
        AggregatedIdentification: The best identification result selected by the judge
    """
    logger.debug(
        f"Starting identification judging for {event_type} with {len(identification_results)} results to evaluate"
    )

    # Create a formatted string of all identification results
    try:
        results_text = await format_identification_results(identification_results)

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        results_text = "Error formatting results"

    # Create input for the judge agent
    judge_input = f"""
    
    ### EVENT TYPE: {event_type}
    
    ### PATIENT NOTE:
    {note_text}
    
    ### IDENTIFICATION RESULTS:
    {results_text}
   
    ### FINAL REMINDER:
    Make sure the final list is correct based on the patient note for each specific temporality.
    """

    # Set up run configuration with Azure provider if specified
    run_config = None
    if azure_provider:
        run_config = RunConfig(model_provider=azure_provider)
        logger.debug(f"Using Azure provider for {event_type} identification judging")

    # Create base context for the identification judge
    try:
        # Get CTCAE subset for this event type
        event_type_lower = event_type.lower()
        event_subset = get_ctcae_subset(event_type_lower)

        # Get the terms and definitions for this event type
        all_terms_data = get_terms_definitions_and_grades(event_type_lower)

        # Format only term names for identification judge context (no definitions to reduce tokens)
        all_definitions = []
        for term, term_data in all_terms_data.items():
            # Add only the term name, not the definition
            all_definitions.append(term)

        # Join all term names
        combined_definitions = "\n".join(all_definitions)

        # Create event context with term names only for identification judge
        base_context = EventContext(
            event_type=event_type,
            event_definition=combined_definitions,
            request_id=getattr(token_tracker, "request_id", None),
        )

        logger.debug(
            f"Created context for {event_type} identification judge with {len(all_definitions)} term names"
        )
    except Exception as e:
        # This will now raise an error because of our validation in EventContext
        logger.error(f"Error creating event context for identification judge: {str(e)}")
        raise

    # Run the judge agent with context
    try:
        judge_result = await Runner.run(
            judge_agent,
            judge_input,
            context=base_context,  # Pass base context to the judge
            run_config=run_config,
        )

        # Log the complete judge output
        log_agent_output("IDENTIFICATION_JUDGING", "Judge", 0, judge_result, event_type)

    except Exception as e:
        logging.error(f"Unexpected error in judge identification: {str(e)}")
        raise

    # Track token usage if tracker provided
    if token_tracker:
        agent_name = f"{event_type.lower()}_identification_judge"
        # Get the model name from the judge agent
        model_obj = getattr(judge_agent, "model", None)
        # Extract the actual model name from the model object
        if model_obj and hasattr(model_obj, "model"):
            model_name = model_obj.model  # This is the deployment name
        else:
            model_name = str(model_obj) if model_obj else None

        # Add debug logging for token extraction
        if hasattr(token_tracker, "debug_token_extraction"):
            token_tracker.debug_token_extraction(judge_result)

        token_tracker.track_usage(agent_name, judge_result, model_name)

    # Extract the judge's decision
    judged_identification = judge_result.final_output

    # Log the raw judge output for debugging
    logger.debug(f"Raw identification judge output for {event_type}: {judged_identification}")
    logger.debug(f"Judge output type: {type(judged_identification)}")

    # Create a new AggregatedIdentification instance if needed
    if not isinstance(judged_identification, AggregatedIdentification):
        # Get attributes from the output (handle both dict and object types)
        if isinstance(judged_identification, dict):
            past_events = judged_identification.get("past_events", [])
            current_events = judged_identification.get("current_events", [])
            reasoning = judged_identification.get("reasoning", None)
            evidence_snippets = judged_identification.get("evidence_snippets", None)
        else:
            past_events = getattr(judged_identification, "past_events", [])
            current_events = getattr(judged_identification, "current_events", [])
            reasoning = getattr(judged_identification, "reasoning", None)
            evidence_snippets = getattr(judged_identification, "evidence_snippets", None)

        judged_identification = AggregatedIdentification(
            past_events=past_events,
            current_events=current_events,
            reasoning=reasoning,
            evidence_snippets=evidence_snippets,
        )

    # Collect evidence snippets if not already provided
    if not judged_identification.evidence_snippets:
        all_evidence = []
        for result in identification_results:
            if result.evidence_snippets:
                all_evidence.extend(result.evidence_snippets)

        # Remove duplicates while preserving order
        unique_evidence = []
        for evidence in all_evidence:
            if evidence not in unique_evidence:
                unique_evidence.append(evidence)

        if unique_evidence:
            judged_identification.evidence_snippets = unique_evidence

    # Sanitize evidence snippets before returning
    if judged_identification.evidence_snippets:
        judged_identification.evidence_snippets = sanitize_for_json(
            judged_identification.evidence_snippets
        )

    logger.debug(f"Completed identification judging for {event_type}")
    logger.debug(
        f"Judge selected: past events={judged_identification.past_events}, current events={judged_identification.current_events}"
    )

    return judged_identification


async def run_parallel_grading(
    event_type: str,
    temporal_context: str,
    events: List[str],
    note_text: str,
    grader_agents: List[Agent],
    azure_provider=None,
    token_tracker=None,
    identification_evidence: Optional[List[str]] = None,
) -> List[GradingResult]:
    """
    Run multiple grader agents in parallel to grade events

    Args:
        event_type: The type of event to grade
        temporal_context: Temporal context for grading ("past" or "current")
        events: The list of events to grade
        note_text: The extracted note text
        grader_agents: List of grader agents to run in parallel
        azure_provider: Optional Azure provider for authentication
        token_tracker: Optional token usage tracker
        identification_evidence: Optional evidence snippets from identification stage

    Returns:
        List[GradingResult]: List of grading results from different agents
    """
    logger.debug(
        f"Starting parallel grading for {event_type} ({temporal_context}) with {len(grader_agents)} agents"
    )
    logger.debug(f"Events to grade: {events}")

    # Get the specific subset data for this event type for grading
    event_type_lower = event_type.lower()
    event_subset = get_ctcae_subset(event_type_lower)

    # Get the grading criteria
    try:
        grading_criteria = format_ctcae_grades_for_prompt(
            event_subset, event_type_lower
        )
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        grading_criteria = "Error loading grading criteria"

    # Get the main term that matches the event type
    main_term = None
    for key in event_subset:
        category_data = event_subset[key]
        for term in category_data:
            if term.lower() == event_type_lower:
                main_term = term
                break
        if main_term:
            break

    # Get term data and definition
    base_definition = ""
    if main_term:
        for key in event_subset:
            if main_term in event_subset[key]:
                # Get the term data to extract definition
                term_data = event_subset[key][main_term]
                base_definition = term_data.get("Definition", "")
                break

    # Create event context with the relevant information
    try:
        # Get CTCAE subset for this event type
        event_type_lower = event_type.lower()
        event_subset = get_ctcae_subset(event_type_lower)

        # Get the terms and definitions for this event type
        all_terms_data = get_terms_definitions_and_grades(event_type_lower)

        # Format only term names for grading context (no definitions needed)
        all_definitions = []
        for term, term_data in all_terms_data.items():
            # Add only the term name for grading agents
            all_definitions.append(term)

        # Join all term names
        combined_definitions = "\n".join(all_definitions)

        # Prepare formatted grading criteria
        formatted_grades = format_ctcae_grades_for_prompt(
            event_subset, event_type_lower
        )

        # Create the event context with CTCAE data
        base_context = EventContext(
            event_type=event_type,
            event_definition=combined_definitions,
            grading_criteria=formatted_grades,
            temporal_context=temporal_context,
            request_id=getattr(token_tracker, "request_id", None),
        )

        logger.debug(
            f"Created context for {event_type} {temporal_context} grading with {len(all_definitions)} term names"
        )
    except Exception as e:
        logger.error(f"Error creating event context for grading: {str(e)}")
        raise

    # Create a formatted string of all events
    events_text = ", ".join(events)

    # Create the common input for all grader agents
    grader_input = f"""
    ### EVENT TYPE
    {event_type}
    
    ### PAST OR CURRENT: 
    {temporal_context} 
    
    ### EVENTS TO GRADE FROM NOTE: 
    {events_text}
    
    ### PATIENT NOTE:
    {note_text}
    
    ### FINAL REMINDER:
    Make sure the final grade is correct based on the grading criteria and evidence in the note for this specific temporality only.

    """

    # Include evidence from identification if available
    if identification_evidence and len(identification_evidence) > 0:
        evidence_text = "\n".join(
            [f"- {evidence}" for evidence in identification_evidence]
        )
        grader_input += f"""
        
        ### IDENTIFICATION AGENT EVIDENCE:
        {evidence_text}
        """

    # Set up run configuration with Azure provider if specified
    run_config = None
    try:
        if azure_provider:
            run_config = RunConfig(model_provider=azure_provider)
            logger.debug(
                f"Using Azure provider for {event_type} {temporal_context} grading"
            )
    except Exception as e:
        logging.error(f"Error setting up Azure provider: {str(e)}")

    # Create tasks for parallel execution
    grading_tasks = []
    for i, agent in enumerate(grader_agents):
        try:
            grading_tasks.append(
                Runner.run(
                    agent,
                    grader_input,
                    context=base_context,  # Pass the event context with temporal context
                    run_config=run_config,
                )
            )
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")

    # Run all grading tasks concurrently
    grading_results = await asyncio.gather(*grading_tasks, return_exceptions=True)

    # Process results
    results = []
    for i, result in enumerate(grading_results):
        if isinstance(result, Exception):
            logger.error(f"Error in grader {i}: {str(result)}")
            continue

        try:
            # Log the complete agent output
            log_agent_output(
                f"{temporal_context.upper()}_GRADING", "Grader", i, result, event_type
            )

            # Track token usage if tracker provided
            if token_tracker:
                agent_name = f"{event_type.lower()}_{temporal_context}_grader_{i}"
                # Get the model name from the grader agent
                model_obj = getattr(grader_agents[i], "model", None)
                # Extract the actual model name from the model object
                if model_obj and hasattr(model_obj, "model"):
                    model_name = model_obj.model  # This is the deployment name
                else:
                    model_name = str(model_obj) if model_obj else None

                # Add debug logging for token extraction
                if hasattr(token_tracker, "debug_token_extraction"):
                    token_tracker.debug_token_extraction(result)

                token_tracker.track_usage(agent_name, result, model_name)

            # Extract the grader output
            grader_output = result.final_output

            # Ensure the temporal context is included
            temporal_context_value = temporal_context
            if hasattr(grader_output, "temporal_context"):
                temporal_context_value = grader_output.temporal_context

            # Get default values for optional fields
            grade = 0
            if hasattr(grader_output, "grade"):
                grade = grader_output.grade

            rationale = ""
            if hasattr(grader_output, "rationale"):
                rationale = grader_output.rationale

            evidence_snippets = None
            if hasattr(grader_output, "evidence_snippets"):
                evidence_snippets = sanitize_for_json(grader_output.evidence_snippets)

            # Create a GradingResult object - simplified for new structure
            try:
                results.append(
                    GradingResult(
                        grader_id=f"grader_{i}",
                        event_name=event_type,
                        grade=grade,
                        temporal_context=temporal_context_value,
                        rationale=rationale,
                        evidence_snippets=evidence_snippets,
                    )
                )
            except Exception as e:
                logging.error(f"Unexpected error creating GradingResult: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing grader result {i}: {str(e)}")

    logger.debug(
        f"Completed parallel grading for {event_type} ({temporal_context}), got {len(results)} valid results"
    )
    return results


async def format_grading_results(
    grading_results: List[GradingResult],
) -> str:
    """Formats grading results clearly for the judge agent."""
    try:
        results_text = "\n\n".join(
            [
                f"**Grader**: {result.grader_id}\n"
                f"**Grade**: {result.grade}\n"
                f"**Rationale**: {sanitize_for_json(result.rationale) or 'No rationale provided'}"
                for result in grading_results
            ]
        )
    except Exception as e:
        logging.error(f"Error formatting grading results: {str(e)}")
        results_text = "Error formatting results"

    return results_text


async def judge_grading(
    event_type: str,
    temporal_context: str,
    grading_results: List[GradingResult],
    events: List[str],
    note_text: str,
    judge_agent: Agent,
    azure_provider=None,
    token_tracker=None,
) -> AggregatedGrading:
    """
    Judge multiple grading results to select the best grade

    Args:
        event_type: The type of event being graded
        temporal_context: Whether this is a "past" or "current" event
        grading_results: List of grading results from different agents
        events: List of events that were graded
        note_text: The extracted note text
        judge_agent: The judge agent to evaluate the results
        azure_provider: Optional Azure provider for authentication
        token_tracker: Optional token usage tracker

    Returns:
        AggregatedGrading: The best grading result selected by the judge
    """
    logger.debug(
        f"Starting grading judging for {event_type} ({temporal_context}) with {len(grading_results)} results to evaluate"
    )

    # Create a formatted string of all grading results - simplified to remove confidence
    try:
        results_text = await format_grading_results(grading_results)

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        results_text = "Error formatting results"

    # Format the events list
    events_list = ", ".join(events)

    # Create input for the judge agent
    judge_input = f"""
    ### EVENT TYPE: 
    {event_type}
    
    ### PAST OR PRESENT: 
    {temporal_context.capitalize()}
    
    ### EVENTS TO GRADE:
    {events_list}
    
    ### PATIENT NOTE:
    {note_text}
    
    ### GRADER RESULTS TO EVALUTATE:
    {results_text}
    
    ### FINAL REMINDER:
    Make sure the final grade is correct based on the grading criteria, evidence in the note and recommendations from graders.
    """

    # Set up run configuration with Azure provider if specified
    run_config = None
    try:
        if azure_provider:
            run_config = RunConfig(model_provider=azure_provider)
            logger.debug(
                f"Using Azure provider for {event_type} {temporal_context} grading judging"
            )
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")

    # Create grading context with temporal information
    try:
        # Get CTCAE subset for this event type
        event_type_lower = event_type.lower()

        # Get just the term names for grading judge (no definitions to reduce tokens)
        all_terms = get_terms_only(event_type_lower)
        combined_definitions = "\n".join(all_terms)

        # Get grading criteria for the grading judge
        event_subset = get_ctcae_subset(event_type_lower)
        formatted_grades = format_ctcae_grades_for_prompt(
            event_subset, event_type_lower
        )

        grading_context = EventContext(
            event_type=event_type,
            event_definition=combined_definitions,
            grading_criteria=formatted_grades,
            temporal_context=temporal_context,
            request_id=getattr(token_tracker, "request_id", None),
        )

        logger.debug(
            f"Created grading judge context for {event_type} {temporal_context} with {len(all_terms)} term names and grading criteria"
        )
    except Exception as e:
        logger.error(f"Error creating grading judge context: {str(e)}")
        raise

    # Run the judge agent with context
    try:
        judge_result = await Runner.run(
            judge_agent,
            judge_input,
            context=grading_context,  # Pass temporal context to the judge
            run_config=run_config,
        )

        # Log the complete judge output
        log_agent_output(
            f"{temporal_context.upper()}_GRADING_JUDGING",
            "Judge",
            0,
            judge_result,
            event_type,
        )

    except Exception as e:
        logger.error(f"Error in grading judge: {str(e)}")
        raise

    # Track token usage if tracker provided
    if token_tracker:
        agent_name = f"{event_type.lower()}_{temporal_context}_grading_judge"
        # Get the model name from the judge agent
        model_obj = getattr(judge_agent, "model", None)
        # Extract the actual model name from the model object
        if model_obj and hasattr(model_obj, "model"):
            model_name = model_obj.model  # This is the deployment name
        else:
            model_name = str(model_obj) if model_obj else None

        # Add debug logging for token extraction
        if hasattr(token_tracker, "debug_token_extraction"):
            token_tracker.debug_token_extraction(judge_result)

        token_tracker.track_usage(agent_name, judge_result, model_name)

    # Extract the judge's decision
    judged_grading = judge_result.final_output

    # Log the raw judge output for debugging
    logger.debug(f"Raw grading judge output for {event_type} ({temporal_context}): {judged_grading}")
    logger.debug(f"Judge output type: {type(judged_grading)}")

    # Create a proper AggregatedGrading instance if needed - simplified for new structure
    if not isinstance(judged_grading, AggregatedGrading):
        # Extract attributes from the output (handle both dict and object types)
        if isinstance(judged_grading, dict):
            grade = judged_grading.get("grade", 0)
            rationale = judged_grading.get("rationale", None) or judged_grading.get("reasoning", None)
            evidence_snippets = judged_grading.get("evidence_snippets", None)
            # Check if judge output already has event_name and temporal_context
            judge_event_name = judged_grading.get("event_name", None)
            judge_temporal_context = judged_grading.get("temporal_context", None)
        else:
            grade = getattr(judged_grading, "grade", 0)
            rationale = getattr(judged_grading, "rationale", None) or getattr(judged_grading, "reasoning", None)
            evidence_snippets = getattr(judged_grading, "evidence_snippets", None)
            # Check if judge output already has event_name and temporal_context
            judge_event_name = getattr(judged_grading, "event_name", None)
            judge_temporal_context = getattr(judged_grading, "temporal_context", None)

        # Create a new AggregatedGrading instance with proper values
        # Always provide event_name and temporal_context from function parameters
        try:
            judged_grading = AggregatedGrading(
                event_name=judge_event_name or event_type,  # Use judge's value if provided, otherwise use function parameter
                grade=grade,
                temporal_context=judge_temporal_context or temporal_context,  # Use judge's value if provided, otherwise use function parameter
                rationale=rationale,
                evidence_snippets=evidence_snippets,
            )
        except Exception as e:
            logger.error(f"Failed to create AggregatedGrading for {event_type} ({temporal_context})")
            logger.error(f"Input data: event_name={judge_event_name or event_type}, grade={grade}, temporal_context={judge_temporal_context or temporal_context}")
            logger.error(f"Error: {str(e)}")
            raise
    else:
        # Make sure event_name is set
        if not judged_grading.event_name:
            judged_grading.event_name = event_type

        # Make sure temporal_context is set
        if not judged_grading.temporal_context:
            judged_grading.temporal_context = temporal_context

    # Sanitize evidence snippets before returning
    if judged_grading.evidence_snippets:
        judged_grading.evidence_snippets = sanitize_for_json(
            judged_grading.evidence_snippets
        )

    logger.debug(f"Completed grading judging for {event_type} ({temporal_context})")
    logger.debug(f"Judge selected grade: {judged_grading.grade}")

    return judged_grading


async def evaluate_event_processing(
    event_type: str,
    event_result: Any,
    note_text: str,
    meta_judge_agent: Agent,
    azure_provider=None,
    context: Optional[EventContext] = None,
    token_tracker=None,
) -> MetaJudgeFeedback:
    """
    Evaluate the overall quality of event processing using a meta-judge agent

    Args:
        event_type: The type of event being evaluated
        event_result: The result of event processing
        note_text: The extracted note text
        meta_judge_agent: The meta-judge agent to evaluate the results
        azure_provider: Optional Azure provider for authentication
        context: Optional event context for dynamic instructions
        token_tracker: Optional token usage tracker

    Returns:
        MetaJudgeFeedback: Feedback from the meta-judge
    """
    logger.debug(f"Starting meta-judge evaluation for {event_type}")

    # Format and provide evidence information for the meta-judge to create a comprehensive overview
    structured_evidence = None
    if hasattr(event_result, "evidence") and event_result.evidence:
        structured_evidence = event_result.evidence

    # Create input for the meta-judge agent
    # Check if event_result is a string (direct output from LLM) or an object
    if isinstance(event_result, str):
        # For string result, just use the raw output
        meta_judge_input = f"""
        Event Type: {event_type}
        
        Patient Note:
        {note_text}
        
        Processing Result:
        Raw Output: {event_result}
        """
    else:
        # For object result, use all the available attributes with comprehensive evidence details
        meta_judge_input = f"""
        Event Type: {event_type}
        
        Patient Note:
        {note_text}
        
        Processing Result Summary:
        Grade: {getattr(event_result, 'grade', 0)} (Past: {getattr(event_result, 'past_grade', 0)}, Current: {getattr(event_result, 'current_grade', 0)})
        Attribution: {getattr(event_result, 'attribution', 0)} (Past: {getattr(event_result, 'past_attribution', 0)}, Current: {getattr(event_result, 'current_attribution', 0)})
        Certainty: {getattr(event_result, 'certainty', 0)} (Past: {getattr(event_result, 'past_certainty', 0)}, Current: {getattr(event_result, 'current_certainty', 0)})
        """

        # Add evidence details to help with user overview creation
        if structured_evidence:
            meta_judge_input += "\n\nEvidence Details:\n"

            # Add identification evidence
            if "identification" in structured_evidence:
                meta_judge_input += "\nIdentification Evidence:\n"
                if (
                    "past" in structured_evidence["identification"]
                    and structured_evidence["identification"]["past"]
                ):
                    past_evidence = structured_evidence["identification"]["past"]
                    meta_judge_input += f"- Past: {join_safe(past_evidence)}\n"
                if (
                    "current" in structured_evidence["identification"]
                    and structured_evidence["identification"]["current"]
                ):
                    current_evidence = structured_evidence["identification"]["current"]
                    meta_judge_input += f"- Current: {join_safe(current_evidence)}\n"

            # Add grading evidence
            if "grading" in structured_evidence:
                meta_judge_input += "\nGrading Evidence:\n"
                if (
                    "past" in structured_evidence["grading"]
                    and structured_evidence["grading"]["past"]
                ):
                    past_evidence = structured_evidence["grading"]["past"]
                    meta_judge_input += f"- Past: {join_safe(past_evidence)}\n"
                if (
                    "current" in structured_evidence["grading"]
                    and structured_evidence["grading"]["current"]
                ):
                    current_evidence = structured_evidence["grading"]["current"]
                    meta_judge_input += f"- Current: {join_safe(current_evidence)}\n"

            # Add attribution evidence
            if "attribution" in structured_evidence:
                meta_judge_input += "\nAttribution Evidence:\n"
                if (
                    "past" in structured_evidence["attribution"]
                    and structured_evidence["attribution"]["past"]
                ):
                    past_evidence = structured_evidence["attribution"]["past"]
                    meta_judge_input += f"- Past: {join_safe(past_evidence)}\n"
                if (
                    "current" in structured_evidence["attribution"]
                    and structured_evidence["attribution"]["current"]
                ):
                    current_evidence = structured_evidence["attribution"]["current"]
                    meta_judge_input += f"- Current: {join_safe(current_evidence)}\n"

            # Add certainty evidence
            if "certainty" in structured_evidence:
                meta_judge_input += "\nCertainty Evidence:\n"
                if (
                    "past" in structured_evidence["certainty"]
                    and structured_evidence["certainty"]["past"]
                ):
                    past_evidence = structured_evidence["certainty"]["past"]
                    meta_judge_input += f"- Past: {join_safe(past_evidence)}\n"
                if (
                    "current" in structured_evidence["certainty"]
                    and structured_evidence["certainty"]["current"]
                ):
                    current_evidence = structured_evidence["certainty"]["current"]
                    meta_judge_input += f"- Current: {join_safe(current_evidence)}\n"

    # Get reasoning if available, otherwise use a default message
    reasoning = (
        getattr(event_result, "reasoning", "No reasoning provided")
        if not isinstance(event_result, str)
        else "Raw LLM output, no structured reasoning available"
    )

    meta_judge_input += f"""
    Reasoning:
    {reasoning}
    
    Your tasks are to:
    1. Evaluate the quality of this event processing result and determine if it needs improvement.
    2. Create a concise user overview (3-4 sentences max) that summarizes the findings for {event_type} in this note.
    
    For the user overview:
    - Summarize whether {event_type} was detected (past and/or current)
    - Include the grade and temporality if detected
    - Highlight key evidence that supports the findings
    - Make it clinically accurate but accessible
    """

    # Set up run configuration with Azure provider if specified
    run_config = None
    if (
        azure_provider is not None
        and not isinstance(azure_provider, EventContext)
        and hasattr(azure_provider, "get_model")
    ):
        # Make sure the provider has the required method and isn't accidentally an EventContext
        run_config = RunConfig(model_provider=azure_provider)
        logger.debug(f"Using Azure provider for {event_type} meta-judge evaluation")
    else:
        # Don't use a provider that lacks the get_model method
        logger.warning(
            f"Azure provider for {event_type} meta-judge is not valid, running without provider"
        )

    # Run the meta-judge agent with run configuration if provided
    meta_judge_result = await Runner.run(
        meta_judge_agent,
        meta_judge_input,
        run_config=run_config,
    )

    # Log the complete meta-judge output
    log_agent_output("META_JUDGE", "MetaJudge", 0, meta_judge_result, event_type)

    # Track token usage if tracker provided
    if token_tracker:
        agent_name = f"{event_type.lower()}_meta_judge"
        # Get the model name from the meta judge agent
        model_obj = getattr(meta_judge_agent, "model", None)
        # Extract the actual model name from the model object
        if model_obj and hasattr(model_obj, "model"):
            model_name = model_obj.model  # This is the deployment name
        else:
            model_name = str(model_obj) if model_obj else None
        token_tracker.track_usage(agent_name, meta_judge_result, model_name)

    # Extract the meta-judge's feedback
    meta_judge_feedback = meta_judge_result.final_output

    # Ensure the event type is set
    meta_judge_feedback.event_type = event_type

    # Log the user overview
    if (
        hasattr(meta_judge_feedback, "user_overview")
        and meta_judge_feedback.user_overview
    ):
        logger.debug(
            f"User overview for {event_type}: {meta_judge_feedback.user_overview}"
        )

    logger.debug(f"Completed meta-judge evaluation for {event_type}")
    logger.debug(f"Needs improvement: {meta_judge_feedback.needs_improvement}")

    return meta_judge_feedback


async def process_event_with_judge(
    event_type: str,
    event_agents: Dict[str, Dict[str, Any]],
    note_text: str,
    max_iterations: int = 1,
    azure_provider=None,
    token_tracker=None,
    request_id: str = None,
    prompt_variant: str = "default",
) -> EnhancedEventResult:
    """
    Process a single event type with the enhanced parallel workflow with judge-based evaluation.

    This function coordinates the full workflow for a single event type:
    1. Run multiple identifier agents in parallel
    2. Judge identification results
    3. For each temporal context (past, current):
       a. Run multiple grader agents in parallel
       b. Judge grading results
    4. Run attribution and certainty agents
    5. Generate a user-friendly overview

    Args:
        event_type: The type of event to process
        event_agents: Dictionary of all specialized agents for this event type
        note_text: The extracted note text
        max_iterations: Maximum number of iterations to run
        azure_provider: Optional Azure provider for authentication
        token_tracker: Optional token usage tracker
        request_id: Optional request ID for tracking
        prompt_variant: Which prompt variant to use

    Returns:
        EnhancedEventResult: The final result of event processing
    """
    logger.debug(f"Starting event processing for {event_type}")

    # Get event subset for this event type - needed for several downstream operations
    event_subset = get_ctcae_subset(event_type.lower())

    # Initialize the processing result
    processing_result = EnhancedEventProcessingResult(event_type=event_type)

    # Generate a request ID if not provided
    if not request_id:
        request_id = str(uuid.uuid4())

    # Log the start of event processing with request ID
    logger.debug(
        f"[RequestID: {request_id}] ======= PROCESSING EVENT {event_type} ======="
    )
    logger.debug(
        f"[RequestID: {request_id}] Note text (first 1000 chars): {note_text[:1000]}..."
    )
    logger.debug(f"[RequestID: {request_id}] Max iterations: {max_iterations}")
    logger.debug(f"[RequestID: {request_id}] Prompt variant: {prompt_variant}")

    event_start_time = datetime.now()
    iterations_completed = 0

    try:
        # Get the agents for this event type - using new split workflow
        event_identifiers = event_agents[event_type]["event_identifiers"]
        event_identification_judge = event_agents[event_type]["event_identification_judge"]
        temporality_classifiers = event_agents[event_type]["temporality_classifiers"]
        temporality_judge = event_agents[event_type]["temporality_judge"]
        past_graders = event_agents[event_type]["past_graders"]
        current_graders = event_agents[event_type]["current_graders"]
        past_grading_judge = event_agents[event_type]["past_grading_judge"]
        current_grading_judge = event_agents[event_type]["current_grading_judge"]
        attribution_detector = event_agents[event_type]["attribution"]
        certainty_assessor = event_agents[event_type]["certainty"]
        meta_judge = event_agents[event_type]["meta_judge"]

        logger.debug(f"[RequestID: {request_id}] Loaded agents for {event_type}:")
        logger.debug(
            f"[RequestID: {request_id}] - {len(event_identifiers)} event identifiers"
        )
        logger.debug(
            f"[RequestID: {request_id}] - {len(temporality_classifiers)} temporality classifiers"
        )
        logger.debug(f"[RequestID: {request_id}] - {len(past_graders)} past graders")
        logger.debug(
            f"[RequestID: {request_id}] - {len(current_graders)} current graders"
        )

        # ABLATION STUDY: Agent reduction is handled in note_processor.py
        # Don't duplicate the logic here

        # Process the event with reflection loop
        for iteration in range(max_iterations):
            iterations_completed = iteration + 1
            logger.debug(
                f"[RequestID: {request_id}] Starting iteration {iterations_completed}/{max_iterations} for {event_type}"
            )

            # Set up run configuration with Azure provider and request ID
            run_config = None
            if azure_provider:
                run_config = RunConfig(model_provider=azure_provider)
                logger.debug(
                    f"[RequestID: {request_id}] Using Azure provider for {event_type}"
                )

            # 1. Run parallel event identification (without temporality)
            event_identification_results = await run_parallel_event_identification(
                event_type,
                note_text,
                event_identifiers,
                azure_provider,
                token_tracker,
            )
            processing_result.event_identification_results = event_identification_results

            # Log event identification results for debugging
            for i, result in enumerate(event_identification_results):
                logger.debug(
                    f"[RequestID: {request_id}][DEBUG] Event Identifier {i} results for {event_type}: event_present={result.event_present}"
                )
                if hasattr(result, "evidence_snippets") and result.evidence_snippets:
                    log_evidence(
                        f"[RequestID: {request_id}]EVENT_IDENTIFIER_{i}",
                        event_type,
                        result.evidence_snippets,
                        note_text,
                    )

            # If no valid identification results, return a minimal result
            if not event_identification_results:
                # Create evidence with explanation
                no_identification_message = (
                    f"No valid {event_type} events identified in patient note."
                )
                identification_evidence = [no_identification_message]

                # Log the default evidence
                log_evidence(
                    "[RequestID: {request_id}]NO_IDENTIFICATION",
                    event_type,
                    identification_evidence,
                    note_text,
                )

                return EnhancedEventResult(
                    event_type=event_type,
                    grade=0,
                    past_grade=0,
                    current_grade=0,
                    attribution=0,
                    certainty=0,
                    attribution_evidence=None,
                    certainty_evidence=None,
                    identification_evidence=identification_evidence,
                    past_grading_evidence=None,
                    current_grading_evidence=None,
                    reasoning=f"No valid {event_type} events identified in patient note.",
                    iterations_completed=iterations_completed,
                )

            # 2. Judge event identification results or use first result for ablation_no_judge
            if prompt_variant == "ablation_no_judge" and event_identification_results:
                logger.debug(
                    f"[RequestID: {request_id}] ABLATION STUDY: Skipping identification judge for {event_type}, using first identifier result"
                )
                # Use the first identification result directly
                first_result = event_identification_results[0]
                judged_event_identification = AggregatedEventIdentification(
                    event_present=first_result.event_present,
                    reasoning=first_result.reasoning,
                    evidence_snippets=first_result.evidence_snippets,
                )
            else:
                # Standard workflow - use the judge to evaluate results
                judged_event_identification = await judge_event_identification(
                    event_type,
                    event_identification_results,
                    note_text,
                    event_identification_judge,
                    azure_provider,
                    token_tracker,
                )

            processing_result.judged_event_identification = judged_event_identification

            # Log judged event identification for debugging
            logger.debug(
                f"[RequestID: {request_id}][DEBUG] Judged event identification for {event_type}: event_present={judged_event_identification.event_present}"
            )
            if (
                hasattr(judged_event_identification, "evidence_snippets")
                and judged_event_identification.evidence_snippets
            ):
                log_evidence(
                    "[RequestID: {request_id}]JUDGED_EVENT_IDENTIFICATION",
                    event_type,
                    judged_event_identification.evidence_snippets,
                    note_text,
                )
            
            # If no events identified, return early
            if not judged_event_identification.event_present:
                logger.debug(f"[RequestID: {request_id}] No events identified for {event_type}, returning early")
                return EnhancedEventResult(
                    event_type=event_type,
                    grade=0,
                    past_grade=0,
                    current_grade=0,
                    attribution=0,
                    certainty=0,
                    attribution_evidence=None,
                    certainty_evidence=None,
                    identification_evidence=[f"No {event_type} events identified in patient note."],
                    past_grading_evidence=None,
                    current_grading_evidence=None,
                    reasoning=f"No {event_type} events identified in patient note.",
                    iterations_completed=iterations_completed,
                )
            
            # 3. Run parallel temporality classification
            temporality_results = await run_parallel_temporality_classification(
                event_type,
                note_text,
                temporality_classifiers,
                azure_provider,
                token_tracker,
            )
            processing_result.temporality_results = temporality_results

            # Log temporality classification results for debugging
            for i, result in enumerate(temporality_results):
                logger.debug(
                    f"[RequestID: {request_id}][DEBUG] Temporality Classifier {i} results for {event_type}: past_events={result.past_events}, current_events={result.current_events}"
                )
                if hasattr(result, "evidence_snippets") and result.evidence_snippets:
                    log_evidence(
                        f"[RequestID: {request_id}]TEMPORALITY_CLASSIFIER_{i}",
                        event_type,
                        result.evidence_snippets,
                        note_text,
                    )
            
            # 4. Judge temporality results or use first result for ablation_no_judge
            if prompt_variant == "ablation_no_judge" and temporality_results:
                logger.debug(
                    f"[RequestID: {request_id}] ABLATION STUDY: Skipping temporality judge for {event_type}, using first classifier result"
                )
                # Use the first temporality result directly
                first_result = temporality_results[0]
                judged_temporality = AggregatedTemporality(
                    past_events=first_result.past_events,
                    current_events=first_result.current_events,
                    reasoning=first_result.reasoning,
                    evidence_snippets=first_result.evidence_snippets,
                )
            else:
                # Standard workflow - use the judge to evaluate results
                judged_temporality = await judge_temporality_classification(
                    event_type,
                    temporality_results,
                    note_text,
                    temporality_judge,
                    azure_provider,
                    token_tracker,
                )
            
            processing_result.judged_temporality = judged_temporality
            
            # Log judged temporality for debugging
            logger.debug(
                f"[RequestID: {request_id}][DEBUG] Judged temporality for {event_type}: past_events={judged_temporality.past_events}, current_events={judged_temporality.current_events}"
            )
            if (
                hasattr(judged_temporality, "evidence_snippets")
                and judged_temporality.evidence_snippets
            ):
                log_evidence(
                    "[RequestID: {request_id}]JUDGED_TEMPORALITY",
                    event_type,
                    judged_temporality.evidence_snippets,
                    note_text,
                )

            # Initialize grading tasks
            past_grading_task = None
            current_grading_task = None

            # 5. Run parallel grading for past events if present
            if judged_temporality.past_events:
                past_grading_results = await run_parallel_grading(
                    event_type,
                    "past",
                    judged_temporality.past_events,
                    note_text,
                    past_graders,
                    azure_provider,
                    token_tracker,
                    judged_temporality.evidence_snippets,
                )
                processing_result.past_grading_results = past_grading_results

                # Log past grading results
                for i, result in enumerate(past_grading_results):
                    logger.debug(
                        f"[RequestID: {request_id}][DEBUG] Past grader {i} results for {event_type}: grade={result.grade}, event_name={result.event_name}"
                    )
                    if (
                        hasattr(result, "evidence_snippets")
                        and result.evidence_snippets
                    ):
                        log_evidence(
                            f"[RequestID: {request_id}]PAST_GRADER_{i}",
                            event_type,
                            result.evidence_snippets,
                            note_text,
                        )

                # If valid past grading results, judge them or use first result for ablation_no_judge
                if past_grading_results:
                    if prompt_variant == "ablation_no_judge":
                        logger.debug(
                            f"[RequestID: {request_id}] ABLATION STUDY: Skipping past grading judge for {event_type}, using first grader result"
                        )
                        # Use the first grading result directly
                        first_result = past_grading_results[0]
                        judged_past_grading = AggregatedGrading(
                            event_name=event_type,
                            grade=first_result.grade,
                            temporal_context="past",
                            rationale=first_result.rationale,
                            evidence_snippets=first_result.evidence_snippets,
                        )
                    else:
                        judged_past_grading = await judge_grading(
                            event_type,
                            "past",
                            past_grading_results,
                            judged_temporality.past_events,
                            note_text,
                            past_grading_judge,
                            azure_provider,
                            token_tracker,
                        )
                    processing_result.judged_past_grading = judged_past_grading
                    processing_result.past_grade = judged_past_grading.grade

                    # Log judged past grading
                    logger.debug(
                        f"[RequestID: {request_id}][DEBUG] Judged past grading for {event_type}: grade={judged_past_grading.grade}"
                    )
                    if (
                        hasattr(judged_past_grading, "evidence_snippets")
                        and judged_past_grading.evidence_snippets
                    ):
                        log_evidence(
                            "[RequestID: {request_id}]JUDGED_PAST_GRADING",
                            event_type,
                            judged_past_grading.evidence_snippets,
                            note_text,
                        )

            # 6. Run parallel grading for current events if present
            if judged_temporality.current_events:
                current_grading_results = await run_parallel_grading(
                    event_type,
                    "current",
                    judged_temporality.current_events,
                    note_text,
                    current_graders,
                    azure_provider,
                    token_tracker,
                    judged_temporality.evidence_snippets,
                )
                processing_result.current_grading_results = current_grading_results

                # Log current grading results
                for i, result in enumerate(current_grading_results):
                    logger.debug(
                        f"[RequestID: {request_id}][DEBUG] Current grader {i} results for {event_type}: grade={result.grade}, event_name={result.event_name}"
                    )
                    if (
                        hasattr(result, "evidence_snippets")
                        and result.evidence_snippets
                    ):
                        log_evidence(
                            f"[RequestID: {request_id}]CURRENT_GRADER_{i}",
                            event_type,
                            result.evidence_snippets,
                            note_text,
                        )

                # If valid current grading results, judge them or use first result for ablation_no_judge
                if current_grading_results:
                    if prompt_variant == "ablation_no_judge":
                        logger.debug(
                            f"[RequestID: {request_id}] ABLATION STUDY: Skipping current grading judge for {event_type}, using first grader result"
                        )
                        # Use the first grading result directly
                        first_result = current_grading_results[0]
                        judged_current_grading = AggregatedGrading(
                            event_name=event_type,
                            grade=first_result.grade,
                            temporal_context="current",
                            rationale=first_result.rationale,
                            evidence_snippets=first_result.evidence_snippets,
                        )
                    else:
                        judged_current_grading = await judge_grading(
                            event_type,
                            "current",
                            current_grading_results,
                            judged_temporality.current_events,
                            note_text,
                            current_grading_judge,
                            azure_provider,
                            token_tracker,
                        )
                    processing_result.judged_current_grading = judged_current_grading
                    processing_result.current_grade = judged_current_grading.grade

                    # Log judged current grading
                    logger.debug(
                        f"[RequestID: {request_id}][DEBUG] Judged current grading for {event_type}: grade={judged_current_grading.grade}"
                    )
                    if (
                        hasattr(judged_current_grading, "evidence_snippets")
                        and judged_current_grading.evidence_snippets
                    ):
                        log_evidence(
                            "[RequestID: {request_id}]JUDGED_CURRENT_GRADING",
                            event_type,
                            judged_current_grading.evidence_snippets,
                            note_text,
                        )

            # 5. Determine the maximum grade
            past_grade = (
                0
                if processing_result.past_grade is None
                else processing_result.past_grade
            )
            current_grade = (
                0
                if processing_result.current_grade is None
                else processing_result.current_grade
            )
            max_grade = max(past_grade, current_grade)
            processing_result.grade = max_grade

            # If no events found, return a minimal result
            if max_grade == 0:
                # Create evidence dictionaries with explanation of no events
                no_events_message = (
                    f"No valid {event_type} events identified in patient note."
                )
                identification_evidence = [no_events_message]

                # Log the default evidence
                log_evidence(
                    "[RequestID: {request_id}]NO_EVENTS",
                    event_type,
                    identification_evidence,
                    note_text,
                )

                return EnhancedEventResult(
                    event_type=event_type,
                    grade=0,
                    past_grade=0,
                    current_grade=0,
                    attribution=0,
                    certainty=0,
                    attribution_evidence=None,
                    certainty_evidence=None,
                    identification_evidence=identification_evidence,
                    past_grading_evidence=None,
                    current_grading_evidence=None,
                    reasoning=f"No valid {event_type} events identified in patient note.",
                    iterations_completed=iterations_completed,
                )

            # 6. Set up for attribution and certainty detection
            # Get terms data first
            event_type_lower = event_type.lower()
            all_terms_data = get_terms_definitions_and_grades(event_type_lower)

            # Format only term names for attribution/certainty context (no definitions needed)
            all_definitions = []
            for term, term_data in all_terms_data.items():
                # Add only the term name since attribution/certainty don't need full definitions
                all_definitions.append(term)

            # Join all term names
            combined_definitions = "\n".join(all_definitions)

            # Validate we have term names
            if not combined_definitions:
                error_msg = f"[RequestID: {request_id}] No term names available for {event_type} attribution/certainty analysis"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Create contexts with term names only
            attribution_context = EventContext(
                event_type=event_type,
                event_definition=combined_definitions,
                request_id=request_id,
            )

            certainty_context = EventContext(
                event_type=event_type,
                event_definition=combined_definitions,
                request_id=request_id,
            )

            # Run attribution detection for past events if they have a grade > 0
            if past_grade is not None and past_grade > 0:
                past_attribution_input = f"""
                ### EVENT TYPE: 
                {event_type}
                
                ### TEMPORAL FOCUS: 
                Past Events Only
                
                ### PAST GRADE: 
                {past_grade}
                
                ### PATIENT NOTE:
                {note_text}
                
                ### REMINDER:
                Focus only on attribution for the PAST {event_type} (Grade {past_grade}) if this attributed to immunotherapy-- do not evaluate any attribution for CURRENT events.
                """

                # Set up run configuration with Azure provider if specified
                run_config = None
                if azure_provider:
                    run_config = RunConfig(model_provider=azure_provider)
                    logger.debug(
                        f"[RequestID: {request_id}]Using Azure provider for {event_type} past attribution detection"
                    )

                try:
                    past_attribution_result = await Runner.run(
                        attribution_detector,
                        past_attribution_input,
                        context=attribution_context,
                        run_config=run_config,
                    )

                    # Track token usage if tracker provided
                    if token_tracker:
                        agent_name = f"[RequestID: {request_id}]{event_type.lower()}_past_attribution"
                        # Get the model name from the attribution detector agent
                        model_obj = getattr(attribution_detector, "model", None)
                        # Extract the actual model name from the model object
                        if model_obj and hasattr(model_obj, "model"):
                            model_name = model_obj.model  # This is the deployment name
                        else:
                            model_name = str(model_obj) if model_obj else None
                        token_tracker.track_usage(
                            agent_name, past_attribution_result, model_name
                        )

                    try:
                        # Extract the attribution value
                        past_attribution_output = past_attribution_result.final_output
                        processing_result.past_attribution = (
                            past_attribution_output.attribution
                        )

                        # Store the past attribution detection for evidence extraction
                        processing_result.past_attribution_detection = (
                            past_attribution_output
                        )

                        # Try to extract and log evidence safely
                        try:
                            evidence = getattr(
                                past_attribution_output, "evidence", None
                            )

                            # Sanitize the evidence immediately
                            if evidence:
                                past_attribution_output.evidence = (
                                    flatten_evidence_list(evidence)
                                )

                            # Log the evidence extraction
                            logger.debug(
                                f"[RequestID: {request_id}]Past attribution evidence for {event_type}: {evidence}"
                            )

                            # Log attribution evidence using our debug function
                            log_evidence(
                                "[RequestID: {request_id}]PAST_ATTRIBUTION",
                                event_type,
                                evidence,
                                note_text,
                            )
                        except Exception as evidence_err:
                            logger.error(
                                f"Error extracting or logging attribution evidence for {event_type}: {str(evidence_err)}"
                            )
                            # Don't rethrow, just continue

                    except Exception as attr_err:
                        logger.error(
                            f"Error processing attribution output for {event_type}: {str(attr_err)}"
                        )
                        processing_result.past_attribution = 0  # Default to 0
                        processing_result.past_attribution_detection = None
                except Exception as run_err:
                    logger.error(
                        f"Error running attribution detection for {event_type}: {str(run_err)}"
                    )
                    processing_result.past_attribution = 0
                    processing_result.past_attribution_detection = None

            # Run attribution detection for current events if they have a grade > 0
            if current_grade is not None and current_grade > 0:
                current_attribution_input = f"""
                ### EVENT TYPE: 
                {event_type}
                
                ### TEMPORAL FOCUS: 
                Current Events Only
                
                ### CURRENT GRADE: 
                {current_grade}
                
                ### PATIENT NOTE:
                {note_text}
                
                ### REMINDER:
                Focus only on attribution for the CURRENT {event_type} (Grade {current_grade}) if this attributed to immunotherapy-- do not evaluate any attribution for PAST events.
                """

                # Set up run configuration with Azure provider if specified
                run_config = None
                if azure_provider:
                    run_config = RunConfig(model_provider=azure_provider)
                    logger.debug(
                        f"[RequestID: {request_id}]Using Azure provider for {event_type} current attribution detection"
                    )

                try:
                    current_attribution_result = await Runner.run(
                        attribution_detector,
                        current_attribution_input,
                        context=attribution_context,
                        run_config=run_config,
                    )

                    # Track token usage if tracker provided
                    if token_tracker:
                        agent_name = f"[RequestID: {request_id}]{event_type.lower()}_current_attribution"
                        # Get the model name from the attribution detector agent
                        model_obj = getattr(attribution_detector, "model", None)
                        # Extract the actual model name from the model object
                        if model_obj and hasattr(model_obj, "model"):
                            model_name = model_obj.model  # This is the deployment name
                        else:
                            model_name = str(model_obj) if model_obj else None
                        token_tracker.track_usage(
                            agent_name, current_attribution_result, model_name
                        )

                    try:
                        # Extract the attribution value
                        current_attribution_output = (
                            current_attribution_result.final_output
                        )
                        processing_result.current_attribution = (
                            current_attribution_output.attribution
                        )

                        # Store the current attribution detection for evidence extraction
                        processing_result.current_attribution_detection = (
                            current_attribution_output
                        )

                        # Try to extract and log evidence safely
                        try:
                            evidence = getattr(
                                current_attribution_output, "evidence", None
                            )

                            # Sanitize the evidence immediately
                            if evidence:
                                current_attribution_output.evidence = (
                                    flatten_evidence_list(evidence)
                                )

                            # Log the evidence extraction
                            logger.debug(
                                f"[RequestID: {request_id}]Current attribution evidence for {event_type}: {evidence}"
                            )

                            # Log attribution evidence using our debug function
                            log_evidence(
                                "[RequestID: {request_id}]CURRENT_ATTRIBUTION",
                                event_type,
                                evidence,
                                note_text,
                            )
                        except Exception as evidence_err:
                            logger.error(
                                f"Error extracting or logging attribution evidence for {event_type}: {str(evidence_err)}"
                            )
                            # Don't rethrow, just continue

                    except Exception as attr_err:
                        logger.error(
                            f"Error processing attribution output for {event_type}: {str(attr_err)}"
                        )
                        processing_result.current_attribution = 0  # Default to 0
                        processing_result.current_attribution_detection = None
                except Exception as run_err:
                    logger.error(
                        f"Error running attribution detection for {event_type}: {str(run_err)}"
                    )
                    processing_result.current_attribution = 0
                    processing_result.current_attribution_detection = None

            # Calculate overall attribution as maximum of past and current
            past_attribution = (
                0
                if processing_result.past_attribution is None
                else processing_result.past_attribution
            )
            current_attribution = (
                0
                if processing_result.current_attribution is None
                else processing_result.current_attribution
            )
            processing_result.attribution = max(past_attribution, current_attribution)

            # Run certainty assessment for past events if they have a grade > 0
            if past_grade is not None and past_grade > 0:
                past_certainty_input = f"""
                ### EVENT TYPE: 
                {event_type}
                
                ### TEMPORAL FOCUS: 
                Past Events Only
                
                ### PAST GRADE: 
                {past_grade}
                
                ### PATIENT NOTE:
                {note_text}
                
                ### REMINDER:
                Focus only on assessing the certainty for the PAST {event_type} (Grade {past_grade})-- do not evaluate any stated certainty for CURRENT events.
                """

                # Set up run configuration with Azure provider if specified
                run_config = None
                if azure_provider:
                    run_config = RunConfig(model_provider=azure_provider)
                    logger.debug(
                        f"[RequestID: {request_id}]Using Azure provider for {event_type} past certainty assessment"
                    )

                try:
                    past_certainty_result = await Runner.run(
                        certainty_assessor,
                        past_certainty_input,
                        context=certainty_context,
                        run_config=run_config,
                    )

                    # Track token usage if tracker provided
                    if token_tracker:
                        agent_name = f"[RequestID: {request_id}]{event_type.lower()}_past_certainty"
                        # Get the model name from the certainty assessor agent
                        model_obj = getattr(certainty_assessor, "model", None)
                        # Extract the actual model name from the model object
                        if model_obj and hasattr(model_obj, "model"):
                            model_name = model_obj.model  # This is the deployment name
                        else:
                            model_name = str(model_obj) if model_obj else None
                        token_tracker.track_usage(
                            agent_name, past_certainty_result, model_name
                        )

                    try:
                        # Extract the certainty value
                        past_certainty_output = past_certainty_result.final_output
                        processing_result.past_certainty = (
                            past_certainty_output.certainty
                        )

                        # Store the past certainty assessment for evidence extraction
                        processing_result.past_certainty_assessment = (
                            past_certainty_output
                        )

                        # Try to extract and log evidence safely
                        try:
                            evidence = getattr(past_certainty_output, "evidence", None)

                            # Sanitize the evidence immediately
                            if evidence:
                                past_certainty_output.evidence = flatten_evidence_list(
                                    evidence
                                )

                            # Log the evidence extraction
                            logger.debug(
                                f"[RequestID: {request_id}]Past certainty evidence for {event_type}: {evidence}"
                            )

                            # Log certainty evidence using our debug function
                            log_evidence(
                                "[RequestID: {request_id}]PAST_CERTAINTY",
                                event_type,
                                evidence,
                                note_text,
                            )
                        except Exception as evidence_err:
                            logger.error(
                                f"Error extracting or logging certainty evidence for {event_type}: {str(evidence_err)}"
                            )
                            # Don't rethrow, just continue

                    except Exception as cert_err:
                        logger.error(
                            f"Error processing certainty output for {event_type}: {str(cert_err)}"
                        )
                        processing_result.past_certainty = 0  # Default to 0
                        processing_result.past_certainty_assessment = None
                except Exception as run_err:
                    logger.error(
                        f"Error running certainty assessment for {event_type}: {str(run_err)}"
                    )
                    processing_result.past_certainty = 0
                    processing_result.past_certainty_assessment = None

            # Run certainty assessment for current events if they have a grade > 0
            if current_grade is not None and current_grade > 0:
                current_certainty_input = f"""
                ### EVENT TYPE: 
                {event_type}
                
                ### TEMPORAL FOCUS: 
                Current Events Only
                
                ### CURRENT GRADE: 
                {current_grade}
                
                ### PATIENT NOTE:
                {note_text}
                
                ### REMINDER:
                Focus only on assessing the certainty for the CURRENT {event_type} (Grade {current_grade})-- do not evaluate any stated certainty for PAST events.
                """

                # Set up run configuration with Azure provider if specified
                run_config = None
                if azure_provider:
                    run_config = RunConfig(model_provider=azure_provider)
                    logger.debug(
                        f"[RequestID: {request_id}]Using Azure provider for {event_type} current certainty assessment"
                    )

                try:
                    current_certainty_result = await Runner.run(
                        certainty_assessor,
                        current_certainty_input,
                        context=certainty_context,
                        run_config=run_config,
                    )

                    # Track token usage if tracker provided
                    if token_tracker:
                        agent_name = f"[RequestID: {request_id}]{event_type.lower()}_current_certainty"
                        # Get the model name from the certainty assessor agent
                        model_obj = getattr(certainty_assessor, "model", None)
                        # Extract the actual model name from the model object
                        if model_obj and hasattr(model_obj, "model"):
                            model_name = model_obj.model  # This is the deployment name
                        else:
                            model_name = str(model_obj) if model_obj else None
                        token_tracker.track_usage(
                            agent_name, current_certainty_result, model_name
                        )

                    try:
                        # Extract the certainty value
                        current_certainty_output = current_certainty_result.final_output
                        processing_result.current_certainty = (
                            current_certainty_output.certainty
                        )

                        # Store the current certainty assessment for evidence extraction
                        processing_result.current_certainty_assessment = (
                            current_certainty_output
                        )

                        # Try to extract and log evidence safely
                        try:
                            evidence = getattr(
                                current_certainty_output, "evidence", None
                            )

                            # Sanitize the evidence immediately
                            if evidence:
                                current_certainty_output.evidence = (
                                    flatten_evidence_list(evidence)
                                )

                            # Log the evidence extraction
                            logger.debug(
                                f"[RequestID: {request_id}]Current certainty evidence for {event_type}: {evidence}"
                            )

                            # Log certainty evidence using our debug function
                            log_evidence(
                                "[RequestID: {request_id}]CURRENT_CERTAINTY",
                                event_type,
                                evidence,
                                note_text,
                            )
                        except Exception as evidence_err:
                            logger.error(
                                f"Error extracting or logging certainty evidence for {event_type}: {str(evidence_err)}"
                            )
                            # Don't rethrow, just continue

                    except Exception as cert_err:
                        logger.error(
                            f"Error processing certainty output for {event_type}: {str(cert_err)}"
                        )
                        processing_result.current_certainty = 0  # Default to 0
                        processing_result.current_certainty_assessment = None
                except Exception as run_err:
                    logger.error(
                        f"Error running certainty assessment for {event_type}: {str(run_err)}"
                    )
                    processing_result.current_certainty = 0
                    processing_result.current_certainty_assessment = None

            # Calculate overall certainty as maximum of past and current
            past_certainty = (
                0
                if processing_result.past_certainty is None
                else processing_result.past_certainty
            )
            current_certainty = (
                0
                if processing_result.current_certainty is None
                else processing_result.current_certainty
            )
            processing_result.certainty = max(past_certainty, current_certainty)

            # Generate an integrated reasoning based on identification, grading, attribution and certainty
            reasoning_parts = []

            # Add identification and temporality reasoning
            identification_reason = f"Identified events: Past={join_safe(processing_result.judged_temporality.past_events or ['None'])}; "
            identification_reason += (
                f"Current={join_safe(processing_result.judged_temporality.current_events or ['None'])}"
            )
            reasoning_parts.append(identification_reason)

            # Add grading reasoning
            if (
                processing_result.past_grade is not None
                and processing_result.past_grade > 0
            ):
                past_grading_reason = f"Past grade: {processing_result.past_grade}"
                if (
                    hasattr(processing_result, "judged_past_grading")
                    and processing_result.judged_past_grading
                    and processing_result.judged_past_grading.rationale
                ):
                    past_grading_reason += (
                        f" - {processing_result.judged_past_grading.rationale}"
                    )
                reasoning_parts.append(past_grading_reason)

            if (
                processing_result.current_grade is not None
                and processing_result.current_grade > 0
            ):
                current_grading_reason = (
                    f"Current grade: {processing_result.current_grade}"
                )
                if (
                    hasattr(processing_result, "judged_current_grading")
                    and processing_result.judged_current_grading
                    and processing_result.judged_current_grading.rationale
                ):
                    current_grading_reason += (
                        f" - {processing_result.judged_current_grading.rationale}"
                    )
                reasoning_parts.append(current_grading_reason)

            # Add attribution reasoning
            if (
                processing_result.attribution is not None
                and processing_result.attribution > 0
            ):
                attribution_reason = (
                    f"Attribution to immunotherapy: {processing_result.attribution} "
                )
                if (
                    past_attribution is not None
                    and past_attribution > 0
                    and current_attribution is not None
                    and current_attribution > 0
                ):
                    attribution_reason += "(present in both past and current events)"
                elif past_attribution is not None and past_attribution > 0:
                    attribution_reason += "(present in past events)"
                elif current_attribution is not None and current_attribution > 0:
                    attribution_reason += "(present in current events)"
                reasoning_parts.append(attribution_reason)

            # Add certainty reasoning
            if processing_result.certainty > 0:
                certainty_reason = (
                    f"Assessment certainty: {processing_result.certainty} "
                )
                if past_certainty > 0 and current_certainty > 0:
                    certainty_reason += (
                        "(high confidence in both past and current events)"
                    )
                elif past_certainty > 0:
                    certainty_reason += "(high confidence in past events)"
                elif current_certainty > 0:
                    certainty_reason += "(high confidence in current events)"
                reasoning_parts.append(certainty_reason)

            # Combine all reasoning
            processing_result.reasoning = ". ".join(reasoning_parts)

            # See if we need another iteration via meta-judge (skip for ablation_no_judge)
            if (
                iteration < max_iterations - 1 and prompt_variant != "ablation_no_judge"
            ):  # Not the last iteration yet and not ablation study
                # Run meta-judge to evaluate result quality
                meta_judge_feedback = await evaluate_event_processing(
                    event_type=event_type,
                    event_result=processing_result,
                    note_text=note_text,
                    meta_judge_agent=meta_judge,
                    azure_provider=azure_provider,
                    context=attribution_context,
                    token_tracker=token_tracker,
                )

                # If meta-judge suggests improvements are needed, continue to next iteration
                if meta_judge_feedback.needs_improvement:
                    processing_result.meta_judge_feedback = meta_judge_feedback
                    logger.debug(
                        f"[RequestID: {request_id}]Meta-judge suggests improvements for {event_type}, continuing to next iteration"
                    )
                    continue
            elif prompt_variant == "ablation_no_judge":
                logger.debug(
                    f"[RequestID: {request_id}] ABLATION STUDY: Skipping meta-judge for {event_type}"
                )
            else:
                # This is the last iteration, generate just the user overview without full meta-judge feedback
                # Use the dedicated overview agent instead of meta-judge
                overview_agent = event_agents[event_type]["overview"]
                user_overview = await generate_user_overview(
                    event_type=event_type,
                    event_result=processing_result,
                    note_text=note_text,
                    overview_agent=overview_agent,  # Use the dedicated overview agent
                    azure_provider=azure_provider,
                    context=attribution_context,
                    token_tracker=token_tracker,
                    request_id=request_id,
                )
                processing_result.user_overview = user_overview
                logger.debug(
                    f"[RequestID: {request_id}][User Overview] Generated dedicated overview on final iteration: {user_overview}"
                )

            # If we reached here, either the meta-judge is satisfied or we've hit max_iterations
            # Create the final enhanced event result

            # Log the evidence right before creating the final result
            logger.debug(
                f"[RequestID: {request_id}][Evidence Debug] Event type: {event_type}"
            )

            # Check if we have meta-judge feedback with a user overview
            user_overview = None
            if (
                hasattr(processing_result, "meta_judge_feedback")
                and processing_result.meta_judge_feedback is not None
                and hasattr(processing_result.meta_judge_feedback, "user_overview")
                and processing_result.meta_judge_feedback.user_overview
            ):
                user_overview = processing_result.meta_judge_feedback.user_overview
                logger.debug(
                    f"[RequestID: {request_id}][User Overview] Using meta-judge provided overview: {user_overview}"
                )
            # If we don't have a user overview from meta-judge, but we have one from general processing
            elif (
                hasattr(processing_result, "user_overview")
                and processing_result.user_overview
            ):
                user_overview = processing_result.user_overview
                logger.debug(
                    f"[RequestID: {request_id}][User Overview] Using processing result overview: {user_overview}"
                )
            # Otherwise, generate a simple user overview based on the available data
            else:
                if processing_result.grade == 0:
                    user_overview = f"No evidence of {event_type} was detected in this clinical note."
                else:
                    # Build a simple overview based on the available data
                    temporal_parts = []
                    if (
                        processing_result.past_grade
                        and processing_result.past_grade > 0
                    ):
                        temporal_parts.append(
                            f"past (grade {processing_result.past_grade})"
                        )
                    if (
                        processing_result.current_grade
                        and processing_result.current_grade > 0
                    ):
                        temporal_parts.append(
                            f"current (grade {processing_result.current_grade})"
                        )

                    temporal_context = " and ".join(temporal_parts)
                    user_overview = f"Evidence of {event_type} was detected in the {temporal_context} context."

                    # Add attribution if available
                    if (
                        processing_result.attribution
                        and processing_result.attribution > 0
                    ):
                        user_overview += f" Attribution to immunotherapy is indicated."

                logger.debug(
                    f"[RequestID: {request_id}][User Overview] Generated basic overview: {user_overview}"
                )

            # Check and log attribution evidence
            has_past_attribution = (
                hasattr(processing_result, "past_attribution_detection")
                and processing_result.past_attribution_detection is not None
            )
            has_current_attribution = (
                hasattr(processing_result, "current_attribution_detection")
                and processing_result.current_attribution_detection is not None
            )

            logger.debug(
                f"[RequestID: {request_id}][Evidence Debug] Has past attribution: {has_past_attribution}"
            )
            if has_past_attribution:
                logger.debug(
                    f"[RequestID: {request_id}][Evidence Debug] Past attribution evidence: {processing_result.past_attribution_detection.evidence}"
                )

            logger.debug(
                f"[RequestID: {request_id}][Evidence Debug] Has current attribution: {has_current_attribution}"
            )
            if has_current_attribution:
                logger.debug(
                    f"[RequestID: {request_id}][Evidence Debug] Current attribution evidence: {processing_result.current_attribution_detection.evidence}"
                )

            # Check and log certainty evidence
            has_past_certainty = (
                hasattr(processing_result, "past_certainty_assessment")
                and processing_result.past_certainty_assessment is not None
            )
            has_current_certainty = (
                hasattr(processing_result, "current_certainty_assessment")
                and processing_result.current_certainty_assessment is not None
            )

            logger.debug(
                f"[RequestID: {request_id}][Evidence Debug] Has past certainty: {has_past_certainty}"
            )
            if has_past_certainty:
                logger.debug(
                    f"[RequestID: {request_id}][Evidence Debug] Past certainty evidence: {processing_result.past_certainty_assessment.evidence}"
                )

            logger.debug(
                f"[RequestID: {request_id}][Evidence Debug] Has current certainty: {has_current_certainty}"
            )
            if has_current_certainty:
                logger.debug(
                    f"[RequestID: {request_id}][Evidence Debug] Current certainty evidence: {processing_result.current_certainty_assessment.evidence}"
                )

            # Check and log grading evidence
            has_past_grading = (
                hasattr(processing_result, "judged_past_grading")
                and processing_result.judged_past_grading is not None
            )
            has_current_grading = (
                hasattr(processing_result, "judged_current_grading")
                and processing_result.judged_current_grading is not None
            )

            logger.debug(
                f"[RequestID: {request_id}][Evidence Debug] Has past grading: {has_past_grading}"
            )
            if has_past_grading:
                logger.debug(
                    f"[RequestID: {request_id}][Evidence Debug] Past grading rationale: {processing_result.judged_past_grading.rationale}"
                )
                logger.debug(
                    f"[RequestID: {request_id}][Evidence Debug] Past grading evidence snippets: {processing_result.judged_past_grading.evidence_snippets}"
                )

            logger.debug(
                f"[RequestID: {request_id}][Evidence Debug] Has current grading: {has_current_grading}"
            )
            if has_current_grading:
                logger.debug(
                    f"[RequestID: {request_id}][Evidence Debug] Current grading rationale: {processing_result.judged_current_grading.rationale}"
                )
                logger.debug(
                    f"[RequestID: {request_id}][Evidence Debug] Current grading evidence snippets: {processing_result.judged_current_grading.evidence_snippets}"
                )

            # Create attribution evidence dictionary
            attribution_evidence = None
            if has_current_attribution or has_past_attribution:
                evidence_text = None
                if has_current_attribution:
                    evidence_text = (
                        processing_result.current_attribution_detection.evidence
                    )
                elif has_past_attribution:
                    evidence_text = (
                        processing_result.past_attribution_detection.evidence
                    )

                if evidence_text:
                    attribution_evidence = [evidence_text]

            # Create certainty evidence dictionary
            certainty_evidence = None
            if has_current_certainty or has_past_certainty:
                evidence_text = None
                if has_current_certainty:
                    evidence_text = (
                        processing_result.current_certainty_assessment.evidence
                    )
                elif has_past_certainty:
                    evidence_text = processing_result.past_certainty_assessment.evidence

                if evidence_text:
                    certainty_evidence = [evidence_text]

            # Create past grading evidence dictionary
            past_grading_evidence = None
            if has_past_grading:
                if processing_result.judged_past_grading.evidence_snippets:
                    past_grading_evidence = (
                        processing_result.judged_past_grading.evidence_snippets
                    )
                elif processing_result.judged_past_grading.rationale:
                    past_grading_evidence = [
                        processing_result.judged_past_grading.rationale
                    ]

            # Create current grading evidence dictionary
            current_grading_evidence = None
            if has_current_grading:
                if processing_result.judged_current_grading.evidence_snippets:
                    current_grading_evidence = (
                        processing_result.judged_current_grading.evidence_snippets
                    )
                elif processing_result.judged_current_grading.rationale:
                    current_grading_evidence = [
                        processing_result.judged_current_grading.rationale
                    ]

            # Create identification evidence from judged event identification and temporality if available
            identification_evidence = None
            all_identification_evidence = []
            
            # Collect evidence from event identification
            if (
                hasattr(processing_result, "judged_event_identification")
                and processing_result.judged_event_identification is not None
            ):
                if (
                    hasattr(
                        processing_result.judged_event_identification,
                        "evidence_snippets",
                    )
                    and processing_result.judged_event_identification.evidence_snippets
                ):
                    all_identification_evidence.extend(
                        processing_result.judged_event_identification.evidence_snippets
                    )
            
            # Collect evidence from temporality classification
            if (
                hasattr(processing_result, "judged_temporality")
                and processing_result.judged_temporality is not None
            ):
                if (
                    hasattr(
                        processing_result.judged_temporality,
                        "evidence_snippets",
                    )
                    and processing_result.judged_temporality.evidence_snippets
                ):
                    all_identification_evidence.extend(
                        processing_result.judged_temporality.evidence_snippets
                    )
            
            # Remove duplicates while preserving order
            if all_identification_evidence:
                seen = set()
                identification_evidence = []
                for evidence in all_identification_evidence:
                    if evidence not in seen:
                        seen.add(evidence)
                        identification_evidence.append(evidence)

            logger.debug(
                f"[RequestID: {request_id}][Evidence Debug] Final attribution_evidence: {attribution_evidence}"
            )
            logger.debug(
                f"[RequestID: {request_id}][Evidence Debug] Final certainty_evidence: {certainty_evidence}"
            )
            logger.debug(
                f"[RequestID: {request_id}][Evidence Debug] Final past_grading_evidence: {past_grading_evidence}"
            )
            logger.debug(
                f"[RequestID: {request_id}][Evidence Debug] Final current_grading_evidence: {current_grading_evidence}"
            )
            logger.debug(
                f"[RequestID: {request_id}][Evidence Debug] Final identification_evidence: {identification_evidence}"
            )

            # Create event result from processed data with structured evidence and user overview
            try:
                # Initialize empty evidence collections
                attribution_evidence = []
                certainty_evidence = []
                identification_evidence = []
                past_grading_evidence = []
                current_grading_evidence = []

                # Safely collect attribution evidence
                if (
                    hasattr(processing_result, "past_attribution_detection")
                    and processing_result.past_attribution_detection
                ):
                    try:
                        evidence = getattr(
                            processing_result.past_attribution_detection,
                            "evidence",
                            None,
                        )
                        if evidence:
                            # Use flatten_evidence_list to handle nested lists
                            attribution_evidence.extend(flatten_evidence_list(evidence))
                    except Exception as e:
                        logger.error(
                            f"Error processing past attribution evidence for {event_type}: {str(e)}"
                        )

                if (
                    hasattr(processing_result, "current_attribution_detection")
                    and processing_result.current_attribution_detection
                ):
                    try:
                        evidence = getattr(
                            processing_result.current_attribution_detection,
                            "evidence",
                            None,
                        )
                        if evidence:
                            # Use flatten_evidence_list to handle nested lists
                            attribution_evidence.extend(flatten_evidence_list(evidence))
                    except Exception as e:
                        logger.error(
                            f"Error processing current attribution evidence for {event_type}: {str(e)}"
                        )

                # Safely collect certainty evidence
                if (
                    hasattr(processing_result, "past_certainty_assessment")
                    and processing_result.past_certainty_assessment
                ):
                    try:
                        evidence = getattr(
                            processing_result.past_certainty_assessment,
                            "evidence",
                            None,
                        )
                        if evidence:
                            # Use flatten_evidence_list to handle nested lists
                            certainty_evidence.extend(flatten_evidence_list(evidence))
                    except Exception as e:
                        logger.error(
                            f"Error processing past certainty evidence for {event_type}: {str(e)}"
                        )

                if (
                    hasattr(processing_result, "current_certainty_assessment")
                    and processing_result.current_certainty_assessment
                ):
                    try:
                        evidence = getattr(
                            processing_result.current_certainty_assessment,
                            "evidence",
                            None,
                        )
                        if evidence:
                            # Use flatten_evidence_list to handle nested lists
                            certainty_evidence.extend(flatten_evidence_list(evidence))
                    except Exception as e:
                        logger.error(
                            f"Error processing current certainty evidence for {event_type}: {str(e)}"
                        )

                # Safely collect identification evidence
                if (
                    hasattr(processing_result, "judged_identification")
                    and processing_result.judged_identification
                ):
                    try:
                        if (
                            hasattr(
                                processing_result.judged_identification,
                                "evidence_snippets",
                            )
                            and processing_result.judged_identification.evidence_snippets
                        ):
                            for (
                                snippet
                            ) in (
                                processing_result.judged_identification.evidence_snippets
                            ):
                                try:
                                    identification_evidence.append(snippet)
                                except Exception as snippet_err:
                                    logger.error(
                                        f"Error adding identification evidence snippet for {event_type}: {str(snippet_err)}"
                                    )
                        elif (
                            hasattr(
                                processing_result.judged_identification, "reasoning"
                            )
                            and processing_result.judged_identification.reasoning
                        ):
                            identification_evidence.append(
                                processing_result.judged_identification.reasoning
                            )
                    except Exception as e:
                        logger.error(
                            f"Error processing identification evidence for {event_type}: {str(e)}"
                        )

                # Safely collect past grading evidence
                if (
                    hasattr(processing_result, "judged_past_grading")
                    and processing_result.judged_past_grading
                ):
                    try:
                        if (
                            hasattr(
                                processing_result.judged_past_grading,
                                "evidence_snippets",
                            )
                            and processing_result.judged_past_grading.evidence_snippets
                        ):
                            for (
                                snippet
                            ) in (
                                processing_result.judged_past_grading.evidence_snippets
                            ):
                                try:
                                    past_grading_evidence.append(snippet)
                                except Exception as snippet_err:
                                    logger.error(
                                        f"Error adding past grading evidence snippet for {event_type}: {str(snippet_err)}"
                                    )
                        elif (
                            hasattr(processing_result.judged_past_grading, "reasoning")
                            and processing_result.judged_past_grading.reasoning
                        ):
                            past_grading_evidence.append(
                                processing_result.judged_past_grading.reasoning
                            )
                    except Exception as e:
                        logger.error(
                            f"Error processing past grading evidence for {event_type}: {str(e)}"
                        )

                # Safely collect current grading evidence
                if (
                    hasattr(processing_result, "judged_current_grading")
                    and processing_result.judged_current_grading
                ):
                    try:
                        if (
                            hasattr(
                                processing_result.judged_current_grading,
                                "evidence_snippets",
                            )
                            and processing_result.judged_current_grading.evidence_snippets
                        ):
                            for (
                                snippet
                            ) in (
                                processing_result.judged_current_grading.evidence_snippets
                            ):
                                try:
                                    current_grading_evidence.append(snippet)
                                except Exception as snippet_err:
                                    logger.error(
                                        f"Error adding current grading evidence snippet for {event_type}: {str(snippet_err)}"
                                    )
                        elif (
                            hasattr(
                                processing_result.judged_current_grading, "reasoning"
                            )
                            and processing_result.judged_current_grading.reasoning
                        ):
                            current_grading_evidence.append(
                                processing_result.judged_current_grading.reasoning
                            )
                    except Exception as e:
                        logger.error(
                            f"Error processing current grading evidence for {event_type}: {str(e)}"
                        )

                # Create structured evidence with consistent keys regardless of what evidence was collected

                # Validate all evidence lists to ensure they contain only strings
                if identification_evidence:
                    identification_evidence = flatten_evidence_list(
                        identification_evidence
                    )
                if past_grading_evidence:
                    past_grading_evidence = flatten_evidence_list(past_grading_evidence)
                if current_grading_evidence:
                    current_grading_evidence = flatten_evidence_list(
                        current_grading_evidence
                    )
                if attribution_evidence:
                    attribution_evidence = flatten_evidence_list(attribution_evidence)
                if certainty_evidence:
                    certainty_evidence = flatten_evidence_list(certainty_evidence)

                structured_evidence = {
                    "identification": {
                        "evidence": (
                            identification_evidence if identification_evidence else []
                        )
                    },
                    "past_grading": {
                        "evidence": (
                            past_grading_evidence if past_grading_evidence else []
                        )
                    },
                    "current_grading": {
                        "evidence": (
                            current_grading_evidence if current_grading_evidence else []
                        )
                    },
                    "attribution": {
                        "evidence": attribution_evidence if attribution_evidence else []
                    },
                    "certainty": {
                        "evidence": certainty_evidence if certainty_evidence else []
                    },
                }

                # Create the final result object
                processing_result = EnhancedEventResult(
                    event_type=event_type,
                    grade=processing_result.grade,
                    past_grade=past_grade if past_grade is not None else 0,
                    current_grade=current_grade if current_grade is not None else 0,
                    attribution=getattr(processing_result, "attribution", 0),
                    past_attribution=getattr(processing_result, "past_attribution", 0),
                    current_attribution=getattr(
                        processing_result, "current_attribution", 0
                    ),
                    certainty=getattr(processing_result, "certainty", 0),
                    past_certainty=getattr(processing_result, "past_certainty", 0),
                    current_certainty=getattr(
                        processing_result, "current_certainty", 0
                    ),
                    evidence=structured_evidence,  # Use the structured evidence format
                    identification_evidence=(
                        identification_evidence if identification_evidence else None
                    ),
                    past_grading_evidence=(
                        past_grading_evidence if past_grading_evidence else None
                    ),
                    current_grading_evidence=(
                        current_grading_evidence if current_grading_evidence else None
                    ),
                    attribution_evidence=(
                        attribution_evidence if attribution_evidence else None
                    ),
                    certainty_evidence=(
                        certainty_evidence if certainty_evidence else None
                    ),
                    user_overview=user_overview,  # Include the user overview
                    reasoning=(
                        processing_result.reasoning
                        if processing_result.reasoning
                        else f"Processing completed with grade {getattr(processing_result, 'grade', 0)}"
                    ),
                    iterations_completed=iterations_completed,
                )

                # Log successful creation of final result
                logger.info(
                    f"Successfully created final result for {event_type} with grade {processing_result.grade}"
                )

                return processing_result

            except Exception as final_err:
                # If anything goes wrong in final result creation, create a minimal result
                logger.error(
                    f"Error creating final result for {event_type}: {str(final_err)}"
                )

                # Create a basic result with just the essential information
                processing_result = EnhancedEventResult(
                    event_type=event_type,
                    grade=getattr(processing_result, "grade", 0),
                    past_grade=past_grade if past_grade is not None else 0,
                    current_grade=current_grade if current_grade is not None else 0,
                    attribution=getattr(processing_result, "attribution", 0),
                    past_attribution=getattr(processing_result, "past_attribution", 0),
                    current_attribution=getattr(
                        processing_result, "current_attribution", 0
                    ),
                    certainty=getattr(processing_result, "certainty", 0),
                    past_certainty=getattr(processing_result, "past_certainty", 0),
                    current_certainty=getattr(
                        processing_result, "current_certainty", 0
                    ),
                    evidence={
                        "identification": {"evidence": []},
                        "past_grading": {"evidence": []},
                        "current_grading": {"evidence": []},
                        "attribution": {"evidence": []},
                        "certainty": {"evidence": []},
                    },
                    user_overview=f"Evidence of {event_type} was processed but encountered an error during evidence collection. The detected grade is {getattr(processing_result, 'grade', 0)}.",
                    reasoning=f"Processing completed with grade {getattr(processing_result, 'grade', 0)} but encountered errors: {str(final_err)}",
                    iterations_completed=iterations_completed,
                )

                return processing_result

    except Exception as e:
        # Log the error and return a minimal result
        logger.error(f"Error in process_event_with_judge for {event_type}: {str(e)}")

        # Return a minimal result with error information
        return EnhancedEventResult(
            event_type=event_type,
            grade=0,
            past_grade=0,
            current_grade=0,
            attribution=0,
            certainty=0,
            evidence={
                "identification": {"evidence": []},
                "past_grading": {"evidence": []},
                "current_grading": {"evidence": []},
                "attribution": {"evidence": []},
                "certainty": {"evidence": []},
            },
            user_overview=f"Error processing {event_type} event: {str(e)}",
            reasoning=f"An error occurred during processing: {str(e)}",
            iterations_completed=iterations_completed,
        )


async def generate_user_overview(
    event_type: str,
    event_result: Any,
    note_text: str,
    overview_agent: Agent,
    azure_provider=None,
    context: Optional[EventContext] = None,
    token_tracker=None,
    request_id: str = None,
) -> str:
    """
    Generate just a user overview summary for an event without full meta-judge evaluation.

    Args:
        event_type: The type of event being evaluated
        event_result: The result of event processing
        note_text: The extracted note text
        overview_agent: The agent to generate the overview
        azure_provider: Optional Azure provider for authentication
        context: Optional event context for dynamic instructions
        token_tracker: Optional token usage tracker
        request_id: Optional request ID for logging

    Returns:
        str: A user-friendly overview of the event findings
    """
    logger.debug(f"[RequestID: {request_id}] Generating user overview for {event_type}")

    # Format and provide evidence information for the overview agent
    structured_evidence = None
    if hasattr(event_result, "evidence") and event_result.evidence:
        # Sanitize the evidence to ensure it's JSON-safe
        structured_evidence = sanitize_for_json(event_result.evidence)

    # Create input for the overview agent
    overview_input = f"""
    Event Type: {event_type}
        
    Processing Result Summary:
    Grade: {getattr(event_result, 'grade', 0)} (Past: {getattr(event_result, 'past_grade', 0)}, Current: {getattr(event_result, 'current_grade', 0)})
    Attribution: {getattr(event_result, 'attribution', 0)} (Past: {getattr(event_result, 'past_attribution', 0)}, Current: {getattr(event_result, 'current_attribution', 0)})
    Certainty: {getattr(event_result, 'certainty', 0)} (Past: {getattr(event_result, 'past_certainty', 0)}, Current: {getattr(event_result, 'current_certainty', 0)})
    """

    # Add evidence details to help with user overview creation
    if structured_evidence:
        overview_input += "\n\nEvidence Details:\n"

        # Add identification evidence
        if "identification" in structured_evidence:
            overview_input += "\nIdentification Evidence:\n"
            evidence_items = structured_evidence["identification"].get("evidence", [])
            if evidence_items:
                # Sanitize each evidence item individually before joining
                sanitized_items = [sanitize_for_json(item) for item in evidence_items]
                overview_input += f"- {join_safe(sanitized_items)}\n"

        # Add grading evidence
        if "past_grading" in structured_evidence:
            overview_input += "\nPast Grading Evidence:\n"
            evidence_items = structured_evidence["past_grading"].get("evidence", [])
            if evidence_items:
                # Sanitize each evidence item individually before joining
                sanitized_items = [sanitize_for_json(item) for item in evidence_items]
                overview_input += f"- {join_safe(sanitized_items)}\n"

        if "current_grading" in structured_evidence:
            overview_input += "\nCurrent Grading Evidence:\n"
            evidence_items = structured_evidence["current_grading"].get("evidence", [])
            if evidence_items:
                # Sanitize each evidence item individually before joining
                sanitized_items = [sanitize_for_json(item) for item in evidence_items]
                overview_input += f"- {join_safe(sanitized_items)}\n"

        # Add attribution evidence
        if "attribution" in structured_evidence:
            overview_input += "\nAttribution Evidence:\n"
            evidence_items = structured_evidence["attribution"].get("evidence", [])
            if evidence_items:
                # Sanitize each evidence item individually before joining
                sanitized_items = [sanitize_for_json(item) for item in evidence_items]
                overview_input += f"- {join_safe(sanitized_items)}\n"

        # Add certainty evidence
        if "certainty" in structured_evidence:
            overview_input += "\nCertainty Evidence:\n"
            evidence_items = structured_evidence["certainty"].get("evidence", [])
            if evidence_items:
                # Sanitize each evidence item individually before joining
                sanitized_items = [sanitize_for_json(item) for item in evidence_items]
                overview_input += f"- {join_safe(sanitized_items)}\n"

    # Get reasoning if available and sanitize it
    reasoning = sanitize_for_json(
        getattr(event_result, "reasoning", "No reasoning provided")
    )
    overview_input += f"""
    Reasoning:
    {reasoning}
    """

    # Set up run configuration with Azure provider if specified
    run_config = None
    if (
        azure_provider is not None
        and not isinstance(azure_provider, EventContext)
        and hasattr(azure_provider, "get_model")
    ):
        run_config = RunConfig(model_provider=azure_provider)
        logger.debug(
            f"[RequestID: {request_id}] Using Azure provider for {event_type} overview generation"
        )

    # Run the overview agent with run configuration if provided
    overview_result = await Runner.run(
        overview_agent,
        overview_input,
        run_config=run_config,
    )

    # Track token usage if tracker provided
    if token_tracker:
        agent_name = f"{event_type.lower()}_overview_generator"
        # Get the model name from the overview agent
        model_obj = getattr(overview_agent, "model", None)
        # Extract the actual model name from the model object
        if model_obj and hasattr(model_obj, "model"):
            model_name = model_obj.model  # This is the deployment name
        else:
            model_name = str(model_obj) if model_obj else None
        token_tracker.track_usage(agent_name, overview_result, model_name)

    # Extract the overview text from the result and sanitize it
    # The overview agent is configured to return a string directly
    overview_text = sanitize_for_json(overview_result.final_output)

    # Log the user overview
    logger.debug(
        f"[RequestID: {request_id}] Generated user overview for {event_type}: {overview_text}"
    )

    return overview_text


##### NEW SPLIT TEMPORALITY WORKFLOW FUNCTIONS #####


async def run_parallel_event_identification(
    event_type: str,
    note_text: str,
    identifier_agents: List[Agent],
    azure_provider=None,
    token_tracker=None,
) -> List[EventIdentificationResult]:
    """
    Run multiple event identifier agents in parallel to identify events (without temporal classification)

    Args:
        event_type: The type of event to identify
        note_text: The extracted note text
        identifier_agents: List of identifier agents to run in parallel
        azure_provider: Optional Azure provider for authentication
        token_tracker: Optional token usage tracker

    Returns:
        List[EventIdentificationResult]: List of event identification results from different agents
    """
    logger.debug(
        f"Starting parallel event identification for {event_type} with {len(identifier_agents)} agents"
    )

    # Create context with the event type
    try:
        # Get CTCAE subset for this event type
        event_type_lower = event_type.lower()
        event_subset = get_ctcae_subset(event_type_lower)

        # Get the terms and definitions for this event type
        all_terms_data = get_terms_definitions_and_grades(event_type_lower)

        # Format only term names for identification context (no definitions to reduce tokens)
        all_definitions = []

        for term, term_data in all_terms_data.items():
            # Add only the term name, not the definition
            all_definitions.append(term)

        # Join all term names
        combined_definitions = "\n".join(all_definitions)

        # Create event context for identification (no grading criteria needed)
        event_context = EventContext(
            event_type=event_type,
            event_definition=combined_definitions,
            grading_criteria="",  # Empty for identification
        )

    except Exception as e:
        logger.error(f"Error creating event context for {event_type}: {str(e)}")
        raise

    # Set up run configuration with Azure provider if specified
    run_config = None
    if azure_provider:
        run_config = RunConfig(model_provider=azure_provider)

    # Create input for identifier agents
    identifier_input = note_text

    # Run all identifier agents in parallel
    identification_tasks = []
    for i, agent in enumerate(identifier_agents):
        try:
            # Add staggered delays to prevent overwhelming the API
            if i > 0:
                stagger_delay = min(i * 0.1, 1.0)  # Max 1 second delay
                logger.debug(
                    f"Adding {stagger_delay:.2f}s stagger delay for agent {i} of {event_type}"
                )
                await asyncio.sleep(stagger_delay)

            logger.debug(f"Starting agent {i} for {event_type}")
            identification_tasks.append(
                Runner.run(
                    agent,
                    identifier_input,
                    context=event_context,
                    run_config=run_config,
                )
            )
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")

    # Run all identification tasks concurrently
    try:
        identification_results = await asyncio.gather(
            *identification_tasks, return_exceptions=True
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

    # Process results
    results = []
    for i, result in enumerate(identification_results):
        if isinstance(result, Exception):
            logger.error(f"Error in identifier {i}: {str(result)}")
            continue

        try:
            # Log the complete agent output
            log_agent_output(
                "EVENT_IDENTIFICATION", "Identifier", i, result, event_type
            )

            # Track token usage if tracker provided
            if token_tracker:
                agent_name = f"{event_type.lower()}_event_identifier_{i}"
                model_obj = getattr(identifier_agents[i], "model", None)
                if model_obj and hasattr(model_obj, "model"):
                    model_name = model_obj.model
                else:
                    model_name = str(model_obj) if model_obj else None

                if hasattr(token_tracker, "debug_token_extraction"):
                    token_tracker.debug_token_extraction(result)

                token_tracker.track_usage(agent_name, result, model_name)

            # Extract the identifier output
            identifier_output = result.final_output

            # Create an EventIdentificationResult object
            try:
                reasoning = ""
                if (
                    hasattr(identifier_output, "reasoning")
                    and identifier_output.reasoning
                ):
                    reasoning = identifier_output.reasoning

                sanitized_evidence_snippets = None
                if (
                    hasattr(identifier_output, "evidence_snippets")
                    and identifier_output.evidence_snippets
                ):
                    sanitized_evidence_snippets = sanitize_for_json(
                        identifier_output.evidence_snippets
                    )

                results.append(
                    EventIdentificationResult(
                        identifier_id=f"identifier_{i}",
                        event_present=identifier_output.event_present,
                        reasoning=reasoning,
                        evidence_snippets=sanitized_evidence_snippets,
                    )
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error creating EventIdentificationResult: {str(e)}"
                )
        except Exception as e:
            logger.error(f"Error processing identifier result {i}: {str(e)}")

    logger.debug(
        f"Completed parallel event identification for {event_type}, got {len(results)} valid results"
    )
    return results


async def judge_event_identification(
    event_type: str,
    identification_results: List[EventIdentificationResult],
    note_text: str,
    judge_agent: Agent,
    azure_provider=None,
    token_tracker=None,
) -> AggregatedEventIdentification:
    """
    Judge multiple event identification results to determine final event presence

    Args:
        event_type: The type of event being identified
        identification_results: List of identification results from different agents
        note_text: The extracted note text
        judge_agent: The judge agent to evaluate the results
        azure_provider: Optional Azure provider for authentication
        token_tracker: Optional token usage tracker

    Returns:
        AggregatedEventIdentification: The final identification result selected by the judge
    """
    logger.debug(
        f"Starting event identification judging for {event_type} with {len(identification_results)} results to evaluate"
    )

    # Create a formatted string of all identification results
    try:
        formatted_results = []
        for result in identification_results:
            snippets = result.evidence_snippets or []
            evidence_snippets = (
                "\n".join([f"- {snippet}" for snippet in snippets]) or "- None"
            )

            formatted_result = (
                f"""**Identifier ID**: {result.identifier_id}

"""
                f"**Event Present**: {result.event_present}\n\n"
                f"**Evidence Snippets**:\n{evidence_snippets}\n\n"
                f"**Reasoning**:\n{result.reasoning}\n"
            )

            formatted_results.append(formatted_result)

        results_text = "\n\n---\n\n".join(formatted_results)

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        results_text = "Error formatting results"

    # Create input for the judge agent
    judge_input = f"""
    
    ### EVENT TYPE: {event_type}
    
    ### PATIENT NOTE:
    {note_text}
    
    ### IDENTIFICATION RESULTS:
    {results_text}
   
    ### FINAL REMINDER:
    Make sure the final decision is correct based on the patient note.
    """

    # Set up run configuration with Azure provider if specified
    run_config = None
    if azure_provider:
        run_config = RunConfig(model_provider=azure_provider)
        logger.debug(
            f"Using Azure provider for {event_type} event identification judging"
        )

    # Create base context for the identification judge
    try:
        event_type_lower = event_type.lower()
        # Get just the term names for identification (no definitions to reduce tokens)
        all_terms = get_terms_only(event_type_lower)
        combined_definitions = "\n".join(all_terms)

        event_context = EventContext(
            event_type=event_type,
            event_definition=combined_definitions,
            grading_criteria="",  # Not needed for identification
        )

    except Exception as e:
        logger.error(f"Error creating event context for {event_type}: {str(e)}")
        raise

    # Run the judge agent
    try:
        judge_result = await Runner.run(
            judge_agent,
            judge_input,
            context=event_context,
            run_config=run_config,
        )
    except Exception as e:
        logger.error(f"Error running judge agent for {event_type}: {str(e)}")
        raise

    # Track token usage if tracker provided
    if token_tracker:
        agent_name = f"{event_type.lower()}_event_identification_judge"
        model_obj = getattr(judge_agent, "model", None)
        if model_obj and hasattr(model_obj, "model"):
            model_name = model_obj.model
        else:
            model_name = str(model_obj) if model_obj else None

        if hasattr(token_tracker, "debug_token_extraction"):
            token_tracker.debug_token_extraction(judge_result)

        token_tracker.track_usage(agent_name, judge_result, model_name)

    # Extract the judge's decision
    judged_identification = judge_result.final_output

    # Log the raw judge output for debugging
    logger.debug(f"Raw event identification judge output for {event_type}: {judged_identification}")
    logger.debug(f"Judge output type: {type(judged_identification)}")

    # Create a new AggregatedEventIdentification instance if needed
    if not isinstance(judged_identification, AggregatedEventIdentification):
        # Initialize variables
        event_present = False
        reasoning = None
        evidence_snippets = None
        
        # Handle both dict and object types
        if isinstance(judged_identification, dict):
            # Try new field name first, fall back to legacy
            event_present = judged_identification.get("event_present", False)
            if not event_present and "identified_events" in judged_identification:
                # Legacy format - convert list to boolean
                event_present = bool(judged_identification.get("identified_events", []))
            reasoning = judged_identification.get("reasoning", None)
            evidence_snippets = judged_identification.get("evidence_snippets", None)
        else:
            # Try new field name first, fall back to legacy
            event_present = getattr(judged_identification, "event_present", False)
            if not event_present and hasattr(judged_identification, "identified_events"):
                # Legacy format - convert list to boolean
                event_present = bool(getattr(judged_identification, "identified_events", []))
            reasoning = getattr(judged_identification, "reasoning", None)
            evidence_snippets = getattr(judged_identification, "evidence_snippets", None)

        judged_identification = AggregatedEventIdentification(
            event_present=event_present,
            reasoning=reasoning,
            evidence_snippets=evidence_snippets,
        )

    # Collect evidence snippets if not already provided
    if not judged_identification.evidence_snippets:
        all_evidence = []
        for result in identification_results:
            if result.evidence_snippets:
                all_evidence.extend(result.evidence_snippets)

        # Remove duplicates while preserving order
        unique_evidence = []
        for evidence in all_evidence:
            if evidence not in unique_evidence:
                unique_evidence.append(evidence)

        judged_identification.evidence_snippets = unique_evidence

    logger.debug(f"Completed event identification judging for {event_type}")
    return judged_identification


async def run_parallel_temporality_classification(
    event_type: str,
    note_text: str,
    classifier_agents: List[Agent],
    azure_provider=None,
    token_tracker=None,
) -> List[TemporalityResult]:
    """
    Run multiple temporality classifier agents in parallel to classify temporal status

    Args:
        event_type: The type of event to classify temporally
        note_text: The extracted note text
        classifier_agents: List of temporality classifier agents to run in parallel
        azure_provider: Optional Azure provider for authentication
        token_tracker: Optional token usage tracker

    Returns:
        List[TemporalityResult]: List of temporality classification results from different agents
    """
    logger.debug(
        f"Starting parallel temporality classification for {event_type} with {len(classifier_agents)} agents"
    )

    # Create context with the event type
    try:
        event_type_lower = event_type.lower()
        all_terms_data = get_terms_definitions_and_grades(event_type_lower)

        all_definitions = []
        for term, term_data in all_terms_data.items():
            # Add only the term name for temporality classification
            all_definitions.append(term)

        combined_definitions = "\n".join(all_definitions)

        event_context = EventContext(
            event_type=event_type,
            event_definition=combined_definitions,
            grading_criteria="",  # Not needed for temporality classification
        )

    except Exception as e:
        logger.error(f"Error creating event context for {event_type}: {str(e)}")
        raise

    # Set up run configuration with Azure provider if specified
    run_config = None
    if azure_provider:
        run_config = RunConfig(model_provider=azure_provider)

    # Create input for classifier agents
    classifier_input = note_text

    # Run all classifier agents in parallel
    classification_tasks = []
    for i, agent in enumerate(classifier_agents):
        try:
            # Add staggered delays to prevent overwhelming the API
            if i > 0:
                stagger_delay = min(i * 0.1, 1.0)  # Max 1 second delay
                logger.debug(
                    f"Adding {stagger_delay:.2f}s stagger delay for agent {i} of {event_type}"
                )
                await asyncio.sleep(stagger_delay)

            logger.debug(f"Starting agent {i} for {event_type}")
            classification_tasks.append(
                Runner.run(
                    agent,
                    classifier_input,
                    context=event_context,
                    run_config=run_config,
                )
            )
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")

    # Run all classification tasks concurrently
    try:
        classification_results = await asyncio.gather(
            *classification_tasks, return_exceptions=True
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

    # Process results
    results = []
    for i, result in enumerate(classification_results):
        if isinstance(result, Exception):
            logger.error(f"Error in classifier {i}: {str(result)}")
            continue

        try:
            # Log the complete agent output
            log_agent_output(
                "TEMPORALITY_CLASSIFICATION", "Classifier", i, result, event_type
            )

            # Track token usage if tracker provided
            if token_tracker:
                agent_name = f"{event_type.lower()}_temporality_classifier_{i}"
                model_obj = getattr(classifier_agents[i], "model", None)
                if model_obj and hasattr(model_obj, "model"):
                    model_name = model_obj.model
                else:
                    model_name = str(model_obj) if model_obj else None

                if hasattr(token_tracker, "debug_token_extraction"):
                    token_tracker.debug_token_extraction(result)

                token_tracker.track_usage(agent_name, result, model_name)

            # Extract the classifier output
            classifier_output = result.final_output

            # Create a TemporalityResult object
            try:
                reasoning = ""
                if (
                    hasattr(classifier_output, "reasoning")
                    and classifier_output.reasoning
                ):
                    reasoning = classifier_output.reasoning

                sanitized_evidence_snippets = None
                if (
                    hasattr(classifier_output, "evidence_snippets")
                    and classifier_output.evidence_snippets
                ):
                    sanitized_evidence_snippets = sanitize_for_json(
                        classifier_output.evidence_snippets
                    )

                results.append(
                    TemporalityResult(
                        classifier_id=f"classifier_{i}",
                        past_events=classifier_output.past_events,
                        current_events=classifier_output.current_events,
                        reasoning=reasoning,
                        evidence_snippets=sanitized_evidence_snippets,
                    )
                )
            except Exception as e:
                logger.error(f"Unexpected error creating TemporalityResult: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing classifier result {i}: {str(e)}")

    logger.debug(
        f"Completed parallel temporality classification for {event_type}, got {len(results)} valid results"
    )
    return results


async def judge_temporality_classification(
    event_type: str,
    temporality_results: List[TemporalityResult],
    note_text: str,
    judge_agent: Agent,
    azure_provider=None,
    token_tracker=None,
) -> AggregatedTemporality:
    """
    Judge multiple temporality classification results to determine final temporal status

    Args:
        event_type: The type of event being classified temporally
        temporality_results: List of temporality results from different agents
        note_text: The extracted note text
        judge_agent: The judge agent to evaluate the results
        azure_provider: Optional Azure provider for authentication
        token_tracker: Optional token usage tracker

    Returns:
        AggregatedTemporality: The final temporal classification result selected by the judge
    """
    logger.debug(
        f"Starting temporality classification judging for {event_type} with {len(temporality_results)} results to evaluate"
    )

    # Create a formatted string of all temporality results
    try:
        formatted_results = []
        for result in temporality_results:
            past_events = (
                "\n".join([f"- {event}" for event in result.past_events]) or "- None"
            )
            current_events = (
                "\n".join([f"- {event}" for event in result.current_events]) or "- None"
            )
            snippets = result.evidence_snippets or []
            evidence_snippets = (
                "\n".join([f"- {snippet}" for snippet in snippets]) or "- None"
            )

            formatted_result = (
                f"""**Classifier ID**: {result.classifier_id}

"""
                f"**Past Events**:\n{past_events}\n\n"
                f"**Current Events**:\n{current_events}\n\n"
                f"**Evidence Snippets**:\n{evidence_snippets}\n\n"
                f"**Reasoning**:\n{result.reasoning}\n"
            )

            formatted_results.append(formatted_result)

        results_text = "\n\n---\n\n".join(formatted_results)

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        results_text = "Error formatting results"

    # Create input for the judge agent
    judge_input = f"""
    
    ### EVENT TYPE: {event_type}
    
    ### PATIENT NOTE:
    {note_text}
    
    ### TEMPORALITY CLASSIFICATION RESULTS:
    {results_text}
   
    ### FINAL REMINDER:
    Make sure the final temporal classification is correct based on the patient note.
    """

    # Set up run configuration with Azure provider if specified
    run_config = None
    if azure_provider:
        run_config = RunConfig(model_provider=azure_provider)
        logger.debug(f"Using Azure provider for {event_type} temporality judging")

    # Create base context for the temporality judge
    try:
        event_type_lower = event_type.lower()
        # Get just the term names for temporality classification (no definitions to reduce tokens)
        all_terms = get_terms_only(event_type_lower)
        combined_definitions = "\n".join(all_terms)

        event_context = EventContext(
            event_type=event_type,
            event_definition=combined_definitions,
            grading_criteria="",  # Not needed for temporality judging
        )

    except Exception as e:
        logger.error(f"Error creating event context for {event_type}: {str(e)}")
        raise

    # Run the judge agent
    try:
        judge_result = await Runner.run(
            judge_agent,
            judge_input,
            context=event_context,
            run_config=run_config,
        )
    except Exception as e:
        logger.error(f"Error running judge agent for {event_type}: {str(e)}")
        raise

    # Track token usage if tracker provided
    if token_tracker:
        agent_name = f"{event_type.lower()}_temporality_judge"
        model_obj = getattr(judge_agent, "model", None)
        if model_obj and hasattr(model_obj, "model"):
            model_name = model_obj.model
        else:
            model_name = str(model_obj) if model_obj else None

        if hasattr(token_tracker, "debug_token_extraction"):
            token_tracker.debug_token_extraction(judge_result)

        token_tracker.track_usage(agent_name, judge_result, model_name)

    # Extract the judge's decision
    judged_temporality = judge_result.final_output

    # Log the raw judge output for debugging
    logger.debug(f"Raw temporality judge output for {event_type}: {judged_temporality}")
    logger.debug(f"Judge output type: {type(judged_temporality)}")

    # Create a new AggregatedTemporality instance if needed
    if not isinstance(judged_temporality, AggregatedTemporality):
        # Handle both dict and object types
        if isinstance(judged_temporality, dict):
            past_events = judged_temporality.get("past_events", [])
            current_events = judged_temporality.get("current_events", [])
            reasoning = judged_temporality.get("reasoning", None)
            evidence_snippets = judged_temporality.get("evidence_snippets", None)
        else:
            past_events = getattr(judged_temporality, "past_events", [])
            current_events = getattr(judged_temporality, "current_events", [])
            reasoning = getattr(judged_temporality, "reasoning", None)
            evidence_snippets = getattr(judged_temporality, "evidence_snippets", None)

        judged_temporality = AggregatedTemporality(
            past_events=past_events,
            current_events=current_events,
            reasoning=reasoning,
            evidence_snippets=evidence_snippets,
        )

    # Collect evidence snippets if not already provided
    if not judged_temporality.evidence_snippets:
        all_evidence = []
        for result in temporality_results:
            if result.evidence_snippets:
                all_evidence.extend(result.evidence_snippets)

        # Remove duplicates while preserving order
        unique_evidence = []
        for evidence in all_evidence:
            if evidence not in unique_evidence:
                unique_evidence.append(evidence)

        judged_temporality.evidence_snippets = unique_evidence

    logger.debug(f"Completed temporality classification judging for {event_type}")
    return judged_temporality
