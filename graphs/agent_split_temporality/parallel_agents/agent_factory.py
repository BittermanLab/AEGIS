"""
Factory module for creating specialized agents for event processing.
"""

import logging
import os
from typing import Dict, Any, List, Optional

# Import Agent from external agents package, not local
from agents import Agent, set_tracing_disabled
from agents.model_settings import ModelSettings

# Local import for EventContext
from .event_context import EventContext

# Import new split workflow prompts
from .prompts.dynamic.identifier import (
    dynamic_identifier_instructions,
)
from .prompts.dynamic.temporality_classifier import (
    dynamic_temporality_classifier_instructions,
)
from .prompts.dynamic.temporality_judge import (
    dynamic_temporality_judge_instructions,
)

# Import legacy prompts for backwards compatibility
from .prompts.dynamic.temporal_identifier import (
    dynamic_temporal_identifier_instructions,
)
from .prompts.dynamic.grader import dynamic_grader_instructions
from .prompts.dynamic.attribution import dynamic_attribution_instructions
from .prompts.dynamic.certainty import dynamic_certainty_instructions

# Import dynamic prompts
from .prompts.dynamic.identification_judge import (
    dynamic_identification_judge_instructions,
)
from .prompts.dynamic.grading_judge import dynamic_grading_judge_instructions

# Import existing prompts that aren't yet converted to dynamic
from .prompts.meta_judge_prompt import META_JUDGE_PROMPT
from .prompts.overview_prompt import OVERVIEW_PROMPT

# Import models
from .models.enhanced_output_models import (
    EventIdentification,
    TemporalityClassification,
    TemporalIdentification,
    AttributionDetection,
    CertaintyAssessment,
    EventGrading,
)
from .models.enhanced_judge_models import (
    AggregatedEventIdentification,
    AggregatedTemporality,
    AggregatedIdentification,
    AggregatedGrading,
    MetaJudgeFeedback,
)

# Import utilities
from .utils.ctcae_utils import (
    get_ctcae_subset,
    format_ctcae_grades_for_prompt,
    get_terms_definitions_and_grades,
)
from .utils.model_config import get_model_for_role, get_model_settings, is_ollama_model
from .utils.unified_provider import get_model_factory, UnifiedModelFactory

logger = logging.getLogger(__name__)

# Set httpx logger to WARNING level to completely suppress HTTP request logs
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

# Disable tracing for all OpenAI agent executions
set_tracing_disabled(True)
logger.info("OpenAI Agents tracing disabled in agent factory module")


def create_enhanced_event_agents_with_judges(
    model_config: str = "default",
    azure_provider=None,
    request_id: str = None,
    event_types: Optional[List[str]] = None,
    provider_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Agent]]:
    """Create specialized agents for each event type with temporal, attribution, certainty, and judge capabilities.

    Uses dynamic prompts with EventContext to focus each agent on its specific event type.

    Args:
        model_config: Configuration for which models to use (default, economy, high_quality, premium)
        azure_provider: Optional Azure provider for authentication (deprecated, use provider_config instead)
        request_id: Unique request ID for thread tracking
        event_types: Optional list of event types to process. If None, default event types will be used.
        provider_config: Optional configuration for model providers
            For Azure: {'azure_endpoint': '...', 'azure_api_version': '...'}
            For Ollama: {'ollama_endpoint': '...'}

    Returns:
        Dictionary mapping event types to dictionaries of specialized agents
    """
    event_agents = {}

    # Generate a request ID if not provided
    if not request_id:
        import uuid

        request_id = str(uuid.uuid4())

    # Log the request ID
    logger.debug(f"[RequestID: {request_id}] Creating enhanced event agents")

    # Initialize the unified model factory
    model_factory = get_model_factory(provider_config)
    logger.info(f"[RequestID: {request_id}] Using model configuration: {model_config}")

    # Detect provider type for logging
    from .utils.model_config import get_model_for_role, is_ollama_model

    for role in ["extractor", "parallel_agent", "judge_agent", "general"]:
        model_name = get_model_for_role(model_config, role)
        if is_ollama_model(model_name):
            endpoint = (
                provider_config.get("ollama_endpoint") if provider_config else None
            )
            logger.info(
                f"[RequestID: {request_id}] Ollama provider will be used for role '{role}' with model '{model_name}' at endpoint '{endpoint}'"
            )
        else:
            endpoint = (
                provider_config.get("azure_endpoint") if provider_config else None
            )
            logger.info(
                f"[RequestID: {request_id}] Azure provider will be used for role '{role}' with model '{model_name}' at endpoint '{endpoint}'"
            )

    ##### BASE CONTEXT #####
    for event_type in event_types:
        # Get all terms, definitions and grades for this event type
        event_type_lower = event_type.lower()
        logger.info(
            f"[RequestID: {request_id}] Getting CTCAE data for event type: {event_type_lower}"
        )

        # Add error handling for CTCAE data retrieval
        try:
            all_terms_data = get_terms_definitions_and_grades(event_type_lower)
        except Exception as e:
            error_msg = f"[RequestID: {request_id}] ERROR retrieving CTCAE data for {event_type}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Hard validation - must have terms data
        if not all_terms_data:
            error_msg = (
                f"[RequestID: {request_id}] ERROR: No terms found for {event_type}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Format all terms, definitions, and grades for inclusion in the agent context
        all_definitions = []
        formatted_grading_criteria = []

        logger.info(
            f"[RequestID: {request_id}] Processing {len(all_terms_data)} terms for {event_type}"
        )

        # Check if the main event type is in the data
        main_term = event_type.capitalize()
        if main_term not in all_terms_data:
            logger.warning(
                f"[RequestID: {request_id}] Main term '{main_term}' not found in CTCAE data"
            )
            # Look for it with different capitalization
            main_term_found = False
            for term in all_terms_data.keys():
                if term.lower() == main_term.lower():
                    main_term = term  # Use the actual capitalization from data
                    main_term_found = True
                    logger.info(
                        f"[RequestID: {request_id}] Found main term with different capitalization: {main_term}"
                    )
                    break

            if not main_term_found:
                logger.warning(
                    f"[RequestID: {request_id}] Main term '{main_term}' not found in any form in CTCAE data"
                )
        else:
            logger.info(
                f"[RequestID: {request_id}] Main term '{main_term}' found in CTCAE data"
            )

        for term, term_data in all_terms_data.items():
            # Add only the term name without definition
            all_definitions.append(term)

            logger.debug(f"[RequestID: {request_id}] Added term: {term}")

            # Check for empty grade values
            missing_grades = []
            for grade in range(1, 6):
                grade_value = term_data.get(f"Grade {grade}", "")
                if not grade_value and grade < 5:  # Grade 5 is often empty
                    missing_grades.append(grade)

            if missing_grades:
                logger.warning(
                    f"[RequestID: {request_id}] Missing grades {missing_grades} for term: {term}"
                )

            term_grading = [
                f"TERM: {term}",
                f"Grade 1: {term_data.get('Grade 1', '')}",
                f"Grade 2: {term_data.get('Grade 2', '')}",
                f"Grade 3: {term_data.get('Grade 3', '')}",
                f"Grade 4: {term_data.get('Grade 4', '')}",
                f"Grade 5: {term_data.get('Grade 5', '')}",
            ]
            formatted_grading_criteria.append("\n".join(term_grading))

        # Hard validation - must have term names
        if not all_definitions:
            error_msg = f"[RequestID: {request_id}] ERROR: No term names generated for {event_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Join all term names and detailed grading criteria
        combined_definitions = "\n".join(all_definitions)
        combined_grading_criteria = "\n\n".join(formatted_grading_criteria)

        # Hard validation - must have non-empty combined term names
        if not combined_definitions.strip():
            error_msg = f"[RequestID: {request_id}] ERROR: Empty combined term names for {event_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Log the formatted data
        logger.info(
            f"[RequestID: {request_id}] Combined term names length: {len(combined_definitions)}"
        )
        logger.info(
            f"[RequestID: {request_id}] Combined grading criteria length: {len(combined_grading_criteria)}"
        )
        if combined_definitions:
            logger.info(
                f"[RequestID: {request_id}] First 100 chars of definitions: {combined_definitions[:100]}"
            )
            # Log individual definitions to debug the content
            if len(all_definitions) > 0:
                logger.debug(
                    f"[RequestID: {request_id}] First definition: {all_definitions[0]}"
                )
                if len(all_definitions) > 1:
                    logger.debug(
                        f"[RequestID: {request_id}] Second definition: {all_definitions[1]}"
                    )
        else:
            error_msg = f"[RequestID: {request_id}] ERROR: Empty combined definitions for {event_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if combined_grading_criteria:
            logger.info(
                f"[RequestID: {request_id}] First 100 chars of grading criteria: {combined_grading_criteria[:100]}"
            )
            # Log individual grading criteria to debug the content
            if len(formatted_grading_criteria) > 0:
                logger.debug(
                    f"[RequestID: {request_id}] First term grading: {formatted_grading_criteria[0][:100]}..."
                )
                if len(formatted_grading_criteria) > 1:
                    logger.debug(
                        f"[RequestID: {request_id}] Second term grading: {formatted_grading_criteria[1][:100]}..."
                    )
        else:
            logger.warning(
                f"[RequestID: {request_id}] Empty combined grading criteria for {event_type}"
            )

        # Create event context with comprehensive event-specific information and request ID
        base_context = EventContext(
            event_type=event_type,
            event_definition=combined_definitions,
            grading_criteria=combined_grading_criteria,
            request_id=request_id,
        )

        # Hard validation - verify context has data
        if not base_context.event_definition:
            error_msg = f"[RequestID: {request_id}] ERROR: Empty event_definition in context for {event_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Verify definition length in context matches our combined_definitions
        if len(base_context.event_definition) != len(combined_definitions):
            error_msg = f"[RequestID: {request_id}] ERROR: Definition length mismatch for {event_type}. combined_definitions={len(combined_definitions)}, context.event_definition={len(base_context.event_definition)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(
            f"[RequestID: {request_id}] Created EventContext with data: event_type={base_context.event_type}, definition_length={len(base_context.event_definition)}, criteria_length={len(base_context.grading_criteria)}"
        )

        ##### TEMPORAL IDENTIFIER AGENTS #####

        # Helper function to get the appropriate model for a role
        def get_model_for_agent(role):
            model_name = get_model_for_role(model_config, role)
            # Check if we need to create a model instance
            if isinstance(model_name, str):
                return model_factory.get_model(model_name)
            return model_name

        # Create multiple temporal identifier agents with slightly different prompts
        temporal_identifier_agents = []
        identifier_variants = [
            (
                "Comprehensive",
                "Focus on comprehensive identification of all possible events.",
            ),
            ("Specific", "Focus on specific, well-evidenced events only."),
            ("Balanced", "Balance comprehensiveness with specificity."),
        ]

        for variant_name, variant_instruction in identifier_variants:
            # Create a dynamic temporal identifier agent using the dynamic instructions
            temporal_identifier_agent = Agent(
                name=f"{event_type} {variant_name} Identifier",
                instructions=dynamic_temporal_identifier_instructions,
                output_type=TemporalIdentification,
                model=get_model_for_agent("parallel_agent"),
            )
            temporal_identifier_agents.append(temporal_identifier_agent)

        # Create identification judge agent with dynamic instructions
        identification_judge_agent = Agent(
            name=f"{event_type} Identification Judge",
            instructions=dynamic_identification_judge_instructions,
            output_type=AggregatedIdentification,
            model=get_model_for_agent("judge_agent"),
            model_settings=get_model_settings(
                get_model_for_role(model_config, "judge_agent")
            ),
        )

        ##### GRADING AGENTS #####

        # Create past context from the base context with temporal context set to "past"
        past_context = EventContext(
            event_type=event_type,
            event_definition=combined_definitions,
            grading_criteria=combined_grading_criteria,
            temporal_context="past",
        )

        # Create multiple grader agents for past events with slight prompt variants
        past_grader_agents = []
        grader_variants = [
            (
                "Conservative",
                "Be conservative in grading, prioritizing patient safety.",
            ),
            ("Evidence-based", "Focus strictly on evidence in the note for grading."),
            ("Guidelines-focused", "Prioritize strict adherence to CTCAE guidelines."),
        ]

        for variant_name, variant_instruction in grader_variants:
            # Create past grader agent with dynamic instructions
            past_grader_agent = Agent(
                name=f"{event_type} Past {variant_name} Grader",
                instructions=dynamic_grader_instructions,
                output_type=EventGrading,
                model=get_model_for_agent("parallel_agent"),
                model_settings=get_model_settings(
                    get_model_for_role(model_config, "parallel_agent")
                ),
            )
            past_grader_agents.append(past_grader_agent)

        # Create multiple grader agents for current events with slight prompt variants
        current_grader_agents = []

        # Create current context from the base context with temporal context set to "current"
        current_context = EventContext(
            event_type=event_type,
            event_definition=combined_definitions,
            grading_criteria=combined_grading_criteria,
            temporal_context="current",
        )

        for variant_name, variant_instruction in grader_variants:
            # Create current grader agent with dynamic instructions
            current_grader_agent = Agent(
                name=f"{event_type} Current {variant_name} Grader",
                instructions=dynamic_grader_instructions,
                output_type=EventGrading,
                model=get_model_for_agent("parallel_agent"),
                model_settings=get_model_settings(
                    get_model_for_role(model_config, "parallel_agent")
                ),
            )
            current_grader_agents.append(current_grader_agent)

        # Create past and current grading judge agents with dynamic instructions
        past_grading_judge_agent = Agent(
            name=f"{event_type} Past Grading Judge",
            instructions=dynamic_grading_judge_instructions,
            output_type=AggregatedGrading,
            model=get_model_for_agent("judge_agent"),
            model_settings=get_model_settings(
                get_model_for_role(model_config, "judge_agent")
            ),
        )

        current_grading_judge_agent = Agent(
            name=f"{event_type} Current Grading Judge",
            instructions=dynamic_grading_judge_instructions,
            output_type=AggregatedGrading,
            model=get_model_for_agent("judge_agent"),
            model_settings=get_model_settings(
                get_model_for_role(model_config, "judge_agent")
            ),
        )

        # Create attribution agent with dynamic instructions
        attribution_agent = Agent(
            name=f"{event_type} Attribution Detector",
            instructions=dynamic_attribution_instructions,  # Use dynamic instructions
            output_type=AttributionDetection,
            model=get_model_for_agent("parallel_agent"),
            model_settings=get_model_settings(
                get_model_for_role(model_config, "parallel_agent")
            ),
        )

        # Create certainty agent with dynamic instructions
        certainty_agent = Agent(
            name=f"{event_type} Certainty Assessor",
            instructions=dynamic_certainty_instructions,  # Use dynamic instructions
            output_type=CertaintyAssessment,
            model=get_model_for_agent("parallel_agent"),
            model_settings=get_model_settings(
                get_model_for_role(model_config, "parallel_agent")
            ),
        )

        # Create meta-judge agent
        meta_judge_agent = Agent(
            name=f"{event_type} Meta-Judge",
            instructions=META_JUDGE_PROMPT,
            output_type=MetaJudgeFeedback,
            model=get_model_for_agent("judge_agent"),
            model_settings=get_model_settings(
                get_model_for_role(model_config, "judge_agent")
            ),
        )

        # Create overview agent - returns string directly
        overview_agent = Agent(
            name=f"{event_type} Overview Generator",
            instructions=OVERVIEW_PROMPT,
            output_type=str,  # Direct string output
            model=get_model_for_agent("judge_agent"),
            model_settings=get_model_settings(
                get_model_for_role(model_config, "judge_agent")
            ),
        )

        event_agents[event_type] = {
            "temporal_identifiers": temporal_identifier_agents,
            "identification_judge": identification_judge_agent,
            "past_graders": past_grader_agents,
            "current_graders": current_grader_agents,
            "past_grading_judge": past_grading_judge_agent,
            "current_grading_judge": current_grading_judge_agent,
            "attribution": attribution_agent,
            "certainty": certainty_agent,
            "meta_judge": meta_judge_agent,
            "overview": overview_agent,  # Add the overview agent
        }

    return event_agents


def create_split_temporality_agents(
    model_config: str = "default",
    azure_provider=None,
    request_id: str = None,
    event_types: Optional[List[str]] = None,
    provider_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Agent]]:
    """Create specialized agents for each event type with split temporality workflow.

    This version splits the temporal identification into separate identification and
    temporality classification steps with their own judges.

    Args:
        model_config: Configuration for which models to use (default, economy, high_quality, premium)
        azure_provider: Optional Azure provider for authentication (deprecated, use provider_config instead)
        request_id: Unique request ID for thread tracking
        event_types: Optional list of event types to process. If None, default event types will be used.
        provider_config: Optional configuration for model providers
            For Azure: {'azure_endpoint': '...', 'azure_api_version': '...'}
            For Ollama: {'ollama_endpoint': '...'}

    Returns:
        Dictionary mapping event types to dictionaries of specialized agents
    """
    event_agents = {}

    # Generate a request ID if not provided
    if not request_id:
        import uuid

        request_id = str(uuid.uuid4())

    # Log the request ID
    logger.debug(f"[RequestID: {request_id}] Creating split temporality event agents")

    # Initialize the unified model factory
    model_factory = get_model_factory(provider_config)
    logger.info(f"[RequestID: {request_id}] Using model configuration: {model_config}")

    # Detect provider type for logging
    from .utils.model_config import get_model_for_role, is_ollama_model

    for role in ["extractor", "parallel_agent", "judge_agent", "general"]:
        model_name = get_model_for_role(model_config, role)
        if is_ollama_model(model_name):
            endpoint = (
                provider_config.get("ollama_endpoint") if provider_config else None
            )
            logger.info(
                f"[RequestID: {request_id}] Ollama provider will be used for role '{role}' with model '{model_name}' at endpoint '{endpoint}'"
            )
        else:
            endpoint = (
                provider_config.get("azure_endpoint") if provider_config else None
            )
            logger.info(
                f"[RequestID: {request_id}] Azure provider will be used for role '{role}' with model '{model_name}' at endpoint '{endpoint}'"
            )

    ##### BASE CONTEXT #####
    for event_type in event_types:
        # Get all terms, definitions and grades for this event type
        event_type_lower = event_type.lower()
        logger.info(
            f"[RequestID: {request_id}] Getting CTCAE data for event type: {event_type_lower}"
        )

        # Add error handling for CTCAE data retrieval
        try:
            all_terms_data = get_terms_definitions_and_grades(event_type_lower)
        except Exception as e:
            error_msg = f"[RequestID: {request_id}] ERROR retrieving CTCAE data for {event_type}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Hard validation - must have terms data
        if not all_terms_data:
            error_msg = (
                f"[RequestID: {request_id}] ERROR: No terms found for {event_type}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Format all terms, definitions, and grades for inclusion in the agent context
        all_definitions = []
        formatted_grading_criteria = []

        logger.info(
            f"[RequestID: {request_id}] Processing {len(all_terms_data)} terms for {event_type}"
        )

        # Check if the main event type is in the data
        main_term = event_type.capitalize()
        if main_term not in all_terms_data:
            logger.warning(
                f"[RequestID: {request_id}] Main term '{main_term}' not found in CTCAE data"
            )
            # Look for it with different capitalization
            main_term_found = False
            for term in all_terms_data.keys():
                if term.lower() == main_term.lower():
                    main_term = term  # Use the actual capitalization from data
                    main_term_found = True
                    logger.info(
                        f"[RequestID: {request_id}] Found main term with different capitalization: {main_term}"
                    )
                    break

            if not main_term_found:
                logger.warning(
                    f"[RequestID: {request_id}] Main term '{main_term}' not found in any form in CTCAE data"
                )
        else:
            logger.info(
                f"[RequestID: {request_id}] Main term '{main_term}' found in CTCAE data"
            )

        for term, term_data in all_terms_data.items():
            definition = term_data.get("Definition", "")
            if not definition:
                logger.warning(
                    f"[RequestID: {request_id}] Empty definition for term: {term}"
                )
            else:
                logger.debug(
                    f"[RequestID: {request_id}] Definition for term {term}: {definition[:50]}..."
                )

            # Add only the term name without definition
            all_definitions.append(term)

            logger.debug(f"[RequestID: {request_id}] Added term: {term}")

            # Check for empty grade values
            missing_grades = []
            for grade in range(1, 6):
                grade_value = term_data.get(f"Grade {grade}", "")
                if not grade_value and grade < 5:  # Grade 5 is often empty
                    missing_grades.append(grade)

            if missing_grades:
                logger.warning(
                    f"[RequestID: {request_id}] Missing grades {missing_grades} for term: {term}"
                )

            term_grading = [
                f"TERM: {term}",
                f"Grade 1: {term_data.get('Grade 1', '')}",
                f"Grade 2: {term_data.get('Grade 2', '')}",
                f"Grade 3: {term_data.get('Grade 3', '')}",
                f"Grade 4: {term_data.get('Grade 4', '')}",
                f"Grade 5: {term_data.get('Grade 5', '')}",
            ]
            formatted_grading_criteria.append("\n".join(term_grading))

        # Hard validation - must have term names
        if not all_definitions:
            error_msg = f"[RequestID: {request_id}] ERROR: No term names generated for {event_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Join all term names and detailed grading criteria
        combined_definitions = "\n".join(all_definitions)
        combined_grading_criteria = "\n\n".join(formatted_grading_criteria)

        # Hard validation - must have non-empty combined term names
        if not combined_definitions.strip():
            error_msg = f"[RequestID: {request_id}] ERROR: Empty combined term names for {event_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Log the formatted data
        logger.info(
            f"[RequestID: {request_id}] Combined term names length: {len(combined_definitions)}"
        )
        logger.info(
            f"[RequestID: {request_id}] Combined grading criteria length: {len(combined_grading_criteria)}"
        )

        # Create event context with comprehensive event-specific information and request ID
        base_context = EventContext(
            event_type=event_type,
            event_definition=combined_definitions,
            grading_criteria=combined_grading_criteria,
            request_id=request_id,
        )

        # Hard validation - verify context has data
        if not base_context.event_definition:
            error_msg = f"[RequestID: {request_id}] ERROR: Empty event_definition in context for {event_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(
            f"[RequestID: {request_id}] Created EventContext with data: event_type={base_context.event_type}, definition_length={len(base_context.event_definition)}, criteria_length={len(base_context.grading_criteria)}"
        )

        ##### SPLIT TEMPORALITY WORKFLOW #####

        # Helper function to get the appropriate model for a role
        def get_model_for_agent(role):
            model_name = get_model_for_role(model_config, role)
            # Check if we need to create a model instance
            if isinstance(model_name, str):
                return model_factory.get_model(model_name)
            return model_name

        # Step 1: Create multiple event identifier agents (without temporal classification)
        event_identifier_agents = []
        identifier_variants = [
            (
                "Comprehensive",
                "Focus on comprehensive identification of all possible events.",
            ),
            ("Specific", "Focus on specific, well-evidenced events only."),
            ("Balanced", "Balance comprehensiveness with specificity."),
        ]

        for variant_name, variant_instruction in identifier_variants:
            # Create event identifier agent using the new dynamic instructions
            event_identifier_agent = Agent(
                name=f"{event_type} {variant_name} Event Identifier",
                instructions=dynamic_identifier_instructions,
                output_type=EventIdentification,
                model=get_model_for_agent("parallel_agent"),
            )
            event_identifier_agents.append(event_identifier_agent)

        # Step 2: Create event identification judge agent
        event_identification_judge_agent = Agent(
            name=f"{event_type} Event Identification Judge",
            instructions=dynamic_identification_judge_instructions,  # We can reuse this for now
            output_type=AggregatedEventIdentification,
            model=get_model_for_agent("judge_agent"),
            model_settings=get_model_settings(
                get_model_for_role(model_config, "judge_agent")
            ),
        )

        # Step 3: Create multiple temporality classifier agents
        temporality_classifier_agents = []
        temporality_variants = [
            ("Conservative", "Be conservative in temporal classification."),
            ("Evidence-based", "Focus strictly on temporal evidence in the note."),
            (
                "Guidelines-focused",
                "Prioritize strict adherence to temporal indicators.",
            ),
        ]

        for variant_name, variant_instruction in temporality_variants:
            # Create temporality classifier agent using the new dynamic instructions
            temporality_classifier_agent = Agent(
                name=f"{event_type} {variant_name} Temporality Classifier",
                instructions=dynamic_temporality_classifier_instructions,
                output_type=TemporalityClassification,
                model=get_model_for_agent("parallel_agent"),
            )
            temporality_classifier_agents.append(temporality_classifier_agent)

        # Step 4: Create temporality judge agent
        temporality_judge_agent = Agent(
            name=f"{event_type} Temporality Judge",
            instructions=dynamic_temporality_judge_instructions,
            output_type=AggregatedTemporality,
            model=get_model_for_agent("judge_agent"),
            model_settings=get_model_settings(
                get_model_for_role(model_config, "judge_agent")
            ),
        )

        ##### GRADING AGENTS (same as original) #####

        # Create past context from the base context with temporal context set to "past"
        past_context = EventContext(
            event_type=event_type,
            event_definition=combined_definitions,
            grading_criteria=combined_grading_criteria,
            temporal_context="past",
        )

        # Create multiple grader agents for past events with slight prompt variants
        past_grader_agents = []
        grader_variants = [
            (
                "Conservative",
                "Be conservative in grading, prioritizing patient safety.",
            ),
            ("Evidence-based", "Focus strictly on evidence in the note for grading."),
            ("Guidelines-focused", "Prioritize strict adherence to CTCAE guidelines."),
        ]

        for variant_name, variant_instruction in grader_variants:
            # Create past grader agent with dynamic instructions
            past_grader_agent = Agent(
                name=f"{event_type} Past {variant_name} Grader",
                instructions=dynamic_grader_instructions,
                output_type=EventGrading,
                model=get_model_for_agent("parallel_agent"),
                model_settings=get_model_settings(
                    get_model_for_role(model_config, "parallel_agent")
                ),
            )
            past_grader_agents.append(past_grader_agent)

        # Create multiple grader agents for current events with slight prompt variants
        current_grader_agents = []

        # Create current context from the base context with temporal context set to "current"
        current_context = EventContext(
            event_type=event_type,
            event_definition=combined_definitions,
            grading_criteria=combined_grading_criteria,
            temporal_context="current",
        )

        for variant_name, variant_instruction in grader_variants:
            # Create current grader agent with dynamic instructions
            current_grader_agent = Agent(
                name=f"{event_type} Current {variant_name} Grader",
                instructions=dynamic_grader_instructions,
                output_type=EventGrading,
                model=get_model_for_agent("parallel_agent"),
                model_settings=get_model_settings(
                    get_model_for_role(model_config, "parallel_agent")
                ),
            )
            current_grader_agents.append(current_grader_agent)

        # Create past and current grading judge agents with dynamic instructions
        past_grading_judge_agent = Agent(
            name=f"{event_type} Past Grading Judge",
            instructions=dynamic_grading_judge_instructions,
            output_type=AggregatedGrading,
            model=get_model_for_agent("judge_agent"),
            model_settings=get_model_settings(
                get_model_for_role(model_config, "judge_agent")
            ),
        )

        current_grading_judge_agent = Agent(
            name=f"{event_type} Current Grading Judge",
            instructions=dynamic_grading_judge_instructions,
            output_type=AggregatedGrading,
            model=get_model_for_agent("judge_agent"),
            model_settings=get_model_settings(
                get_model_for_role(model_config, "judge_agent")
            ),
        )

        # Create attribution agent with dynamic instructions
        attribution_agent = Agent(
            name=f"{event_type} Attribution Detector",
            instructions=dynamic_attribution_instructions,  # Use dynamic instructions
            output_type=AttributionDetection,
            model=get_model_for_agent("parallel_agent"),
            model_settings=get_model_settings(
                get_model_for_role(model_config, "parallel_agent")
            ),
        )

        # Create certainty agent with dynamic instructions
        certainty_agent = Agent(
            name=f"{event_type} Certainty Assessor",
            instructions=dynamic_certainty_instructions,  # Use dynamic instructions
            output_type=CertaintyAssessment,
            model=get_model_for_agent("parallel_agent"),
            model_settings=get_model_settings(
                get_model_for_role(model_config, "parallel_agent")
            ),
        )

        # Create meta-judge agent
        meta_judge_agent = Agent(
            name=f"{event_type} Meta-Judge",
            instructions=META_JUDGE_PROMPT,
            output_type=MetaJudgeFeedback,
            model=get_model_for_agent("judge_agent"),
            model_settings=get_model_settings(
                get_model_for_role(model_config, "judge_agent")
            ),
        )

        # Create overview agent - returns string directly
        overview_agent = Agent(
            name=f"{event_type} Overview Generator",
            instructions=OVERVIEW_PROMPT,
            output_type=str,  # Direct string output
            model=get_model_for_agent("judge_agent"),
            model_settings=get_model_settings(
                get_model_for_role(model_config, "judge_agent")
            ),
        )

        event_agents[event_type] = {
            # Split temporality workflow
            "event_identifiers": event_identifier_agents,
            "event_identification_judge": event_identification_judge_agent,
            "temporality_classifiers": temporality_classifier_agents,
            "temporality_judge": temporality_judge_agent,
            # Legacy grading workflow (unchanged)
            "past_graders": past_grader_agents,
            "current_graders": current_grader_agents,
            "past_grading_judge": past_grading_judge_agent,
            "current_grading_judge": current_grading_judge_agent,
            "attribution": attribution_agent,
            "certainty": certainty_agent,
            "meta_judge": meta_judge_agent,
            "overview": overview_agent,
        }

    return event_agents
