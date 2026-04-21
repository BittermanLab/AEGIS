"""
EventContext module for holding event-specific information for dynamic prompt generation.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


class EventContext:
    """Context class that holds event type information for dynamic prompt generation."""

    def __init__(
        self,
        event_type: str,
        event_definition: str = "",
        grading_criteria: str = "",
        temporal_context: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        """
        Initialize EventContext with event specific information.

        Args:
            event_type: The type of event being processed (e.g. Pneumonitis, Myocarditis, etc.)
            event_definition: The CTCAE definition of the event
            grading_criteria: The CTCAE grading criteria for the event
            temporal_context: The temporal context, either "past" or "current" or None
            request_id: Unique ID for tracking requests across threads
        """
        if not event_type:
            error_msg = "ERROR: event_type cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate event_definition is not empty when creating context
        if not event_definition:
            error_msg = f"ERROR: event_definition cannot be empty for {event_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Log creation
        logger_prefix = f"[RequestID: {request_id}]" if request_id else ""
        logger.info(
            f"{logger_prefix} Creating EventContext: event_type={event_type}, "
            f"definition_length={len(event_definition)}, "
            f"criteria_length={len(grading_criteria)}, "
            f"temporal_context={temporal_context}"
        )

        self.event_type = event_type
        self.event_definition = event_definition
        self.grading_criteria = grading_criteria
        self.temporal_context = temporal_context  # "past" or "current" or None
        self.request_id = request_id
