"""
Event processing models for the enhanced workflow implementation.
These models are used for intermediate processing steps and tracking metadata.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class EventProcessingResult(BaseModel):
    """Model for tracking the processing of a single event type"""

    event_type: str = Field(description="Type of event (e.g., pneumonitis)")
    past_events: List[str] = Field(
        default_factory=list, description="Past events identified"
    )
    current_events: List[str] = Field(
        default_factory=list, description="Current/ongoing events identified"
    )
    evidence_snippets: Optional[Dict[str, str]] = Field(
        default=None, description="Evidence snippets for each event"
    )
    past_grade: int = Field(default=0, description="Grade of past event if present")
    current_grade: int = Field(
        default=0, description="Grade of current event if present"
    )
    max_grade: int = Field(default=0, description="Maximum grade (past or current)")
    attribution: int = Field(
        default=0, description="Attribution to immunotherapy (1 = yes, 0 = no)"
    )
    attribution_evidence: str = Field(
        default="", description="Evidence for attribution assessment"
    )
    certainty: int = Field(default=0, description="Certainty level (0-4 scale)")
    certainty_evidence: str = Field(
        default="", description="Evidence for certainty assessment"
    )
    processing_time: float = Field(
        default=0.0, description="Time taken to process this event type in seconds"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if processing failed"
    )


class EventProcessingMetadata(BaseModel):
    """Model for tracking metadata about event processing"""

    token_usage: Dict[str, Dict[str, Dict[str, int]]] = Field(
        default_factory=dict, description="Token usage by agent and event type"
    )
    processing_times: Dict[str, float] = Field(
        default_factory=dict, description="Processing time by event type"
    )
    errors: Dict[str, str] = Field(
        default_factory=dict, description="Errors by event type"
    )

    def track_token_usage(
        self,
        event_type: str,
        agent_name: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Track token usage for a specific event type and agent"""
        if event_type not in self.token_usage:
            self.token_usage[event_type] = {}

        self.token_usage[event_type][agent_name] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def track_processing_time(self, event_type: str, processing_time: float) -> None:
        """Track processing time for a specific event type"""
        self.processing_times[event_type] = processing_time

    def track_error(self, event_type: str, error_message: str) -> None:
        """Track error for a specific event type"""
        self.errors[event_type] = error_message

    def get_total_token_usage(self) -> Dict[str, int]:
        """Get total token usage across all event types and agents"""
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for event_type in self.token_usage:
            for agent_name in self.token_usage[event_type]:
                usage = self.token_usage[event_type][agent_name]
                total_prompt_tokens += usage["prompt_tokens"]
                total_completion_tokens += usage["completion_tokens"]

        return {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        }

    def get_total_processing_time(self) -> float:
        """Get total processing time across all event types"""
        return sum(self.processing_times.values())

    def has_errors(self) -> bool:
        """Check if any errors occurred during processing"""
        return bool(self.errors)
