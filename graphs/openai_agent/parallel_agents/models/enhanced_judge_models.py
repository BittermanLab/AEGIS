"""
Enhanced judge-based data models for the parallel event processing workflow.
These models support multiple identifiers, graders, and judge-based evaluation.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

from .enhanced_output_models import (
    AttributionDetection,
    CertaintyAssessment,
)


class IdentificationResult(BaseModel):
    """Result from a single identifier agent."""

    identifier_id: str
    past_events: List[str] = Field(
        default_factory=list, description="Past events identified by the identifier"
    )
    current_events: List[str] = Field(
        default_factory=list, description="Current events identified by the identifier"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Reasoning for the identification of the events",
    )
    evidence_snippets: Optional[List[str]] = Field(
        default=None,
        description="Exact string match for any evidence snippets as a list of quotes",
    )


class AggregatedIdentification(BaseModel):
    """Aggregated identification results after judge evaluation."""

    past_events: List[str] = Field(
        default_factory=list, description="Past events identified by the judge"
    )
    current_events: List[str] = Field(
        default_factory=list, description="Current events identified by the judge"
    )
    reasoning: Optional[str] = Field(
        default=None, description="Reasoning for the identification of the events"
    )
    evidence_snippets: Optional[List[str]] = Field(
        default=None,
        description="Aggregated exact string match for any evidence snippets as a list of quotes",
    )


class GradingResult(BaseModel):
    """Result from a single grader agent."""

    grader_id: str
    event_name: str
    grade: int
    temporal_context: str
    rationale: Optional[str] = Field(
        default=None,
        description="Reasoning for the grade chosen with reference to the evidence.",
    )
    evidence_snippets: Optional[List[str]] = Field(
        default=None,
        description="Exact string match for any evidence snippets as a list of quotes",
    )


class AggregatedGrading(BaseModel):
    """Aggregated grading results after judge evaluation."""

    event_name: str
    grade: int
    temporal_context: str  # "past" or "current"
    rationale: Optional[str] = Field(
        default=None,
        description="Reasoning for the grade chosen with reference to the evidence.",
    )
    evidence_snippets: Optional[List[str]] = Field(
        default=None,
        description="Aggregated exact string match for any evidence snippets as a list of quotes",
    )


class MetaJudgeFeedback(BaseModel):
    """Feedback from the meta-judge agent."""

    satisfaction_score: Optional[float] = None  # 0.0-1.0 scale
    event_type: str
    should_reprocess: Optional[bool] = None
    needs_improvement: Optional[bool] = None  # Field without default value
    user_overview: Optional[str] = Field(
        default=None,
        description="A concise summary of the findings for clinical users, highlighting key positives and negatives.",
    )
    identification_feedback: Optional[str] = None
    grading_feedback: Optional[str] = None
    attribution_feedback: Optional[str] = None
    certainty_feedback: Optional[str] = None
    evidence_evaluation: Optional[str] = None  # Evaluation of evidence quality
    improvement_areas: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None


class EnhancedEventProcessingResult(BaseModel):
    """Comprehensive result including all stages of event processing with judge evaluation."""

    event_type: str
    # Identification stage
    identification_results: List[IdentificationResult] = Field(default_factory=list)
    judged_identification: Optional[AggregatedIdentification] = None
    # Grading stage - past
    past_grading_results: List[GradingResult] = Field(default_factory=list)
    judged_past_grading: Optional[AggregatedGrading] = None
    # Grading stage - current
    current_grading_results: List[GradingResult] = Field(default_factory=list)
    judged_current_grading: Optional[AggregatedGrading] = None
    # Final grade results
    past_grade: Optional[int] = None
    current_grade: Optional[int] = None
    grade: Optional[int] = None  # Max of past and current
    # Attribution results - overall and temporal
    attribution: Optional[int] = (
        None  # Overall attribution (for backward compatibility)
    )
    # Combined reasoning from all stages
    reasoning: Optional[str] = None
    past_attribution: Optional[int] = None  # Attribution for past events
    current_attribution: Optional[int] = None  # Attribution for current events
    # Certainty results - overall and temporal
    certainty: Optional[int] = None  # Overall certainty (for backward compatibility)
    past_certainty: Optional[int] = None  # Certainty for past events
    current_certainty: Optional[int] = None  # Certainty for current events
    # Attribution and certainty results (raw outputs)
    past_attribution_detection: Optional[AttributionDetection] = None
    current_attribution_detection: Optional[AttributionDetection] = None
    past_certainty_assessment: Optional[CertaintyAssessment] = None
    current_certainty_assessment: Optional[CertaintyAssessment] = None
    # Evidence collections from various stages - structured format
    evidence: Dict[str, Dict[str, List[str]]] = Field(
        default_factory=lambda: {
            "identification": {"evidence": []},
            "past_grading": {"evidence": []},
            "current_grading": {"evidence": []},
            "past_attribution": {"evidence": []},
            "current_attribution": {"evidence": []},
            "past_certainty": {"evidence": []},
            "current_certainty": {"evidence": []},
        },
        description="Structured evidence with categories and subcategories",
    )
    # Legacy evidence fields for backward compatibility
    identification_evidence: Optional[List[str]] = Field(
        default=None,
        description="Exact string match for any evidence snippets as a list of quotes",
    )
    past_grading_evidence: Optional[List[str]] = Field(
        default=None,
        description="Exact string match for any evidence snippets as a list of quotes",
    )
    current_grading_evidence: Optional[List[str]] = Field(
        default=None,
        description="Exact string match for any evidence snippets as a list of quotes",
    )
    past_attribution_evidence: Optional[List[str]] = Field(
        default=None,
        description="Exact string match for any evidence snippets as a list of quotes",
    )
    current_attribution_evidence: Optional[List[str]] = Field(
        default=None,
        description="Exact string match for any evidence snippets as a list of quotes",
    )
    past_certainty_evidence: Optional[List[str]] = Field(
        default=None,
        description="Exact string match for any evidence snippets as a list of quotes",
    )
    current_certainty_evidence: Optional[List[str]] = Field(
        default=None,
        description="Exact string match for any evidence snippets as a list of quotes",
    )
    # Meta-judge evaluation
    meta_judge_feedback: Optional[MetaJudgeFeedback] = None
    # User overview from meta-judge
    user_overview: Optional[str] = Field(
        default=None,
        description="A concise summary of the findings for clinical users, highlighting key positives and negatives.",
    )
    # Processing metadata
    iterations_completed: Optional[int] = None
    final_reasoning: Optional[str] = None


class EnhancedEventResult(BaseModel):
    """Final result of enhanced event processing with judge evaluation."""

    event_type: str
    # Grade fields
    grade: Optional[int] = None
    past_grade: Optional[int] = None
    current_grade: Optional[int] = None
    # Attribution fields - overall and temporal
    attribution: Optional[int] = None  # Overall attribution
    past_attribution: Optional[int] = None  # Attribution for past events
    current_attribution: Optional[int] = None  # Attribution for current events
    # Certainty fields - overall and temporal
    certainty: Optional[int] = None  # Overall certainty
    past_certainty: Optional[int] = None  # Certainty for past events
    current_certainty: Optional[int] = None  # Certainty for current events

    # Structured evidence dictionary
    evidence: Dict[str, Dict[str, List[str]]] = Field(
        default_factory=lambda: {
            "identification": {"evidence": []},
            "past_grading": {"evidence": []},
            "current_grading": {"evidence": []},
            "attribution": {"evidence": []},
            "certainty": {"evidence": []},
        },
        description="Structured evidence with categories and subcategories",
    )

    # Legacy evidence fields for backward compatibility
    identification_evidence: Optional[List[str]] = Field(
        default=None,
        description="Exact string match for any evidence snippets as a list of quotes",
    )
    past_grading_evidence: Optional[List[str]] = Field(
        default=None,
        description="Exact string match for any evidence snippets as a list of quotes",
    )
    current_grading_evidence: Optional[List[str]] = Field(
        default=None,
        description="Exact string match for any evidence snippets as a list of quotes",
    )
    attribution_evidence: Optional[List[str]] = Field(
        default=None,
        description="Exact string match for any evidence snippets as a list of quotes",
    )
    certainty_evidence: Optional[List[str]] = Field(
        default=None,
        description="Exact string match for any evidence snippets as a list of quotes",
    )

    # User overview
    user_overview: Optional[str] = Field(
        default=None,
        description="A concise summary of the findings for clinical users, highlighting key positives and negatives.",
    )

    # Additional metadata
    reasoning: Optional[str] = None
    iterations_completed: Optional[int] = None
