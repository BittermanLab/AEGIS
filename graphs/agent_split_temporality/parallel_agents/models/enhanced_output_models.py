"""
Enhanced output models for the parallel workflow implementation.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class EventIdentification(BaseModel):
    """Model for event identification (without temporal classification)"""

    event_present: bool = Field(description="Whether the event is present in the note")
    evidence_snippets: Optional[List[str]] = Field(
        default=None,
        description="List of exact string matched text from the notes that can be highlighted to physicians as evidence for the event presence",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Thought process and reasoning for the identification of the event or not, check of evidence string match",
    )


class TemporalityClassification(BaseModel):
    """Model for temporal classification of identified events"""

    past_events: List[str] = Field(
        default_factory=list,
        description="List of past adverse events or events that started prior to the time of the note e.g 'adverse event 1', 'adverse event 2'",
    )
    current_events: List[str] = Field(
        default_factory=list,
        description="List of current adverse events or ongoing events as of the time of the note e.g. e.g 'adverse event 1', 'adverse event 2'",
    )
    evidence_snippets: Optional[List[str]] = Field(
        default=None,
        description="List of exact string matched text from the notes that can be highlighted to physicians as evidence for the temporal classification",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Thought process and reasoning for the temporal classification, check of evidence string match",
    )


class TemporalIdentification(BaseModel):
    """Model for temporally-aware event identification (legacy compatibility)"""

    past_events: List[str] = Field(
        default_factory=list,
        description="List of past adverse events or events that started prior to the time of the note e.g 'adverse event 1', 'adverse event 2'",
    )
    current_events: List[str] = Field(
        default_factory=list,
        description="List of current adverse events or ongoing events as of the time of the note e.g. e.g 'adverse event 1', 'adverse event 2'",
    )
    evidence_snippets: Optional[List[str]] = Field(
        default=None,
        description="List of exact string matched text from the notes that can be highligted to physicians as evidence for the past and current events",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Thought process and reasoning for the identification of the events or not, check of evidence string match",
    )


class AttributionDetection(BaseModel):
    """Model for detection of attribution to immunotherapy"""

    attribution: int = Field(
        description="Attribution to immunotherapy (1 = yes, 0 = no)"
    )
    evidence: str = Field(
        description="Exact string match in the notes for any evidence snippets"
    )


class CertaintyAssessment(BaseModel):
    """Model for assessment of certainty level"""

    certainty: int = Field(description="Certainty level (0-4 scale)")
    evidence: str = Field(
        description="Exact string match in the notes for any evidence snippets"
    )


class EventGrading(BaseModel):
    """Model for grading a single event"""

    event_name: str = Field(description="Name of the event")
    grade: int = Field(description="Grade assigned (0-5)")
    rationale: str = Field(description="Rationale for the grade")
    temporal_context: str = Field(description="Temporal context (past or current)")
    evidence_snippets: Optional[List[str]] = Field(
        default=None,
        description="Exact string match in the notes for any evidence snippets as a list of quotes",
    )


class EventResult(BaseModel):
    """Comprehensive result for a single event type"""

    event_type: str = Field(description="Type of event (e.g., pneumonitis)")
    grade: int = Field(description="Maximum grade (0-5)")
    past_grade: Optional[int] = Field(
        default=None, description="Grade of past event if present"
    )
    current_grade: Optional[int] = Field(
        default=None, description="Grade of current event if present"
    )
    # Overall attribution and certainty (for backward compatibility)
    attribution: int = Field(
        description="Overall attribution to immunotherapy (1 = yes, 0 = no)"
    )
    certainty: int = Field(description="Overall certainty level (0-4 scale)")
    # Separate attribution and certainty for past and current events
    past_attribution: Optional[int] = Field(
        default=None,
        description="Past event attribution to immunotherapy (1 = yes, 0 = no)",
    )
    current_attribution: Optional[int] = Field(
        default=None,
        description="Current event attribution to immunotherapy (1 = yes, 0 = no)",
    )
    past_certainty: Optional[int] = Field(
        default=None, description="Past event certainty level (0-4 scale)"
    )
    current_certainty: Optional[int] = Field(
        default=None, description="Current event certainty level (0-4 scale)"
    )
    reasoning: str = Field(description="Comprehensive reasoning")
    # Structured evidence field
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
        description="Exact string match in the notes for any evidence snippets as a list of quotes",
    )
    past_grading_evidence: Optional[List[str]] = Field(
        default=None,
        description="Exact string match in the notes for any evidence snippets as a list of quotes",
    )
    current_grading_evidence: Optional[List[str]] = Field(
        default=None,
        description="Exact string match in the notes for any evidence snippets as a list of quotes",
    )
    attribution_evidence: Optional[List[str]] = Field(
        default=None,
        description="Exact string match in the notes for any evidence snippets as a list of quotes",
    )
    certainty_evidence: Optional[List[str]] = Field(
        default=None,
        description="Exact string match in the notes for any evidence snippets as a list of quotes",
    )
    # User-friendly overview
    user_overview: Optional[str] = Field(
        default=None,
        description="A concise summary of the findings for clinical users, highlighting key positives and negatives.",
    )


class EnhancedPrediction(BaseModel):
    """Enhanced prediction model with attribution and certainty"""

    past_grades: Dict[str, int] = Field(
        default_factory=dict, description="Past grade values for each event type"
    )
    current_grades: Dict[str, int] = Field(
        default_factory=dict, description="Current grade values for each event type"
    )
    past_attributions: Dict[str, int] = Field(
        default_factory=dict, description="Past attribution values for each event type"
    )
    current_attributions: Dict[str, int] = Field(
        default_factory=dict,
        description="Current attribution values for each event type",
    )
    past_certainties: Dict[str, int] = Field(
        default_factory=dict, description="Past certainty values for each event type"
    )
    current_certainties: Dict[str, int] = Field(
        default_factory=dict, description="Current certainty values for each event type"
    )
    evidences: Dict[str, Dict[str, Dict[str, List[str]]]] = Field(
        default_factory=dict,
        description="Structured evidence with categories and subcategories for each event type",
    )
    user_overviews: Dict[str, str] = Field(
        default_factory=dict, description="User overviews for each event type"
    )

    # Legacy fields for backward compatibility
    # Unified structured evidence dictionary
    evidence: Dict[str, Dict[str, Dict[str, List[str]]]] = Field(
        default_factory=dict,
        description="Evidence organized by condition, stage, and temporal context with structure: {condition: {stage: {temporal_context: [evidence_snippets]}}}",
    )

    # Combined user summary with overviews of all events
    user_summary: str = Field(
        default="",
        description="Combined user summary of all event overviews for clinical users",
    )

    # Reasoning
    reasoning: str = ""

    # Add storage for past and current events
    past_events: Dict[str, List[str]] = Field(
        default_factory=dict, description="Past events for each event type"
    )
    current_events: Dict[str, List[str]] = Field(
        default_factory=dict, description="Current events for each event type"
    )

    def __getattr__(self, name):
        """
        Dynamically handle attribute access for event-specific fields.
        This allows access to fields like pneumonitis_past_grade through the dynamic dictionaries.
        """
        parts = name.split("_")
        if len(parts) >= 2:
            event_type = parts[0]

            if name.endswith("_past_grade"):
                return self.past_grades.get(event_type, 0)
            elif name.endswith("_current_grade"):
                return self.current_grades.get(event_type, 0)
            elif name.endswith("_grade"):  # Handle plain grade field
                # Return the maximum of past and current grades for backward compatibility
                past_grade = self.past_grades.get(event_type, 0)
                current_grade = self.current_grades.get(event_type, 0)
                return max(past_grade, current_grade)
            elif name.endswith("_past_attribution"):
                return self.past_attributions.get(event_type, 0)
            elif name.endswith("_current_attribution"):
                return self.current_attributions.get(event_type, 0)
            elif name.endswith("_attribution"):  # Handle plain attribution field
                # Return 1 if either past or current has attribution
                past_attr = self.past_attributions.get(event_type, 0)
                current_attr = self.current_attributions.get(event_type, 0)
                return max(past_attr, current_attr)
            elif name.endswith("_past_certainty"):
                return self.past_certainties.get(event_type, 0)
            elif name.endswith("_current_certainty"):
                return self.current_certainties.get(event_type, 0)
            elif name.endswith("_certainty"):  # Handle plain certainty field
                # Return the maximum of past and current certainties
                past_cert = self.past_certainties.get(event_type, 0)
                current_cert = self.current_certainties.get(event_type, 0)
                return max(past_cert, current_cert)
            elif name.endswith("_evidence"):
                return self.evidences.get(event_type, {})
            elif name.endswith("_evidence_snippets"):  # Handle evidence_snippets field
                # Return evidence snippets from the evidences structure
                evidence = self.evidences.get(event_type, {})
                if (
                    "identification" in evidence
                    and "evidence" in evidence["identification"]
                ):
                    return evidence["identification"]["evidence"]
                return []
            elif name.endswith("_user_overview"):
                return self.user_overviews.get(event_type, None)
            elif name.endswith("_past_events"):
                return self.past_events.get(event_type, [])
            elif name.endswith("_current_events"):
                return self.current_events.get(event_type, [])

        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")

    def __setattr__(self, name, value):
        """
        Dynamically handle setting attributes for event-specific fields.
        """
        if name.startswith("__") or name in self.__dict__ or name in self.__fields__:
            super().__setattr__(name, value)
            return

        parts = name.split("_")
        if len(parts) >= 2:
            event_type = parts[0]

            if name.endswith("_past_grade"):
                self.past_grades[event_type] = value
                return
            elif name.endswith("_current_grade"):
                self.current_grades[event_type] = value
                return
            elif name.endswith("_grade"):  # Handle plain grade field
                # Set both past and current grades to max for backward compatibility
                max_grade = max(
                    self.past_grades.get(event_type, 0),
                    self.current_grades.get(event_type, 0),
                    value,
                )
                # Don't store the plain grade, but ensure past/current are set
                return
            elif name.endswith("_past_attribution"):
                self.past_attributions[event_type] = value
                return
            elif name.endswith("_current_attribution"):
                self.current_attributions[event_type] = value
                return
            elif name.endswith("_attribution"):  # Handle plain attribution field
                # For backward compatibility, set both past and current
                if value > 0:
                    # Only set if there are actual grades
                    if self.past_grades.get(event_type, 0) > 0:
                        self.past_attributions[event_type] = value
                    if self.current_grades.get(event_type, 0) > 0:
                        self.current_attributions[event_type] = value
                return
            elif name.endswith("_past_certainty"):
                self.past_certainties[event_type] = value
                return
            elif name.endswith("_current_certainty"):
                self.current_certainties[event_type] = value
                return
            elif name.endswith("_certainty"):  # Handle plain certainty field
                # For backward compatibility, set both past and current
                if value > 0:
                    # Only set if there are actual grades
                    if self.past_grades.get(event_type, 0) > 0:
                        self.past_certainties[event_type] = value
                    if self.current_grades.get(event_type, 0) > 0:
                        self.current_certainties[event_type] = value
                return
            elif name.endswith("_evidence"):
                self.evidences[event_type] = value
                return
            elif name.endswith("_evidence_snippets"):  # Handle evidence_snippets field
                # Set evidence snippets in the evidences structure
                evidence = self.evidences.get(event_type, {})
                evidence["identification"]["evidence"] = value
                return
            elif name.endswith("_user_overview"):
                self.user_overviews[event_type] = value
                return
            elif name.endswith("_past_events"):
                self.past_events[event_type] = value
                return
            elif name.endswith("_current_events"):
                self.current_events[event_type] = value
                return

        # If we get here, it's not a recognized dynamic field, so silently ignore
        # instead of calling super().__setattr__ which would raise a Pydantic error
        # This prevents errors when trying to set unrecognized fields
        return

    def dict(self, *args, **kwargs):
        """
        Override the dict method to output a flat representation with event_type_field keys
        instead of nested dictionaries, to maintain backwards compatibility.
        """
        # First get the standard dictionary representation
        base_dict = super().dict(*args, **kwargs)

        # Create the flat representation
        flat_dict = {}

        # Add existing base fields (evidence, user_summary, reasoning)
        if "evidence" in base_dict:
            flat_dict["evidence"] = base_dict["evidence"]
        if "user_summary" in base_dict:
            flat_dict["user_summary"] = base_dict["user_summary"]
        if "reasoning" in base_dict:
            flat_dict["reasoning"] = base_dict["reasoning"]

        # Flatten the nested dictionaries to individual fields
        if "past_grades" in base_dict and isinstance(base_dict["past_grades"], dict):
            for event_type, grade in base_dict["past_grades"].items():
                flat_dict[f"{event_type}_past_grade"] = grade

        if "current_grades" in base_dict and isinstance(
            base_dict["current_grades"], dict
        ):
            for event_type, grade in base_dict["current_grades"].items():
                flat_dict[f"{event_type}_current_grade"] = grade

        if "past_attributions" in base_dict and isinstance(
            base_dict["past_attributions"], dict
        ):
            for event_type, attribution in base_dict["past_attributions"].items():
                flat_dict[f"{event_type}_past_attribution"] = attribution

        if "current_attributions" in base_dict and isinstance(
            base_dict["current_attributions"], dict
        ):
            for event_type, attribution in base_dict["current_attributions"].items():
                flat_dict[f"{event_type}_current_attribution"] = attribution

        if "past_certainties" in base_dict and isinstance(
            base_dict["past_certainties"], dict
        ):
            for event_type, certainty in base_dict["past_certainties"].items():
                flat_dict[f"{event_type}_past_certainty"] = certainty

        if "current_certainties" in base_dict and isinstance(
            base_dict["current_certainties"], dict
        ):
            for event_type, certainty in base_dict["current_certainties"].items():
                flat_dict[f"{event_type}_current_certainty"] = certainty

        if "evidences" in base_dict and isinstance(base_dict["evidences"], dict):
            for event_type, evidence in base_dict["evidences"].items():
                flat_dict[f"{event_type}_evidence"] = evidence

        if "user_overviews" in base_dict and isinstance(
            base_dict["user_overviews"], dict
        ):
            for event_type, overview in base_dict["user_overviews"].items():
                flat_dict[f"{event_type}_user_overview"] = overview

        if "past_events" in base_dict and isinstance(base_dict["past_events"], dict):
            for event_type, events in base_dict["past_events"].items():
                flat_dict[f"{event_type}_past_events"] = events

        if "current_events" in base_dict and isinstance(
            base_dict["current_events"], dict
        ):
            for event_type, events in base_dict["current_events"].items():
                flat_dict[f"{event_type}_current_events"] = events

        return flat_dict
