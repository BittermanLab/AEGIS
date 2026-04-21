from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class NoteInput(BaseModel):
    """Input model for a clinical note"""

    pmrn: str = Field(description="Patient Medical Record Number")
    note_id: str = Field(description="Unique identifier for the note")
    note_type: str = Field(description="Type of clinical note")
    type_name: str = Field(description="Descriptive name of the note type")
    loc_name: str = Field(description="Location name where note was taken")
    date: str = Field(description="Date of the note in ISO format")
    prov_name: str = Field(description="Provider name")
    prov_type: str = Field(description="Provider type")
    line: int = Field(description="Line number in source data")
    note_text: str = Field(description="Full text content of the clinical note")
    labels: Optional[Dict[str, Any]] = Field(
        None, description="Optional gold labels for evaluation"
    )
    timepoint: Optional[str] = Field(None, description="Optional timepoint information")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NoteInput":
        """Convert a dictionary to a NoteInput object with validation"""
        # Validate PMRN
        if not data.get("PMRN"):
            raise ValueError("PMRN is required but missing or empty")

        # Normalize PMRN (in case it's not a string)
        pmrn = str(data["PMRN"]).strip()
        if not pmrn:
            raise ValueError("PMRN is empty after normalization")

        # Extract optional fields
        labels = data.get("labels")
        timepoint = data.get("timepoint")

        # Create the model instance with required fields
        note = cls(
            pmrn=pmrn,
            note_id=str(data["NOTE_ID"]),
            note_type=str(data["NOTE_TYPE"]),
            type_name=str(data["TYPE_NAME"]),
            loc_name=str(data["LOC_NAME"]),
            date=str(data["DATE"]),
            prov_name=str(data["PROV_NAME"]),
            prov_type=str(data["PROV_TYPE"]),
            line=int(data["LINE"]),
            note_text=str(data["NOTE_TEXT"]),
        )

        # Add optional fields if present
        if labels:
            note.labels = labels
        if timepoint:
            note.timepoint = timepoint

        return note


class TokenUsageMetadata(BaseModel):
    """Metadata about token usage for cost tracking"""

    prompt_tokens: int = Field(0, description="Number of tokens in the prompt")
    completion_tokens: int = Field(0, description="Number of tokens in the completion")
    total_tokens: int = Field(0, description="Total number of tokens used")
    prompt_cost: float = Field(0.0, description="Cost for prompt tokens")
    completion_cost: float = Field(0.0, description="Cost for completion tokens")
    total_cost: float = Field(0.0, description="Total cost")
    model_costs: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="Per-model breakdown of costs"
    )
