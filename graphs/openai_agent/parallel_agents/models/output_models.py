from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


# Define NotePrediction directly here
class NotePrediction(BaseModel):
    """Prediction output for a clinical note."""

    PMRN: str
    NOTE_ID: str
    NOTE_TEXT: str
    SHORTENED_NOTE_TEXT: str
    MESSAGES: List[Any]
    PREDICTION: Dict[str, Any]
    PROCESSED_AT: datetime
    PROCESSING_TIME: float
    MODEL_NAME: str
    TOKEN_USAGE: Optional[Any] = None
    ERROR: Optional[str] = None
    COST_REPORT: Optional[Dict[str, Any]] = None
    TOKEN_TRACKER: Optional[Any] = None


class ExtractedNote(BaseModel):
    """Output of the extraction step"""

    reasoning: str = Field(description="Explanation of extraction decisions")
    extracted_note: str = Field(
        description="The exact chunks of text extracted from the note"
    )


class CTCAEHeader(BaseModel):
    """Structure for CTCAE header information"""

    header: str = Field(description="The CTCAE header name")
    definition: str = Field(description="CTCAE header definition")


class RetrievedHeaders(BaseModel):
    """Output of the retrieval step"""

    reasoning: str = Field(description="Explanation of retrieval decisions")
    retrieved_headers: List[CTCAEHeader] = Field(
        description="List of relevant CTCAE headers"
    )


class IdentifiedEvents(BaseModel):
    """Output of the identification step"""

    reasoning: str = Field(description="Explanation of identification process")
    events: List[str] = Field(description="List of identified priority events")


class GradedEvent(BaseModel):
    """Structure for each graded event"""

    event: str = Field(description="The name of the adverse event")
    grade: str = Field(description="CTCAE grade assigned (1-5)")
    rationale: str = Field(description="Explanation for the assigned grade")


class GradedEvents(BaseModel):
    """Output of the grading step"""

    reasoning: str = Field(description="Overall reasoning for grading decisions")
    events: List[GradedEvent] = Field(description="List of graded events")


class FinalOutput(BaseModel):
    """Final standardized output"""

    reasoning: str = Field(description="Overall analysis reasoning")
    pneumonitis_grade: int = Field(description="CTCAE grade for pneumonitis (0-5)")
    myocarditis_grade: int = Field(description="CTCAE grade for myocarditis (0-5)")
    colitis_grade: int = Field(description="CTCAE grade for colitis (0-5)")
    thyroiditis_grade: int = Field(description="CTCAE grade for thyroiditis (0-5)")
    hepatitis_grade: int = Field(description="CTCAE grade for hepatitis (0-5)")
    dermatitis_grade: int = Field(description="CTCAE grade for dermatitis (0-5)")


# Conversion helper function to create a standardized NotePrediction from a dictionary with lowercase keys
def create_note_prediction(data: Dict) -> NotePrediction:
    """Create a standardized NotePrediction instance from a dictionary with lowercase keys"""
    # Map lowercase field names to uppercase
    uppercase_data = {
        "PMRN": data.get("pmrn"),
        "NOTE_ID": data.get("note_id"),
        "NOTE_TEXT": data.get("note_text"),
        "SHORTENED_NOTE_TEXT": data.get("shortened_note_text"),
        "MESSAGES": data.get("messages", []),
        "PREDICTION": data.get("prediction", {}),
        "PROCESSED_AT": data.get("processed_at"),
        "PROCESSING_TIME": data.get("processing_time", 0.0),
        "MODEL_NAME": data.get("model_name", ""),
        "TOKEN_USAGE": data.get("token_usage"),
        "ERROR": data.get("error"),
        "COST_REPORT": data.get("cost_report"),
        "TOKEN_TRACKER": data.get("token_tracker"),
    }
    return NotePrediction(**uppercase_data)
