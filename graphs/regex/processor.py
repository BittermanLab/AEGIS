# file: processor.py
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from dotenv import load_dotenv
import os
import logging
import re

# Get logger instance
logger = logging.getLogger(__name__)


# Define our own model classes to avoid import errors
class NoteInput:
    """Input model for clinical notes."""

    def __init__(
        self,
        pmrn: str,
        note_id: str,
        note_type: Optional[str] = None,
        type_name: Optional[str] = None,
        loc_name: Optional[str] = None,
        date: Optional[str] = None,
        prov_name: Optional[str] = None,
        prov_type: Optional[str] = None,
        line: Optional[int] = None,
        note_text: str = "",
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
    """Prediction model for clinical notes."""

    def __init__(
        self,
        pmrn: str,
        note_id: str,
        note_text: str,
        shortened_note_text: str,
        messages: List[Any],
        prediction: Dict[str, Any],
        processed_at: datetime,
        processing_time: float,
        model_name: str,
        token_usage: Optional[Any] = None,
        error: Optional[str] = None,
    ):
        self.PMRN = pmrn
        self.NOTE_ID = note_id
        self.NOTE_TEXT = note_text
        self.SHORTENED_NOTE_TEXT = shortened_note_text
        self.MESSAGES = messages
        self.PREDICTION = prediction
        self.PROCESSED_AT = processed_at
        self.PROCESSING_TIME = processing_time
        self.MODEL_NAME = model_name
        self.TOKEN_USAGE = token_usage
        self.ERROR = error


# ----------------------------------
# 1) Define the keyword sets for each category
#    Anything not in these sets is considered "pneumonitis."
# ----------------------------------
MYOCARDITIS_TERMS = {"myocarditis"}
COLITIS_TERMS = {"colitis"}
THYROIDITIS_TERMS = {"thyroiditis"}
HEPATITIS_TERMS = {"hepatitis"}
DERMATITIS_TERMS = {"dermatitis"}
PNEUMONITIS_TERMS = {"pneumonitis"}


class RegexProcessor:
    def __init__(self):
        load_dotenv()

        # Read regex terms from file
        self.regex_terms = []
        try:
            # Get the directory of the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Build the path to regex_terms.txt
            terms_file = os.path.join(current_dir, "regex_terms.txt")

            with open(terms_file, "r") as f:
                terms = [line.strip() for line in f if line.strip()]

            # Check for duplicates
            terms_set = set(terms)

            # Store deduplicated terms
            self.regex_terms = sorted(list(terms_set))

            # Pre-process terms for regex
            processed_terms = []
            for term in self.regex_terms:
                # Make hyphen optional
                term = term.replace("-", "[-]?")
                # Convert spaces to "\s+"
                term = term.replace(" ", r"\s+")
                # Escape parentheses
                term = term.replace("(", r"\(").replace(")", r"\)")
                processed_terms.append(term)

            # Create a single regex pattern with word boundaries + case-insensitive
            # (?: ... ) non-capturing group keeps re.findall from returning tuples
            self.pattern = r"(?i)\b(?:" + "|".join(processed_terms) + r")\b"

        except Exception as e:
            logger.error(f"Error loading regex terms: {e}")
            raise

    def detect_keywords(self, text: str) -> dict:
        """
        Detects keywords from regex_terms.txt and categorizes them under:
          myocarditis, colitis, thyroiditis, hepatitis, dermatitis, or pneumonitis.
        """
        try:
            # Find all matches using the pattern
            matches = re.findall(self.pattern, text)

            # Deduplicate and sort matches
            unique_matches = sorted(set(m.strip() for m in matches))

            # Initialize all categories to 0
            myocarditis_grade = 0
            colitis_grade = 0
            thyroiditis_grade = 0
            hepatitis_grade = 0
            dermatitis_grade = 0
            pneumonitis_grade = 0

            # For reasoning, store the matched words in each category
            myocarditis_hits = []
            colitis_hits = []
            thyroiditis_hits = []
            hepatitis_hits = []
            dermatitis_hits = []
            pneumonitis_hits = []

            # Categorize each unique match
            for match in unique_matches:
                normalized = match.lower()

                if any(term in normalized for term in MYOCARDITIS_TERMS):
                    myocarditis_grade = 1
                    myocarditis_hits.append(match)
                elif any(term in normalized for term in COLITIS_TERMS):
                    colitis_grade = 1
                    colitis_hits.append(match)
                elif any(term in normalized for term in THYROIDITIS_TERMS):
                    thyroiditis_grade = 1
                    thyroiditis_hits.append(match)
                elif any(term in normalized for term in HEPATITIS_TERMS):
                    hepatitis_grade = 1
                    hepatitis_hits.append(match)
                elif any(term in normalized for term in DERMATITIS_TERMS):
                    dermatitis_grade = 1
                    dermatitis_hits.append(match)
                elif any(term in normalized for term in PNEUMONITIS_TERMS):
                    # Explicitly check for pneumonitis terms
                    pneumonitis_grade = 1
                    pneumonitis_hits.append(match)
                else:
                    # Default = pneumonitis for any other immune-related terms
                    pneumonitis_grade = 1
                    pneumonitis_hits.append(match)

            # Build a reasoning string that shows what we matched
            # and how we categorized it
            reasoning_parts = []
            if myocarditis_hits:
                reasoning_parts.append(f"Myocarditis: {', '.join(myocarditis_hits)}")
            if colitis_hits:
                reasoning_parts.append(f"Colitis: {', '.join(colitis_hits)}")
            if thyroiditis_hits:
                reasoning_parts.append(f"Thyroiditis: {', '.join(thyroiditis_hits)}")
            if hepatitis_hits:
                reasoning_parts.append(f"Hepatitis: {', '.join(hepatitis_hits)}")
            if dermatitis_hits:
                reasoning_parts.append(f"Dermatitis: {', '.join(dermatitis_hits)}")
            if pneumonitis_hits:
                reasoning_parts.append(f"Pneumonitis: {', '.join(pneumonitis_hits)}")

            if not unique_matches:
                # No matches at all
                reasoning = "No significant immune-related terms found."
            else:
                reasoning = " | ".join(reasoning_parts)

            # Build the final output structure
            result = {
                "reasoning": reasoning,
                "pneumonitis_current_grade": pneumonitis_grade,
                "myocarditis_current_grade": myocarditis_grade,
                "colitis_current_grade": colitis_grade,
                "thyroiditis_current_grade": thyroiditis_grade,
                "hepatitis_current_grade": hepatitis_grade,
                "dermatitis_current_grade": dermatitis_grade,
                # For AE attribution
                "pneumonitis_current_attribution": pneumonitis_grade,
                "myocarditis_current_attribution": myocarditis_grade,
                "colitis_current_attribution": colitis_grade,
                "thyroiditis_current_attribution": thyroiditis_grade,
                "hepatitis_current_attribution": hepatitis_grade,
                "dermatitis_current_attribution": dermatitis_grade,
                # For certainty
                "pneumonitis_current_certainty": pneumonitis_grade,
                "myocarditis_current_certainty": myocarditis_grade,
                "colitis_current_certainty": colitis_grade,
                "thyroiditis_current_certainty": thyroiditis_grade,
                "hepatitis_current_certainty": hepatitis_grade,
                "dermatitis_current_certainty": dermatitis_grade,
            }
            return result

        except Exception as e:
            logger.error(f"Error in detect_keywords: {str(e)}")
            # Return safe default on error
            return {
                "reasoning": f"Error processing text: {str(e)}",
                "pneumonitis_current_grade": 0,
                "myocarditis_current_grade": 0,
                "colitis_current_grade": 0,
                "thyroiditis_current_grade": 0,
                "hepatitis_current_grade": 0,
                "dermatitis_current_grade": 0,
                "pneumonitis_current_attribution": 0,
                "myocarditis_current_attribution": 0,
                "colitis_current_attribution": 0,
                "thyroiditis_current_attribution": 0,
                "hepatitis_current_attribution": 0,
                "dermatitis_current_attribution": 0,
                "pneumonitis_current_certainty": 0,
                "myocarditis_current_certainty": 0,
                "colitis_current_certainty": 0,
                "thyroiditis_current_certainty": 0,
                "hepatitis_current_certainty": 0,
                "dermatitis_current_certainty": 0,
            }

    async def _process_single_note_async(self, note: NoteInput) -> NotePrediction:
        """
        Process a single note using regex pattern matching (async implementation).
        """
        start_time = datetime.now()

        try:
            # Detect keywords and categorize findings
            prediction = self.detect_keywords(note.note_text)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            return NotePrediction(
                pmrn=note.pmrn,
                note_id=note.note_id,
                note_text=note.note_text,
                shortened_note_text=(
                    note.note_text[:500] + "..."
                    if len(note.note_text) > 500
                    else note.note_text
                ),
                messages=[],  # No stepwise LLM messages in regex processing
                prediction=prediction,
                processed_at=datetime.now(),
                processing_time=processing_time,
                model_name="regex_pattern_matcher",
                token_usage=None,  # No token usage in regex processing
                error=None,
            )

        except Exception as e:
            logger.error(f"Error processing note: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()

            return NotePrediction(
                pmrn=note.pmrn,
                note_id=note.note_id,
                note_text=note.note_text,
                shortened_note_text=(
                    note.note_text[:100] + "..."
                    if len(note.note_text) > 100
                    else note.note_text
                ),
                messages=[],
                prediction={},
                processed_at=datetime.now(),
                processing_time=processing_time,
                model_name="regex_pattern_matcher",
                token_usage=None,
                error=str(e),
            )

    def process_single_note(self, note: NoteInput) -> NotePrediction:
        """
        Process a single note using regex pattern matching.
        Non-async version for simpler integration.
        """
        try:
            # Call the async version and run it with a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._process_single_note_async(note))
            loop.close()
            return result
        except Exception as e:
            logger.error(f"Error in non-async process_single_note: {e}")
            # Return minimal error result
            return NotePrediction(
                pmrn=note.pmrn,
                note_id=note.note_id,
                note_text=note.note_text,
                shortened_note_text=(
                    note.note_text[:100] + "..."
                    if len(note.note_text) > 100
                    else note.note_text
                ),
                messages=[],
                prediction={},
                processed_at=datetime.now(),
                processing_time=0.0,
                model_name="regex_pattern_matcher",
                token_usage=None,
                error=str(e),
            )

    async def process_all_notes(self, notes: List[NoteInput]) -> List[NotePrediction]:
        """
        Process multiple notes in parallel using asyncio.gather.
        """
        logger.info(f"Processing {len(notes)} notes")
        predictions = await asyncio.gather(
            *[self._process_single_note_async(note) for note in notes]
        )
        return predictions
