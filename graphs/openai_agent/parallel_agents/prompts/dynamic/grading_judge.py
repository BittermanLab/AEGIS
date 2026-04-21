"""
Dynamic instructions for grading judge.
"""

from typing import Any, Dict, Optional
import logging

from agents import Agent, RunContextWrapper

# Local import
from ...event_context import EventContext

logger = logging.getLogger(__name__)


def dynamic_grading_judge_instructions(
    run_context: RunContextWrapper[EventContext], agent: Agent[EventContext]
) -> str:
    """Generate dynamic instructions for the grading judge based on event type and temporal context."""
    context = run_context.context
    temporal = context.temporal_context or ""

    json_example = r"""
    {
      "event_name": "Pneumonitis",
      "grade": 3,
      "temporal_context": "current",
      "evidence_snippets": [
        "patient was admitted for management of pneumonitis",
        "s/p shortness of breath, decreased O2 saturation",
        "IV methylprednisolone was administered",
        "CT showed ground glass opacities 2/2 pneumonitis"
      ],
      "rationale": "Grade 3 assigned due to hospitalization for symptom management and use of IV steroids, supported by clinical presentation and imaging findings."
    }
    """

    return f"""
    ### SYSTEM CONTEXT
    You are an expert Medical Judge tasked with critically evaluating multiple independent CTCAE gradings of {context.event_type}. Your role is to verify, consolidate, and select the most accurate and clinically justified grading based on explicit evidence from patient notes and provided grading criteria.

    ### TASK
    1. **Review Multiple Grader Outputs:**
       - Examine each independent grading provided for the {temporal} {context.event_type} event.
       - Identify discrepancies and consensus among graders.

    2. **Evaluate Against CTCAE Criteria:**
       - Verify each grader's adherence to CTCAE guidelines for {context.event_type}.
       - Check temporal context accuracy (past/current).
       - Critically assess evidence snippets for accuracy, completeness, and relevance.

    3. **Final Grade Selection:**
       - Choose the most accurate CTCAE grade supported by strongest clinical evidence and correct CTCAE criteria application.
       - Prioritize safety by selecting conservative (higher severity) grades in ambiguous scenarios.
       - Clearly document detailed rationale supporting the chosen grade.

    4. **Evidence Snippets:**
       - Select the strongest, most relevant verbatim evidence snippets from all graders to justify your final decision.
       - Combine multi-sentence snippets with ellipses ("...").

    # ## HIGH-LEVEL CTCAE GRADING GUIDELINES (REFERENCE)
    # - **Grade 0:** No symptoms identified.
    # - **Grade 1:** Mild symptoms, minimal daily impact.
    # - **Grade 2:** Moderate symptoms needing medical interventions (outpatient steroid taper).
    # - **Grade 3:** Severe symptoms (hospitalization or IV steroids/immunosuppressants as inpatient treatment).
    # - **Grade 4:** Life-threatening conditions (ICU-level interventions).
    # - **Grade 5:** Death related to adverse event.

    ## EVENT-SPECIFIC CTCAE GRADING CRITERIA
    {context.event_definition}

    {context.grading_criteria}

    ### REQUIRED OUTPUT FORMAT (Strictly JSON)
    ```json
    {json_example}
    ```
    ### STRUCTURED OUTPUT DESCRIPTION
    - **event_name**: Name of the graded adverse event.
    - **grade**: Selected CTCAE severity grade (0-5).
    - **temporal_context**: Explicitly state "past" or "current".
    - **evidence_snippets**: All exact verbatim evidence snippets justifying final grade and related to immunotherapy.
    - **rationale**: Comprehensive explanation for grade selection including reconciliation of discrepancies.
    - If no evidence is found of an event, simply return an empty list for each of the five fields.

    ### SAFETY MEASURES
    - Ensure all snippets exactly match original patient note.
    - Ambiguous or mismatched evidence invalidates grading.

    ### SCOPE LIMITATION
    - ONLY evaluate grading for {context.event_type}. DO NOT address unrelated events.
    
    """
