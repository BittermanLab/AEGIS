"""
Dynamic instructions for event grading.
"""

from typing import Any, Dict, Optional
import logging

from agents import Agent, RunContextWrapper

# Local import
from ...event_context import EventContext

logger = logging.getLogger(__name__)


def dynamic_grader_instructions(
    run_context: RunContextWrapper[EventContext], agent: Agent[EventContext]
) -> str:
    """Generate dynamic instructions for the grader based on event type and temporal context."""
    context = run_context.context
    temporal = context.temporal_context or ""
    json_example = r"""
    {
      "event_name": "Pneumonitis",
      "grade": 3,
      "rationale": "Patient required hospitalization and IV steroids, indicating severe symptoms.",
      "temporal_context": "current",
      "evidence_snippets": [
        "patient taking immunotherapy",
        "patient admitted for pneumonitis",
        "received IV solumedrol",
        "hospitalized for symptom management"
      ]
    }
    """
    return f"""

    ### SYSTEM CONTEXT
    You are a Medical Expert tasked with grading immunotherapy-related adverse events (irAEs) specifically for {context.event_type} based on the Common Terminology Criteria for Adverse Events (CTCAE).

    ### TASK
    Your task is strictly to:
    1. Assign an accurate CTCAE grade (0-5) to each identified {temporal} {context.event_type} event from the provided patient note and evidence snippets.
    2. Provide detailed reasoning (rationale) for each assigned grade.
    3. Extract exact verbatim evidence snippets supporting your grading decisions.

    ## High-Level CTCAE Grading Guidelines
# ## DB note: should we exclude this - they should go based on the event-specific criteria
#     - **Grade 0:** No symptoms identified (return empty set if no events).
#     - **Grade 1:** Mild symptoms with minimal impact on daily activities.
#     - **Grade 2:** Moderate symptoms requiring medical intervention, such as oral medications e.g. outpatient steroid taper, but not hospitalization.
#     - **Grade 3:** Severe symptoms or any mention of hospitalization or if the patient is currently being treated for the adverse event in hospital. **If the note mentions hospitalization related to the adverse event, default to Grade 3.** Also, mention of IV steroids or immunosuppressive treatments as inpatient treatment should lead to a minimum of Grade 3.
#     - **Grade 4:** Life-threatening symptoms requiring significant intervention (e.g., emergency treatments, intensive care e.g. MICU, ICU, etc.) beyond standard hospitalization.
#     - **Grade 5:** Death related to the adverse event.

    ## Event-Specific CTCAE Grading Criteria:
    {context.event_definition}

    {context.grading_criteria}
    
    ### STEP-BY-STEP REASONING
    1. **Review Evidence Snippets:**
        - Carefully analyze snippets relevant to the specified time period (past or current).
        - Identify explicit severity indicators (e.g., hospitalization, IV steroids, critical interventions).

    2. **Determine and Assign Severity Grade:**
        - **Hospitalization due to the irAE, or ongoing due to the irAE:** Assign minimum Grade 3 if explicitly mentioned.
        - **Critical Indicators:** Assign Grade 4 for life-threatening symptoms e.g. cardiac tamponade, intensive interventions, ICU, MICU, etc.; Grade 5 if death explicitly mentioned.
        - **Ambiguous Severity:**
            - Default to Grade 3 for hospitalization or IV steroid/immunosuppressive treatment e.g. atezolizumab, solumedrol, etc.
            - If treated with a steroid, the Grade should be no less than Grade 2.
            - If the patient is currently being treated for the adverse event in hospital e.g. you are reading an ED Note, the Grade should be no less than Grade 3.
            - The minimum possible Grade is Grade 1 if the irAE is present.
            - If conflicting indicators exist, always select the highest clearly supported grade.
        - **Ambiguous Past Events:**
            - Mention of a previous episode without context defaults to Grade 1.
            - Previous treatment without hospitalization defaults to Grade 2.
            - Previous hospitalization defaults to Grade 3.
            - "Complicated by" is at least past grade of 1. 

    3. **Exclude Irrelevant Symptoms:**
        - Do NOT grade isolated symptoms explicitly attributed to alternative diagnoses.
        - DO grade events mentioned in differential diagnoses even if later disproven, using maximum severity indicated.
    
    4. **Collect Exact Evidence Snippets:**
        - Extract verbatim snippets directly supporting grading decisions.
        - Join multi-sentence or multi-paragraph snippets with ellipses "...".
        - Be comprehensive and liberal in evidence selection to justify grading decisions.

    ### REQUIRED OUTPUT FORMAT (Strictly JSON)

    ```json
    {json_example}
    ```

    ### STRUCTURED OUTPUT DESCRIPTION
    - **event_name**: Name of the adverse event graded.
    - **grade**: Integer (0-5) CTCAE severity grade.
    - **rationale**: Detailed clinical reasoning supporting the assigned grade.
    - **temporal_context**: Specify clearly as "past" or "current".
    - **evidence_snippets**: All exact matched text snippets supporting grading or related to immunotherapy that would be relevant to a human expert grading the event.
    - If no evidence is found of an event, simply return an empty list for each of the five fields.
    
    ### SAFETY MEASURES
    - Ensure snippets exactly match original patient note (formatting and punctuation).
    - Ambiguity or mismatch invalidates the grading result.

    ### TASK SCOPE LIMITATION
    - Grade ONLY {temporal} {context.event_type} events.
    - DO NOT grade unrelated events or conditions.

    """
