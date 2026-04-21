"""
Workflow module for enhanced parallel processing with judge-based evaluation.
"""

from .event_processor import (
    run_parallel_temporality_classification,
    run_parallel_event_identification,
    judge_identification,
    run_parallel_grading,
    judge_grading,
    evaluate_event_processing,
    process_event_with_judge,
)

from .note_processor import (
    process_note_with_judge,
    process_all_notes_with_judge,
)

__all__ = [
    "run_parallel_temporality_classification",
    "run_parallel_event_identification",
    "judge_identification",
    "run_parallel_grading",
    "judge_grading",
    "evaluate_event_processing",
    "process_event_with_judge",
    "process_note_with_judge",
    "process_all_notes_with_judge",
]
