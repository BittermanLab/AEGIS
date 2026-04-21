"""
Dynamic prompts module that generates instructions based on event context.
"""

from .temporal_identifier import dynamic_temporal_identifier_instructions
from .grader import dynamic_grader_instructions
from .attribution import dynamic_attribution_instructions
from .certainty import dynamic_certainty_instructions

__all__ = [
    "dynamic_temporal_identifier_instructions",
    "dynamic_grader_instructions",
    "dynamic_attribution_instructions",
    "dynamic_certainty_instructions",
]
