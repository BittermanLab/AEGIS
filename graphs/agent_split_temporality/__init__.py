"""
OpenAI Agent processing module.

This module provides an adapter to run the OpenAI agent for clinical note processing.
"""

try:
    from .model_config import MODEL_CONFIGS, OpenAIModelConfig
except ModuleNotFoundError:
    MODEL_CONFIGS = {}
    OpenAIModelConfig = None

try:
    from .entry_graph import graph, create_workflow
except ModuleNotFoundError:
    # Allow importing package modules (for tests/docs) without optional runtime deps.
    graph = None

    def create_workflow(*args, **kwargs):  # type: ignore[no-redef]
        """Create workflow when optional dependencies are available."""
        raise ModuleNotFoundError(
            "Missing optional dependency 'agents'. Install runtime dependencies to use split-temporality workflows."
        )
