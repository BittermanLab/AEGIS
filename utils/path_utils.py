from pathlib import Path
from typing import Set, Dict, Tuple

# Configuration Constants
GRAPH_TYPES: Set[str] = {
    "agent",
    "zeroshot",
    "task_specific",
    "full_workflow",
    "guided_full_workflow",
    "openai_agent",  # Add the new OpenAI agent type
    "agent_split_temporality",  # Add the new split temporality agent type
    "openai_zeroshot",
    "regex",
    "ray_agent",
    "ray_zeroshot",
    "ray_regex",
    "itox_agent",
    "itox_zeroshot",
    "itox_regex",
}

MODEL_CONFIGS: Set[str] = {
    "default",
    "4o-mini",
    "4.1-mini",
    "4.1-nano",
    "o3_mini",
    "o3-judge",
    "o3_middle",
    "hybrid",
    "ollama-deepseek-1b",  # Updated name
    "ollama-deepseek-8b",  # Updated name
    "ollama-deepseek-14b",  # Updated name
    "ollama-llama-1b",  # Added specific model
    "ollama-qwen3-14b",
    "vllm-deepseek-r1-8b",
    "vllm-qwen3-8b",
    "vllm-qwen3-14b",
    "vllm-qwen3-32b",
    "vllm-gemma3-4b",
    "vllm-gemma3-12b",
    "vllm-gemma3-27b",
    "vllm-medgemma-27b",
}
PROMPT_VARIANTS: Set[str] = {"default", "detailed", "concise", "base"}

# Directory structure mapping
GRAPH_TYPE_MAP: Dict[str, str] = {
    "agent": "graphs.agent.entry_graph",
    "zeroshot": "graphs.zeroshot.entry_graph",
    "task_specific": "graphs.task_specific.entry_graph",
    "full_workflow": "graphs.full_workflow.entry_graph",
    "guided_full_workflow": "graphs.guided_full_workflow.entry_graph",
    "openai_agent": "graphs.openai_agent.entry_graph",
    "agent_split_temporality": "graphs.agent_split_temporality.entry_graph",
    "openai_zeroshot": "graphs.openai_zeroshot.entry_graph",
    "regex": "graphs.regex.entry_graph",
    "ray_agent": "graphs.ray_agent.entry_graph",
    "ray_zeroshot": "graphs.ray_zeroshot.entry_graph",
    "ray_regex": "graphs.ray_regex.entry_graph",
    "itox_agent": "graphs.itox_agent.entry_graph",
    "itox_zeroshot": "graphs.itox_zeroshot.entry_graph",
    "itox_regex": "graphs.itox_regex.entry_graph",
    "zeroshot_test": "graphs.zeroshot_test.entry_graph",
}


def ensure_evaluation_structure(eval_dir: Path) -> None:
    """
    Create standardized evaluation directory structure.

    Structure:
    evaluation_dir/
    ├── metrics/
    """
    dirs = [
        # Metrics directories
        "metrics/",
    ]

    for dir_path in dirs:
        (eval_dir / dir_path).mkdir(parents=True, exist_ok=True)


def get_artifact_path(
    base_dir: Path,
    graph_type: str,
    artifact_type: str,
    model_config: str = "default",
    prompt_variant: str = "default",
) -> Path:
    """
    Generate standardized paths for artifacts.

    Args:
        base_dir: Base directory for outputs
        graph_type: Type of graph workflow
        artifact_type: Type of artifact (e.g., 'predictions', 'token_usage')
        model_config: Model configuration
        prompt_variant: Prompt variant name
    """
    if graph_type not in GRAPH_TYPE_MAP:
        raise ValueError(f"Invalid graph type: {graph_type}")

    # Use graph_type directly as the folder name
    graph_folder = graph_type

    # All graph types now include model_config and prompt_variant
    path = base_dir / graph_folder / model_config / f"variant_{prompt_variant}"

    return path / f"{artifact_type}.json"


def list_experiment_runs(base_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    List all experiment runs in the base directory.

    Returns:
        Dictionary mapping graph types to their run configurations
    """
    runs: Dict[str, Dict[str, Path]] = {}

    for graph_type in GRAPH_TYPE_MAP:
        runs[graph_type] = {}
        # Use graph_type directly as the folder name
        graph_path = base_dir / graph_type

        if not graph_path.exists():
            continue

        for pred_file in graph_path.rglob("predictions.json"):
            # Extract model_config and prompt_variant from path
            variant_dir = pred_file.parent.name  # variant_{prompt_variant}
            model_config = pred_file.parent.parent.name
            prompt_variant = variant_dir.replace("variant_", "")

            config_key = f"{model_config}_{prompt_variant}"
            runs[graph_type][config_key] = pred_file

    return runs
