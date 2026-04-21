from pathlib import Path
from typing import Set, Dict, Tuple

# Configuration Constants
GRAPH_TYPES: Set[str] = {
    "agent",
    "zeroshot",
    "task_specific",
    "full_workflow",
    "guided_full_workflow",
}

MODEL_CONFIGS: Set[str] = {
    "default",
    "o1",
    "o3_mini",
    "o3_early",
    "o3_late",
    "hybrid",
}
PROMPT_VARIANTS: Set[str] = {"default", "detailed", "concise", "base"}

# Directory structure mapping
GRAPH_TYPE_MAP: Dict[str, Tuple[str, str]] = {
    "agent": ("agent", "base"),
    "zeroshot": ("zeroshot", "base"),
    "task_specific": ("task_specific", "base"),
    "full_workflow": ("full_workflow", "base"),
    "guided_full_workflow": ("guided_full_workflow", "base"),
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

    graph_folder, reasoning_folder = GRAPH_TYPE_MAP[graph_type]

    # All graph types now include model_config and prompt_variant
    path = (
        base_dir
        / graph_folder
        / reasoning_folder
        / model_config
        / f"variant_{prompt_variant}"
    )

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
        graph_folder, _ = GRAPH_TYPE_MAP[graph_type]
        graph_path = base_dir / graph_folder

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
