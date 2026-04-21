import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from pathlib import Path
import json
import logging
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore

logger = logging.getLogger(__name__)


class UsageProcessor:
    """Process and analyze token usage statistics."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)

    def process_usage_file(self, filepath: Path) -> Dict[str, Any]:
        """Process individual usage statistics file."""
        try:
            with open(filepath) as f:
                data = json.load(f)

            # Extract path components relative to base_dir
            rel_path = filepath.relative_to(self.base_dir)
            path_parts = list(rel_path.parts)

            # Expected path structure:
            # sweep_name/graph_type/reasoning_type/model_config/variant_name/token_usage.json
            # e.g. debug/task_specific/base/default/variant_default/token_usage.json

            # Initialize metadata with defaults
            metadata = {
                "sweep_name": None,
                "graph_type": None,
                "reasoning_type": None,
                "model_config": None,
                "prompt_variant": None,
            }

            # Parse path components if we have enough parts
            if len(path_parts) >= 5:
                # Extract variant name without the 'variant_' prefix
                variant_dir = path_parts[4]
                variant_name = (
                    variant_dir.replace("variant_", "")
                    if "variant_" in variant_dir
                    else variant_dir
                )

                metadata.update(
                    {
                        "sweep_name": path_parts[0],  # e.g. 'debug'
                        "graph_type": path_parts[1],  # e.g. 'task_specific'
                        "reasoning_type": path_parts[2],  # e.g. 'base'
                        "model_config": path_parts[3],  # e.g. 'default'
                        "prompt_variant": variant_name,  # e.g. 'default' (from 'variant_default')
                    }
                )

            # Add experiment details from the data
            metadata.update(
                {
                    "experiment_run_name": data.get("experiment_run_name"),
                    "experiment_run_id": data.get("experiment_run_id"),
                }
            )

            # Extract usage statistics
            usage_stats = data.get("overall_stats", {})

            # Process per-model statistics
            model_stats = {}
            for model, stats in data.get("per_model_stats", {}).items():
                for metric, value in stats.items():
                    model_stats[f"{model}_{metric}"] = value

            return {
                **metadata,
                **usage_stats,
                **model_stats,
                "debug_mode": data.get("config_details", {}).get("debug_mode", False),
            }

        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            return {}

    def collect_usage_data(self) -> pd.DataFrame:
        """Collect usage statistics from all files."""
        usage_data = []

        # Track all unique models and their metrics
        all_models = set()
        metric_suffixes = [
            "_prompt_tokens",
            "_completion_tokens",
            "_total_tokens",
            "_prompt_cost",
            "_completion_cost",
            "_total_cost",
        ]

        # First pass: collect all model names
        for usage_file in self.base_dir.rglob("token_usage.json"):
            data = self.process_usage_file(usage_file)
            if data:
                # Extract model names from keys
                for key in data.keys():
                    for suffix in metric_suffixes:
                        if suffix in key:
                            model_name = key.replace(suffix, "")
                            all_models.add(model_name)
                usage_data.append(data)

        # Create DataFrame
        df = pd.DataFrame(usage_data)

        # Ensure all model metrics exist and fill empty values with 0.0
        for model in all_models:
            for suffix in metric_suffixes:
                column = f"{model}{suffix}"
                if column not in df.columns:
                    df[column] = 0.0
                else:
                    # Fill empty values with 0.0
                    df[column] = df[column].fillna(0.0)

        return df


def generate_usage_summaries(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate usage summaries with clear, single-level headers."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Overall usage metrics
    overall_metrics = {
        "prompt_tokens_sum": ("prompt_tokens", "sum"),
        "prompt_tokens_mean": ("prompt_tokens", "mean"),
        "completion_tokens_sum": ("completion_tokens", "sum"),
        "completion_tokens_mean": ("completion_tokens", "mean"),
        "total_tokens_sum": ("total_tokens", "sum"),
        "total_tokens_mean": ("total_tokens", "mean"),
        "total_cost_sum": ("total_cost_usd", "sum"),
        "total_cost_mean": ("total_cost_usd", "mean"),
        "run_time_mean": ("avg_run_time_sec_per_patient", "mean"),
        "total_patients": ("total_patients", "sum"),
    }

    overall_summary = (
        df.groupby(["graph_type", "model_config"])
        .agg(**{name: aggfunc for name, aggfunc in overall_metrics.items()})
        .round(4)
    )
    overall_summary.to_csv(output_dir / "overall_usage.csv")

    # Model-specific metrics
    model_metrics = {}
    for col in df.columns:
        if any(
            x in col
            for x in ["_prompt_tokens", "_completion_tokens", "_total_tokens", "_cost"]
        ):
            model_metrics.update(
                {
                    f"{col}_sum": (col, "sum"),
                    f"{col}_mean": (col, "mean"),
                    f"{col}_std": (col, "std"),
                }
            )

    if model_metrics:
        model_summary = (
            df.groupby(["graph_type", "model_config"]).agg(**model_metrics).round(4)
        )
        model_summary.to_csv(output_dir / "model_specific_usage.csv")

    # Prompt variant analysis
    if "prompt_variant" in df.columns:
        variant_metrics = {
            "total_tokens_mean": ("total_tokens", "mean"),
            "total_tokens_std": ("total_tokens", "std"),
            "total_cost_mean": ("total_cost_usd", "mean"),
            "total_cost_std": ("total_cost_usd", "std"),
            "run_time_mean": ("avg_run_time_sec_per_patient", "mean"),
            "run_time_std": ("avg_run_time_sec_per_patient", "std"),
        }

        variant_summary = (
            df.groupby(["graph_type", "prompt_variant"]).agg(**variant_metrics).round(4)
        )
        variant_summary.to_csv(output_dir / "prompt_variant_usage.csv")
