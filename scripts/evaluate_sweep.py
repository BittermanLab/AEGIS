#!/usr/bin/env python3
"""
Evaluate and compare all experiments within a single sweep.
This script finds all prediction files for each experiment configuration
(graph_type/model/variant) and generates comparison tables.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
from datetime import datetime
import pandas as pd

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# Import our utilities
from generate_results import (
    load_predictions,
    calculate_all_metrics_from_predictions,
    process_all_model_variants,
    create_appendix_3d_processing_summary,
    save_markdown,
    ALLOWED_CONDITIONS,
    logger,
    create_main_table,
    create_appendix_table1,
    create_appendix_3a_binary_performance,
    create_appendix_3b_macro_f1_performance,
    create_appendix_3c_comprehensive_metrics,
    generate_label_count_tables,
    generate_comprehensive_heatmaps,
    extract_confusion_matrices_from_metrics,
    calculate_overall_metrics,
)
from metrics_utils import DEFAULT_F1

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def find_latest_predictions_file(
    output_dir: str, graph_type: str, model_config: str, variant: str = "default"
) -> Optional[Path]:
    """
    Find the most recent predictions file for a specific configuration.

    Args:
        output_dir: Base output directory (e.g., data/synthetic_outputs/debug_agent_sweep/)
        graph_type: Graph type (e.g., openai_agent, regex, openai_zeroshot)
        model_config: Model configuration (e.g., default, 4.1-mini, etc.)
        variant: Prompt variant (default: "default")

    Returns:
        Path to the latest predictions file or None if not found
    """
    # Construct the path to the variant directory
    variant_dir = Path(output_dir) / graph_type / model_config / f"variant_{variant}"

    if not variant_dir.exists():
        logger.warning(f"Directory does not exist: {variant_dir}")
        return None

    # Find all predictions files
    files = list(variant_dir.glob("predictions_*.json"))
    if not files:
        logger.warning(f"No predictions files found in {variant_dir}")
        return None

    # Get the most recent file
    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    return latest_file


def extract_token_and_cost_data(predictions: List[Dict]) -> Dict[str, Dict]:
    """
    Extract token usage and cost data from predictions.

    Returns:
        Dict with aggregated token and cost data per model configuration
    """
    token_data = {}

    for pred in predictions:
        if "token_usage" not in pred:
            continue

        token_usage = pred["token_usage"]
        model_key = token_usage.get("model", "unknown")

        if model_key not in token_data:
            token_data[model_key] = {
                "total_predictions": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "total_prompt_cost": 0.0,
                "total_completion_cost": 0.0,
                "total_cost": 0.0,
                "processing_times": [],
                "cost_rates": token_usage.get(
                    "cost_rates", {"prompt": 0.0, "completion": 0.0}
                ),
            }

        # Aggregate token counts
        token_data[model_key]["total_predictions"] += 1
        token_data[model_key]["total_prompt_tokens"] += token_usage.get(
            "prompt_tokens", 0
        )
        token_data[model_key]["total_completion_tokens"] += token_usage.get(
            "completion_tokens", 0
        )
        token_data[model_key]["total_tokens"] += token_usage.get("total_tokens", 0)
        token_data[model_key]["total_prompt_cost"] += token_usage.get(
            "prompt_cost", 0.0
        )
        token_data[model_key]["total_completion_cost"] += token_usage.get(
            "completion_cost", 0.0
        )
        token_data[model_key]["total_cost"] += token_usage.get("total_cost", 0.0)

        # Track processing times
        if "processing_time" in pred:
            token_data[model_key]["processing_times"].append(pred["processing_time"])

    # Calculate averages
    for model_key, data in token_data.items():
        if data["total_predictions"] > 0:
            data["avg_tokens_per_prediction"] = (
                data["total_tokens"] / data["total_predictions"]
            )
            data["avg_cost_per_prediction"] = (
                data["total_cost"] / data["total_predictions"]
            )

            if data["processing_times"]:
                data["avg_processing_time"] = sum(data["processing_times"]) / len(
                    data["processing_times"]
                )
                data["total_processing_time"] = sum(data["processing_times"])
            else:
                data["avg_processing_time"] = 0.0
                data["total_processing_time"] = 0.0

    return token_data


def create_enhanced_appendix_3d(processing_results: List[Dict], sweep_name: str) -> str:
    """Create enhanced appendix table 3d with token usage and costs."""
    table_header = (
        f"# Appendix Table 3d: Experiment Processing Summary - {sweep_name}\n\n"
    )
    table_header += "Summary of all experiments (graph type/model combinations) processed in this sweep.\n\n"

    # Enhanced headers with cost information
    headers = [
        "Graph Type",
        "Model",
        "Variant",
        "Status",
        "Predictions",
        "Total Tokens",
        "Prompt Tokens",
        "Completion Tokens",
        "Total Cost ($)",
        "Avg Cost/Pred ($)",
        "Avg Time (s)",
        "File Path",
    ]

    table = [
        table_header,
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["-" * (len(h) + 2) for h in headers]) + "|",
    ]

    total_cost = 0.0
    total_tokens = 0
    total_predictions = 0

    # Sort results for consistent display
    sorted_results = sorted(
        processing_results,
        key=lambda x: (
            x.get("graph_type", ""),
            x.get("model", ""),
            x.get("variant", ""),
        ),
    )

    for result in sorted_results:
        # Extract token data if available
        token_info = result.get("token_data", {})

        # Get the first (usually only) model's data
        model_data = {}
        if token_info:
            # Get the first model's data (there's usually only one per run)
            model_data = next(iter(token_info.values()), {})

        row = [
            result.get("graph_type", "N/A"),
            result.get("model", "N/A"),
            result.get("variant", "N/A"),
            result["status"],
            str(result.get("predictions_count", 0)),
            f"{model_data.get('total_tokens', 0):,}" if model_data else "N/A",
            f"{model_data.get('total_prompt_tokens', 0):,}" if model_data else "N/A",
            (
                f"{model_data.get('total_completion_tokens', 0):,}"
                if model_data
                else "N/A"
            ),
            f"${model_data.get('total_cost', 0):.4f}" if model_data else "N/A",
            (
                f"${model_data.get('avg_cost_per_prediction', 0):.4f}"
                if model_data
                else "N/A"
            ),
            f"{model_data.get('avg_processing_time', 0):.2f}" if model_data else "N/A",
            (
                result.get("file_path", "N/A").split("/")[-1]
                if result.get("file_path") != "N/A"
                else "N/A"
            ),
        ]
        table.append("| " + " | ".join(row) + " |")

        # Accumulate totals
        if result["status"] == "Success" and model_data:
            total_cost += model_data.get("total_cost", 0)
            total_tokens += model_data.get("total_tokens", 0)
            total_predictions += result.get("predictions_count", 0)

    # Add summary
    successful = sum(1 for r in processing_results if r["status"] == "Success")
    failed = sum(1 for r in processing_results if r["status"] == "Failed")

    footer = f"\n\n**Summary:**\n"
    footer += f"- Total experiments in sweep: {len(processing_results)}\n"
    footer += f"- Successful: {successful}\n"
    footer += f"- Failed: {failed}\n"
    footer += f"- Total predictions processed: {total_predictions:,}\n"
    footer += f"- Total tokens used: {total_tokens:,}\n"
    footer += f"- Total cost: ${total_cost:.4f}\n"
    if total_predictions > 0:
        footer += (
            f"- Average cost per prediction: ${total_cost/total_predictions:.4f}\n"
        )

    footer += f"\n**Notes:**\n"
    footer += f"- This table shows all experiments run in the {sweep_name} sweep\n"
    footer += f"- Graph Type indicates the approach used (openai_agent, openai_zeroshot, regex, etc.)\n"
    footer += f"- Costs are calculated based on model-specific rates configured in the system\n"

    return "\n".join(table) + footer


def process_sweep_outputs(
    sweep_dir: str, sweep_name: str
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Process all outputs from a sweep, comparing different experiments.

    Returns:
        Tuple of (processing_results, combined_df)
    """
    processing_results = []
    all_dfs = []

    # Check what directories exist in the sweep output
    sweep_path = Path(sweep_dir)
    if not sweep_path.exists():
        logger.error(f"Sweep directory does not exist: {sweep_dir}")
        return processing_results, pd.DataFrame()

    logger.info(f"Processing sweep at: {sweep_dir}")

    # Find all configurations by exploring the directory structure
    configurations = []

    # Find all graph type directories
    for graph_type_dir in sweep_path.iterdir():
        if not graph_type_dir.is_dir():
            continue

        graph_type = graph_type_dir.name
        logger.info(f"Found graph type: {graph_type}")

        # Find all model directories
        for model_dir in graph_type_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_config = model_dir.name

            # Find all variant directories
            for variant_dir in model_dir.iterdir():
                if not variant_dir.is_dir() or not variant_dir.name.startswith(
                    "variant_"
                ):
                    continue

                variant = variant_dir.name.replace("variant_", "")

                configurations.append(
                    {
                        "graph_type": graph_type,
                        "model": model_config,
                        "variant": variant,
                    }
                )

    logger.info(f"Found {len(configurations)} experiment configurations to process")

    # Process each configuration
    for config in configurations:
        result_info = {
            "graph_type": config["graph_type"],
            "model": config["model"],
            "variant": config["variant"],
            "status": "Failed",
            "predictions_count": 0,
            "file_path": "N/A",
            "token_data": {},
        }

        try:
            # Find predictions file
            latest_file = find_latest_predictions_file(
                sweep_dir, config["graph_type"], config["model"], config["variant"]
            )

            if not latest_file:
                logger.warning(f"No predictions file found for {config}")
                processing_results.append(result_info)
                continue

            result_info["file_path"] = str(latest_file)
            logger.info(
                f"Processing {config['graph_type']}/{config['model']}/{config['variant']}: {latest_file}"
            )

            # Load predictions
            predictions = load_predictions(latest_file)
            result_info["predictions_count"] = len(predictions)

            # Extract token and cost data
            token_data = extract_token_and_cost_data(predictions)
            result_info["token_data"] = token_data

            # Calculate metrics
            df = calculate_all_metrics_from_predictions(predictions)

            # Add configuration info
            df["graph_type"] = config["graph_type"]
            df["model"] = config["model"]
            df["variant"] = config["variant"]
            df["experiment"] = (
                f"{config['graph_type']}_{config['model']}_{config['variant']}"
            )

            # Calculate overall metrics across all conditions
            overall_metrics = calculate_overall_metrics(df, ALLOWED_CONDITIONS)

            # Add overall metrics to the dataframe
            for metric_name, metric_value in overall_metrics.items():
                df[metric_name] = metric_value

            all_dfs.append(df)

            result_info["status"] = "Success"

        except Exception as e:
            logger.error(f"Error processing {config}: {e}")
            result_info["error"] = str(e)

        processing_results.append(result_info)

    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    if not combined_df.empty:
        logger.info(f"Combined results from {len(all_dfs)} successful experiments")
        logger.info(
            f"Experiments included: {combined_df['experiment'].unique().tolist()}"
        )

    return processing_results, combined_df


def select_primary_configuration(
    processing_results: List[Dict], combined_df: pd.DataFrame
) -> Optional[Dict]:
    """
    Select the primary configuration for detailed results.
    Priority: openai_agent with 4.1-mini and default variant
    """
    # First try to find openai_agent with 4.1-mini and default variant
    for result in processing_results:
        if (
            result["status"] == "Success"
            and result.get("graph_type") == "openai_agent"
            and result.get("model") == "4.1-mini"
            and result.get("variant") == "default"
        ):
            return result

    # If not found, try any 4.1-mini with default variant
    for result in processing_results:
        if (
            result["status"] == "Success"
            and result.get("model") == "4.1-mini"
            and result.get("variant") == "default"
        ):
            return result

    # Otherwise, return first successful result
    for result in processing_results:
        if result["status"] == "Success":
            return result

    return None


def main():
    """Main function to evaluate a single sweep."""
    parser = argparse.ArgumentParser(
        description="Evaluate all experiments within a sweep"
    )
    parser.add_argument(
        "--sweep-dir",
        required=True,
        help="Directory containing sweep outputs (e.g., data/synthetic_outputs/debug_agent_sweep)",
    )
    parser.add_argument(
        "--sweep-name", required=True, help="Name of the sweep for labeling outputs"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save results (default: results)",
    )

    args = parser.parse_args()

    try:
        logger.info(f"=" * 80)
        logger.info(f"Evaluating sweep: {args.sweep_name}")
        logger.info(f"Sweep directory: {args.sweep_dir}")
        logger.info(f"=" * 80)

        # Create output directories
        output_base = Path(args.output_dir) / "sweeps" / args.sweep_name
        os.makedirs(output_base / "main", exist_ok=True)
        os.makedirs(output_base / "appendix", exist_ok=True)
        os.makedirs(output_base / "visualizations", exist_ok=True)

        # Process all outputs from the sweep
        processing_results, combined_df = process_sweep_outputs(
            args.sweep_dir, args.sweep_name
        )

        if not processing_results:
            logger.error("No results to process")
            return 1

        # Log processing summary
        successful = sum(1 for r in processing_results if r["status"] == "Success")
        logger.info(
            f"\nProcessed {len(processing_results)} experiments: {successful} successful"
        )

        # Create enhanced appendix table 3d with all experiments
        enhanced_table_3d = create_enhanced_appendix_3d(
            processing_results, args.sweep_name
        )
        save_markdown(
            enhanced_table_3d, output_base / "appendix" / "appendix_table3d.md"
        )

        # If we have successful results, generate comparison tables
        if not combined_df.empty:
            logger.info("\nGenerating comparison tables across experiments...")

            # Select primary configuration for main results
            primary_config = select_primary_configuration(
                processing_results, combined_df
            )

            if primary_config:
                logger.info(
                    f"\nPrimary configuration for main results: {primary_config['graph_type']}/{primary_config['model']}/{primary_config['variant']}"
                )

                # Load predictions for primary configuration
                predictions = load_predictions(Path(primary_config["file_path"]))
                df_main = calculate_all_metrics_from_predictions(predictions)

                # Create main table
                main_table = create_main_table(df_main, ALLOWED_CONDITIONS)
                save_markdown(
                    main_table, output_base / "main" / "main_results_table.md"
                )

                # Create appendix table 1
                appendix_table1 = create_appendix_table1(df_main, ALLOWED_CONDITIONS)
                save_markdown(
                    appendix_table1, output_base / "appendix" / "appendix_table1.md"
                )

                # Generate visualizations for primary configuration
                all_matrices = extract_confusion_matrices_from_metrics(df_main)
                if all_matrices:
                    generate_comprehensive_heatmaps(
                        all_matrices, output_base / "visualizations"
                    )

            # Generate comparison tables across all experiments
            logger.info("\nGenerating experiment comparison tables...")

            appendix_table3a = create_appendix_3a_binary_performance(combined_df)
            save_markdown(
                appendix_table3a, output_base / "appendix" / "appendix_table3a.md"
            )

            appendix_table3b = create_appendix_3b_macro_f1_performance(combined_df)
            save_markdown(
                appendix_table3b, output_base / "appendix" / "appendix_table3b.md"
            )

            appendix_table3c = create_appendix_3c_comprehensive_metrics(combined_df)
            save_markdown(
                appendix_table3c, output_base / "appendix" / "appendix_table3c.md"
            )

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Evaluation complete! Results saved to {output_base}/")
        logger.info(f"\nMain results show performance of the primary configuration")
        logger.info(f"Appendix tables 3a-3c compare all experiments in this sweep")
        logger.info(f"Appendix table 3d shows processing details and costs")
        logger.info(f"{'=' * 80}")

        return 0

    except Exception as e:
        logger.error(f"Error in evaluation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
