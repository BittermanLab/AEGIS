#!/usr/bin/env python3
"""
Script to evaluate and aggregate results across all sweeps.
This generates the final paper results by combining results from debug, dev, and prod sweeps.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import json

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from evaluate_sweep import (
    find_latest_predictions_file,
    process_sweep_outputs,
    create_enhanced_appendix_3d,
    extract_token_and_cost_data,
)
from generate_results import (
    load_predictions,
    calculate_all_metrics_from_predictions,
    calculate_overall_metrics,
    create_main_table,
    create_appendix_table1,
    create_appendix_3a_binary_performance,
    create_appendix_3b_macro_f1_performance,
    create_appendix_3c_comprehensive_metrics,
    create_appendix_grade_f1_breakdown,
    create_appendix_cross_temporal_analysis,
    generate_label_count_tables,
    generate_comprehensive_heatmaps,
    extract_confusion_matrices_from_metrics,
    save_markdown,
    ALLOWED_CONDITIONS,
    logger,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def find_all_sweep_outputs(base_output_dir: str) -> Dict[str, Path]:
    """Find all sweep output directories."""
    sweep_dirs = {}
    base_path = Path(base_output_dir)
    
    if not base_path.exists():
        logger.error(f"Base output directory does not exist: {base_output_dir}")
        return sweep_dirs
    
    # Look for sweep directories
    for item in base_path.iterdir():
        if item.is_dir() and item.name.endswith("_sweep"):
            sweep_name = item.name
            sweep_dirs[sweep_name] = item
            logger.info(f"Found sweep directory: {sweep_name} at {item}")
    
    return sweep_dirs


def process_all_sweeps(sweep_dirs: Dict[str, Path]) -> Tuple[pd.DataFrame, List[Dict]]:
    """Process all sweeps and combine results."""
    all_dfs = []
    all_processing_results = []
    
    for sweep_name, sweep_dir in sweep_dirs.items():
        logger.info(f"Processing sweep: {sweep_name}")
        
        try:
            # Process this sweep
            processing_results, combined_df = process_sweep_outputs(str(sweep_dir), sweep_name)
            
            if processing_results:
                # Add sweep name to results
                for result in processing_results:
                    result["sweep"] = sweep_name
                all_processing_results.extend(processing_results)
            
            if not combined_df.empty:
                # Add sweep name to dataframe
                combined_df["sweep"] = sweep_name
                all_dfs.append(combined_df)
                
        except Exception as e:
            logger.error(f"Error processing sweep {sweep_name}: {e}")
            continue
    
    # Combine all dataframes
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
    else:
        combined_df = pd.DataFrame()
    
    return combined_df, all_processing_results


def create_cross_sweep_summary(all_processing_results: List[Dict]) -> str:
    """Create a summary table showing results across all sweeps."""
    table_header = "# Cross-Sweep Summary: All Experiments Across All Sweeps\n\n"
    table_header += "Overview of all model/variant combinations tested across debug, dev, and prod sweeps\n\n"
    
    headers = ["Sweep", "Graph Type", "Model", "Variant", "Status", "Predictions", "Total Cost ($)"]
    
    table = [
        table_header,
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["-" * (len(h) + 2) for h in headers]) + "|"
    ]
    
    # Group by sweep
    sweep_groups = {}
    for result in all_processing_results:
        sweep = result.get("sweep", "unknown")
        if sweep not in sweep_groups:
            sweep_groups[sweep] = []
        sweep_groups[sweep].append(result)
    
    total_cost_all = 0.0
    total_predictions_all = 0
    
    # Process each sweep
    for sweep in sorted(sweep_groups.keys()):
        results = sweep_groups[sweep]
        sweep_total_cost = 0.0
        sweep_total_predictions = 0
        
        for result in sorted(results, key=lambda x: (x.get("graph_type", ""), x.get("model", ""), x.get("variant", ""))):
            # Extract cost info
            token_data = result.get("token_data", {})
            total_cost = 0.0
            if token_data:
                for model_data in token_data.values():
                    total_cost += model_data.get("total_cost", 0.0)
            
            row = [
                sweep,
                result.get("graph_type", "N/A"),
                result.get("model", "N/A"),
                result.get("variant", "N/A"),
                result["status"],
                str(result.get("predictions_count", 0)),
                f"${total_cost:.4f}" if total_cost > 0 else "N/A"
            ]
            table.append("| " + " | ".join(row) + " |")
            
            if result["status"] == "Success":
                sweep_total_cost += total_cost
                sweep_total_predictions += result.get("predictions_count", 0)
        
        # Add sweep subtotal
        table.append(f"| **{sweep} Total** | - | - | - | - | **{sweep_total_predictions}** | **${sweep_total_cost:.4f}** |")
        table.append("| | | | | | | |")  # Empty row for spacing
        
        total_cost_all += sweep_total_cost
        total_predictions_all += sweep_total_predictions
    
    # Add grand total
    table.append(f"| **GRAND TOTAL** | - | - | - | - | **{total_predictions_all}** | **${total_cost_all:.4f}** |")
    
    # Add summary statistics
    successful = sum(1 for r in all_processing_results if r["status"] == "Success")
    failed = sum(1 for r in all_processing_results if r["status"] == "Failed")
    
    footer = f"\n\n**Summary Statistics:**\n"
    footer += f"- Total sweeps processed: {len(sweep_groups)}\n"
    footer += f"- Total configurations tested: {len(all_processing_results)}\n"
    footer += f"- Successful: {successful}\n"
    footer += f"- Failed: {failed}\n"
    footer += f"- Total predictions processed: {total_predictions_all:,}\n"
    footer += f"- Total cost across all sweeps: ${total_cost_all:.4f}\n"
    
    return "\n".join(table) + footer


def select_best_configuration(combined_df: pd.DataFrame) -> Optional[pd.Series]:
    """Select the best performing configuration based on overall metrics."""
    if combined_df.empty:
        return None
    
    # Priority order for selection:
    # 1. prod_agent_sweep with 4.1-mini and default variant
    # 2. Any sweep with 4.1-mini and default variant
    # 3. Highest overall binary F1
    
    # Try to find prod sweep with 4.1-mini default
    mask = (
        (combined_df["sweep"] == "prod_agent_sweep") &
        (combined_df["model"] == "4.1-mini") &
        (combined_df["variant"] == "default")
    )
    if mask.any():
        return combined_df[mask].iloc[0]
    
    # Try to find any 4.1-mini default
    mask = (
        (combined_df["model"] == "4.1-mini") &
        (combined_df["variant"] == "default")
    )
    if mask.any():
        # Get the one from the most complete sweep (prod > dev > debug)
        subset = combined_df[mask]
        for sweep in ["prod_agent_sweep", "dev_agent_sweep", "debug_agent_sweep"]:
            sweep_mask = subset["sweep"] == sweep
            if sweep_mask.any():
                return subset[sweep_mask].iloc[0]
    
    # Otherwise, get the one with highest overall binary F1
    if "overall_binary_f1" in combined_df.columns:
        return combined_df.loc[combined_df["overall_binary_f1"].idxmax()]
    
    # Last resort: just return the first one
    return combined_df.iloc[0]


def main():
    """Main function to run cross-sweep evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate results across all sweeps")
    parser.add_argument(
        "--output-base-dir",
        default="data/synthetic_outputs",
        help="Base directory containing sweep outputs"
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to save final results"
    )
    parser.add_argument(
        "--primary-sweep",
        default="prod_agent_sweep",
        help="Primary sweep to use for main results (default: prod_agent_sweep)"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting cross-sweep evaluation")
        
        # Find all sweep outputs
        sweep_dirs = find_all_sweep_outputs(args.output_base_dir)
        if not sweep_dirs:
            logger.error("No sweep directories found")
            return 1
        
        logger.info(f"Found {len(sweep_dirs)} sweeps: {list(sweep_dirs.keys())}")
        
        # Process all sweeps
        combined_df, all_processing_results = process_all_sweeps(sweep_dirs)
        
        if not all_processing_results:
            logger.error("No results to process across all sweeps")
            return 1
        
        # Create output directories
        os.makedirs(f"{args.results_dir}/main", exist_ok=True)
        os.makedirs(f"{args.results_dir}/appendix", exist_ok=True)
        os.makedirs(f"{args.results_dir}/visualizations", exist_ok=True)
        
        # Create cross-sweep summary
        cross_sweep_summary = create_cross_sweep_summary(all_processing_results)
        save_markdown(cross_sweep_summary, f"{args.results_dir}/appendix/cross_sweep_summary.md")
        
        # Select best configuration for main results
        if not combined_df.empty:
            best_config = select_best_configuration(combined_df)
            
            if best_config is not None:
                logger.info(f"Selected best configuration: {best_config['sweep']}/{best_config['model']}/{best_config['variant']}")
                
                # Find the predictions file for the best configuration
                best_result = None
                for result in all_processing_results:
                    if (result["status"] == "Success" and
                        result.get("sweep") == best_config["sweep"] and
                        result.get("model") == best_config["model"] and
                        result.get("variant") == best_config["variant"]):
                        best_result = result
                        break
                
                if best_result and best_result.get("file_path"):
                    # Load predictions and generate main results
                    predictions = load_predictions(Path(best_result["file_path"]))
                    df_main = calculate_all_metrics_from_predictions(predictions)
                    
                    # Create main table
                    main_table = create_main_table(df_main, ALLOWED_CONDITIONS)
                    save_markdown(main_table, f"{args.results_dir}/main/main_results_table.md")
                    
                    # Create appendix table 1
                    appendix_table1 = create_appendix_table1(df_main, ALLOWED_CONDITIONS)
                    save_markdown(appendix_table1, f"{args.results_dir}/appendix/appendix_table1.md")
                    
                    # Generate additional appendix tables
                    appendix_grade_f1 = create_appendix_grade_f1_breakdown(predictions)
                    save_markdown(appendix_grade_f1, f"{args.results_dir}/appendix/appendix_table4.md")
                    
                    appendix_temporal = create_appendix_cross_temporal_analysis(predictions)
                    save_markdown(appendix_temporal, f"{args.results_dir}/appendix/appendix_table5.md")
                    
                    # Generate visualizations
                    all_matrices = extract_confusion_matrices_from_metrics(df_main)
                    if all_matrices:
                        generate_comprehensive_heatmaps(all_matrices, f"{args.results_dir}/visualizations")
            
            # Generate comparison tables across all configurations
            appendix_table3a = create_appendix_3a_binary_performance(combined_df)
            save_markdown(appendix_table3a, f"{args.results_dir}/appendix/appendix_table3a.md")
            
            appendix_table3b = create_appendix_3b_macro_f1_performance(combined_df)
            save_markdown(appendix_table3b, f"{args.results_dir}/appendix/appendix_table3b.md")
            
            appendix_table3c = create_appendix_3c_comprehensive_metrics(combined_df)
            save_markdown(appendix_table3c, f"{args.results_dir}/appendix/appendix_table3c.md")
        
        # Generate label count tables (these are dataset-specific, not sweep-specific)
        generate_label_count_tables()
        
        logger.info(f"Cross-sweep evaluation complete! Results saved to {args.results_dir}/")
        logger.info("Main results use the best performing configuration")
        logger.info("Appendix tables show comparisons across all sweeps and configurations")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in cross-sweep evaluation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())