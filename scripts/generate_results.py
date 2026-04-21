#!/usr/bin/env python3
"""
Generate final results tables from raw prediction files.

This script reads prediction JSON files, calculates metrics using scikit-learn,
and generates markdown tables and visualizations for the results.
"""

import os
import pandas as pd
import glob
from pathlib import Path
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

# Import our metrics utilities
from metrics_utils import (
    summarise_condition,
    calculate_metrics_for_type,
    convert_to_binary_confusion_matrix,
    calculate_binary_metrics_from_matrix,
    GRADE_CLASSES,
    ATTR_CLASSES,
    CERT_CLASSES,
    DEFAULT_F1,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Configuration variables - these can be overridden when imported
FINAL_RESULTS_DIR = "data/rwd/prod_outputs/prod_agent_sweep/openai_agent/"
# Models for different graph types
OPENAI_AGENT_MODELS = ["4.1-mini", "4.1-nano", "o4-mini"]
OPENAI_SPLIT_TEMPORALITY_MODELS = ["4.1-mini", "4.1-nano", "o4-mini"]
ZEROSHOT_MODELS = ["4.1-mini", "4.1-nano", "o4-mini"]
REGEX_MODELS = ["default"]
ALL_MODELS = (
    OPENAI_AGENT_MODELS
    + OPENAI_SPLIT_TEMPORALITY_MODELS
    + ZEROSHOT_MODELS
    + REGEX_MODELS
)
MODELS = ALL_MODELS  # For backward compatibility
VARIANTS = ["variant_default", "variant_ablation_no_judge", "variant_ablation_single"]
TAKE_MOST_RECENT = True

# Default model/variant for main results
DEFAULT_MODEL = "4.1-mini"
DEFAULT_VARIANT = "variant_default"

# Define allowed conditions
ALLOWED_CONDITIONS = [
    "thyroiditis",
    "hepatitis",
    "colitis",
    "pneumonitis",
    "myocarditis",
    "dermatitis",
]

# Define mapping of snake_case to standard condition names
CONDITION_MAPPING = {
    "pneumonitis": "Pneumonitis",
    "myocarditis": "Myocarditis",
    "colitis": "Colitis",
    "thyroiditis": "Thyroiditis",
    "hepatitis": "Hepatitis",
    "dermatitis": "Dermatitis",
}

# Mapping between true label format and predicted label format
TRUE_LABEL_MAPPING = {
    "pneumonitis": "Pneum",
    "colitis": "Col",
    "hepatitis": "Hep",
    "dermatitis": "Derm",
    "thyroiditis": "Thyr",
    "myocarditis": "Myo",
}

# Define fields for label counting
FIELDS = [
    "CurrentGrade_Pneum",
    "CurrentAttr_Pneum",
    "CurrentCert_Pneum",
    "PastMaxGrade_Pneum",
    "PastAttr_Pneum",
    "PastCert_Pneum",
    "CurrentGrade_Col",
    "CurrentAttr_Col",
    "CurrentCert_Col",
    "PastMaxGrade_Col",
    "PastAttr_Col",
    "PastCert_Col",
    "CurrentGrade_Hep",
    "CurrentAttr_Hep",
    "CurrentCert_Hep",
    "PastMaxGrade_Hep",
    "PastAttr_Hep",
    "PastCert_Hep",
    "CurrentGrade_Derm",
    "CurrentAttr_Derm",
    "CurrentCert_Derm",
    "PastMaxGrade_Derm",
    "PastAttr_Derm",
    "PastCert_Derm",
    "CurrentGrade_Thyr",
    "CurrentAttr_Thyr",
    "CurrentCert_Thyr",
    "PastMaxGrade_Thyr",
    "PastAttr_Thyr",
    "PastCert_Thyr",
    "CurrentGrade_Myo",
    "CurrentAttr_Myo",
    "CurrentCert_Myo",
    "PastMaxGrade_Myo",
    "PastAttr_Myo",
    "PastCert_Myo",
]

# Field groups by condition
FIELD_GROUPS = {
    "Pneumonitis": ["Pneum"],
    "Colitis": ["Col"],
    "Hepatitis": ["Hep"],
    "Dermatitis": ["Derm"],
    "Thyroiditis": ["Thyr"],
    "Myocarditis": ["Myo"],
}

# Field suffix mapping
field_suffix_mapping = {
    "Pneum": "Pneumonitis",
    "Col": "Colitis",
    "Hep": "Hepatitis",
    "Derm": "Dermatitis",
    "Thyr": "Thyroiditis",
    "Myo": "Myocarditis",
}


def find_latest_predictions_file(model: str, variant: str) -> Path:
    """
    Find the most recent predictions file for a specific model and variant.

    Args:
        model: Model name (e.g., "4.1-mini")
        variant: Variant name (e.g., "variant_default")

    Returns:
        Path to the latest predictions file
    """
    base_dir = Path(FINAL_RESULTS_DIR) / model / variant
    files = list(base_dir.glob("predictions_*.json"))
    if not files:
        raise FileNotFoundError(f"No predictions files found in {base_dir}")
    latest_file = max(files, key=lambda x: x.name)
    return latest_file


def load_predictions(file_path: Path) -> List[Dict]:
    """Load predictions from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def extract_labels_from_predictions(
    predictions: List[Dict], condition: str, temporal: str, label_type: str
) -> Tuple[List[int], List[int]]:
    """
    Extract true and predicted labels for a specific condition, temporal, and type.

    Args:
        predictions: List of prediction dictionaries
        condition: Condition name (e.g., 'pneumonitis')
        temporal: 'current' or 'past'
        label_type: 'grade', 'attribution', or 'certainty'

    Returns:
        Tuple of (true_labels, predicted_labels)
    """
    true_labels = []
    pred_labels = []

    # Get the condition suffix for true labels
    suffix = TRUE_LABEL_MAPPING.get(condition, "")

    # Construct the true key name based on temporal and label_type
    true_key = ""
    if temporal == "current":
        if label_type == "grade":
            true_key = f"CurrentGrade_{suffix}"
        elif label_type == "attribution":
            true_key = f"CurrentAttr_{suffix}"
        elif label_type == "certainty":
            true_key = f"CurrentCert_{suffix}"
    else:  # past
        if label_type == "grade":
            true_key = f"PastMaxGrade_{suffix}"
        elif label_type == "attribution":
            true_key = f"PastAttr_{suffix}"
        elif label_type == "certainty":
            true_key = f"PastCert_{suffix}"

    # Construct the predicted key name
    pred_key = f"{condition}_{temporal}_{label_type}"

    # Extract labels
    for pred in predictions:
        if "true_labels" in pred and "predicted_labels" in pred:
            true_val = pred["true_labels"].get(true_key, 0)
            pred_val = pred["predicted_labels"].get(pred_key, 0)

            # Convert to int and handle None/empty values
            true_labels.append(int(true_val) if true_val is not None else 0)
            pred_labels.append(int(pred_val) if pred_val is not None else 0)

    return true_labels, pred_labels


def calculate_all_metrics_from_predictions(predictions: List[Dict]) -> pd.DataFrame:
    """
    Calculate all metrics from raw predictions.

    Returns:
        DataFrame with one row containing all calculated metrics
    """
    metrics_dict: Dict[str, Any] = {}

    # Process each condition
    for condition in ALLOWED_CONDITIONS:
        # Process each temporal and type combination
        for temporal in ["current", "past"]:
            for label_type in ["grade", "attribution", "certainty"]:
                # Extract labels
                y_true, y_pred = extract_labels_from_predictions(
                    predictions, condition, temporal, label_type
                )

                if not y_true or not y_pred:
                    logger.warning(
                        f"No labels found for {condition}_{temporal}_{label_type}"
                    )
                    continue

                # Calculate metrics
                metrics = calculate_metrics_for_type(y_true, y_pred, label_type)

                # Log binary metrics for grade current to debug
                if (
                    condition == "pneumonitis"
                    and temporal == "current"
                    and label_type == "grade"
                ):
                    cm = metrics["multi_class"]["confusion_matrix"]  # type: np.ndarray
                    binary_cm = convert_to_binary_confusion_matrix(cm)
                    logger.info(
                        f"Pneumonitis Current Grade - Multi-class CM shape: {cm.shape}"
                    )
                    logger.info(f"Pneumonitis Current Grade - Binary CM:\n{binary_cm}")
                    logger.info(
                        f"Binary metrics: Precision={metrics['binary_precision']:.3f}, "
                        f"Recall={metrics['binary_recall']:.3f}, F1={metrics['binary_f1']:.3f}"
                    )

                # Store in dictionary with appropriate column names
                prefix = f"{condition}_{label_type}_{temporal}"
                metrics_dict[f"{prefix}_macro_f1"] = metrics["macro_f1"]
                metrics_dict[f"{prefix}_binary_f1"] = metrics["binary_f1"]
                metrics_dict[f"{prefix}_binary_macro_f1"] = metrics[
                    "binary_f1"
                ]  # Same as binary_f1
                metrics_dict[f"{prefix}_binary_precision"] = metrics["binary_precision"]
                metrics_dict[f"{prefix}_binary_recall"] = metrics["binary_recall"]

                # Store confusion matrix for later use
                metrics_dict[f"{prefix}_confusion_matrix"] = metrics["multi_class"][
                    "confusion_matrix"
                ]  # type: np.ndarray

    # Create DataFrame with one row
    df = pd.DataFrame([metrics_dict])
    return df


def extract_confusion_matrices_from_metrics(
    df: pd.DataFrame,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract confusion matrices from the metrics DataFrame.

    Returns:
        Nested dictionary: {condition -> {attribute_temporality -> confusion_matrix}}
    """
    all_matrices: Dict[str, Dict[str, np.ndarray]] = {}

    for condition in ALLOWED_CONDITIONS:
        all_matrices[condition] = {}

        for temporal in ["current", "past"]:
            for label_type in ["grade", "attribution", "certainty"]:
                col_name = f"{condition}_{label_type}_{temporal}_confusion_matrix"

                if col_name in df.columns and df[col_name].iloc[0] is not None:
                    # Store with the expected key format
                    key = f"{label_type}_{temporal}"
                    all_matrices[condition][key] = df[col_name].iloc[0]

    return all_matrices


def round_values(value):
    """Format numeric values with 2 decimal places."""
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return value


def create_main_table(df: pd.DataFrame, conditions: List[str]) -> str:
    """Create the main table with current grade metrics for all conditions."""
    # Define metrics for main table
    metrics = [
        ("Current Grade Multi-class Macro F1", "_grade_current_macro_f1"),
        ("Current Grade Binary F1", "_grade_current_binary_macro_f1"),
        ("Current Grade Binary Precision", "_grade_current_binary_precision"),
        ("Current Grade Binary Recall", "_grade_current_binary_recall"),
    ]

    # Create header text
    table_header = "# Table 1: Main Results\n\nCurrent irAE Grade Classification Performance Across Conditions\n\n"

    # Create header row
    headers = ["Condition"] + [metric[0] for metric in metrics]
    table = ["| " + " | ".join(headers) + " |"]
    table.append("| " + " | ".join(["---" for _ in headers]) + " |")

    # Collect all values for each metric
    metric_values: Dict[str, List[float]] = {
        col_suffix: [] for _, col_suffix in metrics
    }

    # Add condition rows
    condition_rows = []
    for condition in conditions:
        row = [condition.replace("_", " ").title()]
        for _, col_suffix in metrics:
            column = f"{condition}{col_suffix}"
            if column in df.columns and df[column].iloc[0] is not None:
                value = df[column].iloc[0]
                row.append(round_values(value))
                metric_values[col_suffix].append(value)
            else:
                row.append("N/A")
        condition_rows.append("| " + " | ".join(row) + " |")

    # Calculate overall averages
    overall_row = ["Overall (Average Across Conditions)"]
    for display_name, col_suffix in metrics:
        values = metric_values[col_suffix]
        if values:
            avg_value = sum(values) / len(values)
            overall_row.append(round_values(avg_value))
        else:
            overall_row.append("N/A")

    # Add overall row first, then condition rows
    table.append("| " + " | ".join(overall_row) + " |")
    table.extend(condition_rows)

    # Add footer text
    table_footer = "\n\n**Notes:**\n"
    table_footer += f"- Current Grade Multi-class Macro F1: Macro-averaged F1 score across all {GRADE_CLASSES} grade classes (0-{GRADE_CLASSES-1}) for current conditions. For classes with no examples, F1 is set to {DEFAULT_F1}.\n"
    table_footer += "- Current Grade Binary F1/Precision/Recall: Macro-averaged metrics for binary classification (Grade 0 vs Grade 1+) for current conditions. These are the average of metrics for both class 0 and class 1.\n"
    table_footer += "- Overall (Average Across Conditions): Simple unweighted average of each metric across all 6 conditions (Thyroiditis, Hepatitis, Colitis, Pneumonitis, Myocarditis, Dermatitis).\n"

    return table_header + "\n".join(table) + table_footer


def create_appendix_table1(df: pd.DataFrame, conditions: List[str]) -> str:
    """Create appendix table 1 with macro F1 for current/past grade/cert/attr."""
    # Define metrics with clearer names
    metrics = [
        ("Current Grade Multi-class Macro F1", "_grade_current_macro_f1"),
        ("Current Grade Binary F1", "_grade_current_binary_macro_f1"),
        ("Past Grade Multi-class Macro F1", "_grade_past_macro_f1"),
        ("Past Grade Binary F1", "_grade_past_binary_macro_f1"),
        ("Current Attribution Binary F1", "_attribution_current_macro_f1"),
        (
            "Current Attribution Binary F1 (Same)",
            "_attribution_current_binary_macro_f1",
        ),
        ("Past Attribution Binary F1", "_attribution_past_macro_f1"),
        ("Past Attribution Binary F1 (Same)", "_attribution_past_binary_macro_f1"),
        ("Current Certainty Multi-class Macro F1", "_certainty_current_macro_f1"),
        ("Current Certainty Binary F1", "_certainty_current_binary_macro_f1"),
        ("Past Certainty Multi-class Macro F1", "_certainty_past_macro_f1"),
        ("Past Certainty Binary F1", "_certainty_past_binary_macro_f1"),
    ]

    # Create header text
    table_header = "# Appendix Table 1: Comprehensive Metric Breakdown\n\nF1 Scores Across All Classification Categories and Temporalities\n\n"

    # Create header row
    headers = ["Condition"] + [metric[0] for metric in metrics]
    table = ["| " + " | ".join(headers) + " |"]
    table.append("| " + " | ".join(["---" for _ in headers]) + " |")

    # Collect all values for each metric
    metric_values: Dict[str, List[float]] = {
        col_suffix: [] for _, col_suffix in metrics
    }

    # Add condition rows
    for condition in conditions:
        row = [condition.replace("_", " ").title()]
        for _, col_suffix in metrics:
            column = f"{condition}{col_suffix}"
            if column in df.columns and df[column].iloc[0] is not None:
                value = df[column].iloc[0]
                row.append(round_values(value))
                metric_values[col_suffix].append(value)
            else:
                row.append("N/A")
        table.append("| " + " | ".join(row) + " |")

    # Calculate overall averages
    overall_row = ["Overall (Average)"]
    for display_name, col_suffix in metrics:
        values = metric_values[col_suffix]
        if values:
            avg_value = sum(values) / len(values)
            overall_row.append(round_values(avg_value))
        else:
            overall_row.append("N/A")

    # Add overall row at the end
    table.append("| " + " | ".join(overall_row) + " |")

    # Add footer text
    table_footer = "\n\n**Notes:**\n"
    table_footer += f"- Grade Multi-class: Severity classification on full {GRADE_CLASSES}-class scale (0-{GRADE_CLASSES-1})\n"
    table_footer += f"- Grade Binary: Binary classification (Grade 0 vs Grade 1+) with macro-averaged metrics (average of F1 for both classes)\n"
    table_footer += f"- Attribution: Drug causality assessment ({ATTR_CLASSES} classes: 0=Not related, 1=Related). Note: Attribution is inherently binary, so multi-class and binary metrics are the same.\n"
    table_footer += f"- Certainty Multi-class: Diagnostic confidence on full {CERT_CLASSES}-class scale (0-{CERT_CLASSES-1})\n"
    table_footer += f"- Certainty Binary: Binary classification (Certainty 0 vs Certainty 1+) with macro-averaged metrics\n"
    table_footer += "- Current: Present at the time of assessment\n"
    table_footer += "- Past: Historical occurrence (maximum grade recorded)\n"
    table_footer += f"- Multi-class Macro F1: Macro-averaged F1 score across all classes. For classes with no examples, F1 is set to {DEFAULT_F1}\n"
    table_footer += "- Binary F1: Macro-averaged F1 score for binary classification (average of F1 for both class 0 and class 1)\n"
    table_footer += (
        "- Overall (Average): Simple unweighted average across all 6 conditions\n"
    )

    return table_header + "\n".join(table) + table_footer


def analyze_individual_jsons(individual_jsons_dir: str) -> Dict[str, Dict[int, int]]:
    """Analyzes individual JSON files and returns aggregated counts."""
    json_files = glob.glob(os.path.join(individual_jsons_dir, "*.json"))
    field_counts: Dict[str, Dict[int, int]] = {}

    # Initialize counts for each field
    for field in FIELDS:
        field_counts[field] = {}

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # Count occurrences of each value for each field
            labels = data.get("labels", {})
            for field in FIELDS:
                value = labels.get(field, 0)
                if isinstance(value, str) and value.strip() == "":
                    value = 0
                value = int(value) if value is not None else 0
                field_counts[field][value] = field_counts[field].get(value, 0) + 1

        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")

    return field_counts


def generate_label_count_tables(dataset: str = "irae") -> None:
    """Generate detailed label count tables for test set."""
    logger.info(f"Generating detailed label count tables for {dataset} test set")

    # Define path for irae test set
    test_dir = "data/rwd/irae_test_set_resolved"

    # Get counts for test set
    if os.path.exists(test_dir):
        test_counts = analyze_individual_jsons(test_dir)
        logger.info(f"Analyzed {test_dir}, found data for {len(test_counts)} fields")
    else:
        logger.error(f"Test directory not found: {test_dir}")
        return

    # Categories for appendix tables 2a-f
    categories = [
        ("Grade", "Current", "2a"),
        ("Grade", "Past", "2b"),
        ("Attribution", "Current", "2c"),
        ("Attribution", "Past", "2d"),
        ("Certainty", "Current", "2e"),
        ("Certainty", "Past", "2f"),
    ]

    # For each category, create a detailed table
    for field_type, temporality, table_id in categories:
        logger.info(f"Creating detailed table for {temporality} {field_type}")

        # Create header text
        table_header = (
            f"# Appendix Table {table_id}: {temporality} {field_type} Support\n\n"
        )
        table_header += (
            f"Distribution of {temporality} {field_type} Values in the Test Set\n\n"
        )

        # Determine the maximum class value
        max_class = 0
        if field_type == "Grade":
            max_class = GRADE_CLASSES - 1
        elif field_type == "Attribution":
            max_class = ATTR_CLASSES - 1
        elif field_type == "Certainty":
            max_class = CERT_CLASSES - 1

        # Headers for the table
        headers = ["Condition"]
        for i in range(max_class + 1):
            headers.append(f"Value {i}")
        headers.append("Total")

        table = [table_header]
        table.append("| " + " | ".join(headers) + " |")
        table.append("| " + " | ".join(["---" for _ in headers]) + " |")

        # Add rows for each condition
        for condition_name, suffixes in FIELD_GROUPS.items():
            for suffix in suffixes:
                # Determine the field name
                field = ""
                if temporality == "Current":
                    if field_type == "Grade":
                        field = f"Current{field_type}_{suffix}"
                    elif field_type == "Attribution":
                        field = f"CurrentAttr_{suffix}"
                    elif field_type == "Certainty":
                        field = f"CurrentCert_{suffix}"
                else:  # Past
                    if field_type == "Grade":
                        field = f"PastMax{field_type}_{suffix}"
                    elif field_type == "Attribution":
                        field = f"PastAttr_{suffix}"
                    elif field_type == "Certainty":
                        field = f"PastCert_{suffix}"

                # Get counts for this field
                field_counts = test_counts.get(field, {})

                # Calculate totals
                total_count = sum(field_counts.values())

                # Add row to table
                row = [condition_name]
                for value in range(max_class + 1):
                    row.append(str(field_counts.get(value, 0)))
                row.append(str(total_count))

                table.append("| " + " | ".join(row) + " |")

                # Only include each condition once
                break

        # Add footer with value descriptions
        footer = "\n\n**Notes:**\n"

        if field_type == "Grade":
            value_desc = [
                "No condition",
                "Mild",
                "Moderate",
                "Severe",
                "Life-threatening",
                "Death",
            ]
            for i in range(min(GRADE_CLASSES, len(value_desc))):
                footer += f"- Value {i}: {value_desc[i]}\n"
        elif field_type == "Attribution":
            value_desc = ["Not related", "Related"]
            for i in range(min(ATTR_CLASSES, len(value_desc))):
                footer += f"- Value {i}: {value_desc[i]}\n"
        elif field_type == "Certainty":
            value_desc = [
                "Not present",
                "Low certainty",
                "Moderate certainty",
                "High certainty",
                "Absolute certainty",
            ]
            for i in range(min(CERT_CLASSES, len(value_desc))):
                footer += f"- Value {i}: {value_desc[i]}\n"

        footer += f"- Total: Total number of examples in the test set for {temporality} {field_type}\n"

        table.append(footer)

        # Save the table
        save_markdown("\n".join(table), f"results/appendix/appendix_table{table_id}.md")


def generate_comprehensive_heatmaps(all_matrices: Dict, output_dir: str) -> None:
    """Generate comprehensive heatmaps for each condition."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Generating comprehensive confusion matrix heatmaps in {output_dir}")

    # Define the grid layout
    col_labels = ["Current", "Past"]
    row_labels = ["Grade", "Binary Detection", "Attribution", "Certainty"]

    # Define attribute-temporality keys
    grid_keys = [
        [("grade_current", "grade_past")],
        [("binary_grade_current", "binary_grade_past")],
        [("attribution_current", "attribution_past")],
        [("certainty_current", "certainty_past")],
    ]

    # Process each condition
    for condition, matrices in all_matrices.items():
        display_name = CONDITION_MAPPING.get(condition, condition.title())

        if not matrices:
            logger.warning(f"No matrices found for {condition}, skipping visualization")
            continue

        # Create figure with subplots
        fig, axs = plt.subplots(4, 2, figsize=(8.27, 11.69))
        fig.suptitle(f"Confusion Matrices - {display_name}", fontsize=16)

        # Add descriptive subtitle
        fig.text(
            0.5,
            0.92,
            "Model Prediction vs. True Labels across Classification Types",
            ha="center",
            fontsize=12,
            style="italic",
        )

        # Improved spacing
        plt.subplots_adjust(
            wspace=0.3, hspace=0.6, left=0.1, right=0.9, top=0.88, bottom=0.08
        )

        # Fill in each subplot
        for row_idx, row in enumerate(grid_keys):
            for pair_idx, key_pair in enumerate(row):
                for col_idx, key in enumerate(key_pair):
                    ax = axs[row_idx, col_idx]

                    # Special handling for binary detection matrices
                    if "binary_grade" in key:
                        temporality = key.split("_")[-1]
                        grade_key = f"grade_{temporality}"

                        if grade_key in matrices:
                            cm_array = matrices[grade_key]
                            binary_cm = convert_to_binary_confusion_matrix(cm_array)

                            sns.heatmap(
                                binary_cm,
                                annot=True,
                                fmt="d",
                                cmap="Blues",
                                cbar=False,
                                square=True,
                                xticklabels=["Grade 0", "Grade 1+"],
                                yticklabels=["Grade 0", "Grade 1+"],
                                ax=ax,
                            )

                            ax.set_title(f"{col_labels[col_idx]} Binary Detection")
                            ax.set_xlabel("Predicted Label")
                            ax.set_ylabel("True Label")
                        else:
                            ax.text(
                                0.5,
                                0.5,
                                "No Data Available",
                                horizontalalignment="center",
                                verticalalignment="center",
                                transform=ax.transAxes,
                            )
                            ax.set_title(
                                f"{col_labels[col_idx]} Binary Detection (N/A)"
                            )
                            ax.axis("off")

                    # Regular handling for other columns
                    elif key in matrices:
                        cm_array = matrices[key]
                        attr_type = key.split("_")[0]

                        # Set tick labels based on attribute type
                        if attr_type == "grade":
                            tick_labels = [f"{i}" for i in range(GRADE_CLASSES)]
                        elif attr_type == "attribution":
                            tick_labels = ["None", "IO related"]
                        elif attr_type == "certainty":
                            tick_labels = [
                                "None",
                                "Unlikely",
                                "Possible",
                                "Likely",
                                "Certain",
                            ]
                        else:
                            tick_labels = [str(i) for i in range(cm_array.shape[1])]

                        # Generate heatmap
                        sns.heatmap(
                            cm_array,
                            annot=True,
                            fmt="d",
                            cmap="Blues",
                            cbar=False,
                            square=True,
                            xticklabels=tick_labels[: cm_array.shape[1]],
                            yticklabels=tick_labels[: cm_array.shape[0]],
                            ax=ax,
                        )

                        ax.set_title(f"{col_labels[col_idx]} {row_labels[row_idx]}")
                        ax.set_xlabel("Predicted Label")
                        ax.set_ylabel("True Label")

                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "No Data Available",
                            horizontalalignment="center",
                            verticalalignment="center",
                            transform=ax.transAxes,
                        )
                        ax.set_title(
                            f"{col_labels[col_idx]} {row_labels[row_idx]} (N/A)"
                        )
                        ax.axis("off")

        # Add timestamp
        fig.text(
            0.5,
            0.01,
            f"Generated: {datetime.now().strftime('%Y-%m-%d')}",
            ha="center",
            fontsize=8,
            color="gray",
        )

        # Save as PDF
        output_file = os.path.join(output_dir, f"confusion_matrices_{condition}.pdf")
        plt.savefig(output_file, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)

        logger.info(f"Saved confusion matrices for {condition} to {output_file}")


def save_markdown(content: str, filepath: str) -> None:
    """Save markdown content to a file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)
    logger.info(f"Saved: {filepath}")


def extract_and_verify_metrics(
    all_matrices: Dict, conditions: List[str]
) -> Dict[str, Dict[str, float]]:
    """Extract and verify binary metrics directly from confusion matrices."""
    verified_metrics = {}

    for condition in conditions:
        # Check if this condition exists in matrices
        if condition not in all_matrices:
            continue

        # Get grade_current matrix
        if "grade_current" in all_matrices[condition]:
            cm_array = all_matrices[condition]["grade_current"]

            # Print the full confusion matrix
            logger.info(f"\n{condition.title()} Grade Current Confusion Matrix:")
            logger.info(f"Shape: {cm_array.shape}")
            logger.info(f"Matrix:\n{cm_array}")

            # Convert to binary confusion matrix and print
            binary_cm = convert_to_binary_confusion_matrix(cm_array)
            logger.info(f"Binary confusion matrix:\n{binary_cm}")
            logger.info(
                f"TN={binary_cm[0,0]}, FP={binary_cm[0,1]}, "
                f"FN={binary_cm[1,0]}, TP={binary_cm[1,1]}"
            )

            # Calculate binary metrics
            binary_metrics = calculate_binary_metrics_from_matrix(cm_array)
            logger.info(
                f"Calculated metrics: Precision={binary_metrics['binary_precision']:.4f}, "
                f"Recall={binary_metrics['binary_recall']:.4f}, "
                f"F1={binary_metrics['binary_f1']:.4f}"
            )

            # Store metrics for this condition
            verified_metrics[condition] = binary_metrics

    return verified_metrics


def create_appendix_3d_processing_summary(processing_results: List[Dict]) -> str:
    """Create appendix table 3d showing which model/variant combinations were processed."""
    table_header = "# Appendix Table 3d: Model/Variant Processing Summary\n\n"
    table_header += "Status of All Model and Variant Combinations\n\n"

    headers = ["Model", "Variant", "Status", "Predictions Count", "File Path"]

    table = [table_header]
    table.append("| " + " | ".join(headers) + " |")
    table.append("| " + " | ".join(["---" for _ in headers]) + " |")

    # Sort by model then variant
    sorted_results = sorted(
        processing_results, key=lambda x: (x["model"], x["variant"])
    )

    # Add rows
    for result in sorted_results:
        row = [
            result["model"],
            result["variant"].replace("variant_", ""),
            result["status"],
            str(result.get("predictions_count", "N/A")),
            result.get("file_path", "N/A"),
        ]
        table.append("| " + " | ".join(row) + " |")

    # Add summary
    successful = sum(1 for r in processing_results if r["status"] == "Success")
    failed = sum(1 for r in processing_results if r["status"] == "Failed")

    footer = f"\n\n**Summary:**\n"
    footer += f"- Total combinations attempted: {len(processing_results)}\n"
    footer += f"- Successful: {successful}\n"
    footer += f"- Failed: {failed}\n"

    return "\n".join(table) + footer


def create_appendix_grade_f1_breakdown(predictions: List[Dict]) -> str:
    """Create appendix table showing F1 scores for each grade level and various binary thresholds."""
    table_header = "# Appendix Table 4: Per-Grade F1 Score Breakdown for Current Grade Classification\n\n"
    table_header += "F1 Scores for Individual Grade Levels and Binary Classification at Different Thresholds\n\n"

    # Calculate per-grade F1 scores for each condition
    grade_f1_data = {}

    for condition in ALLOWED_CONDITIONS:
        # Extract labels for current grade
        y_true, y_pred = extract_labels_from_predictions(
            predictions, condition, "current", "grade"
        )

        if not y_true or not y_pred:
            continue

        # Calculate per-class F1 scores
        from sklearn.metrics import f1_score

        f1_per_class = f1_score(
            y_true,
            y_pred,
            labels=list(range(GRADE_CLASSES)),
            average=None,
            zero_division=DEFAULT_F1,
        )

        # Store per-grade F1 scores
        grade_f1_data[condition] = {
            "per_grade_f1": f1_per_class,
            "y_true": y_true,
            "y_pred": y_pred,
        }

    # Part 1: Individual Grade F1 Scores
    headers = (
        ["Condition"]
        + [f"Grade {i} F1" for i in range(GRADE_CLASSES)]
        + ["Macro F1 (All Grades)"]
    )

    table = [table_header]
    table.append("**Part 1: F1 Scores for Individual Grade Levels**\n")
    table.append("| " + " | ".join(headers) + " |")
    table.append("| " + " | ".join(["---" for _ in headers]) + " |")

    # Collect values for overall calculation
    all_grade_f1s: List[List[float]] = [[] for _ in range(GRADE_CLASSES)]
    all_macro_f1s = []

    for condition in ALLOWED_CONDITIONS:
        row = [condition.title()]

        if condition in grade_f1_data:
            per_grade_f1 = grade_f1_data[condition]["per_grade_f1"]

            # Add F1 for each grade
            for i in range(GRADE_CLASSES):
                f1_val = per_grade_f1[i] if i < len(per_grade_f1) else DEFAULT_F1
                row.append(round_values(f1_val))
                all_grade_f1s[i].append(f1_val)

            # Add macro F1
            macro_f1 = np.mean(per_grade_f1)
            row.append(round_values(macro_f1))
            all_macro_f1s.append(macro_f1)
        else:
            row.extend(["N/A"] * (GRADE_CLASSES + 1))

        table.append("| " + " | ".join(row) + " |")

    # Add overall row
    overall_row = ["Overall (Average)"]
    for i in range(GRADE_CLASSES):
        if all_grade_f1s[i]:
            overall_row.append(round_values(np.mean(all_grade_f1s[i])))
        else:
            overall_row.append("N/A")

    if all_macro_f1s:
        overall_row.append(round_values(np.mean(all_macro_f1s)))
    else:
        overall_row.append("N/A")

    table.append("| " + " | ".join(overall_row) + " |")

    # Part 2: Binary F1 Scores at Different Thresholds
    table.append("\n**Part 2: Binary F1 Scores at Different Thresholds**\n")

    binary_thresholds = [
        ("0 vs 1-5", lambda x: 0 if x == 0 else 1),
        ("0-1 vs 2-5", lambda x: 0 if x <= 1 else 1),
        ("0-2 vs 3-5", lambda x: 0 if x <= 2 else 1),
        ("0-3 vs 4-5", lambda x: 0 if x <= 3 else 1),
        ("0-4 vs 5", lambda x: 0 if x <= 4 else 1),
    ]

    headers = ["Condition"] + [
        f"{threshold[0]} Macro F1" for threshold in binary_thresholds
    ]
    table.append("| " + " | ".join(headers) + " |")
    table.append("| " + " | ".join(["---" for _ in headers]) + " |")

    # Collect values for overall calculation
    threshold_f1s: List[List[float]] = [[] for _ in binary_thresholds]

    for condition in ALLOWED_CONDITIONS:
        row = [condition.title()]

        if condition in grade_f1_data:
            y_true = grade_f1_data[condition]["y_true"]
            y_pred = grade_f1_data[condition]["y_pred"]

            for idx, (threshold_name, threshold_func) in enumerate(binary_thresholds):
                # Convert to binary using threshold
                y_true_binary = [threshold_func(y) for y in y_true]
                y_pred_binary = [threshold_func(y) for y in y_pred]

                # Calculate macro F1 for this binary classification
                binary_f1 = f1_score(
                    y_true_binary,
                    y_pred_binary,
                    average="macro",
                    zero_division=DEFAULT_F1,
                )
                row.append(round_values(binary_f1))
                threshold_f1s[idx].append(binary_f1)
        else:
            row.extend(["N/A"] * len(binary_thresholds))

        table.append("| " + " | ".join(row) + " |")

    # Add overall row
    overall_row = ["Overall (Average)"]
    for threshold_f1_list in threshold_f1s:
        if threshold_f1_list:
            overall_row.append(round_values(np.mean(threshold_f1_list)))
        else:
            overall_row.append("N/A")

    table.append("| " + " | ".join(overall_row) + " |")

    # Add footer
    footer = "\n\n**Notes:**\n"
    footer += f"- Part 1 shows F1 scores for each individual grade level (0-{GRADE_CLASSES-1})\n"
    footer += f"- Macro F1 (All Grades) is the unweighted average of F1 scores across all {GRADE_CLASSES} grades\n"
    footer += f"- Part 2 shows macro-averaged binary F1 scores at different severity thresholds\n"
    footer += "- For binary classification, macro F1 is the average of F1 scores for both classes (negative and positive)\n"
    footer += f"- All F1 scores use zero_division={DEFAULT_F1} to handle classes with no predictions\n"
    footer += "- Overall (Average) shows the simple unweighted average across all 6 conditions\n"

    return "\n".join(table) + footer


def create_appendix_cross_temporal_analysis(predictions: List[Dict]) -> str:
    """Create appendix table showing performance when considering maximum values across current/past temporality."""
    table_header = "# Appendix Table 5: Cross-Temporal Maximum Analysis\n\n"
    table_header += (
        "Performance when taking maximum values across Current and Past temporalities\n"
    )
    table_header += "This analysis shows whether errors are due to detection issues or temporality assignment\n\n"

    # Store results for each condition and metric type
    temporal_analysis_data: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for condition in ALLOWED_CONDITIONS:
        temporal_analysis_data[condition] = {}

        for label_type in ["grade", "attribution", "certainty"]:
            # Extract current and past labels
            y_true_current, y_pred_current = extract_labels_from_predictions(
                predictions, condition, "current", label_type
            )
            y_true_past, y_pred_past = extract_labels_from_predictions(
                predictions, condition, "past", label_type
            )

            if not y_true_current or not y_true_past:
                continue

            # Compute maximum across temporalities
            y_true_max = [
                max(curr, past) for curr, past in zip(y_true_current, y_true_past)
            ]
            y_pred_max = [
                max(curr, past) for curr, past in zip(y_pred_current, y_pred_past)
            ]

            # Calculate metrics on max values
            metrics_max = calculate_metrics_for_type(y_true_max, y_pred_max, label_type)

            # Also calculate metrics for current and past separately for comparison
            metrics_current = calculate_metrics_for_type(
                y_true_current, y_pred_current, label_type
            )
            metrics_past = calculate_metrics_for_type(
                y_true_past, y_pred_past, label_type
            )

            temporal_analysis_data[condition][label_type] = {
                "current": metrics_current,
                "past": metrics_past,
                "max": metrics_max,
                "y_true_max": y_true_max,
                "y_pred_max": y_pred_max,
            }

    # Part 1: Grade Analysis
    table = [table_header]
    table.append("**Part 1: Grade Cross-Temporal Analysis**\n")

    headers = [
        "Condition",
        "Current Only Binary F1",
        "Past Only Binary F1",
        "Max(Current,Past) Binary F1",
        "Current Only Multi-class F1",
        "Past Only Multi-class F1",
        "Max(Current,Past) Multi-class F1",
    ]

    table.append("| " + " | ".join(headers) + " |")
    table.append("| " + " | ".join(["---" for _ in headers]) + " |")

    # Collect values for overall calculation
    current_binary_f1s = []
    past_binary_f1s = []
    max_binary_f1s = []
    current_multi_f1s = []
    past_multi_f1s = []
    max_multi_f1s = []

    for condition in ALLOWED_CONDITIONS:
        row = [condition.title()]

        if (
            condition in temporal_analysis_data
            and "grade" in temporal_analysis_data[condition]
        ):
            data = temporal_analysis_data[condition]["grade"]

            # Binary F1s
            current_binary_f1 = data["current"]["binary_f1"]
            past_binary_f1 = data["past"]["binary_f1"]
            max_binary_f1 = data["max"]["binary_f1"]

            row.extend(
                [
                    round_values(current_binary_f1),
                    round_values(past_binary_f1),
                    round_values(max_binary_f1),
                ]
            )

            current_binary_f1s.append(current_binary_f1)
            past_binary_f1s.append(past_binary_f1)
            max_binary_f1s.append(max_binary_f1)

            # Multi-class F1s
            current_multi_f1 = data["current"]["macro_f1"]
            past_multi_f1 = data["past"]["macro_f1"]
            max_multi_f1 = data["max"]["macro_f1"]

            row.extend(
                [
                    round_values(current_multi_f1),
                    round_values(past_multi_f1),
                    round_values(max_multi_f1),
                ]
            )

            current_multi_f1s.append(current_multi_f1)
            past_multi_f1s.append(past_multi_f1)
            max_multi_f1s.append(max_multi_f1)
        else:
            row.extend(["N/A"] * 6)

        table.append("| " + " | ".join(row) + " |")

    # Add overall row
    overall_row = ["Overall (Average)"]
    overall_row.extend(
        [
            round_values(np.mean(current_binary_f1s)) if current_binary_f1s else "N/A",
            round_values(np.mean(past_binary_f1s)) if past_binary_f1s else "N/A",
            round_values(np.mean(max_binary_f1s)) if max_binary_f1s else "N/A",
            round_values(np.mean(current_multi_f1s)) if current_multi_f1s else "N/A",
            round_values(np.mean(past_multi_f1s)) if past_multi_f1s else "N/A",
            round_values(np.mean(max_multi_f1s)) if max_multi_f1s else "N/A",
        ]
    )
    table.append("| " + " | ".join(overall_row) + " |")

    # Part 2: Attribution and Certainty Analysis
    table.append("\n**Part 2: Attribution and Certainty Cross-Temporal Analysis**\n")

    headers = [
        "Condition",
        "Metric Type",
        "Current Only Binary F1",
        "Past Only Binary F1",
        "Max(Current,Past) Binary F1",
        "Temporal Assignment Error Rate",
    ]

    table.append("| " + " | ".join(headers) + " |")
    table.append("| " + " | ".join(["---" for _ in headers]) + " |")

    for condition in ALLOWED_CONDITIONS:
        for label_type in ["attribution", "certainty"]:
            row = [condition.title(), label_type.title()]

            if (
                condition in temporal_analysis_data
                and label_type in temporal_analysis_data[condition]
            ):
                data = temporal_analysis_data[condition][label_type]

                current_f1 = data["current"]["binary_f1"]
                past_f1 = data["past"]["binary_f1"]
                max_f1 = data["max"]["binary_f1"]

                row.extend(
                    [
                        round_values(current_f1),
                        round_values(past_f1),
                        round_values(max_f1),
                    ]
                )

                # Calculate temporal assignment error rate
                # This is the proportion of cases where max detection is correct but temporal assignment is wrong
                y_true_max = data["y_true_max"]
                y_pred_max = data["y_pred_max"]

                # Count cases where max is correctly detected (both > 0 or both == 0)
                max_correct = sum(
                    1
                    for t, p in zip(y_true_max, y_pred_max)
                    if (t > 0 and p > 0) or (t == 0 and p == 0)
                )

                # Improvement from max over better of current/past
                better_individual = max(current_f1, past_f1)
                improvement = max_f1 - better_individual

                row.append(
                    f"+{improvement:.3f}" if improvement >= 0 else f"{improvement:.3f}"
                )
            else:
                row.extend(["N/A"] * 4)

            table.append("| " + " | ".join(row) + " |")

    # Part 3: Confusion Matrix Analysis for Grade Max Values
    table.append(
        "\n**Part 3: Example Confusion Matrices for Grade Max(Current,Past) - Selected Conditions**\n"
    )

    # Show confusion matrices for 2 example conditions
    example_conditions = ["pneumonitis", "hepatitis"]

    for condition in example_conditions:
        if (
            condition in temporal_analysis_data
            and "grade" in temporal_analysis_data[condition]
        ):
            data = temporal_analysis_data[condition]["grade"]
            cm = data["max"]["multi_class"]["confusion_matrix"]

            table.append(
                f"\n{condition.title()} - Grade Max(Current,Past) Confusion Matrix:"
            )
            table.append("```")
            table.append(
                "True\\Pred " + " ".join([f"{i:3d}" for i in range(cm.shape[1])])
            )
            for i in range(cm.shape[0]):
                row_str = f"Grade {i}:  " + " ".join(
                    [f"{cm[i,j]:3d}" for j in range(cm.shape[1])]
                )
                table.append(row_str)
            table.append("```")

    # Add footer
    footer = "\n\n**Notes:**\n"
    footer += "- This analysis takes the maximum value between Current and Past for both true and predicted labels\n"
    footer += "- If Max(Current,Past) F1 ≈ max(Current F1, Past F1), errors are mainly detection issues\n"
    footer += "- If Max(Current,Past) F1 >> max(Current F1, Past F1), errors are mainly temporal assignment issues\n"
    footer += "- Binary F1: Macro-averaged F1 for binary classification (0 vs 1+)\n"
    footer += "- Multi-class F1: Macro-averaged F1 across all classes\n"
    footer += "- Temporal Assignment Error Rate: Improvement in F1 when using max values (positive means temporal confusion exists)\n"
    footer += "- Example: If true current=3, past=0 and predicted current=0, past=3, then max detection is correct but temporal assignment is wrong\n"

    return "\n".join(table) + footer


def process_all_model_variants() -> Tuple[pd.DataFrame, List[Dict]]:
    """Process all model and variant combinations and return comprehensive DataFrame and processing results."""
    all_results = []
    processing_results = []

    for model in MODELS:
        for variant in VARIANTS:
            result_info = {
                "model": model,
                "variant": variant,
                "status": "Failed",
                "predictions_count": 0,
                "file_path": "N/A",
            }

            try:
                # Find predictions file
                latest_file = find_latest_predictions_file(model, variant)
                # Use the string path rather than the Path object for relative_to
                result_info["file_path"] = str(latest_file)

                logger.info(f"Processing {model}/{variant}: {latest_file}")

                # Load predictions
                predictions = load_predictions(latest_file)
                result_info["predictions_count"] = len(predictions)

                # Calculate metrics
                df = calculate_all_metrics_from_predictions(predictions)

                # Add model and variant info
                df["model"] = model
                df["variant"] = variant
                df["model_variant"] = f"{model}_{variant}"

                # Calculate overall metrics across all conditions
                overall_metrics = calculate_overall_metrics(df, ALLOWED_CONDITIONS)
                df["overall_macro_f1"] = overall_metrics["overall_macro_f1"]
                df["overall_binary_f1"] = overall_metrics["overall_binary_f1"]
                df["overall_binary_precision"] = overall_metrics[
                    "overall_binary_precision"
                ]
                df["overall_binary_recall"] = overall_metrics["overall_binary_recall"]

                all_results.append(df)
                result_info["status"] = "Success"

            except Exception as e:
                logger.warning(f"Error processing {model}/{variant}: {e}")
                result_info["error"] = str(e)

            processing_results.append(result_info)

    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        return combined_df, processing_results
    else:
        return pd.DataFrame(), processing_results


def calculate_overall_metrics(
    df: pd.DataFrame, conditions: List[str]
) -> Dict[str, float]:
    """Calculate overall metrics across all conditions."""
    macro_f1_values = []
    binary_f1_values = []
    binary_precision_values = []
    binary_recall_values = []

    for condition in conditions:
        # Collect current grade metrics
        macro_col = f"{condition}_grade_current_macro_f1"
        binary_f1_col = f"{condition}_grade_current_binary_f1"
        binary_prec_col = f"{condition}_grade_current_binary_precision"
        binary_rec_col = f"{condition}_grade_current_binary_recall"

        if macro_col in df.columns and df[macro_col].iloc[0] is not None:
            macro_f1_values.append(df[macro_col].iloc[0])
        if binary_f1_col in df.columns and df[binary_f1_col].iloc[0] is not None:
            binary_f1_values.append(df[binary_f1_col].iloc[0])
        if binary_prec_col in df.columns and df[binary_prec_col].iloc[0] is not None:
            binary_precision_values.append(df[binary_prec_col].iloc[0])
        if binary_rec_col in df.columns and df[binary_rec_col].iloc[0] is not None:
            binary_recall_values.append(df[binary_rec_col].iloc[0])

    return {
        "overall_macro_f1": np.mean(macro_f1_values) if macro_f1_values else 0,
        "overall_binary_f1": np.mean(binary_f1_values) if binary_f1_values else 0,
        "overall_binary_precision": (
            np.mean(binary_precision_values) if binary_precision_values else 0
        ),
        "overall_binary_recall": (
            np.mean(binary_recall_values) if binary_recall_values else 0
        ),
    }


def create_appendix_3a_binary_performance(combined_df: pd.DataFrame) -> str:
    """Create appendix table 3a showing binary classification performance across models/variants."""
    table_header = "# Appendix Table 3a: Binary Classification Performance Across Model/Variant Combinations\n\n"
    table_header += "Binary F1, Precision, and Recall for Current Grade Binary Classification (Grade 0 vs Grade 1+)\n\n"

    # Create headers
    headers = [
        "Graph Type",
        "Model",
        "Variant",
        "Overall Binary F1 (Avg)",
        "Overall Binary Precision (Avg)",
        "Overall Binary Recall (Avg)",
    ]
    for condition in ALLOWED_CONDITIONS:
        headers.extend(
            [
                f"{condition.title()} Binary F1",
                f"{condition.title()} Binary Precision",
                f"{condition.title()} Binary Recall",
            ]
        )

    table = [table_header]
    table.append("| " + " | ".join(headers) + " |")
    table.append("| " + " | ".join(["---" for _ in headers]) + " |")

    # Sort by overall binary F1
    combined_df = combined_df.sort_values("overall_binary_f1", ascending=False)

    # Add rows
    for _, row in combined_df.iterrows():
        row_data = [
            row.get("graph_type", "unknown"),
            row["model"],
            row["variant"].replace("variant_", ""),
            round_values(row["overall_binary_f1"]),
            round_values(row["overall_binary_precision"]),
            round_values(row["overall_binary_recall"]),
        ]

        # Add condition-specific metrics
        for condition in ALLOWED_CONDITIONS:
            f1_col = f"{condition}_grade_current_binary_f1"
            prec_col = f"{condition}_grade_current_binary_precision"
            rec_col = f"{condition}_grade_current_binary_recall"

            f1_val = row[f1_col] if f1_col in row and pd.notna(row[f1_col]) else "N/A"
            prec_val = (
                row[prec_col] if prec_col in row and pd.notna(row[prec_col]) else "N/A"
            )
            rec_val = (
                row[rec_col] if rec_col in row and pd.notna(row[rec_col]) else "N/A"
            )

            row_data.extend(
                [
                    round_values(f1_val) if f1_val != "N/A" else "N/A",
                    round_values(prec_val) if prec_val != "N/A" else "N/A",
                    round_values(rec_val) if rec_val != "N/A" else "N/A",
                ]
            )

        table.append("| " + " | ".join(row_data) + " |")

    # Add footer
    footer = "\n\n**Notes:**\n"
    footer += "- Binary classification: Grade 0 (no condition) vs Grade 1+ (condition present)\n"
    footer += "- All binary F1/Precision/Recall values shown are macro-averaged (average of metrics for both class 0 and class 1)\n"
    footer += "- Overall Binary F1/Precision/Recall (Avg): Simple unweighted average of the respective metric across all 6 conditions (Thyroiditis, Hepatitis, Colitis, Pneumonitis, Myocarditis, Dermatitis)\n"
    footer += "- Individual condition metrics: Binary classification performance for Current Grade only\n"
    footer += "- Graph Types: openai_agent (main approach), openai_zeroshot (baseline), regex (rule-based baseline, binary only)\n"
    footer += "- Models: 4.1-mini, 4.1-nano (OpenAI agent), 4o-mini, o3-mini (zeroshot), default (regex)\n"
    footer += "- Note: Regex baseline only outputs binary predictions (0 or 1) for presence/absence detection\n"
    footer += "- Variants: default (full system), ablation_no_judge (without judge model), ablation_single (single-stage)\n"
    footer += "- Table sorted by Overall Binary F1 in descending order\n"

    return "\n".join(table) + footer


def create_appendix_3b_macro_f1_performance(combined_df: pd.DataFrame) -> str:
    """Create appendix table 3b showing macro F1 performance across models/variants."""
    table_header = "# Appendix Table 3b: Multi-class Macro F1 Performance Across Model/Variant Combinations\n\n"
    table_header += "Macro F1 Scores for Current Grade Multi-class Classification (6 classes: 0-5)\n\n"

    # Create headers
    headers = ["Graph Type", "Model", "Variant", "Overall Multi-class Macro F1 (Avg)"]
    for condition in ALLOWED_CONDITIONS:
        headers.append(f"{condition.title()} Multi-class Macro F1")

    table = [table_header]
    table.append("| " + " | ".join(headers) + " |")
    table.append("| " + " | ".join(["---" for _ in headers]) + " |")

    # Sort by overall macro F1
    combined_df = combined_df.sort_values("overall_macro_f1", ascending=False)

    # Add rows
    for _, row in combined_df.iterrows():
        row_data = [
            row.get("graph_type", "unknown"),
            row["model"],
            row["variant"].replace("variant_", ""),
            round_values(row["overall_macro_f1"]),
        ]

        # Add condition-specific metrics
        for condition in ALLOWED_CONDITIONS:
            col = f"{condition}_grade_current_macro_f1"
            val = row[col] if col in row and pd.notna(row[col]) else "N/A"
            row_data.append(round_values(val) if val != "N/A" else "N/A")

        table.append("| " + " | ".join(row_data) + " |")

    # Add footer
    footer = "\n\n**Notes:**\n"
    footer += f"- Multi-class Macro F1: Macro-averaged F1 score across all {GRADE_CLASSES} grade classes (0-5) where each class gets equal weight\n"
    footer += f"- Classes with no true positive examples receive F1 score of {DEFAULT_F1} to avoid undefined values\n"
    footer += "- Overall Multi-class Macro F1 (Avg): Simple unweighted average of Multi-class Macro F1 scores across all 6 conditions\n"
    footer += "- All metrics shown are for Current Grade only (not Past Grade)\n"
    footer += "- Graph Types: openai_agent (main approach), openai_zeroshot (baseline), regex (rule-based baseline)\n"
    footer += "- Note: Regex baseline only outputs binary (0/1) predictions, so its multi-class F1 may not be directly comparable\n"
    footer += "- Table sorted by Overall Multi-class Macro F1 in descending order\n"

    return "\n".join(table) + footer


def create_appendix_3c_comprehensive_metrics(combined_df: pd.DataFrame) -> str:
    """Create appendix table 3c showing all metric types across models/variants."""
    table_header = "# Appendix Table 3c: Comprehensive Performance Metrics\n\n"
    table_header += "All Classification Tasks (Grade, Attribution, Certainty) Across Temporalities - Averaged Across All Conditions\n\n"

    # Create headers
    headers = ["Graph Type", "Model", "Variant"]
    metric_types = [
        ("Current Grade Multi-class Macro F1", "grade_current"),
        ("Current Grade Binary F1", "grade_current"),
        ("Past Grade Multi-class Macro F1", "grade_past"),
        ("Past Grade Binary F1", "grade_past"),
        ("Current Attribution Binary F1", "attribution_current"),
        ("Past Attribution Binary F1", "attribution_past"),
        ("Current Certainty Multi-class Macro F1", "certainty_current"),
        ("Current Certainty Binary F1", "certainty_current"),
        ("Past Certainty Multi-class Macro F1", "certainty_past"),
        ("Past Certainty Binary F1", "certainty_past"),
    ]

    # Add headers based on metric types
    for display_name, metric_key in metric_types:
        headers.append(display_name)

    table = [table_header]
    table.append("| " + " | ".join(headers) + " |")
    table.append("| " + " | ".join(["---" for _ in headers]) + " |")

    # Calculate comprehensive overall score for sorting
    def calculate_comprehensive_score(row):
        scores = []
        for condition in ALLOWED_CONDITIONS:
            # Grade metrics
            for temporal in ["current", "past"]:
                macro_col = f"{condition}_grade_{temporal}_macro_f1"
                binary_col = f"{condition}_grade_{temporal}_binary_f1"
                if macro_col in row and pd.notna(row[macro_col]):
                    scores.append(row[macro_col])
                if binary_col in row and pd.notna(row[binary_col]):
                    scores.append(row[binary_col])

            # Attribution metrics (binary only)
            for temporal in ["current", "past"]:
                col = f"{condition}_attribution_{temporal}_macro_f1"
                if col in row and pd.notna(row[col]):
                    scores.append(row[col])

            # Certainty metrics
            for temporal in ["current", "past"]:
                macro_col = f"{condition}_certainty_{temporal}_macro_f1"
                binary_col = f"{condition}_certainty_{temporal}_binary_f1"
                if macro_col in row and pd.notna(row[macro_col]):
                    scores.append(row[macro_col])
                if binary_col in row and pd.notna(row[binary_col]):
                    scores.append(row[binary_col])

        return np.mean(scores) if scores else 0

    combined_df["comprehensive_score"] = combined_df.apply(
        calculate_comprehensive_score, axis=1
    )
    combined_df = combined_df.sort_values("comprehensive_score", ascending=False)

    # Add rows
    for _, row in combined_df.iterrows():
        row_data = [
            row.get("graph_type", "unknown"),
            row["model"],
            row["variant"].replace("variant_", ""),
        ]

        # Current Grade Multi-class Macro F1
        vals = []
        for condition in ALLOWED_CONDITIONS:
            col = f"{condition}_grade_current_macro_f1"
            if col in row and pd.notna(row[col]):
                vals.append(row[col])
        row_data.append(round_values(np.mean(vals)) if vals else "N/A")

        # Current Grade Binary F1
        vals = []
        for condition in ALLOWED_CONDITIONS:
            col = f"{condition}_grade_current_binary_f1"
            if col in row and pd.notna(row[col]):
                vals.append(row[col])
        row_data.append(round_values(np.mean(vals)) if vals else "N/A")

        # Past Grade Multi-class Macro F1
        vals = []
        for condition in ALLOWED_CONDITIONS:
            col = f"{condition}_grade_past_macro_f1"
            if col in row and pd.notna(row[col]):
                vals.append(row[col])
        row_data.append(round_values(np.mean(vals)) if vals else "N/A")

        # Past Grade Binary F1
        vals = []
        for condition in ALLOWED_CONDITIONS:
            col = f"{condition}_grade_past_binary_f1"
            if col in row and pd.notna(row[col]):
                vals.append(row[col])
        row_data.append(round_values(np.mean(vals)) if vals else "N/A")

        # Current Attribution Binary F1
        vals = []
        for condition in ALLOWED_CONDITIONS:
            col = f"{condition}_attribution_current_macro_f1"
            if col in row and pd.notna(row[col]):
                vals.append(row[col])
        row_data.append(round_values(np.mean(vals)) if vals else "N/A")

        # Past Attribution Binary F1
        vals = []
        for condition in ALLOWED_CONDITIONS:
            col = f"{condition}_attribution_past_macro_f1"
            if col in row and pd.notna(row[col]):
                vals.append(row[col])
        row_data.append(round_values(np.mean(vals)) if vals else "N/A")

        # Current Certainty Multi-class Macro F1
        vals = []
        for condition in ALLOWED_CONDITIONS:
            col = f"{condition}_certainty_current_macro_f1"
            if col in row and pd.notna(row[col]):
                vals.append(row[col])
        row_data.append(round_values(np.mean(vals)) if vals else "N/A")

        # Current Certainty Binary F1
        vals = []
        for condition in ALLOWED_CONDITIONS:
            col = f"{condition}_certainty_current_binary_f1"
            if col in row and pd.notna(row[col]):
                vals.append(row[col])
        row_data.append(round_values(np.mean(vals)) if vals else "N/A")

        # Past Certainty Multi-class Macro F1
        vals = []
        for condition in ALLOWED_CONDITIONS:
            col = f"{condition}_certainty_past_macro_f1"
            if col in row and pd.notna(row[col]):
                vals.append(row[col])
        row_data.append(round_values(np.mean(vals)) if vals else "N/A")

        # Past Certainty Binary F1
        vals = []
        for condition in ALLOWED_CONDITIONS:
            col = f"{condition}_certainty_past_binary_f1"
            if col in row and pd.notna(row[col]):
                vals.append(row[col])
        row_data.append(round_values(np.mean(vals)) if vals else "N/A")

        table.append("| " + " | ".join(row_data) + " |")

    # Add footer
    footer = "\n\n**Notes:**\n"
    footer += "- All values shown are simple unweighted averages across all 6 conditions (Thyroiditis, Hepatitis, Colitis, Pneumonitis, Myocarditis, Dermatitis)\n"
    footer += f"- Grade Multi-class: Severity classification on {GRADE_CLASSES}-class scale (0-5); Binary: Grade 0 vs Grade 1+ with macro-averaged metrics\n"
    footer += f"- Attribution: Drug causality assessment ({ATTR_CLASSES} classes - inherently binary: 0=Not related, 1=Related)\n"
    footer += f"- Certainty Multi-class: Diagnostic confidence on {CERT_CLASSES}-class scale (0-4); Binary: Certainty 0 vs Certainty 1+ with macro-averaged metrics\n"
    footer += "- Binary F1 values are macro-averaged (average of F1 for both class 0 and class 1)\n"
    footer += "- Current: Present conditions at time of assessment\n"
    footer += "- Past: Historical conditions (using maximum recorded grade)\n"
    footer += "- Regex baseline: Only outputs binary (0/1) values, so multi-class metrics reflect binary performance\n"
    footer += "- Table sorted by comprehensive score (average of all metrics)\n"

    return "\n".join(table) + footer


def main():
    """Main function to generate all results."""
    try:
        # PART 1: Process default model/variant for main tables and visualizations
        logger.info(
            f"PART 1: Processing default model ({DEFAULT_MODEL}) and variant ({DEFAULT_VARIANT})"
        )

        # Find the latest predictions file for the default model/variant
        latest_file = find_latest_predictions_file(DEFAULT_MODEL, DEFAULT_VARIANT)
        logger.info(f"Using predictions file: {latest_file}")

        # Load predictions
        predictions = load_predictions(latest_file)
        logger.info(f"Loaded {len(predictions)} predictions")

        # Calculate all metrics from predictions
        df = calculate_all_metrics_from_predictions(predictions)
        logger.info(f"Calculated metrics for {len(df.columns)} metric columns")

        # Extract conditions
        conditions = ALLOWED_CONDITIONS
        logger.info(f"Processing {len(conditions)} conditions: {', '.join(conditions)}")

        # Create results directories
        os.makedirs("results/main", exist_ok=True)
        os.makedirs("results/appendix", exist_ok=True)
        os.makedirs("results/visualizations", exist_ok=True)

        # Extract confusion matrices for verification
        all_matrices = extract_confusion_matrices_from_metrics(df)
        if all_matrices:
            # Verify metrics
            verified_metrics = extract_and_verify_metrics(all_matrices, conditions)
            logger.info(f"Verified metrics for {len(verified_metrics)} conditions")

        # Create and save main table
        main_table = create_main_table(df, conditions)
        save_markdown(main_table, "results/main/main_results_table.md")

        # Create and save appendix table 1
        appendix_table1 = create_appendix_table1(df, conditions)
        save_markdown(appendix_table1, "results/appendix/appendix_table1.md")

        # Generate label count tables
        generate_label_count_tables()

        # Generate visualizations
        if all_matrices:
            generate_comprehensive_heatmaps(all_matrices, "results/visualizations")
        else:
            logger.warning("No confusion matrices found for visualization")

        # PART 2: Process ALL model/variant combinations for comparison tables
        logger.info("PART 2: Processing all model/variant combinations")

        # Process all model/variant combinations
        combined_df, processing_results = process_all_model_variants()

        if not combined_df.empty:
            # Create and save appendix tables for model/variant comparisons
            appendix_table3a = create_appendix_3a_binary_performance(combined_df)
            save_markdown(appendix_table3a, "results/appendix/appendix_table3a.md")

            appendix_table3b = create_appendix_3b_macro_f1_performance(combined_df)
            save_markdown(appendix_table3b, "results/appendix/appendix_table3b.md")

            appendix_table3c = create_appendix_3c_comprehensive_metrics(combined_df)
            save_markdown(appendix_table3c, "results/appendix/appendix_table3c.md")
        else:
            logger.warning("No data available for model/variant comparison tables")

        # Create processing summary table regardless of success
        appendix_table3d = create_appendix_3d_processing_summary(processing_results)
        save_markdown(appendix_table3d, "results/appendix/appendix_table3d.md")

        # Create appendix grade F1 breakdown
        appendix_grade_f1_breakdown = create_appendix_grade_f1_breakdown(predictions)
        save_markdown(
            appendix_grade_f1_breakdown, "results/appendix/appendix_table4.md"
        )

        # Create appendix cross-temporal analysis
        appendix_cross_temporal_analysis = create_appendix_cross_temporal_analysis(
            predictions
        )
        save_markdown(
            appendix_cross_temporal_analysis, "results/appendix/appendix_table5.md"
        )

        logger.info("All tables and visualizations generated successfully!")

    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
