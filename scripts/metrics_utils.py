#!/usr/bin/env python3
"""
Metrics utilities for calculating performance metrics using scikit-learn.
Handles grade, attribution, and certainty metrics with proper macro averaging.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
    precision_score,
    recall_score,
)


# Configuration constants
GRADE_CLASSES = 6  # Grades 0-5
ATTR_CLASSES = 2  # Attribution 0-1
CERT_CLASSES = 5  # Certainty 0-4
DEFAULT_F1 = 1.0  # Default F1 for missing classes


def _confusion_and_prf(
    y_true: List[int],
    y_pred: List[int],
    labels: List[int],
    zero_division: float = 1.0,
) -> Dict[str, np.ndarray | float]:
    """
    Calculate confusion matrix and performance metrics.

    Returns:
        Dictionary containing:
        - confusion_matrix: (n_classes × n_classes) confusion matrix
        - per_class_precision: precision for each class
        - per_class_recall: recall for each class
        - per_class_f1: f1 for each class
        - macro_precision: macro-averaged precision
        - macro_recall: macro-averaged recall
        - macro_f1: macro-averaged f1
    """
    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Calculate per-class metrics with zero_division handling
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=zero_division,
    )

    # Calculate macro averages
    macro_prec = np.mean(prec)
    macro_rec = np.mean(rec)
    macro_f1 = np.mean(f1)

    return {
        "confusion_matrix": cm,
        "per_class_precision": prec,
        "per_class_recall": rec,
        "per_class_f1": f1,
        "per_class_support": support,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
    }


def grade_metrics(
    y_true: List[int],
    y_pred: List[int],
    n_classes: int = GRADE_CLASSES,
) -> Dict[str, np.ndarray | float]:
    """
    Calculate multi-class grade performance metrics (0-5).
    """
    labels = list(range(n_classes))
    return _confusion_and_prf(y_true, y_pred, labels, zero_division=DEFAULT_F1)


def attribution_metrics(
    y_true: List[int],
    y_pred: List[int],
    n_classes: int = ATTR_CLASSES,
) -> Dict[str, np.ndarray | float]:
    """
    Calculate attribution performance metrics (0-1).
    """
    labels = list(range(n_classes))
    return _confusion_and_prf(y_true, y_pred, labels, zero_division=DEFAULT_F1)


def certainty_metrics(
    y_true: List[int],
    y_pred: List[int],
    n_classes: int = CERT_CLASSES,
) -> Dict[str, np.ndarray | float]:
    """
    Calculate certainty performance metrics (0-4).
    """
    labels = list(range(n_classes))
    return _confusion_and_prf(y_true, y_pred, labels, zero_division=DEFAULT_F1)


def binary_metrics(
    y_true: List[int],
    y_pred: List[int],
    positive_label: Optional[int] = None,
) -> Dict[str, np.ndarray | float]:
    """
    Calculate binary metrics (0 vs non-zero).

    Args:
        y_true: True labels
        y_pred: Predicted labels
        positive_label: If specified, treats this label as positive class.
                       If None, treats 0 as negative and all others as positive.
    """
    y_true_bin = _to_binary(y_true, positive_label)
    y_pred_bin = _to_binary(y_pred, positive_label)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])

    # Calculate metrics directly for positive class (class 1)
    # Using pos_label=1 to get metrics for the "has condition" class
    precision = precision_score(
        y_true_bin, y_pred_bin, pos_label=1, zero_division=DEFAULT_F1
    )
    recall = recall_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=DEFAULT_F1)
    f1 = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=DEFAULT_F1)

    # Also get per-class metrics for completeness
    prec_per_class, rec_per_class, f1_per_class, support = (
        precision_recall_fscore_support(
            y_true_bin,
            y_pred_bin,
            labels=[0, 1],
            average=None,
            zero_division=DEFAULT_F1,
        )
    )

    # Macro averages
    macro_precision = np.mean(prec_per_class)
    macro_recall = np.mean(rec_per_class)
    macro_f1 = np.mean(f1_per_class)

    return {
        "confusion_matrix": cm,
        "per_class_precision": prec_per_class,
        "per_class_recall": rec_per_class,
        "per_class_f1": f1_per_class,
        "per_class_support": support,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "binary_precision": precision,  # Precision for positive class
        "binary_recall": recall,  # Recall for positive class
        "binary_f1": f1,  # F1 for positive class
    }


def _to_binary(
    labels: List[int],
    positive_label: Optional[int] = None,
) -> List[int]:
    """Convert labels to binary (0 vs non-zero)."""
    if positive_label is None:
        # 0 remains 0, everything else becomes 1
        return [0 if x == 0 else 1 for x in labels]
    else:
        # Specific label becomes 1, everything else becomes 0
        return [1 if x == positive_label else 0 for x in labels]


def calculate_metrics_for_type(
    y_true: List[int],
    y_pred: List[int],
    metric_type: str,
) -> Dict[str, float | np.ndarray]:
    """
    Calculate metrics based on the type (grade, attribution, or certainty).

    Args:
        y_true: True labels
        y_pred: Predicted labels
        metric_type: One of 'grade', 'attribution', 'certainty'

    Returns:
        Dictionary with both multi-class and binary metrics
    """
    # Get the appropriate metrics function
    if metric_type == "grade":
        multi_metrics = grade_metrics(y_true, y_pred)
    elif metric_type == "attribution":
        multi_metrics = attribution_metrics(y_true, y_pred)
    elif metric_type == "certainty":
        multi_metrics = certainty_metrics(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

    # Get binary metrics
    bin_metrics = binary_metrics(y_true, y_pred)

    # Combine results
    return {
        "multi_class": multi_metrics,
        "binary": bin_metrics,
        "macro_f1": multi_metrics["macro_f1"],
        "binary_f1": bin_metrics["macro_f1"],  # Use macro F1 for binary classification
        "binary_precision": bin_metrics[
            "macro_precision"
        ],  # Use macro precision for binary classification
        "binary_recall": bin_metrics[
            "macro_recall"
        ],  # Use macro recall for binary classification
        "positive_class_f1": bin_metrics["binary_f1"],  # F1 for positive class only
        "positive_class_precision": bin_metrics[
            "binary_precision"
        ],  # Precision for positive class only
        "positive_class_recall": bin_metrics[
            "binary_recall"
        ],  # Recall for positive class only
    }


def summarise_condition(
    condition: str,
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, float | np.ndarray]:
    """
    Calculate all metrics for a single condition.

    Returns dictionary with 'grade' and 'binary' sub-dictionaries.
    """
    # Calculate grade metrics (6-way classification)
    grade_result = grade_metrics(y_true, y_pred)

    # Calculate binary metrics (0 vs 1+)
    binary_result = binary_metrics(y_true, y_pred)

    return {
        "condition": condition,
        "grade": grade_result,
        "binary": binary_result,
    }


def convert_to_binary_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    """
    Convert a multi-class confusion matrix to binary (0 vs non-zero).

    In sklearn confusion matrices:
    - Rows represent TRUE labels
    - Columns represent PREDICTED labels

    Args:
        cm: Multi-class confusion matrix

    Returns:
        2x2 binary confusion matrix
    """
    binary_cm = np.zeros((2, 2), dtype=int)

    # True Negative: true 0, predicted 0
    binary_cm[0, 0] = cm[0, 0]

    # False Positive: true 0, predicted 1+ (non-zero)
    binary_cm[0, 1] = cm[0, 1:].sum()

    # False Negative: true 1+ (non-zero), predicted 0
    binary_cm[1, 0] = cm[1:, 0].sum()

    # True Positive: true 1+ (non-zero), predicted 1+ (non-zero)
    binary_cm[1, 1] = cm[1:, 1:].sum()

    return binary_cm


def calculate_binary_metrics_from_matrix(cm: np.ndarray) -> Dict[str, float]:
    """
    Calculate binary metrics from a confusion matrix.

    Args:
        cm: Confusion matrix (can be multi-class)

    Returns:
        Dictionary with binary_precision, binary_recall, binary_f1 (all macro-averaged)
    """
    # Convert to binary if needed
    if cm.shape[0] > 2:
        binary_cm = convert_to_binary_confusion_matrix(cm)
    else:
        binary_cm = cm

    # Extract values following sklearn convention: rows = true, cols = predicted
    # Binary confusion matrix layout:
    #           Pred 0  Pred 1
    # True 0      TN      FP
    # True 1      FN      TP

    tn = binary_cm[0, 0]  # true 0, pred 0
    fp = binary_cm[0, 1]  # true 0, pred 1
    fn = binary_cm[1, 0]  # true 1, pred 0
    tp = binary_cm[1, 1]  # true 1, pred 1

    # Calculate metrics for positive class (class 1) with zero division handling
    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else DEFAULT_F1
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else DEFAULT_F1
    f1_pos = (
        2 * (precision_pos * recall_pos) / (precision_pos + recall_pos)
        if (precision_pos + recall_pos) > 0
        else 0.0
    )

    # Calculate metrics for negative class (class 0) with zero division handling
    precision_neg = tn / (tn + fn) if (tn + fn) > 0 else DEFAULT_F1
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else DEFAULT_F1
    f1_neg = (
        2 * (precision_neg * recall_neg) / (precision_neg + recall_neg)
        if (precision_neg + recall_neg) > 0
        else 0.0
    )

    # Calculate macro averages
    macro_precision = (precision_pos + precision_neg) / 2
    macro_recall = (recall_pos + recall_neg) / 2
    macro_f1 = (f1_pos + f1_neg) / 2

    return {
        "binary_precision": macro_precision,  # Macro-averaged precision
        "binary_recall": macro_recall,  # Macro-averaged recall
        "binary_f1": macro_f1,  # Macro-averaged F1
        "positive_class_precision": precision_pos,  # Positive class precision
        "positive_class_recall": recall_pos,  # Positive class recall
        "positive_class_f1": f1_pos,  # Positive class F1
    }
