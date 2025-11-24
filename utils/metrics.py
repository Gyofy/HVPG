import os
from typing import Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)


def find_best_cutoff_youden(labels, predictions, num_thresholds: int = 101) -> float:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    best_cutoff = 0.0
    best_youden = -1.0
    cutoffs = np.linspace(0, 1, num_thresholds)

    for cutoff in cutoffs:
        pred_labels = (predictions > cutoff).astype(int)
        cm = confusion_matrix(labels, pred_labels)
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        youden = sensitivity + specificity - 1

        if youden > best_youden:
            best_youden = youden
            best_cutoff = cutoff

    return best_cutoff


def compute_metrics(labels, predictions, cutoff: float):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    binary_preds = (predictions > cutoff).astype(int)

    cm = confusion_matrix(labels, binary_preds)
    report = classification_report(labels, binary_preds)
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)

    return cm, report, fpr, tpr, roc_auc


def save_confusion_matrix(cm, output_path: str, title: str = "Confusion Matrix"):
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.savefig(output_path)
    plt.close()


def save_roc_curve(fpr, tpr, roc_auc: float, output_path: str, title: str):
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close()


def save_text_report(report: str, output_path: str, header: str | None = None):
    with open(output_path, "w") as f:
        if header:
            f.write(header + "\n")
        f.write(report)


def compute_mean_roc(fprs: Iterable[np.ndarray], tprs: Iterable[np.ndarray], num_points: int = 100):
    mean_fpr = np.linspace(0, 1, num_points)
    interp_tprs = []

    for fpr, tpr in zip(fprs, tprs):
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    return mean_fpr, mean_tpr, mean_auc


def save_mean_roc_curve(fprs, tprs, output_path: str, label: str = "Mean ROC"):
    mean_fpr, mean_tpr, mean_auc = compute_mean_roc(fprs, tprs)
    plt.figure(figsize=(7, 7))
    for fpr, tpr in zip(fprs, tprs):
        plt.plot(fpr, tpr, color="gray", alpha=0.4, lw=1)
    plt.plot(mean_fpr, mean_tpr, color="black", lw=2, label=f"{label} (AUC = {mean_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Cross-Validated ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return mean_auc


def save_overall_metrics(
    labels,
    predictions,
    cutoff: float,
    cm_path: str,
    roc_path: str,
    report_path: str,
    title_prefix: str = "Overall",
):
    cm, report, fpr, tpr, roc_auc = compute_metrics(labels, predictions, cutoff)
    save_confusion_matrix(cm, cm_path, title=f"{title_prefix} Confusion Matrix")
    save_roc_curve(fpr, tpr, roc_auc, roc_path, title=f"{title_prefix} ROC Curve")
    save_text_report(report, report_path, header=f"{title_prefix} Classification Report")
    return cm, report, roc_auc


