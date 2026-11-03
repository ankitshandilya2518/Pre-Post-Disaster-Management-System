"""
Model Evaluation and Confusion Matrix Generation
================================================
Comprehensive evaluation suite for the xBD Building Damage Severity Classifier.
Calculates:
- 4x4 Confusion Matrix (None, Minor, Major, Destroyed)
- Per-severity Precision, Recall, and F1-Scores
- Overall Accuracy and Macro/Weighted F1
- Comparative benchmark between Dual-Branch and Single-Branch models
- Saves visual confusion matrix plots and metrics JSON
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TORCH_AVAILABLE = False

from data.xbd_loader import DAMAGE_CLASSES, IDX_TO_CLASS
from utils.visualization import (
    create_confusion_matrix_figure,
    create_metrics_bar_figure,
)


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Computes confusion matrix and per-class precision, recall, F1-scores.
    """
    num_classes = len(DAMAGE_CLASSES)
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    metrics: Dict[str, Any] = {}
    f1_list = []
    prec_list = []
    rec_list = []

    for c in range(num_classes):
        class_name = IDX_TO_CLASS[c]
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        total_gt = cm[c, :].sum()

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics[class_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1-score": round(f1, 4),
            "support": int(total_gt),
        }
        prec_list.append(precision)
        rec_list.append(recall)
        f1_list.append(f1)

    total_samples = len(y_true)
    correct_samples = int((y_true == y_pred).sum())
    overall_accuracy = float(correct_samples / total_samples) if total_samples > 0 else 0.0

    metrics["overall_accuracy"] = round(overall_accuracy, 4)
    metrics["macro_precision"] = round(float(np.mean(prec_list)), 4)
    metrics["macro_recall"] = round(float(np.mean(rec_list)), 4)
    metrics["macro_f1"] = round(float(np.mean(f1_list)), 4)
    metrics["total_samples"] = int(total_samples)

    return cm, metrics


def evaluate_samples(
    model: Any,
    samples: List[Dict],
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Evaluates Dual-Branch model across a list of sample dictionaries.
    Works seamlessly with both PyTorch and NumPy model engines.
    """
    all_preds = []
    all_targets = []

    for s in samples:
        pre_p = Path(s["pre_path"])
        post_p = Path(s["post_path"])

        pre_img = Image.open(pre_p).convert("RGB")
        post_img = Image.open(post_p).convert("RGB")

        pred_label, _, probs, _ = model.predict(pre_img, post_img)
        pred_idx = int(np.argmax(probs))

        all_preds.append(pred_idx)
        all_targets.append(int(s["severity"]))

    y_true = np.array(all_targets, dtype=int)
    y_pred = np.array(all_preds, dtype=int)

    cm, metrics = compute_classification_metrics(y_true, y_pred)
    return cm, metrics, y_true, y_pred


def evaluate_dual_branch(
    model: Any,
    dataloader: Any,
    device: Any = None,
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Evaluates Dual-Branch PyTorch model across a DataLoader.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for DataLoader evaluation. Use evaluate_samples() instead.")

    model.eval()
    if device is not None:
        model.to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for pre, post, label, _ in dataloader:
            if device is not None:
                pre = pre.to(device)
                post = post.to(device)
            logits = model(pre, post)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(label.numpy())

    y_true = np.array(all_targets, dtype=int)
    y_pred = np.array(all_preds, dtype=int)

    cm, metrics = compute_classification_metrics(y_true, y_pred)
    return cm, metrics, y_true, y_pred


def save_evaluation_artifacts(
    cm: np.ndarray,
    metrics: Dict[str, Any],
    output_dir: Union[str, Path] = "models_saved",
    prefix: str = "dual_branch",
):
    """Saves metrics JSON and confusion matrix PNG."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    json_path = out_path / f"{prefix}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "confusion_matrix": cm.tolist(),
                "classes": DAMAGE_CLASSES,
            },
            f,
            indent=2,
        )

    # Save Confusion Matrix Image
    fig_cm = create_confusion_matrix_figure(
        cm,
        classes=["None (0)", "Minor (1)", "Major (2)", "Destroyed (3)"],
        normalize=True,
        title=f"xBD Damage Severity Confusion Matrix ({prefix.replace('_', ' ').title()})",
    )
    cm_img_path = out_path / f"{prefix}_confusion_matrix.png"
    fig_cm.savefig(cm_img_path, bbox_inches="tight", dpi=180)
    plt.close(fig_cm)

    # Save Metrics Bar Chart
    fig_bar = create_metrics_bar_figure(metrics)
    bar_img_path = out_path / f"{prefix}_metrics_bar.png"
    fig_bar.savefig(bar_img_path, bbox_inches="tight", dpi=180)
    plt.close(fig_bar)

    print(f"Artifacts saved in {out_path}:")
    print(f"  - {json_path}")
    print(f"  - {cm_img_path}")
    print(f"  - {bar_img_path}")
