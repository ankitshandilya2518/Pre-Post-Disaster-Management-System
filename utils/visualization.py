"""
Visualization and Computer Vision Rendering Utilities
======================================================
Generates publication-grade figures, damage attention heatmaps,
side-by-side satellite comparisons, and per-severity confusion matrices.
"""

import io
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Set local Matplotlib cache directory
mpl_cache = Path(__file__).resolve().parent.parent / ".matplotlib_cache"
mpl_cache.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_cache)

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data.xbd_loader import CLASS_COLORS, CLASS_HEX, DAMAGE_CLASSES


def create_confusion_matrix_figure(
    cm: np.ndarray,
    classes: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = "Damage Severity Confusion Matrix",
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Renders high-resolution, dark-themed confusion matrix with counts and percentages.
    """
    classes = classes or ["None (0)", "Minor (1)", "Major (2)", "Destroyed (3)"]
    cm_display = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9) if normalize else cm

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=180)
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#0b0f19")

    im = ax.imshow(cm_display, interpolation="nearest", cmap="magma" if cmap == "magma" else "Blues")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Normalized Rate" if normalize else "Sample Count", rotation=-90, va="bottom", color="#94a3b8", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#94a3b8", labelcolor="#94a3b8")
    cbar.outline.set_edgecolor("#334155")

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, color="#cbd5e1", fontsize=9, fontweight="bold")
    ax.set_yticklabels(classes, color="#cbd5e1", fontsize=9, fontweight="bold")
    ax.set_title(title, color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Ground Truth Severity", color="#94a3b8", fontsize=10, labelpad=8)
    ax.set_xlabel("Predicted Severity", color="#94a3b8", fontsize=10, labelpad=8)

    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")

    for spine in ax.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(1.2)

    # Annotate cells with count and percentage
    thresh = cm_display.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_pct = cm[i, j] / (cm[i].sum() + 1e-9)
            text_str = f"{val_pct:.1%}\n({int(cm[i, j])})" if normalize else f"{int(cm[i, j])}"
            ax.text(
                j,
                i,
                text_str,
                ha="center",
                va="center",
                color="#0b0f19" if cm_display[i, j] > thresh else "#f8fafc",
                fontsize=8.5,
                fontweight="bold",
            )

    fig.tight_layout()
    return fig


def create_metrics_bar_figure(metrics_dict: Dict[str, Dict[str, float]]) -> plt.Figure:
    """Renders Precision, Recall, and F1-score grouped bar chart per class with dark aesthetics."""
    classes = [c.capitalize() for c in DAMAGE_CLASSES]
    x = np.arange(len(classes))
    width = 0.26

    precisions = [metrics_dict[c.lower()]["precision"] * 100 for c in DAMAGE_CLASSES]
    recalls = [metrics_dict[c.lower()]["recall"] * 100 for c in DAMAGE_CLASSES]
    f1s = [metrics_dict[c.lower()]["f1-score"] * 100 for c in DAMAGE_CLASSES]

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=180)
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#0f172a")

    rects1 = ax.bar(x - width, precisions, width, label="Precision", color="#38bdf8", edgecolor="#0284c7", linewidth=0.8)
    rects2 = ax.bar(x, recalls, width, label="Recall", color="#4ade80", edgecolor="#16a34a", linewidth=0.8)
    rects3 = ax.bar(x + width, f1s, width, label="F1-Score", color="#fb923c", edgecolor="#ea580c", linewidth=0.8)

    ax.set_ylabel("Score (%)", color="#94a3b8", fontsize=9.5)
    ax.set_title("Per-Severity Classification Performance", color="#f8fafc", fontsize=11.5, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, color="#cbd5e1", fontsize=9.5, fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    ax.set_ylim(0, 115)
    ax.legend(loc="lower right", facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.15, color="#94a3b8")

    for spine in ax.spines.values():
        spine.set_color("#334155")

    # Value labels
    for rects in [rects1, rects2, rects3]:
        for r in rects:
            h = r.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(r.get_x() + r.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                color="#f1f5f9",
            )

    fig.tight_layout()
    return fig


def blend_heatmap_on_image(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.55,
    colormap_name: str = "turbo",
) -> Image.Image:
    """
    Blends continuous damage activation heatmap onto satellite image.
    """
    cmap = plt.get_cmap(colormap_name)
    rgba_heatmap = cmap(heatmap)  # [H, W, 4]
    rgb_heatmap = (rgba_heatmap[:, :, :3] * 255).astype(np.uint8)

    # Resize heatmap to match image dimensions
    heatmap_pil = Image.fromarray(rgb_heatmap).resize(image.size, Image.BILINEAR)
    blended = Image.blend(image.convert("RGB"), heatmap_pil, alpha=alpha)
    return blended


def draw_tactical_bounding_box(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    label: str = "BUILDING",
    confidence: float = 0.95,
    color_rgb: Tuple[int, int, int] = (56, 189, 248),
) -> Image.Image:
    """
    Draws a tactical military/GIS-style bounding box with corner reticles and badge.
    """
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    # Main rectangle with thin stroke
    draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=2)

    # Tactical corner brackets (5px length)
    c_len = min(max(w // 5, 6), 14)
    # Top-Left
    draw.line([(x1, y1), (x1 + c_len, y1)], fill=color_rgb, width=3)
    draw.line([(x1, y1), (x1, y1 + c_len)], fill=color_rgb, width=3)
    # Top-Right
    draw.line([(x2, y1), (x2 - c_len, y1)], fill=color_rgb, width=3)
    draw.line([(x2, y1), (x2, y1 + c_len)], fill=color_rgb, width=3)
    # Bottom-Left
    draw.line([(x1, y2), (x1 + c_len, y2)], fill=color_rgb, width=3)
    draw.line([(x1, y2), (x1, y2 - c_len)], fill=color_rgb, width=3)
    # Bottom-Right
    draw.line([(x2, y2), (x2 - c_len, y2)], fill=color_rgb, width=3)
    draw.line([(x2, y2), (x2, y2 - c_len)], fill=color_rgb, width=3)

    # Label badge at top-left
    tag_text = f"{label.upper()} [{int(confidence * 100)}%]"
    badge_w = len(tag_text) * 7 + 8
    badge_h = 16
    tag_y = max(y1 - badge_h - 2, 2)
    draw.rectangle([x1, tag_y, x1 + badge_w, tag_y + badge_h], fill=(15, 23, 42))
    draw.rectangle([x1, tag_y, x1 + badge_w, tag_y + badge_h], outline=color_rgb, width=1)
    draw.text((x1 + 4, tag_y + 1), tag_text, fill=color_rgb)

    return img


def create_side_by_side_comparison(
    pre_crop: Image.Image,
    post_crop: Image.Image,
    heatmap: Optional[np.ndarray] = None,
    pred_label: str = "none",
    confidence: float = 0.95,
    target_size: Tuple[int, int] = (256, 256),
) -> Image.Image:
    """Creates a unified 3-panel triage visual: Pre-Disaster | Post-Disaster | Damage Heatmap."""
    pre_res = pre_crop.resize(target_size, Image.BILINEAR).convert("RGB")
    post_res = post_crop.resize(target_size, Image.BILINEAR).convert("RGB")

    if heatmap is not None:
        heat_blend = blend_heatmap_on_image(post_res, heatmap, alpha=0.5)
    else:
        heat_blend = post_res

    w, h = target_size
    banner_h = 44
    canvas_w = w * 3 + 20
    canvas_h = h + banner_h + 20

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(11, 15, 25))
    draw = ImageDraw.Draw(canvas)

    # Paste panels
    canvas.paste(pre_res, (5, banner_h + 10))
    canvas.paste(post_res, (w + 10, banner_h + 10))
    canvas.paste(heat_blend, (w * 2 + 15, banner_h + 10))

    label_color = CLASS_COLORS.get(
        {"none": 0, "minor": 1, "major": 2, "destroyed": 3}.get(pred_label, 0), (255, 255, 255)
    )

    draw.text((10, 12), "PRE-DISASTER CROP", fill=(148, 163, 184))
    draw.text((w + 15, 12), "POST-DISASTER CROP", fill=(148, 163, 184))
    draw.text((w * 2 + 20, 12), "ATTENTION HEATMAP", fill=(148, 163, 184))

    badge_text = f"Predicted Severity: {pred_label.upper()} ({confidence * 100:.1f}% confidence)"
    draw.rectangle([5, banner_h + h + 12, canvas_w - 5, canvas_h - 2], fill=(15, 23, 42))
    draw.text((15, banner_h + h + 13), badge_text, fill=label_color)

    return canvas
