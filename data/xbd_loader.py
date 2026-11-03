"""
xBD Dataset Loader and Preprocessor
===================================
Handles parsing xBD (xView2) satellite imagery and annotation files:
- Parses pre- and post-disaster geo-referenced/pixel JSON polygon annotations.
- Maps xBD damage classes: 'no-damage' (0), 'minor-damage' (1), 'major-damage' (2), 'destroyed' (3).
- Extracts paired building crops with spatial contextual margin.
- Provides PyTorch Dataset and DataLoader with dual-stream augmentations.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    Dataset = object
    DataLoader = None
    T = None
    TORCH_AVAILABLE = False

# Standard xBD Damage Severity Mapping
DAMAGE_CLASSES = ["none", "minor", "major", "destroyed"]
CLASS_TO_IDX = {
    "no-damage": 0,
    "none": 0,
    "minor-damage": 1,
    "minor": 1,
    "major-damage": 2,
    "major": 2,
    "destroyed": 3,
}
IDX_TO_CLASS = {0: "none", 1: "minor", 2: "major", 3: "destroyed"}
CLASS_COLORS = {
    0: (46, 204, 113),   # Green - No Damage
    1: (241, 196, 15),   # Yellow - Minor
    2: (230, 126, 34),   # Orange - Major
    3: (231, 76, 60),    # Red - Destroyed
}
CLASS_HEX = {
    0: "#2ecc71",
    1: "#f1c40f",
    2: "#e67e22",
    3: "#e74c3c",
}


def parse_wkt_polygon(wkt_str: str) -> List[Tuple[float, float]]:
    """Extract coordinates from a WKT POLYGON string."""
    match = re.search(r"POLYGON\s*\(\((.*?)\)\)", wkt_str)
    if not match:
        return []
    coords_text = match.group(1).strip()
    pts = []
    for pair in coords_text.split(","):
        parts = pair.strip().split()
        if len(parts) >= 2:
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return pts


def get_polygon_bbox(pts: List[Tuple[float, float]], margin_pct: float = 0.15,
                     img_w: int = 1024, img_h: int = 1024) -> Tuple[int, int, int, int]:
    """Calculate padded bounding box [xmin, ymin, xmax, ymax] clamped to image dimensions."""
    if not pts:
        return 0, 0, 64, 64
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    w = max(xmax - xmin, 16)
    h = max(ymax - ymin, 16)

    pad_x = w * margin_pct
    pad_y = h * margin_pct

    xmin = max(0, int(xmin - pad_x))
    ymin = max(0, int(ymin - pad_y))
    xmax = min(img_w, int(xmax + pad_x))
    ymax = min(img_h, int(ymax + pad_y))

    return xmin, ymin, xmax, ymax


class XBDCropDataset(Dataset):
    """
    PyTorch Dataset yielding paired pre- and post-disaster building crops.
    Returns:
        pre_img (Tensor [3, H, W])
        post_img (Tensor [3, H, W])
        label (int: 0, 1, 2, 3)
        info (dict with metadata)
    """

    def __init__(
        self,
        samples: List[Dict],
        transform: Optional[Any] = None,
        is_train: bool = False,
    ):
        self.samples = samples
        self.is_train = is_train

        if TORCH_AVAILABLE and T is not None:
            self.default_transform = T.Compose([
                T.Resize((128, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.default_transform = lambda img: (
                np.transpose(np.array(img.resize((128, 128)).convert("RGB"), dtype=np.float32) / 255.0, (2, 0, 1))
            )

        self.transform = transform or self.default_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, int, Dict]:
        item = self.samples[idx]

        if "pre_crop" in item and "post_crop" in item:
            # In-memory PIL Image or numpy array
            pre_img = item["pre_crop"]
            post_img = item["post_crop"]
            if isinstance(pre_img, np.ndarray):
                pre_img = Image.fromarray(pre_img)
            if isinstance(post_img, np.ndarray):
                post_img = Image.fromarray(post_img)
        else:
            # Load from disk
            pre_path = item["pre_img_path"]
            post_path = item["post_img_path"]
            bbox = item.get("bbox", None)

            pre_full = Image.open(pre_path).convert("RGB")
            post_full = Image.open(post_path).convert("RGB")

            if bbox is not None:
                pre_img = pre_full.crop(bbox)
                post_img = post_full.crop(bbox)
            else:
                pre_img = pre_full
                post_img = post_full

        # Apply synchronized spatial augmentations during training
        if self.is_train:
            # Random horizontal flip applied identically to pre and post
            if np.random.rand() > 0.5:
                pre_img = pre_img.transpose(Image.FLIP_LEFT_RIGHT)
                post_img = post_img.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.rand() > 0.5:
                pre_img = pre_img.transpose(Image.FLIP_TOP_BOTTOM)
                post_img = post_img.transpose(Image.FLIP_TOP_BOTTOM)

        pre_tensor = self.transform(pre_img)
        post_tensor = self.transform(post_img)
        label = int(item["label"])

        metadata = {
            "disaster": item.get("disaster", "unknown"),
            "building_id": item.get("building_id", str(idx)),
            "damage_class": IDX_TO_CLASS[label],
        }

        return pre_tensor, post_tensor, label, metadata


def get_data_transforms(img_size: int = 128):
    """Train and validation transforms for pre/post imagery."""
    if TORCH_AVAILABLE and T is not None:
        train_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        norm_fn = lambda img: np.transpose(
            np.array(img.resize((img_size, img_size)).convert("RGB"), dtype=np.float32) / 255.0, (2, 0, 1)
        )
        train_transform = norm_fn
        val_transform = norm_fn

    return train_transform, val_transform
