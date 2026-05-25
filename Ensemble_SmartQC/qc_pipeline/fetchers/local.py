"""
fetchers/local.py
=================
Load annotations from local files — YOLO format or COCO JSON.

Usage:
    from qc_pipeline.fetchers.local import load_annotations

    # From YOLO .txt files
    annotations = load_annotations(
        format="yolo",
        images_dir="dataset/images",
        labels_dir="dataset/labels",
        class_map_path="dataset/class_mapping.json",
    )

    # From COCO JSON
    annotations = load_annotations(
        format="coco",
        coco_json_path="dataset/annotations.json",
        images_dir="dataset/images",
    )

Returns a list of dicts, each with keys:
    id, image_id, file_name, category_name, bbox ([x,y,w,h] in pixels)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def load_annotations(
    format: str = "coco",
    images_dir: str = "",
    labels_dir: str = "",
    coco_json_path: str = "",
    class_map_path: str = "",
    split: str = "train",
) -> List[Dict[str, Any]]:
    """
    Unified annotation loader.

    Parameters
    ----------
    format : str
        ``"coco"`` or ``"yolo"``.
    images_dir : str
        Directory containing image files.
    labels_dir : str
        Directory containing YOLO .txt label files (YOLO mode only).
    coco_json_path : str
        Path to COCO annotations JSON (COCO mode only).
    class_map_path : str
        Path to class_mapping.json (YOLO mode only — maps IDs to names).
    split : str
        Dataset split subfolder to use in YOLO mode (default ``"train"``).

    Returns
    -------
    list of annotation dicts
    """
    if format.lower() == "coco":
        return _load_coco(coco_json_path, images_dir)
    elif format.lower() == "yolo":
        return _load_yolo(images_dir, labels_dir, class_map_path, split)
    else:
        raise ValueError(f"Unsupported format '{format}'. Use 'coco' or 'yolo'.")


def _load_coco(
    coco_json_path: str,
    images_dir: str,
) -> List[Dict[str, Any]]:
    """Load from COCO JSON format."""
    with open(coco_json_path) as f:
        coco = json.load(f)

    img_lookup = {img["id"]: img for img in coco.get("images", [])}
    cat_lookup = {c["id"]: c["name"] for c in coco.get("categories", [])}

    annotations = []
    for ann in coco.get("annotations", []):
        img_info = img_lookup.get(ann["image_id"])
        if img_info is None:
            continue
        annotations.append({
            "id":            str(ann["id"]),
            "image_id":      ann["image_id"],
            "file_name":     img_info["file_name"],
            "category_name": cat_lookup.get(ann["category_id"], "unknown"),
            "category_id":   ann["category_id"],
            "bbox":          ann["bbox"],   # [x, y, w, h] in pixels
        })
    return annotations


def _load_yolo(
    images_dir: str,
    labels_dir: str,
    class_map_path: str,
    split: str,
) -> List[Dict[str, Any]]:
    """Load from YOLO .txt label files."""
    # Load class map
    with open(class_map_path) as f:
        cm = json.load(f)
    id_to_name = {c["id"]: c["name"] for c in cm["classes"]}

    img_dir = os.path.join(images_dir, split) if split else images_dir
    lbl_dir = os.path.join(labels_dir, split) if split else labels_dir

    annotations = []
    ann_id = 0

    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        stem    = os.path.splitext(fname)[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        img_path = os.path.join(img_dir, fname)

        if not os.path.exists(lbl_path):
            continue

        from PIL import Image
        try:
            img = Image.open(img_path)
            W, H = img.size
        except Exception:
            continue

        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                # Convert normalised cx,cy,w,h to pixel x,y,w,h (top-left)
                pw = w * W
                ph = h * H
                px = cx * W - pw / 2
                py = cy * H - ph / 2

                annotations.append({
                    "id":            str(ann_id),
                    "image_id":      stem,
                    "file_name":     fname,
                    "category_name": id_to_name.get(cls_id, f"class_{cls_id}"),
                    "category_id":   cls_id,
                    "bbox":          [px, py, pw, ph],
                })
                ann_id += 1

    return annotations
