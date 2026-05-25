"""
fetchers/labellerr.py
=====================
Load annotations directly from the Labellerr platform via its SDK.

Usage:
    from qc_pipeline.fetchers.labellerr import load_from_labellerr

    annotations = load_from_labellerr(
        api_key="your_labellerr_api_key",
        project_id="your_project_id",
        batch_id="optional_batch_id",   # omit to fetch all batches
    )

Returns the same list-of-dicts format as fetchers/local.py so it
can be passed directly to SmartQCValidator.fit() and .validate().

Note: Requires the Labellerr Python SDK.
    pip install labellerr-sdk     (or follow Labellerr platform docs)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def load_from_labellerr(
    api_key: str,
    project_id: str,
    batch_id: Optional[str] = None,
    status_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch annotations from the Labellerr platform.

    Parameters
    ----------
    api_key : str
        Your Labellerr API key (from platform settings).
    project_id : str
        The project ID to fetch annotations from.
    batch_id : str or None
        Optional — fetch a specific batch. Omit to fetch all.
    status_filter : str or None
        Optional — filter by annotation status (e.g. ``"completed"``).

    Returns
    -------
    list of annotation dicts (same schema as fetchers/local.py)
    """
    try:
        from labellerr import Client  # type: ignore
    except ImportError:
        raise ImportError(
            "Labellerr SDK not installed. Run: pip install labellerr-sdk\n"
            "Or load annotations from local files using fetchers/local.py."
        )

    client = Client(api_key=api_key)

    logger.info("Fetching annotations from Labellerr project %s ...", project_id)

    kwargs: Dict[str, Any] = {"project_id": project_id}
    if batch_id:
        kwargs["batch_id"] = batch_id
    if status_filter:
        kwargs["status"] = status_filter

    raw = client.get_annotations(**kwargs)

    annotations = []
    for item in raw:
        for ann in item.get("annotations", []):
            bbox = ann.get("bbox", ann.get("bounding_box", {}))
            if isinstance(bbox, dict):
                x = bbox.get("x", bbox.get("left", 0))
                y = bbox.get("y", bbox.get("top",  0))
                w = bbox.get("width",  bbox.get("w", 0))
                h = bbox.get("height", bbox.get("h", 0))
            elif isinstance(bbox, list) and len(bbox) >= 4:
                x, y, w, h = bbox[:4]
            else:
                continue

            annotations.append({
                "id":            str(ann.get("id", "")),
                "image_id":      item.get("image_id", ""),
                "file_name":     item.get("file_name", item.get("filename", "")),
                "category_name": ann.get("label", ann.get("class_name", "unknown")),
                "category_id":   ann.get("class_id", -1),
                "bbox":          [float(x), float(y), float(w), float(h)],
            })

    logger.info("Fetched %d annotations from Labellerr.", len(annotations))
    return annotations
