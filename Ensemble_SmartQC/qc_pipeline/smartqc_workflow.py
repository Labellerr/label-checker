"""
qc_pipeline/smartqc_workflow.py
================================
Thin adapter that wires SmartQCValidator into the label-checker
QCValidationWorkflow interface.

Usage
-----
::

    from qc_pipeline.smartqc_workflow import SmartQCWorkflow

    workflow = SmartQCWorkflow(
        output_dir="output",
        yolo_model_path="runs/detect/best.pt",   # optional
    )
    state, results, summary = workflow.run(
        coco_json_path="annotations.json",
        images_dir="images/",
        pdf_path="guidelines.pdf",               # optional
    )

    print(f"Accuracy: {summary.matches / summary.total * 100:.1f}%")

The ``results`` list contains one dict per annotation in the label-checker
result schema (``annotation_id``, ``label``, ``prediction_label``,
``is_match``, ``confidence``, ``rationale``) plus SmartQC-specific fields
(``qc_flag``, ``quality_score``, signal scores, weights).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .validators.smartqc import SmartQCValidator

logger = logging.getLogger(__name__)


@dataclass
class SmartQCSummary:
    total:              int   = 0
    matches:            int   = 0   # ACCEPT count
    mismatches:         int   = 0   # REVIEW + FLAG + REJECT
    flags:              int   = 0   # FLAG count
    rejects:            int   = 0   # REJECT count
    reviews:            int   = 0   # REVIEW count
    average_confidence: float = 0.0
    mean_clip_score:    float = 0.0
    clip_mismatches:    int   = 0
    proto_match_rate:   float = 0.0
    mean_iou:           float = 0.0
    false_positives:    int   = 0


@dataclass
class SmartQCState:
    """Mirrors the LangGraph shared-state structure used by QCValidationWorkflow."""
    pdf_text:        str                   = ""
    guidelines:      Dict[str, str]        = field(default_factory=dict)
    categories:      List[str]             = field(default_factory=list)
    results:         List[Dict[str, Any]]  = field(default_factory=list)
    errors:          List[str]             = field(default_factory=list)
    status_message:  str                   = ""


class SmartQCWorkflow:
    """
    Drop-in replacement for ``QCValidationWorkflow`` backed by SmartQC's
    three-signal engine.

    Parameters
    ----------
    output_dir : str
        Directory for CSV and log outputs.
    yolo_model_path : str or None
        Path to a trained YOLOv8 ``best.pt``.
    device : str or None
        Torch device override (``"cuda"`` / ``"cpu"``).
    use_grayscale : bool
        Grayscale DINOv2 crops (default ``True``, matches training setup).
    batch_size : int
        Batch size for model inference.
    yolo_conf : float
        YOLO confidence threshold.
    """

    def __init__(
        self,
        output_dir: str = "output",
        yolo_model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_grayscale: bool = True,
        batch_size: int = 16,
        yolo_conf: float = 0.25,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self._validator = SmartQCValidator(
            device=device,
            use_grayscale=use_grayscale,
            batch_size=batch_size,
            yolo_model_path=yolo_model_path,
            yolo_conf=yolo_conf,
        )

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        coco_json_path: str,
        images_dir: str,
        pdf_path: Optional[str] = None,
        max_files: Optional[int] = None,
        confidence_threshold: float = 0.5,
    ) -> Tuple[SmartQCState, List[Dict[str, Any]], SmartQCSummary]:
        """
        Execute the full SmartQC pipeline.

        Parameters
        ----------
        coco_json_path : str
            Path to a COCO-format ``annotations.json``.
        images_dir : str
            Root directory containing the images referenced in the JSON.
        pdf_path : str or None
            Optional guidelines PDF.  Text is extracted and stored in state
            but SmartQC's embedding signals do not depend on it.
        max_files : int or None
            Cap the number of annotations processed (useful for testing).
        confidence_threshold : float
            Minimum ``confidence`` (quality score) for an annotation to be
            counted as a match in the summary.

        Returns
        -------
        (state, results, summary)
        """
        state = SmartQCState()

        # ── Optional: guidelines extraction ──────────────────────────────────
        if pdf_path and os.path.exists(pdf_path):
            state.pdf_text  = self._extract_pdf_text(pdf_path)
            state.guidelines = self._parse_guidelines(state.pdf_text)
            state.status_message = "Guidelines extracted"
            logger.info("Extracted %d label definitions from PDF.", len(state.guidelines))

        # ── Load COCO annotations ─────────────────────────────────────────────
        annotations, img_lookup, cat_lookup = self._load_coco(coco_json_path)
        if max_files:
            annotations = annotations[:max_files]

        state.categories = list(cat_lookup.values())

        # Attach file_name and category_name directly on each annotation dict
        # so the validator and fit() can consume them without extra lookups.
        enriched = []
        for ann in annotations:
            img_info = img_lookup.get(ann["image_id"])
            if img_info is None:
                continue
            ann = dict(ann)
            ann["file_name"]      = img_info["file_name"]
            ann["category_name"]  = cat_lookup.get(ann["category_id"], "unknown")
            enriched.append(ann)

        # ── Fit prototypes ────────────────────────────────────────────────────
        logger.info("Fitting prototypes on %d annotations ...", len(enriched))
        state.status_message = "Fitting prototypes"
        try:
            self._validator.fit(enriched, images_dir)
        except Exception as exc:
            logger.error("Fit failed: %s", exc)
            state.errors.append(str(exc))
            return state, [], SmartQCSummary()

        # ── Validate each annotation ──────────────────────────────────────────
        state.status_message = "Validating annotations"
        results: List[Dict[str, Any]] = []

        # Cache open images to avoid re-opening the same file many times
        img_cache: Dict[str, Image.Image] = {}
        # Cache YOLO predictions per image
        pred_cache: Dict[str, List[Dict]] = {}

        for ann in enriched:
            img_path = os.path.join(images_dir, ann["file_name"])
            try:
                if img_path not in img_cache:
                    img_cache[img_path] = Image.open(img_path).convert("RGB")
                image = img_cache[img_path]

                # Run YOLO once per image and cache
                if self._validator._yolo_model is not None:
                    if img_path not in pred_cache:
                        pred_cache[img_path] = self._validator._run_yolo(image)
                    preds = pred_cache[img_path]
                else:
                    preds = None

                result = self._validator.validate(
                    annotation=ann,
                    image=image,
                    image_predictions=preds,
                    guidelines=state.guidelines or None,
                )
                results.append(result)
            except Exception as exc:
                logger.debug("Error on annotation %s: %s", ann.get("id"), exc)
                state.errors.append(f"ann {ann.get('id')}: {exc}")

        state.results = results

        # ── Save outputs ──────────────────────────────────────────────────────
        self._save_outputs(results)

        # ── Build summary ─────────────────────────────────────────────────────
        summary = self._build_summary(results, confidence_threshold)
        state.status_message = (
            f"Complete — {summary.matches}/{summary.total} accepted"
        )
        return state, results, summary

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _load_coco(
        path: str,
    ) -> Tuple[List[Dict], Dict[int, Dict], Dict[int, str]]:
        with open(path) as f:
            coco = json.load(f)
        img_lookup = {img["id"]: img for img in coco.get("images", [])}
        cat_lookup = {c["id"]: c["name"] for c in coco.get("categories", [])}
        return coco.get("annotations", []), img_lookup, cat_lookup

    @staticmethod
    def _extract_pdf_text(pdf_path: str) -> str:
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as exc:
            logger.warning("PDF extraction failed: %s", exc)
            return ""

    @staticmethod
    def _parse_guidelines(text: str) -> Dict[str, str]:
        """
        Best-effort extraction of label definitions from free-form PDF text.
        Lines like ``barcode: a sticker containing machine-readable data``
        are parsed into {label: definition}.
        """
        guidelines: Dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip().lower().replace(" ", "_")
                if 1 < len(key) < 40 and len(val.strip()) > 5:
                    guidelines[key] = val.strip()
        return guidelines

    def _save_outputs(self, results: List[Dict[str, Any]]) -> None:
        import csv

        if not results:
            return

        out_path = os.path.join(self.output_dir, "qc_results.json")
        with open(out_path, "w") as f:
            json.dump({"results": results, "total": len(results)}, f, indent=2)

        csv_path = os.path.join(self.output_dir, "qc_results.csv")
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        logger.info("Outputs saved to %s/", self.output_dir)

    @staticmethod
    def _build_summary(
        results: List[Dict[str, Any]],
        confidence_threshold: float,
    ) -> SmartQCSummary:
        if not results:
            return SmartQCSummary()

        total   = len(results)
        flags   = {r["annotation_id"]: r.get("qc_flag", "FLAG") for r in results}
        accept  = sum(1 for f in flags.values() if f == "ACCEPT")
        review  = sum(1 for f in flags.values() if f == "REVIEW")
        flag    = sum(1 for f in flags.values() if f == "FLAG")
        reject  = sum(1 for f in flags.values() if f == "REJECT")

        avg_conf    = sum(r.get("confidence", 0.0) for r in results) / total
        clip_scores = [r.get("clip_lbl_score", 0.0) for r in results if "clip_lbl_score" in r]
        mean_clip   = sum(clip_scores) / len(clip_scores) if clip_scores else 0.0
        clip_mm     = sum(r.get("clip_mismatch", 0) for r in results)
        proto_m     = [r.get("proto_match", 0) for r in results if "proto_match" in r]
        mean_proto  = sum(proto_m) / len(proto_m) if proto_m else 0.0
        ious        = [r.get("best_iou", 0.0) for r in results if "best_iou" in r]
        mean_iou    = sum(ious) / len(ious) if ious else 0.0
        fps         = sum(r.get("is_false_positive", 0) for r in results)

        return SmartQCSummary(
            total=total,
            matches=accept,
            mismatches=total - accept,
            flags=flag,
            rejects=reject,
            reviews=review,
            average_confidence=round(avg_conf, 4),
            mean_clip_score=round(mean_clip, 4),
            clip_mismatches=clip_mm,
            proto_match_rate=round(mean_proto, 4),
            mean_iou=round(mean_iou, 4),
            false_positives=fps,
        )
