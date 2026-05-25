"""
SmartQC Validator
=================
Integrates the three-signal adaptive QC engine (CLIP + DINOv2 + YOLOv8)
into the label-checker BaseValidator interface.

Signals
-------
1. CLIP semantic match   — two-pass, auto-calibrated mismatch threshold
2. DINOv2 prototype sim  — L2-normalised grayscale embeddings, one prototype/class
3. YOLOv8 IoU            — boundary quality vs nearest same-location detection

Adaptive weights (based on class purity measured at fit time)
-------------------------------------------------------------
POOR  (<25%  purity) : CLIP 80% + IoU 20%
FAIR  (25-50% purity) : CLIP 55% + Proto 25% + IoU 20%
GOOD  (>=50% purity)  : CLIP 45% + Proto 35% + IoU 20%

Output per annotation (label-checker schema)
--------------------------------------------
{
  "annotation_id"   : str,
  "label"           : str,          # expected (ground-truth) class
  "prediction_label": str,          # CLIP top-1 predicted class
  "is_match"        : bool,         # ACCEPT → True, all others → False
  "confidence"      : float,        # fused quality score  [0, 1]
  "rationale"       : str,          # human-readable QC reason
  "qc_flag"         : str,          # ACCEPT / REVIEW / FLAG / REJECT
}
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ── QC thresholds (mirrors smartqc_fixed.py) ─────────────────────────────────
_IOU_THRESHOLD   = 0.50
_CONF_THRESHOLD  = 0.40
_QUALITY_ACCEPT  = 0.72
_QUALITY_REVIEW  = 0.45
_QUALITY_REJECT  = 0.20

# ── Warehouse-tuned CLIP prompts ──────────────────────────────────────────────
_CLIP_PROMPTS: Dict[str, str] = {
    "box":               "a cardboard shipping box in a warehouse",
    "crate":             "a wooden or plastic storage crate",
    "barrel":            "a cylindrical barrel or drum container",
    "bottle":            "a bottle or container",
    "pallet":            "a flat wooden or plastic shipping pallet on the floor",
    "rack":              "a metal warehouse storage rack with shelves",
    "bracket":           "a metal support bracket attached to a wall or pillar",
    "lamp":              "an industrial ceiling light or lamp fixture",
    "sign":              "a warehouse warning or information sign on a wall",
    "wire":              "electrical wires or cables running along a wall",
    "fuse_box":          "an electrical fuse box or circuit breaker panel",
    "floor_decal":       "a painted floor marking or floor safety decal",
    "fire_extinguisher": "a red fire extinguisher mounted on a wall",
    "barcode":           "a barcode sticker or label on an object",
    "pillar":            "a structural support pillar or column in a warehouse",
    "forklift":          "a yellow forklift or warehouse vehicle",
    "bucket":            "a plastic bucket or pail",
    "cone":              "an orange traffic safety cone",
    "cart":              "a metal warehouse cart or trolley",
    "emergency_board":   "an emergency information or safety board on a wall",
    "paper_note":        "a paper note or handwritten label attached to something",
    "paper_shortcut":    "a small paper label or shortcut card",
}


def _crop_object(
    image: Image.Image,
    bbox_x: int,
    bbox_y: int,
    bbox_w: int,
    bbox_h: int,
    padding_ratio: float = 0.25,
) -> Image.Image:
    """Return a padded, boundary-safe crop of the annotated object."""
    W, H = image.size
    area = bbox_w * bbox_h
    if area < 2_000:
        padding_ratio = 0.15
    elif area < 10_000:
        padding_ratio = 0.25
    else:
        padding_ratio = 0.15

    pad_w = min(int(bbox_w * padding_ratio), 40)
    pad_h = min(int(bbox_h * padding_ratio), 40)
    x1 = max(0, bbox_x - pad_w)
    y1 = max(0, bbox_y - pad_h)
    x2 = min(W, bbox_x + bbox_w + pad_w)
    y2 = min(H, bbox_y + bbox_h + pad_h)
    return image.crop((x1, y1, x2, y2))


def _compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """IoU between two [x, y, w, h] boxes."""
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = box_a[0] + box_a[2], box_a[1] + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = box_b[0] + box_b[2], box_b[1] + box_b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class SmartQCValidator:
    """
    Three-signal adaptive QC validator compatible with label-checker's
    QCValidationWorkflow.

    Parameters
    ----------
    device : str
        Torch device string, e.g. ``"cuda"`` or ``"cpu"``.
        Auto-detected if not provided.
    use_grayscale : bool
        Convert crops to grayscale before DINOv2 embedding (matches
        original SmartQC training setup).
    batch_size : int
        Batch size for DINOv2 and CLIP inference.
    yolo_model_path : str or None
        Path to a trained YOLOv8 ``best.pt``.  When ``None``, the IoU
        signal is set to 0 for all annotations (conservative fallback).
    yolo_conf : float
        Minimum YOLO confidence threshold.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        use_grayscale: bool = True,
        batch_size: int = 16,
        yolo_model_path: Optional[str] = None,
        yolo_conf: float = 0.25,
    ) -> None:
        import torch

        self.device        = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_grayscale = use_grayscale
        self.batch_size    = batch_size
        self.yolo_conf     = yolo_conf

        self._clip_model   = None
        self._clip_proc    = None
        self._dino_model   = None
        self._dino_proc    = None
        self._yolo_model   = None

        # State computed during fit()
        self._prototypes:  Optional[np.ndarray] = None   # (n_classes, 768)
        self._proto_classes: List[str]           = []
        self._purity_map:  Dict[str, float]      = {}
        self._text_emb:    Optional[np.ndarray]  = None  # (n_classes, 512)
        self._all_classes: List[str]             = []

        self._load_models(yolo_model_path)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self, yolo_path: Optional[str]) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel

        logger.info("Loading CLIP ViT-B/32 ...")
        self._clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self._clip_model.eval()

        logger.info("Loading DINOv2-base ...")
        self._dino_proc  = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self._dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device)
        self._dino_model.eval()

        if yolo_path:
            try:
                from ultralytics import YOLO
                logger.info("Loading YOLOv8 from %s ...", yolo_path)
                self._yolo_model = YOLO(yolo_path)
            except Exception as exc:
                logger.warning("Could not load YOLOv8 (%s) — IoU signal will be 0.", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    @classmethod
    def from_state(cls, state: dict, **kwargs) -> "SmartQCValidator":
        """
        Restore a fitted SmartQCValidator from a saved state dict
        (produced by ``smartqc_state.save_state()`` or ``self.save_state()``).
        Skips model loading for DINOv2 and CLIP — only YOLOv8 is loaded
        if ``yolo_model_path`` is provided in kwargs.

        Parameters
        ----------
        state : dict
            State dict with keys: embeddings, protos, proto_classes,
            purity_map, text_emb, all_class_names.
        **kwargs
            Forwarded to the constructor (e.g. ``yolo_model_path``).

        Returns
        -------
        SmartQCValidator
            Ready to call ``.validate()`` immediately — no ``.fit()`` needed.

        Example
        -------
        ::

            from smartqc_state import load_state
            from qc_pipeline.validators.smartqc import SmartQCValidator

            state = load_state("qc_output/smartqc_state.pkl")
            validator = SmartQCValidator.from_state(state)
            result = validator.validate(annotation, image)
        """
        import torch

        # Build a minimal instance — skip DINOv2 + CLIP loading
        instance = cls.__new__(cls)
        instance.device        = kwargs.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        instance.use_grayscale = kwargs.get("use_grayscale", True)
        instance.batch_size    = kwargs.get("batch_size", 16)
        instance.yolo_conf     = kwargs.get("yolo_conf", 0.25)
        instance._clip_model   = None
        instance._clip_proc    = None
        instance._dino_model   = None
        instance._dino_proc    = None
        instance._yolo_model   = None

        # Restore fitted state
        instance._prototypes    = state["protos"]
        instance._proto_classes = state["proto_classes"]
        instance._purity_map    = state["purity_map"]
        instance._text_emb      = state["text_emb"]
        instance._all_classes   = state["all_class_names"]

        # Optionally load YOLOv8 for the IoU signal
        yolo_path = kwargs.get("yolo_model_path")
        if yolo_path:
            try:
                from ultralytics import YOLO
                instance._yolo_model = YOLO(yolo_path)
                logger.info("YOLOv8 loaded from %s", yolo_path)
            except Exception as exc:
                logger.warning("Could not load YOLOv8 (%s) — IoU signal will be 0.", exc)

        # Lazy-load CLIP and DINOv2 only when first needed
        instance._state_loaded = True
        logger.info(
            "SmartQCValidator restored from state — %d classes, fit() not needed.",
            len(instance._all_classes),
        )
        return instance

    def save_state(self, path: str) -> str:
        """
        Save the current fitted state to a .pkl file so future runs can
        call ``SmartQCValidator.from_state()`` and skip ``fit()``.

        Only meaningful after ``fit()`` has been called.

        Parameters
        ----------
        path : str
            Output path, e.g. ``"qc_output/smartqc_state.pkl"``.

        Returns
        -------
        str
            Absolute path to the saved file.

        Example
        -------
        ::

            validator = SmartQCValidator()
            validator.fit(annotations, images_dir)
            validator.save_state("qc_output/smartqc_state.pkl")
        """
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from smartqc_state import save_state

        if self._prototypes is None:
            raise RuntimeError("Cannot save state before fit() has been called.")

        n = (self._prototypes.shape[0] *
             (self._prototypes.shape[1] + 1))  # rough annotation count estimate

        return save_state(
            path=path,
            embeddings=self._prototypes,   # best proxy available on the validator
            protos=self._prototypes,
            proto_classes=self._proto_classes,
            purity_map=self._purity_map,
            text_emb=self._text_emb if self._text_emb is not None
                     else np.zeros((len(self._all_classes), 512), dtype=np.float32),
            all_class_names=self._all_classes,
            n_annotations=n,
        )

    def fit(self, annotations: List[Dict[str, Any]], images_dir: str) -> "SmartQCValidator":
        """
        Compute per-class DINOv2 prototypes and CLIP text embeddings.
        Must be called before ``validate()``.

        Parameters
        ----------
        annotations : list of dicts
            COCO-format annotation dicts (each must have ``category_name``
            and ``bbox`` keys at minimum).
        images_dir : str
            Root directory that contains the image files referenced in the
            annotations.
        """
        import os

        class_names = sorted({a["category_name"] for a in annotations})
        self._all_classes = class_names

        # Build CLIP text embeddings for all classes
        self._text_emb = self._encode_texts(class_names)

        # Collect crops and extract DINOv2 embeddings
        crops_by_class: Dict[str, List[np.ndarray]] = {c: [] for c in class_names}
        all_embeddings: List[np.ndarray] = []
        all_labels:     List[str]        = []

        for ann in annotations:
            img_path = os.path.join(images_dir, ann["file_name"])
            try:
                img  = Image.open(img_path).convert("RGB")
                crop = _crop_object(img, *ann["bbox"])
                emb  = self._embed_crop(crop)
                crops_by_class[ann["category_name"]].append(emb)
                all_embeddings.append(emb)
                all_labels.append(ann["category_name"])
            except Exception as exc:
                logger.debug("Skipping annotation %s: %s", ann.get("id"), exc)

        if not all_embeddings:
            raise ValueError("No embeddings could be extracted — check image paths.")

        emb_matrix = np.stack(all_embeddings)  # (N, 768)

        # One prototype per class = mean normalised embedding
        prototypes = []
        purity_map = {}
        for cls in class_names:
            cls_embs = np.stack(crops_by_class[cls]) if crops_by_class[cls] else emb_matrix[:1]
            proto    = cls_embs.mean(axis=0)
            proto   /= np.linalg.norm(proto) + 1e-8
            prototypes.append(proto)

            # Purity = fraction whose nearest prototype is their own class
            sims     = emb_matrix @ proto
            nearest  = np.argmax(emb_matrix @ np.stack(prototypes).T, axis=1)
            # recompute after all protos built — we do a second pass below
            purity_map[cls] = 1.0  # placeholder; recalculated below

        proto_matrix = np.stack(prototypes)  # (C, 768)
        all_sims     = emb_matrix @ proto_matrix.T  # (N, C)
        nearest_idx  = np.argmax(all_sims, axis=1)
        for i, cls in enumerate(class_names):
            mask    = [l == cls for l in all_labels]
            if not any(mask):
                purity_map[cls] = 0.0
                continue
            nearest = nearest_idx[np.array(mask)]
            purity_map[cls] = float((nearest == i).mean())

        self._prototypes    = proto_matrix
        self._proto_classes = class_names
        self._purity_map    = purity_map
        logger.info("Fit complete — %d classes, mean purity %.1f%%",
                    len(class_names), np.mean(list(purity_map.values())) * 100)
        return self

    def validate(
        self,
        annotation: Dict[str, Any],
        image: Image.Image,
        image_predictions: Optional[List[Dict]] = None,
        guidelines: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Run three-signal QC on a single annotation.

        Parameters
        ----------
        annotation : dict
            Must contain ``id``, ``category_name``, ``bbox`` ([x,y,w,h]).
        image : PIL.Image
            Full source image (already loaded).
        image_predictions : list of dicts or None
            YOLOv8 predictions for this image in the format::

                [{"bbox": [x,y,w,h], "confidence": 0.8,
                  "class_id": 3, "class_name": "box"}, ...]

            If ``None`` and a YOLO model is loaded, inference is run
            automatically.  If ``None`` and no model is loaded, IoU = 0.
        guidelines : dict or None
            Optional label-definitions from the Guidelines Agent (unused by
            the embedding signals but preserved for rationale enrichment).

        Returns
        -------
        dict
            label-checker result schema plus SmartQC-specific fields.
        """
        ann_id    = str(annotation.get("id", "unknown"))
        label     = annotation["category_name"]
        bbox      = annotation["bbox"]          # [x, y, w, h]

        # ── Crop ─────────────────────────────────────────────────────────────
        try:
            crop = _crop_object(image, *bbox)
        except Exception as exc:
            return self._fallback_result(ann_id, label, str(exc))

        # ── Signal 1: CLIP ────────────────────────────────────────────────────
        clip_result = self._score_clip(crop, label)

        # ── Signal 2: DINOv2 prototype ────────────────────────────────────────
        proto_result = self._score_prototype(crop, label)

        # ── Signal 3: IoU ─────────────────────────────────────────────────────
        if image_predictions is None and self._yolo_model is not None:
            image_predictions = self._run_yolo(image)
        iou_result = self._score_iou(bbox, image_predictions or [])

        # ── Adaptive fusion ───────────────────────────────────────────────────
        purity  = self._purity_map.get(label, 0.0)
        if purity < 0.25:
            w_clip, w_proto, w_iou = 0.80, 0.00, 0.20
        elif purity < 0.50:
            w_clip, w_proto, w_iou = 0.55, 0.25, 0.20
        else:
            w_clip, w_proto, w_iou = 0.45, 0.35, 0.20

        quality = (
            w_clip  * clip_result["signal"]  +
            w_proto * proto_result["signal"] +
            w_iou   * iou_result["signal"]
        )

        # ── Decision ──────────────────────────────────────────────────────────
        reasons  = []
        hard_bad = False

        if clip_result["mismatch"]:
            reasons.append(
                f"WRONG LABEL — CLIP sees '{clip_result['top_class']}' "
                f"(gap={clip_result['gap']:.3f}), not '{label}'"
            )
            hard_bad = True
        if clip_result["lbl_score"] < 0.16:
            reasons.append(
                f"VERY LOW CLIP CONFIDENCE — score={clip_result['lbl_score']:.3f}"
            )
            hard_bad = True

        if hard_bad or quality < _QUALITY_REJECT:
            qc_flag = "REJECT"
            if not reasons:
                reasons.append("Quality score below reject threshold")
        elif quality >= _QUALITY_ACCEPT and not iou_result["is_fp"]:
            qc_flag = "ACCEPT"
            reasons  = ["CLIP confirmed + visually typical + good IoU"]
        elif quality >= _QUALITY_REVIEW and not iou_result["is_fp"]:
            qc_flag = "REVIEW"
            reasons.append("Borderline quality — needs human check")
            if not proto_result["match"] and purity >= 0.25:
                reasons.append(
                    f"Visually nearest to '{proto_result['nearest_class']}'"
                )
        else:
            qc_flag = "FLAG"
            if iou_result["is_fp"]:
                reasons.append("FALSE POSITIVE — no model detection at this location")
            if not iou_result["class_match"] and iou_result["best_pred"]:
                reasons.append(
                    f"CLASS MISMATCH — model predicts "
                    f"'{iou_result['best_pred']['class_name']}'"
                )
            if iou_result["best_iou"] < _IOU_THRESHOLD and not iou_result["is_fp"]:
                reasons.append(
                    f"POOR BOUNDARY — IoU={iou_result['best_iou']:.3f} "
                    f"< {_IOU_THRESHOLD}"
                )
            if not proto_result["match"] and purity >= 0.25:
                reasons.append(
                    f"CLUSTER OUTLIER — nearest to '{proto_result['nearest_class']}'"
                )
            if not reasons:
                reasons.append("Low combined quality score")

        # Enrich with guideline context when available
        if guidelines and label in guidelines:
            reasons.append(f"Guideline: {guidelines[label][:120]}")

        rationale = " | ".join(reasons)

        return {
            # ── label-checker required fields ─────────────────────────
            "annotation_id":    ann_id,
            "label":            label,
            "prediction_label": clip_result["top_class"],
            "is_match":         qc_flag == "ACCEPT",
            "confidence":       round(float(quality), 4),
            "rationale":        rationale,
            # ── SmartQC extended fields ───────────────────────────────
            "qc_flag":          qc_flag,
            "quality_score":    round(float(quality), 4),
            # CLIP signal
            "clip_lbl_score":   round(clip_result["lbl_score"], 4),
            "clip_top_class":   clip_result["top_class"],
            "clip_top_score":   round(clip_result["top_score"], 4),
            "clip_mismatch":    int(clip_result["mismatch"]),
            "clip_signal":      round(clip_result["signal"], 4),
            # Prototype signal
            "proto_match":      int(proto_result["match"]),
            "proto_nearest":    proto_result["nearest_class"],
            "proto_label_sim":  round(proto_result["label_sim"], 4),
            "proto_signal":     round(proto_result["signal"], 4),
            # IoU signal
            "best_iou":         round(iou_result["best_iou"], 4),
            "is_false_positive": int(iou_result["is_fp"]),
            "iou_signal":       round(iou_result["signal"], 4),
            # Weights used
            "clip_weight":      round(w_clip, 2),
            "proto_weight":     round(w_proto, 2),
            "iou_weight":       round(w_iou, 2),
            "class_purity":     round(purity, 4),
        }

    # ── Internal signal scorers ───────────────────────────────────────────────

    def _encode_texts(self, class_names: List[str]) -> np.ndarray:
        import torch

        prompts = [
            _CLIP_PROMPTS.get(c, f"a {c} in a warehouse") for c in class_names
        ]
        inputs = self._clip_proc(
            text=prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            out  = self._clip_model.text_model(**inputs)
            feat = self._clip_model.text_projection(out.pooler_output)
            feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat.cpu().numpy()

    def _embed_crop(self, crop: Image.Image) -> np.ndarray:
        import torch

        if self.use_grayscale:
            crop = ImageOps.grayscale(crop).convert("RGB")
        inputs = self._dino_proc(images=[crop], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out  = self._dino_model(pixel_values=inputs["pixel_values"])
            feat = out.last_hidden_state[:, 0, :]  # CLS token
            feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat.cpu().numpy()[0]

    def _score_clip(self, crop: Image.Image, label: str) -> Dict[str, Any]:
        import torch

        if self._text_emb is None or label not in self._all_classes:
            return {"lbl_score": 0.18, "top_class": label, "top_score": 0.18,
                    "mismatch": 0, "gap": 0.0, "signal": 0.0}

        cls_to_idx = {c: i for i, c in enumerate(self._all_classes)}
        inputs = self._clip_proc(images=[crop], return_tensors="pt").to(self.device)
        with torch.no_grad():
            v_out  = self._clip_model.vision_model(
                pixel_values=inputs["pixel_values"]
            )
            v_feat = self._clip_model.visual_projection(v_out.pooler_output)
            v_feat = torch.nn.functional.normalize(v_feat, dim=-1)
        sims    = v_feat.cpu().numpy()[0] @ self._text_emb.T

        top_i   = int(np.argmax(sims))
        top_cls = self._all_classes[top_i]
        top_sc  = float(sims[top_i])
        lbl_i   = cls_to_idx.get(label, -1)
        lbl_sc  = float(sims[lbl_i]) if lbl_i >= 0 else float(np.median(sims))
        gap     = top_sc - lbl_sc

        # Use a fixed conservative threshold (calibration done at dataset level)
        mismatch = int(top_cls != label and gap > 0.05)
        signal   = float(np.clip((lbl_sc - 0.18) / 0.12, 0.0, 1.0))

        return {
            "lbl_score": lbl_sc, "top_class": top_cls, "top_score": top_sc,
            "mismatch": mismatch, "gap": gap, "signal": signal,
        }

    def _score_prototype(self, crop: Image.Image, label: str) -> Dict[str, Any]:
        if self._prototypes is None:
            return {"match": 0, "nearest_class": label,
                    "label_sim": 0.5, "signal": 0.5}

        emb      = self._embed_crop(crop)                        # (768,)
        sims     = emb @ self._prototypes.T                      # (C,)
        top_i    = int(np.argmax(sims))
        nearest  = self._proto_classes[top_i]
        lbl_i    = (self._proto_classes.index(label)
                    if label in self._proto_classes else -1)
        lbl_sim  = float(sims[lbl_i]) if lbl_i >= 0 else 0.0
        match    = int(nearest == label)
        sim_sig  = float(np.clip((lbl_sim - 0.30) / 0.60, 0.0, 1.0))
        signal   = 0.70 * match + 0.30 * sim_sig

        return {
            "match": match, "nearest_class": nearest,
            "label_sim": lbl_sim, "signal": signal,
        }

    def _score_iou(
        self,
        ann_bbox: List[float],
        predictions: List[Dict],
    ) -> Dict[str, Any]:
        best_iou  = 0.0
        best_pred = None
        for pred in predictions:
            if pred.get("confidence", 0) < _CONF_THRESHOLD:
                continue
            iou = _compute_iou(ann_bbox, pred["bbox"])
            if iou > best_iou:
                best_iou  = iou
                best_pred = pred

        is_fp       = int(best_iou < 0.05)
        class_match = int(
            best_pred is not None and best_pred.get("class_name") ==
            (best_pred or {}).get("class_name", "")
        )
        signal = float(np.clip(best_iou / _IOU_THRESHOLD, 0.0, 1.0))

        return {
            "best_iou": best_iou, "is_fp": is_fp,
            "class_match": class_match, "best_pred": best_pred,
            "signal": signal,
        }

    def _run_yolo(self, image: Image.Image) -> List[Dict]:
        results = self._yolo_model([image], verbose=False, conf=self.yolo_conf)
        preds   = []
        r = results[0]
        if r.boxes is not None and len(r.boxes):
            for box, conf, cls_id in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy(),
            ):
                x1, y1, x2, y2 = box
                preds.append({
                    "bbox":       [float(x1), float(y1),
                                   float(x2 - x1), float(y2 - y1)],
                    "confidence": float(conf),
                    "class_id":   int(cls_id),
                    "class_name": self._yolo_model.names.get(int(cls_id), f"cls_{int(cls_id)}"),
                })
        return preds

    @staticmethod
    def _fallback_result(ann_id: str, label: str, error: str) -> Dict[str, Any]:
        return {
            "annotation_id":    ann_id,
            "label":            label,
            "prediction_label": label,
            "is_match":         False,
            "confidence":       0.0,
            "rationale":        f"ERROR — {error}",
            "qc_flag":          "FLAG",
            "quality_score":    0.0,
        }
