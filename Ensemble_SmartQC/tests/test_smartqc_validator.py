"""
tests/test_smartqc_validator.py
================================
Unit tests for SmartQCValidator and SmartQCWorkflow.

Run with:
    pytest tests/test_smartqc_validator.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_rgb_image(w: int = 64, h: int = 64) -> Image.Image:
    return Image.fromarray(
        np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    )


def _make_annotation(
    ann_id: int = 1,
    category_name: str = "box",
    bbox: List[float] = None,
    file_name: str = "img_001.jpg",
) -> Dict[str, Any]:
    return {
        "id":            ann_id,
        "image_id":      1,
        "category_id":   1,
        "category_name": category_name,
        "file_name":     file_name,
        "bbox":          bbox or [10, 10, 40, 40],
    }


def _make_coco_json(tmpdir: str, n_images: int = 2, n_ann_each: int = 3) -> str:
    """Write a minimal COCO JSON and matching dummy images to tmpdir."""
    images, annotations = [], []
    img_dir = os.path.join(tmpdir, "images")
    os.makedirs(img_dir, exist_ok=True)
    categories = [
        {"id": 1, "name": "box"},
        {"id": 2, "name": "pallet"},
    ]
    ann_id = 1
    for i in range(1, n_images + 1):
        fname = f"img_{i:03d}.jpg"
        _make_rgb_image(128, 128).save(os.path.join(img_dir, fname))
        images.append({"id": i, "file_name": fname, "width": 128, "height": 128})
        for j in range(n_ann_each):
            annotations.append({
                "id":          ann_id,
                "image_id":    i,
                "category_id": 1 + (j % 2),
                "bbox":        [10 + j * 5, 10, 30, 30],
            })
            ann_id += 1

    coco_path = os.path.join(tmpdir, "annotations.json")
    with open(coco_path, "w") as f:
        json.dump({
            "images":      images,
            "annotations": annotations,
            "categories":  categories,
        }, f)
    return coco_path, img_dir


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _mock_validator() -> "SmartQCValidator":  # noqa: F821
    """
    Return a SmartQCValidator with all heavy model calls patched out so tests
    run without GPU or model downloads.
    """
    import torch
    from qc_pipeline.validators.smartqc import SmartQCValidator

    with (
        patch("qc_pipeline.validators.smartqc.CLIPProcessor.from_pretrained",
              return_value=MagicMock()),
        patch("qc_pipeline.validators.smartqc.CLIPModel.from_pretrained",
              return_value=MagicMock()),
        patch("qc_pipeline.validators.smartqc.AutoImageProcessor.from_pretrained",
              return_value=MagicMock()),
        patch("qc_pipeline.validators.smartqc.AutoModel.from_pretrained",
              return_value=MagicMock()),
    ):
        v = SmartQCValidator(device="cpu")

    # Inject synthetic prototypes and text embeddings
    v._all_classes   = ["box", "pallet", "rack"]
    v._proto_classes = ["box", "pallet", "rack"]
    v._prototypes    = np.eye(3, 768, dtype=np.float32)  # identity-style
    v._text_emb      = np.eye(3, 512, dtype=np.float32)
    v._purity_map    = {"box": 0.70, "pallet": 0.40, "rack": 0.20}

    # Patch internal embedding/scoring to return deterministic values
    v._embed_crop   = lambda crop: np.array([1.0] + [0.0] * 767, dtype=np.float32)
    v._score_clip   = lambda crop, label: {
        "lbl_score": 0.25, "top_class": label, "top_score": 0.25,
        "mismatch": 0, "gap": 0.0, "signal": 0.58,
    }
    v._score_prototype = lambda crop, label: {
        "match": 1, "nearest_class": label,
        "label_sim": 0.80, "signal": 0.85,
    }
    return v


# ── SmartQCValidator unit tests ───────────────────────────────────────────────

class TestSmartQCValidatorOutputSchema:
    """Validate that every required label-checker field is present."""

    REQUIRED_FIELDS = {
        "annotation_id", "label", "prediction_label",
        "is_match", "confidence", "rationale",
    }
    SMARTQC_FIELDS = {
        "qc_flag", "quality_score",
        "clip_lbl_score", "clip_top_class", "clip_mismatch",
        "proto_match", "proto_nearest",
        "best_iou", "is_false_positive",
        "clip_weight", "proto_weight", "iou_weight",
    }

    def test_accept_verdict_schema(self):
        v   = _mock_validator()
        img = _make_rgb_image()
        ann = _make_annotation()
        res = v.validate(ann, img, image_predictions=[])
        for field in self.REQUIRED_FIELDS | self.SMARTQC_FIELDS:
            assert field in res, f"Missing field: {field}"

    def test_is_match_true_only_for_accept(self):
        v   = _mock_validator()
        img = _make_rgb_image()
        ann = _make_annotation()
        res = v.validate(ann, img, image_predictions=[])
        assert isinstance(res["is_match"], bool)
        assert res["is_match"] == (res["qc_flag"] == "ACCEPT")

    def test_confidence_in_unit_range(self):
        v   = _mock_validator()
        img = _make_rgb_image()
        res = v.validate(_make_annotation(), img, image_predictions=[])
        assert 0.0 <= res["confidence"] <= 1.0

    def test_qc_flag_is_valid(self):
        v   = _mock_validator()
        img = _make_rgb_image()
        res = v.validate(_make_annotation(), img, image_predictions=[])
        assert res["qc_flag"] in ("ACCEPT", "REVIEW", "FLAG", "REJECT")

    def test_weights_sum_to_one(self):
        v   = _mock_validator()
        img = _make_rgb_image()
        res = v.validate(_make_annotation(), img, image_predictions=[])
        total = res["clip_weight"] + res["proto_weight"] + res["iou_weight"]
        assert abs(total - 1.0) < 1e-5, f"Weights sum to {total}, expected 1.0"


class TestAdaptiveWeights:
    """Weights must follow the purity tiers defined in smartqc_fixed.py."""

    def _validate_with_purity(self, purity: float, label: str = "box"):
        v = _mock_validator()
        v._purity_map = {label: purity}
        img = _make_rgb_image()
        return v.validate(_make_annotation(category_name=label), img, image_predictions=[])

    def test_poor_class_uses_clip_heavy(self):
        res = self._validate_with_purity(0.10)
        assert res["clip_weight"]  == pytest.approx(0.80)
        assert res["proto_weight"] == pytest.approx(0.00)
        assert res["iou_weight"]   == pytest.approx(0.20)

    def test_fair_class_weights(self):
        res = self._validate_with_purity(0.35)
        assert res["clip_weight"]  == pytest.approx(0.55)
        assert res["proto_weight"] == pytest.approx(0.25)
        assert res["iou_weight"]   == pytest.approx(0.20)

    def test_good_class_weights(self):
        res = self._validate_with_purity(0.70)
        assert res["clip_weight"]  == pytest.approx(0.45)
        assert res["proto_weight"] == pytest.approx(0.35)
        assert res["iou_weight"]   == pytest.approx(0.20)


class TestIoUSignal:
    """IoU scoring and REJECT/FLAG thresholds."""

    def test_no_predictions_gives_zero_iou(self):
        v   = _mock_validator()
        img = _make_rgb_image()
        res = v.validate(_make_annotation(), img, image_predictions=[])
        assert res["best_iou"] == pytest.approx(0.0)
        assert res["is_false_positive"] == 1

    def test_perfect_prediction_gives_high_iou(self):
        v   = _mock_validator()
        img = _make_rgb_image(128, 128)
        ann = _make_annotation(bbox=[10, 10, 40, 40])
        preds = [{"bbox": [10, 10, 40, 40], "confidence": 0.9,
                  "class_id": 1, "class_name": "box"}]
        res = v.validate(ann, img, image_predictions=preds)
        assert res["best_iou"] == pytest.approx(1.0, abs=1e-4)

    def test_low_conf_prediction_ignored(self):
        v   = _mock_validator()
        img = _make_rgb_image(128, 128)
        ann = _make_annotation(bbox=[10, 10, 40, 40])
        preds = [{"bbox": [10, 10, 40, 40], "confidence": 0.10,
                  "class_id": 1, "class_name": "box"}]
        res = v.validate(ann, img, image_predictions=preds)
        # confidence 0.10 < _CONF_THRESHOLD (0.40) — should be ignored
        assert res["best_iou"] == pytest.approx(0.0)


class TestCLIPMismatchTrigger:
    """A CLIP mismatch should force a REJECT regardless of quality score."""

    def test_mismatch_forces_reject(self):
        v   = _mock_validator()
        img = _make_rgb_image()
        ann = _make_annotation(category_name="box")

        # Inject a mismatch
        def bad_clip(crop, label):
            return {
                "lbl_score": 0.20, "top_class": "forklift", "top_score": 0.30,
                "mismatch": 1, "gap": 0.10, "signal": 0.30,
            }
        v._score_clip = bad_clip
        res = v.validate(ann, img, image_predictions=[])
        assert res["qc_flag"] == "REJECT"
        assert res["is_match"] is False
        assert "WRONG LABEL" in res["rationale"]


# ── Factory tests ─────────────────────────────────────────────────────────────

class TestCreateValidatorFactory:

    def test_smartqc_registered(self):
        from qc_pipeline.validators import create_validator
        with (
            patch("qc_pipeline.validators.smartqc.CLIPProcessor.from_pretrained",
                  return_value=MagicMock()),
            patch("qc_pipeline.validators.smartqc.CLIPModel.from_pretrained",
                  return_value=MagicMock()),
            patch("qc_pipeline.validators.smartqc.AutoImageProcessor.from_pretrained",
                  return_value=MagicMock()),
            patch("qc_pipeline.validators.smartqc.AutoModel.from_pretrained",
                  return_value=MagicMock()),
        ):
            v = create_validator("smartqc")
        from qc_pipeline.validators.smartqc import SmartQCValidator
        assert isinstance(v, SmartQCValidator)

    def test_unknown_provider_raises(self):
        from qc_pipeline.validators import create_validator
        with pytest.raises(ValueError, match="Unknown provider"):
            create_validator("nonexistent_provider")

    def test_smartqc_no_api_key_needed(self):
        """SmartQC must not raise when api_key is None."""
        from qc_pipeline.validators import create_validator
        with (
            patch("qc_pipeline.validators.smartqc.CLIPProcessor.from_pretrained",
                  return_value=MagicMock()),
            patch("qc_pipeline.validators.smartqc.CLIPModel.from_pretrained",
                  return_value=MagicMock()),
            patch("qc_pipeline.validators.smartqc.AutoImageProcessor.from_pretrained",
                  return_value=MagicMock()),
            patch("qc_pipeline.validators.smartqc.AutoModel.from_pretrained",
                  return_value=MagicMock()),
        ):
            v = create_validator("smartqc", api_key=None, model_name=None)
        assert v is not None


# ── SmartQCWorkflow integration test ─────────────────────────────────────────

class TestSmartQCWorkflow:

    def test_run_returns_correct_types(self):
        from qc_pipeline.smartqc_workflow import SmartQCWorkflow, SmartQCState, SmartQCSummary

        with tempfile.TemporaryDirectory() as tmpdir:
            coco_path, img_dir = _make_coco_json(tmpdir, n_images=2, n_ann_each=2)
            out_dir = os.path.join(tmpdir, "output")

            workflow = SmartQCWorkflow(output_dir=out_dir)

            # Patch the validator so no real models are loaded
            workflow._validator = _mock_validator()
            # Patch fit so it doesn't require actual embeddings
            workflow._validator.fit = MagicMock(return_value=workflow._validator)

            state, results, summary = workflow.run(
                coco_json_path=coco_path,
                images_dir=img_dir,
            )

        assert isinstance(state,   SmartQCState)
        assert isinstance(results, list)
        assert isinstance(summary, SmartQCSummary)
        assert summary.total == len(results)

    def test_outputs_written(self):
        from qc_pipeline.smartqc_workflow import SmartQCWorkflow

        with tempfile.TemporaryDirectory() as tmpdir:
            coco_path, img_dir = _make_coco_json(tmpdir, n_images=1, n_ann_each=2)
            out_dir = os.path.join(tmpdir, "output")

            workflow = SmartQCWorkflow(output_dir=out_dir)
            workflow._validator = _mock_validator()
            workflow._validator.fit = MagicMock(return_value=workflow._validator)
            workflow.run(coco_json_path=coco_path, images_dir=img_dir)

        assert os.path.exists(os.path.join(out_dir, "qc_results.json"))
        assert os.path.exists(os.path.join(out_dir, "qc_results.csv"))
