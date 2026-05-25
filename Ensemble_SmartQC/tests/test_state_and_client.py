"""
tests/test_state_and_client.py
================================
Tests for smartqc_state.py (pkl save/load) and qc_pipeline/client.py.
All tests run without GPU or model downloads — heavy dependencies are mocked.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Make sure local imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Fixtures ─────────────────────────────────────────────────────

def _make_state(n_classes=3, n_ann=12):
    return {
        "embeddings":      np.random.randn(n_ann, 768).astype(np.float32),
        "protos":          np.eye(n_classes, 768, dtype=np.float32),
        "proto_classes":   ["box", "pallet", "forklift"][:n_classes],
        "purity_map":      {"box": 0.60, "pallet": 0.40, "forklift": 0.72},
        "text_emb":        np.eye(n_classes, 512, dtype=np.float32),
        "all_class_names": ["box", "pallet", "forklift"][:n_classes],
        "thresholds": {
            "QUALITY_ACCEPT": 0.72,
            "QUALITY_REJECT": 0.20,
            "IOU_THRESHOLD":  0.50,
        },
        "meta": {
            "saved_at":      "2026-05-24T12:00:00",
            "n_annotations": n_ann,
            "n_classes":     n_classes,
            "emb_shape":     [n_ann, 768],
            "proto_shape":   [n_classes, 768],
            "text_emb_shape":[n_classes, 512],
            "version":       "1.0",
        },
    }

def _make_coco_json(tmpdir, n_images=2, n_ann_each=3):
    img_dir = os.path.join(tmpdir, "images")
    os.makedirs(img_dir, exist_ok=True)
    images, annotations = [], []
    categories = [
        {"id": 1, "name": "box"},
        {"id": 2, "name": "pallet"},
    ]
    ann_id = 1
    for i in range(1, n_images + 1):
        fname = f"img_{i:03d}.jpg"
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(img_dir, fname))
        images.append({"id": i, "file_name": fname, "width": 64, "height": 64})
        for j in range(n_ann_each):
            annotations.append({
                "id":          ann_id,
                "image_id":    i,
                "category_id": 1 + (j % 2),
                "bbox":        [5, 5, 20, 20],
            })
            ann_id += 1
    coco_path = os.path.join(tmpdir, "annotations.json")
    with open(coco_path, "w") as f:
        json.dump({"images": images, "annotations": annotations,
                   "categories": categories}, f)
    return coco_path, img_dir


# ── smartqc_state tests ──────────────────────────────────────────

class TestSaveState:
    def test_file_created(self, tmp_path):
        from smartqc_state import save_state
        state  = _make_state()
        outpath = str(tmp_path / "smartqc_state.pkl")
        save_state(
            path=outpath,
            embeddings=state["embeddings"],
            protos=state["protos"],
            proto_classes=state["proto_classes"],
            purity_map=state["purity_map"],
            text_emb=state["text_emb"],
            all_class_names=state["all_class_names"],
            n_annotations=12,
        )
        assert os.path.exists(outpath)
        assert os.path.getsize(outpath) > 0

    def test_round_trip(self, tmp_path):
        from smartqc_state import save_state, load_state
        state   = _make_state()
        outpath = str(tmp_path / "smartqc_state.pkl")
        save_state(
            path=outpath,
            embeddings=state["embeddings"],
            protos=state["protos"],
            proto_classes=state["proto_classes"],
            purity_map=state["purity_map"],
            text_emb=state["text_emb"],
            all_class_names=state["all_class_names"],
            n_annotations=12,
        )
        loaded = load_state(outpath)
        assert loaded is not None
        assert loaded["proto_classes"] == state["proto_classes"]
        assert loaded["all_class_names"] == state["all_class_names"]
        np.testing.assert_array_almost_equal(loaded["protos"], state["protos"])
        np.testing.assert_array_almost_equal(loaded["text_emb"], state["text_emb"])

    def test_meta_fields_present(self, tmp_path):
        from smartqc_state import save_state, load_state
        state   = _make_state()
        outpath = str(tmp_path / "state.pkl")
        save_state(
            path=outpath,
            embeddings=state["embeddings"],
            protos=state["protos"],
            proto_classes=state["proto_classes"],
            purity_map=state["purity_map"],
            text_emb=state["text_emb"],
            all_class_names=state["all_class_names"],
            n_annotations=12,
        )
        loaded = load_state(outpath)
        meta = loaded["meta"]
        assert "saved_at"      in meta
        assert "n_annotations" in meta
        assert "n_classes"     in meta
        assert "version"       in meta
        assert meta["n_annotations"] == 12
        assert meta["n_classes"]     == 3


class TestLoadState:
    def test_returns_none_for_missing_file(self):
        from smartqc_state import load_state
        result = load_state("/nonexistent/path/state.pkl")
        assert result is None

    def test_returns_none_for_corrupted_file(self, tmp_path):
        from smartqc_state import load_state
        bad = str(tmp_path / "bad.pkl")
        with open(bad, "w") as f:
            f.write("this is not a pickle file")
        result = load_state(bad)
        assert result is None


class TestValidateState:
    def test_valid_state_passes(self):
        from smartqc_state import validate_state
        assert validate_state(_make_state()) is True

    def test_missing_key_fails(self):
        from smartqc_state import validate_state
        state = _make_state()
        del state["protos"]
        assert validate_state(state) is False

    def test_wrong_embedding_dim_fails(self):
        from smartqc_state import validate_state
        state = _make_state()
        state["embeddings"] = np.zeros((10, 512))   # wrong dim (should be 768)
        assert validate_state(state) is False

    def test_proto_class_mismatch_fails(self):
        from smartqc_state import validate_state
        state = _make_state()
        state["proto_classes"] = ["box"]             # only 1 class, protos has 3 rows
        assert validate_state(state) is False


# ── SmartQCValidator.from_state tests ────────────────────────────

class TestValidatorFromState:
    def _mock_validator_class(self):
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
            from qc_pipeline.validators.smartqc import SmartQCValidator
            return SmartQCValidator

    def test_from_state_restores_classes(self):
        SmartQCValidator = self._mock_validator_class()
        state = _make_state()
        v = SmartQCValidator.from_state(state)
        assert v._all_classes   == state["all_class_names"]
        assert v._proto_classes == state["proto_classes"]
        assert v._purity_map    == state["purity_map"]
        np.testing.assert_array_equal(v._prototypes, state["protos"])
        np.testing.assert_array_equal(v._text_emb,   state["text_emb"])

    def test_from_state_no_fit_needed(self):
        """validate() should work immediately after from_state() without fit()."""
        SmartQCValidator = self._mock_validator_class()
        state = _make_state()
        v = SmartQCValidator.from_state(state)

        # Patch the internal scorers so no models are needed
        v._score_clip = lambda crop, label: {
            "lbl_score": 0.22, "top_class": label, "top_score": 0.22,
            "mismatch": 0, "gap": 0.0, "signal": 0.33,
        }
        v._score_prototype = lambda crop, label: {
            "match": 1, "nearest_class": label,
            "label_sim": 0.65, "signal": 0.75,
        }

        ann = {"id": "1", "category_name": "box", "bbox": [5, 5, 20, 20]}
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        result = v.validate(ann, img, image_predictions=[])

        assert "qc_flag" in result
        assert result["qc_flag"] in ("ACCEPT", "REVIEW", "FLAG", "REJECT")
        assert "confidence" in result
        assert "rationale" in result


# ── QCResults tests ───────────────────────────────────────────────

class TestQCResults:
    def _make_results(self):
        from qc_pipeline.client import QCResults
        records = [
            {"annotation_id": "1", "label": "box",     "qc_flag": "ACCEPT",  "confidence": 0.80, "quality_score": 0.80},
            {"annotation_id": "2", "label": "box",     "qc_flag": "REVIEW",  "confidence": 0.55, "quality_score": 0.55},
            {"annotation_id": "3", "label": "pallet",  "qc_flag": "FLAG",    "confidence": 0.35, "quality_score": 0.35},
            {"annotation_id": "4", "label": "pallet",  "qc_flag": "REJECT",  "confidence": 0.12, "quality_score": 0.12},
            {"annotation_id": "5", "label": "forklift","qc_flag": "ACCEPT",  "confidence": 0.85, "quality_score": 0.85},
        ]
        return QCResults(records=records)

    def test_accepted(self):
        r = self._make_results()
        assert len(r.accepted()) == 2

    def test_rejected(self):
        r = self._make_results()
        assert len(r.rejected()) == 1

    def test_needs_human(self):
        r = self._make_results()
        assert len(r.needs_human()) == 3   # REVIEW + FLAG + REJECT

    def test_by_class(self):
        r = self._make_results()
        assert len(r.by_class("box"))     == 2
        assert len(r.by_class("pallet"))  == 2
        assert len(r.by_class("forklift"))== 1

    def test_worst(self):
        r = self._make_results()
        worst = r.worst(2)
        assert len(worst) == 2
        assert worst[0]["annotation_id"] == "4"   # lowest score

    def test_summary_keys(self):
        r = self._make_results()
        s = r.summary()
        for key in ("total", "accept", "review", "flag", "reject",
                    "accept_pct", "auto_handled_pct", "needs_human_pct"):
            assert key in s

    def test_summary_totals(self):
        r = self._make_results()
        s = r.summary()
        assert s["total"]  == 5
        assert s["accept"] == 2
        assert s["reject"] == 1

    def test_class_report_shape(self):
        r  = self._make_results()
        df = r.class_report()
        assert len(df) == 3   # box, pallet, forklift
        assert "reject_pct" in df.columns
        assert "mean_quality" in df.columns

    def test_export_csv(self, tmp_path):
        r   = self._make_results()
        out = str(tmp_path / "out.csv")
        r.export_csv(out)
        assert os.path.exists(out)
        import pandas as pd
        df = pd.read_csv(out)
        assert len(df) == 5

    def test_export_json(self, tmp_path):
        r   = self._make_results()
        out = str(tmp_path / "out.json")
        r.export_json(out)
        assert os.path.exists(out)
        with open(out) as f:
            data = json.load(f)
        assert data["total"] == 5
        assert len(data["results"]) == 5

    def test_len(self):
        r = self._make_results()
        assert len(r) == 5

    def test_repr(self):
        r    = self._make_results()
        s    = repr(r)
        assert "QCResults" in s
        assert "total=5"   in s


# ── SmartQCClient smoke test ──────────────────────────────────────

class TestSmartQCClientSmoke:
    """Smoke test — verifies the client wires things together correctly."""

    def test_run_returns_qc_results(self, tmp_path):
        from qc_pipeline.client import SmartQCClient, QCResults
        from qc_pipeline.smartqc_workflow import SmartQCWorkflow, SmartQCSummary, SmartQCState

        coco_path, img_dir = _make_coco_json(str(tmp_path))
        out_dir = str(tmp_path / "output")

        # Mock the workflow so no real models run
        mock_records = [
            {"annotation_id": str(i), "label": "box", "qc_flag": "ACCEPT",
             "confidence": 0.80, "quality_score": 0.80, "is_match": True, "rationale": "ok"}
            for i in range(6)
        ]
        mock_summary = SmartQCSummary(
            total=6, matches=6, mismatches=0, flags=0, rejects=0, reviews=0,
            average_confidence=0.80, mean_clip_score=0.24,
            clip_mismatches=0, proto_match_rate=0.90, mean_iou=0.75, false_positives=0
        )

        with patch.object(SmartQCWorkflow, "run",
                          return_value=(SmartQCState(), mock_records, mock_summary)):
            client = SmartQCClient(
                coco_json_path=coco_path,
                images_dir=img_dir,
                output_dir=out_dir,
            )
            results = client.run()

        assert isinstance(results, QCResults)
        assert len(results) == 6
        assert results.summary()["total"] == 6

    def test_state_saved_on_first_run(self, tmp_path):
        """When state_path is set and no state exists, state should be saved."""
        from qc_pipeline.client import SmartQCClient
        from qc_pipeline.smartqc_workflow import SmartQCWorkflow, SmartQCSummary, SmartQCState
        from qc_pipeline.validators.smartqc import SmartQCValidator

        coco_path, img_dir = _make_coco_json(str(tmp_path))
        out_dir    = str(tmp_path / "output")
        state_path = str(tmp_path / "state.pkl")

        mock_records = [
            {"annotation_id": "1", "label": "box", "qc_flag": "ACCEPT",
             "confidence": 0.80, "quality_score": 0.80, "is_match": True, "rationale": "ok"}
        ]
        mock_summary = SmartQCSummary(total=1, matches=1)
        mock_validator = MagicMock(spec=SmartQCValidator)
        mock_validator._prototypes    = np.eye(3, 768, dtype=np.float32)
        mock_validator._proto_classes = ["box", "pallet", "forklift"]
        mock_validator._purity_map    = {"box": 0.6, "pallet": 0.4, "forklift": 0.7}
        mock_validator._text_emb      = np.eye(3, 512, dtype=np.float32)
        mock_validator._all_classes   = ["box", "pallet", "forklift"]
        mock_validator.save_state     = MagicMock()

        with (
            patch.object(SmartQCWorkflow, "run",
                         return_value=(SmartQCState(), mock_records, mock_summary)),
            patch.object(SmartQCWorkflow, "__init__", return_value=None),
        ):
            from qc_pipeline import smartqc_workflow
            client = SmartQCClient(
                coco_json_path=coco_path,
                images_dir=img_dir,
                output_dir=out_dir,
                state_path=state_path,
            )
            # Manually set workflow's validator for the test
            import qc_pipeline.smartqc_workflow as wf_mod
            original_workflow_class = wf_mod.SmartQCWorkflow

            class PatchedWorkflow(original_workflow_class):
                def __init__(self, **kwargs):
                    self._validator = mock_validator
                    self.output_dir = out_dir

                def run(self, **kwargs):
                    return SmartQCState(), mock_records, mock_summary

            with patch("qc_pipeline.client.SmartQCWorkflow", PatchedWorkflow):
                results = client.run()

        # save_state should have been called because state_path was set
        mock_validator.save_state.assert_called_once_with(state_path)
