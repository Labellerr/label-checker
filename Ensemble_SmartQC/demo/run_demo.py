"""
SmartQC — Minimal Demo
======================
Run this to see SmartQC in action on a tiny synthetic dataset.
No real images or GPU needed — generates dummy data and runs the
CLIP + DINOv2 + IoU pipeline end-to-end.

Usage:
    cd smartqc/
    python demo/run_demo.py

What it does:
    1. Creates 5 synthetic 64x64 RGB images in a temp folder
    2. Creates a minimal COCO JSON annotation file
    3. Runs SmartQCWorkflow (fit + validate)
    4. Prints results to terminal
    5. Cleans up temp files
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

# Add parent folder to path so imports work when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def make_synthetic_dataset(tmpdir: str, n_images: int = 3, n_ann_each: int = 3):
    """Create dummy images and a COCO JSON file in tmpdir."""
    img_dir = os.path.join(tmpdir, "images")
    os.makedirs(img_dir, exist_ok=True)

    categories = [
        {"id": 1, "name": "box"},
        {"id": 2, "name": "forklift"},
        {"id": 3, "name": "cone"},
    ]

    images, annotations = [], []
    ann_id = 1

    for i in range(1, n_images + 1):
        fname = f"img_{i:03d}.jpg"
        # Coloured synthetic image
        colour = [
            (180, 120, 80),   # box-ish brown
            (240, 200, 30),   # forklift yellow
            (220, 90,  30),   # cone orange
        ][i % 3]
        arr = np.full((128, 128, 3), colour, dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(img_dir, fname))
        images.append({"id": i, "file_name": fname, "width": 128, "height": 128})

        for j in range(n_ann_each):
            annotations.append({
                "id":          ann_id,
                "image_id":    i,
                "category_id": 1 + (j % 3),
                "bbox":        [10 + j * 15, 10, 40, 40],
            })
            ann_id += 1

    coco_path = os.path.join(tmpdir, "annotations.json")
    with open(coco_path, "w") as f:
        json.dump({
            "images":      images,
            "annotations": annotations,
            "categories":  categories,
        }, f, indent=2)

    return coco_path, img_dir


def mock_validator():
    """
    Return a SmartQCValidator with all heavy model calls mocked.
    This lets the demo run without a GPU or model downloads.
    """
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

    # Inject synthetic prototypes
    import numpy as np
    v._all_classes   = ["box", "forklift", "cone"]
    v._proto_classes = ["box", "forklift", "cone"]
    v._prototypes    = np.eye(3, 768, dtype=np.float32)
    v._text_emb      = np.eye(3, 512, dtype=np.float32)
    v._purity_map    = {"box": 0.60, "forklift": 0.75, "cone": 0.55}

    # Deterministic scoring
    v._embed_crop = lambda crop: np.array([1.0] + [0.0] * 767, dtype=np.float32)
    v._score_clip = lambda crop, label: {
        "lbl_score": 0.24, "top_class": label, "top_score": 0.24,
        "mismatch": 0, "gap": 0.0, "signal": 0.50,
    }
    v._score_prototype = lambda crop, label: {
        "match": 1, "nearest_class": label,
        "label_sim": 0.75, "signal": 0.80,
    }
    v.fit = MagicMock(return_value=v)
    return v


def main():
    print("\n" + "="*60)
    print("  SmartQC — Demo Run")
    print("  (Mocked models — no GPU or downloads needed)")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = os.path.join(tmpdir, "output")

        print("\n[1/3] Creating synthetic dataset...")
        coco_path, img_dir = make_synthetic_dataset(tmpdir)
        print(f"      → {coco_path}")

        print("\n[2/3] Running SmartQCWorkflow...")
        from qc_pipeline.smartqc_workflow import SmartQCWorkflow
        workflow = SmartQCWorkflow(output_dir=out_dir)
        workflow._validator = mock_validator()

        state, results, summary = workflow.run(
            coco_json_path=coco_path,
            images_dir=img_dir,
        )

        print("\n[3/3] Results:\n")
        print(f"  Total annotations : {summary.total}")
        print(f"  ACCEPT            : {summary.matches}  (auto-cleared)")
        print(f"  REVIEW+FLAG+REJECT: {summary.mismatches}")
        print(f"  Avg quality score : {summary.average_confidence:.3f}")
        print()
        print("  Per-annotation verdicts:")
        print(f"  {'ID':<12} {'Class':<18} {'Verdict':<8} {'Score'}")
        print("  " + "-"*52)
        for r in results[:10]:
            print(f"  {str(r['annotation_id']):<12} "
                  f"{r['label']:<18} "
                  f"{r['qc_flag']:<8} "
                  f"{r.get('quality_score', r.get('confidence', 0)):.3f}")
        if len(results) > 10:
            print(f"  ... and {len(results)-10} more")

        print("\n  Output files:")
        for f in ["qc_results.json", "qc_results.csv"]:
            path = os.path.join(out_dir, f)
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"  {f}  ({size:,} bytes)")

    print("\n" + "="*60)
    print("  Demo complete.")
    print("  To run on real data: edit DATASET_PATH in smartqc_fixed.py")
    print("  and run: python smartqc_fixed.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
