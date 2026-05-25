"""
qc_pipeline/client.py — SmartQCClient
======================================
A simple, beginner-friendly entry point to SmartQC.

If you are new to this project, start here. You don't need to know
how CLIP, DINOv2, or YOLOv8 work internally. Just point the client
at your dataset and call ``run()``.

Quickstart
----------
::

    from qc_pipeline.client import SmartQCClient

    client = SmartQCClient(
        coco_json_path="dataset/annotations.json",
        images_dir="dataset/images/",
        output_dir="qc_output/",
        yolo_model_path="models/best.pt",   # optional
    )
    results = client.run()

    # Inspect results
    print(results.summary())
    print(results.rejected())          # all REJECT annotations
    print(results.by_class("wire"))    # all annotations for a specific class
    results.export_csv("my_results.csv")

With caching (skip re-computing embeddings on repeat runs)
-----------------------------------------------------------
::

    client = SmartQCClient(
        coco_json_path="dataset/annotations.json",
        images_dir="dataset/images/",
        state_path="qc_output/smartqc_state.pkl",  # auto-saved + loaded
    )
    results = client.run()   # first run: ~24 min. Second run: ~2 min.

With the Labellerr platform
-----------------------------
::

    client = SmartQCClient.from_labellerr(
        api_key="your_labellerr_api_key",
        project_id="your_project_id",
        images_dir="dataset/images/",
    )
    results = client.run()
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image


# ── Results container ─────────────────────────────────────────────

@dataclass
class QCResults:
    """
    Holds all SmartQC results for a dataset run.
    Returned by ``SmartQCClient.run()``.
    """
    records:     List[Dict[str, Any]] = field(default_factory=list)
    _summary_cache: Optional[Dict] = field(default=None, repr=False)

    # ── Filtering helpers ─────────────────────────────────────────

    def accepted(self) -> List[Dict]:
        """All ACCEPT annotations — auto-cleared, no human review needed."""
        return [r for r in self.records if r.get("qc_flag") == "ACCEPT"]

    def to_review(self) -> List[Dict]:
        """All REVIEW annotations — borderline, quick human check."""
        return [r for r in self.records if r.get("qc_flag") == "REVIEW"]

    def flagged(self) -> List[Dict]:
        """All FLAG annotations — suspicious, careful human review."""
        return [r for r in self.records if r.get("qc_flag") == "FLAG"]

    def rejected(self) -> List[Dict]:
        """All REJECT annotations — clearly wrong, discard or re-annotate."""
        return [r for r in self.records if r.get("qc_flag") == "REJECT"]

    def needs_human(self) -> List[Dict]:
        """Everything that needs a human: REVIEW + FLAG + REJECT."""
        return [r for r in self.records
                if r.get("qc_flag") in ("REVIEW", "FLAG", "REJECT")]

    def by_class(self, class_name: str) -> List[Dict]:
        """All annotations for a specific class, e.g. ``by_class("wire")``."""
        return [r for r in self.records
                if r.get("label") == class_name or r.get("category_name") == class_name]

    def by_verdict(self, verdict: str) -> List[Dict]:
        """All annotations with a specific verdict string."""
        return [r for r in self.records if r.get("qc_flag") == verdict.upper()]

    def worst(self, n: int = 20) -> List[Dict]:
        """The n lowest-quality annotations (highest risk, sorted ascending)."""
        return sorted(self.records,
                      key=lambda r: r.get("confidence", r.get("quality_score", 1.0)))[:n]

    # ── Summary ───────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Dataset-level statistics dict."""
        if self._summary_cache:
            return self._summary_cache

        total  = len(self.records)
        if total == 0:
            return {"total": 0}

        counts = {}
        for v in ("ACCEPT", "REVIEW", "FLAG", "REJECT"):
            counts[v] = sum(1 for r in self.records if r.get("qc_flag") == v)

        scores = [r.get("confidence", r.get("quality_score", 0.0)) for r in self.records]

        s = {
            "total":              total,
            "accept":             counts["ACCEPT"],
            "review":             counts["REVIEW"],
            "flag":               counts["FLAG"],
            "reject":             counts["REJECT"],
            "accept_pct":         round(counts["ACCEPT"] / total * 100, 1),
            "auto_handled_pct":   round((counts["ACCEPT"] + counts["REJECT"]) / total * 100, 1),
            "needs_human_pct":    round((counts["REVIEW"] + counts["FLAG"]) / total * 100, 1),
            "mean_quality_score": round(float(np.mean(scores)), 4),
        }

        clip_mm = sum(r.get("clip_mismatch", 0) for r in self.records)
        fps     = sum(r.get("is_false_positive", 0) for r in self.records)
        if clip_mm:  s["clip_mismatches"]  = int(clip_mm)
        if fps:      s["false_positives"]  = int(fps)

        self._summary_cache = s
        return s

    def print_summary(self) -> None:
        """Print a formatted summary table to stdout."""
        s = self.summary()
        total = s["total"]
        print(f"\n{'='*55}")
        print(f"  SmartQC Results Summary")
        print(f"{'='*55}")
        print(f"  Total annotations  : {total:,}")
        print(f"  ACCEPT          : {s['accept']:>6,}  ({s['accept_pct']:.1f}%)")
        print(f"  REVIEW          : {s['review']:>6,}  ({s['review']/(total or 1)*100:.1f}%)")
        print(f"  FLAG            : {s['flag']:>6,}  ({s['flag']/(total or 1)*100:.1f}%)")
        print(f"  REJECT          : {s['reject']:>6,}  ({s['reject']/(total or 1)*100:.1f}%)")
        print(f"  ─────────────────────────────────────────────────")
        print(f"  Auto-handled       : {s['auto_handled_pct']:.1f}%  (no human needed)")
        print(f"  Needs human        : {s['needs_human_pct']:.1f}%")
        print(f"  Mean quality score : {s['mean_quality_score']:.3f}")
        if "clip_mismatches"  in s: print(f"  Label mismatches   : {s['clip_mismatches']:,}")
        if "false_positives"  in s: print(f"  False positives    : {s['false_positives']:,}")
        print(f"{'='*55}\n")

    def class_report(self) -> pd.DataFrame:
        """
        Per-class breakdown as a pandas DataFrame.
        Columns: class, total, accept, review, flag, reject,
                 accept_pct, reject_pct, mean_quality.
        """
        from collections import defaultdict
        data: Dict[str, Dict] = defaultdict(lambda: {
            "total": 0, "ACCEPT": 0, "REVIEW": 0, "FLAG": 0, "REJECT": 0, "scores": []
        })
        for r in self.records:
            cls   = r.get("label") or r.get("category_name") or "unknown"
            flag  = r.get("qc_flag", "FLAG")
            score = r.get("confidence", r.get("quality_score", 0.0))
            data[cls]["total"] += 1
            data[cls][flag]    += 1
            data[cls]["scores"].append(score)

        rows = []
        for cls, d in sorted(data.items()):
            t = d["total"]
            rows.append({
                "class":        cls,
                "total":        t,
                "accept":       d["ACCEPT"],
                "review":       d["REVIEW"],
                "flag":         d["FLAG"],
                "reject":       d["REJECT"],
                "accept_pct":   round(d["ACCEPT"] / t * 100, 1),
                "reject_pct":   round(d["REJECT"] / t * 100, 1),
                "mean_quality": round(float(np.mean(d["scores"])), 4),
            })
        rows.sort(key=lambda r: r["reject_pct"], reverse=True)
        return pd.DataFrame(rows)

    # ── Export ────────────────────────────────────────────────────

    def export_csv(self, path: str) -> str:
        """Write all results to a CSV file. Returns the absolute path."""
        pd.DataFrame(self.records).to_csv(path, index=False)
        print(f"  Exported {len(self.records):,} rows → {os.path.abspath(path)}")
        return os.path.abspath(path)

    def export_json(self, path: str) -> str:
        """Write all results to a JSON file. Returns the absolute path."""
        with open(path, "w") as f:
            json.dump({"total": len(self.records), "results": self.records}, f, indent=2)
        print(f"  Exported {len(self.records):,} records → {os.path.abspath(path)}")
        return os.path.abspath(path)

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        s = self.summary()
        return (f"QCResults(total={s['total']}, accept={s['accept']}, "
                f"review={s['review']}, flag={s['flag']}, reject={s['reject']})")


# ── SmartQCClient ─────────────────────────────────────────────────

class SmartQCClient:
    """
    Beginner-friendly entry point to SmartQC.

    One object. One ``run()`` call. Everything else is handled.

    Parameters
    ----------
    coco_json_path : str
        Path to COCO annotations JSON file.
    images_dir : str
        Directory containing the images referenced in the JSON.
    output_dir : str
        Where to write CSV, JSON, and visual outputs. Created if needed.
    yolo_model_path : str or None
        Path to a trained YOLOv8 ``best.pt``.
        Optional — omit to use CLIP + DINOv2 only (no boundary signal).
    state_path : str or None
        Path to a ``.pkl`` state file.
        On first run: state is saved here after fitting.
        On subsequent runs: state is loaded and fit() is skipped (~15 min saved).
    device : str or None
        ``"cuda"`` or ``"cpu"``. Auto-detected if not provided.
    pdf_path : str or None
        Path to a labelling guidelines PDF (optional). Text is extracted
        and used to enrich QC rationale strings.
    max_annotations : int or None
        Cap the number of annotations processed (useful for testing).

    Examples
    --------
    Minimal::

        client = SmartQCClient(
            coco_json_path="annotations.json",
            images_dir="images/",
        )
        results = client.run()
        results.print_summary()

    With caching::

        client = SmartQCClient(
            coco_json_path="annotations.json",
            images_dir="images/",
            state_path="qc_output/smartqc_state.pkl",
        )
        results = client.run()

    With everything::

        client = SmartQCClient(
            coco_json_path="annotations.json",
            images_dir="images/",
            output_dir="qc_output/",
            yolo_model_path="models/best.pt",
            state_path="qc_output/smartqc_state.pkl",
            pdf_path="guidelines.pdf",
        )
        results = client.run()
        results.print_summary()
        results.export_csv("results.csv")
        df = results.class_report()
        print(df.head(10))
    """

    def __init__(
        self,
        coco_json_path: str,
        images_dir: str,
        output_dir: str = "qc_output",
        yolo_model_path: Optional[str] = None,
        state_path: Optional[str] = None,
        device: Optional[str] = None,
        pdf_path: Optional[str] = None,
        max_annotations: Optional[int] = None,
    ) -> None:
        self.coco_json_path   = coco_json_path
        self.images_dir       = images_dir
        self.output_dir       = output_dir
        self.yolo_model_path  = yolo_model_path
        self.state_path       = state_path
        self.device           = device
        self.pdf_path         = pdf_path
        self.max_annotations  = max_annotations

        os.makedirs(output_dir, exist_ok=True)

    @classmethod
    def from_labellerr(
        cls,
        api_key: str,
        project_id: str,
        images_dir: str,
        batch_id: Optional[str] = None,
        **kwargs,
    ) -> "SmartQCClient":
        """
        Create a client that loads annotations from the Labellerr platform.

        Parameters
        ----------
        api_key : str
            Your Labellerr API key.
        project_id : str
            The project to fetch annotations from.
        images_dir : str
            Local directory containing the images.
        batch_id : str or None
            Optional batch ID to fetch a specific batch.
        **kwargs
            Forwarded to ``SmartQCClient.__init__`` (output_dir, yolo_model_path, etc.)

        Example
        -------
        ::

            client = SmartQCClient.from_labellerr(
                api_key="YOUR_KEY",
                project_id="proj_abc123",
                images_dir="images/",
                output_dir="qc_output/",
            )
            results = client.run()
        """
        from qc_pipeline.fetchers.labellerr import load_from_labellerr
        import tempfile

        # Fetch from Labellerr and write to a temp COCO JSON
        annotations = load_from_labellerr(
            api_key=api_key,
            project_id=project_id,
            batch_id=batch_id,
        )

        # Build a minimal COCO JSON from the fetched annotations
        tmpdir       = tempfile.mkdtemp()
        coco_path    = os.path.join(tmpdir, "annotations.json")
        image_ids    = sorted({a["image_id"] for a in annotations})
        cat_names    = sorted({a["category_name"] for a in annotations})
        cat_id_map   = {name: i + 1 for i, name in enumerate(cat_names)}

        coco = {
            "images": [{"id": img_id, "file_name": img_id} for img_id in image_ids],
            "categories": [{"id": v, "name": k} for k, v in cat_id_map.items()],
            "annotations": [
                {
                    "id":          a["id"],
                    "image_id":    a["image_id"],
                    "category_id": cat_id_map.get(a["category_name"], 0),
                    "bbox":        a["bbox"],
                }
                for a in annotations
            ],
        }
        with open(coco_path, "w") as f:
            json.dump(coco, f)

        print(f"  Labellerr: fetched {len(annotations):,} annotations → {coco_path}")
        return cls(coco_json_path=coco_path, images_dir=images_dir, **kwargs)

    # ── Main entry point ──────────────────────────────────────────

    def run(self) -> QCResults:
        """
        Run the full SmartQC pipeline and return results.

        On first call: fits prototypes and saves state (if state_path set).
        On subsequent calls: loads state from cache and skips fitting.

        Returns
        -------
        QCResults
            Filterable, exportable results container.
        """
        t_start = time.time()
        print(f"\n{'='*55}")
        print(f"  SmartQC Client — starting pipeline")
        print(f"{'='*55}")

        # ── 1. Load state or fit from scratch ─────────────────────
        from qc_pipeline.smartqc_workflow import SmartQCWorkflow
        from smartqc_state import load_state

        state    = load_state(self.state_path) if self.state_path else None
        workflow = SmartQCWorkflow(
            output_dir=self.output_dir,
            yolo_model_path=self.yolo_model_path,
            device=self.device,
        )

        if state is not None:
            from qc_pipeline.validators.smartqc import SmartQCValidator
            workflow._validator = SmartQCValidator.from_state(
                state,
                yolo_model_path=self.yolo_model_path,
                device=self.device,
            )
            print(" Loaded from cached state — fit() skipped.")

        # ── 2. Run pipeline ────────────────────────────────────────
        _, records, summary = workflow.run(
            coco_json_path=self.coco_json_path,
            images_dir=self.images_dir,
            pdf_path=self.pdf_path,
            max_files=self.max_annotations,
        )

        # ── 3. Save state if requested and not loaded from cache ───
        if self.state_path and state is None:
            try:
                workflow._validator.save_state(self.state_path)
            except Exception as exc:
                print(f"  ⚠️  Could not save state: {exc}")

        # ── 4. Write summary JSON ──────────────────────────────────
        summary_path = os.path.join(self.output_dir, "qc_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "total":    summary.total,
                "matches":  summary.matches,
                "reviews":  summary.reviews,
                "flags":    summary.flags,
                "rejects":  summary.rejects,
                "average_confidence": summary.average_confidence,
                "mean_clip_score":    summary.mean_clip_score,
                "clip_mismatches":    summary.clip_mismatches,
                "mean_iou":           summary.mean_iou,
                "false_positives":    summary.false_positives,
            }, f, indent=2)

        elapsed = time.time() - t_start
        results = QCResults(records=records)
        results.print_summary()
        print(f"  Total runtime: {elapsed/60:.1f} min")
        print(f"  Outputs in  : {os.path.abspath(self.output_dir)}\n")
        return results

    def __repr__(self) -> str:
        return (
            f"SmartQCClient(\n"
            f"  coco_json_path = {self.coco_json_path!r}\n"
            f"  images_dir     = {self.images_dir!r}\n"
            f"  output_dir     = {self.output_dir!r}\n"
            f"  state_path     = {self.state_path!r}\n"
            f"  yolo_model_path= {self.yolo_model_path!r}\n"
            f")"
        )
