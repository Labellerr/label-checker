"""
smartqc_state.py — Save and Load Fitted Pipeline State
=======================================================
Saves the expensive-to-compute parts of the SmartQC pipeline to a .pkl
file so that subsequent runs skip re-computing DINOv2 embeddings and
prototypes entirely (saves ~15 of the 24-minute runtime).

What gets saved
---------------
- DINOv2 embeddings array     (shape: N_annotations × 768)
- Class prototype matrix      (shape: N_classes × 768)
- Prototype class list        (list of class name strings)
- Purity map                  (dict: class_name → float)
- CLIP text embeddings        (shape: N_classes × 512)
- All class names             (ordered list)
- QC thresholds               (snapshot of config at save time)
- Timestamp + dataset stats   (for traceability)

Usage — saving after a run
--------------------------
    from smartqc_state import save_state, load_state

    # After running the pipeline (all variables come from smartqc_fixed.py)
    save_state(
        path="qc_output/smartqc_state.pkl",
        embeddings=embeddings,
        protos=protos,
        proto_classes=proto_classes,
        purity_map=purity_map,
        text_emb=text_emb,           # from run_clip_validation()
        all_class_names=all_class_names,
        n_annotations=len(df),
    )

Usage — loading to skip fit() on a repeat run
---------------------------------------------
    state = load_state("qc_output/smartqc_state.pkl")

    # state keys:
    #   embeddings, protos, proto_classes, purity_map,
    #   text_emb, all_class_names, thresholds, meta

    # Restore into SmartQCValidator without re-running fit():
    from qc_pipeline.validators.smartqc import SmartQCValidator

    validator = SmartQCValidator.from_state(state)
    result = validator.validate(annotation, image, image_predictions)

Usage — loading into the main pipeline (smartqc_fixed.py)
----------------------------------------------------------
    # At the top of the pipeline's main block, add:
    state = load_state("qc_output/smartqc_state.pkl")
    if state:
        embeddings    = state["embeddings"]
        protos        = state["protos"]
        proto_classes = state["proto_classes"]
        purity_map    = state["purity_map"]
        text_emb      = state["text_emb"]
        print("  ✅ Loaded from cache — skipped Steps 4 and 5")
    else:
        # Run Steps 4 and 5 normally, then save
        embeddings, valid = extract_embeddings(df, dinov2, dinov2_proc)
        df, protos, proto_classes, purity_map = assign_prototypes(df, embeddings)
        text_emb = run_clip_text_embeddings(clip, clip_proc, all_class_names)
        save_state("qc_output/smartqc_state.pkl", embeddings, protos,
                   proto_classes, purity_map, text_emb, all_class_names, len(df))
"""

from __future__ import annotations

import os
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


# ── QC thresholds (mirrors smartqc_fixed.py defaults) ─────────────
_DEFAULT_THRESHOLDS = {
    "QUALITY_ACCEPT":  0.72,
    "QUALITY_REVIEW":  0.45,
    "QUALITY_REJECT":  0.20,
    "IOU_THRESHOLD":   0.50,
    "CONF_THRESHOLD":  0.40,
}


def save_state(
    path: str,
    embeddings: np.ndarray,
    protos: np.ndarray,
    proto_classes: List[str],
    purity_map: Dict[str, float],
    text_emb: np.ndarray,
    all_class_names: List[str],
    n_annotations: int,
    thresholds: Optional[Dict[str, float]] = None,
) -> str:
    """
    Pickle the fitted SmartQC pipeline state to disk.

    Parameters
    ----------
    path : str
        Output path for the .pkl file (e.g. ``"qc_output/smartqc_state.pkl"``).
    embeddings : np.ndarray
        DINOv2 embedding matrix, shape (N_annotations, 768).
    protos : np.ndarray
        Class prototype matrix, shape (N_classes, 768). L2-normalised.
    proto_classes : list of str
        Class names corresponding to rows of ``protos``.
    purity_map : dict
        Maps class name → prototype purity float [0, 1].
    text_emb : np.ndarray
        CLIP text embeddings for all classes, shape (N_classes, 512).
    all_class_names : list of str
        Ordered class names (same order as ``text_emb`` rows).
    n_annotations : int
        Number of annotations in the dataset (for traceability).
    thresholds : dict or None
        QC thresholds to snapshot. Defaults to the standard SmartQC values.

    Returns
    -------
    str
        Absolute path to the saved file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    state = {
        # Core fitted data
        "embeddings":     embeddings,
        "protos":         protos,
        "proto_classes":  proto_classes,
        "purity_map":     purity_map,
        "text_emb":       text_emb,
        "all_class_names": all_class_names,

        # Config snapshot
        "thresholds": thresholds or _DEFAULT_THRESHOLDS,

        # Traceability metadata
        "meta": {
            "saved_at":      datetime.now().isoformat(timespec="seconds"),
            "n_annotations": n_annotations,
            "n_classes":     len(proto_classes),
            "emb_shape":     list(embeddings.shape),
            "proto_shape":   list(protos.shape),
            "text_emb_shape": list(text_emb.shape),
            "version":       "1.0",
        },
    }

    with open(path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(path) / 1_048_576
    print(f"\n  💾 State saved → {path}")
    print(f"     Embeddings : {embeddings.shape}  ({size_mb:.1f} MB total)")
    print(f"     Classes    : {len(proto_classes)}")
    print(f"     Purity map : {len(purity_map)} entries")
    print(f"     Saved at   : {state['meta']['saved_at']}")
    print(f"\n  ✅ Next run will load from cache and skip Steps 4–5 (~15 min saved).")

    return os.path.abspath(path)


def load_state(path: str) -> Optional[Dict[str, Any]]:
    """
    Load a previously saved SmartQC state.

    Returns ``None`` (with a warning) if the file doesn't exist or is
    corrupted — so the caller can fall back to running fit() normally.

    Parameters
    ----------
    path : str
        Path to the .pkl file produced by ``save_state()``.

    Returns
    -------
    dict or None
        State dict with keys: ``embeddings``, ``protos``, ``proto_classes``,
        ``purity_map``, ``text_emb``, ``all_class_names``, ``thresholds``, ``meta``.
    """
    if not os.path.exists(path):
        print(f"  ℹ️  No cached state at '{path}' — will compute from scratch.")
        return None

    try:
        t0 = time.time()
        with open(path, "rb") as f:
            state = pickle.load(f)

        meta     = state.get("meta", {})
        elapsed  = time.time() - t0
        size_mb  = os.path.getsize(path) / 1_048_576

        print(f"\n  📂 State loaded from cache → {path}")
        print(f"     Saved at     : {meta.get('saved_at', 'unknown')}")
        print(f"     Annotations  : {meta.get('n_annotations', '?'):,}")
        print(f"     Classes      : {meta.get('n_classes', '?')}")
        print(f"     Emb shape    : {meta.get('emb_shape', '?')}")
        print(f"     File size    : {size_mb:.1f} MB")
        print(f"     Load time    : {elapsed:.2f}s")
        print(f"\n  ✅ Steps 4 and 5 skipped (DINOv2 embeddings + prototypes restored).")

        return state

    except Exception as exc:
        print(f"  ⚠️  Failed to load state from '{path}': {exc}")
        print(f"      Will recompute from scratch.")
        return None


def state_info(path: str) -> None:
    """
    Print a human-readable summary of a saved state file without loading
    the full embedding arrays.

    Parameters
    ----------
    path : str
        Path to the .pkl file.
    """
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    size_mb = os.path.getsize(path) / 1_048_576
    with open(path, "rb") as f:
        state = pickle.load(f)

    meta      = state.get("meta", {})
    purity    = state.get("purity_map", {})
    thresholds = state.get("thresholds", {})
    classes   = state.get("proto_classes", [])

    print(f"\n{'='*55}")
    print(f"  SmartQC State File: {os.path.basename(path)}")
    print(f"{'='*55}")
    print(f"  Saved at         : {meta.get('saved_at', '?')}")
    print(f"  Version          : {meta.get('version', '?')}")
    print(f"  File size        : {size_mb:.1f} MB")
    print(f"  Annotations      : {meta.get('n_annotations', '?'):,}")
    print(f"  Classes          : {meta.get('n_classes', '?')}")
    print(f"  Embedding shape  : {meta.get('emb_shape', '?')}")
    print(f"  Prototype shape  : {meta.get('proto_shape', '?')}")
    print(f"  Text emb shape   : {meta.get('text_emb_shape', '?')}")
    print(f"\n  QC Thresholds:")
    for k, v in thresholds.items():
        print(f"    {k:<22}: {v}")
    print(f"\n  Class Purity:")
    for cls in sorted(classes):
        p = purity.get(cls, 0.0)
        tier = "GOOD" if p >= 0.50 else "FAIR" if p >= 0.25 else "POOR"
        bar  = "█" * int(p * 20) + "░" * (20 - int(p * 20))
        print(f"    {cls:<22} [{bar}] {p*100:5.1f}%  [{tier}]")
    print(f"{'='*55}\n")


def validate_state(state: Dict[str, Any]) -> bool:
    """
    Check that all required keys are present and shapes are consistent.

    Parameters
    ----------
    state : dict
        State dict returned by ``load_state()``.

    Returns
    -------
    bool
        ``True`` if valid, ``False`` if something is missing or mismatched.
    """
    required = ["embeddings", "protos", "proto_classes",
                "purity_map", "text_emb", "all_class_names"]

    for key in required:
        if key not in state:
            print(f"  ❌ State missing required key: '{key}'")
            return False

    emb    = state["embeddings"]
    protos = state["protos"]
    t_emb  = state["text_emb"]
    cls    = state["proto_classes"]

    if emb.ndim != 2 or emb.shape[1] != 768:
        print(f"  ❌ embeddings shape {emb.shape} invalid (expected N×768)")
        return False
    if protos.ndim != 2 or protos.shape[1] != 768:
        print(f"  ❌ protos shape {protos.shape} invalid (expected C×768)")
        return False
    if protos.shape[0] != len(cls):
        print(f"  ❌ protos rows ({protos.shape[0]}) ≠ proto_classes length ({len(cls)})")
        return False
    if t_emb.ndim != 2 or t_emb.shape[1] != 512:
        print(f"  ❌ text_emb shape {t_emb.shape} invalid (expected C×512)")
        return False

    print("  ✅ State is valid.")
    return True


# ── CLI usage ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        print("\nCLI usage:")
        print("  python smartqc_state.py info  <path/to/smartqc_state.pkl>")
        print("  python smartqc_state.py check <path/to/smartqc_state.pkl>")
        sys.exit(0)

    cmd  = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else "qc_output/smartqc_state.pkl"

    if cmd == "info":
        state_info(path)
    elif cmd == "check":
        state = load_state(path)
        if state:
            validate_state(state)
    else:
        print(f"Unknown command '{cmd}'. Use 'info' or 'check'.")
