# SmartQC — Intelligent Annotation Quality Control

> **New to this project?** Read [ARCHITECTURE.md](./ARCHITECTURE.md) first — it explains everything in plain English, no AI background needed.

SmartQC is a three-signal annotation quality control engine that automatically audits every annotation in an object detection dataset and assigns a verdict: **ACCEPT**, **REVIEW**, **FLAG**, or **REJECT**.

It was built as an ISB AMPBA Capstone 2026 project with Labellerr AI (Tensor Matics Inc.) and is designed to plug directly into this `label-checker` framework.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Point it at your dataset

Edit the paths at the top of `smartqc_fixed.py`:

```python
DATASET_PATH   = "/path/to/your/dataset"   # ← change this
YOLO_IMAGE_DIR = os.path.join(DATASET_PATH, "yolo", "images")
YOLO_LABEL_DIR = os.path.join(DATASET_PATH, "yolo", "labels")
CLASS_MAP_FILE = os.path.join(DATASET_PATH, "class_mapping.json")
```

### 3. Run the pipeline

```bash
python smartqc_fixed.py
```

Results appear in `qc_output/` — CSV, JSON summary, and visual crop folders.

### 4. Query results via API

```bash
python smartqc_api.py
# Open http://localhost:8000/qc/summary
```

---

## Use SmartQC inside label-checker

SmartQC is registered as a provider in the `create_validator()` factory — no API key needed.

```python
from qc_pipeline.validators import create_validator

# Drop-in replacement for Gemini/OpenAI/Anthropic validators
validator = create_validator(
    provider="smartqc",
    yolo_model_path="path/to/best.pt"   # optional — omit for CLIP+DINOv2 only
)
```

Or run the full workflow:

```python
from qc_pipeline.smartqc_workflow import SmartQCWorkflow

workflow = SmartQCWorkflow(output_dir="output")
state, results, summary = workflow.run(
    coco_json_path="annotations.json",
    images_dir="images/",
    pdf_path="labelling_guidelines.pdf"   # optional
)

print(f"Accepted:  {summary.matches}/{summary.total}")
print(f"Precision: {summary.average_confidence:.1%}")
```

---

## How it works — the short version

Three AI models run in parallel on every annotation:

| Signal | Model | Checks | Weight |
|--------|-------|--------|--------|
| Semantic | CLIP ViT-B/32 | Does the image crop match the label? | 45–80% |
| Prototype | DINOv2-base | Does it look like a typical example of its class? | 0–35% |
| Boundary | YOLOv8 (trained) | Is the bounding box accurately placed? | 20% (fixed) |

Weights adapt per class based on **prototype purity** — how visually separable each class is. For visually ambiguous classes (wire, bracket), DINOv2 is unreliable so its weight drops to 0% and CLIP carries 80%. For distinctive classes (forklift, fire_extinguisher), DINOv2 is trustworthy and earns 35%.

**→ For the full explanation see [ARCHITECTURE.md](./ARCHITECTURE.md)**

---

## Results on the warehouse dataset

| Metric | Value |
|--------|-------|
| Total annotations | 39,182 |
| Flagging precision | **95.83%** |
| Throughput | **5,000 / min** on GPU (~24 min end-to-end) |
| Auto-handled (no human needed) | **26.9%** |
| Label mismatches caught | 3,143 |
| False positives caught | 1,519 |
| Missed objects identified | 15,298 |

---

## Files in this folder

```
smartqc/
├── smartqc_fixed.py          ← Main pipeline — run this
├── smartqc_api.py            ← REST API server (6 endpoints)
├── smartqc_state.py          ← Save/load fitted state as .pkl (skips ~15 min on repeat runs)
├── requirements.txt          ← Python dependencies
├── class_mapping.json        ← Example class map (replace with yours)
├── README.md                 ← This file
├── ARCHITECTURE.md           ← Full system explanation (start here if new)
│
├── qc_pipeline/              ← label-checker integration
│   ├── client.py             ← SmartQCClient — beginner-friendly entry point (start here)
│   ├── validators/
│   │   ├── __init__.py       ← create_validator() factory (smartqc registered here)
│   │   └── smartqc.py        ← SmartQCValidator (+ from_state() + save_state())
│   ├── smartqc_workflow.py   ← SmartQCWorkflow (drop-in for QCValidationWorkflow)
│   └── fetchers/             ← Data source adapters (local + labellerr)
│
├── tests/
│   ├── test_smartqc_validator.py   ← 14 unit tests (no GPU needed)
│   └── test_state_and_client.py    ← 18 tests for pkl state + SmartQCClient
│
└── demo/
    └── run_demo.py           ← Minimal working example (no GPU, no downloads)
```

---

## Caching — skip the slow step on repeat runs

The most expensive part of the pipeline is computing DINOv2 embeddings (~15 of 24 minutes).
Save the fitted state after the first run and every subsequent run loads it in ~2 seconds:

```python
from qc_pipeline.client import SmartQCClient

client = SmartQCClient(
    coco_json_path="annotations.json",
    images_dir="images/",
    state_path="qc_output/smartqc_state.pkl",  # auto-saved first run, auto-loaded after
)
results = client.run()
```

Or with the state module directly:

```python
from smartqc_state import save_state, load_state, state_info

# Inspect a saved state file
state_info("qc_output/smartqc_state.pkl")

# CLI
python smartqc_state.py info  qc_output/smartqc_state.pkl
python smartqc_state.py check qc_output/smartqc_state.pkl
```

---

## SmartQCClient — the simplest way in

If you're new to the project, use `SmartQCClient` instead of calling the pipeline directly:

```python
from qc_pipeline.client import SmartQCClient

client = SmartQCClient(
    coco_json_path="annotations.json",
    images_dir="images/",
    output_dir="qc_output/",
    yolo_model_path="models/best.pt",          # optional
    state_path="qc_output/smartqc_state.pkl",  # optional — enables caching
)
results = client.run()

# Filter results
results.print_summary()
results.rejected()                # list of all REJECT annotations
results.by_class("wire")          # all annotations for one class
results.worst(20)                 # 20 lowest-quality annotations

# Export
results.export_csv("my_results.csv")
df = results.class_report()       # pandas DataFrame, sorted by reject rate
```

---

## Running tests

```bash
# No GPU required — all heavy models are mocked
pytest tests/test_smartqc_validator.py -v
```

---

## Configuration reference

Key settings at the top of `smartqc_fixed.py`:

| Setting | Default | What it does |
|---------|---------|-------------|
| `DATASET_PATH` | — | Root of your dataset folder |
| `QUALITY_ACCEPT` | `0.72` | Score threshold for auto-accept |
| `QUALITY_REJECT` | `0.20` | Score threshold for auto-reject |
| `IOU_THRESHOLD` | `0.50` | Minimum IoU for good boundary |
| `BATCH_SIZE` | `16` | Images per GPU batch (reduce if OOM) |
| `USE_GRAYSCALE` | `True` | Use grayscale DINOv2 embeddings |
| `BACKGROUND_CLASSES` | `wall,ceiling,floor` | Classes to skip |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError` on startup | Check `DATASET_PATH` and `CLASS_MAP_FILE` paths |
| `CUDA out of memory` | Reduce `BATCH_SIZE` from 16 to 8 or 4 |
| YOLOv8 not found | Pipeline runs in mock mode — IoU signal = 0, CLIP+DINOv2 still run |
| Tests failing | Run `pip install pytest` then `pytest tests/ -v` |

---

*ISB AMPBA Capstone 2026 · Labellerr AI (Tensor Matics Inc.)*
