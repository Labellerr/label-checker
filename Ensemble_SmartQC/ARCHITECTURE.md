# SmartQC — Architecture

> **Who is this for?**  
> Anyone new to this project — whether you're a developer, annotator, or reviewer. You don't need to know AI or Python to understand how SmartQC works. Read top-to-bottom and everything will make sense.

---

## Table of Contents

1. [What SmartQC Does — In Plain English](#1-what-smartqc-does--in-plain-english)
2. [The Big Picture — How It All Fits Together](#2-the-big-picture--how-it-all-fits-together)
3. [The 11-Step Pipeline — What Happens When You Run It](#3-the-11-step-pipeline--what-happens-when-you-run-it)
4. [The Three Quality Signals — What Each One Checks](#4-the-three-quality-signals--what-each-one-checks)
5. [Adaptive Weights — Why Different Classes Get Different Treatment](#5-adaptive-weights--why-different-classes-get-different-treatment)
6. [The Four Verdicts — What SmartQC Decides](#6-the-four-verdicts--what-smartqc-decides)
7. [File and Folder Structure](#7-file-and-folder-structure)
8. [Key Configuration — What You Can Change](#8-key-configuration--what-you-can-change)
9. [Inputs and Outputs](#9-inputs-and-outputs)
10. [The REST API — Using SmartQC Over HTTP](#10-the-rest-api--using-smartqc-over-http)
11. [Integration with label-checker](#11-integration-with-label-checker)
12. [How to Run It — Step by Step](#12-how-to-run-it--step-by-step)
13. [Glossary — What the Technical Terms Mean](#13-glossary--what-the-technical-terms-mean)

---

## 1. What SmartQC Does — In Plain English

When humans label images for AI training (drawing boxes around objects and naming them), they make mistakes:

- They label a barrel as a crate
- They draw a box that doesn't quite cover the whole object
- They miss objects entirely — a lamp on the ceiling, a fuse box in the background

These mistakes, called **annotation errors**, silently damage AI models. The model learns wrong things and performs poorly in the real world.

**SmartQC automatically checks every single annotation** in a dataset and gives it one of four verdicts:

| Verdict | Meaning | What happens next |
|---------|---------|-------------------|
|  **ACCEPT** | Annotation looks correct | Goes straight to training |
|  **REVIEW** | Borderline — might be fine | Human does a quick check |
|  **FLAG** | Looks suspicious | Human reviews carefully |
|  **REJECT** | Clearly wrong | Discarded or re-annotated |

It does this using **three AI models working together**, not just one. It checked 39,182 annotations in 24 minutes.

---

## 2. The Big Picture — How It All Fits Together

```
┌─────────────────────────────────────────────────────────────────────┐
│                         YOUR DATASET                                │
│          (images + annotation files in YOLO or COCO format)         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SmartQC Pipeline                               │
│                                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────────┐  │
│   │  CLIP    │    │ DINOv2   │    │ YOLOv8   │    │  Adaptive   │  │
│   │ Semantic │    │Prototype │    │  Boundary│    │   Fusion    │  │
│   │  Check   │    │  Match   │    │   IoU    │    │  (weights)  │  │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘    └──────┬──────┘  │
│        │               │               │                  │         │
│        └───────────────┴───────────────┘                  │         │
│                          score per signal                  │         │
│                                    └──────────────────────►│         │
│                                                            │         │
│                                              quality score │         │
│                                                            ▼         │
│                                          ACCEPT / REVIEW / FLAG / REJECT
└─────────────────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
      REST API         CSV Report     Visual folders
   (live queries)  (one row per ann)  (crops by verdict)
```

---

## 3. The 11-Step Pipeline — What Happens When You Run It

Each step runs in order. You don't need to trigger them manually — running `smartqc_fixed.py` executes all 11 steps automatically.

```
Step 1 ── Load class mapping
           Read class_mapping.json. Map class IDs to names (e.g. 3 → "forklift").
           Identify background classes to skip (wall, ceiling, floor).

Step 2 ── Ingest annotations
           Read all annotation files — either YOLO .txt or COCO JSON format.
           Convert bounding boxes to pixel coordinates. Build one row per annotation.

Step 3 ── Load AI models
           Load DINOv2-base (768-dim visual embeddings).
           Load CLIP ViT-B/32 (semantic image-text matching).
           Load YOLOv8 (object detection for boundary checking).

Step 4 ── Extract DINOv2 embeddings
           Crop each annotated object out of its image.
           Convert crop to grayscale (removes colour noise).
           Pass through DINOv2 → 768-dimensional vector per object.

Step 5 ── Assign class prototypes
           Compute mean embedding per class (the "average look" of each class).
           Measure prototype purity: how visually separable is each class?
           Print purity table — this drives the adaptive weight selection in Step 8.

Step 6 ── Run CLIP validation
           Score each crop against all 22 class text prompts.
           Detect label mismatches (CLIP says "barrel" but label says "crate").
           Auto-calibrate mismatch threshold from the dataset's own score distribution.

Step 7 ── Run YOLOv8 inference
           Run the trained warehouse detection model over all images (batched, 16/batch).
           Get predicted bounding boxes with confidence scores.
           Cache predictions per image (not per annotation — efficient).

Step 8 ── Compute QC metrics (adaptive fusion)
           For each annotation, select weights based on class purity:
             POOR purity (<25%):  CLIP 80%  + IoU 20%
             FAIR purity (25-50%): CLIP 55% + Proto 25% + IoU 20%
             GOOD purity (≥50%):  CLIP 45% + Proto 35% + IoU 20%
           Fuse the three signals → one quality score [0, 1].
           Apply hard overrides: CLIP mismatch or score < 0.16 → REJECT.
           Assign ACCEPT / REVIEW / FLAG / REJECT verdict.

Step 9 ── Detect false negatives
           Find objects that YOLOv8 detected but are NOT in the annotation file.
           These are real objects that human annotators missed entirely.

Step 10 ── Build metrics matrix
            Compute per-class and dataset-level statistics:
            precision, recall, F1, mean IoU, CLIP scores, purity, accept/reject rates.

Step 11 ── Save all outputs
            Write CSV (one row per annotation with all scores and verdict).
            Write visual QC folders (ACCEPT / REVIEW / FLAG / REJECT).
            Write JSON summary. Start REST API.
```

---

## 4. The Three Quality Signals — What Each One Checks

Think of SmartQC as three independent inspectors, each checking a different thing. Their findings are combined into one verdict.

---

### Signal 1 — CLIP (Semantic check)

**What it checks:** Does the object in the image actually match the label?

**How it works:**
- CLIP is an AI model that understands both images and text together
- We describe each class in plain English: `"a red fire extinguisher mounted on a wall"`, `"a yellow forklift or warehouse vehicle"`, etc.
- CLIP scores how well the cropped object matches each of the 22 descriptions
- If the label says "bucket" but the crop scores highest for "barrel" — that's a mismatch

**Why it matters:** This catches the most common annotation error — labelling the wrong class.

**Weight range:** 45–80% (higher weight for classes that are harder to tell apart visually)

---

### Signal 2 — DINOv2 Prototype (Visual similarity check)

**What it checks:** Does this object look like a typical example of its class?

**How it works:**
- DINOv2 is an AI model that turns any image into a 768-number "fingerprint" of its visual appearance
- We compute one prototype (average fingerprint) per class across the whole dataset
- Each annotation's fingerprint is compared to its class prototype
- If a "box" annotation's fingerprint is much closer to the "crate" prototype — that's suspicious

**Why it matters:** Catches subtle annotation errors where the class name is plausible but visually wrong.

**Weight range:** 0–35% (suppressed to 0% for classes where the prototype is unreliable — see Section 5)

---

### Signal 3 — YOLOv8 IoU (Boundary quality check)

**What it checks:** Is the bounding box drawn accurately around the object?

**How it works:**
- A YOLOv8 model trained on warehouse images independently detects all objects
- For each annotation, we measure the overlap (IoU = Intersection over Union) between the human-drawn box and the model's predicted box
- IoU of 1.0 = perfect overlap. IoU below 0.5 = poor boundary quality

**Why it matters:** Catches annotations where the class is right but the box is too small, too large, or badly positioned.

**Weight:** Always 20% (fixed — boundary quality matters for every class equally)

---

## 5. Adaptive Weights — Why Different Classes Get Different Treatment

This is the key innovation in SmartQC. Most QC systems use the same fixed weights for all classes. SmartQC doesn't — because **different classes need different validation strategies**.

### The problem

Some warehouse objects look nearly identical to each other under a visual AI model:
- `wire`, `bracket`, `fuse_box` — all small, dark, wall-mounted objects
- `paper_note`, `paper_shortcut` — both small paper labels

For these classes, the DINOv2 prototype is **unreliable** — it can't tell them apart, so its vote would add noise, not signal.

Other classes have very distinctive visual signatures:
- `fire_extinguisher` — red cylinder, always wall-mounted
- `forklift` — large yellow vehicle
- `cone` — orange triangle

For these, the DINOv2 prototype is **highly reliable** and deserves more influence.

### How purity is measured

**Prototype purity** = the fraction of a class's objects whose nearest prototype match is their own class.

- `fire_extinguisher` purity: **72%** → most fire extinguisher crops are closest to the fire extinguisher prototype ✅
- `paper_shortcut` purity: **9%** → almost none of its crops match its own prototype reliably ❌

### The three weight tiers

```
┌─────────────────┬──────────────────────────┬──────┬───────┬──────┐
│ Tier            │ Example classes          │ CLIP │ Proto │ IoU  │
├─────────────────┼──────────────────────────┼──────┼───────┼──────┤
│ GOOD (≥50%)     │ fire_extinguisher,       │  45% │   35% │  20% │
│                 │ forklift, cone, pallet   │      │       │      │
├─────────────────┼──────────────────────────┼──────┼───────┼──────┤
│ FAIR (25–50%)   │ box, barrel, crate,      │  55% │   25% │  20% │
│                 │ cart, sign, pillar, lamp │      │       │      │
├─────────────────┼──────────────────────────┼──────┼───────┼──────┤
│ POOR (<25%)     │ wire, bracket, fuse_box, │  80% │    0% │  20% │
│                 │ paper_note, paper_shortcut│     │       │      │
└─────────────────┴──────────────────────────┴──────┴───────┴──────┘
```

> **In plain English:** For hard-to-distinguish classes (POOR), we trust CLIP's language-based understanding more. For visually distinctive classes (GOOD), we trust the visual prototype more. IoU always contributes 20% because boundary quality matters equally for all classes.

---

## 6. The Four Verdicts — What SmartQC Decides

Every annotation gets exactly one verdict. Here's how the thresholds work:

```
Quality score (0.0 → 1.0)
│
├── ≥ 0.72 ──────────────────────────── ✅ ACCEPT
│                                         Auto-accepted. Goes to training.
│
├── 0.45 – 0.72 ─────────────────────── 🔵 REVIEW
│                                         Borderline. Quick human check.
│
├── 0.20 – 0.45 ─────────────────────── 🟡 FLAG
│                                         Suspicious. Careful human review.
│
└── < 0.20 ──────────────────────────── ❌ REJECT
                                          Auto-rejected. Re-annotate.

HARD OVERRIDES (bypass the score entirely):
  • CLIP mismatch detected → always REJECT
  • CLIP confidence < 0.16 → always REJECT
```

### Results on the warehouse dataset (39,182 annotations)

```
 ACCEPT   6,514  (16.6%) — automatically cleared, no human needed
 REVIEW  15,607  (39.8%) — borderline, human spot-check
 FLAG    13,027  (33.2%) — suspicious, prioritise for review
 REJECT   4,034  (10.3%) — automatically discarded
```

---

## 7. File and Folder Structure

```
smartqc/
│
├── smartqc_fixed.py          ← Main pipeline — run this to QC a dataset
├── smartqc_api.py            ← REST API server — run after the pipeline
├── class_mapping.json        ← Maps class IDs to names and categories
│
├── qc_pipeline/              ← label-checker integration
│   ├── validators/
│   │   ├── __init__.py       ← create_validator() factory
│   │   ├── smartqc.py        ← SmartQCValidator (label-checker compatible)
│   │   ├── gemini.py
│   │   ├── openai.py
│   │   └── anthropic.py
│   ├── smartqc_workflow.py   ← SmartQCWorkflow (drop-in for QCValidationWorkflow)
│   └── fetchers/
│       ├── local.py          ← Load annotations from local files
│       └── labellerr.py      ← Load annotations from Labellerr platform
│
├── tests/
│   └── test_smartqc_validator.py   ← 14 unit tests (no GPU required)
│
├── qc_output/                ← Created automatically when pipeline runs
│   ├── ACCEPT/               ← Visual crops of accepted annotations
│   ├── REVIEW/               ← Visual crops of borderline annotations
│   ├── FLAG/                 ← Visual crops of suspicious annotations
│   ├── REJECT/               ← Visual crops of rejected annotations
│   └── METRICS/
│       ├── qc_results.csv    ← One row per annotation, all scores + verdict
│       ├── qc_summary.json   ← Dataset-level statistics
│       └── object_prototypes.csv  ← Per-class purity scores
│
├── ARCHITECTURE.md           ← This file
└── README.md                 ← Quick start guide
```

---

## 8. Key Configuration — What You Can Change

All configuration lives at the top of `smartqc_fixed.py`. You don't need to change the AI model code — just edit these values.

```python
# ── PATHS ──────────────────────────────────────────────────────────
DATASET_PATH   = "/path/to/your/dataset"   # ← Change this to your dataset folder
YOLO_IMAGE_DIR = os.path.join(DATASET_PATH, "yolo", "images")
YOLO_LABEL_DIR = os.path.join(DATASET_PATH, "yolo", "labels")
CLASS_MAP_FILE = os.path.join(DATASET_PATH, "class_mapping.json")

# ── MODELS ─────────────────────────────────────────────────────────
DINOV2_MODEL   = "facebook/dinov2-base"    # ← Can upgrade to dinov2-large for better accuracy
USE_GRAYSCALE  = True                      # ← Set False to use colour embeddings
BATCH_SIZE     = 16                        # ← Reduce to 8 if you get GPU out-of-memory errors

# ── QC THRESHOLDS ──────────────────────────────────────────────────
QUALITY_ACCEPT = 0.72   # ← Raise this to auto-accept fewer annotations (more conservative)
QUALITY_REVIEW = 0.45   # ← Lower boundary of the REVIEW band
QUALITY_REJECT = 0.20   # ← Below this = auto-reject
IOU_THRESHOLD  = 0.50   # ← Minimum IoU to consider a boundary acceptable
CONF_THRESHOLD = 0.40   # ← Ignore YOLOv8 predictions below this confidence

# ── CLIP PROMPTS ────────────────────────────────────────────────────
# To add a new class or improve accuracy for an existing one,
# edit the CLIP_PROMPTS dictionary:
CLIP_PROMPTS = {
    "box":    "a cardboard shipping box in a warehouse",
    "forklift": "a yellow forklift or warehouse vehicle",
    # ... add your class here: "my_class": "description of my class"
}

# ── CLASSES TO SKIP ────────────────────────────────────────────────
BACKGROUND_CLASSES = {"wall", "ceiling", "floor"}   # ← Add classes you want to skip
```

### Adding a new object class

1. Add the class to `class_mapping.json` with a new ID and name
2. Add a descriptive prompt to `CLIP_PROMPTS` in `smartqc_fixed.py`
3. Re-run the pipeline — purity will be computed automatically

---

## 9. Inputs and Outputs

### Inputs — what SmartQC needs

| Input | Format | Description |
|-------|--------|-------------|
| Images | `.jpg`, `.png` | The warehouse images that were annotated |
| Annotations (YOLO) | `.txt` files | One file per image: `class_id cx cy w h` per line, normalised 0–1 |
| Annotations (COCO) | `annotations.json` | Standard COCO JSON format with images, annotations, categories |
| Class mapping | `class_mapping.json` | Maps numeric class IDs to human-readable names |
| YOLOv8 model | `best.pt` | Trained object detection model (optional — mock mode available) |

### Outputs — what SmartQC produces

| Output | Location | Description |
|--------|----------|-------------|
| QC results CSV | `qc_output/METRICS/qc_results.csv` | One row per annotation. All signal scores, weights, purity, verdict, rationale. |
| QC summary JSON | `qc_output/METRICS/qc_summary.json` | Dataset totals: counts per verdict, precision, recall, F1, mean IoU. |
| Purity scores | `qc_output/METRICS/object_prototypes.csv` | Per-class purity, weight tier, mean CLIP score. |
| Visual ACCEPT crops | `qc_output/ACCEPT/` | Image crops of auto-accepted annotations with score overlay. |
| Visual REVIEW crops | `qc_output/REVIEW/` | Image crops of borderline annotations. |
| Visual FLAG crops | `qc_output/FLAG/` | Image crops of suspicious annotations. |
| Visual REJECT crops | `qc_output/REJECT/` | Image crops of rejected annotations. |
| REST API | `http://localhost:8000` | Live HTTP endpoints for querying results. |

### CSV output columns (key fields)

| Column | What it means |
|--------|---------------|
| `annotation_id` | Unique ID of the annotation |
| `file_name` | Image file the annotation belongs to |
| `category_name` | The label the human annotator assigned |
| `qc_flag` | SmartQC verdict: ACCEPT / REVIEW / FLAG / REJECT |
| `quality_score` | Overall quality score 0.0–1.0 |
| `clip_lbl_score` | CLIP's confidence the crop matches its label |
| `clip_top_class` | Which class CLIP thinks it actually is |
| `clip_mismatch` | 1 if CLIP disagrees with the label |
| `proto_match` | 1 if DINOv2 prototype agrees with the label |
| `best_iou` | Best overlap with any YOLOv8 prediction |
| `is_false_positive` | 1 if no model detection overlaps this annotation |
| `clip_weight` | Weight given to CLIP for this annotation's class |
| `proto_weight` | Weight given to DINOv2 prototype |
| `iou_weight` | Weight given to IoU (always 0.20) |
| `class_purity` | Prototype purity of this annotation's class |
| `rationale` | Human-readable explanation of the verdict |

---

## 10. The REST API — Using SmartQC Over HTTP

After the pipeline runs, start the API server:

```bash
python smartqc_api.py
# API available at http://localhost:8000
```

### Available endpoints

| Endpoint | What it returns |
|----------|----------------|
| `GET /qc/health` | Check if the API is running |
| `GET /qc/summary` | Dataset-level statistics (total, accept/review/flag/reject counts, precision, recall, F1) |
| `GET /qc/results` | All annotation results (paginated — add `?page=2&limit=100`) |
| `GET /qc/results/{annotation_id}` | Single annotation result by ID |
| `GET /qc/flagged` | Only REVIEW + FLAG + REJECT annotations |
| `GET /qc/stats/by-class` | Per-class breakdown of accept/reject rates and signal scores |

### Example — check a specific annotation

```bash
curl http://localhost:8000/qc/results/12345
```

```json
{
  "annotation_id": "12345",
  "label": "forklift",
  "prediction_label": "forklift",
  "is_match": true,
  "confidence": 0.791,
  "rationale": "CLIP confirmed + visually typical + good IoU",
  "qc_flag": "ACCEPT",
  "quality_score": 0.791,
  "clip_lbl_score": 0.261,
  "best_iou": 0.72,
  "class_purity": 0.68
}
```

---

## 11. Integration with label-checker

SmartQC is a registered provider in Labellerr's [label-checker](https://github.com/Labellerr/label-checker) framework.

### Use SmartQC via the factory (simplest)

```python
from qc_pipeline.validators import create_validator

# No API key needed — SmartQC runs entirely on local models
validator = create_validator(
    provider="smartqc",
    yolo_model_path="path/to/best.pt"   # optional
)
```

### Use SmartQC as a workflow (full pipeline)

```python
from qc_pipeline.smartqc_workflow import SmartQCWorkflow

workflow = SmartQCWorkflow(
    output_dir="output",
    yolo_model_path="path/to/best.pt"   # optional
)

state, results, summary = workflow.run(
    coco_json_path="annotations.json",
    images_dir="images/",
    pdf_path="labelling_guidelines.pdf"  # optional
)

print(f"Accepted:  {summary.matches}/{summary.total}")
print(f"Precision: {summary.average_confidence:.1%}")
```

### Compare SmartQC with an LLM-based validator

```python
from qc_pipeline.validators import create_validator

smartqc  = create_validator("smartqc")
gemini   = create_validator("gemini",    api_key="...", model_name="gemini-2.5-flash")
claude   = create_validator("anthropic", api_key="...", model_name="claude-sonnet-4-6")
```

---

## 12. How to Run It — Step by Step

### Prerequisites

```bash
# Install Python dependencies
pip install torch transformers ultralytics pillow pandas tqdm scikit-learn fastapi uvicorn

# GPU is recommended but not required
# Without GPU: pipeline runs in CPU mode (~3–5x slower)
```

### Option A — Run the full pipeline (recommended)

```bash
# 1. Edit DATASET_PATH at the top of smartqc_fixed.py to point to your dataset
# 2. Run the pipeline
python smartqc_fixed.py

# Output will appear in qc_output/ folder
# API starts automatically at http://localhost:8000
```

### Option B — Run via label-checker workflow

```python
from qc_pipeline.smartqc_workflow import SmartQCWorkflow

workflow = SmartQCWorkflow(output_dir="output")
state, results, summary = workflow.run(
    coco_json_path="path/to/annotations.json",
    images_dir="path/to/images/"
)
```

### Option C — Run tests (no GPU needed)

```bash
pytest tests/test_smartqc_validator.py -v
# All 14 tests use mocked models — runs on any machine
```

### What to expect in the terminal

```
════════════════════════════════════════════════════════════════
  SmartQC — Intelligent Annotation Quality Control
════════════════════════════════════════════════════════════════

  ──────────────────────────────────────────────────────────
  ▶  Class Mapping Summary
  ──────────────────────────────────────────────────────────
    Total classes loaded              : 25
    QC classes (excl. bg)             : 22
    Device                            : cuda

  ──────────────────────────────────────────────────────────
  ▶  Prototype Purity Table
  ──────────────────────────────────────────────────────────
    fire_extinguisher    ████████████████████░░░░  72%  [GOOD]
    forklift             ██████████████████░░░░░░  68%  [GOOD]
    ...
    paper_shortcut       ███░░░░░░░░░░░░░░░░░░░░░   9%  [POOR]

  ──────────────────────────────────────────────────────────
  ▶  QC Results
  ──────────────────────────────────────────────────────────
    ACCEPT    6,514  (16.6%)
    REVIEW   15,607  (39.8%)
    FLAG     13,027  (33.2%)
    REJECT    4,034  (10.3%)

    Precision : 95.83%
    Recall    : 69.54%
    F1        : 0.8060
    Mean IoU  : 0.8614
```

---

## 13. Glossary — What the Technical Terms Mean

| Term | Plain English meaning |
|------|-----------------------|
| **Annotation** | A labelled box drawn around an object in an image (e.g. a box drawn around a forklift, labelled "forklift") |
| **Bounding box** | The rectangle drawn around an object |
| **CLIP** | An AI model that understands both images and text. Used here to check whether an image crop matches its label's description. |
| **DINOv2** | An AI model that turns images into numerical "fingerprints" (embeddings). Used here to compare objects to class prototypes. |
| **Embedding** | A list of 768 numbers that represents what an image looks like to an AI model |
| **Prototype** | The average embedding of all objects in one class — represents the "typical look" of that class |
| **Purity** | How reliably each object in a class matches its own prototype vs other classes. High purity = visually distinctive class. |
| **IoU** | Intersection over Union — a measure of how much two bounding boxes overlap. 1.0 = perfect. 0.0 = no overlap. |
| **False positive** | An annotation that exists but shouldn't — the model finds no real object there |
| **False negative** | A real object in the image that was never annotated — the model finds it, but no human label exists |
| **Precision** | Of all annotations, what fraction are correct? (95.83%) |
| **Recall** | Of all real objects, what fraction were annotated? (69.54% — low because many objects were missed) |
| **F1 score** | The balance between precision and recall (0.8060) |
| **Adaptive weights** | Different importance given to each signal depending on how reliable it is for each class |
| **YOLO format** | A common annotation file format: one .txt file per image, one line per annotation |
| **COCO JSON** | Another common annotation format: one .json file for the whole dataset |
| **REST API** | A way to query SmartQC results over HTTP, like a website backend |
| **label-checker** | Labellerr's open-source QC framework that SmartQC integrates with |

---

## Questions?

- **Pipeline doesn't start?** Check that `DATASET_PATH` in `smartqc_fixed.py` points to the right folder and that `class_mapping.json` exists there.
- **Out of memory error?** Reduce `BATCH_SIZE` from 16 to 8 or 4.
- **YOLOv8 not found?** The pipeline runs in mock mode without a trained model — IoU signal will be 0 for all annotations, but CLIP and DINOv2 still run normally.
- **Tests failing?** Run `pip install pytest` then `pytest tests/ -v`.

---

*SmartQC · ISB AMPBA Capstone 2026 · Labellerr AI (Tensor Matics Inc.) · [github.com/Labellerr/label-checker](https://github.com/Labellerr/label-checker)*
