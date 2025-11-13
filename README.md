## Gemini QC Validation Pipeline

Pipeline for validating human-labeled bounding boxes and polygons with Google's Gemini multimodal models.

### Features

- Load images with COCO-format annotations and generate bounding-box or polygon crops.
- Submit crops to Gemini with configurable prompts and retries.
- Produce per-annotation JSON results including confidence scores and rationales.
- Summaries for match accuracy and aggregate confidence.

### Requirements

- Python 3.10+
- Packages: `Pillow`, `requests`, `pytest` (for tests)
- Gemini API key with access to the selected model (default: `gemini-1.5-pro`)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> If `requirements.txt` does not exist yet, install packages manually:
> `pip install Pillow requests pytest`.

### Usage

```bash
export GEMINI_API_KEY="your-api-key"
python -m qc_pipeline.run_validation \
  --images-dir /path/to/images \
  --coco-json /path/to/annotations.json \
  --output /tmp/validation_results.json \
  --guidelines ./label_guidelines.txt \
  --padding 8
```

Key arguments:

- `--api-key-env`: environment variable holding the Gemini API key (defaults to `GEMINI_API_KEY`).
- `--disable-polygon-mask`: skip masking polygons to keep rectangular crops.
- `--model-name`: specify alternative Gemini multimodal model.

### Detailed Flow

- **Load data**  
  Run the CLI directly with `--images-dir`/`--coco-json`, or wrap it in an interactive front end that prompts for a local folder containing JPEG/PNG images plus the COCO JSON. `load_coco_dataset` links each annotation to its image and category so later stages see consistent triplets.

- **Create crops**  
  `crop_annotation` generates a rectangular crop for each annotation, using the bounding box or deriving one from polygon segments. Optional padding expands the crop while respecting image bounds. When `mask_polygon=True`, the polygon area becomes opaque and everything else transparent. Before invoking Gemini, ensure the crop meets any model minimum size; you can upscale, skip, or batch undersized crops for a fallback embedding path.

- **Talk to Gemini**  
  `GeminiValidator` converts crops to PNG bytes, fills the prompt template with the human label and optional guideline text, and enforces a JSON response containing `prediction_label`, `confidence`, and `rationale`. The built-in retry logic handles transient API errors. Use `validate_batch` if you add embedding-aware batching.

- **Log results**  
  Each response is stored with image/annotation IDs, geometry metadata, Gemini’s label/confidence, the rationale, and an `is_match` flag (case-insensitive comparison to the human label), making it easy to highlight mismatches.

- **Summaries & future work**  
  The CLI aggregates totals, match/mismatch counts, overall accuracy, and mean confidence, writing everything to the output JSON. Downstream tooling can focus on mismatches to quantify classification errors. Leave a placeholder for plugging in an open-source model alongside Gemini so you can compare scores or fall back when needed.

### Output

Results are saved as JSON containing `entries` per annotation and a `summary` section with totals, matches, accuracy, and mean confidence. See `docs/gemini_qc_flow.md` for schema details.

### Documentation

- Feasibility assessment: `docs/gemini_qc_feasibility.md`
- Validation flow and schema: `docs/gemini_qc_flow.md`

### Testing

```bash
pytest
```

