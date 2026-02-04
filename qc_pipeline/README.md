# QC Pipeline

AI-powered annotation quality validation supporting multiple providers and data sources.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch web UI
python run_qc_app.py
# Opens http://localhost:7860
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              QC Pipeline                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │   Gradio UI  │     │   Fetchers   │     │  Validators  │                │
│  │   (app.py)   │────▶│              │────▶│              │                │
│  └──────────────┘     │  • local     │     │  • gemini    │                │
│         │             │  • labellerr │     │  • openai    │                │
│         │             └──────────────┘     │  • anthropic │                │
│         │                    │             └──────────────┘                │
│         │                    │                    │                        │
│         ▼                    ▼                    ▼                        │
│  ┌─────────────────────────────────────────────────────────┐              │
│  │                    QC Workflow                           │              │
│  │  ┌─────────────────┐    ┌─────────────────┐            │              │
│  │  │ Guidelines Agent│───▶│ Validation Agent│            │              │
│  │  │ (PDF → labels)  │    │ (crop → verify) │            │              │
│  │  └─────────────────┘    └─────────────────┘            │              │
│  └─────────────────────────────────────────────────────────┘              │
│                              │                                             │
│                              ▼                                             │
│                    ┌──────────────────┐                                   │
│                    │     Results      │                                   │
│                    │  • JSON report   │                                   │
│                    │  • Crops         │                                   │
│                    │  • Statistics    │                                   │
│                    └──────────────────┘                                   │
│                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Package Structure

```
qc_pipeline/
├── app.py                    # Gradio web interface
├── qc_workflow.py            # Main workflow orchestrator
├── langgraph_validator.py    # LangGraph state machine
│
├── validators/               # Multi-provider AI validators
│   ├── __init__.py          # Factory & exports
│   ├── base.py              # BaseValidator ABC
│   ├── gemini.py            # Google Gemini (LangChain)
│   ├── openai.py            # OpenAI GPT (LangChain)
│   └── anthropic.py         # Anthropic Claude (LangChain)
│
├── fetchers/                 # Data source handlers
│   ├── __init__.py          # Factory & exports
│   ├── base.py              # Protocol & FetchResult
│   ├── local.py             # File upload handler
│   └── labellerr.py         # Labellerr platform
│
├── guidelines_extractor.py   # PDF → structured labels
├── data_loader.py           # COCO JSON parser
├── image_utils.py           # Crop generation
├── pdf_utils.py             # PDF text extraction
└── gemini_validator.py      # Legacy direct API validator
```

## Data Flow

```
Input                    Processing                      Output
─────                    ──────────                      ──────

┌────────────┐
│ COCO JSON  │──┐
└────────────┘  │        ┌─────────────────────┐
                ├───────▶│    Data Loader      │
┌────────────┐  │        │  Parse annotations  │
│   Images   │──┘        └──────────┬──────────┘
└────────────┘                      │
                                    ▼
┌────────────┐           ┌─────────────────────┐        ┌────────────┐
│ Guidelines │──────────▶│ Guidelines Extractor│───────▶│ Label Defs │
│    PDF     │           │   AI extracts labels │        └──────┬─────┘
└────────────┘           └─────────────────────┘               │
                                                               │
                         ┌─────────────────────┐               │
                         │    Crop Generator   │               │
                         │  Extract each object│               │
                         └──────────┬──────────┘               │
                                    │                          │
                                    ▼                          ▼
                         ┌─────────────────────────────────────────┐
                         │              AI Validator               │
                         │  • Build prompt with guidelines         │
                         │  • Send image + text to LLM            │
                         │  • Parse JSON response                 │
                         │  • Compare: expected vs predicted      │
                         └──────────────────┬──────────────────────┘
                                            │
                                            ▼
                         ┌─────────────────────────────────────────┐
                         │               Results                   │
                         │  • qc_results.json                     │
                         │  • Cropped images                      │
                         │  • Match/mismatch statistics           │
                         └─────────────────────────────────────────┘
```

## Validation Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Two-Agent Workflow                           │
└─────────────────────────────────────────────────────────────────────┘

          ┌─────────────────────────────────────────────────────┐
          │                 LangGraph State                      │
          │  • pdf_text      • guidelines_dict                  │
          │  • categories    • status_message                   │
          └─────────────────────────────────────────────────────┘
                 │                              ▲
                 ▼                              │
┌────────────────────────────┐    ┌────────────────────────────┐
│     Guidelines Agent       │    │     Validation Agent       │
│                            │    │                            │
│  1. Extract PDF text       │───▶│  1. Load guidelines        │
│  2. Send to AI             │    │  2. For each annotation:   │
│  3. Parse label definitions│    │     • Crop object          │
│  4. Store in state         │    │     • Build prompt         │
│                            │    │     • Validate with AI     │
│  [Runs if PDF provided]    │    │     • Record result        │
└────────────────────────────┘    └────────────────────────────┘
```

## Supported Providers

| Provider  | Models                                                        |
|-----------|---------------------------------------------------------------|
| Gemini    | `gemini-3-flash-preview`, `gemini-3-pro-preview`, `gemini-2.5-flash` |
| OpenAI    | `gpt-5-mini-2025-08-07`, `gpt-5.2-pro-2025-12-11`, `gpt-5.2-2025-12-11` |
| Anthropic | `claude-opus-4-5-20251101`, `claude-haiku-4-5-20251001`, `claude-sonnet-4-5-20250929` |

## Supported Data Sources

| Source    | Description                                      |
|-----------|--------------------------------------------------|
| Upload    | Local COCO JSON + images (individual or ZIP)     |
| Labellerr | Fetch directly from Labellerr platform via SDK   |

## Usage

### Web UI (Recommended)

```bash
python run_qc_app.py
```

### Python API

```python
from qc_pipeline import QCValidationWorkflow, create_validator

# Create validator (choose provider)
validator = create_validator(
    provider="gemini",  # or "openai", "anthropic"
    api_key="your-key",
    model_name="gemini-3-flash-preview",
)

# Run workflow
workflow = QCValidationWorkflow(
    output_dir="output",
    validator=validator,
)

state, results, summary = workflow.run(
    pdf_path="guidelines.pdf",  # optional
    coco_json_path="annotations.json",
    images_dir="images/",
    max_files=100,
    confidence_threshold=0.5,
)

print(f"Accuracy: {summary.matches / summary.gemini_validations * 100:.1f}%")
```

### Multi-Provider Examples

```python
from qc_pipeline.validators import create_validator

# Gemini
gemini = create_validator("gemini", "key", "gemini-3-flash-preview")

# OpenAI  
openai = create_validator("openai", "key", "gpt-5-mini-2025-08-07")

# Anthropic
claude = create_validator("anthropic", "key", "claude-sonnet-4-5-20250929")
```

## Output

```
~/.qc_pipeline/output/run/
├── annotations.json         # Fetched COCO annotations
├── images/                  # Downloaded images
├── crops/                   # Cropped objects
│   ├── crop_123_dog.png
│   └── crop_456_cat.png
├── qc_results.json          # Validation results
└── validation_prompts.log   # Debug log
```

### Result Format

```json
{
  "results": [
    {
      "annotation_id": 123,
      "label": "cat",
      "prediction_label": "dog",
      "is_match": false,
      "confidence": 0.95,
      "rationale": "This is mislabeled. Visual evidence shows canine features..."
    }
  ],
  "summary": {
    "total_annotations": 100,
    "matches": 85,
    "mismatches": 15,
    "average_confidence": 0.87
  }
}
```

## Dependencies

**Required:**
```
pillow>=10.0.0
gradio>=4.0.0
pypdf>=4.0.0
pydantic>=2.0.0
langchain-core>=0.2.0
langgraph>=0.0.20
```

**Provider SDKs (at least one):**
```
langchain-google-genai>=2.0.0    # Gemini
langchain-openai>=0.1.0          # OpenAI
langchain-anthropic>=0.1.0       # Anthropic
```

**Optional:**
```
labellerr                        # Labellerr platform integration
```

## Environment Variables

```bash
# AI Provider Keys (at least one)
export GOOGLE_API_KEY="..."
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."

# Labellerr (optional)
export LABELLERR_API_KEY="..."
export LABELLERR_API_SECRET="..."
```
