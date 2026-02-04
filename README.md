# Agentic Label Checker

AI-powered annotation validation using multi-agent workflows. Validates image annotations with Gemini, OpenAI, or Anthropic vision models.

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Launch
python run_qc_app.py

# 3. Open http://localhost:7860
```

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│    ┌──────────┐      ┌──────────────┐      ┌──────────────┐            │
│    │  Upload  │      │  Guidelines  │      │     AI       │            │
│    │  Data    │─────▶│  Extraction  │─────▶│  Validation  │            │
│    └──────────┘      └──────────────┘      └──────────────┘            │
│         │                   │                     │                     │
│         │                   │                     │                     │
│         ▼                   ▼                     ▼                     │
│    ┌──────────┐      ┌──────────────┐      ┌──────────────┐            │
│    │ COCO JSON│      │ Label        │      │ Results:     │            │
│    │ + Images │      │ Definitions  │      │ • Matches    │            │
│    └──────────┘      └──────────────┘      │ • Mismatches │            │
│                                            │ • Confidence │            │
│                                            └──────────────┘            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Provider** | Gemini, OpenAI, Anthropic |
| **Multi-Source** | File upload or Labellerr platform |
| **Guidelines** | Extract label definitions from PDF |
| **Web UI** | Gradio interface at localhost:7860 |
| **Python API** | Full programmatic access |

## Supported Models

| Provider | Models |
|----------|--------|
| **Gemini** | gemini-3-flash-preview, gemini-3-pro-preview, gemini-2.5-flash |
| **OpenAI** | gpt-5-mini-2025-08-07, gpt-5.2-pro-2025-12-11, gpt-5.2-2025-12-11 |
| **Anthropic** | claude-opus-4-5-20251101, claude-haiku-4-5-20251001, claude-sonnet-4-5-20250929 |

## Python API

```python
from qc_pipeline import QCValidationWorkflow, create_validator

# Create any validator
validator = create_validator(
    provider="gemini",      # or "openai", "anthropic"
    api_key="your-key",
    model_name="gemini-3-flash-preview",
)

# Run validation
workflow = QCValidationWorkflow(output_dir="output", validator=validator)
state, results, summary = workflow.run(
    coco_json_path="annotations.json",
    images_dir="images/",
)

print(f"Accuracy: {summary.matches / summary.gemini_validations * 100:.1f}%")
```

## Workflow Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Two-Agent LangGraph Workflow                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                        Shared State                              │   │
│   │   pdf_text │ guidelines │ categories │ results │ errors         │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│          │                                        ▲                     │
│          ▼                                        │                     │
│   ┌────────────────────┐              ┌────────────────────┐           │
│   │  Guidelines Agent  │              │  Validation Agent  │           │
│   │                    │              │                    │           │
│   │  • Read PDF        │─────────────▶│  • Load COCO       │           │
│   │  • Extract labels  │              │  • Crop objects    │           │
│   │  • Store in state  │              │  • Call AI model   │           │
│   │                    │              │  • Compare labels  │           │
│   │  [Optional]        │              │  • Save results    │           │
│   └────────────────────┘              └────────────────────┘           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
label-checker/
├── run_qc_app.py              # Launch web UI
├── requirements.txt           # Dependencies
│
├── qc_pipeline/               # Core package
│   ├── app.py                 # Gradio web interface
│   ├── qc_workflow.py         # Main orchestrator
│   ├── langgraph_validator.py # State machine
│   │
│   ├── validators/            # AI providers
│   │   ├── gemini.py
│   │   ├── openai.py
│   │   └── anthropic.py
│   │
│   ├── fetchers/              # Data sources
│   │   ├── local.py           # File upload
│   │   └── labellerr.py       # Platform SDK
│   │
│   ├── data_loader.py         # COCO parser
│   ├── image_utils.py         # Crop generator
│   └── guidelines_extractor.py# PDF processor
│
└── demo/                      # Example scripts
    └── labellerr-integration/ # Legacy Labellerr demo
```

## Requirements

- Python 3.10+
- At least one AI provider API key
- COCO format annotations

**Install all dependencies:**
```bash
pip install -r requirements.txt
```

**Provider-specific:**
```bash
pip install langchain-google-genai   # Gemini
pip install langchain-openai         # OpenAI
pip install langchain-anthropic      # Anthropic
pip install labellerr                # Labellerr platform
```

## Documentation

| Document | Description |
|----------|-------------|
| [QC Pipeline README](qc_pipeline/README.md) | Package architecture & API |
| [Architecture](ARCHITECTURE.md) | System design & diagrams |
| [Labellerr Integration](demo/labellerr-integration/README.md) | Platform setup guide |

## Example Output

```
✅ Validation Complete

## Summary

| Metric | Value |
|--------|-------|
| Total Annotations | 100 |
| Validations | 100 |
| Matches | 92 (92.0%) |
| Mismatches | 8 |
| Avg Confidence | 0.894 |

### Issues Found
⚠️ cat → dog (0.95) - "Clear canine features..."
⚠️ chair → table (0.87) - "Four legs with flat surface..."
```

## License

MIT
