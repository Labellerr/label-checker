# QC Pipeline Architecture

## Overview

The QC Pipeline is a **general-purpose annotation validation system** built on:
- **Multi-provider AI**: Gemini, OpenAI, Anthropic via LangChain
- **Multi-source data**: Local files, Labellerr platform
- **Two-agent workflow**: LangGraph for stateful orchestration

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           QC Pipeline System                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        Gradio Web UI (app.py)                        │   │
│   │                                                                      │   │
│   │   ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐       │   │
│   │   │  📁 Data  │  │ 🤖 Model  │  │📋 Guide-  │  │ ⚙️ Settings│       │   │
│   │   │  Source   │  │  Config   │  │  lines    │  │           │       │   │
│   │   └───────────┘  └───────────┘  └───────────┘  └───────────┘       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│               ┌──────────────────────┼──────────────────────┐               │
│               │                      │                      │               │
│               ▼                      ▼                      ▼               │
│   ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐      │
│   │     Fetchers      │  │    Validators     │  │   Guidelines      │      │
│   │   (fetchers/)     │  │   (validators/)   │  │   Extractor       │      │
│   │                   │  │                   │  │                   │      │
│   │  • LocalFetcher   │  │  • GeminiValidator│  │  • PDF parsing    │      │
│   │  • LabellerrFetch │  │  • OpenAIValidator│  │  • Label extract  │      │
│   │                   │  │  • AnthropicValid │  │  • Multi-provider │      │
│   └───────────────────┘  └───────────────────┘  └───────────────────┘      │
│               │                      │                      │               │
│               └──────────────────────┼──────────────────────┘               │
│                                      │                                       │
│                                      ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    QC Workflow (qc_workflow.py)                      │   │
│   │                                                                      │   │
│   │   ┌─────────────────────────────────────────────────────────────┐   │   │
│   │   │              LangGraph State Machine                         │   │   │
│   │   │                                                              │   │   │
│   │   │   ┌─────────────────┐         ┌─────────────────┐           │   │   │
│   │   │   │   Guidelines    │────────▶│   Validation    │           │   │   │
│   │   │   │     Agent       │         │     Agent       │           │   │   │
│   │   │   └─────────────────┘         └─────────────────┘           │   │   │
│   │   │                                                              │   │   │
│   │   └─────────────────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│                                      ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                            Output                                    │   │
│   │                                                                      │   │
│   │    qc_results.json    crops/    validation_prompts.log              │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Validators (`validators/`)

All validators inherit from `BaseValidator` and use LangChain for a unified interface.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         BaseValidator (ABC)                          │
│                                                                      │
│   Shared Logic:                                                      │
│   • Prompt templates (default, guidelines-enhanced)                  │
│   • Image encoding (base64)                                          │
│   • Response parsing (JSON extraction)                               │
│   • Retry logic with backoff                                         │
│   • Annotation context building                                      │
│                                                                      │
│   Abstract Methods:                                                  │
│   • _create_llm() → LangChain chat model                            │
│   • _build_multimodal_message() → Provider-specific format          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│   │ GeminiValidator │  │ OpenAIValidator │  │AnthropicValidator│    │
│   │                 │  │                 │  │                 │    │
│   │ ChatGoogleGenAI │  │   ChatOpenAI    │  │  ChatAnthropic  │    │
│   │                 │  │                 │  │                 │    │
│   │ Image format:   │  │ Image format:   │  │ Image format:   │    │
│   │ inline_data     │  │ image_url       │  │ source.base64   │    │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Available Models:**

| Provider | Models |
|----------|--------|
| Gemini | `gemini-3-flash-preview`, `gemini-3-pro-preview`, `gemini-2.5-flash` |
| OpenAI | `gpt-5-mini-2025-08-07`, `gpt-5.2-pro-2025-12-11`, `gpt-5.2-2025-12-11` |
| Anthropic | `claude-opus-4-5-20251101`, `claude-haiku-4-5-20251001`, `claude-sonnet-4-5-20250929` |

### 2. Fetchers (`fetchers/`)

```
┌─────────────────────────────────────────────────────────────────────┐
│                      DataFetcherProtocol                             │
│                                                                      │
│   def fetch_data(output_dir, **kwargs) -> FetchResult               │
│                                                                      │
│   FetchResult:                                                       │
│   • coco_json_path: Path                                            │
│   • images_dir: Path                                                │
│   • metadata: Dict                                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────────────┐    ┌─────────────────────────┐       │
│   │     LocalFileFetcher    │    │    LabellerrFetcher     │       │
│   │                         │    │                         │       │
│   │  Handles:               │    │  Handles:               │       │
│   │  • Individual images    │    │  • SDK authentication   │       │
│   │  • ZIP archives         │    │  • Export creation      │       │
│   │  • COCO JSON files      │    │  • Status polling       │       │
│   │                         │    │  • Image download       │       │
│   └─────────────────────────┘    └─────────────────────────┘       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3. Workflow (`qc_workflow.py`)

```
┌─────────────────────────────────────────────────────────────────────┐
│                      QCValidationWorkflow                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   __init__(output_dir, validator)                                   │
│                                                                      │
│   run(coco_json_path, images_dir, ...) -> (state, results, summary) │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Execution Flow:                                                    │
│                                                                      │
│   1. Load COCO Dataset                                              │
│      └─▶ Parse annotations, categories, images                      │
│                                                                      │
│   2. Run LangGraph Workflow                                         │
│      ├─▶ Guidelines Agent (if PDF provided)                         │
│      │   └─▶ Extract label definitions                              │
│      └─▶ Validation Agent                                           │
│          └─▶ Ready to validate                                      │
│                                                                      │
│   3. Validate Each Annotation                                       │
│      ├─▶ Crop object from image                                     │
│      ├─▶ Build prompt (with/without guidelines)                     │
│      ├─▶ Call validator.validate_crop()                             │
│      └─▶ Record result (match/mismatch/confidence)                  │
│                                                                      │
│   4. Generate Summary                                               │
│      ├─▶ Statistics (matches, mismatches, accuracy)                 │
│      ├─▶ Confidence by category                                     │
│      └─▶ Low confidence & mismatch lists                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              Data Flow                                      │
└────────────────────────────────────────────────────────────────────────────┘

INPUTS                         PROCESSING                         OUTPUTS
──────                         ──────────                         ───────

┌──────────────┐
│  COCO JSON   │───────────┐
│              │           │
│ {            │           │    ┌─────────────────────┐
│  "images",   │           ├───▶│    Data Loader      │
│  "annotations│           │    │                     │
│  "categories"│           │    │  • Parse JSON       │
│ }            │           │    │  • Build dataset    │
└──────────────┘           │    │  • Map images→annot │
                           │    └──────────┬──────────┘
┌──────────────┐           │               │
│   Images     │───────────┘               │
│              │                           │
│  image1.jpg  │                           ▼
│  image2.png  │               ┌─────────────────────┐
│  ...         │               │   Image Utils       │
└──────────────┘               │                     │
                               │  • Crop bbox/polygon│
                               │  • Apply padding    │
┌──────────────┐               │  • Convert to bytes │
│  Guidelines  │               └──────────┬──────────┘
│     PDF      │───────┐                   │
│              │       │                   │
│ "Label X:    │       │                   ▼
│  defined as..│       │       ┌─────────────────────┐
│  visual      │       └──────▶│ Guidelines Extractor│
│  features:.."│               │                     │
└──────────────┘               │  • Parse PDF text   │──────┐
                               │  • AI extraction    │      │
                               │  • Structure labels │      │
                               └─────────────────────┘      │
                                                            │
                               ┌─────────────────────┐      │
                               │     Validator       │◀─────┘
                               │                     │
                               │  For each crop:     │
                               │  1. Build prompt    │
                               │  2. Add guidelines  │
                               │  3. Send to AI      │
                               │  4. Parse response  │
                               │  5. Compare labels  │
                               └──────────┬──────────┘
                                          │
                                          ▼
                               ┌─────────────────────────────────────┐
                               │              OUTPUTS                │
                               │                                     │
                               │  qc_results.json                    │
                               │  ├── results[]                      │
                               │  │   ├── annotation_id              │
                               │  │   ├── label (expected)           │
                               │  │   ├── prediction_label           │
                               │  │   ├── is_match                   │
                               │  │   ├── confidence                 │
                               │  │   └── rationale                  │
                               │  └── summary                        │
                               │      ├── matches                    │
                               │      ├── mismatches                 │
                               │      └── average_confidence         │
                               │                                     │
                               │  crops/                             │
                               │  ├── crop_123_dog.png               │
                               │  └── crop_456_cat.png               │
                               │                                     │
                               │  validation_prompts.log             │
                               └─────────────────────────────────────┘
```

## LangGraph State Machine

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                                │
└─────────────────────────────────────────────────────────────────────┘

                         ┌─────────────────┐
                         │     START       │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Check for PDF  │
                         └────────┬────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
              PDF provided               No PDF provided
                    │                           │
                    ▼                           │
         ┌─────────────────┐                    │
         │   Guidelines    │                    │
         │     Agent       │                    │
         │                 │                    │
         │ • Extract text  │                    │
         │ • Call AI       │                    │
         │ • Parse labels  │                    │
         │ • Store state   │                    │
         └────────┬────────┘                    │
                  │                             │
                  └─────────────┬───────────────┘
                                │
                                ▼
                     ┌─────────────────┐
                     │   Validation    │
                     │     Agent       │
                     │                 │
                     │ • Load state    │
                     │ • Signal ready  │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │      END        │
                     └─────────────────┘


State Schema (QCValidationState):
──────────────────────────────────
{
  pdf_path: Optional[str]
  pdf_text: Optional[str]
  guidelines_dict: Dict[str, Any]
  categories_found: List[str]
  status_message: str
  error: Optional[str]
  gemini_api_key: str
}
```

## Prompt Engineering

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Prompt Templates                              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     DEFAULT_PROMPT_TEMPLATE                          │
│                                                                      │
│  You are an expert visual classifier. The annotator labeled this    │
│  cropped image as: "{expected_label}".                              │
│                                                                      │
│  📊 ANNOTATION DATA:                                                │
│  Category: {category_name}                                          │
│  {annotation_attributes}                                            │
│                                                                      │
│  🔍 VALIDATION REQUIREMENTS:                                        │
│  - If MATCHES, confirm clearly                                      │
│  - If DOES NOT MATCH, state: "This is mislabeled..."               │
│  - Provide specific visual evidence                                 │
│                                                                      │
│  Respond as JSON: {prediction_label, confidence, rationale}         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                   GUIDELINES_PROMPT_TEMPLATE                         │
│                                                                      │
│  You are an expert visual classifier...                             │
│                                                                      │
│  📋 LABEL DEFINITION for "{expected_label}":                        │
│  Definition: {definition}                                           │
│                                                                      │
│  Key Visual Characteristics to verify:                              │
│  {characteristics_list}                                             │
│                                                                      │
│  🔍 VALIDATION PROTOCOL:                                            │
│  1. Examine visual features systematically                          │
│  2. Compare against each characteristic                             │
│  3. Note any deviations                                             │
│  4. Make final determination                                        │
│                                                                      │
│  Confidence Guidelines:                                             │
│  - 0.9+: Clear match/mismatch                                       │
│  - 0.7-0.9: Most characteristics present                            │
│  - 0.5-0.7: Ambiguous                                               │
│  - <0.5: Likely mismatch                                            │
└─────────────────────────────────────────────────────────────────────┘
```

## Adding New Components

### Adding a New AI Provider

1. Create `validators/newprovider.py`:

```python
from langchain_newprovider import ChatNewProvider
from qc_pipeline.validators.base import BaseValidator

class NewProviderValidator(BaseValidator):
    def _create_llm(self):
        return ChatNewProvider(
            model=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature,
        )
    
    def _build_multimodal_message(self, prompt, image_b64):
        # Provider-specific message format
        ...
```

2. Register in `validators/__init__.py`:

```python
PROVIDERS["newprovider"] = {
    "name": "New Provider",
    "models": ["model-a", "model-b"],
    "default_model": "model-a",
    "get_validator": lambda: NewProviderValidator,
}
```

### Adding a New Data Source

1. Create `fetchers/newsource.py`:

```python
from qc_pipeline.fetchers.base import FetchResult

class NewSourceFetcher:
    def fetch_data(self, output_dir, **kwargs) -> FetchResult:
        # Fetch logic
        return FetchResult(
            coco_json_path=...,
            images_dir=...,
            metadata={...},
        )
```

2. Register in `fetchers/base.py`:

```python
PLATFORMS["newsource"] = {
    "name": "New Source",
    "requires_credentials": True,
    "credentials": ["api_key", "project_id"],
}
```

## File Reference

| File | Purpose |
|------|---------|
| `app.py` | Gradio web interface |
| `qc_workflow.py` | Main orchestrator |
| `langgraph_validator.py` | LangGraph state machine |
| `guidelines_extractor.py` | PDF → structured labels |
| `data_loader.py` | COCO JSON parser |
| `image_utils.py` | Crop generation |
| `pdf_utils.py` | PDF text extraction |
| `validators/base.py` | BaseValidator ABC |
| `validators/gemini.py` | Gemini LangChain validator |
| `validators/openai.py` | OpenAI LangChain validator |
| `validators/anthropic.py` | Anthropic LangChain validator |
| `fetchers/base.py` | FetchResult & Protocol |
| `fetchers/local.py` | File upload handler |
| `fetchers/labellerr.py` | Labellerr SDK integration |
| `gemini_validator.py` | Legacy direct API validator |
