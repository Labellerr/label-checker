# System Architecture

High-level architecture of the Agentic Label Checker system.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Agentic Label Checker                               │
│                                                                              │
│   "AI-powered annotation validation using multi-agent workflows"            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                           ┌─────────────────┐                               │
│                           │   Web UI        │                               │
│                           │   (Gradio)      │                               │
│                           │   localhost:7860│                               │
│                           └────────┬────────┘                               │
│                                    │                                        │
│       ┌────────────────────────────┼────────────────────────────┐          │
│       │                            │                            │          │
│       ▼                            ▼                            ▼          │
│  ┌─────────────┐          ┌─────────────────┐          ┌─────────────┐    │
│  │   Data      │          │   QC Pipeline   │          │    AI       │    │
│  │   Sources   │─────────▶│   Workflow      │◀─────────│   Providers │    │
│  │             │          │                 │          │             │    │
│  │ • Upload    │          │ • LangGraph     │          │ • Gemini    │    │
│  │ • Labellerr │          │ • Two agents    │          │ • OpenAI    │    │
│  │             │          │ • Results       │          │ • Anthropic │    │
│  └─────────────┘          └─────────────────┘          └─────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. QC Pipeline (`qc_pipeline/`)

The main validation engine:

```
qc_pipeline/
├── app.py                    # Web UI
├── qc_workflow.py            # Orchestrator
├── langgraph_validator.py    # State machine
│
├── validators/               # AI providers
│   ├── gemini.py
│   ├── openai.py
│   └── anthropic.py
│
├── fetchers/                 # Data sources
│   ├── local.py
│   └── labellerr.py
│
└── (utilities)
    ├── data_loader.py        # COCO parser
    ├── image_utils.py        # Crop generator
    ├── pdf_utils.py          # PDF extraction
    └── guidelines_extractor.py
```

### 2. Two-Agent Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LangGraph Workflow                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                     Shared State                             │   │
│   │   • pdf_text        • guidelines_dict    • categories       │   │
│   │   • status_message  • error              • api_key          │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│                    ┌─────────────────────┐                          │
│                    │       START         │                          │
│                    └──────────┬──────────┘                          │
│                               │                                      │
│                    ┌──────────▼──────────┐                          │
│                    │   PDF Provided?     │                          │
│                    └──────────┬──────────┘                          │
│                     Yes       │       No                            │
│                    ┌──────────┴──────────┐                          │
│                    ▼                     │                          │
│         ┌─────────────────┐              │                          │
│         │   Guidelines    │              │                          │
│         │     Agent       │              │                          │
│         │                 │              │                          │
│         │ Extract label   │              │                          │
│         │ definitions     │              │                          │
│         │ from PDF        │              │                          │
│         └────────┬────────┘              │                          │
│                  │                       │                          │
│                  └───────────┬───────────┘                          │
│                              ▼                                      │
│                   ┌─────────────────┐                               │
│                   │   Validation    │                               │
│                   │     Agent       │                               │
│                   │                 │                               │
│                   │ Prepare for     │                               │
│                   │ crop validation │                               │
│                   └────────┬────────┘                               │
│                            │                                        │
│                            ▼                                        │
│                   ┌─────────────────┐                               │
│                   │       END       │                               │
│                   └─────────────────┘                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3. Validation Flow

```
For each annotation:

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    Image     │────▶│    Crop      │────▶│   Prompt     │
│              │     │   Object     │     │   Builder    │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Result     │◀────│    Parse     │◀────│   AI Model   │
│  Recording   │     │   Response   │     │   (Vision)   │
└──────────────┘     └──────────────┘     └──────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────┐
│                        Output                             │
│                                                          │
│   {                                                      │
│     "annotation_id": 123,                                │
│     "label": "cat",              ← Expected              │
│     "prediction_label": "dog",   ← AI prediction         │
│     "is_match": false,                                   │
│     "confidence": 0.95,                                  │
│     "rationale": "Visual evidence shows canine..."      │
│   }                                                      │
└──────────────────────────────────────────────────────────┘
```

## Data Flow

```
INPUT                     PROCESS                      OUTPUT
─────                     ───────                      ──────

┌───────────────┐
│   COCO JSON   │──┐
│   (annotations│  │
│    + images)  │  │      ┌─────────────────────┐
└───────────────┘  ├─────▶│    Data Loader      │
                   │      │    Parse & Link     │
┌───────────────┐  │      └──────────┬──────────┘
│    Images     │──┘                 │
│   (jpg/png)   │                    │
└───────────────┘                    │
                                     ▼
                          ┌─────────────────────┐
                          │    Crop Generator   │
                          │    Extract objects  │
                          └──────────┬──────────┘
                                     │
┌───────────────┐                    │
│  Guidelines   │────────────────────┤
│  PDF (opt.)   │                    │
└───────────────┘                    │
       │                             │
       ▼                             ▼
┌─────────────────┐       ┌─────────────────────┐
│   Guidelines    │──────▶│     Validator       │
│   Extractor     │       │   (AI Vision)       │
│                 │       │                     │
│ Label defs ──▶  │       │  • Build prompts    │
└─────────────────┘       │  • Call AI model    │
                          │  • Parse responses  │
                          │  • Compare labels   │
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────────────────┐
                          │           OUTPUTS               │
                          │                                 │
                          │   ~/.qc_pipeline/output/        │
                          │   ├── qc_results.json          │
                          │   ├── crops/                   │
                          │   │   ├── crop_123_dog.png     │
                          │   │   └── crop_456_cat.png     │
                          │   └── validation_prompts.log   │
                          │                                 │
                          │   Summary:                      │
                          │   • Matches: 92 (92%)          │
                          │   • Mismatches: 8              │
                          │   • Avg confidence: 0.89       │
                          └─────────────────────────────────┘
```

## Provider Abstraction

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BaseValidator (ABC)                           │
│                                                                      │
│   Common functionality:                                              │
│   • Prompt templates                                                │
│   • Image encoding (base64)                                         │
│   • Response parsing (JSON)                                         │
│   • Retry logic                                                     │
│                                                                      │
│   Abstract methods (provider-specific):                             │
│   • _create_llm()                                                   │
│   • _build_multimodal_message()                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│   │     Gemini      │  │     OpenAI      │  │    Anthropic    │    │
│   │                 │  │                 │  │                 │    │
│   │ langchain-      │  │ langchain-      │  │ langchain-      │    │
│   │ google-genai    │  │ openai          │  │ anthropic       │    │
│   │                 │  │                 │  │                 │    │
│   │ • gemini-3-     │  │ • gpt-5-mini    │  │ • claude-opus   │    │
│   │   flash-preview │  │ • gpt-5.2-pro   │  │ • claude-haiku  │    │
│   │ • gemini-3-pro  │  │ • gpt-5.2       │  │ • claude-sonnet │    │
│   │ • gemini-2.5    │  │                 │  │                 │    │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Technologies

| Layer | Technology |
|-------|------------|
| **Web UI** | Gradio |
| **State Machine** | LangGraph |
| **AI Integration** | LangChain |
| **Data Format** | COCO JSON |
| **Image Processing** | Pillow |
| **PDF Processing** | pypdf |
| **Data Validation** | Pydantic |

## Key Design Decisions

1. **LangChain for AI providers** - Unified interface across Gemini, OpenAI, Anthropic
2. **LangGraph for workflow** - Stateful multi-agent orchestration
3. **Protocol-based abstraction** - Easy to add new providers/sources
4. **Lazy loading** - Only import what's needed
5. **Gradio for UI** - Quick to build, good UX

## See Also

- [QC Pipeline Architecture](qc_pipeline/ARCHITECTURE.md) - Detailed component docs
- [QC Pipeline README](qc_pipeline/README.md) - Usage and API
- [Labellerr Integration](demo/labellerr-integration/README.md) - Platform setup
