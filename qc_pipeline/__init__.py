"""
QC Validation Pipeline - Multi-provider annotation quality validation.

This package provides tools for validating annotation quality using
AI vision models from multiple providers (Gemini, OpenAI, Anthropic).

Main components:
- validators: Multi-provider LLM validators for image validation
- fetchers: Data fetchers for various platforms (local, Labellerr)
- data_loader: COCO dataset loading utilities
- image_utils: Image cropping and processing
- qc_workflow: Orchestration of the validation workflow
- guidelines_extractor: Extract structured guidelines from PDFs
- app: Gradio web UI for the pipeline
"""

# Core data loading
from .data_loader import CocoDataset, load_coco_dataset  # noqa: F401

# Image utilities
from .image_utils import CropConfig, crop_annotation, image_to_png_bytes  # noqa: F401

# Workflow
from .qc_workflow import QCValidationWorkflow, ValidationResult, ValidationSummary  # noqa: F401

# Guidelines extraction
from .guidelines_extractor import (  # noqa: F401
    GuidelinesExtractor,
    LabelDefinition,
    ExtractedGuidelines,
    create_guidelines_extractor,
    PROVIDER_MODELS,
)

# Validators (lazy imports available)
from .validators import (  # noqa: F401
    BaseValidator,
    ValidationResponse,
    PROVIDERS,
    create_validator,
)

# Fetchers (lazy imports available)
from .fetchers import (  # noqa: F401
    DataFetcherProtocol,
    FetchResult,
    PLATFORMS,
    LocalFileFetcher,
    create_local_fetcher,
)

# Legacy Gemini validator (for backward compatibility)
from .gemini_validator import GeminiValidator, GeminiResponse, create_demo_validator  # noqa: F401

__all__ = [
    # Data loading
    "CocoDataset",
    "load_coco_dataset",
    # Image utilities
    "CropConfig",
    "crop_annotation",
    "image_to_png_bytes",
    # Workflow
    "QCValidationWorkflow",
    "ValidationResult",
    "ValidationSummary",
    # Guidelines
    "GuidelinesExtractor",
    "LabelDefinition",
    "ExtractedGuidelines",
    "create_guidelines_extractor",
    "PROVIDER_MODELS",
    # Validators
    "BaseValidator",
    "ValidationResponse",
    "PROVIDERS",
    "create_validator",
    # Fetchers
    "DataFetcherProtocol",
    "FetchResult",
    "PLATFORMS",
    "LocalFileFetcher",
    "create_local_fetcher",
    # Legacy
    "GeminiValidator",
    "GeminiResponse",
    "create_demo_validator",
]
