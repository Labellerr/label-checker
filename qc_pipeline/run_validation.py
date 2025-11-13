from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from .data_loader import CocoDataset, load_coco_dataset
from .gemini_validator import GeminiValidator
from .image_utils import CropConfig, crop_annotation, image_to_png_bytes


@dataclass
class ValidationEntry:
    image_id: int
    annotation_id: int
    expected_label: str
    prediction_label: str
    confidence: float
    is_match: bool
    rationale: Optional[str]
    bbox: Optional[list]
    polygon: Optional[list]


@dataclass
class ValidationSummary:
    total: int
    matches: int
    mismatches: int
    accuracy: float
    average_confidence: float


def _load_guidelines(path: Optional[Path]) -> Optional[str]:
    if not path:
        return None
    return path.read_text(encoding="utf-8")


def _match_labels(predicted: str, expected: str) -> bool:
    return predicted.strip().lower() == expected.strip().lower()


def _build_summary(entries: List[ValidationEntry]) -> ValidationSummary:
    total = len(entries)
    matches = sum(1 for entry in entries if entry.is_match)
    mismatches = total - matches
    avg_conf = sum(entry.confidence for entry in entries) / total if total else 0.0
    accuracy = matches / total if total else 0.0
    return ValidationSummary(
        total=total,
        matches=matches,
        mismatches=mismatches,
        accuracy=accuracy,
        average_confidence=avg_conf,
    )


def _serialize(entries: List[ValidationEntry], summary: ValidationSummary) -> Dict[str, object]:
    return {
        "entries": [asdict(entry) for entry in entries],
        "summary": asdict(summary),
    }


def run_validation(
    dataset: CocoDataset,
    images_root: Path,
    validator: GeminiValidator,
    guidelines: Optional[str],
    crop_config: CropConfig,
) -> Dict[str, object]:
    entries: List[ValidationEntry] = []
    image_cache: Dict[int, Image.Image] = {}
    try:
        for image_info, annotation, category in dataset.iter_annotations():
            if image_info.id not in image_cache:
                image_path = image_info.full_path(images_root)
                with Image.open(image_path) as img:
                    image_cache[image_info.id] = img.convert("RGB")

            base_image = image_cache[image_info.id]
            crop = crop_annotation(base_image, annotation, crop_config)
            crop_bytes = image_to_png_bytes(crop)

            response = validator.validate_crop(
                crop_bytes=crop_bytes,
                expected_label=category.name,
                guidelines=guidelines,
            )

            entries.append(
                ValidationEntry(
                    image_id=image_info.id,
                    annotation_id=annotation.id,
                    expected_label=category.name,
                    prediction_label=response.prediction_label,
                    confidence=response.confidence,
                    is_match=_match_labels(response.prediction_label, category.name),
                    rationale=response.rationale,
                    bbox=list(annotation.bbox) if annotation.bbox else None,
                    polygon=[list(seg) for seg in (annotation.segmentation or [])]
                    if annotation.segmentation
                    else None,
                )
            )
    finally:
        for image in image_cache.values():
            image.close()

    summary = _build_summary(entries)
    return _serialize(entries, summary)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemini QC validation on annotated crops.")
    parser.add_argument("--images-dir", required=True, type=Path, help="Directory containing source images.")
    parser.add_argument("--coco-json", required=True, type=Path, help="Path to COCO annotations JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Path to write validation results (JSON).")
    parser.add_argument("--guidelines", type=Path, help="Optional text file with labeling guidelines.")
    parser.add_argument(
        "--api-key-env",
        default="GEMINI_API_KEY",
        help="Environment variable that stores the Gemini API key (default: GEMINI_API_KEY).",
    )
    parser.add_argument("--model-name", default="gemini-1.5-pro", help="Gemini model name to invoke.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Optional nucleus sampling parameter.")
    parser.add_argument("--padding", type=int, default=4, help="Pixel padding added around crops.")
    parser.add_argument(
        "--disable-polygon-mask",
        action="store_true",
        help="Disable polygon masking, returning rectangular crops only.",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retry attempts for Gemini API calls.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout for Gemini API requests (seconds).")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {args.api_key_env} is not set.")

    dataset = load_coco_dataset(args.coco_json)
    validator = GeminiValidator(
        api_key=api_key,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_retries=args.max_retries,
        timeout_seconds=args.timeout,
    )
    guidelines = _load_guidelines(args.guidelines)
    crop_config = CropConfig(
        padding=args.padding,
        mask_polygon=not args.disable_polygon_mask,
    )

    result = run_validation(dataset, args.images_dir, validator, guidelines, crop_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

