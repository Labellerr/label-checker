#!/usr/bin/env python3
"""
End-to-end demo script for QC pipeline.

Loads test dataset, crops annotations, validates with Gemini (if API key available),
and generates a comprehensive report with confidence scores.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

# Add parent directory to path to import qc_pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

from qc_pipeline.data_loader import load_coco_dataset
from qc_pipeline.gemini_validator_demo import create_demo_validator
from qc_pipeline.image_utils import CropConfig, crop_annotation, image_to_png_bytes


@dataclass
class DemoResult:
    """Result for a single annotation validation."""
    image_id: int
    annotation_id: int
    image_file: str
    label: str
    crop_path: str
    confidence: Optional[float]
    gemini_response: Optional[str]
    error: Optional[str]


@dataclass
class DemoSummary:
    """Summary statistics for the demo run."""
    total_annotations: int
    total_crops_saved: int
    gemini_validations: int
    average_confidence: Optional[float]
    confidence_by_category: Dict[str, float]


def save_crop(crop: Image.Image, output_dir: Path, annotation_id: int, label: str) -> Path:
    """Save a crop image to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize label for filename
    safe_label = label.replace(" ", "_").replace("/", "_")
    crop_filename = f"crop_{annotation_id}_{safe_label}.png"
    crop_path = output_dir / crop_filename
    crop.save(crop_path)
    return crop_path


def run_demo(
    dataset_dir: Path,
    coco_json_name: str,
    output_dir: Path,
    use_gemini: bool = True,
    api_key: Optional[str] = None,
) -> None:
    """Run the end-to-end demo."""
    print("=" * 70)
    print("QC Pipeline Demo - End-to-End Validation")
    print("=" * 70)
    
    # Load dataset
    coco_path = dataset_dir / coco_json_name
    print(f"\n[1/5] Loading dataset from: {coco_path}")
    dataset = load_coco_dataset(coco_path)
    print(f"  ✓ Loaded {len(dataset.annotations)} annotations from {len(dataset.images)} images")
    print(f"  ✓ Categories: {', '.join(cat.name for cat in dataset.categories.values())}")
    
    # Initialize Gemini validator if requested
    validator = None
    if use_gemini:
        if api_key:
            print(f"\n[2/5] Initializing Gemini validator...")
            validator = create_demo_validator(
                api_key=api_key,
                model_name="gemini-2.5-flash",
                temperature=0.2,
                max_retries=3,
            )
            print("  ✓ Gemini validator ready (demo prompt: 'Is this a [label]?')")
        else:
            print(f"\n[2/5] Skipping Gemini validation (no API key provided)")
    else:
        print(f"\n[2/5] Gemini validation disabled")
    
    # Process annotations
    print(f"\n[3/5] Processing annotations and generating crops...")
    results: List[DemoResult] = []
    crops_dir = output_dir / "crops"
    image_cache: Dict[int, Image.Image] = {}
    
    try:
        for idx, (image_info, annotation, category) in enumerate(dataset.iter_annotations(), 1):
            print(f"  [{idx}/{len(dataset.annotations)}] Processing annotation {annotation.id} ({category.name})...", end=" ")
            
            # Load image (with caching)
            if image_info.id not in image_cache:
                image_path = dataset_dir / image_info.file_name
                with Image.open(image_path) as img:
                    image_cache[image_info.id] = img.convert("RGB")
            
            base_image = image_cache[image_info.id]
            
            # Generate crop
            try:
                crop_config = CropConfig(padding=8, mask_polygon=True)
                crop = crop_annotation(base_image, annotation, crop_config)
                crop_path = save_crop(crop, crops_dir, annotation.id, category.name)
                
                # Validate with Gemini if available
                confidence = None
                gemini_response = None
                error = None
                
                if validator:
                    try:
                        crop_bytes = image_to_png_bytes(crop)
                        response = validator.validate_crop(
                            crop_bytes=crop_bytes,
                            expected_label=category.name,
                            guidelines=None,
                        )
                        confidence = response.confidence
                        gemini_response = response.raw_text
                        print(f"✓ (confidence: {confidence:.2f})")
                    except Exception as e:
                        error = str(e)
                        print(f"✗ (error: {error})")
                else:
                    print("✓")
                
                results.append(DemoResult(
                    image_id=image_info.id,
                    annotation_id=annotation.id,
                    image_file=image_info.file_name,
                    label=category.name,
                    crop_path=str(crop_path.relative_to(output_dir)),
                    confidence=confidence,
                    gemini_response=gemini_response,
                    error=error,
                ))
            except Exception as e:
                print(f"✗ (crop error: {e})")
                results.append(DemoResult(
                    image_id=image_info.id,
                    annotation_id=annotation.id,
                    image_file=image_info.file_name,
                    label=category.name,
                    crop_path="",
                    confidence=None,
                    gemini_response=None,
                    error=str(e),
                ))
    finally:
        # Close cached images
        for img in image_cache.values():
            img.close()
    
    # Generate summary
    print(f"\n[4/5] Generating summary statistics...")
    successful_crops = [r for r in results if r.crop_path]
    gemini_results = [r for r in results if r.confidence is not None]
    
    avg_confidence = None
    if gemini_results:
        avg_confidence = sum(r.confidence for r in gemini_results) / len(gemini_results)
    
    # Confidence by category
    confidence_by_cat: Dict[str, List[float]] = {}
    for r in gemini_results:
        if r.label not in confidence_by_cat:
            confidence_by_cat[r.label] = []
        confidence_by_cat[r.label].append(r.confidence)
    
    confidence_by_category = {
        label: sum(scores) / len(scores)
        for label, scores in confidence_by_cat.items()
    }
    
    summary = DemoSummary(
        total_annotations=len(results),
        total_crops_saved=len(successful_crops),
        gemini_validations=len(gemini_results),
        average_confidence=avg_confidence,
        confidence_by_category=confidence_by_category,
    )
    
    # Save results
    print(f"\n[5/5] Saving results to: {output_dir / 'demo_results.json'}")
    output = {
        "results": [asdict(r) for r in results],
        "summary": asdict(summary),
    }
    
    results_path = output_dir / "demo_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("  ✓ Results saved")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total annotations:     {summary.total_annotations}")
    print(f"Crops saved:           {summary.total_crops_saved}")
    print(f"Gemini validations:    {summary.gemini_validations}")
    if summary.average_confidence is not None:
        print(f"Average confidence:    {summary.average_confidence:.3f}")
    if summary.confidence_by_category:
        print("\nConfidence by category:")
        for label, conf in summary.confidence_by_category.items():
            print(f"  {label:20s} {conf:.3f}")
    print("=" * 70)
    print(f"\n✓ Demo complete! Check {output_dir} for results.")


def main() -> None:
    """Main entry point."""
    # Configuration
    dataset_dir = Path(__file__).parent.parent / "test_dataset"
    coco_json = "export-#6w0PnZX02NffcYns8L1C.json"
    output_dir = Path(__file__).parent / "output"
    
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    use_gemini = api_key is not None
    
    if not use_gemini:
        print("ℹ️  GEMINI_API_KEY not set - running without Gemini validation")
        print("   Set GEMINI_API_KEY environment variable to enable validation\n")
    
    run_demo(
        dataset_dir=dataset_dir,
        coco_json_name=coco_json,
        output_dir=output_dir,
        use_gemini=use_gemini,
        api_key=api_key,
    )


if __name__ == "__main__":
    main()

