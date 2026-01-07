#!/usr/bin/env python3
"""
Gradio UI for Labellerr SDK integration with QC pipeline.

Allows users to fetch annotations from Labellerr by status,
download images, and run Gemini validation on the annotations.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qc_pipeline.data_loader import load_coco_dataset
from qc_pipeline.gemini_validator import create_demo_validator
from qc_pipeline.image_utils import CropConfig, crop_annotation, image_to_png_bytes

from labellerr_fetcher import fetch_labellerr_data


@dataclass
class ValidationResult:
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
class ValidationSummary:
    """Summary statistics for validation run."""
    total_annotations: int
    total_crops_saved: int
    gemini_validations: int
    average_confidence: Optional[float]
    confidence_by_category: Dict[str, float]
    low_confidence_annotations: List[Dict]


def run_qc_validation(
    api_key: str,
    api_secret: str,
    client_id: str,
    project_id: str,
    gemini_api_key: str,
    statuses: List[str],
    max_files: int = 50,
    confidence_threshold: float = 0.5,
) -> Tuple[str, str, str, List[Tuple[str, str]]]:
    """
    Main function to run QC validation on Labellerr data.
    
    Args:
        api_key: Labellerr API key
        api_secret: Labellerr API secret
        client_id: Labellerr client ID
        project_id: Labellerr project ID
        gemini_api_key: Gemini API key
        statuses: List of annotation statuses to filter
        max_files: Maximum number of files to process
        confidence_threshold: Threshold for flagging low confidence
        
    Returns:
        Tuple of (status_message, results_json, summary_text, low_confidence_gallery)
    """
    try:
        # Validate inputs
        if not all([api_key, api_secret, client_id, project_id]):
            return "❌ Error: All Labellerr credentials are required", "", "", []
        
        if not gemini_api_key:
            return "❌ Error: Gemini API key is required", "", "", []
        
        if not statuses:
            return "❌ Error: Please select at least one annotation status", "", "", []
        
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Fetch data from Labellerr
        status_msg = "📥 Fetching data from Labellerr...\n"
        print(status_msg)
        
        coco_json_path, images_dir, metadata = fetch_labellerr_data(
            api_key=api_key,
            api_secret=api_secret,
            client_id=client_id,
            project_id=project_id,
            statuses=statuses,
            output_dir=output_dir,
        )
        
        status_msg += f"✓ Downloaded {metadata['total_images']} images\n"
        status_msg += f"✓ Annotations saved to: {coco_json_path}\n\n"
        
        # Step 2: Load dataset
        status_msg += "📂 Loading COCO dataset...\n"
        print(status_msg)
        
        dataset = load_coco_dataset(coco_json_path)
        status_msg += f"✓ Loaded {len(dataset.annotations)} annotations from {len(dataset.images)} images\n"
        status_msg += f"✓ Categories: {', '.join(cat.name for cat in dataset.categories.values())}\n\n"
        
        # Step 3: Initialize Gemini validator
        status_msg += "🤖 Initializing Gemini validator...\n"
        print(status_msg)
        
        validator = create_demo_validator(
            api_key=gemini_api_key,
            model_name="gemini-2.0-flash-exp",
            temperature=0.2,
            max_retries=3,
        )
        status_msg += "✓ Gemini validator ready\n\n"
        
        # Step 4: Process annotations
        status_msg += f"🔍 Processing annotations (max {max_files})...\n"
        print(status_msg)
        
        results: List[ValidationResult] = []
        crops_dir = output_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        
        image_cache: Dict[int, Image.Image] = {}
        
        try:
            annotations_to_process = list(dataset.iter_annotations())[:int(max_files)]
            
            for idx, (image_info, annotation, category) in enumerate(annotations_to_process, 1):
                print(f"  [{idx}/{len(annotations_to_process)}] Processing {category.name}...", end=" ")
                
                # Load image (with caching)
                if image_info.id not in image_cache:
                    image_path = images_dir / image_info.file_name
                    if not image_path.exists():
                        print(f"✗ (image not found: {image_path})")
                        continue
                    
                    with Image.open(image_path) as img:
                        image_cache[image_info.id] = img.convert("RGB")
                
                base_image = image_cache[image_info.id]
                
                # Generate crop
                try:
                    crop_config = CropConfig(padding=8, mask_polygon=True)
                    crop = crop_annotation(base_image, annotation, crop_config)
                    
                    # Save crop
                    safe_label = category.name.replace(" ", "_").replace("/", "_")
                    crop_filename = f"crop_{annotation.id}_{safe_label}.png"
                    crop_path = crops_dir / crop_filename
                    crop.save(crop_path)
                    
                    # Validate with Gemini
                    confidence = None
                    gemini_response = None
                    error = None
                    
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
                        print(f"✗ (validation error: {error})")
                    
                    results.append(ValidationResult(
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
                    results.append(ValidationResult(
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
        
        # Step 5: Generate summary
        status_msg += f"\n📊 Generating summary...\n"
        
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
        
        # Find low confidence annotations
        low_confidence = [
            {
                "annotation_id": r.annotation_id,
                "label": r.label,
                "confidence": r.confidence,
                "crop_path": str(output_dir / r.crop_path),
            }
            for r in gemini_results
            if r.confidence < confidence_threshold
        ]
        
        summary = ValidationSummary(
            total_annotations=len(results),
            total_crops_saved=len(successful_crops),
            gemini_validations=len(gemini_results),
            average_confidence=avg_confidence,
            confidence_by_category=confidence_by_category,
            low_confidence_annotations=low_confidence,
        )
        
        # Save results
        results_path = output_dir / "qc_results.json"
        output_data = {
            "metadata": metadata,
            "results": [asdict(r) for r in results],
            "summary": asdict(summary),
        }
        
        results_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
        
        status_msg += f"✓ Results saved to: {results_path}\n\n"
        status_msg += "=" * 70 + "\n"
        status_msg += "✅ QC VALIDATION COMPLETE\n"
        status_msg += "=" * 70 + "\n"
        
        # Format summary text
        avg_conf_str = f"{summary.average_confidence:.3f}" if summary.average_confidence else "N/A"
        summary_text = f"""
📊 SUMMARY STATISTICS
{'=' * 70}
Total annotations:        {summary.total_annotations}
Crops saved:              {summary.total_crops_saved}
Gemini validations:       {summary.gemini_validations}
Average confidence:       {avg_conf_str}
Low confidence count:     {len(low_confidence)} (< {confidence_threshold})

Confidence by category:
"""
        for label, conf in summary.confidence_by_category.items():
            summary_text += f"  • {label:30s} {conf:.3f}\n"
        
        # Prepare low confidence gallery
        low_conf_gallery = []
        for item in low_confidence[:20]:  # Limit to 20 images
            crop_path = item["crop_path"]
            if Path(crop_path).exists():
                caption = f"{item['label']} (conf: {item['confidence']:.2f})"
                low_conf_gallery.append((crop_path, caption))
        
        return status_msg, json.dumps(output_data, indent=2), summary_text, low_conf_gallery
    
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nPlease check your credentials and try again."
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return error_msg, "", "", []


def create_gradio_interface():
    """Create and configure Gradio interface."""
    
    with gr.Blocks(title="Labellerr QC Validation", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔍 Labellerr QC Validation Pipeline
        
        Fetch annotations from Labellerr by status and validate them using Gemini AI.
        
        **Workflow:**
        1. Enter your Labellerr and Gemini credentials
        2. Select annotation statuses to validate
        3. Click "Fetch & Run QC" to start validation
        4. Review results and low-confidence annotations
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔐 Labellerr Credentials")
                api_key = gr.Textbox(
                    label="API Key",
                    placeholder="Enter Labellerr API Key",
                    type="password",
                )
                api_secret = gr.Textbox(
                    label="API Secret",
                    placeholder="Enter Labellerr API Secret",
                    type="password",
                )
                client_id = gr.Textbox(
                    label="Client ID",
                    placeholder="Enter Client ID",
                )
                project_id = gr.Textbox(
                    label="Project ID",
                    placeholder="Enter Project ID",
                )
                
                gr.Markdown("### 🤖 Gemini API")
                gemini_api_key = gr.Textbox(
                    label="Gemini API Key",
                    placeholder="Enter Gemini API Key",
                    type="password",
                )
                
                gr.Markdown("### ⚙️ Settings")
                statuses = gr.CheckboxGroup(
                    choices=["review", "r_assigned", "client_review", "cr_assigned", "accepted"],
                    label="Annotation Statuses",
                    value=["accepted"],
                    info="Select which annotation statuses to validate",
                )
                
                with gr.Row():
                    max_files = gr.Number(
                        label="Max Files",
                        value=50,
                        minimum=1,
                        maximum=1000,
                        step=1,
                    )
                    confidence_threshold = gr.Slider(
                        label="Low Confidence Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                    )
                
                run_button = gr.Button("🚀 Fetch & Run QC", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results")
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=15,
                    max_lines=20,
                    show_label=True,
                )
                
                summary_output = gr.Textbox(
                    label="Summary Statistics",
                    lines=10,
                    show_label=True,
                )
                
                with gr.Accordion("📄 Full Results JSON", open=False):
                    results_json = gr.Textbox(
                        label="Results",
                        lines=20,
                        max_lines=30,
                        show_label=False,
                    )
                
                with gr.Accordion("⚠️ Low Confidence Annotations", open=True):
                    low_conf_gallery = gr.Gallery(
                        label="Low Confidence Crops",
                        show_label=True,
                        columns=4,
                        rows=2,
                        height="auto",
                    )
        
        # Connect button to function
        run_button.click(
            fn=run_qc_validation,
            inputs=[
                api_key,
                api_secret,
                client_id,
                project_id,
                gemini_api_key,
                statuses,
                max_files,
                confidence_threshold,
            ],
            outputs=[
                status_output,
                results_json,
                summary_output,
                low_conf_gallery,
            ],
        )
        
        gr.Markdown("""
        ---
        ### 📝 Notes
        - Export filters annotations by status, but downloads all images referenced by those annotations
        - Only filtered annotations are validated with Gemini
        - Results are saved to `demo/labellerr-integration/output/qc_results.json`
        - Crops are saved to `demo/labellerr-integration/output/crops/`
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )

