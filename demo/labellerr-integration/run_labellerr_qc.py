#!/usr/bin/env python3
"""
Gradio UI for Labellerr SDK integration with QC pipeline.

Allows users to fetch annotations from Labellerr by status,
download images, and run Gemini validation on the annotations.
"""
from __future__ import annotations

import json
import logging
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
from qc_pipeline.pdf_utils import extract_text_from_pdf
from qc_pipeline.guidelines_extractor import GuidelinesExtractor
from qc_pipeline.qc_workflow import QCValidationWorkflow

from labellerr_fetcher import fetch_labellerr_data

# Configure logging
logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path) -> Path:
    """
    Setup logging to both console and file.
    
    Args:
        output_dir: Directory to save log file
        
    Returns:
        Path to log file
    """
    log_file = output_dir / "gemini_prompts.log"
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    
    return log_file


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


def process_guidelines_pdf(
    pdf_file,
    gemini_api_key: str,
) -> Tuple[str, dict]:
    """
    Process uploaded PDF and extract annotation guidelines.
    
    Args:
        pdf_file: Gradio file upload object
        gemini_api_key: Gemini API key for extraction
        
    Returns:
        Tuple of (status_message, guidelines_dict)
    """
    if not pdf_file:
        return "⚠️ Please upload a PDF file", {}
    
    if not gemini_api_key:
        return "⚠️ Please provide Gemini API key first", {}
    
    try:
        status = "📄 Extracting text from PDF..."
        print(status)
        
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(Path(pdf_file.name))
        
        if not pdf_text.strip():
            return "❌ Error: PDF appears to be empty or unreadable", {}
        
        status = " Extracting label definitions using Gemini..."
        print(status)
        
        # Extract structured guidelines using Gemini
        extractor = GuidelinesExtractor(api_key=gemini_api_key)
        guidelines = extractor.extract(pdf_text)
        
        # Convert to dict for easy lookup: {label_name: definition_dict}
        guidelines_dict = {
            label.label_name.lower(): label.model_dump()
            for label in guidelines.labels
        }
        
        # Save extracted guidelines to file for inspection
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        guidelines_json_path = output_dir / "extracted_guidelines.json"
        with open(guidelines_json_path, "w", encoding="utf-8") as f:
            json.dump(guidelines_dict, f, indent=2)
        print(f"Saved extracted guidelines to: {guidelines_json_path}")
        
        # Build detailed success message showing what was extracted
        label_names = list(guidelines_dict.keys())
        status_msg = f"✅ Successfully extracted {len(label_names)} labels:\n\n"
        
        # Show details for each label
        for label_name, label_info in guidelines_dict.items():
            status_msg += f"📌 {label_name.upper()}\n"
            status_msg += f"   Definition: {label_info.get('definition', 'N/A')[:150]}...\n"
            
            characteristics = label_info.get('key_characteristics', [])
            if characteristics:
                status_msg += f"   Visual Features ({len(characteristics)}):\n"
                for char in characteristics[:3]:  # Show first 3
                    status_msg += f"      • {char[:100]}...\n"
                if len(characteristics) > 3:
                    status_msg += f"      ... and {len(characteristics) - 3} more\n"
            else:
                status_msg += "   ⚠️ No visual characteristics extracted!\n"
            
            status_msg += "\n"
        
        status_msg += (
            f"✅ Full extraction saved to: {guidelines_json_path}\n\n"
            "These guidelines will be used during validation to provide "
            "Gemini with precise label definitions and visual characteristics."
        )
        
        print(f"Extracted guidelines for labels: {label_names}")
        print(f"Full guidelines saved to: {guidelines_json_path}")
        return status_msg, guidelines_dict
    
    except Exception as e:
        error_msg = f"❌ Error processing PDF: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, {}


def run_qc_validation(
    api_key: str,
    api_secret: str,
    client_id: str,
    project_id: str,
    gemini_api_key: str,
    statuses: List[str],
    max_files: int = 50,
    confidence_threshold: float = 0.5,
    export_timeout: int = 900,
    guidelines_dict: dict = None,
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
        export_timeout: Maximum time to wait for export completion in seconds (default: 900 = 15 minutes)
        guidelines_dict: Optional dict of label definitions from PDF guidelines
        
    Returns:
        Tuple of (status_message, results_json, summary_text, low_confidence_gallery)
    """
    # Default to empty dict if None
    if guidelines_dict is None:
        guidelines_dict = {}
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
        
        # Setup logging to capture prompts
        log_file = setup_logging(output_dir)
        logger.info("Starting QC validation run")
        logger.info(f"Guidelines provided: {len(guidelines_dict)} labels")
        if guidelines_dict:
            logger.info(f"Guidelines labels: {list(guidelines_dict.keys())}")
        
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
            export_timeout=export_timeout,
        )
        
        status_msg += f"✓ Downloaded {metadata['total_images']} images\n"
        status_msg += f"✓ Annotations saved to: {coco_json_path}\n\n"
        
        # Step 2: Run LangGraph workflow for validation
        status_msg += " Starting QC Validation Workflow...\n"
        print(status_msg)
        
        # Create workflow
        workflow = QCValidationWorkflow(
            gemini_api_key=gemini_api_key,
            output_dir=output_dir,
        )
        
        # Determine PDF path - for now we don't support PDF upload in this flow
        # Guidelines are extracted separately via process_guidelines_pdf
        pdf_path = None
        
        # Run workflow with pre-extracted guidelines if available
        final_state, results, summary = workflow.run(
            pdf_path=pdf_path,
            coco_json_path=coco_json_path,
            images_dir=images_dir,
            max_files=max_files,
            confidence_threshold=confidence_threshold,
            pre_extracted_guidelines=guidelines_dict if guidelines_dict else None,
        )
        
        # Check for workflow errors
        if final_state.get("error"):
            return f"❌ Workflow error: {final_state['error']}", "", "", []
        
        # Step 3: Format final status message
        status_msg += "=" * 70 + "\n"
        status_msg += "✅ QC VALIDATION COMPLETE\n"
        status_msg += "=" * 70 + "\n"
        
        logger.info("QC validation complete")
        logger.info(f"Processed {summary.total_annotations} annotations")
        logger.info(f"Average confidence: {summary.average_confidence}")
        
        # Format summary text
        avg_conf_str = f"{summary.average_confidence:.3f}" if summary.average_confidence else "N/A"
        summary_text = f"""
📊 SUMMARY STATISTICS
{'=' * 70}
Total annotations:        {summary.total_annotations}
Crops saved:              {summary.total_crops_saved}
Gemini validations:       {summary.gemini_validations}
Average confidence:       {avg_conf_str}
Low confidence count:     {len(summary.low_confidence_annotations)} (< {confidence_threshold})

Confidence by category:
"""
        for label, conf in summary.confidence_by_category.items():
            summary_text += f"  • {label:30s} {conf:.3f}\n"
        
        # Prepare low confidence gallery
        low_conf_gallery = []
        for item in summary.low_confidence_annotations[:20]:  # Limit to 20 images
            crop_path = item["crop_path"]
            if Path(crop_path).exists():
                caption = f"{item['label']} (conf: {item['confidence']:.2f})"
                low_conf_gallery.append((crop_path, caption))
        
        # Load results JSON for display
        results_path = output_dir / "qc_results.json"
        with open(results_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)
        
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
        # Session state for guidelines (temporary, per-session)
        guidelines_state = gr.State(value={})
        
        gr.Markdown("""
        # 🔍 Labellerr QC Validation Pipeline
        
        Fetch annotations from Labellerr by status and validate them using Gemini AI.
        
        **Workflow:**
        1. Enter your Labellerr and Gemini credentials
        2. (Optional) Upload annotation guidelines PDF
        3. Select annotation statuses to validate
        4. Click "Fetch & Run QC" to start validation
        5. Review results and low-confidence annotations
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
                
                gr.Markdown("###  Gemini API")
                gemini_api_key = gr.Textbox(
                    label="Gemini API Key",
                    placeholder="Enter Gemini API Key",
                    type="password",
                )
                
                # Annotation Guidelines (Optional)
                with gr.Accordion("📋 Annotation Guidelines (Optional)", open=False):
                    gr.Markdown(
                        "Upload a PDF with annotation guidelines to improve validation accuracy. "
                        "Gemini will use label definitions and characteristics from the PDF."
                    )
                    
                    pdf_upload = gr.File(
                        label="Guidelines PDF",
                        file_types=[".pdf"],
                        file_count="single",
                    )
                    process_guidelines_btn = gr.Button(
                        "📄 Process Guidelines",
                        variant="secondary",
                        size="sm",
                    )
                    guidelines_status = gr.Textbox(
                        label="Guidelines Status",
                        interactive=False,
                        lines=4,
                        placeholder="Upload a PDF and click 'Process Guidelines' to extract label definitions..."
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
                
                export_timeout = gr.Number(
                    label="Export Timeout (seconds)",
                    value=900,
                    minimum=60,
                    maximum=3600,
                    step=60,
                    info="Maximum time to wait for export completion (default: 900 = 15 minutes)",
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
        
        # Connect guidelines processing button
        process_guidelines_btn.click(
            fn=process_guidelines_pdf,
            inputs=[
                pdf_upload,
                gemini_api_key,
            ],
            outputs=[
                guidelines_status,
                guidelines_state,
            ],
        )
        
        # Connect validation button
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
                export_timeout,
                guidelines_state,  # Pass session guidelines
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
        - **Optional**: Upload annotation guidelines PDF to provide Gemini with precise label definitions
        - Guidelines are session-scoped (cleared when browser closes)
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

