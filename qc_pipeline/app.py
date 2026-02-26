#!/usr/bin/env python3
"""
QC Pipeline - Gradio Web UI

A general-purpose annotation validation interface supporting:
- Multiple data sources (local upload, Labellerr)
- Multiple AI providers (Gemini, OpenAI, Anthropic)
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import gradio as gr

from qc_pipeline.pdf_utils import extract_text_from_pdf
from qc_pipeline.guidelines_extractor import GuidelinesExtractor
from qc_pipeline.qc_workflow import QCValidationWorkflow
from qc_pipeline.validators import PROVIDERS, create_validator
from qc_pipeline.fetchers import create_local_fetcher
from qc_pipeline.fetchers.base import FetchResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path.home() / ".qc_pipeline" / "output"


# =============================================================================
# Core Functions
# =============================================================================

def extract_guidelines(pdf_file, provider: str, api_key: str, model: str) -> Tuple[str, dict]:
    """Extract annotation guidelines from PDF using AI."""
    if not pdf_file:
        return "⚠️ Upload a PDF file first", {}
    if not api_key:
        return "⚠️ Enter API key in Model Configuration tab first", {}
    
    try:
        logger.info(f"Extracting guidelines with {provider}/{model}")
        pdf_text = extract_text_from_pdf(Path(pdf_file.name))
        
        if not pdf_text.strip():
            return "❌ PDF is empty or unreadable", {}
        
        extractor = GuidelinesExtractor(api_key=api_key, provider=provider, model=model)
        guidelines = extractor.extract(pdf_text)
        
        guidelines_dict = {
            label.label_name.lower(): label.model_dump()
            for label in guidelines.labels
        }
        
        # Format success message
        msg = f"✅ Extracted {len(guidelines_dict)} labels:\n\n"
        for name, info in guidelines_dict.items():
            msg += f"**{name.upper()}**\n"
            msg += f"  {info.get('definition', 'N/A')[:100]}...\n\n"
        
        return msg, guidelines_dict
        
    except Exception as e:
        logger.error(f"Guidelines extraction failed: {e}", exc_info=True)
        return f"❌ Error: {e}", {}


def fetch_data(
    source: str,
    # Upload params
    images, coco_json, images_zip,
    # Labellerr params  
    ll_key, ll_secret, ll_client, ll_project, ll_statuses, ll_timeout,
    output_dir: Path,
) -> Tuple[FetchResult, str]:
    """Fetch data from selected source."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if source == "upload":
        if not coco_json and not images and not images_zip:
            raise ValueError("Upload COCO JSON and/or images")
        
        fetcher = create_local_fetcher(output_dir)
        result = fetcher.fetch_data(
            output_dir=output_dir,
            image_files=[Path(f.name) for f in images] if images else [],
            coco_json_path=Path(coco_json.name) if coco_json else None,
            images_zip_path=Path(images_zip.name) if images_zip else None,
        )
        return result, f"✅ Loaded {result.metadata.get('total_images', 0)} images"
    
    elif source == "labellerr":
        from qc_pipeline.fetchers.labellerr import create_labellerr_fetcher, is_labellerr_available
        
        if not is_labellerr_available():
            raise ImportError("Labellerr SDK not installed. Run: pip install labellerr")
        
        if not all([ll_key, ll_secret, ll_client, ll_project]):
            raise ValueError("All Labellerr credentials required")
        if not ll_statuses:
            raise ValueError("Select at least one annotation status")
        
        fetcher = create_labellerr_fetcher(
            api_key=ll_key, api_secret=ll_secret,
            client_id=ll_client, project_id=ll_project,
        )
        result = fetcher.fetch_data(
            output_dir=output_dir,
            statuses=ll_statuses,
            export_timeout=ll_timeout,
        )
        return result, f"✅ Fetched {result.metadata.get('total_images', 0)} images from Labellerr"
    
    raise ValueError(f"Unknown source: {source}")


def run_validation(
    # Data source
    source, images, coco_json, images_zip,
    ll_key, ll_secret, ll_client, ll_project, ll_statuses, ll_timeout,
    # Model
    provider, api_key, model,
    # Settings
    max_files, confidence_threshold,
    # Guidelines
    guidelines_dict,
) -> Tuple[str, str, list, list]:
    """Run QC validation pipeline."""
    guidelines_dict = guidelines_dict or {}
    
    if not api_key:
        return "❌ API key required", "", [], []
    
    try:
        output_dir = DEFAULT_OUTPUT_DIR / "run"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Fetch data
        logger.info("📥 Fetching data...")
        try:
            result, fetch_msg = fetch_data(
                source, images, coco_json, images_zip,
                ll_key, ll_secret, ll_client, ll_project, ll_statuses, ll_timeout,
                output_dir,
            )
        except Exception as e:
            error = f"❌ Fetch error: {e}"
            if "500" in str(e).lower():
                error += "\n\n⚠️ Server error - try again in a few minutes"
            return error, "", [], []
        
        # 2. Create validator
        logger.info(f"Creating validator: {provider}/{model}")
        try:
            validator = create_validator(provider, api_key, model, temperature=0.1)
        except ImportError as e:
            return f"❌ {e}", "", [], []
        
        # 3. Run workflow
        logger.info("Starting QC workflow")
        workflow = QCValidationWorkflow(output_dir=output_dir, validator=validator)
        
        final_state, results, summary = workflow.run(
            pdf_path=None,
            coco_json_path=result.coco_json_path,
            images_dir=result.images_dir,
            max_files=int(max_files),
            confidence_threshold=confidence_threshold,
            pre_extracted_guidelines=guidelines_dict or None,
        )
        
        if final_state.get("error"):
            return f"❌ {final_state['error']}", "", [], []
        
        # 4. Format results
        accuracy = (summary.matches / summary.gemini_validations * 100) if summary.gemini_validations > 0 else 0
        avg_conf = f"{summary.average_confidence:.3f}" if summary.average_confidence else "N/A"
        
        status = f"""✅ **Validation Complete**

{fetch_msg}
Validated {summary.gemini_validations} annotations
"""
        
        summary_text = f"""## Summary

| Metric | Value |
|--------|-------|
| Total Annotations | {summary.total_annotations} |
| Validations | {summary.gemini_validations} |
| Matches | {summary.matches} ({accuracy:.1f}%) |
| Mismatches | {summary.mismatches} |
| Avg Confidence | {avg_conf} |
| Low Confidence | {len(summary.low_confidence_annotations)} |

### Confidence by Category
"""
        for label, conf in summary.confidence_by_category.items():
            summary_text += f"- **{label}**: {conf:.3f}\n"
        
        # Build correct matches gallery
        correct_gallery = []
        mismatch_ids = {item["annotation_id"] for item in summary.mismatched_annotations}
        low_conf_ids = {item["annotation_id"] for item in summary.low_confidence_annotations}
        
        # Load full results to get all annotations
        results_path = output_dir / "qc_results.json"
        if results_path.exists():
            with open(results_path) as f:
                full_results = json.load(f)
            
            for item in full_results.get("results", []):
                ann_id = item.get("annotation_id")
                if item.get("is_match") and ann_id not in mismatch_ids and ann_id not in low_conf_ids:
                    crop_path = item.get("crop_path", "")
                    if crop_path:
                        # Handle both relative and absolute paths
                        path = Path(crop_path)
                        if not path.is_absolute():
                            path = output_dir / crop_path
                        if path.exists():
                            caption = f"✓ {item['label']} ({item.get('confidence', 0):.2f})"
                            correct_gallery.append((str(path), caption))
        
        # Build issues gallery (mismatches + low confidence)
        issues_gallery = []
        for item in summary.mismatched_annotations[:20]:
            path = item.get("crop_path", "")
            if path and Path(path).exists():
                caption = f"⚠️ {item['label']} → {item['prediction_label']} ({item['confidence']:.2f})"
                issues_gallery.append((path, caption))
        
        for item in summary.low_confidence_annotations[:20]:
            if item["annotation_id"] not in mismatch_ids:
                path = item.get("crop_path", "")
                if path and Path(path).exists():
                    status_icon = "✓" if item.get("is_match") else "✗"
                    caption = f"{status_icon} {item['label']} ({item['confidence']:.2f}) - low conf"
                    issues_gallery.append((path, caption))
        
        return status, summary_text, correct_gallery, issues_gallery
        
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        return f"❌ {e}", "", [], []


# =============================================================================
# Gradio Interface
# =============================================================================

def create_app():
    """Create Gradio application."""
    
    with gr.Blocks(title="QC Pipeline", theme=gr.themes.Soft()) as app:
        guidelines_state = gr.State(value={})
        
        gr.Markdown("""
# 🔍 QC Validation Pipeline

Validate annotation quality using AI vision models (Gemini, OpenAI, Anthropic).
""")
        
        with gr.Tabs():
            # === Tab 1: Data Source ===
            with gr.Tab("📁 Data"):
                source = gr.Radio(
                    ["upload", "labellerr"],
                    value="upload",
                    label="Source",
                )
                
                # Upload section
                with gr.Group(visible=True) as upload_grp:
                    gr.Markdown("### Upload Files")
                    with gr.Row():
                        coco_json = gr.File(label="COCO JSON", file_types=[".json"])
                        images_zip = gr.File(label="Images ZIP", file_types=[".zip"])
                    images = gr.File(label="Or individual images", file_count="multiple", file_types=["image"])
                
                # Labellerr section
                with gr.Group(visible=False) as labellerr_grp:
                    gr.Markdown("### Labellerr")
                    with gr.Row():
                        ll_key = gr.Textbox(label="API Key", type="password")
                        ll_secret = gr.Textbox(label="API Secret", type="password")
                    with gr.Row():
                        ll_client = gr.Textbox(label="Client ID")
                        ll_project = gr.Textbox(label="Project ID")
                    ll_statuses = gr.CheckboxGroup(
                        ["reviewer_layer", "client_reviewer_layer", "completed"],
                        value=["completed"],
                        label="Statuses",
                    )
                    ll_timeout = gr.Number(label="Timeout (s)", value=900, minimum=60)
                
                source.change(
                    lambda s: (gr.update(visible=s=="upload"), gr.update(visible=s=="labellerr")),
                    [source], [upload_grp, labellerr_grp]
                )
            
            # === Tab 2: Model ===
            with gr.Tab("🤖 Model"):
                provider = gr.Dropdown(list(PROVIDERS.keys()), value="gemini", label="Provider")
                model = gr.Dropdown(
                    PROVIDERS["gemini"]["models"],
                    value=PROVIDERS["gemini"]["default_model"],
                    label="Model",
                )
                api_key = gr.Textbox(label="API Key", type="password")
                
                provider.change(
                    lambda p: gr.update(
                        choices=PROVIDERS[p]["models"],
                        value=PROVIDERS[p]["default_model"]
                    ),
                    [provider], [model]
                )
            
            # === Tab 3: Guidelines ===
            with gr.Tab("📋 Guidelines"):
                gr.Markdown("Upload annotation guidelines PDF to improve accuracy (optional).")
                pdf = gr.File(label="Guidelines PDF", file_types=[".pdf"])
                extract_btn = gr.Button("Extract Guidelines", variant="secondary")
                guidelines_output = gr.Markdown()
                
                extract_btn.click(
                    extract_guidelines,
                    [pdf, provider, api_key, model],
                    [guidelines_output, guidelines_state],
                )
            
            # === Tab 4: Settings ===
            with gr.Tab("⚙️ Settings"):
                with gr.Row():
                    max_files = gr.Number(label="Max Annotations", value=50, minimum=1, maximum=1000)
                    confidence_threshold = gr.Slider(
                        label="Low Confidence Threshold",
                        minimum=0, maximum=1, value=0.5, step=0.05,
                    )
        
        # Run button
        gr.Markdown("---")
        run_btn = gr.Button("🚀 Run Validation", variant="primary", size="lg")
        
        # Results
        with gr.Row():
            with gr.Column():
                status = gr.Markdown(label="Status")
                summary = gr.Markdown(label="Summary")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ✅ Correct Labels")
                correct_gallery = gr.Gallery(label="Correctly Identified", columns=4, height="auto")
            with gr.Column():
                gr.Markdown("### ⚠️ Issues Found")
                issues_gallery = gr.Gallery(label="Mismatches & Low Confidence", columns=4, height="auto")
        
        run_btn.click(
            run_validation,
            [
                source, images, coco_json, images_zip,
                ll_key, ll_secret, ll_client, ll_project, ll_statuses, ll_timeout,
                provider, api_key, model,
                max_files, confidence_threshold,
                guidelines_state,
            ],
            [status, summary, correct_gallery, issues_gallery],
        )
    
    return app


def main():
    """Launch the application."""
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
