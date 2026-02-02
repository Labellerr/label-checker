"""QC Validation Workflow Orchestrator using LangGraph."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from qc_pipeline.data_loader import CocoDataset, load_coco_dataset
from qc_pipeline.image_utils import CropConfig, crop_annotation, image_to_png_bytes
from qc_pipeline.langgraph_validator import QCValidationState, build_guidelines_graph

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result for a single annotation validation."""
    image_id: int
    annotation_id: int
    image_file: str
    label: str
    crop_path: str
    confidence: Optional[float]
    prediction_label: Optional[str]
    is_match: Optional[bool]
    gemini_response: Optional[str]
    error: Optional[str]


@dataclass
class ValidationSummary:
    """Summary statistics for validation run."""
    total_annotations: int
    total_crops_saved: int
    gemini_validations: int
    matches: int
    mismatches: int
    average_confidence: Optional[float]
    confidence_by_category: Dict[str, float]
    low_confidence_annotations: List[Dict]
    mismatched_annotations: List[Dict]


class QCValidationWorkflow:
    """
    Main workflow orchestrator for QC validation using LangGraph.
    
    This class manages the two-agent workflow:
    1. GuidelinesAgent: Extracts visual characteristics from PDF (if provided)
    2. ValidationAgent: Validates crops using guidelines from shared state
    """
    
    def __init__(self, gemini_api_key: str, output_dir: Path):
        """
        Initialize the workflow.
        
        Args:
            gemini_api_key: Google API key for Gemini
            output_dir: Directory to save outputs
        """
        self.gemini_api_key = gemini_api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build the LangGraph workflow
        self.graph = build_guidelines_graph()
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup file logging for guidelines and validation prompts."""
        # Guidelines extraction log
        guidelines_log = self.output_dir / "guidelines_extraction.log"
        guidelines_handler = logging.FileHandler(guidelines_log, mode='w')
        guidelines_handler.setLevel(logging.INFO)
        guidelines_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # Validation prompts log
        validation_log = self.output_dir / "validation_prompts.log"
        validation_handler = logging.FileHandler(validation_log, mode='w')
        validation_handler.setLevel(logging.INFO)
        validation_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # Add handlers to relevant loggers
        logging.getLogger('qc_pipeline.langgraph_validator').addHandler(guidelines_handler)
        logging.getLogger('qc_pipeline.qc_workflow').addHandler(validation_handler)
        logging.getLogger('qc_pipeline.gemini_validator').addHandler(validation_handler)
        
        logger.info(f"Logging setup complete:")
        logger.info(f"  Guidelines log: {guidelines_log}")
        logger.info(f"  Validation log: {validation_log}")
    
    def run(
        self,
        pdf_path: Optional[str],
        coco_json_path: Path,
        images_dir: Path,
        max_files: int = 50,
        confidence_threshold: float = 0.5,
        pre_extracted_guidelines: Optional[Dict[str, Dict]] = None,
    ) -> Tuple[Dict, List[ValidationResult], ValidationSummary]:
        """
        Execute the complete QC validation workflow.
        
        Args:
            pdf_path: Optional path to annotation guidelines PDF
            coco_json_path: Path to COCO JSON annotations
            images_dir: Directory containing images
            max_files: Maximum number of annotations to validate
            confidence_threshold: Threshold for flagging low confidence
            pre_extracted_guidelines: Optional pre-extracted guidelines dict
            
        Returns:
            Tuple of (state_dict, validation_results, summary)
        """
        logger.info("=" * 80)
        logger.info("STARTING QC VALIDATION WORKFLOW")
        logger.info("=" * 80)
        
        # Step 1: Load COCO dataset
        logger.info(f"Loading COCO dataset from: {coco_json_path}")
        dataset = load_coco_dataset(coco_json_path)
        
        # Limit annotations (convert max_files to int for slicing)
        max_files_int = int(max_files)
        annotations_to_process = dataset.annotations[:max_files_int]
        logger.info(f"Processing {len(annotations_to_process)} annotations (limited to {max_files_int})")
        
        # Step 2: Initialize state and run guidelines extraction (if PDF provided)
        initial_state: QCValidationState = {
            "pdf_path": pdf_path if pdf_path else None,
            "pdf_text": None,
            "guidelines_dict": pre_extracted_guidelines if pre_extracted_guidelines else {},
            "categories_found": [],
            "pdf_processed": False,
            "gemini_api_key": self.gemini_api_key,
            "output_dir": str(self.output_dir),
            "status_messages": [],
            "error": None,
        }
        
        # Run the graph (guidelines extraction + preparation)
        logger.info("Running LangGraph workflow (guidelines extraction)")
        final_state = self.graph.invoke(initial_state)
        
        # Check for errors
        if final_state.get("error"):
            logger.error(f"Workflow error: {final_state['error']}")
            return final_state, [], ValidationSummary(
                total_annotations=0,
                total_crops_saved=0,
                gemini_validations=0,
                matches=0,
                mismatches=0,
                average_confidence=None,
                confidence_by_category={},
                low_confidence_annotations=[],
                mismatched_annotations=[]
            )
        
        # Log status messages
        for msg in final_state.get("status_messages", []):
            logger.info(msg)
        
        # Step 3: Run validation loop
        logger.info("=" * 80)
        logger.info("STARTING VALIDATION AGENT")
        logger.info("=" * 80)
        
        guidelines_dict = final_state.get("guidelines_dict", {})
        results, summary = self._validate_annotations(
            dataset=dataset,
            annotations=annotations_to_process,
            images_dir=images_dir,
            guidelines_dict=guidelines_dict,
            confidence_threshold=confidence_threshold,
        )
        
        # Step 4: Save results
        self._save_results(results, summary, final_state)
        
        logger.info("=" * 80)
        logger.info("WORKFLOW COMPLETE")
        logger.info("=" * 80)
        
        return final_state, results, summary
    
    def _validate_annotations(
        self,
        dataset: CocoDataset,
        annotations: List,
        images_dir: Path,
        guidelines_dict: Dict[str, Dict],
        confidence_threshold: float,
    ) -> Tuple[List[ValidationResult], ValidationSummary]:
        """
        Validate annotations using the ValidationAgent.
        
        This method processes each crop and validates it with Gemini,
        using guidelines from state if available.
        """
        # Import here to avoid circular imports
        from qc_pipeline.gemini_validator import GeminiValidator, DEMO_PROMPT_TEMPLATE
        
        validator = GeminiValidator(
            api_key=self.gemini_api_key,
            model_name="gemini-3-flash-preview",
            prompt_template=DEMO_PROMPT_TEMPLATE,
            temperature=0.1,  # Strict, deterministic responses
            enable_code_execution=True,  # Enable code execution for counting/analysis
        )
        
        crops_dir = self.output_dir / "crops"
        crops_dir.mkdir(exist_ok=True)
        
        results = []
        image_cache = {}
        
        logger.info(f"Validating {len(annotations)} annotations...")
        if guidelines_dict:
            logger.info(f"Using guidelines for categories: {list(guidelines_dict.keys())}")
        else:
            logger.info("No guidelines provided, using default prompts")
        
        for idx, annotation in enumerate(annotations, 1):
            image_info = dataset.images.get(annotation.image_id)
            category = dataset.categories.get(annotation.category_id)
            
            if not image_info or not category:
                logger.warning(f"Skipping annotation {annotation.id}: missing image or category")
                continue
            
            logger.info(f"[{idx}/{len(annotations)}] Processing {image_info.file_name} - {category.name}")
            
            # Load image (with caching)
            if annotation.image_id not in image_cache:
                image_path = images_dir / image_info.file_name
                if not image_path.exists():
                    logger.warning(f"Image not found: {image_path}")
                    continue
                
                with Image.open(image_path) as img:
                    image_cache[annotation.image_id] = img.convert("RGB")
            
            base_image = image_cache[annotation.image_id]
            
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
                prediction_label = None
                is_match = None
                gemini_response = None
                error = None
                
                try:
                    crop_bytes = image_to_png_bytes(crop)
                    
                    # Check if we have guidelines for this category
                    category_lower = category.name.lower()
                    has_guidelines = category_lower in guidelines_dict
                    
                    # Log prompt for first 3 crops and all with guidelines
                    log_this_prompt = (idx <= 3) or has_guidelines
                    
                    if log_this_prompt and has_guidelines:
                        logger.info(f"Using guidelines for category: {category.name}")
                        label_def = guidelines_dict[category_lower]
                        logger.info(f"  Definition: {label_def.get('definition', 'N/A')}")
                        logger.info(f"  Visual features: {len(label_def.get('key_characteristics', []))}")
                    
                    response = validator.validate_crop(
                        crop_bytes=crop_bytes,
                        expected_label=category.name,
                        guidelines=None,
                        label_definitions=guidelines_dict,
                        log_prompt=log_this_prompt,
                    )
                    confidence = response.confidence
                    prediction_label = response.prediction_label
                    gemini_response = response.raw_text
                    
                    # Check if prediction matches expected label (case-insensitive)
                    is_match = prediction_label.strip().lower() == category.name.strip().lower()
                    
                    if has_guidelines:
                        match_str = "✓ MATCH" if is_match else "✗ MISMATCH"
                        logger.info(f"  ✓ Validated (confidence: {confidence:.2f}, {match_str}, predicted: {prediction_label}, using guidelines)")
                    else:
                        match_str = "✓ MATCH" if is_match else "✗ MISMATCH"
                        logger.info(f"  ✓ Validated (confidence: {confidence:.2f}, {match_str}, predicted: {prediction_label})")
                        
                except Exception as e:
                    error = str(e)
                    logger.error(f"  ✗ Validation error: {error}")
                
                result = ValidationResult(
                    image_id=annotation.image_id,
                    annotation_id=annotation.id,
                    image_file=image_info.file_name,
                    label=category.name,
                    crop_path=str(crop_path.relative_to(self.output_dir)),
                    confidence=confidence,
                    prediction_label=prediction_label,
                    is_match=is_match,
                    gemini_response=gemini_response,
                    error=error,
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"  ✗ Crop generation error: {e}")
                continue
        
        # Generate summary
        summary = self._generate_summary(results, confidence_threshold)
        
        return results, summary
    
    def _generate_summary(
        self,
        results: List[ValidationResult],
        confidence_threshold: float,
    ) -> ValidationSummary:
        """Generate summary statistics from validation results."""
        successful_crops = [r for r in results if r.crop_path]
        gemini_results = [r for r in results if r.confidence is not None]
        
        # Count matches and mismatches
        matches = sum(1 for r in gemini_results if r.is_match is True)
        mismatches = sum(1 for r in gemini_results if r.is_match is False)
        
        # Calculate average confidence
        if gemini_results:
            avg_confidence = sum(r.confidence for r in gemini_results) / len(gemini_results)
        else:
            avg_confidence = None
        
        # Calculate confidence by category
        confidence_by_cat = {}
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
                "prediction_label": r.prediction_label,
                "confidence": r.confidence,
                "is_match": r.is_match,
                "crop_path": str(self.output_dir / r.crop_path),
            }
            for r in gemini_results
            if r.confidence < confidence_threshold
        ]
        
        # Find mismatched annotations (label doesn't match prediction)
        mismatched = [
            {
                "annotation_id": r.annotation_id,
                "label": r.label,
                "prediction_label": r.prediction_label,
                "confidence": r.confidence,
                "crop_path": str(self.output_dir / r.crop_path),
            }
            for r in gemini_results
            if r.is_match is False
        ]
        
        return ValidationSummary(
            total_annotations=len(results),
            total_crops_saved=len(successful_crops),
            gemini_validations=len(gemini_results),
            matches=matches,
            mismatches=mismatches,
            average_confidence=avg_confidence,
            confidence_by_category=confidence_by_category,
            low_confidence_annotations=low_confidence,
            mismatched_annotations=mismatched,
        )
    
    def _save_results(
        self,
        results: List[ValidationResult],
        summary: ValidationSummary,
        state: Dict,
    ):
        """Save validation results to JSON file."""
        results_path = self.output_dir / "qc_results.json"
        
        output_data = {
            "workflow_state": {
                "pdf_processed": state.get("pdf_processed"),
                "categories_found": state.get("categories_found", []),
                "guidelines_used": bool(state.get("guidelines_dict")),
            },
            "results": [asdict(r) for r in results],
            "summary": asdict(summary),
        }
        
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to: {results_path}")
