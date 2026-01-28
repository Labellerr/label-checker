"""LangGraph-based QC validation with stateful agent workflow."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph

from qc_pipeline.guidelines_extractor import GuidelinesExtractor
from qc_pipeline.pdf_utils import extract_text_from_pdf

logger = logging.getLogger(__name__)


class QCValidationState(TypedDict):
    """State for the QC validation workflow."""
    
    # Guidelines extraction
    pdf_path: Optional[str]
    pdf_text: Optional[str]
    guidelines_dict: Dict[str, Dict]  # {category: {definition, key_characteristics, ...}}
    categories_found: List[str]
    pdf_processed: bool
    
    # Validation inputs
    gemini_api_key: str
    output_dir: str
    
    # Status messages
    status_messages: List[str]
    error: Optional[str]


def guidelines_agent_node(state: QCValidationState) -> QCValidationState:
    """
    Agent 1: Extract annotation guidelines from PDF.
    
    This agent:
    1. Extracts text from the PDF
    2. Uses Gemini to parse visual characteristics for each category
    3. Stores the structured guidelines in state
    
    If guidelines_dict is already populated (from session state),
    it skips extraction and uses the existing guidelines.
    """
    logger.info("=== Guidelines Agent Started ===")
    
    # Check if guidelines are already provided (from session state)
    existing_guidelines = state.get("guidelines_dict", {})
    if existing_guidelines:
        logger.info(f"Using pre-extracted guidelines with {len(existing_guidelines)} categories")
        categories_found = list(existing_guidelines.keys())
        return {
            **state,
            "pdf_processed": True,
            "guidelines_dict": existing_guidelines,
            "categories_found": categories_found,
            "status_messages": [f"✓ Using pre-extracted guidelines for {len(categories_found)} categories"]
        }
    
    pdf_path = state.get("pdf_path")
    if not pdf_path:
        logger.info("No PDF or pre-extracted guidelines provided")
        return {
            **state,
            "pdf_processed": False,
            "guidelines_dict": {},
            "categories_found": [],
            "status_messages": ["ℹ️ No annotation guidelines provided"]
        }
    
    try:
        # Extract text from PDF
        logger.info(f"Extracting text from PDF: {pdf_path}")
        pdf_text = extract_text_from_pdf(Path(pdf_path))
        
        if not pdf_text or not pdf_text.strip():
            logger.warning("PDF text is empty")
            return {
                **state,
                "pdf_processed": False,
                "guidelines_dict": {},
                "categories_found": [],
                "error": "PDF appears to be empty or unreadable"
            }
        
        logger.info(f"Extracted {len(pdf_text)} characters from PDF")
        
        # Extract structured guidelines using Gemini
        logger.info("Extracting label definitions using Gemini...")
        extractor = GuidelinesExtractor(api_key=state["gemini_api_key"])
        guidelines = extractor.extract(pdf_text)
        
        # Convert to dict for easy lookup
        guidelines_dict = {
            label.label_name.lower(): label.model_dump()
            for label in guidelines.labels
        }
        
        categories_found = list(guidelines_dict.keys())
        logger.info(f"Extracted guidelines for {len(categories_found)} categories: {categories_found}")
        
        # Save to file
        output_dir = Path(state["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        guidelines_json_path = output_dir / "extracted_guidelines.json"
        with open(guidelines_json_path, "w", encoding="utf-8") as f:
            json.dump(guidelines_dict, f, indent=2)
        logger.info(f"Saved guidelines to: {guidelines_json_path}")
        
        # Build status message with visual characteristics
        status_msg = f"✅ Extracted guidelines for {len(categories_found)} categories:\n"
        for category, info in guidelines_dict.items():
            characteristics = info.get("key_characteristics", [])
            status_msg += f"  • {category.upper()}: {len(characteristics)} visual features\n"
        
        return {
            **state,
            "pdf_text": pdf_text,
            "guidelines_dict": guidelines_dict,
            "categories_found": categories_found,
            "pdf_processed": True,
            "status_messages": [status_msg]
        }
        
    except Exception as e:
        logger.error(f"Error in guidelines agent: {e}", exc_info=True)
        return {
            **state,
            "pdf_processed": False,
            "guidelines_dict": {},
            "categories_found": [],
            "error": f"Failed to extract guidelines: {str(e)}"
        }


def should_extract_guidelines(state: QCValidationState) -> str:
    """
    Conditional edge: determine if we should extract guidelines.
    
    Returns:
        "extract_guidelines" if PDF is provided OR pre-extracted guidelines exist
        "validate" if no PDF and no guidelines (skip to validation)
    """
    existing_guidelines = state.get("guidelines_dict", {})
    pdf_path = state.get("pdf_path")
    
    if existing_guidelines:
        logger.info("Pre-extracted guidelines found, routing to guidelines agent")
        return "extract_guidelines"
    elif pdf_path and Path(pdf_path).exists():
        logger.info("PDF provided, routing to guidelines extraction")
        return "extract_guidelines"
    else:
        logger.info("No PDF or guidelines provided, routing directly to validation")
        return "validate"


def prepare_validation_node(state: QCValidationState) -> QCValidationState:
    """
    Prepare state for validation by logging readiness.
    
    This node runs after guidelines extraction (if any) and before validation loop.
    """
    logger.info("=== Preparing for Validation ===")
    
    status_msg = "📊 Ready to validate annotations"
    if state.get("pdf_processed"):
        status_msg += f" using guidelines for {len(state.get('categories_found', []))} categories"
    
    logger.info(status_msg)
    
    return {
        **state,
        "status_messages": state.get("status_messages", []) + [status_msg]
    }


def build_guidelines_graph() -> StateGraph:
    """
    Build the LangGraph workflow for guidelines extraction.
    
    Graph structure:
        START -> conditional_check
            -> (if PDF/guidelines) extract_guidelines -> prepare_validation -> END
            -> (if no PDF) prepare_validation -> END
    
    The actual validation loop is handled by QCValidationWorkflow
    to avoid complex graph state management.
    """
    workflow = StateGraph(QCValidationState)
    
    # Add nodes
    workflow.add_node("extract_guidelines", guidelines_agent_node)
    workflow.add_node("prepare_validation", prepare_validation_node)
    
    # Set entry point with conditional routing
    workflow.set_conditional_entry_point(
        should_extract_guidelines,
        {
            "extract_guidelines": "extract_guidelines",
            "validate": "prepare_validation"
        }
    )
    
    # Add edges
    workflow.add_edge("extract_guidelines", "prepare_validation")
    workflow.add_edge("prepare_validation", "__end__")
    
    # Compile and return
    return workflow.compile()
