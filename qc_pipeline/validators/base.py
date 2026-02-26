"""Base validator class and response dataclass for LLM-based validation."""
from __future__ import annotations

import base64
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationResponse:
    """Response from a validation request."""
    prediction_label: str
    confidence: float
    rationale: Optional[str]
    raw_text: str


# Prompt templates shared across all validators
DEFAULT_PROMPT_TEMPLATE = (
    "You are validating the classification of an object crop extracted from an annotated image. "
    "The annotator labeled this as: \"{expected_label}\".\n\n"
    "{annotation_context}"
    "🔴 VALIDATION INSTRUCTIONS:\n"
    "1. Examine the image CAREFULLY and SYSTEMATICALLY\n"
    "2. If the label involves quantities (e.g., \"four wheeler\"), COUNT the relevant features\n"
    "3. Verify ALL visual characteristics that define this category\n"
    "4. Check if this could be confused with a similar but different category\n"
    "5. Be STRICT and CONSERVATIVE with your confidence assessment\n\n"
    "Review the crop and decide whether the object's class matches the annotated label. "
    "Respond strictly as a JSON object with the following keys:\n"
    "{{\n"
    '  "prediction_label": "<model_label>",\n'
    '  "confidence": <number between 0 and 1>,\n'
    '  "rationale": "<detailed explanation of what you observed and verified. If the visual evidence does not match the annotated label, explicitly state: \'This is mislabeled. The annotator labeled it as {expected_label}, but it should be <correct_label> because <specific visual evidence>.\'>"\n'
    "}}\n\n"
    "⚠️ IMPORTANT: Only use high confidence (>0.8) if you are ABSOLUTELY CERTAIN. "
    "If you are unsure or cannot verify all features, use a low confidence score. "
    "If mislabeled, clearly state what it was incorrectly labeled as and what it should be."
)

DEMO_PROMPT_TEMPLATE = (
    "You are an expert visual classifier. The annotator labeled this cropped image as: \"{expected_label}\".\n\n"
    "{annotation_context}"
    "Your task is to validate whether the visual content matches this label.\n\n"
    "🔍 VALIDATION REQUIREMENTS:\n"
    "- If the visual evidence MATCHES the annotated label, confirm this clearly\n"
    "- If the visual evidence DOES NOT MATCH, explicitly state: \"This is mislabeled. The annotator labeled it as '{expected_label}', but the visual evidence shows it should be '<correct_label>' because <specific visual reasons>\"\n"
    "- Provide specific visual evidence to support your conclusion\n\n"
    "Respond strictly as a JSON object with the following keys:\n"
    "{{\n"
    '  "prediction_label": "<your label>",\n'
    '  "confidence": <number between 0 and 1>,\n'
    '  "rationale": "<detailed step-by-step explanation: what you counted, what you verified, why you are confident or uncertain. If mislabeled, explicitly state what the annotator labeled it as and what it should be with specific visual evidence>"\n'
    "}}\n"
)

GUIDELINES_PROMPT_TEMPLATE = (
    "You are an expert visual validator. The annotator labeled this as: \"{expected_label}\"\n\n"
    "📋 LABEL DEFINITION:\n{definition}\n\n"
    "🔍 KEY VISUAL CHARACTERISTICS TO VERIFY:\n{characteristics}\n\n"
    "{annotation_context}"
    "🔴 STRICT VALIDATION PROTOCOL - FOLLOW EACH STEP:\n\n"
    "STEP 1 - OBSERVE THE IMAGE:\n"
    "- Look at the entire image systematically\n"
    "- Identify all visible objects, components, and features\n"
    "- Note anything that stands out\n\n"
    "STEP 2 - COUNT QUANTIFIABLE FEATURES:\n"
    "- If the label involves quantities (e.g., \"four wheeler\" = 4 wheels, \"two wheeler\" = 2 wheels), COUNT them\n"
    "- Verify counts EXACTLY match what the label implies\n"
    "- Use code execution if needed to analyze or count features\n\n"
    "STEP 3 - VERIFY EACH CHARACTERISTIC:\n"
    "Go through EACH characteristic listed above one by one:\n"
    "- Does the image show this characteristic? YES/NO\n"
    "- Can you verify it with certainty? YES/NO\n"
    "- Mark down any mismatches or uncertainties\n\n"
    "STEP 4 - CROSS-CHECK FOR CONFUSION:\n"
    "- Could this object be mistaken for a similar category?\n"
    "- What features DISTINGUISH it from similar objects?\n"
    "- Are there any contradictory visual cues?\n\n"
    "STEP 5 - FINAL VERIFICATION:\n"
    "- Review your analysis again\n"
    "- Double-check your counts and observations\n"
    "- Be HONEST about any uncertainty\n\n"
    "⚠️ STRICT RULES:\n"
    "- ONLY give high confidence (0.8-1.0) if you verified ALL characteristics\n"
    "- Use medium confidence (0.4-0.7) if you have ANY doubt or missing information\n"
    "- Use low confidence (0.0-0.3) if characteristics don't match or you cannot verify\n"
    "- DO NOT confuse similar categories (e.g., two wheeler vs four wheeler, eyes vs lips)\n"
    "- BE CONSERVATIVE - better to express doubt than give wrong high-confidence predictions\n\n"
    "🔍 RATIONALE REQUIREMENTS:\n"
    "- If the visual evidence MATCHES the annotated label \"{expected_label}\", confirm this clearly\n"
    "- If the visual evidence DOES NOT MATCH, explicitly state: \"This is mislabeled. The annotator labeled it as '{expected_label}', but the visual evidence shows it should be '<correct_label>' because <specific reasons>\"\n"
    "- Always provide specific visual evidence to support your conclusion\n"
    "- Be clear and direct about any mislabeling\n\n"
    "Respond strictly as a JSON object with the following keys:\n"
    "{{\n"
    '  "prediction_label": "<your label>",\n'
    '  "confidence": <number between 0 and 1>,\n'
    '  "rationale": "<step-by-step analysis: what you counted, which characteristics you verified, what matches/mismatches you found. If mislabeled, explicitly state the annotator labeled it as X but it should be Y because of specific visual evidence>"\n'
    "}}\n"
)


class BaseValidator(ABC):
    """
    Base class for all LLM validators using LangChain.
    
    Provides common functionality for building prompts, parsing responses,
    and handling multimodal messages with images.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str,
        temperature: float = 0.1,
        max_retries: int = 3,
        prompt_template: str = DEMO_PROMPT_TEMPLATE,
    ):
        """
        Initialize the validator.
        
        Args:
            api_key: API key for the provider
            model_name: Name of the model to use
            temperature: Temperature for generation (default: 0.1 for deterministic)
            max_retries: Maximum number of retries on failure
            prompt_template: Prompt template to use for validation
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.prompt_template = prompt_template
        self.llm = self._create_llm()
    
    @abstractmethod
    def _create_llm(self):
        """Create the LangChain LLM instance. Must be implemented by subclasses."""
        ...
    
    @abstractmethod
    def _build_multimodal_message(self, prompt: str, image_b64: str):
        """
        Build a multimodal message with text and image.
        
        Args:
            prompt: The text prompt
            image_b64: Base64-encoded image data
            
        Returns:
            A message object suitable for the LLM
        """
        ...
    
    def _encode_image(self, crop_bytes: bytes) -> str:
        """Encode image bytes to base64 string."""
        return base64.b64encode(crop_bytes).decode("utf-8")
    
    def _build_annotation_context(self, annotation_data: Optional[Dict[str, Any]]) -> str:
        """Build annotation context string from annotation data."""
        if not annotation_data:
            return ""
        
        context_parts = []
        
        if "category_name" in annotation_data:
            context_parts.append(f"Category: {annotation_data['category_name']}")
        
        if "attributes" in annotation_data and annotation_data["attributes"]:
            context_parts.append("Annotation Attributes:")
            for attr in annotation_data["attributes"]:
                if isinstance(attr, dict):
                    attr_name = attr.get("name", "Unknown")
                    attr_value = attr.get("value", "N/A")
                    context_parts.append(f"  - {attr_name}: {attr_value}")
                else:
                    context_parts.append(f"  - {attr}")
        
        if "metadata" in annotation_data:
            context_parts.append(f"Additional metadata: {annotation_data['metadata']}")
        
        if context_parts:
            return "📊 ANNOTATION DATA:\n" + "\n".join(context_parts) + "\n\n"
        return ""
    
    def _build_prompt(
        self,
        expected_label: str,
        guidelines: Optional[str] = None,
        label_definitions: Optional[Dict[str, Any]] = None,
        annotation_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the validation prompt."""
        annotation_context = self._build_annotation_context(annotation_data)
        
        # Check for structured label definition
        label_def = None
        if label_definitions:
            label_def = label_definitions.get(expected_label) or label_definitions.get(expected_label.lower())
        
        if label_def:
            characteristics = "\n".join(
                f"- {c}" for c in label_def.get("key_characteristics", [])
            )
            return GUIDELINES_PROMPT_TEMPLATE.format(
                expected_label=expected_label,
                definition=label_def.get("definition", "Not specified"),
                characteristics=characteristics or "Not specified",
                annotation_context=annotation_context,
            )
        
        prompt = self.prompt_template.format(
            expected_label=expected_label,
            annotation_context=annotation_context,
        )
        
        if guidelines:
            prompt += f"\n\nGuidelines:\n{guidelines.strip()}"
        
        return prompt
    
    def _parse_response(self, response_text) -> ValidationResponse:
        """Parse LLM response text into ValidationResponse."""
        # Handle different response types from LangChain
        if isinstance(response_text, dict):
            # LangChain multimodal responses come as {'type': 'text', 'text': '...'}
            if 'text' in response_text:
                text = response_text['text']
            else:
                text = str(response_text)
        elif isinstance(response_text, list):
            # Some responses come as lists of content parts
            if len(response_text) > 0:
                first_item = response_text[0]
                if isinstance(first_item, dict) and 'text' in first_item:
                    text = first_item['text']
                elif isinstance(first_item, str):
                    text = first_item
                else:
                    text = str(first_item)
            else:
                text = str(response_text)
        elif isinstance(response_text, str):
            text = response_text
        else:
            text = str(response_text)
        
        text = text.strip()
        
        # Strip markdown code fences if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Response not valid JSON: {text}") from exc
        
        label = parsed.get("prediction_label")
        confidence = parsed.get("confidence")
        rationale = parsed.get("rationale")
        
        if not isinstance(label, str):
            raise ValueError("Response missing prediction_label string")
        if not isinstance(confidence, (int, float)):
            raise ValueError("Response missing numeric confidence")
        
        return ValidationResponse(
            prediction_label=label,
            confidence=float(confidence),
            rationale=rationale if isinstance(rationale, str) else None,
            raw_text=text,
        )
    
    def _extract_text_from_response(self, response) -> str:
        """
        Extract text content from various LangChain response formats.
        
        LangChain responses can come in different formats:
        - AIMessage with content as string
        - AIMessage with content as list of dicts (multimodal)
        - Dict with 'text' key
        - List of content items
        """
        # If it's an AIMessage or similar with .content attribute
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = response
        
        # If content is a string, return directly
        if isinstance(content, str):
            return content
        
        # If content is a list (multimodal response)
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    # Handle dict items with 'text' key
                    if 'text' in item:
                        text_parts.append(item['text'])
                    elif 'type' in item and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
            if text_parts:
                return '\n'.join(text_parts)
        
        # If content is a dict with 'text' key
        if isinstance(content, dict):
            if 'text' in content:
                return content['text']
        
        # Fallback: convert to string
        return str(content)
    
    def validate_crop(
        self,
        crop_bytes: bytes,
        expected_label: str,
        guidelines: Optional[str] = None,
        label_definitions: Optional[Dict[str, Any]] = None,
        annotation_data: Optional[Dict[str, Any]] = None,
        log_prompt: bool = False,
    ) -> ValidationResponse:
        """
        Validate a crop against an expected label.
        
        Args:
            crop_bytes: PNG image bytes of the crop
            expected_label: Expected label for this crop
            guidelines: Optional raw text guidelines
            label_definitions: Optional dict mapping label names to their definitions
            annotation_data: Optional dict with full annotation data
            log_prompt: If True, log the prompt being sent
            
        Returns:
            ValidationResponse with prediction, confidence, and rationale
        """
        prompt = self._build_prompt(expected_label, guidelines, label_definitions, annotation_data)
        image_b64 = self._encode_image(crop_bytes)
        
        if log_prompt:
            logger.info("=" * 80)
            logger.info(f"VALIDATION PROMPT for label '{expected_label}':")
            logger.info("-" * 80)
            logger.info(prompt)
            logger.info("=" * 80)
        
        message = self._build_multimodal_message(prompt, image_b64)
        
        attempt = 0
        last_error = None
        while attempt < self.max_retries:
            attempt += 1
            try:
                response = self.llm.invoke([message])
                response_text = self._extract_text_from_response(response)
                return self._parse_response(response_text)
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                import time
                sleep_seconds = min(2 ** (attempt - 1), 30)
                logger.warning(
                    "Validation failed (attempt %s/%s): %s. Retrying in %ss",
                    attempt,
                    self.max_retries,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
        
        raise last_error
    
    def validate_batch(
        self,
        items: List[Dict[str, Any]],
        guidelines: Optional[str] = None,
        label_definitions: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationResponse]:
        """
        Validate a batch of crops.
        
        Args:
            items: List of dicts with 'crop_bytes', 'expected_label', and optionally 'annotation_data'
            guidelines: Optional raw text guidelines
            label_definitions: Optional dict mapping label names to their definitions
            
        Returns:
            List of ValidationResponse objects
        """
        responses = []
        for item in items:
            crop_bytes = item["crop_bytes"]
            expected_label = item["expected_label"]
            annotation_data = item.get("annotation_data")
            responses.append(
                self.validate_crop(
                    crop_bytes, expected_label, guidelines, label_definitions, annotation_data
                )
            )
        return responses
