from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests

logger = logging.getLogger(__name__)


DEFAULT_PROMPT_TEMPLATE = (
    "You are validating the classification of an object crop extracted from an annotated image. "
    "The expected label provided by a human annotator is: \"{expected_label}\".\n\n"
    "🔴 VALIDATION INSTRUCTIONS:\n"
    "1. Examine the image CAREFULLY and SYSTEMATICALLY\n"
    "2. If the label involves quantities (e.g., \"four wheeler\"), COUNT the relevant features\n"
    "3. Verify ALL visual characteristics that define this category\n"
    "4. Check if this could be confused with a similar but different category\n"
    "5. Be STRICT and CONSERVATIVE with your confidence assessment\n\n"
    "Review the crop and decide whether the object's class matches the expected label. "
    "Respond strictly as a JSON object with the following keys:\n"
    "{{\n"
    '  "prediction_label": "<model_label>",\n'
    '  "confidence": <number between 0 and 1>,\n'
    '  "rationale": "<detailed explanation of what you observed and verified>"\n'
    "}}\n\n"
    "⚠️ IMPORTANT: Only use high confidence (>0.8) if you are ABSOLUTELY CERTAIN. "
    "If you are unsure or cannot verify all features, use a low confidence score."
)

# Simplified prompt template for demo: "Is this a [label]?"
DEMO_PROMPT_TEMPLATE = (
    "You are an expert visual classifier. Your task is to determine if this cropped image is showing the image of \"{expected_label}\".\n\n"
    "Respond strictly as a JSON object with the following keys:\n"
    "{{\n"
    '  "prediction_label": "<your label>",\n'
    '  "confidence": <number between 0 and 1>,\n'
    '  "rationale": "<detailed step-by-step explanation: what you counted, what you verified, why you are confident or uncertain>"\n'
    "}}\n"
)

# Guidelines-enhanced prompt template
GUIDELINES_PROMPT_TEMPLATE = (
    "You are an expert visual validator. The expected label is: \"{expected_label}\"\n\n"
    "📋 LABEL DEFINITION:\n{definition}\n\n"
    "🔍 KEY VISUAL CHARACTERISTICS TO VERIFY:\n{characteristics}\n\n"
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
    "Respond strictly as a JSON object with the following keys:\n"
    "{{\n"
    '  "prediction_label": "<your label>",\n'
    '  "confidence": <number between 0 and 1>,\n'
    '  "rationale": "<step-by-step analysis: what you counted, which characteristics you verified, what matches/mismatches you found, why you are confident or uncertain>"\n'
    "}}\n"
)




@dataclass
class GeminiResponse:
    prediction_label: str
    confidence: float
    rationale: Optional[str]
    raw_text: str


class GeminiValidator:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3-flash-preview",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        temperature: float = 0.1,  # Lower temperature for more deterministic, careful responses
        top_p: float | None = None,
        max_retries: int = 3,
        timeout_seconds: int = 30,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        enable_code_execution: bool = True,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.prompt_template = prompt_template
        self.enable_code_execution = enable_code_execution

    def _endpoint(self) -> str:
        return f"{self.base_url}/models/{self.model_name}:generateContent"

    def _build_guidelines_prompt(
        self,
        expected_label: str,
        label_definition: Optional[Dict[str, Any]],
    ) -> str:
        """
        Build prompt with guidelines context if available.
        
        Args:
            expected_label: The expected label for the crop
            label_definition: Dict with 'definition', 'key_characteristics', etc.
            
        Returns:
            Formatted prompt string
        """
        if label_definition:
            characteristics = "\n".join(
                f"- {c}" for c in label_definition.get("key_characteristics", [])
            )
            return GUIDELINES_PROMPT_TEMPLATE.format(
                expected_label=expected_label,
                definition=label_definition.get("definition", "Not specified"),
                characteristics=characteristics or "Not specified",
            )
        return self.prompt_template.format(expected_label=expected_label)

    def _build_request_body(
        self,
        crop_bytes: bytes,
        expected_label: str,
        guidelines: Optional[str],
        label_definitions: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], str]:
        """Build request body and return both body and the prompt text for logging."""
        # Check if we have a structured label definition
        label_def = None
        if label_definitions:
            # Try exact match first, then case-insensitive
            label_def = label_definitions.get(expected_label)
            if not label_def:
                label_def = label_definitions.get(expected_label.lower())
        
        # Build appropriate prompt
        if label_def:
            prompt = self._build_guidelines_prompt(expected_label, label_def)
        else:
            prompt = self.prompt_template.format(expected_label=expected_label)
            if guidelines:
                prompt += f"\n\nGuidelines:\n{guidelines.strip()}"

        parts = [
            {"text": prompt},
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(crop_bytes).decode("utf-8"),
                }
            },
        ]

        generation_config: Dict[str, Any] = {"temperature": self.temperature}
        if self.top_p is not None:
            generation_config["topP"] = self.top_p

        body = {"contents": [{"parts": parts}], "generationConfig": generation_config}
        
        # Add code execution tool if enabled
        if self.enable_code_execution:
            body["tools"] = [{"codeExecution": {}}]
        
        return body, prompt

    def _execute_request(self, body: Dict[str, Any]) -> Dict[str, Any]:
        url = self._endpoint()
        headers = {"x-goog-api-key": self.api_key}

        response = requests.post(
            url,
            headers=headers,
            json=body,
            timeout=self.timeout_seconds,
        )
        if response.status_code != 200:
            logger.warning("Gemini API error %s: %s", response.status_code, response.text)
            response.raise_for_status()
        return response.json()

    def _parse_candidate(self, payload: Dict[str, Any]) -> GeminiResponse:
        candidates = payload.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini response missing candidates")
        content = candidates[0].get("content", {})
        parts: Sequence[Dict[str, Any]] = content.get("parts", [])
        if not parts:
            raise ValueError("Gemini response missing content parts")

        # Extract text from parts (may include code execution results)
        text_parts = []
        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "executableCode" in part:
                # Log code execution if present
                logger.info(f"Code execution detected: {part.get('executableCode', {}).get('code', '')}")
            elif "codeExecutionResult" in part:
                # Log execution result
                result = part.get("codeExecutionResult", {})
                logger.info(f"Code execution result: {result.get('output', '')}")
        
        text = "\n".join(text_parts).strip()
        if not text:
            raise ValueError("Gemini response missing text output")

        # Strip markdown code fences if present
        if text.startswith("```json"):
            text = text[7:]  # Remove ```json
        elif text.startswith("```"):
            text = text[3:]  # Remove ```
        if text.endswith("```"):
            text = text[:-3]  # Remove trailing ```
        text = text.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Gemini response not valid JSON: {text}") from exc

        label = parsed.get("prediction_label")
        confidence = parsed.get("confidence")
        rationale = parsed.get("rationale")

        if not isinstance(label, str):
            raise ValueError("Gemini JSON missing prediction_label string")
        if not isinstance(confidence, (int, float)):
            raise ValueError("Gemini JSON missing numeric confidence")

        return GeminiResponse(
            prediction_label=label,
            confidence=float(confidence),
            rationale=rationale if isinstance(rationale, str) else None,
            raw_text=text,
        )

    def validate_crop(
        self,
        crop_bytes: bytes,
        expected_label: str,
        guidelines: Optional[str] = None,
        label_definitions: Optional[Dict[str, Any]] = None,
        log_prompt: bool = False,
    ) -> GeminiResponse:
        """
        Validate a crop against an expected label.
        
        Args:
            crop_bytes: PNG image bytes of the crop
            expected_label: Expected label for this crop
            guidelines: Optional raw text guidelines (legacy)
            label_definitions: Optional dict mapping label names to their definitions
                              Format: {label_name: {"definition": str, "key_characteristics": [str], ...}}
            log_prompt: If True, log the prompt being sent to Gemini
        
        Returns:
            GeminiResponse with prediction, confidence, and rationale
        """
        body, prompt = self._build_request_body(crop_bytes, expected_label, guidelines, label_definitions)
        
        # Log the prompt if requested
        if log_prompt:
            logger.info("=" * 80)
            logger.info(f"GEMINI PROMPT for label '{expected_label}':")
            logger.info("-" * 80)
            logger.info(prompt)
            logger.info("=" * 80)

        attempt = 0
        while True:
            attempt += 1
            try:
                payload = self._execute_request(body)
                return self._parse_candidate(payload)
            except (requests.RequestException, ValueError) as exc:
                if attempt >= self.max_retries:
                    raise
                sleep_seconds = min(2 ** (attempt - 1), 30)
                logger.warning(
                    "Gemini validation failed (attempt %s/%s): %s. Retrying in %ss",
                    attempt,
                    self.max_retries,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

    def validate_batch(
        self,
        items: Iterable[Dict[str, Any]],
        guidelines: Optional[str] = None,
        label_definitions: Optional[Dict[str, Any]] = None,
    ) -> List[GeminiResponse]:
        """
        Validate a batch of crops.
        
        Args:
            items: Iterable of dicts with 'crop_bytes' and 'expected_label'
            guidelines: Optional raw text guidelines (legacy)
            label_definitions: Optional dict mapping label names to their definitions
        
        Returns:
            List of GeminiResponse objects
        """
        responses: List[GeminiResponse] = []
        for item in items:
            crop_bytes = item["crop_bytes"]
            expected_label = item["expected_label"]
            responses.append(
                self.validate_crop(crop_bytes, expected_label, guidelines, label_definitions)
            )
        return responses



def create_demo_validator(
    api_key: str,
    model_name: str = "gemini-3-flash-preview",
    temperature: float = 0.1,  # Lower temperature for stricter, more careful validation
    max_retries: int = 3,
    enable_code_execution: bool = True,
) -> GeminiValidator:
    """Create a Gemini validator configured for demo usage with code execution enabled."""
    return GeminiValidator(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_retries=max_retries,
        prompt_template=DEMO_PROMPT_TEMPLATE,
        enable_code_execution=enable_code_execution,
    )
