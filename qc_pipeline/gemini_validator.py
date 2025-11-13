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
    "Review the crop and decide whether the object's class matches the expected label. "
    "Respond strictly as a JSON object with the following keys:\n"
    "{{\n"
    '  "prediction_label": "<model_label>",\n'
    '  "confidence": <number between 0 and 1>,\n'
    '  "rationale": "<optional short explanation>"\n'
    "}}\n"
    "If you are unsure, pick the closest label and use a low confidence score."
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
        model_name: str = "gemini-2.5-flash",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        temperature: float = 0.2,
        top_p: float | None = None,
        max_retries: int = 3,
        timeout_seconds: int = 30,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.prompt_template = prompt_template

    def _endpoint(self) -> str:
        return f"{self.base_url}/models/{self.model_name}:generateContent"

    def _build_request_body(
        self,
        crop_bytes: bytes,
        expected_label: str,
        guidelines: Optional[str],
    ) -> Dict[str, Any]:
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

        return {"contents": [{"parts": parts}], "generationConfig": generation_config}

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

        text = parts[0].get("text", "").strip()
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
    ) -> GeminiResponse:
        body = self._build_request_body(crop_bytes, expected_label, guidelines)

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
    ) -> List[GeminiResponse]:
        responses: List[GeminiResponse] = []
        for item in items:
            crop_bytes = item["crop_bytes"]
            expected_label = item["expected_label"]
            responses.append(self.validate_crop(crop_bytes, expected_label, guidelines))
        return responses

