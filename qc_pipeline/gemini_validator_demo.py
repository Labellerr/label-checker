"""
Demo-specific Gemini validator with simplified prompt for "Is this a [label]?" validation.
"""
from __future__ import annotations

from .gemini_validator import GeminiValidator

# Simplified prompt template for demo: "Is this a [label]?"
DEMO_PROMPT_TEMPLATE = (
    "Look at this cropped image and determine if it contains a \"{expected_label}\".\n\n"
    "Respond strictly as a JSON object with the following keys:\n"
    "{{\n"
    '  "prediction_label": "<your label>",\n'
    '  "confidence": <number between 0 and 1>,\n'
    '  "rationale": "<optional explanation>"\n'
    "}}\n\n"
    "If the image clearly shows a {expected_label}, use high confidence (0.8-1.0).\n"
    "If you're unsure or it partially matches, use medium confidence (0.4-0.7).\n"
    "If it doesn't match at all, use low confidence (0.0-0.3)."
)


def create_demo_validator(
    api_key: str,
    model_name: str = "gemini-1.5-pro",
    temperature: float = 0.2,
    max_retries: int = 3,
) -> GeminiValidator:
    """Create a Gemini validator configured for demo usage."""
    return GeminiValidator(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_retries=max_retries,
        prompt_template=DEMO_PROMPT_TEMPLATE,
    )

