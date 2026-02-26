"""Gemini validator using LangChain for vision-based validation."""
from __future__ import annotations

import logging
from typing import Optional

from langchain_core.messages import HumanMessage

from qc_pipeline.validators.base import BaseValidator, DEMO_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# Available Gemini vision models
GEMINI_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-flash",
]


class GeminiValidator(BaseValidator):
    """
    Gemini-based validator using LangChain's ChatGoogleGenerativeAI.
    
    Uses Google's Gemini models with vision capabilities for image validation.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3-flash-preview",
        temperature: float = 0.1,
        max_retries: int = 3,
        prompt_template: str = DEMO_PROMPT_TEMPLATE,
    ):
        """
        Initialize the Gemini validator.
        
        Args:
            api_key: Google API key
            model_name: Model to use (default: gemini-3-flash-preview)
            temperature: Temperature for generation
            max_retries: Maximum retries on failure
            prompt_template: Prompt template for validation
        """
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_retries=max_retries,
            prompt_template=prompt_template,
        )
    
    def _create_llm(self):
        """Create the LangChain ChatGoogleGenerativeAI instance."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai is required for Gemini validation. "
                "Install it with: pip install langchain-google-genai"
            )
        
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=self.temperature,
            max_retries=self.max_retries,
        )
    
    def _build_multimodal_message(self, prompt: str, image_b64: str):
        """
        Build a multimodal message for Gemini models.
        
        Uses the LangChain format for Google's multimodal API.
        """
        return HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{image_b64}",
                },
            ]
        )


def create_gemini_validator(
    api_key: str,
    model_name: str = "gemini-3-flash-preview",
    temperature: float = 0.1,
    max_retries: int = 3,
) -> GeminiValidator:
    """
    Create a Gemini validator configured for QC validation.
    
    Args:
        api_key: Google API key
        model_name: Model to use
        temperature: Temperature for generation
        max_retries: Maximum retries on failure
        
    Returns:
        Configured GeminiValidator instance
    """
    return GeminiValidator(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_retries=max_retries,
        prompt_template=DEMO_PROMPT_TEMPLATE,
    )
