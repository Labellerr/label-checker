"""OpenAI validator using LangChain for vision-based validation."""
from __future__ import annotations

import logging
from typing import Optional

from langchain_core.messages import HumanMessage

from qc_pipeline.validators.base import BaseValidator, DEMO_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# Available OpenAI vision models
OPENAI_MODELS = [
    "gpt-5-mini-2025-08-07",
    "gpt-5.2-pro-2025-12-11",
    "gpt-5.2-2025-12-11",
]


class OpenAIValidator(BaseValidator):
    """
    OpenAI-based validator using LangChain's ChatOpenAI.
    
    Uses OpenAI's vision models (GPT-4o, GPT-4 Turbo) for image validation.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-5-mini-2025-08-07",
        temperature: float = 0.1,
        max_retries: int = 3,
        prompt_template: str = DEMO_PROMPT_TEMPLATE,
    ):
        """
        Initialize the OpenAI validator.
        
        Args:
            api_key: OpenAI API key
            model_name: Model to use (default: gpt-5-mini-2025-08-07)
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
        """Create the LangChain ChatOpenAI instance."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai is required for OpenAI validation. "
                "Install it with: pip install langchain-openai"
            )
        
        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature,
            max_retries=self.max_retries,
        )
    
    def _build_multimodal_message(self, prompt: str, image_b64: str):
        """
        Build a multimodal message for OpenAI vision models.
        
        Uses the OpenAI vision API format with image_url containing base64 data.
        """
        return HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                        "detail": "high",  # Use high detail for better accuracy
                    },
                },
            ]
        )


def create_openai_validator(
    api_key: str,
    model_name: str = "gpt-5-mini-2025-08-07",
    temperature: float = 0.1,
    max_retries: int = 3,
) -> OpenAIValidator:
    """
    Create an OpenAI validator configured for QC validation.
    
    Args:
        api_key: OpenAI API key
        model_name: Model to use
        temperature: Temperature for generation
        max_retries: Maximum retries on failure
        
    Returns:
        Configured OpenAIValidator instance
    """
    return OpenAIValidator(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_retries=max_retries,
        prompt_template=DEMO_PROMPT_TEMPLATE,
    )
