"""Anthropic validator using LangChain for vision-based validation."""
from __future__ import annotations

import logging
from typing import Optional

from langchain_core.messages import HumanMessage

from qc_pipeline.validators.base import BaseValidator, DEMO_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# Available Anthropic vision models
ANTHROPIC_MODELS = [
    "claude-opus-4-5-20251101",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
]


class AnthropicValidator(BaseValidator):
    """
    Anthropic-based validator using LangChain's ChatAnthropic.
    
    Uses Anthropic's Claude models with vision capabilities for image validation.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "claude-sonnet-4-5-20250929",
        temperature: float = 0.1,
        max_retries: int = 3,
        prompt_template: str = DEMO_PROMPT_TEMPLATE,
    ):
        """
        Initialize the Anthropic validator.
        
        Args:
            api_key: Anthropic API key
            model_name: Model to use (default: claude-sonnet-4-5-20250929)
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
        """Create the LangChain ChatAnthropic instance."""
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic is required for Anthropic validation. "
                "Install it with: pip install langchain-anthropic"
            )
        
        return ChatAnthropic(
            model=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature,
            max_retries=self.max_retries,
        )
    
    def _build_multimodal_message(self, prompt: str, image_b64: str):
        """
        Build a multimodal message for Anthropic Claude models.
        
        Uses the Anthropic vision API format with base64 image data.
        """
        return HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    },
                },
            ]
        )


def create_anthropic_validator(
    api_key: str,
    model_name: str = "claude-sonnet-4-5-20250929",
    temperature: float = 0.1,
    max_retries: int = 3,
) -> AnthropicValidator:
    """
    Create an Anthropic validator configured for QC validation.
    
    Args:
        api_key: Anthropic API key
        model_name: Model to use
        temperature: Temperature for generation
        max_retries: Maximum retries on failure
        
    Returns:
        Configured AnthropicValidator instance
    """
    return AnthropicValidator(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_retries=max_retries,
        prompt_template=DEMO_PROMPT_TEMPLATE,
    )
