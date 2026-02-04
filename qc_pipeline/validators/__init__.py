"""Validators package for multi-provider LLM validation."""
from __future__ import annotations

from qc_pipeline.validators.base import (
    BaseValidator,
    ValidationResponse,
    DEFAULT_PROMPT_TEMPLATE,
    DEMO_PROMPT_TEMPLATE,
    GUIDELINES_PROMPT_TEMPLATE,
)

# Lazy imports for validators to avoid requiring all dependencies
def get_gemini_validator():
    """Get the GeminiValidator class (requires langchain-google-genai)."""
    from qc_pipeline.validators.gemini import GeminiValidator, GEMINI_MODELS, create_gemini_validator
    return GeminiValidator, GEMINI_MODELS, create_gemini_validator

def get_openai_validator():
    """Get the OpenAIValidator class (requires langchain-openai)."""
    from qc_pipeline.validators.openai import OpenAIValidator, OPENAI_MODELS, create_openai_validator
    return OpenAIValidator, OPENAI_MODELS, create_openai_validator

def get_anthropic_validator():
    """Get the AnthropicValidator class (requires langchain-anthropic)."""
    from qc_pipeline.validators.anthropic import AnthropicValidator, ANTHROPIC_MODELS, create_anthropic_validator
    return AnthropicValidator, ANTHROPIC_MODELS, create_anthropic_validator


# Provider configurations for UI
PROVIDERS = {
    "gemini": {
        "name": "Google Gemini",
        "models": ["gemini-3-flash-preview", "gemini-3-pro-preview", "gemini-2.5-flash"],
        "default_model": "gemini-3-flash-preview",
        "api_key_env": "GOOGLE_API_KEY",
        "get_validator": get_gemini_validator,
    },
    "openai": {
        "name": "OpenAI",
        "models": ["gpt-5-mini-2025-08-07", "gpt-5.2-pro-2025-12-11", "gpt-5.2-2025-12-11"],
        "default_model": "gpt-5-mini-2025-08-07",
        "api_key_env": "OPENAI_API_KEY",
        "get_validator": get_openai_validator,
    },
    "anthropic": {
        "name": "Anthropic",
        "models": ["claude-opus-4-5-20251101", "claude-haiku-4-5-20251001", "claude-sonnet-4-5-20250929"],
        "default_model": "claude-sonnet-4-5-20250929",
        "api_key_env": "ANTHROPIC_API_KEY",
        "get_validator": get_anthropic_validator,
    },
}


def create_validator(
    provider: str,
    api_key: str,
    model_name: str = None,
    temperature: float = 0.1,
    max_retries: int = 3,
) -> BaseValidator:
    """
    Factory function to create a validator for any supported provider.
    
    Args:
        provider: Provider name ('gemini', 'openai', 'anthropic')
        api_key: API key for the provider
        model_name: Model to use (defaults to provider's default)
        temperature: Temperature for generation
        max_retries: Maximum retries on failure
        
    Returns:
        Configured validator instance
        
    Raises:
        ValueError: If provider is not supported
        ImportError: If required dependencies are not installed
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}. Choose from: {list(PROVIDERS.keys())}")
    
    provider_config = PROVIDERS[provider]
    model_name = model_name or provider_config["default_model"]
    
    ValidatorClass, _, _ = provider_config["get_validator"]()
    
    return ValidatorClass(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_retries=max_retries,
    )


__all__ = [
    "BaseValidator",
    "ValidationResponse",
    "DEFAULT_PROMPT_TEMPLATE",
    "DEMO_PROMPT_TEMPLATE",
    "GUIDELINES_PROMPT_TEMPLATE",
    "PROVIDERS",
    "create_validator",
    "get_gemini_validator",
    "get_openai_validator",
    "get_anthropic_validator",
]
