"""
qc_pipeline/validators/__init__.py
===================================
Factory and exports for all QC validators.

Supported providers
-------------------
- ``"gemini"``   — Google Gemini via langchain-google-genai
- ``"openai"``   — OpenAI GPT via langchain-openai
- ``"anthropic"`` — Anthropic Claude via langchain-anthropic
- ``"smartqc"``  — Three-signal adaptive engine (CLIP + DINOv2 + YOLOv8)
                    No external LLM API key required.
"""

from __future__ import annotations

from typing import Any, Optional


def create_validator(
    provider: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs: Any,
):
    """
    Instantiate a validator for the given provider.

    Parameters
    ----------
    provider : str
        One of ``"gemini"``, ``"openai"``, ``"anthropic"``, ``"smartqc"``.
    api_key : str or None
        Provider API key.  Not required for ``"smartqc"``.
    model_name : str or None
        Model string.  Not required for ``"smartqc"``.
    **kwargs
        Extra keyword arguments forwarded to the validator constructor.
        For ``"smartqc"``:
            - ``device``           (str)  — e.g. ``"cuda"`` or ``"cpu"``
            - ``yolo_model_path``  (str)  — path to ``best.pt``
            - ``use_grayscale``    (bool) — default ``True``
            - ``batch_size``       (int)  — default ``16``
            - ``yolo_conf``        (float)— default ``0.25``

    Returns
    -------
    Validator instance.

    Examples
    --------
    >>> # LLM-based validators (existing)
    >>> v = create_validator("gemini", api_key="...", model_name="gemini-2.5-flash")
    >>> v = create_validator("openai", api_key="...", model_name="gpt-5-mini-2025-08-07")
    >>> v = create_validator("anthropic", api_key="...", model_name="claude-sonnet-4-6")

    >>> # SmartQC — no API key needed
    >>> v = create_validator("smartqc")
    >>> v = create_validator("smartqc", yolo_model_path="runs/detect/best.pt")
    """
    provider = provider.lower().strip()

    if provider == "gemini":
        from .gemini import GeminiValidator
        return GeminiValidator(api_key=api_key, model_name=model_name, **kwargs)

    elif provider == "openai":
        from .openai import OpenAIValidator
        return OpenAIValidator(api_key=api_key, model_name=model_name, **kwargs)

    elif provider == "anthropic":
        from .anthropic import AnthropicValidator
        return AnthropicValidator(api_key=api_key, model_name=model_name, **kwargs)

    elif provider == "smartqc":
        from .smartqc import SmartQCValidator
        return SmartQCValidator(**kwargs)

    else:
        supported = ("gemini", "openai", "anthropic", "smartqc")
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: {supported}"
        )


__all__ = ["create_validator"]
