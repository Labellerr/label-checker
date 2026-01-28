"""Extract structured annotation guidelines from PDF text using Gemini."""
from __future__ import annotations

from typing import List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


class LabelDefinition(BaseModel):
    """Definition of a single annotation label."""
    
    label_name: str = Field(description="The exact name used for the label")
    definition: str = Field(description="Clear description of what this label represents")
    key_characteristics: List[str] = Field(
        description="Visual features to identify this label"
    )
    examples: Optional[List[str]] = Field(
        default=None,
        description="Specific examples if provided"
    )
    common_mistakes: Optional[str] = Field(
        default=None,
        description="Edge cases or errors to avoid"
    )


class ExtractedGuidelines(BaseModel):
    """All label definitions extracted from guidelines PDF."""
    
    labels: List[LabelDefinition] = Field(
        description="List of all label definitions found in the guidelines"
    )


EXTRACTION_PROMPT = """Analyze this annotation guidelines document and extract structured information about each label/class.

🎯 CRITICAL: The key_characteristics field is the MOST IMPORTANT. It must contain detailed VISUAL descriptions that help an AI model recognize objects in images.

For EACH label mentioned, extract:

1. **label_name**: The exact name used for the label

2. **definition**: Clear, concise description of WHAT should be annotated with this label (1-2 sentences)

3. **key_characteristics**: A LIST of SPECIFIC VISUAL FEATURES. This is CRITICAL - be extremely detailed!
   MUST include:
   - Shape: "rectangular", "cylindrical", "curved", "linear", "angular", etc.
   - Size/Scale: "large", "small", "spans multiple meters", "compact", relative size to other objects
   - Color: "metallic silver", "dark", "reflective", "matte black", color patterns
   - Material/Texture: "metal", "glass", "smooth surface", "corrugated", "glossy", "textured"
   - Structure: "tubular", "flat panel", "framework", "solid", "hollow", "multi-layered"
   - Position/Orientation: "horizontal", "vertical", "tilted", "mounted on", "attached to"
   - Distinctive features: unique identifying marks, patterns, connections to other parts
   - Components: visible parts, attachments, sub-structures
   
   Each characteristic should be a separate item in the list.
   Aim for 5-10 specific visual features per label.
   
   Example GOOD characteristics:
   - "Long metallic cylindrical tube running horizontally"
   - "Dark reflective rectangular panels arranged in a grid pattern"
   - "Silver metal framework with diagonal cross-bracing"
   
   Example BAD characteristics (too vague):
   - "Important component"
   - "Used for support"
   - "Part of the system"

4. **examples**: Specific examples or usage scenarios if provided (list, optional)

5. **common_mistakes**: Edge cases, visual similarities with other labels, or annotation errors to avoid (optional)

⚠️ REMEMBER: An AI will use these characteristics to visually identify objects in images. Be specific about APPEARANCE, not just function!

Guidelines document:
{pdf_text}
"""


class GuidelinesExtractor:
    """
    Extract structured annotation guidelines from PDF text using Gemini.
    
    Uses LangChain's structured output feature to reliably extract
    label definitions into Pydantic models.
    """
    
    def __init__(self, api_key: str, model: str = "gemini-3-flash-preview"):
        """
        Initialize the guidelines extractor.
        
        Args:
            api_key: Google API key for Gemini
            model: Gemini model to use (default: gemini-3-flash-preview)
        """
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.1,
        )
    
    def extract(self, pdf_text: str) -> ExtractedGuidelines:
        """
        Extract structured guidelines from PDF text.
        
        Args:
            pdf_text: Raw text extracted from PDF
            
        Returns:
            ExtractedGuidelines object with list of LabelDefinition objects
            
        Raises:
            Exception: If extraction fails or returns invalid data
        """
        if not pdf_text or not pdf_text.strip():
            raise ValueError("PDF text is empty")
        
        try:
            # Create structured output LLM
            structured_llm = self.llm.with_structured_output(ExtractedGuidelines)
            
            # Format prompt with PDF text
            prompt = EXTRACTION_PROMPT.format(pdf_text=pdf_text)
            
            # Invoke and get structured response
            result = structured_llm.invoke(prompt)
            
            if not result.labels:
                raise ValueError("No labels extracted from guidelines")
            
            return result
        
        except Exception as e:
            raise Exception(f"Failed to extract guidelines: {str(e)}") from e
