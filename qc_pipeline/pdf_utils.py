"""Utility functions for PDF text extraction."""
from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: Path | str) -> str:
    """
    Extract all text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file to extract text from
        
    Returns:
        Concatenated text content from all pages
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If PDF cannot be read or parsed
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        reader = PdfReader(pdf_path)
        text_parts = []
        
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text.strip())
        
        if not text_parts:
            raise ValueError("No text content found in PDF")
        
        return "\n\n".join(text_parts)
    
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {str(e)}") from e
