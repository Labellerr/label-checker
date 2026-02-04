"""Base protocol and types for data fetchers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Protocol, Tuple


@dataclass
class FetchResult:
    """Result from a data fetch operation."""
    coco_json_path: Path
    images_dir: Path
    metadata: Dict[str, Any]
    
    def __iter__(self):
        """Allow tuple unpacking: coco_path, images_dir, metadata = fetch_result"""
        return iter((self.coco_json_path, self.images_dir, self.metadata))


class DataFetcherProtocol(Protocol):
    """
    Protocol for data fetchers.
    
    All data fetchers must implement this interface to be compatible
    with the QC pipeline.
    """
    
    def fetch_data(
        self,
        output_dir: Path,
        **kwargs,
    ) -> FetchResult:
        """
        Fetch annotation data and images.
        
        Args:
            output_dir: Directory to save fetched data
            **kwargs: Provider-specific arguments
            
        Returns:
            FetchResult with paths to COCO JSON and images directory
        """
        ...


# Available platforms for UI
PLATFORMS = {
    "local": {
        "name": "Upload Files",
        "description": "Upload images and COCO JSON directly",
        "requires_credentials": False,
    },
    "labellerr": {
        "name": "Labellerr",
        "description": "Fetch from Labellerr annotation platform",
        "requires_credentials": True,
        "credentials": ["api_key", "api_secret", "client_id", "project_id"],
    },
}
