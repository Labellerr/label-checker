"""Local file fetcher for handling direct file uploads."""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from qc_pipeline.fetchers.base import FetchResult

logger = logging.getLogger(__name__)


class LocalFileFetcher:
    """
    Fetcher for local file uploads.
    
    Handles:
    - Individual image files
    - Zip archives containing images
    - COCO JSON annotation files
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the local file fetcher.
        
        Args:
            output_dir: Directory to save processed files. If None, uses temp dir.
        """
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_data(
        self,
        output_dir: Path,
        image_files: Optional[List[Union[str, Path]]] = None,
        coco_json_path: Optional[Union[str, Path]] = None,
        images_zip_path: Optional[Union[str, Path]] = None,
    ) -> FetchResult:
        """
        Process uploaded files and prepare them for QC validation.
        
        Args:
            output_dir: Directory to save processed data
            image_files: List of image file paths
            coco_json_path: Path to COCO JSON file
            images_zip_path: Path to zip file containing images
            
        Returns:
            FetchResult with paths to processed data
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Process images
        image_count = 0
        
        # Handle zip file
        if images_zip_path:
            image_count += self._extract_images_from_zip(images_zip_path, images_dir)
        
        # Handle individual image files
        if image_files:
            image_count += self._copy_image_files(image_files, images_dir)
        
        # Process COCO JSON
        if coco_json_path:
            processed_json_path = self._process_coco_json(coco_json_path, output_dir, images_dir)
        else:
            # Generate a basic COCO JSON from images if none provided
            processed_json_path = self._generate_coco_json(images_dir, output_dir)
        
        metadata = {
            "source": "local_upload",
            "total_images": image_count,
            "images_dir": str(images_dir),
            "coco_json": str(processed_json_path),
        }
        
        logger.info(f"Processed {image_count} images from local upload")
        
        return FetchResult(
            coco_json_path=processed_json_path,
            images_dir=images_dir,
            metadata=metadata,
        )
    
    def _extract_images_from_zip(self, zip_path: Union[str, Path], images_dir: Path) -> int:
        """Extract images from a zip file."""
        zip_path = Path(zip_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        count = 0
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                # Skip directories and hidden files
                if member.endswith('/') or member.startswith('__MACOSX'):
                    continue
                
                # Check if it's an image
                ext = Path(member).suffix.lower()
                if ext in image_extensions:
                    # Extract to images_dir with flat structure
                    filename = Path(member).name
                    target_path = images_dir / filename
                    
                    # Handle duplicates
                    if target_path.exists():
                        stem = target_path.stem
                        suffix = target_path.suffix
                        i = 1
                        while target_path.exists():
                            target_path = images_dir / f"{stem}_{i}{suffix}"
                            i += 1
                    
                    with zf.open(member) as src, open(target_path, 'wb') as dst:
                        dst.write(src.read())
                    count += 1
        
        logger.info(f"Extracted {count} images from {zip_path.name}")
        return count
    
    def _copy_image_files(self, image_files: List[Union[str, Path]], images_dir: Path) -> int:
        """Copy individual image files to the images directory."""
        count = 0
        for img_path in image_files:
            img_path = Path(img_path)
            if img_path.exists() and img_path.is_file():
                target_path = images_dir / img_path.name
                
                # Handle duplicates
                if target_path.exists():
                    stem = target_path.stem
                    suffix = target_path.suffix
                    i = 1
                    while target_path.exists():
                        target_path = images_dir / f"{stem}_{i}{suffix}"
                        i += 1
                
                shutil.copy2(img_path, target_path)
                count += 1
        
        return count
    
    def _process_coco_json(
        self,
        coco_json_path: Union[str, Path],
        output_dir: Path,
        images_dir: Path,
    ) -> Path:
        """Process and validate COCO JSON file."""
        coco_json_path = Path(coco_json_path)
        
        with open(coco_json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # Update image paths to point to local images directory
        for img in coco_data.get('images', []):
            filename = Path(img.get('file_name', '')).name
            # Check if image exists in images_dir
            local_path = images_dir / filename
            if local_path.exists():
                img['file_name'] = filename
        
        # Save processed COCO JSON
        processed_path = output_dir / "annotations.json"
        with open(processed_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2)
        
        return processed_path
    
    def _generate_coco_json(self, images_dir: Path, output_dir: Path) -> Path:
        """Generate a basic COCO JSON from images in directory (no annotations)."""
        from PIL import Image
        
        images = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        
        for idx, img_path in enumerate(sorted(images_dir.iterdir())):
            if img_path.suffix.lower() in image_extensions:
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                except Exception as e:
                    logger.warning(f"Could not read image {img_path}: {e}")
                    continue
                
                images.append({
                    "id": idx,
                    "file_name": img_path.name,
                    "width": width,
                    "height": height,
                })
        
        coco_data = {
            "info": {
                "description": "Generated from uploaded images",
                "version": "1.0",
            },
            "images": images,
            "annotations": [],
            "categories": [],
        }
        
        processed_path = output_dir / "annotations.json"
        with open(processed_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2)
        
        logger.info(f"Generated COCO JSON with {len(images)} images (no annotations)")
        return processed_path


def create_local_fetcher(output_dir: Optional[Path] = None) -> LocalFileFetcher:
    """Create a local file fetcher instance."""
    return LocalFileFetcher(output_dir=output_dir)
