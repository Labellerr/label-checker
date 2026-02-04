"""
Labellerr SDK integration module.

Handles fetching annotations and images from Labellerr platform
using the SDK, filtered by annotation status.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests

from qc_pipeline.fetchers.base import FetchResult

logger = logging.getLogger(__name__)

# Add SDK path - check multiple possible locations
_SDK_PATHS = [
    Path(__file__).parent.parent.parent.parent / "SDKPython-1",  # From qc_pipeline/fetchers/
    Path(__file__).parent.parent.parent / "SDKPython-1",  # Alternative
    Path.home() / "SDKPython-1",  # Home directory
]

for sdk_path in _SDK_PATHS:
    if sdk_path.exists():
        sys.path.insert(0, str(sdk_path))
        logger.debug(f"Added Labellerr SDK path: {sdk_path}")
        break

# Try to import Labellerr SDK
_LABELLERR_AVAILABLE = False
try:
    from labellerr import LabellerrClient
    from labellerr.core import schemas
    from labellerr.core.projects import LabellerrProject
    _LABELLERR_AVAILABLE = True
except ImportError:
    LabellerrClient = None
    schemas = None
    LabellerrProject = None


def is_labellerr_available() -> bool:
    """Check if Labellerr SDK is available."""
    return _LABELLERR_AVAILABLE


# Status mapping from UI-friendly names to API status names
STATUS_MAPPING = {
    "reviewer_layer": ["review", "r_assigned"],
    "client_reviewer_layer": ["client_review", "cr_assigned"],
    "completed": ["accepted"],
}


def map_statuses_to_api(ui_statuses: List[str]) -> List[str]:
    """
    Convert UI-friendly status names to Labellerr API status names.
    
    Args:
        ui_statuses: List of UI status names (e.g., ["reviewer_layer", "completed"])
        
    Returns:
        List of API status names (e.g., ["review", "r_assigned", "accepted"])
    """
    api_statuses = []
    for status in ui_statuses:
        if status in STATUS_MAPPING:
            api_statuses.extend(STATUS_MAPPING[status])
        else:
            # Pass through unknown statuses (backwards compatibility)
            api_statuses.append(status)
    return api_statuses


class LabellerrFetcher:
    """Handles fetching data from Labellerr platform."""
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        client_id: str,
        project_id: str,
    ):
        """
        Initialize Labellerr fetcher.
        
        Args:
            api_key: Labellerr API key
            api_secret: Labellerr API secret
            client_id: Labellerr client ID
            project_id: Labellerr project ID
        """
        if not _LABELLERR_AVAILABLE:
            raise ImportError(
                "Labellerr SDK is not installed. Please install it to use Labellerr integration."
            )
        
        self.client = LabellerrClient(api_key, api_secret, client_id)
        self.client_id = client_id
        self.project_id = project_id
        self.project = LabellerrProject(self.client, project_id=project_id)
    
    def fetch_data(
        self,
        output_dir: Path,
        statuses: List[str],
        export_timeout: int = 900,
        poll_interval: float = 3.0,
    ) -> FetchResult:
        """
        Fetch annotation data and images from Labellerr.
        
        Args:
            output_dir: Directory to save fetched data
            statuses: List of annotation statuses to filter
            export_timeout: Maximum time to wait for export completion
            poll_interval: Time between status checks
            
        Returns:
            FetchResult with paths to COCO JSON and images directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Map UI statuses to API statuses
        api_statuses = map_statuses_to_api(statuses)
        
        # Create and download export
        coco_json_path, export_metadata = self.create_and_download_export(
            statuses=api_statuses,
            output_dir=output_dir,
            timeout=export_timeout,
            poll_interval=poll_interval,
        )
        
        # Download all images referenced in the export
        downloaded_images = self.download_project_images(
            coco_json_path=coco_json_path,
            output_dir=output_dir,
        )
        
        images_dir = output_dir / "images"
        
        metadata = {
            "source": "labellerr",
            "export_id": export_metadata.get("report_id"),
            "statuses": statuses,
            "api_statuses": api_statuses,
            "total_images": len(downloaded_images),
            "coco_json": str(coco_json_path),
            "images_dir": str(images_dir),
        }
        
        return FetchResult(
            coco_json_path=coco_json_path,
            images_dir=images_dir,
            metadata=metadata,
        )
    
    def create_and_download_export(
        self,
        statuses: List[str],
        output_dir: Path,
        export_name: str = "QC Export",
        timeout: int = 900,
        poll_interval: float = 3.0,
    ) -> Tuple[Path, Dict]:
        """
        Create export filtered by status and download COCO JSON.
        
        Args:
            statuses: List of annotation statuses to export (e.g., ["accepted", "review"])
            output_dir: Directory to save the export
            export_name: Name for the export
            timeout: Maximum time to wait for export completion in seconds (default: 900 = 15 minutes)
            poll_interval: Time between status checks in seconds (default: 3.0)
            
        Returns:
            Tuple of (path to downloaded COCO JSON, export metadata)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create export configuration
        export_config = schemas.CreateExportParams(
            export_name=export_name,
            export_description=f"QC validation export for statuses: {', '.join(statuses)}",
            export_format="coco_json",  # COCO JSON format
            statuses=statuses,
            export_destination=schemas.ExportDestination.LOCAL,
        )
        
        logger.info(f"Creating export for statuses: {statuses}...")
        print(f"Creating export for statuses: {statuses}...")
        export = self.project.create_export(export_config)
        report_id = export.report_id
        logger.info(f"Export created with ID: {report_id}")
        print(f"Export created with ID: {report_id}")
        
        # Poll until export is ready with detailed debugging
        logger.info(f"Waiting for export to complete (timeout: {timeout}s, poll interval: {poll_interval}s)...")
        print(f"Waiting for export to complete (timeout: {timeout}s, poll interval: {poll_interval}s)...")
        print("(This may take a few minutes depending on the number of annotations...)")
        
        start_time = time.time()
        last_status = None
        poll_count = 0
        export_status = None
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Export timed out after {timeout}s. Last status: {last_status}")
            
            poll_count += 1
            try:
                # Use the project's check_export_status method directly for better debugging
                result = self.project.check_export_status([report_id])
                
                logger.debug(f"[Poll #{poll_count}] Elapsed: {elapsed:.1f}s")
                
                # The API returns data in "response" key, not "status" key
                # Try both keys for compatibility
                status_list = result.get("status", []) or result.get("response", [])
                
                if status_list:
                    # Find our export in the status list
                    for status_item in status_list:
                        if status_item.get("report_id") == report_id:
                            export_status = status_item
                            last_status = export_status
                            break
                    
                    if export_status:
                        is_completed = export_status.get("is_completed", False)
                        export_state = export_status.get("export_status", "unknown")
                        download_url = export_status.get("download_url")
                        
                        logger.debug(f"  export_status: {export_state}, is_completed: {is_completed}")
                        
                        # Check for failure
                        if export_state.lower() == "failed":
                            raise ValueError(f"Export failed: {export_status}")
                        
                        # Check for success (both is_completed=True AND export_status="created")
                        if is_completed and export_state.lower() == "created" and download_url:
                            logger.info(f"Export completed after {elapsed:.1f}s")
                            print(f"\n✓ Export completed after {elapsed:.1f}s")
                            break
                        elif is_completed and not download_url:
                            logger.debug("Export marked complete but no download URL yet...")
                    else:
                        logger.debug(f"Report ID {report_id} not found in status list")
                else:
                    logger.debug(f"No status/response list in response: {list(result.keys())}")
                    last_status = result
                
            except ValueError:
                raise  # Re-raise export failures
            except Exception as e:
                logger.warning(f"[Poll #{poll_count}] Error checking status: {type(e).__name__}: {e}")
            
            # Wait before next poll
            time.sleep(poll_interval)
        
        # Extract download URL
        if not export_status:
            raise ValueError(f"Export status not found for report_id: {report_id}")
            
        download_url = export_status.get("download_url")
        if not download_url:
            raise ValueError(f"No download URL in export response: {export_status}")
        
        # Extract the actual URL string if it's a dictionary
        if isinstance(download_url, dict):
            download_url = download_url.get("url")
            if not download_url:
                raise ValueError("No URL found in download_url dictionary")
        
        logger.info(f"Downloading export from: {download_url[:50]}...")
        print(f"Export completed. Downloading...")
        
        # Download the export file
        response = requests.get(download_url, timeout=60)
        response.raise_for_status()
        
        # Save to output directory
        coco_json_path = output_dir / "annotations.json"
        coco_json_path.write_bytes(response.content)
        
        logger.info(f"Export downloaded to: {coco_json_path}")
        print(f"Export downloaded to: {coco_json_path}")
        
        return coco_json_path, export_status
    
    def extract_image_refs_from_coco(self, coco_json_path: Path) -> Set[int]:
        """
        Extract unique image IDs from COCO JSON.
        
        Args:
            coco_json_path: Path to COCO JSON file
            
        Returns:
            Set of unique image IDs referenced in the annotations
        """
        with open(coco_json_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)
        
        # Extract unique image IDs from annotations
        image_ids = set()
        for annotation in coco_data.get("annotations", []):
            image_id = annotation.get("image_id")
            if image_id is not None:
                image_ids.add(image_id)
        
        logger.info(f"Found {len(image_ids)} unique images referenced in annotations")
        return image_ids
    
    def _get_file_signed_url(self, file_id: str) -> Optional[str]:
        """
        Get signed URL for a file from Labellerr API using the correct endpoint.
        
        Args:
            file_id: Labellerr file ID
            
        Returns:
            Signed URL for the file, or None if not available
        """
        try:
            unique_id = str(uuid.uuid4())
            
            # Use the correct endpoint for getting signed URLs
            url = f"https://api.labellerr.com/cdn-web/files_links?project_id={self.project_id}&client_id={self.client_id}&uuid={unique_id}"
            
            payload = json.dumps({"file_ids": [file_id]})
            
            # Make POST request with the file_ids in the body
            response = self.client.make_request(
                "POST", 
                url, 
                request_id=unique_id,
                data=payload,
                extra_headers={"Content-Type": "application/json"}
            )
            
            # Extract the signed URL from response
            if isinstance(response, dict):
                file_links = (response.get("fileLinks") or 
                             response.get("file_links") or 
                             response.get("response", {}).get("fileLinks") or
                             response.get("response", {}).get("file_links"))
                
                if file_links and len(file_links) > 0:
                    signed_url = file_links[0].get("url") or file_links[0].get("signed_url") or file_links[0].get("link")
                    if signed_url:
                        return signed_url
            
            return None
        except Exception as e:
            logger.warning(f"API error getting signed URL: {e}")
            return None
    
    def download_project_images(
        self,
        coco_json_path: Path,
        output_dir: Path,
    ) -> Dict[int, Path]:
        """
        Download images referenced in COCO JSON from Labellerr project.
        
        Args:
            coco_json_path: Path to COCO JSON file
            output_dir: Directory to save images
            
        Returns:
            Dictionary mapping image_id to local file path
        """
        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Load COCO data to get image info
        with open(coco_json_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)
        
        if not isinstance(coco_data, dict):
            raise ValueError(f"Expected COCO JSON dict format, got {type(coco_data)}. "
                           "Make sure export_format is set to 'coco_json'")
        
        images_info = coco_data.get("images", [])
        logger.info(f"Downloading {len(images_info)} images...")
        print(f"Downloading {len(images_info)} images...")
        
        downloaded_images = {}
        
        for idx, image_info in enumerate(images_info, 1):
            image_id = image_info.get("id")
            file_name = image_info.get("file_name", f"image_{image_id}.jpg")
            
            # Get image URL from COCO data (Labellerr exports include URLs)
            image_url = image_info.get("coco_url") or image_info.get("url") or image_info.get("file_url")
            
            # If no URL, try to fetch using labellerr_file_id
            if not image_url and "labellerr_file_id" in image_info:
                labellerr_file_id = image_info["labellerr_file_id"]
                print(f"  [{idx}/{len(images_info)}] Fetching {file_name} from Labellerr API...", end=" ")
                image_url = self._get_file_signed_url(labellerr_file_id)
                
                if not image_url:
                    print("✗ (no signed URL returned)")
                    continue
            
            if not image_url:
                logger.warning(f"No URL for image {image_id}, skipping")
                print(f"  [{idx}/{len(images_info)}] Warning: No URL for image {image_id}, skipping")
                continue
            
            # Download image
            try:
                print(f"  [{idx}/{len(images_info)}] Downloading {file_name}...", end=" ")
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                
                # Save to images directory
                image_path = images_dir / file_name
                image_path.write_bytes(response.content)
                
                downloaded_images[image_id] = image_path
                print("✓")
                
            except Exception as e:
                print(f"✗ (error: {e})")
                continue
        
        logger.info(f"Successfully downloaded {len(downloaded_images)}/{len(images_info)} images")
        print(f"Successfully downloaded {len(downloaded_images)}/{len(images_info)} images")
        return downloaded_images


def create_labellerr_fetcher(
    api_key: str,
    api_secret: str,
    client_id: str,
    project_id: str,
) -> LabellerrFetcher:
    """Create a Labellerr fetcher instance."""
    return LabellerrFetcher(
        api_key=api_key,
        api_secret=api_secret,
        client_id=client_id,
        project_id=project_id,
    )


# Backwards compatibility function
def fetch_labellerr_data(
    api_key: str,
    api_secret: str,
    client_id: str,
    project_id: str,
    statuses: List[str],
    output_dir: Path,
    export_timeout: int = 900,
    poll_interval: float = 3.0,
) -> Tuple[Path, Path, Dict]:
    """
    High-level function to fetch all data from Labellerr.
    
    Args:
        api_key: Labellerr API key
        api_secret: Labellerr API secret
        client_id: Labellerr client ID
        project_id: Labellerr project ID
        statuses: List of annotation statuses to filter
        output_dir: Directory to save all outputs
        export_timeout: Maximum time to wait for export completion in seconds
        poll_interval: Time between status checks in seconds
        
    Returns:
        Tuple of (annotations_path, images_dir, metadata)
    """
    fetcher = create_labellerr_fetcher(api_key, api_secret, client_id, project_id)
    result = fetcher.fetch_data(
        output_dir=Path(output_dir),
        statuses=statuses,
        export_timeout=export_timeout,
        poll_interval=poll_interval,
    )
    return result.coco_json_path, result.images_dir, result.metadata
