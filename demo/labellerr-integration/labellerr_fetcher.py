"""
Labellerr SDK integration module.

Handles fetching annotations and images from Labellerr platform
using the SDK, filtered by annotation status.
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests

# Add SDK path - adjust this based on where SDK is installed
SDK_PATH = Path(__file__).parent.parent.parent.parent / "SDKPython-1"
if SDK_PATH.exists():
    sys.path.insert(0, str(SDK_PATH))

try:
    from labellerr import LabellerrClient
    from labellerr.core import schemas
    from labellerr.core.projects import LabellerrProject
except ImportError as e:
    raise ImportError(
        f"Failed to import Labellerr SDK. Make sure SDK is installed or path is correct. Error: {e}"
    )


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
        self.client = LabellerrClient(api_key, api_secret, client_id)
        self.client_id = client_id
        self.project_id = project_id
        self.project = LabellerrProject(self.client, project_id=project_id)
    
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
        
        print(f"Creating export for statuses: {statuses}...")
        export = self.project.create_export(export_config)
        report_id = export.report_id
        print(f"Export created with ID: {report_id}")
        
        # Poll until export is ready with detailed debugging
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
                
                print(f"\n[Poll #{poll_count}] Elapsed: {elapsed:.1f}s")
                
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
                        
                        print(f"  export_status: {export_state}")
                        print(f"  is_completed: {is_completed}")
                        print(f"  download_url: {download_url is not None}")
                        
                        # Check for failure
                        if export_state.lower() == "failed":
                            raise ValueError(f"Export failed: {export_status}")
                        
                        # Check for success (both is_completed=True AND export_status="created")
                        if is_completed and export_state.lower() == "created" and download_url:
                            print(f"\n✓ Export completed after {elapsed:.1f}s")
                            break
                        elif is_completed and not download_url:
                            print(f"  Warning: Export marked complete but no download URL yet...")
                    else:
                        print(f"  Report ID {report_id} not found in status list")
                else:
                    print(f"  No status/response list in response: {list(result.keys())}")
                    last_status = result
                
            except ValueError:
                raise  # Re-raise export failures
            except Exception as e:
                print(f"\n[Poll #{poll_count}] Error checking status: {type(e).__name__}: {e}")
            
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
        
        print(f"Export completed. Downloading from: {download_url}")
        
        # Download the export file
        response = requests.get(download_url, timeout=60)
        response.raise_for_status()
        
        # Save to output directory
        coco_json_path = output_dir / "annotations.json"
        coco_json_path.write_bytes(response.content)
        
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
        
        print(f"Found {len(image_ids)} unique images referenced in annotations")
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
            # client_id must be in query params (as per team requirement)
            url = f"https://api.labellerr.com/cdn-web/files_links?project_id={self.project_id}&client_id={self.client_id}&uuid={unique_id}"
            
            payload = json.dumps({"file_ids": [file_id]})
            
            # Make POST request with the file_ids in the body
            # SDK will automatically add api_key, api_secret, client_id to headers
            response = self.client.make_request(
                "POST", 
                url, 
                request_id=unique_id,
                data=payload,
                extra_headers={"Content-Type": "application/json"}
            )
            
            # Extract the signed URL from response
            # Actual format: {"fileLinks": [{"file_id": "...", "url": "..."}]}
            if isinstance(response, dict):
                # Try different possible response structures (camelCase and snake_case)
                file_links = (response.get("fileLinks") or 
                             response.get("file_links") or 
                             response.get("response", {}).get("fileLinks") or
                             response.get("response", {}).get("file_links"))
                
                if file_links and len(file_links) > 0:
                    # The URL is in the 'url' field, not 'signed_url'
                    signed_url = file_links[0].get("url") or file_links[0].get("signed_url") or file_links[0].get("link")
                    if signed_url:
                        return signed_url
            
            return None
        except Exception as e:
            print(f"(API error getting signed URL: {e})")
            import traceback
            traceback.print_exc()
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
                print(f"  [{idx}/{len(images_info)}] Warning: No URL for image {image_id}, skipping")
                continue
            
            # Download image
            try:
                if "Fetching" not in locals().get('last_msg', ''):
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
        
        print(f"Successfully downloaded {len(downloaded_images)}/{len(images_info)} images")
        return downloaded_images


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
        export_timeout: Maximum time to wait for export completion in seconds (default: 900 = 15 minutes)
        poll_interval: Time between status checks in seconds (default: 3.0)
        
    Returns:
        Tuple of (annotations_path, images_dir, metadata)
    """
    fetcher = LabellerrFetcher(api_key, api_secret, client_id, project_id)
    
    # Create and download export
    coco_json_path, export_metadata = fetcher.create_and_download_export(
        statuses=statuses,
        output_dir=output_dir,
        timeout=export_timeout,
        poll_interval=poll_interval,
    )
    
    # Download all images referenced in the export
    downloaded_images = fetcher.download_project_images(
        coco_json_path=coco_json_path,
        output_dir=output_dir,
    )
    
    images_dir = output_dir / "images"
    
    metadata = {
        "export_id": export_metadata.get("report_id"),
        "statuses": statuses,
        "total_images": len(downloaded_images),
        "coco_json": str(coco_json_path),
        "images_dir": str(images_dir),
    }
    
    return coco_json_path, images_dir, metadata

