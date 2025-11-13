from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import json


@dataclass(frozen=True)
class Category:
    id: int
    name: str
    supercategory: Optional[str] = None


@dataclass(frozen=True)
class ImageInfo:
    id: int
    file_name: str
    width: int
    height: int
    coco_url: Optional[str] = None
    flickr_url: Optional[str] = None
    license: Optional[int] = None
    date_captured: Optional[str] = None

    def full_path(self, image_root: Path) -> Path:
        return (image_root / self.file_name).resolve()


@dataclass(frozen=True)
class Annotation:
    id: int
    image_id: int
    category_id: int
    bbox: Optional[Sequence[float]] = None
    segmentation: Optional[Sequence[Sequence[float]]] = None
    area: Optional[float] = None
    iscrowd: Optional[int] = None

    def has_polygon(self) -> bool:
        return bool(self.segmentation)

    def has_bbox(self) -> bool:
        return bool(self.bbox) and len(self.bbox) == 4


class CocoDataset:
    def __init__(
        self,
        categories: Dict[int, Category],
        images: Dict[int, ImageInfo],
        annotations: List[Annotation],
    ) -> None:
        self._categories = categories
        self._images = images
        self._annotations = annotations

    @property
    def categories(self) -> Dict[int, Category]:
        return self._categories

    @property
    def images(self) -> Dict[int, ImageInfo]:
        return self._images

    @property
    def annotations(self) -> Sequence[Annotation]:
        return self._annotations

    def iter_annotations(
        self,
    ) -> Iterator[Tuple[ImageInfo, Annotation, Category]]:
        for ann in self._annotations:
            image = self._images.get(ann.image_id)
            if image is None:
                raise KeyError(f"Annotation {ann.id} references missing image {ann.image_id}")
            category = self._categories.get(ann.category_id)
            if category is None:
                raise KeyError(
                    f"Annotation {ann.id} references missing category {ann.category_id}"
                )
            yield image, ann, category


def load_coco_dataset(json_path: Path | str) -> CocoDataset:
    """Load a dataset from a COCO-format JSON file."""
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = {
        entry["id"]: Category(
            id=entry["id"],
            name=entry["name"],
            supercategory=entry.get("supercategory"),
        )
        for entry in coco.get("categories", [])
    }

    images = {
        entry["id"]: ImageInfo(
            id=entry["id"],
            file_name=entry["file_name"],
            width=entry["width"],
            height=entry["height"],
            coco_url=entry.get("coco_url"),
            flickr_url=entry.get("flickr_url"),
            license=entry.get("license"),
            date_captured=entry.get("date_captured"),
        )
        for entry in coco.get("images", [])
    }

    annotations: List[Annotation] = []
    for entry in coco.get("annotations", []):
        segmentation = entry.get("segmentation")
        # Ensure polygon segments are normalized as list of lists.
        if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], (int, float)):
            segmentation = [segmentation]  # type: ignore[assignment]
        annotations.append(
            Annotation(
                id=entry["id"],
                image_id=entry["image_id"],
                category_id=entry["category_id"],
                bbox=entry.get("bbox"),
                segmentation=segmentation,  # type: ignore[arg-type]
                area=entry.get("area"),
                iscrowd=entry.get("iscrowd"),
            )
        )

    return CocoDataset(categories=categories, images=images, annotations=annotations)

