from __future__ import annotations

from typing import List

from PIL import Image, ImageChops

from qc_pipeline.data_loader import Annotation
from qc_pipeline.image_utils import CropConfig, crop_annotation


def _create_base_image() -> Image.Image:
    img = Image.new("RGB", (64, 64), "black")
    for x in range(16, 48):
        for y in range(16, 48):
            img.putpixel((x, y), (255, 0, 0))
    return img


def test_crop_bbox_without_padding() -> None:
    base = _create_base_image()
    annotation = Annotation(
        id=1,
        image_id=1,
        category_id=1,
        bbox=[16, 16, 32, 32],
    )

    crop = crop_annotation(base, annotation, CropConfig(padding=0, mask_polygon=False))
    assert crop.size == (32, 32)
    assert crop.getpixel((0, 0)) == (255, 0, 0)


def test_polygon_mask_applies_transparency() -> None:
    base = Image.new("RGB", (32, 32), "white")
    annotation = Annotation(
        id=2,
        image_id=1,
        category_id=1,
        segmentation=[[8, 8, 24, 8, 24, 24, 8, 24]],
    )

    crop = crop_annotation(base, annotation, CropConfig(padding=2, mask_polygon=True))
    assert crop.mode == "RGBA"
    assert crop.size == (20, 20)
    # Corner outside polygon should be transparent.
    assert crop.getchannel("A").getpixel((0, 0)) == 0
    # Center should remain opaque.

