from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw

from .data_loader import Annotation


@dataclass
class CropConfig:
    padding: int = 0
    mask_polygon: bool = True
    background_color: Tuple[int, int, int, int] = (0, 0, 0, 0)


def _polygon_bbox(segments: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for segment in segments:
        for i, value in enumerate(segment):
            if i % 2 == 0:
                xs.append(value)
            else:
                ys.append(value)
    if not xs or not ys:
        raise ValueError("Polygon segmentation is empty")
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return min_x, min_y, max_x, max_y


def _expand_box(
    left: float,
    top: float,
    right: float,
    bottom: float,
    padding: int,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    pad = max(0, padding)
    return (
        max(int(left) - pad, 0),
        max(int(top) - pad, 0),
        min(int(right) + pad, width),
        min(int(bottom) + pad, height),
    )


def crop_annotation(image: Image.Image, annotation: Annotation, config: CropConfig | None = None) -> Image.Image:
    if config is None:
        config = CropConfig()

    if annotation.has_bbox():
        x, y, w, h = annotation.bbox  # type: ignore[misc]
        left, top = x, y
        right, bottom = x + w, y + h
    elif annotation.has_polygon():
        left, top, right, bottom = _polygon_bbox(annotation.segmentation or [])
    else:
        raise ValueError(f"Annotation {annotation.id} missing bbox and polygon")

    left_i, top_i, right_i, bottom_i = _expand_box(
        left, top, right, bottom, config.padding, image.width, image.height
    )

    crop = image.crop((left_i, top_i, right_i, bottom_i))

    if annotation.has_polygon() and config.mask_polygon:
        mask = Image.new("L", crop.size, 0)
        draw = ImageDraw.Draw(mask)
        for segment in annotation.segmentation or []:
            points = []
            for i in range(0, len(segment), 2):
                px = segment[i] - left_i
                py = segment[i + 1] - top_i
                points.append((px, py))
            if points:
                draw.polygon(points, fill=255)
        crop = crop.convert("RGBA")
        crop.putalpha(mask)
        if config.background_color[3] > 0:
            background = Image.new("RGBA", crop.size, config.background_color)
            background.alpha_composite(crop)
            crop = background
    return crop


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

