from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps


@dataclass(frozen=True)
class PreparedImage:
    path: Path
    compressed: bool
    original_size: int
    final_size: int
    original_dimensions: tuple[int, int]
    final_dimensions: tuple[int, int]


def _rgb_image(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def _has_alpha(image: Image.Image) -> bool:
    return image.mode in {"RGBA", "LA"} or "transparency" in image.info


def _png_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, "PNG", optimize=True, compress_level=9)
    return output.getvalue()


def _best_scaled_png(image: Image.Image, target_size: int) -> tuple[bytes, tuple[int, int]] | None:
    low, high = 0.0, 1.0
    best = None
    tested_sizes: set[tuple[int, int]] = set()
    for _ in range(12):
        scale = (low + high) / 2.0
        size = max(1, round(image.width * scale)), max(1, round(image.height * scale))
        if size in tested_sizes:
            break
        tested_sizes.add(size)
        resized = image.resize(size, Image.Resampling.LANCZOS)
        payload = _png_bytes(resized)
        if len(payload) <= target_size:
            best = payload, size
            low = scale
        else:
            high = scale
    if best is not None:
        return best
    smallest = image.resize((1, 1), Image.Resampling.LANCZOS)
    payload = _png_bytes(smallest)
    return (payload, (1, 1)) if len(payload) <= target_size else None


def _jpeg_bytes(image: Image.Image, quality: int) -> bytes:
    output = BytesIO()
    image.save(output, "JPEG", quality=quality, optimize=True, progressive=True, subsampling=2)
    return output.getvalue()


def _best_jpeg(image: Image.Image, target_size: int) -> bytes | None:
    low, high = 72, 95
    best = None
    while low <= high:
        quality = (low + high) // 2
        payload = _jpeg_bytes(image, quality)
        if len(payload) <= target_size:
            best = payload
            low = quality + 1
        else:
            high = quality - 1
    return best


def prepare_image_attachment(path: Path, max_bytes: int) -> PreparedImage:
    if max_bytes <= 0:
        raise ValueError("image upload limit must be positive")
    original_size = path.stat().st_size
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).copy()
    original_dimensions = image.size
    if original_size <= max_bytes:
        return PreparedImage(path, False, original_size, original_size, original_dimensions, original_dimensions)

    target_size = max(1, max_bytes - min(4096, max_bytes // 20))
    optimized_png = path.with_name(f"{path.stem}-compressed.png")
    image.save(optimized_png, "PNG", optimize=True, compress_level=9)
    optimized_size = optimized_png.stat().st_size
    if optimized_size <= target_size:
        return PreparedImage(
            optimized_png,
            True,
            original_size,
            optimized_size,
            original_dimensions,
            original_dimensions,
        )

    if _has_alpha(image):
        fitted = _best_scaled_png(image, target_size)
        if fitted is None:
            raise ValueError(f"could not compress {path.name} below {max_bytes} bytes")
        payload, final_dimensions = fitted
        optimized_png.write_bytes(payload)
        return PreparedImage(
            optimized_png,
            True,
            original_size,
            len(payload),
            original_dimensions,
            final_dimensions,
        )

    working = _rgb_image(image)
    jpeg_path = path.with_name(f"{path.stem}-compressed.jpg")
    while True:
        payload = _best_jpeg(working, target_size)
        if payload is not None:
            jpeg_path.write_bytes(payload)
            return PreparedImage(
                jpeg_path,
                True,
                original_size,
                len(payload),
                original_dimensions,
                working.size,
            )

        lowest_quality = _jpeg_bytes(working, 72)
        if working.width == 1 and working.height == 1:
            raise ValueError(f"could not compress {path.name} below {max_bytes} bytes")
        scale = min(0.92, (target_size / len(lowest_quality)) ** 0.5 * 0.96)
        new_size = (
            max(1, round(working.width * scale)),
            max(1, round(working.height * scale)),
        )
        if new_size == working.size:
            new_size = max(1, working.width - 1), max(1, working.height - 1)
        working = working.resize(new_size, Image.Resampling.LANCZOS)
