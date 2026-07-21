"""Stitch same-sized screenshots of vertically scrolling content."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


ImageSource = str | Path | Image.Image


@dataclass(frozen=True)
class StitchSettings:
    crop_top: int = 0
    crop_bottom: int = 0
    window_height: int = 50
    similarity_threshold: float = 99.0

    @property
    def similarity_fraction(self) -> float:
        return self.similarity_threshold / 100.0

    def as_dict(self) -> dict[str, int | float]:
        similarity_threshold = float(self.similarity_threshold)
        threshold = int(similarity_threshold) if similarity_threshold.is_integer() else similarity_threshold
        return {
            "crop_top": self.crop_top,
            "crop_bottom": self.crop_bottom,
            "window_height": self.window_height,
            "similarity_threshold": threshold,
        }


@dataclass(frozen=True)
class Match:
    shift: int
    previous_y: int
    next_y: int
    window_height: int
    similarity: float
    mean_squared_error: float
    information: float


@dataclass(frozen=True)
class StitchResult:
    image: Image.Image
    matches: tuple[Match, ...]
    placements: tuple[int, ...]
    debug_paths: tuple[Path, ...] = ()


class AlignmentError(RuntimeError):
    pass


class StitchingError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        image_index: int,
        partial_stitch: Image.Image,
        failed_image: Image.Image,
        debug_paths: Sequence[Path] = (),
    ) -> None:
        super().__init__(message)
        self.image_index = image_index
        self.partial_stitch = partial_stitch
        self.failed_image = failed_image
        self.debug_paths = tuple(debug_paths)


@dataclass(frozen=True)
class _Candidate:
    shift: int
    previous_y: int
    next_y: int
    coarse_error: float
    coarse_quality: float


def parse_stitch_settings(values: Mapping[str, object]) -> StitchSettings:
    try:
        settings = StitchSettings(
            crop_top=int(values.get("crop_top", 0)),
            crop_bottom=int(values.get("crop_bottom", 0)),
            window_height=int(values.get("window_height", 50)),
            similarity_threshold=float(values.get("similarity_threshold", 99)),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("stitch settings must contain numeric values") from exc
    validate_stitch_settings(settings)
    return settings


def validate_stitch_settings(settings: StitchSettings) -> None:
    if settings.crop_top < 0:
        raise ValueError("crop_top must be a nonnegative integer")
    if settings.crop_bottom < 0:
        raise ValueError("crop_bottom must be a nonnegative integer")
    if settings.window_height < 0:
        raise ValueError("window_height must be a nonnegative integer")
    if not 0 <= settings.similarity_threshold <= 100:
        raise ValueError("similarity_threshold must be between 0 and 100")


def _load_image(source: ImageSource) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    with Image.open(source) as image:
        return image.convert("RGB")


def _window_means(values: np.ndarray, size: int) -> np.ndarray:
    cumulative = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(values, dtype=np.float64)))
    return (cumulative[size:] - cumulative[:-size]) / size


def _information_by_window(image: Image.Image, window_height: int) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    horizontal = np.abs(np.diff(gray, axis=1)).mean(axis=1)
    return _window_means(horizontal, window_height)


def _activity_by_window(
    previous: Image.Image,
    next_image: Image.Image,
    window_height: int,
    pixel_tolerance: int = 12,
) -> np.ndarray:
    previous_pixels = np.asarray(previous, dtype=np.int16)
    next_pixels = np.asarray(next_image, dtype=np.int16)
    changed = (np.abs(previous_pixels - next_pixels).max(axis=2) > pixel_tolerance).mean(axis=1)
    return _window_means(changed, window_height)


def _row_features(image: Image.Image, search_width: int) -> np.ndarray:
    width = min(search_width, image.width)
    reduced = image.resize((width, image.height), resample=Image.Resampling.BILINEAR).convert("L")
    return np.asarray(reduced, dtype=np.float32)


def _pairwise_row_errors(previous: Image.Image, next_image: Image.Image, search_width: int) -> np.ndarray:
    previous_rows = _row_features(previous, search_width)
    next_rows = _row_features(next_image, search_width)
    feature_count = previous_rows.shape[1]
    errors = (
        np.square(previous_rows).mean(axis=1)[:, None]
        + np.square(next_rows).mean(axis=1)[None, :]
        - (2.0 / feature_count) * (previous_rows @ next_rows.T)
    )
    return np.maximum(errors, 0.0)


def find_vertical_match(
    previous: Image.Image,
    next_image: Image.Image,
    *,
    window_height: int = 50,
    similarity_threshold: float = 0.99,
    min_shift: int = 1,
    max_shift: int | None = None,
    search_width: int = 96,
    min_activity: float = 0.03,
    candidates_to_verify: int = 256,
) -> Match:
    if previous.size != next_image.size:
        raise ValueError(f"Images must have identical dimensions; got {previous.size} and {next_image.size}.")
    if not 1 <= window_height < previous.height:
        raise ValueError(f"window_height must be between 1 and {previous.height - 1} for these images.")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")
    if min_shift < 1:
        raise ValueError("min_shift must be at least 1")
    if search_width < 8:
        raise ValueError("search_width must be at least 8")
    if not 0.0 <= min_activity <= 1.0:
        raise ValueError("min_activity must be between 0 and 1")

    greatest_shift = previous.height - window_height
    if max_shift is None:
        max_shift = greatest_shift
    if max_shift < min_shift:
        raise ValueError("max_shift must be greater than or equal to min_shift")
    max_shift = min(max_shift, greatest_shift)

    maximum_mse = (1.0 - similarity_threshold) * (255.0**2)
    coarse_maximum_mse = maximum_mse * 1.25 + 1.0
    row_errors = _pairwise_row_errors(previous, next_image, search_width)
    information = _information_by_window(next_image, window_height)
    activity = _activity_by_window(previous, next_image, window_height)
    candidates: list[_Candidate] = []

    for shift in range(min_shift, max_shift + 1):
        diagonal = np.diagonal(row_errors, offset=-shift)
        if diagonal.size < window_height:
            continue
        errors = _window_means(diagonal, window_height)
        count = errors.size
        active = (
            (activity[:count] >= min_activity)
            & (activity[shift : shift + count] >= min_activity)
            & (errors <= coarse_maximum_mse)
        )
        indices = np.flatnonzero(active)
        if indices.size == 0:
            continue

        qualities = errors[indices] / (1.0 + information[indices])
        keep = min(2, indices.size)
        if indices.size > keep:
            selected = np.argpartition(qualities, keep - 1)[:keep]
            indices = indices[selected]
            qualities = qualities[selected]

        for next_y, quality in zip(indices, qualities, strict=True):
            candidates.append(
                _Candidate(
                    shift=shift,
                    previous_y=shift + int(next_y),
                    next_y=int(next_y),
                    coarse_error=float(errors[next_y]),
                    coarse_quality=float(quality),
                )
            )

    if not candidates:
        raise AlignmentError("No active full-width window met the requested similarity threshold.")

    candidates.sort(key=lambda candidate: candidate.coarse_quality)
    previous_pixels = np.asarray(previous, dtype=np.float32)
    next_pixels = np.asarray(next_image, dtype=np.float32)
    verified: list[tuple[float, float, _Candidate]] = []
    for candidate in candidates[:candidates_to_verify]:
        previous_window = previous_pixels[candidate.previous_y : candidate.previous_y + window_height]
        next_window = next_pixels[candidate.next_y : candidate.next_y + window_height]
        mse = float(np.square(previous_window - next_window).mean())
        similarity = max(0.0, 1.0 - mse / (255.0**2))
        if similarity >= similarity_threshold:
            verified.append((candidate.coarse_quality, mse, candidate))

    if not verified:
        best = candidates[0]
        coarse_similarity = max(0.0, 1.0 - best.coarse_error / (255.0**2))
        raise AlignmentError(
            "Candidate overlaps were found, but none passed full-resolution "
            f"verification (best coarse similarity: {coarse_similarity:.4%})."
        )

    _, mse, candidate = min(verified, key=lambda item: (item[0], item[1], -item[2].shift))
    return Match(
        shift=candidate.shift,
        previous_y=candidate.previous_y,
        next_y=candidate.next_y,
        window_height=window_height,
        similarity=max(0.0, 1.0 - mse / (255.0**2)),
        mean_squared_error=mse,
        information=float(information[candidate.next_y]),
    )


def _save_debug_overlay(
    current_stitch: Image.Image,
    next_image: Image.Image,
    next_placement: int,
    match: Match,
    pair_number: int,
    debug_dir: Path,
) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)
    height = max(current_stitch.height, next_placement + next_image.height)
    size = (current_stitch.width, height)
    stitch_layer = Image.new("RGBA", size, (0, 0, 0, 0))
    stitch_layer.paste(current_stitch.convert("RGBA"), (0, 0))
    next_layer = Image.new("RGBA", size, (0, 0, 0, 0))
    next_layer.paste(next_image.convert("RGBA"), (0, next_placement))
    overlay = Image.blend(stitch_layer, next_layer, 0.5)

    compared_y = next_placement + match.next_y
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(
        (1, compared_y, current_stitch.width - 2, compared_y + match.window_height - 1),
        outline=(0, 102, 255, 255),
        width=4,
    )
    path = debug_dir / f"pair_{pair_number:02d}_overlay.png"
    overlay.save(path)
    return path


def stitch_image_sequence(
    images: Sequence[ImageSource],
    *,
    crop_top: int = 0,
    crop_bottom: int = 0,
    window_height: int = 50,
    similarity_threshold: float = 0.99,
    min_shift: int = 1,
    max_shift: int | None = None,
    search_width: int = 96,
    min_activity: float = 0.03,
    debug: bool = False,
    debug_dir: str | Path = "stitch_debug",
) -> StitchResult:
    if len(images) < 2:
        raise ValueError("At least two images are required.")
    if crop_top < 0 or crop_bottom < 0:
        raise ValueError("crop_top and crop_bottom must be nonnegative")

    loaded = [_load_image(source) for source in images]
    expected_size = loaded[0].size
    for index, image in enumerate(loaded[1:], start=2):
        if image.size != expected_size:
            raise StitchingError(
                f"Image {index} has size {image.size}; expected {expected_size}.",
                image_index=index,
                partial_stitch=loaded[0].copy(),
                failed_image=image.copy(),
            )

    stitch = loaded[0].copy()
    placements = [0]
    matches: list[Match] = []
    debug_paths: list[Path] = []
    debug_path = Path(debug_dir)
    for index, (previous, next_image) in enumerate(zip(loaded, loaded[1:]), start=1):
        try:
            match = find_vertical_match(
                previous,
                next_image,
                window_height=window_height,
                similarity_threshold=similarity_threshold,
                min_shift=min_shift,
                max_shift=max_shift,
                search_width=search_width,
                min_activity=min_activity,
            )
        except (AlignmentError, ValueError) as error:
            raise StitchingError(
                f"Could not stitch image {index + 1}: {error}",
                image_index=index + 1,
                partial_stitch=stitch.copy(),
                failed_image=next_image.copy(),
                debug_paths=debug_paths,
            ) from error

        next_placement = placements[-1] + match.shift
        if debug:
            debug_paths.append(_save_debug_overlay(stitch, next_image, next_placement, match, index, debug_path))

        new_height = max(stitch.height, next_placement + next_image.height)
        combined = Image.new("RGB", (stitch.width, new_height))
        combined.paste(stitch, (0, 0))
        paste_y = next_placement + match.next_y
        tail = next_image.crop((0, match.next_y, next_image.width, next_image.height))
        combined.paste(tail, (0, paste_y))
        stitch = combined
        placements.append(next_placement)
        matches.append(match)

    if crop_top + crop_bottom >= stitch.height:
        raise StitchingError(
            f"crop_top={crop_top} and crop_bottom={crop_bottom} remove all of the {stitch.height}-pixel image.",
            image_index=len(loaded),
            partial_stitch=stitch.copy(),
            failed_image=loaded[-1].copy(),
            debug_paths=debug_paths,
        )
    if crop_top or crop_bottom:
        stitch = stitch.crop((0, crop_top, stitch.width, stitch.height - crop_bottom))

    return StitchResult(
        image=stitch,
        matches=tuple(matches),
        placements=tuple(placements),
        debug_paths=tuple(debug_paths),
    )
