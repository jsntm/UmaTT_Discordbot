"""Stitch screenshots of vertically scrolling content."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
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
    x_shift: int = 0
    content_left: int = 0
    content_right: int = 0
    comparison_left: int = 0
    comparison_right: int = 0
    inlier_count: int = 0


@dataclass(frozen=True)
class StitchResult:
    image: Image.Image
    matches: tuple[Match, ...]
    placements: tuple[int, ...]
    debug_paths: tuple[Path, ...] = ()
    x_placements: tuple[int, ...] = ()


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


@dataclass(frozen=True)
class _FeatureSet:
    scale: float
    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray | None


@dataclass(frozen=True)
class _TranslationCandidate:
    x_shift: int
    y_shift: int
    inlier_count: int
    match_fraction: float
    rank: float
    verification_bounds: tuple[int, int, int, int]
    content_left: int
    content_right: int


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


def _extract_feature_set(image: Image.Image, scale: float) -> _FeatureSet:
    gray = np.asarray(image.convert("L"))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    detector = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return _FeatureSet(scale, tuple(keypoints), descriptors)


def _candidate_match_fraction(
    previous_pixels: np.ndarray,
    next_pixels: np.ndarray,
    x_shift: int,
    y_shift: int,
    bounds: tuple[int, int, int, int],
) -> float:
    left = max(0, x_shift, bounds[0])
    top = max(0, y_shift, bounds[1])
    right = min(previous_pixels.shape[1], x_shift + next_pixels.shape[1], bounds[2])
    bottom = min(previous_pixels.shape[0], y_shift + next_pixels.shape[0], bounds[3])
    if right - left < 32 or bottom - top < 16:
        return 0.0

    previous_region = previous_pixels[top:bottom, left:right]
    next_region = next_pixels[top - y_shift : bottom - y_shift, left - x_shift : right - x_shift]
    previous_gray = cv2.cvtColor(previous_region, cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(next_region, cv2.COLOR_RGB2GRAY)
    horizontal = np.maximum(
        cv2.absdiff(previous_gray[:, 1:], previous_gray[:, :-1]),
        cv2.absdiff(next_gray[:, 1:], next_gray[:, :-1]),
    )
    vertical = np.maximum(
        cv2.absdiff(previous_gray[1:, :], previous_gray[:-1, :]),
        cv2.absdiff(next_gray[1:, :], next_gray[:-1, :]),
    )
    informative = np.zeros(previous_gray.shape, dtype=bool)
    informative[:, :-1] |= horizontal > 8
    informative[:-1, :] |= vertical > 8
    differences = np.abs(previous_gray.astype(np.int16) - next_gray.astype(np.int16))[informative]
    if differences.size < 200:
        return 0.0
    return float(np.mean(differences <= 16))


def _translation_candidates(
    previous: Image.Image,
    next_image: Image.Image,
    previous_features: _FeatureSet,
    next_features: _FeatureSet,
    *,
    window_height: int,
    min_shift: int,
    max_shift: int | None,
) -> list[_TranslationCandidate]:
    if previous_features.descriptors is None or next_features.descriptors is None:
        raise AlignmentError("Not enough visual features were found in both screenshots.")
    if previous_features.scale != next_features.scale:
        raise ValueError("Feature sets must use the same analysis scale.")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(previous_features.descriptors, next_features.descriptors, k=2)
    good_matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance]
    if len(good_matches) < 4:
        raise AlignmentError(f"Not enough unambiguous feature matches were found ({len(good_matches)}).")

    scale = previous_features.scale
    greatest_shift = previous.height - min(window_height, previous.height - 1)
    allowed_shift = greatest_shift if max_shift is None else min(max_shift, greatest_shift)
    previous_points = []
    next_points = []
    for match in good_matches:
        previous_point = np.asarray(previous_features.keypoints[match.queryIdx].pt, dtype=np.float32)
        next_point = np.asarray(next_features.keypoints[match.trainIdx].pt, dtype=np.float32)
        y_shift = float((previous_point[1] - next_point[1]) / scale)
        if min_shift <= y_shift <= allowed_shift:
            previous_points.append(previous_point)
            next_points.append(next_point)
    if len(previous_points) < 4:
        raise AlignmentError("No feature cluster represented downward scrolling motion.")

    previous_array = np.asarray(previous_points, dtype=np.float32)
    next_array = np.asarray(next_points, dtype=np.float32)
    offsets = previous_array - next_array
    remaining = np.arange(len(previous_array))
    previous_pixels = np.asarray(previous.convert("RGB"))
    next_pixels = np.asarray(next_image.convert("RGB"))
    candidates: list[_TranslationCandidate] = []

    for _ in range(12):
        if remaining.size < 4:
            break
        translation, inlier_mask = cv2.estimateTranslation2D(
            next_array[remaining],
            previous_array[remaining],
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
            maxIters=2000,
            confidence=0.99,
            refineIters=0,
        )
        if translation is None or not np.isfinite(translation).all():
            break
        translation = np.asarray(translation, dtype=np.float32)
        all_inliers = np.linalg.norm(offsets - translation, axis=1) <= 2.5
        inlier_indices = np.flatnonzero(all_inliers)
        if inlier_indices.size >= 4:
            full_shift = np.median(offsets[inlier_indices], axis=0) / scale
            x_shift, y_shift = (int(round(value)) for value in full_shift)
            duplicate = any(
                candidate.x_shift == x_shift and candidate.y_shift == y_shift
                for candidate in candidates
            )
            if not duplicate and min_shift <= y_shift <= allowed_shift:
                aligned_width = min(previous.width, x_shift + next_image.width) - max(0, x_shift)
                aligned_height = min(previous.height, y_shift + next_image.height) - max(0, y_shift)
                if aligned_width >= 32 and aligned_height >= window_height:
                    points = previous_array[inlier_indices] / scale
                    span_x = max(1.0, float(np.ptp(points[:, 0])))
                    span_y = max(1.0, float(np.ptp(points[:, 1])))
                    verification_margin_x = max(24.0, span_x * 0.08)
                    verification_margin_y = max(24.0, span_y * 0.08)
                    verification_bounds = (
                        int(np.floor(points[:, 0].min() - verification_margin_x)),
                        int(np.floor(points[:, 1].min() - verification_margin_y)),
                        int(np.ceil(points[:, 0].max() + verification_margin_x)),
                        int(np.ceil(points[:, 1].max() + verification_margin_y)),
                    )
                    match_fraction = _candidate_match_fraction(
                        previous_pixels,
                        next_pixels,
                        x_shift,
                        y_shift,
                        verification_bounds,
                    )
                    content_margin = max(96.0, span_x * 0.5)
                    candidates.append(
                        _TranslationCandidate(
                            x_shift=x_shift,
                            y_shift=y_shift,
                            inlier_count=int(inlier_indices.size),
                            match_fraction=match_fraction,
                            rank=float(inlier_indices.size) * match_fraction**2,
                            verification_bounds=verification_bounds,
                            content_left=int(np.floor(points[:, 0].min() - content_margin)),
                            content_right=int(np.ceil(points[:, 0].max() + content_margin)),
                        )
                    )

        local_inliers = np.flatnonzero(inlier_mask.ravel()) if inlier_mask is not None else np.empty(0, dtype=int)
        if local_inliers.size < 3:
            break
        remaining = np.delete(remaining, local_inliers)

    if not candidates:
        raise AlignmentError("Feature matches were found, but no usable scrolling translation could be verified.")

    static_cutoff = max(min_shift, 4, min(50, round(min(previous.height, next_image.height) * 0.02)))
    moving_candidates = [candidate for candidate in candidates if candidate.y_shift >= static_cutoff]
    ranked = moving_candidates or candidates
    advancing_candidates = [
        candidate
        for candidate in ranked
        if candidate.y_shift + next_image.height > previous.height
    ]
    ranked = advancing_candidates or ranked
    return sorted(ranked, key=lambda candidate: candidate.rank, reverse=True)


def _comparison_match(
    previous: Image.Image,
    next_image: Image.Image,
    candidate: _TranslationCandidate,
    *,
    window_height: int,
    similarity_threshold: float,
    min_activity: float,
) -> Match:
    left = max(0, candidate.x_shift, candidate.verification_bounds[0])
    right = min(previous.width, candidate.x_shift + next_image.width, candidate.verification_bounds[2])
    top = max(0, candidate.y_shift)
    bottom = min(previous.height, candidate.y_shift + next_image.height)
    if right - left < 32 or bottom - top < window_height:
        raise AlignmentError("The translated screenshots do not have a large enough informative overlap.")

    previous_pixels = np.asarray(previous.convert("RGB"), dtype=np.float32)[top:bottom, left:right]
    next_pixels = np.asarray(next_image.convert("RGB"), dtype=np.float32)[
        top - candidate.y_shift : bottom - candidate.y_shift,
        left - candidate.x_shift : right - candidate.x_shift,
    ]
    row_errors = np.square(previous_pixels - next_pixels).mean(axis=(1, 2))
    previous_gray = cv2.cvtColor(previous_pixels.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    row_information = np.abs(np.diff(previous_gray, axis=1)).mean(axis=1)
    errors = _window_means(row_errors, window_height)
    information = _window_means(row_information, window_height)
    similarities = np.maximum(0.0, 1.0 - errors / (255.0**2))
    minimum_information = max(0.5, min_activity * 20.0)
    valid = np.flatnonzero((similarities >= similarity_threshold) & (information >= minimum_information))
    if valid.size == 0:
        best_similarity = float(similarities.max()) if similarities.size else 0.0
        raise AlignmentError(
            "A scrolling translation was found, but no informative comparison window passed "
            f"the similarity threshold (best: {best_similarity:.4%})."
        )

    window_index = int(valid[-1])
    previous_y = top + window_index
    next_y = previous_y - candidate.y_shift
    content_left = max(0, min(previous.width - 1, candidate.content_left))
    content_right = max(content_left + 1, min(previous.width, candidate.content_right))
    return Match(
        shift=candidate.y_shift,
        previous_y=previous_y,
        next_y=next_y,
        window_height=window_height,
        similarity=float(similarities[window_index]),
        mean_squared_error=float(errors[window_index]),
        information=float(information[window_index]),
        x_shift=candidate.x_shift,
        content_left=content_left,
        content_right=content_right,
        comparison_left=left,
        comparison_right=right,
        inlier_count=candidate.inlier_count,
    )


def find_translation_match(
    previous: Image.Image,
    next_image: Image.Image,
    *,
    window_height: int = 50,
    similarity_threshold: float = 0.99,
    min_shift: int = 1,
    max_shift: int | None = None,
    search_width: int = 96,
    min_activity: float = 0.03,
    previous_features: _FeatureSet | None = None,
    next_features: _FeatureSet | None = None,
) -> Match:
    if not 1 <= window_height < previous.height:
        raise ValueError(f"window_height must be between 1 and {previous.height - 1} for these images.")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")
    if min_shift < 1:
        raise ValueError("min_shift must be at least 1")

    if previous_features is None or next_features is None:
        scale = min(1.0, 1600.0 / max(previous.width, previous.height, next_image.width, next_image.height))
        previous_features = _extract_feature_set(previous, scale)
        next_features = _extract_feature_set(next_image, scale)

    feature_error: AlignmentError | None = None
    try:
        candidates = _translation_candidates(
            previous,
            next_image,
            previous_features,
            next_features,
            window_height=window_height,
            min_shift=min_shift,
            max_shift=max_shift,
        )
        errors = []
        for candidate in candidates:
            try:
                return _comparison_match(
                    previous,
                    next_image,
                    candidate,
                    window_height=window_height,
                    similarity_threshold=similarity_threshold,
                    min_activity=min_activity,
                )
            except AlignmentError as exc:
                errors.append(str(exc))
        feature_error = AlignmentError(errors[0] if errors else "No feature translation passed verification.")
    except (AlignmentError, cv2.error) as exc:
        feature_error = AlignmentError(str(exc))

    if previous.size == next_image.size:
        try:
            legacy = find_vertical_match(
                previous,
                next_image,
                window_height=window_height,
                similarity_threshold=similarity_threshold,
                min_shift=min_shift,
                max_shift=max_shift,
                search_width=search_width,
                min_activity=min_activity,
            )
            values = {
                **legacy.__dict__,
                "content_right": previous.width,
                "comparison_right": previous.width,
            }
            return Match(**values)
        except (AlignmentError, ValueError):
            pass
    raise feature_error or AlignmentError("The screenshots could not be aligned.")


def _save_debug_overlay(
    current_stitch: Image.Image,
    current_origin: tuple[int, int],
    previous_placement: tuple[int, int],
    next_image: Image.Image,
    next_placement: tuple[int, int],
    match: Match,
    pair_number: int,
    debug_dir: Path,
) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)
    left = min(current_origin[0], next_placement[0])
    top = min(current_origin[1], next_placement[1])
    right = max(current_origin[0] + current_stitch.width, next_placement[0] + next_image.width)
    bottom = max(current_origin[1] + current_stitch.height, next_placement[1] + next_image.height)
    size = (right - left, bottom - top)
    stitch_layer = Image.new("RGBA", size, (0, 0, 0, 0))
    stitch_layer.alpha_composite(current_stitch.convert("RGBA"), (current_origin[0] - left, current_origin[1] - top))
    next_layer = Image.new("RGBA", size, (0, 0, 0, 0))
    translucent_next = next_image.convert("RGBA")
    translucent_next.putalpha(128)
    next_layer.alpha_composite(translucent_next, (next_placement[0] - left, next_placement[1] - top))
    overlay = Image.alpha_composite(stitch_layer, next_layer)

    compared_left = previous_placement[0] + match.comparison_left - left
    compared_right = previous_placement[0] + match.comparison_right - left
    compared_y = previous_placement[1] + match.previous_y - top
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(
        (compared_left, compared_y, compared_right - 1, compared_y + match.window_height - 1),
        outline=(0, 102, 255, 255),
        width=4,
    )
    path = debug_dir / f"pair_{pair_number:02d}_overlay.png"
    overlay.save(path)
    return path


def _composite_layer(
    canvas: Image.Image,
    canvas_origin: tuple[int, int],
    layer: Image.Image,
    layer_placement: tuple[int, int],
) -> tuple[Image.Image, tuple[int, int]]:
    left = min(canvas_origin[0], layer_placement[0])
    top = min(canvas_origin[1], layer_placement[1])
    right = max(canvas_origin[0] + canvas.width, layer_placement[0] + layer.width)
    bottom = max(canvas_origin[1] + canvas.height, layer_placement[1] + layer.height)
    combined = Image.new("RGBA", (right - left, bottom - top), (0, 0, 0, 0))
    combined.alpha_composite(canvas.convert("RGBA"), (canvas_origin[0] - left, canvas_origin[1] - top))
    combined.alpha_composite(layer.convert("RGBA"), (layer_placement[0] - left, layer_placement[1] - top))
    return combined, (left, top)


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
    crop_auto: bool = False,
    debug: bool = False,
    debug_dir: str | Path = "stitch_debug",
) -> StitchResult:
    if len(images) < 2:
        raise ValueError("At least two images are required.")
    if crop_top < 0 or crop_bottom < 0:
        raise ValueError("crop_top and crop_bottom must be nonnegative")

    loaded = [_load_image(source) for source in images]
    analysis_scale = min(1.0, 1600.0 / max(max(image.width, image.height) for image in loaded))
    feature_sets = [_extract_feature_set(image, analysis_scale) for image in loaded]

    stitch = loaded[0].convert("RGBA")
    stitch_origin = (0, 0)
    placements = [0]
    x_placements = [0]
    matches: list[Match] = []
    auto_regions: list[tuple[int, int]] = []
    debug_paths: list[Path] = []
    debug_path = Path(debug_dir)
    for index, (previous, next_image) in enumerate(zip(loaded, loaded[1:]), start=1):
        try:
            match = find_translation_match(
                previous,
                next_image,
                window_height=window_height,
                similarity_threshold=similarity_threshold,
                min_shift=min_shift,
                max_shift=max_shift,
                search_width=search_width,
                min_activity=min_activity,
                previous_features=feature_sets[index - 1],
                next_features=feature_sets[index],
            )
        except (AlignmentError, ValueError) as error:
            raise StitchingError(
                f"Could not stitch image {index + 1}: {error}",
                image_index=index + 1,
                partial_stitch=stitch.copy(),
                failed_image=next_image.copy(),
                debug_paths=debug_paths,
            ) from error

        previous_placement = (x_placements[-1], placements[-1])
        next_placement = (
            previous_placement[0] + match.x_shift,
            previous_placement[1] + match.shift,
        )
        if debug:
            debug_paths.append(
                _save_debug_overlay(
                    stitch,
                    stitch_origin,
                    previous_placement,
                    next_image,
                    next_placement,
                    match,
                    index,
                    debug_path,
                )
            )

        tail = next_image.crop((0, match.next_y, next_image.width, next_image.height))
        tail_placement = (next_placement[0], next_placement[1] + match.next_y)
        stitch, stitch_origin = _composite_layer(stitch, stitch_origin, tail, tail_placement)
        placements.append(next_placement[1])
        x_placements.append(next_placement[0])
        matches.append(match)
        if match.content_right > match.content_left:
            auto_regions.append(
                (
                    previous_placement[0] + match.content_left,
                    previous_placement[0] + match.content_right,
                )
            )

    if crop_auto and auto_regions:
        auto_left = max(stitch_origin[0], min(region[0] for region in auto_regions))
        auto_right = min(stitch_origin[0] + stitch.width, max(region[1] for region in auto_regions))
        if auto_right - auto_left >= 32:
            left = auto_left - stitch_origin[0]
            right = auto_right - stitch_origin[0]
            stitch = stitch.crop((left, 0, right, stitch.height))
            stitch_origin = (auto_left, stitch_origin[1])

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
        x_placements=tuple(x_placements),
    )
