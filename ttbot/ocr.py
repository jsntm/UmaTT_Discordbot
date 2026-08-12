from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from ttbot import config
from ttbot.names import NameMatcher
from ttbot.storage import get_ocr_setting, set_ocr_setting


ScreenshotType = Literal["top", "bottom"]
SCORE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[,.]\s*\d{3}|\d{1,3}(?:(?:[,.]\s*|\s+)\d{3})+|\d{3,})(?![A-Za-z0-9])"
)


@dataclass(frozen=True)
class OCRRow:
    name: str
    score: int
    raw_name: str = ""


@dataclass
class OCRResult:
    screenshot_type: str
    rows: list[OCRRow]
    raw_text: str
    region: tuple[int, int, int, int]
    highlight_path: Path | None = None
    region_mode: str = "manual"


class OCRFailure(Exception):
    def __init__(self, message: str, result: OCRResult | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.result = result

    def to_user_message(self) -> str:
        if not self.result:
            return self.message
        x1, y1, x2, y2 = self.result.region
        raw = self.result.raw_text.strip() or "(no OCR text returned)"
        raw = raw[:1200]
        return (
            f"{self.result.screenshot_type} screenshot OCR output was malformed: {self.message}\n"
            f"{self.result.region_mode.title()} OCR region top-left ({x1}, {y1}), bottom-right ({x2}, {y2})\n"
            f"Raw OCR output:\n```text\n{raw}\n```"
        )


class OCRService:
    PANEL_ASPECT_RATIO = 0.58
    AUTO_MAX_CROP_PIXELS = 1_300_000
    # Stable score-panel geometry lets the Close button reproduce the header-derived crop.
    CLOSE_BUTTON_WIDTH_RATIO = 0.4026
    CLOSE_PANEL_BOTTOM_GAP_RATIO = 0.0378
    CLOSE_PANEL_HEADER_HEIGHT_RATIO = 0.0877

    def __init__(self, matcher: NameMatcher) -> None:
        self.matcher = matcher
        self._easyocr_reader = None

    def process_image(
        self,
        user_id: str,
        image_path: Path,
        screenshot_type: ScreenshotType,
        work_dir: Path,
        *,
        update_coords: tuple[int | None, int | None, int | None, int | None] | None = None,
        candidate_names: Iterable[str] | None = None,
        manual: bool = False,
    ) -> OCRResult:
        candidate_names = tuple(candidate_names) if candidate_names is not None else None
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if update_coords is not None:
            manual = True
            current_region = self._scaled_region(user_id, screenshot_type, image.size)
            x1, y1, x2, y2 = (
                current if updated is None else updated
                for current, updated in zip(current_region, update_coords)
            )
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > image.width or y2 > image.height:
                raise OCRFailure("coordinates must be inside the image and bottom-right must be below/right of top-left")
            setting = {"top_left": [x1, y1], "bottom_right": [x2, y2], "base_size": [image.width, image.height]}
            set_ocr_setting(user_id, screenshot_type, setting)

        if manual:
            region = self._scaled_region(user_id, screenshot_type, image.size)
            region_mode = "manual"
            highlight_color = "blue"
        else:
            try:
                region = self.detect_ocr_region(image)
            except ValueError as exc:
                raise OCRFailure(
                    f"automatic region detection failed: {exc}. Retry with `manual:True` or adjust `/change-ocr`."
                ) from exc
            region_mode = "automatic"
            highlight_color = "red"
        highlight_path = self._write_highlight(
            image,
            region,
            work_dir / f"{screenshot_type}-{region_mode}-ocr-region.png",
            color=highlight_color,
        )
        crop = image.crop(region)
        if not manual:
            crop = self._limit_crop_size(crop)
        crop_path = work_dir / f"{screenshot_type}-crop.png"
        crop.save(crop_path)

        if candidate_names is None:
            raw_text = self._extract_text(crop_path, screenshot_type, work_dir)
            rows = self.parse_rows(raw_text)
        else:
            raw_text = self._extract_text(crop_path, screenshot_type, work_dir, candidate_names)
            rows = self.parse_rows(raw_text, candidate_names=candidate_names)
        result = OCRResult(
            screenshot_type=screenshot_type,
            rows=rows,
            raw_text=raw_text,
            region=region,
            highlight_path=highlight_path,
            region_mode=region_mode,
        )
        if not rows:
            raise OCRFailure("no score rows could be parsed", result)
        return result

    def _scaled_region(self, user_id: str, screenshot_type: str, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
        setting = get_ocr_setting(user_id, screenshot_type)
        base_w, base_h = setting.get("base_size", [image_size[0], image_size[1]])
        left, top = setting["top_left"]
        right, bottom = setting["bottom_right"]
        scale_x = image_size[0] / max(1, base_w)
        scale_y = image_size[1] / max(1, base_h)
        x1 = round(left * scale_x)
        y1 = round(top * scale_y)
        x2 = round(right * scale_x)
        y2 = round(bottom * scale_y)
        x1 = max(0, min(image_size[0] - 1, x1))
        y1 = max(0, min(image_size[1] - 1, y1))
        x2 = max(x1 + 1, min(image_size[0], x2))
        y2 = max(y1 + 1, min(image_size[1], y2))
        return x1, y1, x2, y2

    def detect_ocr_region(self, image: Image.Image) -> tuple[int, int, int, int]:
        import cv2
        import numpy as np

        pixels = np.asarray(image.convert("RGB"))
        image_height, image_width = pixels.shape[:2]
        hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
        green = cv2.inRange(
            hsv,
            np.array([35, 100, 80], dtype=np.uint8),
            np.array([95, 255, 255], dtype=np.uint8),
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(3, image_width // 100), max(3, image_height // 300)),
        )
        green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        minimum_width = max(120, round(min(image_width, image_height) * 0.50))
        candidates = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            height_ratio = height / max(1, width)
            if width < minimum_width or not 0.065 <= height_ratio <= 0.13:
                continue
            coverage = float(np.count_nonzero(green[y : y + height, x : x + width])) / (width * height)
            if coverage < 0.55:
                continue
            candidates.append((coverage * width * height, x, y, width, height))

        for _, header_x, header_y, header_width, header_height in sorted(candidates, reverse=True):
            panel_height = round(header_width / self.PANEL_ASPECT_RATIO)
            if header_y + panel_height > image_height:
                continue
            left = max(0, header_x + round(header_width * 0.20))
            top = max(0, header_y + header_height)
            right = min(image_width, header_x + header_width - round(header_width * 0.03))
            bottom = header_y + panel_height
            if right - left >= 100 and bottom - top >= 100:
                return left, top, right, bottom

        return self._detect_close_button_region(pixels)

    def _detect_close_button_region(self, pixels) -> tuple[int, int, int, int]:
        import cv2

        image_height, image_width = pixels.shape[:2]
        minimum_dimension = min(image_width, image_height)
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        button_candidates = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            aspect_ratio = width / max(1, height)
            rectangularity = cv2.contourArea(contour) / max(1, width * height)
            if (
                y > image_height * 0.55
                and width > minimum_dimension * 0.12
                and minimum_dimension * 0.025 < height < minimum_dimension * 0.15
                and 3.2 < aspect_ratio < 4.2
                and rectangularity > 0.90
            ):
                button_candidates.append((width * height, x, y, width, height))
        if not button_candidates:
            raise ValueError("the green Score Info header and Close button were not found")

        _, button_x, button_y, button_width, button_height = max(button_candidates)
        button_center = button_x + button_width / 2
        expected_panel_width = button_width / self.CLOSE_BUTTON_WIDTH_RATIO
        panel_candidates = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if (
                2.2 * button_width < width < 2.8 * button_width
                and x < button_center < x + width
                and y < button_y - 3 * button_height
                and y + height >= button_y + button_height
                and height > width
            ):
                rectangularity = cv2.contourArea(contour) / max(1, width * height)
                panel_candidates.append(
                    (
                        abs(width - expected_panel_width),
                        -height,
                        x,
                        y,
                        width,
                        height,
                        rectangularity,
                    )
                )

        geometric_bottom = round(
            button_y
            + button_height
            + expected_panel_width * self.CLOSE_PANEL_BOTTOM_GAP_RATIO
        )
        geometric_top = (
            geometric_bottom
            - round(expected_panel_width / self.PANEL_ASPECT_RATIO)
            + round(expected_panel_width * self.CLOSE_PANEL_HEADER_HEIGHT_RATIO)
        )
        if panel_candidates:
            _, _, panel_x, panel_y, panel_width, panel_height, rectangularity = min(panel_candidates)
            panel_top = panel_y + max(1, round(panel_width * 0.0015))
            top = (
                panel_top
                if rectangularity >= 0.99 or abs(panel_top - geometric_top) <= 5
                else geometric_top
            )
            left = panel_x + round(panel_width * 0.1995)
            right = panel_x + panel_width - round(panel_width * 0.029)
            bottom = panel_y + panel_height - max(2, round(panel_width * 0.006))
        else:
            panel_x = round(button_center - expected_panel_width / 2)
            left = panel_x + round(expected_panel_width * 0.20)
            right = panel_x + round(expected_panel_width * 0.97)
            top = geometric_top
            bottom = geometric_bottom

        left = max(0, left)
        top = max(0, top)
        right = min(image_width, right)
        bottom = min(image_height, bottom)
        if right - left < 100 or bottom - top < 100:
            raise ValueError("the Close button was found, but the detected Score Info panel is too small")
        return left, top, right, bottom

    def write_manual_highlight(
        self,
        user_id: str,
        image_path: Path,
        screenshot_type: ScreenshotType,
        output_path: Path,
    ) -> tuple[Path, tuple[int, int, int, int]]:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        region = self._scaled_region(user_id, screenshot_type, image.size)
        return self._write_highlight(image, region, output_path, color="blue"), region

    def _limit_crop_size(self, crop: Image.Image) -> Image.Image:
        if crop.width * crop.height <= self.AUTO_MAX_CROP_PIXELS:
            return crop
        scale = (self.AUTO_MAX_CROP_PIXELS / (crop.width * crop.height)) ** 0.5
        size = max(1, round(crop.width * scale)), max(1, round(crop.height * scale))
        return crop.resize(size, Image.Resampling.LANCZOS)

    def _write_highlight(
        self,
        image: Image.Image,
        region: tuple[int, int, int, int],
        path: Path,
        *,
        color: str = "blue",
    ) -> Path:
        overlay = image.convert("RGBA")
        highlight = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(highlight)
        if color == "red":
            fill, outline = (255, 45, 45, 45), (220, 25, 25, 255)
        else:
            fill, outline = (50, 130, 255, 48), (30, 105, 255, 255)
        line_width = max(3, round(min(image.size) * 0.006))
        draw.rectangle(region, fill=fill, outline=outline, width=line_width)
        highlighted = Image.alpha_composite(overlay, highlight).convert("RGB")
        highlighted.save(path)
        return path

    def _extract_text(
        self,
        crop_path: Path,
        screenshot_type: str,
        work_dir: Path,
        candidate_names: Iterable[str] | None = None,
    ) -> str:
        provider = config.OCR_PROVIDER
        if provider in {"auto", "openai"} and config.OPENAI_MODEL and self._openai_available():
            try:
                return self._extract_with_openai(crop_path, screenshot_type, candidate_names)
            except Exception as exc:
                if provider == "openai":
                    raise OCRFailure(f"OpenAI OCR failed: {exc}") from exc
        if provider in {"auto", "easyocr"} and self._easyocr_available():
            try:
                return self._extract_with_easyocr(crop_path)
            except Exception as exc:
                if provider == "easyocr":
                    raise OCRFailure(f"EasyOCR failed: {exc}") from exc
        if provider in {"auto", "tesseract"} and shutil.which("tesseract"):
            return self._extract_with_tesseract(crop_path, work_dir)
        raise OCRFailure("no OCR backend is available. Set OPENAI_API_KEY, install EasyOCR, or install/configure tesseract.")

    def _openai_available(self) -> bool:
        if not config.OPENAI_MODEL or not config.OPENAI_API_KEY:
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return bool(config.OPENAI_MODEL and config.OPENAI_MODEL.strip())

    def _easyocr_available(self) -> bool:
        try:
            import easyocr  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            return False
        return True

    def _extract_with_openai(
        self,
        crop_path: Path,
        screenshot_type: str,
        candidate_names: Iterable[str] | None = None,
    ) -> str:
        from openai import OpenAI

        client = OpenAI()
        encoded = base64.b64encode(crop_path.read_bytes()).decode("ascii")
        allowed_names = ", ".join(candidate_names or self.matcher.names)
        prompt = (
            "Extract Uma Musume score rows from this cropped score list screenshot. "
            f"This is the {screenshot_type} screenshot of a two-screenshot list. "
            "Return strict JSON only: an array of objects with keys name and score. "
            "Use integer scores with no commas. Include a boundary row if both name and score are readable. "
            "Use the closest official name from this allowed list when confident: "
            f"{allowed_names}"
        )
        image_url = f"data:image/png;base64,{encoded}"
        try:
            response = client.responses.create(
                model=config.OPENAI_MODEL,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": image_url},
                        ],
                    }
                ],
            )
            return response.output_text
        except AttributeError:
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
            )
            return response.choices[0].message.content or ""

    def _extract_with_tesseract(self, crop_path: Path, work_dir: Path) -> str:
        preprocessed = work_dir / f"{crop_path.stem}-tesseract.png"
        image = Image.open(crop_path).convert("RGB")
        gray = ImageOps.grayscale(image)
        gray = ImageEnhance.Contrast(gray).enhance(2.4)
        gray = gray.resize((gray.width * 2, gray.height * 2), Image.Resampling.LANCZOS)
        gray.save(preprocessed)
        output_base = work_dir / f"{crop_path.stem}-tesseract"
        completed = subprocess.run(
            ["tesseract", str(preprocessed), str(output_base), "-psm", "6"],
            check=False,
            capture_output=True,
            text=True,
        )
        output_path = output_base.with_suffix(".txt")
        if output_path.exists():
            return output_path.read_text(encoding="utf-8", errors="replace")
        return (completed.stdout or "") + "\n" + (completed.stderr or "")

    def _extract_with_easyocr(self, crop_path: Path) -> str:
        import easyocr
        import numpy as np
        import torch

        if not config.EASYOCR_GPU:
            torch.set_num_threads(config.EASYOCR_CPU_THREADS)

        if self._easyocr_reader is None:
            self._easyocr_reader = easyocr.Reader(["en"], gpu=config.EASYOCR_GPU, verbose=False)

        image = Image.open(crop_path).convert("RGB")
        detections = self._easyocr_reader.readtext(
            np.array(image),
            detail=1,
            paragraph=False,
            decoder="greedy",
            batch_size=1,
        )
        return self._easyocr_detections_to_lines(detections)

    def _easyocr_detections_to_lines(self, detections: list[tuple[object, str, float]]) -> str:
        items = []
        for box, text, confidence in detections:
            points = list(box)
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            items.append(
                {
                    "x": min(xs),
                    "y": sum(ys) / len(ys),
                    "text": str(text),
                    "confidence": float(confidence),
                }
            )
        items.sort(key=lambda item: item["y"])

        groups: list[dict[str, object]] = []
        for item in items:
            if not groups or abs(float(groups[-1]["y"]) - float(item["y"])) > 55:
                groups.append({"y": item["y"], "items": [item]})
                continue
            group_items = groups[-1]["items"]
            if not isinstance(group_items, list):
                continue
            group_items.append(item)
            groups[-1]["y"] = sum(float(group_item["y"]) for group_item in group_items) / len(group_items)

        lines = []
        for group in groups:
            group_items = group["items"]
            if not isinstance(group_items, list):
                continue
            parts = [str(item["text"]) for item in sorted(group_items, key=lambda item: float(item["x"]))]
            line = self._clean_easyocr_line(" ".join(parts))
            if re.search(r"\d[\d,\s.]{2,}\s*(?:pts?|p)?", line, flags=re.IGNORECASE):
                lines.append(line)
        return "\n".join(lines)

    def _clean_easyocr_line(self, line: str) -> str:
        line = re.sub(
            r"\b(?:l?leading|charge|mvps?|empress|otherworldly|front[- ]runner|score info|close)\b",
            " ",
            line,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\s+", " ", line).strip()

    def parse_rows(self, raw_text: str, *, candidate_names: Iterable[str] | None = None) -> list[OCRRow]:
        candidate_names = tuple(candidate_names) if candidate_names is not None else None
        json_rows = self._parse_json_rows(raw_text, candidate_names)
        if json_rows:
            return json_rows
        return self._parse_text_rows(raw_text, candidate_names)

    def _parse_json_rows(
        self,
        raw_text: str,
        candidate_names: Iterable[str] | None = None,
    ) -> list[OCRRow]:
        text = raw_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
            text = re.sub(r"```$", "", text).strip()
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            text = match.group(0)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []
        rows = []
        if not isinstance(payload, list):
            return []
        for item in payload:
            if not isinstance(item, dict):
                continue
            raw_name = str(item.get("name", "")).strip()
            score = self._coerce_score(item.get("score", ""))
            name_match = self.matcher.match(raw_name, candidates=candidate_names)
            if name_match and score:
                rows.append(OCRRow(name=name_match.name, score=score, raw_name=raw_name))
        return rows

    def _parse_text_rows(
        self,
        raw_text: str,
        candidate_names: Iterable[str] | None = None,
    ) -> list[OCRRow]:
        rows: list[OCRRow] = []
        for raw_line in raw_text.splitlines():
            line = raw_line.strip()
            if not line or self._is_noise_line(line):
                continue
            score_match = self._find_score_match(line)
            if not score_match:
                continue
            score = self._coerce_score(score_match.group(0))
            if not score:
                continue
            name_text = line[: score_match.start()]
            name_match = self.matcher.match_in_text(name_text, candidates=candidate_names)
            if not name_match:
                continue
            rows.append(OCRRow(name=name_match.name, score=score, raw_name=name_text.strip()))
        return rows

    def _find_score_match(self, line: str) -> re.Match[str] | None:
        point_markers = list(re.finditer(r"\b(?:pts?|p)\b", line, flags=re.IGNORECASE))
        score_text = line[: point_markers[-1].start()] if point_markers else line
        matches = list(SCORE_TOKEN_RE.finditer(score_text))
        return matches[-1] if matches else None

    def _is_noise_line(self, line: str) -> bool:
        normalized = line.lower()
        return any(piece in normalized for piece in ["score info", "leading the charge", "close", "rank"])

    def _coerce_score(self, value: object) -> int | None:
        digits = re.sub(r"\D", "", str(value))
        if len(digits) < 3:
            return None
        try:
            score = int(digits)
        except ValueError:
            return None
        return score if score > 0 else None

    def merge_rows(self, row_groups: Iterable[list[OCRRow]]) -> list[OCRRow]:
        merged: list[OCRRow] = []
        seen: set[str] = set()
        for rows in row_groups:
            for row in rows:
                key = row.name.lower()
                if key in seen:
                    continue
                merged.append(row)
                seen.add(key)
        return merged

    def format_rows(self, rows: Iterable[OCRRow]) -> str:
        lines = [f"{row.name} {row.score:,} pts" for row in rows]
        return "\n".join(lines) if lines else "No score rows were parsed."
