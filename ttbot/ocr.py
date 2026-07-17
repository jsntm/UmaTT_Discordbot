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
            f"OCR region top-left ({x1}, {y1}), bottom-right ({x2}, {y2})\n"
            f"Raw OCR output:\n```text\n{raw}\n```"
        )


class OCRService:
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
    ) -> OCRResult:
        image = Image.open(image_path).convert("RGB")
        if update_coords is not None:
            current_region = self._scaled_region(user_id, screenshot_type, image.size)
            x1, y1, x2, y2 = (
                current if updated is None else updated
                for current, updated in zip(current_region, update_coords)
            )
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > image.width or y2 > image.height:
                raise OCRFailure("coordinates must be inside the image and bottom-right must be below/right of top-left")
            setting = {"top_left": [x1, y1], "bottom_right": [x2, y2], "base_size": [image.width, image.height]}
            set_ocr_setting(user_id, screenshot_type, setting)

        region = self._scaled_region(user_id, screenshot_type, image.size)
        highlight_path = self._write_highlight(image, region, work_dir / f"{screenshot_type}-ocr-region.png")
        crop = image.crop(region)
        crop_path = work_dir / f"{screenshot_type}-crop.png"
        crop.save(crop_path)

        raw_text = self._extract_text(crop_path, screenshot_type, work_dir)
        rows = self.parse_rows(raw_text)
        result = OCRResult(screenshot_type=screenshot_type, rows=rows, raw_text=raw_text, region=region, highlight_path=highlight_path)
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

    def _write_highlight(self, image: Image.Image, region: tuple[int, int, int, int], path: Path) -> Path:
        overlay = image.convert("RGBA")
        blue = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(blue)
        draw.rectangle(region, fill=(50, 130, 255, 48), outline=(30, 105, 255, 255), width=8)
        highlighted = Image.alpha_composite(overlay, blue).convert("RGB")
        highlighted.save(path)
        return path

    def _extract_text(self, crop_path: Path, screenshot_type: str, work_dir: Path) -> str:
        provider = config.OCR_PROVIDER
        if provider in {"auto", "openai"} and config.OPENAI_MODEL and self._openai_available():
            try:
                return self._extract_with_openai(crop_path, screenshot_type)
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

    def _extract_with_openai(self, crop_path: Path, screenshot_type: str) -> str:
        from openai import OpenAI

        client = OpenAI()
        encoded = base64.b64encode(crop_path.read_bytes()).decode("ascii")
        allowed_names = ", ".join(self.matcher.names)
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

    def parse_rows(self, raw_text: str) -> list[OCRRow]:
        json_rows = self._parse_json_rows(raw_text)
        if json_rows:
            return json_rows
        return self._parse_text_rows(raw_text)

    def _parse_json_rows(self, raw_text: str) -> list[OCRRow]:
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
            name_match = self.matcher.match(raw_name)
            if name_match and score:
                rows.append(OCRRow(name=name_match.name, score=score, raw_name=raw_name))
        return rows

    def _parse_text_rows(self, raw_text: str) -> list[OCRRow]:
        rows: list[OCRRow] = []
        for raw_line in raw_text.splitlines():
            line = raw_line.strip()
            if not line or self._is_noise_line(line):
                continue
            score_match = re.search(r"(\d[\d,\s.]{2,})\s*(?:pts?|p)?", line, flags=re.IGNORECASE)
            if not score_match:
                continue
            score = self._coerce_score(score_match.group(1))
            if not score:
                continue
            name_text = line[: score_match.start()]
            name_match = self.matcher.match_in_text(name_text)
            if not name_match:
                continue
            rows.append(OCRRow(name=name_match.name, score=score, raw_name=name_text.strip()))
        return rows

    def _is_noise_line(self, line: str) -> bool:
        normalized = line.lower()
        return any(piece in normalized for piece in ["score info", "leading the charge", "close", "rank"])

    def _coerce_score(self, value: object) -> int | None:
        digits = re.sub(r"\D", "", str(value))
        if len(digits) < 4:
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
