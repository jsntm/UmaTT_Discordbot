from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ttbot import config
from ttbot.names import NameMatcher
from ttbot.ocr import OCRFailure, OCRService


def score_rows(rows, desired) -> tuple[int, int]:
    desired_map = {row.name: row.score for row in desired}
    actual_map = {row.name: row.score for row in rows}
    names_ok = sum(1 for name in desired_map if name in actual_map)
    scores_ok = sum(1 for name, score in desired_map.items() if actual_map.get(name) == score)
    return names_ok, scores_ok


def process_pair(service: OCRService, folder: Path, provider: str):
    old_provider = config.OCR_PROVIDER
    config.OCR_PROVIDER = provider
    try:
        with tempfile.TemporaryDirectory(dir=config.TMP_DIR) as tmp_name:
            tmp = Path(tmp_name)
            top = service.process_image("fixture", image_path(folder, "top"), "top", tmp)
            bottom = service.process_image("fixture", image_path(folder, "bottom"), "bottom", tmp)
            return service.merge_rows([top.rows, bottom.rows])
    finally:
        config.OCR_PROVIDER = old_provider


def image_path(folder: Path, stem: str) -> Path:
    for suffix in (".jpg", ".jpeg", ".png"):
        path = folder / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No {stem}.jpg/.jpeg/.png found in {folder}")


def check_provider(service: OCRService, fixture_folders: list[Path], provider: str) -> None:
    print(f"\nChecking {provider} fallback on screenshot pairs")
    total_names = 0
    total_scores = 0
    total_rows = 0
    for folder in fixture_folders:
        desired = service.parse_rows((folder / "desired.txt").read_text(encoding="utf-8"))
        try:
            merged = process_pair(service, folder, provider)
        except OCRFailure as exc:
            print(f"  {folder.name}: {provider} failed: {exc}")
            continue
        names_ok, scores_ok = score_rows(merged, desired)
        total_names += names_ok
        total_scores += scores_ok
        total_rows += len(desired)
        print(f"  {folder.name}: names {names_ok}/{len(desired)}, exact scores {scores_ok}/{len(desired)}")
    if total_rows:
        print(f"  total: names {total_names}/{total_rows}, exact scores {total_scores}/{total_rows}")


def main() -> None:
    matcher = NameMatcher.from_reference_file(config.REFERENCE_NAMES_FILE)
    service = OCRService(matcher)
    fixture_root = config.ROOT_DIR / "test_screenshots"
    fixture_folders = sorted([folder for folder in fixture_root.iterdir() if folder.is_dir()], key=lambda path: path.name)

    print("Checking desired.txt parsing")
    for folder in fixture_folders:
        desired = (folder / "desired.txt").read_text(encoding="utf-8")
        rows = service.parse_rows(desired)
        status = "ok" if len(rows) == 15 else "check"
        print(f"  {folder.name}: {len(rows)} rows parsed from desired.txt [{status}]")

    if importlib.util.find_spec("easyocr"):
        check_provider(service, fixture_folders, "easyocr")
    else:
        print("\nNo EasyOCR package found; skipping EasyOCR fixture check.")

    if shutil.which("tesseract"):
        check_provider(service, fixture_folders, "tesseract")
    else:
        print("\nNo tesseract executable found; skipping Tesseract fixture check.")


if __name__ == "__main__":
    main()
