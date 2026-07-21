from __future__ import annotations

import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image

from ttbot import config


REQUIRED_COLUMNS = {"gametora_id", "outfit_id", "uma_name", "outfit_name"}


@dataclass(frozen=True)
class ReferenceRow:
    gametora_id: str
    outfit_id: str
    uma_name: str
    outfit_name: str

    @property
    def thumbnail_path(self) -> Path:
        return config.UMA_THUMBS_DIR / self.gametora_id / f"{self.gametora_id}{self.outfit_id}.png"

    @property
    def thumbnail_url(self) -> str:
        filename = f"chara_stand_{self.gametora_id}_{self.gametora_id}{self.outfit_id}.png"
        return f"https://gametora.com/images/umamusume/characters/thumb/{filename}"


@dataclass(frozen=True)
class ThumbnailDownloadReport:
    downloaded: int
    skipped: int
    failures: tuple[str, ...]


def read_reference_rows(path: Path = config.REFERENCE_CSV_FILE) -> list[ReferenceRow]:
    if not path.exists():
        raise RuntimeError(f"Reference CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError("Reference CSV is missing columns: " + ", ".join(sorted(missing)))
        rows = []
        for line_number, raw in enumerate(reader, start=2):
            values = {key: (raw.get(key) or "").strip() for key in REQUIRED_COLUMNS}
            if not all(values.values()):
                raise RuntimeError(f"Reference CSV line {line_number} contains an empty required value")
            if not values["gametora_id"].isdigit() or not values["outfit_id"].isdigit():
                raise RuntimeError(f"Reference CSV line {line_number} has a nonnumeric ID")
            rows.append(ReferenceRow(**values))
    if not rows:
        raise RuntimeError("Reference CSV does not contain any umas")
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def generate_reference_files(rows: list[ReferenceRow] | None = None) -> list[ReferenceRow]:
    rows = rows or read_reference_rows()
    ids_by_name: dict[str, str] = {}
    outfits_by_id: dict[str, list[tuple[int, str]]] = {}
    seen_outfits: set[tuple[str, str]] = set()
    for row in rows:
        existing_id = ids_by_name.setdefault(row.uma_name, row.gametora_id)
        if existing_id != row.gametora_id:
            raise RuntimeError(f"{row.uma_name} has conflicting gametora IDs in the reference CSV")
        outfit_key = row.gametora_id, row.outfit_id
        if outfit_key in seen_outfits:
            raise RuntimeError(f"Duplicate outfit ID {row.gametora_id}/{row.outfit_id} in the reference CSV")
        seen_outfits.add(outfit_key)
        outfits_by_id.setdefault(row.gametora_id, []).append((int(row.outfit_id), row.outfit_name))

    uma_names = {name: ids_by_name[name] for name in sorted(ids_by_name, key=str.casefold)}
    outfit_names = {
        gametora_id: [name for _, name in sorted(outfits, key=lambda item: item[0])]
        for gametora_id, outfits in sorted(outfits_by_id.items(), key=lambda item: int(item[0]))
    }
    _write_json(config.UMA_NAMES_FILE, uma_names)
    _write_json(config.OUTFIT_NAMES_FILE, outfit_names)
    return rows


def _download_thumbnail(row: ReferenceRow, timeout: float) -> None:
    request = Request(row.thumbnail_url, headers={"User-Agent": "TTDiscordBot/1.0"})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read()
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
    row.thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = row.thumbnail_path.with_suffix(".png.tmp")
    temporary.write_bytes(payload)
    temporary.replace(row.thumbnail_path)


def download_missing_thumbnails(
    rows: list[ReferenceRow] | None = None,
    *,
    workers: int = 12,
    timeout: float = 20.0,
) -> ThumbnailDownloadReport:
    rows = rows or read_reference_rows()
    missing = [row for row in rows if not row.thumbnail_path.exists()]
    skipped = len(rows) - len(missing)
    failures = []
    downloaded = 0
    if not missing:
        return ThumbnailDownloadReport(downloaded=0, skipped=skipped, failures=())

    with ThreadPoolExecutor(max_workers=min(workers, len(missing))) as executor:
        futures = {executor.submit(_download_thumbnail, row, timeout): row for row in missing}
        for future in as_completed(futures):
            row = futures[future]
            try:
                future.result()
                downloaded += 1
            except Exception as exc:
                failures.append(f"{row.gametora_id}{row.outfit_id} ({row.uma_name}, {row.outfit_name}): {exc}")
    return ThumbnailDownloadReport(downloaded=downloaded, skipped=skipped, failures=tuple(sorted(failures)))
