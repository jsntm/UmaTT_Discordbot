from __future__ import annotations

import csv
import json
import random
import string
from pathlib import Path
from typing import Iterable

from ttbot import config
from ttbot.constants import ALL_UMAS_COLUMNS, CURRENT_TEAM_COLUMNS, DEFAULT_OCR_SETTINGS, RECORDS_COLUMNS


def _read_csv(path: Path, columns: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({column: (row.get(column) or "") for column in columns})
        return rows


def _write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


class UserStore:
    def __init__(self, user_id: str) -> None:
        self.user_id = str(user_id)
        self.path = config.USERS_DIR / self.user_id
        self.path.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    @property
    def all_umas_path(self) -> Path:
        return self.path / "all_umas.csv"

    @property
    def current_team_path(self) -> Path:
        return self.path / "current_team.csv"

    @property
    def records_path(self) -> Path:
        return self.path / "records.csv"

    def _ensure_tables(self) -> None:
        if not self.all_umas_path.exists():
            _write_csv(self.all_umas_path, ALL_UMAS_COLUMNS, [])
        if not self.current_team_path.exists():
            _write_csv(self.current_team_path, CURRENT_TEAM_COLUMNS, [])
        if not self.records_path.exists():
            _write_csv(self.records_path, RECORDS_COLUMNS, [])

    def read_all_umas(self) -> list[dict[str, str]]:
        return _read_csv(self.all_umas_path, ALL_UMAS_COLUMNS)

    def write_all_umas(self, rows: Iterable[dict[str, object]]) -> None:
        _write_csv(self.all_umas_path, ALL_UMAS_COLUMNS, rows)

    def read_current_team(self) -> list[dict[str, str]]:
        return _read_csv(self.current_team_path, CURRENT_TEAM_COLUMNS)

    def write_current_team(self, rows: Iterable[dict[str, object]]) -> None:
        _write_csv(self.current_team_path, CURRENT_TEAM_COLUMNS, rows)

    def read_records(self) -> list[dict[str, str]]:
        return _read_csv(self.records_path, RECORDS_COLUMNS)

    def write_records(self, rows: Iterable[dict[str, object]]) -> None:
        _write_csv(self.records_path, RECORDS_COLUMNS, rows)

    def append_records(self, rows: Iterable[dict[str, object]]) -> None:
        current = self.read_records()
        current.extend({key: str(value) for key, value in row.items()} for row in rows)
        self.write_records(current)

    def generate_uma_id(self) -> str:
        existing = {row["uma_id"].lower() for row in self.read_all_umas()}
        alphabet = string.ascii_lowercase + string.digits
        while True:
            code = "".join(random.choice(alphabet) for _ in range(5))
            if code not in existing:
                return code

    def find_uma(self, uma_id: str) -> dict[str, str] | None:
        return next((row for row in self.read_all_umas() if row["uma_id"].lower() == uma_id.lower()), None)

    def count_records(self, uma_id: str) -> int:
        return sum(1 for row in self.read_records() if row["uma_id"].lower() == uma_id.lower())


def _read_all_settings() -> dict[str, dict[str, dict[str, list[int]]]]:
    if not config.OCR_SETTINGS_FILE.exists():
        return {}
    try:
        return json.loads(config.OCR_SETTINGS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_all_settings(settings: dict[str, object]) -> None:
    config.OCR_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.OCR_SETTINGS_FILE.write_text(json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")


def get_ocr_setting(user_id: str, screenshot_type: str) -> dict[str, list[int]]:
    settings = _read_all_settings()
    user_settings = settings.get(str(user_id), {})
    selected = user_settings.get(screenshot_type) or DEFAULT_OCR_SETTINGS[screenshot_type]
    return json.loads(json.dumps(selected))


def set_ocr_setting(user_id: str, screenshot_type: str, setting: dict[str, list[int]]) -> None:
    settings = _read_all_settings()
    settings.setdefault(str(user_id), {})[screenshot_type] = setting
    _write_all_settings(settings)
