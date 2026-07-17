from __future__ import annotations

from datetime import datetime, timezone
import re

from ttbot.constants import POSITIONS, STRATEGIES, TRACKS


UMA_ID_RE = re.compile(r"^[a-z0-9]{5}$", re.IGNORECASE)


def parse_track(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in TRACKS:
        raise ValueError(f"track must be one of: {', '.join(TRACKS)}")
    return normalized


def parse_strategy(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in STRATEGIES:
        raise ValueError(f"strategy must be one of: {', '.join(STRATEGIES)}")
    return normalized


def parse_position(value: int | str) -> int:
    try:
        position = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("position must be 1, 2, or 3") from exc
    if position not in POSITIONS:
        raise ValueError("position must be 1, 2, or 3")
    return position


def parse_positive_int(value: int | str, label: str) -> int:
    try:
        number = int(str(value).replace(",", "").strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if number <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return number


def parse_date(value: str) -> str:
    raw = value.strip()
    if raw.lower() == "today":
        return datetime.now(timezone.utc).strftime("%m/%d/%Y")
    raw = raw.replace("-", "/")
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            parsed = datetime.strptime(raw, fmt)
            return parsed.strftime("%m/%d/%Y")
        except ValueError:
            pass
    raise ValueError("date_acquired must look like MM/DD/YYYY or MM-DD-YYYY")


def parse_record_datetime(value: str) -> tuple[str, str]:
    raw = value.strip().replace("-", "/")
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            parsed = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            return parsed.isoformat(timespec="seconds"), parsed.strftime("%m/%d/%Y 00:00:00")
        except ValueError:
            pass
    raise ValueError("datetime must look like MM/DD/YYYY or MM-DD-YYYY")


def normalize_uma_id(value: str) -> str:
    normalized = value.strip().lower()
    if not UMA_ID_RE.fullmatch(normalized):
        raise ValueError("uma_id must be a 5 digit alphanumeric code")
    return normalized
