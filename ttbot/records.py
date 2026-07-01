from __future__ import annotations

from dataclasses import dataclass

from ttbot.storage import UserStore
from ttbot.team import TeamError, format_uma_id
from ttbot.validation import normalize_uma_id, parse_positive_int, parse_record_datetime


@dataclass(frozen=True)
class RecordEditResult:
    index: int
    name: str
    outfit: str
    old_score: str
    new_score: str


@dataclass(frozen=True)
class RecordAddResult:
    index: int
    uma_id: str
    outfit: str
    name: str
    rating: str
    date_acquired: str
    track: str
    position: str
    strategy: str
    display_datetime: str
    score: str


def _parse_index(value: int | str, label: str = "record_index") -> int:
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise TeamError(f"{label} must be a positive integer") from exc
    if index <= 0:
        raise TeamError(f"{label} must be a positive integer")
    return index


def _all_by_id(store: UserStore) -> dict[str, dict[str, str]]:
    return {row["uma_id"].lower(): row for row in store.read_all_umas()}


def _current_by_id(store: UserStore) -> dict[str, dict[str, str]]:
    return {row["uma_id"].lower(): row for row in store.read_current_team()}


def _joined_indexed_row(store: UserStore, index: int, record: dict[str, str]) -> dict[str, str] | None:
    all_row = _all_by_id(store).get(record["uma_id"].lower())
    if not all_row:
        return None
    return {"index": str(index), **all_row, **record}


def build_indexed_records(store: UserStore) -> list[dict[str, str]]:
    rows = []
    for index, record in enumerate(store.read_records(), start=1):
        joined = _joined_indexed_row(store, index, record)
        if joined:
            rows.append(joined)
    return rows


def edit_record_score(store: UserStore, record_index: int, score: int) -> RecordEditResult:
    index = _parse_index(record_index)
    try:
        parsed_score = parse_positive_int(score, "score")
    except ValueError as exc:
        raise TeamError(str(exc)) from exc

    records = store.read_records()
    if index > len(records):
        raise TeamError(f"record_index {index} is out of bounds. There are only {len(records)} records.")
    record = records[index - 1]
    all_row = _all_by_id(store).get(record["uma_id"].lower())
    if not all_row:
        raise TeamError(f"record at index {index} references missing uma {format_uma_id(record['uma_id'])}.")
    old_score = record["score"]
    record["score"] = str(parsed_score)
    store.write_records(records)
    return RecordEditResult(index, all_row["name"], all_row["outfit"], old_score, str(parsed_score))


def add_manual_record(store: UserStore, raw_uma_id: str, raw_datetime: str, score: int) -> RecordAddResult:
    try:
        uma_id = normalize_uma_id(raw_uma_id)
        parsed_score = parse_positive_int(score, "score")
        stored_datetime, display_datetime = parse_record_datetime(raw_datetime)
    except ValueError as exc:
        raise TeamError(str(exc)) from exc

    all_row = _all_by_id(store).get(uma_id)
    if not all_row:
        raise TeamError(f"{format_uma_id(uma_id)} does not correspond to a valid uma.")
    current_row = _current_by_id(store).get(uma_id)
    if not current_row:
        raise TeamError(f"{format_uma_id(uma_id)} is valid, but is not in your current team.")

    index = len(store.read_records()) + 1
    store.append_records(
        [
            {
                "uma_id": uma_id,
                "time_of_screenshot": stored_datetime,
                "track": current_row["track"],
                "position": current_row["position"],
                "strategy": current_row["strategy"],
                "score": str(parsed_score),
            }
        ]
    )
    return RecordAddResult(
        index=index,
        uma_id=uma_id,
        outfit=all_row["outfit"],
        name=all_row["name"],
        rating=all_row["rating"],
        date_acquired=all_row["date_acquired"],
        track=current_row["track"],
        position=current_row["position"],
        strategy=current_row["strategy"],
        display_datetime=display_datetime,
        score=str(parsed_score),
    )


def preview_delete_records(store: UserStore, start_index: int, end_index: int | None = None) -> list[dict[str, str]]:
    start = _parse_index(start_index, "start_index")
    end = _parse_index(end_index if end_index is not None else start_index, "end_index")
    if end < start:
        raise TeamError("end_index must be greater than or equal to start_index.")
    records = store.read_records()
    if not records:
        raise TeamError("You do not have any records to delete.")
    if start > len(records) or end > len(records):
        raise TeamError(f"start_index and end_index must be between 1 and {len(records)}.")
    rows = []
    for index in range(start, end + 1):
        joined = _joined_indexed_row(store, index, records[index - 1])
        if joined:
            rows.append(joined)
    return rows


def delete_record_range(store: UserStore, start_index: int, end_index: int | None = None) -> int:
    start = _parse_index(start_index, "start_index")
    end = _parse_index(end_index if end_index is not None else start_index, "end_index")
    if end < start:
        raise TeamError("end_index must be greater than or equal to start_index.")
    records = store.read_records()
    if start > len(records) or end > len(records):
        raise TeamError(f"start_index and end_index must be between 1 and {len(records)}.")
    del records[start - 1 : end]
    store.write_records(records)
    return end - start + 1
