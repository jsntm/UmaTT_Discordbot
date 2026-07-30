from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ttbot.constants import POSITION_LABELS, POSITIONS, TRACKS
from ttbot.names import NameMatcher
from ttbot.ocr import OCRRow
from ttbot.storage import UserStore
from ttbot.validation import parse_date, parse_position, parse_positive_int, parse_strategy, parse_track


class TeamError(Exception):
    pass


@dataclass(frozen=True)
class JoinedUma:
    uma_id: str
    outfit: str
    name: str
    rating: str
    date_acquired: str
    track: str = ""
    position: str = ""
    strategy: str = ""


@dataclass(frozen=True)
class ReplaceResult:
    added: JoinedUma
    removed: JoinedUma | None
    removed_record_count: int = 0


@dataclass(frozen=True)
class SwapResult:
    messages: list[str]


@dataclass(frozen=True)
class AddedRecord:
    index: int
    entry: JoinedUma
    score: int


@dataclass(frozen=True)
class OCRAddResult:
    added: list[AddedRecord]
    warnings: list[str]
    score_outliers: list[AddedRecord] = field(default_factory=list)


def format_uma_id(uma_id: str) -> str:
    return f"`{uma_id}`"


def _join(all_row: dict[str, str], current_row: dict[str, str] | None = None) -> JoinedUma:
    current_row = current_row or {}
    return JoinedUma(
        uma_id=all_row["uma_id"],
        outfit=all_row["outfit"],
        name=all_row["name"],
        rating=all_row["rating"],
        date_acquired=all_row["date_acquired"],
        track=current_row.get("track", ""),
        position=current_row.get("position", ""),
        strategy=current_row.get("strategy", ""),
    )


def _maps(store: UserStore) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    all_by_id = {row["uma_id"].lower(): row for row in store.read_all_umas()}
    current_by_id = {row["uma_id"].lower(): row for row in store.read_current_team()}
    return all_by_id, current_by_id


def _format_position(position: str | int) -> str:
    try:
        return POSITION_LABELS[int(position)]
    except (ValueError, KeyError):
        return str(position)


def format_team_entry(entry: JoinedUma) -> str:
    base = f"{format_uma_id(entry.uma_id)} {entry.name} ({entry.outfit}) rating {entry.rating} acquired {entry.date_acquired}"
    if entry.track:
        return f"{base} running as {entry.strategy} for {entry.track} as {_format_position(entry.position)}"
    return base


def format_added_uma(entry: JoinedUma) -> str:
    return (
        f"{format_uma_id(entry.uma_id)} {entry.name} ({entry.outfit}) rating {entry.rating} acquired {entry.date_acquired} "
        f"added as {entry.strategy} for {entry.track} as {_format_position(entry.position)}"
    )


def format_removed_uma(entry: JoinedUma, record_count: int) -> str:
    removed = format_team_entry(entry)
    if record_count == 0:
        return f"{removed} had no records, it is now deleted."
    return f"{removed} with {record_count} records is no longer in your team"


def _validate_name(matcher: NameMatcher, raw_name: str) -> str:
    match = matcher.match(raw_name)
    if not match:
        raise TeamError(f"name `{raw_name}` did not match a known uma name closely enough.")
    return match.name


def _validate_outfit(matcher: NameMatcher, official_name: str, raw_outfit: str) -> str:
    match = matcher.match_outfit(official_name, raw_outfit)
    if not match:
        gametora_id = matcher.gametora_id(official_name)
        outfits = matcher.outfits_by_id.get(gametora_id, []) if gametora_id else []
        choices = ""
        if outfits:
            choices = f" Possible outfits for {official_name}: " + ", ".join(f"`{outfit}`" for outfit in outfits) + "."
        raise TeamError(
            f"outfit `{raw_outfit}` did not match a known outfit for {official_name} closely enough."
            f"{choices}"
        )
    return match.outfit


def _validate_common(track: str, position: int, strategy: str, rating: int | None = None, date_acquired: str | None = None) -> tuple[str, int, str, int | None, str | None]:
    try:
        parsed_track = parse_track(track)
        parsed_position = parse_position(position)
        parsed_strategy = parse_strategy(strategy)
        parsed_rating = parse_positive_int(rating, "rating") if rating is not None else None
        parsed_date = parse_date(date_acquired) if date_acquired is not None else None
    except ValueError as exc:
        raise TeamError(str(exc)) from exc
    return parsed_track, parsed_position, parsed_strategy, parsed_rating, parsed_date


def _current_joined(store: UserStore) -> list[JoinedUma]:
    all_by_id, current_by_id = _maps(store)
    joined = []
    for current in current_by_id.values():
        all_row = all_by_id.get(current["uma_id"].lower())
        if all_row:
            joined.append(_join(all_row, current))
    return joined


def _check_name_available(store: UserStore, official_name: str, *, ignore_uma_id: str | None = None) -> None:
    all_by_id, current_by_id = _maps(store)
    for uma_id, current in current_by_id.items():
        if ignore_uma_id and uma_id == ignore_uma_id.lower():
            continue
        all_row = all_by_id.get(uma_id)
        if all_row and all_row["name"].lower() == official_name.lower():
            raise TeamError(f"{official_name} is already in your current team as {current['track']} {_format_position(current['position'])}.")


def _check_slot_available(
    current_rows: list[dict[str, str]],
    track: str,
    position: int,
    *,
    ignore_uma_id: str | None = None,
) -> None:
    same_track = [row for row in current_rows if row["track"] == track and row["uma_id"].lower() != (ignore_uma_id or "").lower()]
    if len(same_track) >= 3:
        raise TeamError(f"{track} already has three umas.")
    for row in same_track:
        if int(row["position"]) == position:
            raise TeamError(f"{track} already has a runner in position {_format_position(position)}.")


def replace_team_slot(
    store: UserStore,
    matcher: NameMatcher,
    track: str,
    position: int,
    strategy: str,
    outfit: str,
    name: str,
    rating: int,
    date_acquired: str,
) -> ReplaceResult:
    parsed_track, parsed_position, parsed_strategy, parsed_rating, parsed_date = _validate_common(track, position, strategy, rating, date_acquired)
    official_name = _validate_name(matcher, name)
    official_outfit = _validate_outfit(matcher, official_name, outfit)
    current_rows = store.read_current_team()
    existing_slot = next((row for row in current_rows if row["track"] == parsed_track and int(row["position"]) == parsed_position), None)
    ignore = existing_slot["uma_id"] if existing_slot else None
    _check_name_available(store, official_name, ignore_uma_id=ignore)

    all_rows = store.read_all_umas()
    uma_id = store.generate_uma_id()
    new_all = {
        "uma_id": uma_id,
        "outfit": official_outfit,
        "name": official_name,
        "rating": str(parsed_rating),
        "date_acquired": parsed_date,
    }
    all_rows.append(new_all)

    removed_joined = None
    removed_record_count = 0
    if existing_slot:
        removed_all = next((row for row in all_rows if row["uma_id"].lower() == existing_slot["uma_id"].lower()), None)
        if removed_all:
            removed_joined = _join(removed_all, existing_slot)
            removed_record_count = store.count_records(existing_slot["uma_id"])
        current_rows = [row for row in current_rows if row["uma_id"].lower() != existing_slot["uma_id"].lower()]
        if removed_record_count == 0:
            all_rows = [row for row in all_rows if row["uma_id"].lower() != existing_slot["uma_id"].lower()]
    else:
        _check_slot_available(current_rows, parsed_track, parsed_position)

    new_current = {"uma_id": uma_id, "track": parsed_track, "position": str(parsed_position), "strategy": parsed_strategy}
    current_rows.append(new_current)
    store.write_all_umas(all_rows)
    store.write_current_team(current_rows)
    return ReplaceResult(_join(new_all, new_current), removed_joined, removed_record_count)


def swap_team_members(store: UserStore, uma_1_id: str, uma_2_id: str) -> SwapResult:
    all_by_id, current_by_id = _maps(store)
    if uma_1_id not in all_by_id:
        raise TeamError(f"{format_uma_id(uma_1_id)} does not correspond to a valid uma.")
    if uma_2_id not in all_by_id:
        raise TeamError(f"{format_uma_id(uma_2_id)} does not correspond to a valid uma.")
    in_1 = current_by_id.get(uma_1_id)
    in_2 = current_by_id.get(uma_2_id)
    if not in_1 and not in_2:
        raise TeamError("Both umas are outside your current team, so there is no team slot to swap.")

    current_rows = store.read_current_team()
    all_rows = store.read_all_umas()
    messages: list[str] = []
    if in_1 and in_2:
        old_1 = dict(in_1)
        old_2 = dict(in_2)
        for row in current_rows:
            if row["uma_id"].lower() == uma_1_id:
                row["track"], row["position"] = old_2["track"], old_2["position"]
            if row["uma_id"].lower() == uma_2_id:
                row["track"], row["position"] = old_1["track"], old_1["position"]
        store.write_current_team(current_rows)
        uma_1 = _join(all_by_id[uma_1_id], {**old_1, "track": old_2["track"], "position": old_2["position"]})
        uma_2 = _join(all_by_id[uma_2_id], {**old_2, "track": old_1["track"], "position": old_1["position"]})
        messages.append(f"swapped {format_uma_id(uma_1.uma_id)} {uma_1.strategy} {uma_1.name} ({uma_1.outfit}) to {old_2['track']} as {_format_position(old_2['position'])}")
        messages.append(f"swapped {format_uma_id(uma_2.uma_id)} {uma_2.strategy} {uma_2.name} ({uma_2.outfit}) to {old_1['track']} as {_format_position(old_1['position'])}")
        return SwapResult(messages)

    entering_id, removed_id = (uma_1_id, uma_2_id) if in_2 else (uma_2_id, uma_1_id)
    removed_current = in_2 if in_2 else in_1
    assert removed_current is not None
    entering_all = all_by_id[entering_id]
    _check_name_available(store, entering_all["name"], ignore_uma_id=removed_id)
    current_rows = [row for row in current_rows if row["uma_id"].lower() != removed_id]
    new_current = {
        "uma_id": entering_id,
        "track": removed_current["track"],
        "position": removed_current["position"],
        "strategy": removed_current["strategy"],
    }
    current_rows.append(new_current)
    removed_all = all_by_id[removed_id]
    removed_joined = _join(removed_all, removed_current)
    removed_record_count = store.count_records(removed_id)
    if removed_record_count == 0:
        all_rows = [row for row in all_rows if row["uma_id"].lower() != removed_id]
    store.write_all_umas(all_rows)
    store.write_current_team(current_rows)

    entering = _join(entering_all, new_current)
    messages.append(f"added back {format_uma_id(entering.uma_id)} {entering.outfit} {entering.name} as {entering.strategy} for {entering.track} as {_format_position(entering.position)}")
    messages.append(format_removed_uma(removed_joined, removed_record_count))
    return SwapResult(messages)


def update_uma(
    store: UserStore,
    matcher: NameMatcher,
    uma_id: str,
    track: str | None,
    position: int | None,
    strategy: str | None,
    outfit: str | None,
    name: str | None,
    rating: int | None,
    date_acquired: str | None,
) -> tuple[JoinedUma, JoinedUma]:
    all_rows = store.read_all_umas()
    current_rows = store.read_current_team()
    all_row = next((row for row in all_rows if row["uma_id"].lower() == uma_id), None)
    if not all_row:
        raise TeamError(f"{format_uma_id(uma_id)} does not correspond to a valid uma.")
    current_row = next((row for row in current_rows if row["uma_id"].lower() == uma_id), None)
    old_entry = _join(dict(all_row), dict(current_row) if current_row else None)

    target_name = all_row["name"]
    if name is not None:
        official_name = _validate_name(matcher, name)
        if current_row:
            _check_name_available(store, official_name, ignore_uma_id=uma_id)
        all_row["name"] = official_name
        target_name = official_name
    if outfit is not None or name is not None:
        all_row["outfit"] = _validate_outfit(matcher, target_name, outfit if outfit is not None else all_row["outfit"])
    if rating is not None:
        try:
            all_row["rating"] = str(parse_positive_int(rating, "rating"))
        except ValueError as exc:
            raise TeamError(str(exc)) from exc
    if date_acquired is not None:
        try:
            all_row["date_acquired"] = parse_date(date_acquired)
        except ValueError as exc:
            raise TeamError(str(exc)) from exc

    if any(value is not None for value in [track, position, strategy]):
        if not current_row:
            raise TeamError("track, position, and strategy can only be edited for umas in the current team.")
        try:
            new_track = parse_track(track) if track is not None else current_row["track"]
            new_position = parse_position(position) if position is not None else int(current_row["position"])
            new_strategy = parse_strategy(strategy) if strategy is not None else current_row["strategy"]
        except ValueError as exc:
            raise TeamError(str(exc)) from exc
        _check_slot_available(current_rows, new_track, new_position, ignore_uma_id=uma_id)
        current_row["track"] = new_track
        current_row["position"] = str(new_position)
        current_row["strategy"] = new_strategy

    store.write_all_umas(all_rows)
    store.write_current_team(current_rows)
    new_entry = _join(all_row, current_row)
    return old_entry, new_entry


def ensure_full_team(store: UserStore) -> None:
    current = store.read_current_team()
    occupied = {(row["track"], int(row["position"])) for row in current if row.get("position", "").isdigit()}
    missing = [(track, position) for track in TRACKS for position in POSITIONS if (track, position) not in occupied]
    if missing:
        details = format_current_team(store, show_missing=True)
        raise TeamError(f"Your current team is not fully set. Missing slots are marked below.\n{details}")


def format_current_team(store: UserStore, *, show_missing: bool = False) -> str:
    all_by_id, current_by_id = _maps(store)
    if not current_by_id and not show_missing:
        raise TeamError("That user does not have a current team associated with them.")
    by_slot = {(row["track"], int(row["position"])): row for row in current_by_id.values() if row.get("position", "").isdigit()}
    table_rows: list[dict[str, str]] = []
    for track in TRACKS:
        for position in POSITIONS:
            row = by_slot.get((track, position))
            label = f"{track} {position}"
            if not row:
                table_rows.append({"track": label, "name": "MISSING" if show_missing else "empty", "id": "", "rating": "", "date": ""})
                continue
            uma = all_by_id.get(row["uma_id"].lower())
            if not uma:
                table_rows.append({"track": label, "name": "missing uma", "id": row["uma_id"], "rating": "", "date": ""})
                continue
            date_text = _display_date(uma["date_acquired"])
            table_rows.append(
                {
                    "track": label,
                    "name": f"{uma['name']} ({uma['outfit']})",
                    "id": uma["uma_id"],
                    "rating": uma["rating"],
                    "date": date_text,
                }
            )
    return _format_current_team_table(table_rows)


def _format_current_team_table(rows: list[dict[str, str]]) -> str:
    columns = [("track", "track"), ("name", "name (outfit)"), ("id", "id"), ("rating", "rating"), ("date", "date")]
    widths = {
        key: max(len(label), *(len(str(row[key])) for row in rows))
        for key, label in columns
    }

    def render(row: dict[str, str] | None = None) -> str:
        values = {key: label if row is None else str(row[key]) for key, label in columns}
        return " | ".join(values[key].ljust(widths[key]) for key, _ in columns)

    separator = "-+-".join("-" * widths[key] for key, _ in columns)
    return "```text\n" + "\n".join([render(), separator, *(render(row) for row in rows)]) + "\n```"


def _display_date(value: str) -> str:
    try:
        parsed = datetime.strptime(value, "%m/%d/%Y")
        return parsed.strftime("%b %-d %Y")
    except ValueError:
        try:
            parsed = datetime.strptime(value, "%m/%d/%Y")
            return parsed.strftime("%b %#d %Y")
        except ValueError:
            return value


def get_team_name_map(store: UserStore) -> dict[str, JoinedUma]:
    return {entry.name.lower(): entry for entry in _current_joined(store)}


def get_current_team_entries(store: UserStore) -> list[JoinedUma]:
    track_order = {track: index for index, track in enumerate(TRACKS)}
    return sorted(
        _current_joined(store),
        key=lambda entry: (track_order.get(entry.track, 99), int(entry.position) if entry.position.isdigit() else 99),
    )


def ocr_bijection_issues(store: UserStore, rows: list[OCRRow]) -> list[str]:
    name_map = get_team_name_map(store)
    counts = Counter(row.name.lower() for row in rows)
    recognized = set(counts)
    current_names = set(name_map)
    issues = []
    extras = sorted(recognized - current_names)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    missing = sorted(current_names - recognized)
    recognized_names = {row.name.lower(): row.name for row in rows}
    if extras:
        issues.append("names not in the current team: " + ", ".join(recognized_names[name] for name in extras))
    if duplicates:
        issues.append("duplicate names: " + ", ".join(name_map[name].name if name in name_map else recognized_names[name] for name in duplicates))
    if missing:
        issues.append("missing names: " + ", ".join(name_map[name].name for name in missing))
    return issues


def _find_ocr_score_outliers(added: list[AddedRecord]) -> list[AddedRecord]:
    if len(added) < 8:
        return []
    # A clear digit-length consensus and a multi-fold gap target OCR digit errors without policing variance.
    digit_counts = Counter(len(str(record.score)) for record in added)
    typical_digits, typical_count = digit_counts.most_common(1)[0]
    if typical_count < len(added) * 0.75:
        return []

    median_score = float(statistics.median(record.score for record in added))
    return [
        record
        for record in added
        if len(str(record.score)) != typical_digits
        and (
            record.score >= median_score * 2.5
            or record.score * 4 <= median_score
        )
    ]


def add_records_from_ocr(store: UserStore, rows: list[OCRRow], timestamp: datetime) -> OCRAddResult:
    if not rows:
        raise TeamError("OCR did not return any score rows.")
    name_map = get_team_name_map(store)
    current_names = set(name_map)
    when = timestamp.astimezone(timezone.utc).isoformat(timespec="seconds")
    start_index = len(store.read_records()) + 1
    records = []
    added = []
    warnings = []
    added_names: set[str] = set()
    skipped_extra: list[str] = []
    skipped_duplicate: list[str] = []
    for row in rows:
        key = row.name.lower()
        if key not in current_names:
            skipped_extra.append(row.name)
            continue
        if key in added_names:
            skipped_duplicate.append(row.name)
            continue
        entry = name_map[key]
        records.append(
            {
                "uma_id": entry.uma_id,
                "time_of_screenshot": when,
                "track": entry.track,
                "position": entry.position,
                "strategy": entry.strategy,
                "score": str(row.score),
            }
        )
        added.append(AddedRecord(start_index + len(added), entry, row.score))
        added_names.add(key)
    missing = sorted(current_names - added_names)
    if skipped_extra:
        warnings.append("Skipped OCR names not in your current team: " + ", ".join(sorted(set(skipped_extra))))
    if skipped_duplicate:
        warnings.append("Skipped duplicate OCR names: " + ", ".join(sorted(set(skipped_duplicate))))
    if missing:
        warnings.append("Missing from OCR: " + ", ".join(name_map[name].name for name in missing))
    if not records:
        raise TeamError("OCR did not produce any records that match your current team.")
    score_outliers = _find_ocr_score_outliers(added)
    store.append_records(records)
    return OCRAddResult(added, warnings, score_outliers)
