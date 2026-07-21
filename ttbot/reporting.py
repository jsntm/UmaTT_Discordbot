from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import math
import random
import statistics
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image
from ttbot import config  # Sets MPLCONFIGDIR before matplotlib imports.

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch

from ttbot.constants import ALL_UMAS_COLUMNS, RECORDS_EXPORT_COLUMNS, STRATEGIES, TRACKS
from ttbot.names import NameMatcher, normalize_name
from ttbot.storage import UserStore
from ttbot.team import TeamError
from ttbot.validation import UMA_ID_RE, parse_date


@dataclass(frozen=True)
class UmaSelector:
    source_kind: str
    source_value: str
    uma_ids: tuple[str, ...]
    track: str | None = None
    ace_status: str | None = None
    strategy: str | None = None

    @property
    def source(self) -> str:
        return self.source_value

    @property
    def canonical_key(self) -> tuple[str, str, str | None, str | None, str | None]:
        return self.source_kind, self.source_value, self.track, self.ace_status, self.strategy


@dataclass
class BoxplotSeries:
    label: str
    scores: list[int]
    track: str = ""
    position: str = "99"
    is_current: bool = False
    thumbnail_path: Path | None = None


def _all_by_id(store: UserStore) -> dict[str, dict[str, str]]:
    return {row["uma_id"].lower(): row for row in store.read_all_umas()}


def _current_by_id(store: UserStore) -> dict[str, dict[str, str]]:
    return {row["uma_id"].lower(): row for row in store.read_current_team()}


def _joined_record(store: UserStore, record: dict[str, str]) -> dict[str, str] | None:
    all_row = _all_by_id(store).get(record["uma_id"].lower())
    if not all_row:
        return None
    return {**all_row, **record}


def build_records_export(
    store: UserStore,
    filter_value: str = "all",
    matcher: NameMatcher | None = None,
) -> list[dict[str, str]]:
    records = store.read_records()
    if not records:
        raise TeamError("That user does not have records associated with them.")
    selectors = parse_uma_selectors(store, filter_value, matcher)
    current_configs = set(_current_config_keys(store))
    selected = []
    for index, record in enumerate(records, start=1):
        key = _record_config_key(record)
        if any(_selector_matches_config(selector, key, current_configs) for selector in selectors):
            selected.append((index, record))
    rows = [{"index": str(index), **joined} for index, record in selected if (joined := _joined_record(store, record))]
    if not rows:
        raise TeamError("No records matched that filter.")
    return rows


def format_summary_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "```text\nname | track | avg | median | min | max | stdev\n```"
    columns = [
        ("name", "name"),
        ("track", "track"),
        ("average score", "avg"),
        ("median score", "median"),
        ("min score", "min"),
        ("max score", "max"),
        ("stdev score", "stdev"),
    ]
    widths = {}
    for key, label in columns:
        widths[key] = max(len(label), *(len(str(row[key])) for row in rows))

    def render(row: dict[str, str] | None = None) -> str:
        values = {key: label if row is None else str(row[key]) for key, label in columns}
        return " | ".join(values[key].ljust(widths[key]) for key, _ in columns)

    separator = "-+-".join("-" * widths[key] for key, _ in columns)
    lines = [render(), separator]
    lines.extend(render(row) for row in rows)
    return "```text\n" + "\n".join(lines) + "\n```"


def write_records_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _write_csv(path, RECORDS_EXPORT_COLUMNS, rows)


def write_all_umas_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _write_csv(path, ALL_UMAS_COLUMNS, rows)


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _matching_current_records(store: UserStore) -> list[tuple[dict[str, str], dict[str, str], dict[str, str]]]:
    all_rows = _all_by_id(store)
    current_rows = _current_by_id(store)
    matches = []
    for record in store.read_records():
        uma_id = record["uma_id"].lower()
        current = current_rows.get(uma_id)
        all_row = all_rows.get(uma_id)
        if not current or not all_row:
            continue
        record_ace_status = _ace_status(record["position"])
        current_ace_status = _ace_status(current["position"])
        if (
            record["track"] == current["track"]
            and record_ace_status
            and record_ace_status == current_ace_status
            and record["strategy"] == current["strategy"]
        ):
            matches.append((all_row, current, record))
    return matches


def _ace_status(position: str | int) -> str:
    value = str(position).strip()
    if value == "1":
        return "ace"
    if value in {"2", "3"}:
        return "nonace"
    return ""


ConfigKey = tuple[str, str, str, str]


def _record_config_key(row: dict[str, str]) -> ConfigKey:
    return (
        row.get("uma_id", "").strip().lower(),
        row.get("track", "").strip().lower(),
        _ace_status(row.get("position", "")),
        row.get("strategy", "").strip().lower(),
    )


def _current_config_keys(store: UserStore) -> list[ConfigKey]:
    keys = []
    for row in store.read_current_team():
        key = _record_config_key(row)
        if key[0] and key[1] in TRACKS and key[2] and key[3] in STRATEGIES:
            keys.append(key)
    return keys


def _default_matcher() -> NameMatcher:
    return NameMatcher.from_reference_files(
        config.UMA_NAMES_FILE,
        config.UMA_NAME_ALIASES_FILE,
        config.OUTFIT_NAMES_FILE,
        config.OUTFIT_NAME_ALIASES_FILE,
        config.UMA_THUMBS_DIR,
    )


def _uses_space_separated_qualifiers(source: str, matcher: NameMatcher, all_rows: dict[str, dict[str, str]]) -> bool:
    normalized = normalize_name(source)
    if normalized in matcher.by_normalized or normalized in matcher.aliases:
        return False
    words = source.strip().split()
    removed = False
    while len(words) > 1:
        candidate = words[-1].lower()
        candidate = "medium" if candidate == "med" else candidate
        if candidate not in {*TRACKS, "ace", "nonace", *STRATEGIES}:
            break
        words.pop()
        removed = True
    if not removed:
        return False
    remaining = " ".join(words).strip()
    if remaining.lower() in all_rows or UMA_ID_RE.fullmatch(remaining):
        return True
    return matcher.match(remaining) is not None


def parse_uma_selectors(store: UserStore, value: str, matcher: NameMatcher | None = None) -> list[UmaSelector]:
    matcher = matcher or _default_matcher()
    all_rows = _all_by_id(store)
    current_ids = tuple(_current_by_id(store))
    selectors = []
    errors = []

    for raw_item in value.split(","):
        item = raw_item.strip()
        safe_item = item.replace("`", "'") or "(empty)"
        if not item:
            errors.append("`(empty)` is malformed")
            continue

        pieces = [piece.strip() for piece in item.split("-")]
        source_text = pieces[0]
        if not source_text or any(not piece for piece in pieces[1:]):
            errors.append(f"`{safe_item}` contains an empty dash-separated specifier")
            continue
        if _uses_space_separated_qualifiers(source_text, matcher, all_rows):
            errors.append(f"`{safe_item}` must use dashes between its source and specifiers")
            continue

        qualifiers: dict[str, str | None] = {"track": None, "ace_status": None, "strategy": None}
        item_errors = []
        for raw_token in pieces[1:]:
            token = raw_token.lower()
            token = "medium" if token == "med" else token
            if token in TRACKS:
                qualifier_type = "track"
            elif token in {"ace", "nonace"}:
                qualifier_type = "ace_status"
            elif token in STRATEGIES:
                qualifier_type = "strategy"
            else:
                item_errors.append(f"unknown specifier `{raw_token}`")
                continue
            if qualifiers[qualifier_type] is not None:
                item_errors.append(f"more than one {qualifier_type.replace('_', ' ')} specifier")
                continue
            qualifiers[qualifier_type] = token

        source_lower = source_text.lower()
        if source_lower == "current":
            if not current_ids:
                item_errors.append("that user does not have a current team")
            source_kind = "current"
            source_value = "current"
            uma_ids = current_ids
        elif source_lower == "all":
            source_kind = "all"
            source_value = "all"
            uma_ids = tuple(all_rows)
        elif source_lower in all_rows:
            source_kind = "id"
            source_value = source_lower
            uma_ids = (source_lower,)
        else:
            name_match = matcher.match(source_text)
            if name_match:
                matching_ids = tuple(
                    uma_id
                    for uma_id, row in all_rows.items()
                    if normalize_name(row.get("name", "")) == normalize_name(name_match.name)
                )
                if matching_ids:
                    source_kind = "name"
                    source_value = name_match.name
                    uma_ids = matching_ids
                else:
                    source_kind = "name"
                    source_value = name_match.name
                    uma_ids = ()
                    item_errors.append(f"that user has no saved {name_match.name}")
            else:
                source_kind = "unknown"
                source_value = source_text
                uma_ids = ()
                if UMA_ID_RE.fullmatch(source_text):
                    item_errors.append(f"uma_id `{source_lower}` was not found")
                else:
                    item_errors.append("source did not match current, all, a saved uma_id, or a known uma name")

        if item_errors:
            errors.append(f"`{safe_item}`: " + "; ".join(item_errors))
            continue
        selectors.append(UmaSelector(source_kind, source_value, uma_ids, **qualifiers))

    if errors:
        raise TeamError("Invalid uma filters:\n" + "\n".join(f"- {error}" for error in errors))
    if not selectors:
        raise TeamError("filter must contain at least one uma_id, uma name, current, or all")
    return selectors


def parse_custom_uma_selectors(
    store: UserStore,
    value: str,
    matcher: NameMatcher | None = None,
) -> list[UmaSelector]:
    return parse_uma_selectors(store, value, matcher)


def _selector_matches_config(selector: UmaSelector, key: ConfigKey, current_configs: set[ConfigKey]) -> bool:
    uma_id, track, ace_status, strategy = key
    return (
        uma_id in selector.uma_ids
        and (selector.source_kind != "current" or key in current_configs)
        and (selector.track is None or selector.track == track)
        and (selector.ace_status is None or selector.ace_status == ace_status)
        and (selector.strategy is None or selector.strategy == strategy)
    )


def _normalize_sort_key(sort_key: str) -> str:
    sort_key = "avg" if sort_key == "average" else sort_key
    if sort_key not in {"track", "avg", "median", "min", "max", "stdev"}:
        raise TeamError("sort must be one of: track, avg, median, min, max, stdev")
    return sort_key


def _normalize_order(order: str) -> str:
    normalized = order.strip().lower()
    if normalized not in {"ascending", "descending"}:
        raise TeamError("order must be ascending or descending")
    return normalized


def _track_sort_tuple(current: dict[str, str], label: str = "") -> tuple[int, int, str]:
    track_order = {track: index for index, track in enumerate(TRACKS)}
    position = int(current.get("position", "99")) if str(current.get("position", "")).isdigit() else 99
    return track_order.get(current.get("track", ""), 99), position, label


def build_summary_rows(store: UserStore, sort_key: str, order: str = "descending") -> list[dict[str, str]]:
    sort_key = _normalize_sort_key(sort_key)
    order = _normalize_order(order)
    grouped: dict[str, dict[str, object]] = {}
    for all_row, current, record in _matching_current_records(store):
        key = all_row["uma_id"].lower()
        grouped.setdefault(key, {"all": all_row, "current": current, "scores": []})
        grouped[key]["scores"].append(int(record["score"]))

    rows: list[dict[str, str]] = []
    for item in grouped.values():
        scores = item["scores"]
        if not scores:
            continue
        average = statistics.mean(scores)
        median = statistics.median(scores)
        stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        all_row = item["all"]
        current = item["current"]
        rows.append(
            {
                "name": all_row["name"],
                "track": current["track"],
                "position": current["position"],
                "average score": f"{average:.0f}",
                "median score": f"{median:.0f}",
                "min score": str(min(scores)),
                "max score": str(max(scores)),
                "stdev score": f"{stdev:.0f}",
            }
        )

    if sort_key == "track":
        track_order = {track: index for index, track in enumerate(TRACKS)}
        rows.sort(
            key=lambda row: (
                -track_order.get(row["track"], 99) if order == "ascending" else track_order.get(row["track"], 99),
                int(row.get("position", "99")),
                row["name"],
            )
        )
    else:
        column = "average score" if sort_key == "avg" else f"{sort_key} score"
        rows.sort(
            key=lambda row: (
                -int(row[column]) if order == "descending" else int(row[column]),
                *_track_sort_tuple({"track": row["track"], "position": row.get("position", "99")}, row["name"]),
            )
        )
    return rows


def _score_metric(scores: list[int], sort_key: str) -> float:
    if not scores:
        return math.inf
    if sort_key == "avg":
        return statistics.mean(scores)
    if sort_key == "median":
        return statistics.median(scores)
    if sort_key == "min":
        return min(scores)
    if sort_key == "max":
        return max(scores)
    if sort_key == "stdev":
        return statistics.stdev(scores) if len(scores) > 1 else 0.0
    return math.inf


def _sort_boxplot_series(series: list[BoxplotSeries], sort_key: str, order: str) -> list[BoxplotSeries]:
    def track_key(item: BoxplotSeries) -> tuple[int, int, str]:
        track, position, label = _track_sort_tuple({"track": item.track, "position": item.position}, item.label)
        return (-track if order == "ascending" else track), position, label

    if sort_key == "track":
        return sorted(series, key=track_key)

    populated = [item for item in series if item.scores]
    empty = [item for item in series if not item.scores]

    def numeric_key(item: BoxplotSeries) -> tuple[float, int, int, str]:
        metric = _score_metric(item.scores, sort_key)
        metric_key = -metric if order == "descending" else metric
        return metric_key, *track_key(item)

    return sorted(populated, key=numeric_key) + sorted(empty, key=track_key)


def _render_boxplot(
    series: list[BoxplotSeries],
    path: Path,
    *,
    title: str,
    sort_key: str,
    order: str,
    wrap_labels: bool = False,
    highlight_current: bool = False,
) -> None:
    if not series:
        raise TeamError("No score configurations matched that request.")

    positions = list(range(1, len(series) + 1))
    non_empty = [(position, item) for position, item in zip(positions, series) if item.scores]
    all_scores = [score for item in series for score in item.scores]
    per_column_width = 1.05 if wrap_labels else 0.78
    fig_width = max(10, min(50, len(series) * per_column_width))
    fig_height = 8.0 if wrap_labels else 7.2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=160)

    if non_empty:
        boxplot = ax.boxplot(
            [item.scores for _, item in non_empty],
            positions=[position for position, _ in non_empty],
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#0f5132", "linewidth": 2},
            boxprops={"facecolor": "#a7d8de", "edgecolor": "#284b63", "linewidth": 1.35},
            whiskerprops={"color": "#284b63", "linewidth": 1.2},
            capprops={"color": "#284b63", "linewidth": 1.2},
        )
        for patch, (_, item) in zip(boxplot["boxes"], non_empty):
            if highlight_current and item.is_current:
                patch.set_facecolor("#d8c7f2")
                patch.set_edgecolor("#6d3ca0")
                patch.set_hatch("///")

    if all_scores:
        low = min(all_scores)
        high = max(all_scores)
        padding = max(1000, (high - low) * 0.08)
        y_min, y_max = max(0, low - padding), high + padding
    else:
        y_min, y_max = 0, 1
    ax.set_ylim(y_min, y_max)
    no_data_y = y_min + (y_max - y_min) / 2

    rng = random.Random(17)
    for position, item in zip(positions, series):
        if not item.scores:
            ax.text(position, no_data_y, "no data", ha="center", va="center", rotation=90, color="#8a97a3", fontsize=8)
            continue
        jittered = [position + rng.uniform(-0.13, 0.13) for _ in item.scores]
        ax.scatter(jittered, item.scores, s=18, color="#f97316", alpha=0.45, edgecolors="none", zorder=3)

    for position, item in zip(positions, series):
        if item.thumbnail_path is None:
            continue
        try:
            with Image.open(item.thumbnail_path) as thumbnail:
                thumbnail.thumbnail((72, 72))
                pixels = np.asarray(thumbnail.convert("RGBA"))
            image_box = OffsetImage(pixels, zoom=0.58)
            annotation = AnnotationBbox(
                image_box,
                (position, -0.02),
                xycoords=("data", "axes fraction"),
                box_alignment=(0.5, 1),
                frameon=False,
                annotation_clip=False,
                pad=0,
            )
            ax.add_artist(annotation)
        except (OSError, ValueError):
            continue

    labels = [textwrap.fill(item.label, width=30, break_long_words=False) if wrap_labels else item.label for item in series]
    ax.set_ylabel("score")
    ax.set_title(f"{title} (sorted by {sort_key}, {order})")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color="#d9e2ec", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=55 if wrap_labels else 45, labelsize=7 if wrap_labels else 8, pad=58)
    if highlight_current and any(item.is_current for item in series):
        ax.legend(
            handles=[Patch(facecolor="#d8c7f2", edgecolor="#6d3ca0", hatch="///", label="Currently on team")],
            loc="upper right",
            frameon=False,
        )
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.52 if wrap_labels else 0.43)
    fig.savefig(path)
    plt.close(fig)


def build_boxplot(
    store: UserStore,
    path: Path,
    sort_key: str = "median",
    order: str = "descending",
    matcher: NameMatcher | None = None,
) -> None:
    matcher = matcher or _default_matcher()
    sort_key = _normalize_sort_key(sort_key)
    order = _normalize_order(order)
    all_rows = _all_by_id(store)
    current_rows = _current_by_id(store)
    if not current_rows:
        raise TeamError("That user does not have a current team associated with them.")

    grouped: dict[str, list[int]] = {uma_id: [] for uma_id in current_rows}
    for all_row, current, record in _matching_current_records(store):
        grouped.setdefault(all_row["uma_id"].lower(), []).append(int(record["score"]))

    series = []
    for uma_id, current in current_rows.items():
        all_row = all_rows.get(uma_id)
        if not all_row:
            continue
        series.append(
            BoxplotSeries(
                label=all_row["name"],
                scores=grouped.get(uma_id, []),
                track=current["track"],
                position=current["position"],
                is_current=True,
                thumbnail_path=matcher.thumbnail_path(all_row["name"], all_row["outfit"]),
            )
        )
    ordered = _sort_boxplot_series(series, sort_key, order)
    _render_boxplot(ordered, path, title="Current Team Score Distribution", sort_key=sort_key, order=order)


def _merged_selector_label(selector: UmaSelector, all_rows: dict[str, dict[str, str]]) -> str:
    if selector.source_kind in {"current", "all", "name"}:
        base = selector.source_value
    else:
        uma = all_rows[selector.source_value]
        base = f"{uma['name']} ({uma['outfit']}) {selector.source_value}"
    qualifiers = [selector.ace_status, selector.track, selector.strategy]
    return " ".join([base, *(qualifier for qualifier in qualifiers if qualifier)])


def _date_cutoff(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        normalized = parse_date(value)
    except ValueError as exc:
        raise TeamError(str(exc).replace("date_acquired", "date_after")) from exc
    return datetime.strptime(normalized, "%m/%d/%Y")


def _eligible_uma_ids(all_rows: dict[str, dict[str, str]], cutoff: datetime | None) -> set[str]:
    if cutoff is None:
        return set(all_rows)
    eligible = set()
    for uma_id, row in all_rows.items():
        try:
            acquired = datetime.strptime(row.get("date_acquired", ""), "%m/%d/%Y")
        except ValueError:
            continue
        if acquired >= cutoff:
            eligible.add(uma_id)
    return eligible


def _thumbnail_for_ids(
    uma_ids: set[str],
    all_rows: dict[str, dict[str, str]],
    matcher: NameMatcher,
) -> Path | None:
    if len(uma_ids) != 1:
        return None
    row = all_rows[next(iter(uma_ids))]
    return matcher.thumbnail_path(row["name"], row["outfit"])


def build_custom_boxplot_series(
    store: UserStore,
    umas: str,
    merge_same_uma: bool = False,
    sort_key: str = "median",
    order: str = "descending",
    matcher: NameMatcher | None = None,
    date_after: str | None = None,
) -> list[BoxplotSeries]:
    matcher = matcher or _default_matcher()
    sort_key = _normalize_sort_key(sort_key)
    order = _normalize_order(order)
    selectors = parse_uma_selectors(store, umas, matcher)
    all_rows = _all_by_id(store)
    current_rows = _current_by_id(store)
    current_ids = set(current_rows)
    eligible_ids = _eligible_uma_ids(all_rows, _date_cutoff(date_after))

    scores_by_config: dict[ConfigKey, list[int]] = {}
    for record in store.read_records():
        uma_id, track, ace_status, strategy = _record_config_key(record)
        if uma_id not in all_rows or track not in TRACKS or not ace_status or strategy not in STRATEGIES:
            continue
        try:
            score = int(record["score"])
        except ValueError:
            continue
        scores_by_config.setdefault((uma_id, track, ace_status, strategy), []).append(score)

    current_configs = set(_current_config_keys(store))

    def matching_configs(selector: UmaSelector) -> list[ConfigKey]:
        candidates = current_configs if selector.source_kind == "current" else set(scores_by_config)
        return [
            key
            for key in candidates
            if key[0] in eligible_ids and _selector_matches_config(selector, key, current_configs)
        ]

    series = []
    if merge_same_uma:
        seen_selectors = set()
        track_order = {track: index for index, track in enumerate(TRACKS)}
        for selector in selectors:
            if selector.canonical_key in seen_selectors:
                continue
            seen_selectors.add(selector.canonical_key)
            selectable_ids = set(selector.uma_ids) & eligible_ids
            if not selectable_ids:
                continue
            configs = matching_configs(selector)
            scores = [score for key in configs for score in scores_by_config.get(key, [])]
            represented_ids = {key[0] for key in configs} or selectable_ids
            represented_tracks = {key[1] for key in configs}
            represented_statuses = {key[2] for key in configs}
            representative_track = selector.track or min(represented_tracks, key=lambda track: track_order[track], default="")
            if selector.ace_status:
                representative_position = "1" if selector.ace_status == "ace" else "2"
            else:
                representative_position = "1" if "ace" in represented_statuses else "2" if "nonace" in represented_statuses else "99"
            series.append(
                BoxplotSeries(
                    label=_merged_selector_label(selector, all_rows),
                    scores=scores,
                    track=representative_track,
                    position=representative_position,
                    is_current=bool(represented_ids) and represented_ids <= current_ids,
                    thumbnail_path=_thumbnail_for_ids(represented_ids, all_rows, matcher),
                )
            )
    else:
        seen_configs = set()
        for selector in selectors:
            for key in matching_configs(selector):
                if key in seen_configs:
                    continue
                seen_configs.add(key)
                uma_id, track, ace_status, strategy = key
                uma = all_rows[uma_id]
                series.append(
                    BoxplotSeries(
                        label=f"{uma['name']} ({uma['outfit']}) {uma_id} {ace_status} {track} {strategy}",
                        scores=list(scores_by_config.get(key, [])),
                        track=track,
                        position="1" if ace_status == "ace" else "2",
                        is_current=uma_id in current_ids,
                        thumbnail_path=matcher.thumbnail_path(uma["name"], uma["outfit"]),
                    )
                )

    if not series:
        raise TeamError("No score configurations matched those uma selections and specifiers.")
    return _sort_boxplot_series(series, sort_key, order)


def build_custom_boxplot(
    store: UserStore,
    path: Path,
    umas: str,
    merge_same_uma: bool = False,
    sort_key: str = "median",
    order: str = "descending",
    matcher: NameMatcher | None = None,
    date_after: str | None = None,
) -> None:
    series = build_custom_boxplot_series(store, umas, merge_same_uma, sort_key, order, matcher, date_after)
    _render_boxplot(
        series,
        path,
        title="Custom Uma Score Distribution",
        sort_key=_normalize_sort_key(sort_key),
        order=_normalize_order(order),
        wrap_labels=True,
        highlight_current=True,
    )
