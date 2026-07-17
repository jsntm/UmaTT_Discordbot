from __future__ import annotations

import csv
from dataclasses import dataclass
import math
import random
import re
import statistics
import textwrap
from pathlib import Path

from ttbot import config  # Sets MPLCONFIGDIR before matplotlib imports.

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ttbot.constants import ALL_UMAS_COLUMNS, RECORDS_EXPORT_COLUMNS, STRATEGIES, TRACKS
from ttbot.storage import UserStore
from ttbot.team import TeamError
from ttbot.validation import normalize_uma_id


@dataclass(frozen=True)
class CustomUmaSelector:
    source: str
    track: str | None = None
    ace_status: str | None = None
    strategy: str | None = None

    @property
    def canonical_key(self) -> tuple[str, str | None, str | None, str | None]:
        return self.source, self.track, self.ace_status, self.strategy


@dataclass
class BoxplotSeries:
    label: str
    scores: list[int]
    track: str = ""
    position: str = "99"


def _all_by_id(store: UserStore) -> dict[str, dict[str, str]]:
    return {row["uma_id"].lower(): row for row in store.read_all_umas()}


def _current_by_id(store: UserStore) -> dict[str, dict[str, str]]:
    return {row["uma_id"].lower(): row for row in store.read_current_team()}


def _joined_record(store: UserStore, record: dict[str, str]) -> dict[str, str] | None:
    all_row = _all_by_id(store).get(record["uma_id"].lower())
    if not all_row:
        return None
    return {**all_row, **record}


def build_records_export(store: UserStore, scope: str) -> list[dict[str, str]]:
    scope = scope.strip().lower()
    records = store.read_records()
    if not records:
        raise TeamError("That user does not have records associated with them.")
    current_ids = set(_current_by_id(store))
    all_ids = set(_all_by_id(store))
    indexed_records = list(enumerate(records, start=1))
    if scope == "current":
        selected = [(index, record) for index, record in indexed_records if record["uma_id"].lower() in current_ids]
    elif scope == "all":
        selected = indexed_records
    else:
        try:
            uma_id = normalize_uma_id(scope)
        except ValueError as exc:
            raise TeamError("scope must be current, all, or a valid 5 digit alphanumeric uma code") from exc
        if uma_id not in all_ids:
            raise TeamError(f"{uma_id} does not correspond to a valid uma for that user.")
        selected = [(index, record) for index, record in indexed_records if record["uma_id"].lower() == uma_id]
    rows = [{"index": str(index), **joined} for index, record in selected if (joined := _joined_record(store, record))]
    if not rows:
        raise TeamError("No records matched that scope.")
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
        return _track_sort_tuple({"track": item.track, "position": item.position}, item.label)

    if sort_key == "track":
        return sorted(series, key=track_key, reverse=order == "descending")

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
) -> None:
    if not series:
        raise TeamError("No score configurations matched that request.")

    positions = list(range(1, len(series) + 1))
    non_empty = [(position, item.scores) for position, item in zip(positions, series) if item.scores]
    all_scores = [score for item in series for score in item.scores]
    per_column_width = 1.05 if wrap_labels else 0.78
    fig_width = max(10, min(50, len(series) * per_column_width))
    fig_height = 7.4 if wrap_labels else 6.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=160)

    if non_empty:
        ax.boxplot(
            [scores for _, scores in non_empty],
            positions=[position for position, _ in non_empty],
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#0f5132", "linewidth": 2},
            boxprops={"facecolor": "#a7d8de", "edgecolor": "#284b63", "linewidth": 1.35},
            whiskerprops={"color": "#284b63", "linewidth": 1.2},
            capprops={"color": "#284b63", "linewidth": 1.2},
        )

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

    labels = [textwrap.fill(item.label, width=30, break_long_words=False) if wrap_labels else item.label for item in series]
    ax.set_ylabel("score")
    ax.set_title(f"{title} (sorted by {sort_key}, {order})")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color="#d9e2ec", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=55 if wrap_labels else 45, labelsize=7 if wrap_labels else 8)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.42 if wrap_labels else 0.32)
    fig.savefig(path)
    plt.close(fig)


def build_boxplot(store: UserStore, path: Path, sort_key: str = "median", order: str = "descending") -> None:
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
            )
        )
    ordered = _sort_boxplot_series(series, sort_key, order)
    _render_boxplot(ordered, path, title="Current Team Score Distribution", sort_key=sort_key, order=order)


def parse_custom_uma_selectors(store: UserStore, value: str) -> list[CustomUmaSelector]:
    all_rows = _all_by_id(store)
    selectors = []
    errors = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            errors.append("`(empty)` is malformed")
            continue
        tokens = [token for token in re.split(r"[\s-]+", item.lower()) if token]
        source = tokens[0]
        if source not in {"current", "all"}:
            try:
                source = normalize_uma_id(source)
            except ValueError:
                errors.append(f"`{item.replace('`', "'")}` has a malformed uma_id")
                continue
            if source not in all_rows:
                errors.append(f"`{item.replace('`', "'")}`: uma_id `{source}` was not found")
                continue

        qualifiers: dict[str, str | None] = {"track": None, "ace_status": None, "strategy": None}
        item_errors = []
        for token in tokens[1:]:
            token = "medium" if token == "med" else token
            if token in TRACKS:
                qualifier_type = "track"
            elif token in {"ace", "nonace"}:
                qualifier_type = "ace_status"
            elif token in STRATEGIES:
                qualifier_type = "strategy"
            else:
                item_errors.append(f"unknown specifier `{token}`")
                continue
            if qualifiers[qualifier_type] is not None:
                item_errors.append(f"more than one {qualifier_type.replace('_', ' ')} specifier")
                continue
            qualifiers[qualifier_type] = token
        if item_errors:
            errors.append(f"`{item.replace('`', "'")}`: " + "; ".join(item_errors))
            continue
        selectors.append(CustomUmaSelector(source=source, **qualifiers))

    if errors:
        raise TeamError("Invalid uma selections:\n" + "\n".join(f"- {error}" for error in errors))
    if not selectors:
        raise TeamError("umas must contain at least one uma_id, current, or all")
    return selectors


def _selector_matches_config(selector: CustomUmaSelector, key: tuple[str, str, str, str]) -> bool:
    uma_id, track, ace_status, strategy = key
    return (
        (selector.source in {"current", "all"} or selector.source == uma_id)
        and (selector.track is None or selector.track == track)
        and (selector.ace_status is None or selector.ace_status == ace_status)
        and (selector.strategy is None or selector.strategy == strategy)
    )


def _merged_selector_label(selector: CustomUmaSelector, all_rows: dict[str, dict[str, str]]) -> str:
    if selector.source in {"current", "all"}:
        base = selector.source
    else:
        uma = all_rows[selector.source]
        base = f"{uma['name']} ({uma['outfit']}) {selector.source}"
    qualifiers = [selector.ace_status, selector.track, selector.strategy]
    return " ".join([base, *(qualifier for qualifier in qualifiers if qualifier)])


def build_custom_boxplot_series(
    store: UserStore,
    umas: str,
    merge_same_uma: bool = False,
    sort_key: str = "median",
    order: str = "descending",
) -> list[BoxplotSeries]:
    sort_key = _normalize_sort_key(sort_key)
    order = _normalize_order(order)
    selectors = parse_custom_uma_selectors(store, umas)
    all_rows = _all_by_id(store)
    current_rows = _current_by_id(store)
    if any(selector.source == "current" for selector in selectors) and not current_rows:
        raise TeamError("That user does not have a current team associated with them.")

    scores_by_config: dict[tuple[str, str, str, str], list[int]] = {}
    for record in store.read_records():
        uma_id = record["uma_id"].lower()
        track = record["track"].lower()
        ace_status = _ace_status(record["position"])
        strategy = record["strategy"].lower()
        if uma_id not in all_rows or track not in TRACKS or not ace_status or strategy not in STRATEGIES:
            continue
        try:
            score = int(record["score"])
        except ValueError:
            continue
        scores_by_config.setdefault((uma_id, track, ace_status, strategy), []).append(score)

    current_configs = []
    for uma_id, current in current_rows.items():
        track = current["track"].lower()
        ace_status = _ace_status(current["position"])
        strategy = current["strategy"].lower()
        if uma_id in all_rows and track in TRACKS and ace_status and strategy in STRATEGIES:
            current_configs.append((uma_id, track, ace_status, strategy))

    def matching_configs(selector: CustomUmaSelector) -> list[tuple[str, str, str, str]]:
        candidates = current_configs if selector.source == "current" else list(scores_by_config)
        return [key for key in candidates if _selector_matches_config(selector, key)]

    series = []
    if merge_same_uma:
        seen_selectors = set()
        track_order = {track: index for index, track in enumerate(TRACKS)}
        for selector in selectors:
            if selector.canonical_key in seen_selectors:
                continue
            seen_selectors.add(selector.canonical_key)
            configs = matching_configs(selector)
            scores = [score for key in configs for score in scores_by_config.get(key, [])]
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
) -> None:
    series = build_custom_boxplot_series(store, umas, merge_same_uma, sort_key, order)
    _render_boxplot(
        series,
        path,
        title="Custom Uma Score Distribution",
        sort_key=_normalize_sort_key(sort_key),
        order=_normalize_order(order),
        wrap_labels=True,
    )
