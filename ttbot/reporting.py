from __future__ import annotations

import csv
import math
import random
import statistics
from pathlib import Path

from ttbot import config  # Sets MPLCONFIGDIR before matplotlib imports.

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ttbot.constants import ALL_UMAS_COLUMNS, RECORDS_EXPORT_COLUMNS, TRACKS
from ttbot.storage import UserStore
from ttbot.team import TeamError
from ttbot.validation import normalize_uma_id


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
        if (
            record["track"] == current["track"]
            and record["position"] == current["position"]
            and record["strategy"] == current["strategy"]
        ):
            matches.append((all_row, current, record))
    return matches


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


def build_boxplot(store: UserStore, path: Path, sort_key: str = "median", order: str = "descending") -> None:
    sort_key = _normalize_sort_key(sort_key)
    order = _normalize_order(order)

    all_rows = _all_by_id(store)
    current_rows = _current_by_id(store)
    if not current_rows:
        raise TeamError("That user does not have a current team associated with them.")

    grouped: dict[str, list[int]] = {uma_id: [] for uma_id in current_rows}
    labels: dict[str, str] = {}
    current_meta: dict[str, dict[str, str]] = {}
    for uma_id, current in current_rows.items():
        all_row = all_rows.get(uma_id)
        if not all_row:
            continue
        labels[uma_id] = all_row["name"]
        current_meta[uma_id] = current

    for all_row, current, record in _matching_current_records(store):
        key = all_row["uma_id"].lower()
        grouped.setdefault(key, []).append(int(record["score"]))
        labels[key] = all_row["name"]

    def score_metric(scores: list[int]) -> float:
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

    def track_item_key(item: tuple[str, list[int]]) -> tuple[int, int, str]:
        uma_id, scores = item
        current = current_meta.get(uma_id, {})
        return _track_sort_tuple(current, labels.get(uma_id, uma_id))

    items = [(uma_id, scores) for uma_id, scores in grouped.items() if uma_id in labels]
    if sort_key == "track":
        ordered = sorted(items, key=track_item_key, reverse=order == "descending")
    else:
        non_empty_items = [(uma_id, scores) for uma_id, scores in items if scores]
        empty_items = [(uma_id, scores) for uma_id, scores in items if not scores]

        def numeric_item_key(item: tuple[str, list[int]]) -> tuple[float, int, int, str]:
            uma_id, scores = item
            metric = score_metric(scores)
            metric_key = -metric if order == "descending" else metric
            track_order, position, label = track_item_key(item)
            return metric_key, track_order, position, label

        ordered = sorted(non_empty_items, key=numeric_item_key) + sorted(empty_items, key=track_item_key)
    names = [labels[key] for key, _ in ordered]
    positions = list(range(1, len(ordered) + 1))
    non_empty = [(position, scores) for position, (_, scores) in zip(positions, ordered) if scores]
    all_scores = [score for _, scores in ordered for score in scores]

    fig_width = max(10, len(ordered) * 0.78)
    fig, ax = plt.subplots(figsize=(fig_width, 6.5), dpi=160)
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

    rng = random.Random(17)
    for position, (_, scores) in zip(positions, ordered):
        if not scores:
            ax.text(position, 0.5, "no data", ha="center", va="center", rotation=90, color="#8a97a3", fontsize=8)
            continue
        jittered = [position + rng.uniform(-0.13, 0.13) for _ in scores]
        ax.scatter(jittered, scores, s=18, color="#f97316", alpha=0.45, edgecolors="none", zorder=3)

    ax.set_ylabel("score")
    ax.set_title(f"Current Team Score Distribution (sorted by {sort_key}, {order})")
    ax.set_xticks(positions)
    ax.set_xticklabels(names)
    if all_scores:
        low = min(all_scores)
        high = max(all_scores)
        padding = max(1000, (high - low) * 0.08)
        ax.set_ylim(max(0, low - padding), high + padding)
    else:
        ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#d9e2ec", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.32)
    fig.savefig(path)
    plt.close(fig)
