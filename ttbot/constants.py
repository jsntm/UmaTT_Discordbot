from __future__ import annotations


TRACKS = ["sprint", "mile", "medium", "long", "dirt"]
STRATEGIES = ["front", "pace", "late", "end"]
POSITIONS = [1, 2, 3]
POSITION_LABELS = {1: "ace", 2: "2", 3: "3"}
SORT_KEYS = ["track", "avg", "median", "min", "max", "stdev"]
ORDER_KEYS = ["ascending", "descending"]

ALL_UMAS_COLUMNS = ["uma_id", "outfit", "name", "rating", "date_acquired"]
CURRENT_TEAM_COLUMNS = ["uma_id", "track", "position", "strategy"]
RECORDS_COLUMNS = ["uma_id", "time_of_screenshot", "track", "position", "strategy", "score"]
RECORDS_EXPORT_COLUMNS = [
    "index",
    "uma_id",
    "outfit",
    "name",
    "rating",
    "date_acquired",
    "time_of_screenshot",
    "track",
    "position",
    "strategy",
    "score",
]
SUMMARY_COLUMNS = ["name", "track", "average score", "median score", "min score", "max score", "stdev score"]

DEFAULT_OCR_SETTINGS = {
    "top": {
        "top_left": [220, 430],
        "bottom_right": [970, 2080],
        "base_size": [1080, 2340],
    },
    "bottom": {
        "top_left": [220, 280],
        "bottom_right": [970, 1900],
        "base_size": [1080, 2340],
    },
}
