from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
REFERENCE_NAMES_FILE = ROOT_DIR / "reference_data" / "all_uma_names.txt"
TOKEN_FILE = ROOT_DIR / "umaTTauth.txt"
DATA_DIR = Path(os.environ.get("UMA_BOT_DATA_DIR", ROOT_DIR / "data"))
USERS_DIR = DATA_DIR / "users"
TMP_DIR = DATA_DIR / "tmp"
MPLCONFIG_DIR = DATA_DIR / "matplotlib"
OCR_SETTINGS_FILE = DATA_DIR / "ocr_settings.json"
STITCH_SETTINGS_FILE = DATA_DIR / "stitch_settings.json"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

KEEP_IMAGES = os.environ.get("UMA_BOT_KEEP_IMAGES", "").strip().lower() in {"1", "true", "yes", "on"}
OCR_PROVIDER = os.environ.get("UMA_BOT_OCR_PROVIDER", "auto").strip().lower()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
EASYOCR_GPU = os.environ.get("UMA_BOT_EASYOCR_GPU", "").strip().lower() in {"1", "true", "yes", "on"}
EASYOCR_CPU_THREADS = max(1, int(os.environ.get("UMA_BOT_EASYOCR_CPU_THREADS", "1")))
MAX_DISCORD_FILE_BYTES = int(os.environ.get("MAX_DISCORD_FILE_BYTES", str(8 * 1024 * 1024)))


def ensure_runtime_dirs() -> None:
    USERS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)


def read_discord_token() -> str:
    if not TOKEN_FILE.exists():
        raise RuntimeError(f"Discord token file not found: {TOKEN_FILE}")
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(f"Discord token file is empty: {TOKEN_FILE}")
    return token


ensure_runtime_dirs()
