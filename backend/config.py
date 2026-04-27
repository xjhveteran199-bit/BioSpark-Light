r"""
BioSpark-Light configuration.

Local desktop app — all writable state lives in the user's per-OS data dir
(via platformdirs), NOT next to the executable. This keeps the app portable
and avoids permission issues on Program Files / /Applications installs.

Resolved layout (Windows example):
  %LOCALAPPDATA%\BioSpark-Light\
    biospark.db          # SQLite (training history + checkpoints metadata)
    uploads\             # User-uploaded raw datasets (Mode A/B/C inputs)
    checkpoints\<uid>\   # Per-user warm-start chain (.pt files)
"""

import sys
from pathlib import Path

from platformdirs import user_data_dir

APP_NAME = "BioSpark-Light"

# All user-writable paths land here.
#   Windows: %LOCALAPPDATA%\BioSpark-Light
#   macOS:   ~/Library/Application Support/BioSpark-Light
#   Linux:   ~/.local/share/BioSpark-Light
# We pass appauthor=False so Windows skips the redundant <Author>\<App> nesting.
USER_DATA_DIR = Path(user_data_dir(APP_NAME, appauthor=False))
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Source-tree dirs (read-only at runtime).
#
# Two cases:
#   * Source: __file__ = .../BioSpark-Light/backend/config.py
#                 → BASE_DIR = .../BioSpark-Light
#                 → frontend at BASE_DIR / "frontend"
#   * PyInstaller frozen (one-folder): everything bundled under sys._MEIPASS,
#                 with the spec's `datas` having added frontend → "frontend".
#                 → frontend at sys._MEIPASS / "frontend"
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    BASE_DIR = Path(sys._MEIPASS)  # type: ignore[attr-defined]
else:
    BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Writable dirs (per-user)
UPLOAD_DIR = USER_DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS_DIR = USER_DATA_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# Default DB path — used by database.py if DATABASE_URL not set
DB_FILE = USER_DATA_DIR / "biospark.db"

# Server (loopback only — desktop app, never expose externally)
HOST = "127.0.0.1"
PORT = 8765

# Upload limits — same as web version
MAX_FILE_SIZE_MB = 200
