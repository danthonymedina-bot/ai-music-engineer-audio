import os
from pathlib import Path

STORAGE_DIR = Path(os.environ.get("AUDIO_ENGINE_STORAGE", "./storage")).resolve()
MAX_UPLOAD_MB = int(os.environ.get("AUDIO_ENGINE_MAX_UPLOAD_MB", "50"))
DEFAULT_MODEL = os.environ.get("AUDIO_ENGINE_DEFAULT_MODEL", "htdemucs")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
MAX_DURATION_S = int(os.environ.get("AUDIO_ENGINE_MAX_DURATION_S", "600"))
