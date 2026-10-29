import os
from pathlib import Path

STORAGE_DIR = Path(os.environ.get("AUDIO_ENGINE_STORAGE", "/tmp/audio-engine")).resolve()
MAX_UPLOAD_MB = int(os.environ.get("AUDIO_ENGINE_MAX_UPLOAD_MB", "40"))
DEFAULT_MODEL = os.environ.get("AUDIO_ENGINE_DEFAULT_MODEL", "auto")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
MAX_DURATION_S = int(os.environ.get("AUDIO_ENGINE_MAX_DURATION_S", "300"))
MAX_LANES = max(1, int(os.environ.get("AUDIO_ENGINE_MAX_LANES", "32")))  # edit-model lanes per project
