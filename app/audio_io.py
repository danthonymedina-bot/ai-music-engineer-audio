import subprocess
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf


def load(path: Path):
    """Returns ((channels, samples) float32, sr). Falls back to ffmpeg for mp3/m4a/aac/ogg-opus."""
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), tmp.name], check=True)
            data, sr = sf.read(tmp.name, dtype="float32", always_2d=True)
    return data.T, sr


def save(path: Path, x: np.ndarray, sr: int):
    x = np.clip(np.asarray(x), -1, 1)
    sf.write(str(path), x.T if x.ndim == 2 else x, sr, subtype="PCM_16")


def duration_s(path: Path):
    """Seconds, or None if undecodable. soundfile header first, then ffprobe."""
    try:
        return sf.info(str(path)).duration
    except Exception:
        pass
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip())
    except Exception:
        return None
