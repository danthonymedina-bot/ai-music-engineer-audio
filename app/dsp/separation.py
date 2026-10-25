"""Stem separation via Demucs (only cloud backend). Imported lazily via subprocess."""
import importlib.util
import subprocess
import sys
from pathlib import Path

DEMUCS_MODELS = {"htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra"}


def demucs_available():
    return importlib.util.find_spec("demucs") is not None


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def separate(input_path: Path, out_dir: Path, model: str = "htdemucs") -> dict:
    """Returns {stem_name: Path}."""
    if model not in DEMUCS_MODELS:
        raise ValueError(f"unknown model {model}")
    if not demucs_available():
        raise RuntimeError("demucs not installed (pip install -r requirements-ml.txt)")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if _cuda() else "cpu"
    # -j 1 and segment keep RAM within ~2 GB on CPU
    subprocess.run([sys.executable, "-m", "demucs", "-n", model, "-d", device, "-j", "1",
                    "-o", str(out_dir), str(input_path)], check=True)
    return {p.stem: p for p in (out_dir / model / input_path.stem).glob("*.wav")}
