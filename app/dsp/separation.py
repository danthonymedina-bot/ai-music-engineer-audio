"""Separation backend selection. SEPARATION_BACKEND = auto | onnx | dsp | demucs.

auto: onnx (Open-Unmix int8, if model files present) -> demucs (if installed) -> dsp (classical, lower quality).
"""
import importlib.util

import numpy as np
import os
import subprocess
import sys
from pathlib import Path

DEMUCS_MODELS = {"htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra"}
MODELS = DEMUCS_MODELS | {"auto", "umxhq", "dsp"}
QUALITY = {"demucs": "high", "onnx": "medium", "dsp": "low (classical DSP, not ML)"}


def demucs_available():
    return importlib.util.find_spec("demucs") is not None


def onnx_available():
    from .onnx_separation import onnx_available as f
    return f()


def resolve_backend(model: str = "auto") -> str:
    want = os.environ.get("SEPARATION_BACKEND", "auto")
    if model == "dsp":
        return "dsp"
    if model == "umxhq":
        want = "onnx"
    elif model in DEMUCS_MODELS and model != "htdemucs":
        want = "demucs"
    if want == "auto":
        if onnx_available():
            return "onnx"
        if demucs_available():
            return "demucs"
        return "dsp"
    if want == "onnx" and not onnx_available():
        raise RuntimeError("ONNX model files not found (ONNX_MODEL_DIR)")
    if want == "demucs" and not demucs_available():
        raise RuntimeError("demucs not installed")
    return want


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def separate(input_path: Path, out_dir: Path, model: str = "auto") -> dict:
    """Returns {stem_name: Path}; also writes nothing else. Backend recorded by caller via resolve_backend."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model}")
    backend = resolve_backend(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    if backend == "demucs":
        name = model if model in DEMUCS_MODELS else "htdemucs"
        subprocess.run([sys.executable, "-m", "demucs", "-n", name, "-d", "cuda" if _cuda() else "cpu", "-j", "1",
                        "-o", str(out_dir), str(input_path)], check=True)
        return {p.stem: p for p in (out_dir / name / input_path.stem).glob("*.wav")}
    import soundfile as sf
    from ..audio_io import load
    x, sr = load(input_path)
    if backend == "onnx":
        from .onnx_separation import SR as MSR, iter_onnx
        if sr != MSR:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=MSR)
            sr = MSR
        chunks = iter_onnx(x, sr)
    else:
        from .dsp_separation import iter_dsp
        chunks = iter_dsp(x, sr)
    # Stream each chunk straight to disk so only one chunk of stems is ever in RAM.
    paths, writers = {}, {}
    try:
        for part in chunks:
            for k, v in part.items():
                if k not in writers:
                    paths[k] = out_dir / f"{k}.wav"
                    writers[k] = sf.SoundFile(str(paths[k]), "w", sr, v.shape[0], "PCM_16")
                writers[k].write(np.clip(v.T, -1, 1))
    finally:
        for w in writers.values():
            w.close()
    return paths
