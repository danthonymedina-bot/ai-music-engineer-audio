"""Pitch shift / time stretch. Prefers Rubber Band (pyrubberband -> rubberband CLI), falls back to librosa."""
import shutil
import numpy as np


def rubberband_cli_available() -> bool:
    return shutil.which("rubberband") is not None


def rubberband_file(src: str, dst: str, semitones=0.0, tempo=1.0):
    """File-to-file via the rubberband CLI (streams; ~constant memory). tempo>1 = faster."""
    import subprocess
    subprocess.run(["rubberband", "-q", "-t", repr(1.0 / tempo), "-p", repr(float(semitones)), src, dst],
                   check=True, capture_output=True)


def rubberband_available() -> bool:
    if shutil.which("rubberband") is None:
        return False
    try:
        import pyrubberband  # noqa: F401
        return True
    except ImportError:
        return False


def _per_channel(fn, x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return fn(x)
    return np.stack([fn(c) for c in x])


def pitch_tempo(x, sr, semitones=0.0, tempo=1.0, engine="auto"):
    """tempo>1 = faster/shorter. Returns (y, engine_used)."""
    if tempo <= 0:
        raise ValueError("tempo must be > 0")
    if engine == "rubberband" or (engine == "auto" and rubberband_available()):
        import pyrubberband as prb

        def f(c):
            y = c
            if tempo != 1.0:
                y = prb.time_stretch(y, sr, tempo)
            if semitones:
                y = prb.pitch_shift(y, sr, semitones)
            return y
        return _per_channel(f, x).astype(np.float32), "rubberband"
    import librosa

    def g(c):
        y = c
        if tempo != 1.0:
            y = librosa.effects.time_stretch(y, rate=tempo)
        if semitones:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
        return y
    return _per_channel(g, x).astype(np.float32), "librosa"
