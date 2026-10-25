"""Multi-track overlay: per-track gain (dB), constant-power pan, offset (ms), then peak limiter."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Track:
    audio: np.ndarray  # (samples,) mono or (C, samples)
    gain_db: float = 0.0
    pan: float = 0.0  # -1 L .. +1 R
    offset_ms: float = 0.0


def pan_gains(pan: float):
    """Constant-power (sin/cos) law: L^2 + R^2 == 1."""
    theta = (float(np.clip(pan, -1, 1)) + 1) * np.pi / 4
    return np.cos(theta), np.sin(theta)


def _to_stereo(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        return np.stack([a, a])
    if a.shape[0] == 1:
        return np.repeat(a, 2, axis=0)
    return a[:2]


def limiter(x, ceiling_db=-1.0, release_ms=50.0, sr=44100):
    """Instant-attack peak limiter with exponential release. Guarantees |y| <= ceiling."""
    ceil = 10 ** (ceiling_db / 20)
    peak = np.max(np.abs(x), axis=0)
    target = np.minimum(1.0, ceil / np.maximum(peak, 1e-12))
    if np.all(target >= 1.0):
        return x.astype(np.float32)
    rel = np.exp(-1.0 / (release_ms * 1e-3 * sr))
    g = np.empty_like(target)
    cur = 1.0
    for i, t in enumerate(target):
        cur = t if t < cur else t + (cur - t) * rel
        g[i] = cur
    return (x * g).astype(np.float32)


def mix(tracks, sr, ceiling_db=-1.0, limit=True):
    if not tracks:
        raise ValueError("no tracks")
    prepared = []
    for t in tracks:
        st = _to_stereo(t.audio)
        off = int(round(max(t.offset_ms, 0) * sr / 1000))
        gl, gr = pan_gains(t.pan)
        # sqrt(2) so a centred track passes at unity gain
        st = st * (10 ** (t.gain_db / 20)) * np.array([[gl], [gr]], dtype=np.float32) * np.sqrt(2)
        prepared.append((off, st))
    n = max(off + st.shape[1] for off, st in prepared)
    out = np.zeros((2, n), dtype=np.float32)
    for off, st in prepared:
        out[:, off:off + st.shape[1]] += st
    return limiter(out, ceiling_db, sr=sr) if limit else out
