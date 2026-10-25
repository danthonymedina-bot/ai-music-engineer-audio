"""Tone shaping: RBJ biquad shelves/peaking EQ + tanh saturation. Pure numpy/scipy."""
import numpy as np
from scipy.signal import sosfilt


def _biquad(kind: str, f0: float, sr: int, gain_db: float, q: float = 0.707) -> np.ndarray:
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * q)
    if kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == "lowshelf":
        sa = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) - (A - 1) * cw + sa), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sa)]
        a = [(A + 1) + (A - 1) * cw + sa, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sa]
    elif kind == "highshelf":
        sa = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
        a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    else:
        raise ValueError(kind)
    b, a = np.array(b) / a[0], np.array(a) / a[0]
    return np.concatenate([b, a])[None, :]


def eq(x, sr, low_gain_db=0.0, mid_gain_db=0.0, high_gain_db=0.0,
       low_freq=120.0, mid_freq=1000.0, high_freq=6000.0, mid_q=1.0):
    """x: (samples,) or (channels, samples)."""
    x = np.asarray(x, dtype=np.float32)
    sos = [s for g, s in ((low_gain_db, lambda: _biquad("lowshelf", low_freq, sr, low_gain_db)),
                          (mid_gain_db, lambda: _biquad("peak", mid_freq, sr, mid_gain_db, mid_q)),
                          (high_gain_db, lambda: _biquad("highshelf", high_freq, sr, high_gain_db))) if g]
    if not sos:
        return x.copy()
    sos = np.concatenate([f() for f in sos]).astype(np.float32)
    if x.ndim == 1:
        return sosfilt(sos, x).astype(np.float32)
    out = np.empty_like(x)
    for c in range(x.shape[0]):  # per channel, float32: bounded memory on 512 MB hosts
        out[c] = sosfilt(sos, x[c])
    return out


def saturate(x, drive=0.0, mix=1.0):
    """tanh soft clip. drive in dB pre-gain; mix 0..1 dry/wet."""
    x = np.asarray(x, dtype=np.float32)
    if drive <= 0 or mix <= 0:
        return x
    g = np.float32(10 ** (drive / 20))
    wet = np.tanh(g * x)
    wet *= np.float32(mix / np.tanh(g))  # full-scale stays full-scale
    wet += np.float32(1 - mix) * x
    return wet


def tone(x, sr, low=0.0, mid=0.0, high=0.0, drive=0.0, mix=0.0, **kw):
    return saturate(eq(x, sr, low, mid, high, **kw), drive, mix)
