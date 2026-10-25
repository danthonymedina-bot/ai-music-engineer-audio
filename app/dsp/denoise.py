"""Spectral-gating denoise via noisereduce with presets."""
import numpy as np
from scipy.signal import butter, sosfilt

PRESETS = {
    "wind": dict(highpass_hz=100.0, stationary=False, prop_decrease=0.8),  # rumble HPF + non-stationary gate
    "air": dict(highpass_hz=None, stationary=False, prop_decrease=0.7),    # gentle room/AC gate
    "hiss": dict(highpass_hz=None, stationary=True, prop_decrease=1.0),    # stationary broadband hiss
}


def highpass(x, sr, cutoff=100.0, order=4):
    sos = butter(order, cutoff, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, x, axis=-1).astype(np.float32)


def denoise(x, sr, preset="hiss", prop_decrease=None):
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset}")
    import noisereduce as nr
    p = PRESETS[preset]
    y = np.asarray(x, dtype=np.float32)
    if p["highpass_hz"]:
        y = highpass(y, sr, p["highpass_hz"])
    kw = dict(sr=sr, stationary=p["stationary"],
              prop_decrease=prop_decrease if prop_decrease is not None else p["prop_decrease"],
              freq_mask_smooth_hz=100)  # default 500 Hz smears masks onto tonal content
    if p["stationary"]:
        # Profile noise from the quietest window rather than the whole clip, so sustained notes aren't learned as noise.
        kw["y_noise"] = quietest_window(y, sr)
    else:
        kw.update(time_constant_s=2.0, thresh_n_mult_nonstationary=1.0)
    kw["chunk_size"] = 131072  # bounds noisereduce's internal float64 buffers
    if y.ndim == 2:  # channel by channel to halve peak memory
        noise = kw.pop("y_noise", None)
        out = np.empty_like(y)
        for c in range(y.shape[0]):
            out[c] = _chunked(nr, y[c], None if noise is None else noise[c], sr, kw)
        return out
    return _chunked(nr, y, kw.pop("y_noise", None), sr, kw)


def _chunked(nr, y, noise, sr, kw, chunk_s=30.0, pad_s=2.0):
    """Denoise in 30 s segments (+2 s context each side) so memory doesn't grow with song length."""
    n, chunk, pad = len(y), int(chunk_s * sr), int(pad_s * sr)
    out = np.empty(n, np.float32)
    for start in range(0, n, chunk):
        a, b = max(0, start - pad), min(n, start + chunk + pad)
        seg = nr.reduce_noise(y=y[a:b], y_noise=noise, **kw)
        lo = start - a
        out[start:start + min(chunk, n - start)] = seg[lo:lo + min(chunk, n - start)]
    return out


def quietest_window(x, sr, win_s=0.5):
    mono = x if x.ndim == 1 else x.mean(axis=0)
    w = max(int(win_s * sr), 1)
    if len(mono) <= w:
        return x
    hop = w // 2
    starts = range(0, len(mono) - w + 1, hop)
    best = min(starts, key=lambda i: float(np.mean(mono[i:i + w] ** 2)))
    return x[..., best:best + w]
