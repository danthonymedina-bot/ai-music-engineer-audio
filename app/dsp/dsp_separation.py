"""Classical DSP stem separation (LOWER QUALITY, no ML). Memory-bounded: processes in chunks.

Stems:
  vocals        centre-panned harmonic content in the vocal band (stereo similarity mask x HPSS harmonic)
  drums         percussive part of HPSS
  bass          harmonic content below ~160 Hz
  other         residual (mixture - vocals - drums - bass)
Pure functions on numpy arrays: (C, N) float32 -> dict[str, (C, N)].
"""
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import istft, stft

# scipy only (no librosa/numba) to keep the 512 MB free-tier footprint low.
N_FFT, HOP = 2048, 512
KERNEL = 17


def _stft(x, sr):
    return stft(x, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="even", padded=True)[2].astype(np.complex64)


def _istft(S, sr, n):
    y = istft(S, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=True)[1]
    out = np.zeros(n, np.float32)
    out[:min(n, len(y))] = y[:n]
    return out


def hpss_masks(mag, kernel=KERNEL, power=2.0):
    """Fitzgerald median-filter HPSS soft masks (harmonic, percussive)."""
    h = median_filter(mag, size=(1, kernel), mode="reflect")
    p = median_filter(mag, size=(kernel, 1), mode="reflect")
    hp, pp = h ** power, p ** power
    tot = hp + pp + 1e-10
    return (hp / tot).astype(np.float32), (pp / tot).astype(np.float32)


def _sep_chunk(x: np.ndarray, sr: int) -> dict:
    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)
    n = x.shape[1]
    L, R = _stft(x[0], sr), _stft(x[1], sr)
    M = 0.5 * (L + R)
    H, P = hpss_masks(np.abs(M))
    del M
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    # stereo similarity: 1 when L == R (centre-panned), ~0 when hard-panned / decorrelated
    sim = (2 * np.abs(L * np.conj(R)) / (np.abs(L) ** 2 + np.abs(R) ** 2 + 1e-10)).astype(np.float32) ** 4
    vocal_band = ((freqs > 180) & (freqs < 8000)).astype(np.float32)[:, None]
    bass_band = (freqs < 160).astype(np.float32)[:, None]
    masks = {"vocals": H * sim * vocal_band, "drums": P, "bass": H * bass_band}
    del sim
    stems = {k: np.stack([_istft(S * m, sr, n) for S in (L, R)]) for k, m in masks.items()}
    stems["other"] = (x[:2] - stems["vocals"] - stems["drums"] - stems["bass"]).astype(np.float32)
    return stems


def iter_dsp(x: np.ndarray, sr: int, chunk_s: float = 10.0, pad_s: float = 1.0):
    """Yields dict[stem -> (2, n_chunk)] for consecutive chunks; peak memory independent of song length."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float32))
    n = x.shape[1]
    chunk, pad = int(chunk_s * sr), int(pad_s * sr)
    for start in range(0, n, chunk):
        a, b = max(0, start - pad), min(n, start + chunk + pad)
        res = _sep_chunk(x[:, a:b], sr)
        lo = start - a
        hi = lo + min(chunk, n - start)
        yield {k: v[:, lo:hi] for k, v in res.items()}


def separate_dsp(x: np.ndarray, sr: int, **kw) -> dict:
    parts = list(iter_dsp(x, sr, **kw))
    return {k: np.concatenate([p[k] for p in parts], axis=1) for k in parts[0]}
