"""Synthetic drums + plucks and click metrics shared by the edit tests (and scripts/edit_metrics.py)."""
import numpy as np

SR = 44100


def pluck(f0, dur, sr=SR, decay=3.0, seed=0):
    t = np.arange(int(dur * sr)) / sr
    rng = np.random.default_rng(seed)
    y = sum((0.6 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi)) for k in range(1, 6))
    attack = np.minimum(1.0, t / 0.002)
    return attack * np.exp(-decay * t) * y


def kick(sr=SR, dur=0.35):
    t = np.arange(int(dur * sr)) / sr
    f = 50 + 100 * np.exp(-t * 30)
    return 0.9 * np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 9) * np.minimum(1, t / 0.001)


def noise_hit(sr=SR, dur=0.2, decay=30.0, seed=1, amp=0.4):
    t = np.arange(int(dur * sr)) / sr
    return amp * np.random.default_rng(seed).standard_normal(t.size) * np.exp(-t * decay) * np.minimum(1, t / 0.001)


def place(buf, x, at):
    a = int(at)
    b = min(buf.size, a + x.size)
    buf[a:b] += x[: b - a]


def drums_and_plucks(seconds=6.0, sr=SR, notes=(220.0, 277.18, 329.63, 246.94), seed=0):
    """Onsets on a 0.25 s grid: drums on k*0.5, plucks on 0.25 + k*0.5. Mid-points (x.125 / x.375) are quiet-ish
    sustain regions, ideal for splice tests. Returns (2, N) float64 with a slight stereo difference."""
    n = int(seconds * sr)
    y = np.zeros(n)
    for k in range(int(seconds / 0.5)):
        place(y, kick() if k % 2 == 0 else noise_hit(seed=seed + k, decay=25), k * 0.5 * sr)
        place(y, noise_hit(dur=0.05, decay=120, amp=0.15, seed=seed + 100 + k), k * 0.5 * sr)
        place(y, pluck(notes[k % len(notes)], 1.0, seed=seed + k), (0.25 + k * 0.5) * sr)
    y *= 0.8 / np.max(np.abs(y))
    return np.stack([y, 0.9 * y])


def plucks(seconds=6.0, sr=SR, notes=(392.0, 493.88, 587.33, 440.0), seed=7):
    """Sustained plucks with onsets every 0.5 s (k*0.5)."""
    n = int(seconds * sr)
    y = np.zeros(n)
    for k in range(int(seconds / 0.5)):
        place(y, pluck(notes[k % len(notes)], 1.2, decay=1.5, seed=seed + k), k * 0.5 * sr)
    y *= 0.8 / np.max(np.abs(y))
    return np.stack([y, y])


# ---------------- click metrics
# A splice click is a discontinuity in the waveform or its slope. Both show up as a spike in the second
# difference (a crude >5 kHz high-pass), which musical content near a splice point does not produce.

def slope_jump(x, a, b):
    """Max derivative jump: max |x[n+1] - 2 x[n] + x[n-1]| in [a, b) (any channel)."""
    return float(np.max(np.abs(np.diff(x[:, a:b], 2, axis=1))))


def hf_rms(x, a, b, win=44):
    """1 ms-window RMS of the 2nd difference in [a, b): (peak dBFS, largest window-to-window rise in dB)."""
    seg = np.diff(x[0, a:b], 2)
    n = seg.size // win
    r = 20 * np.log10(np.sqrt(np.mean(seg[: n * win].reshape(n, win) ** 2, axis=1)) + 1e-9)
    return float(r.max()), float(np.max(np.diff(r)))


def rms_step(x, a, b, win=441, hop=220):
    """Largest |dB change| between consecutive 10 ms broadband RMS windows (5 ms hop) in [a, b)."""
    frames = np.lib.stride_tricks.sliding_window_view(x[0, a:b], win)[::hop]
    r = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9)
    return float(np.max(np.abs(np.diff(r))))


def splice_metrics(y, refs, t, half):
    """Metrics of y around timeline sample t vs the worst of the reference sources [(signal, sample), ...]."""
    def ref(fn):
        return [fn(r, s - half, s + half) for r, s in refs]
    hf = hf_rms(y, t - half, t + half)
    rhf = ref(hf_rms)
    return {
        "slopeJump": slope_jump(y, t - half, t + half), "refSlopeJump": max(ref(slope_jump)),
        "hfPeakDb": hf[0], "refHfPeakDb": max(r[0] for r in rhf),
        "hfRiseDb": hf[1], "refHfRiseDb": max(r[1] for r in rhf),
        "rmsStepDb": rms_step(y, t - half, t + half), "refRmsStepDb": max(ref(rms_step)),
    }
