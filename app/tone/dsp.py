"""Freestyle tone DSP. Every stage is a small object with explicit filter state, so the same code serves:
  * streaming file->file rendering (state carried block to block: exact, constant memory), and
  * the edit-lane registry (fresh state + `lookbehind()` samples of history: block-exact to ~1e-6).
All math in float64 on (channels, n) blocks. Defaults/disabled stages are skipped => flat profile == exact identity.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from ..dsp.tone import _biquad
from .profile import NYQUIST_FRACTION, TONE_ORDER


def _fmax(sr):
    return NYQUIST_FRACTION * sr


def _clampf(f, sr, path, warns):
    hi = _fmax(sr)
    if f > hi:
        warns.append(f"{path}: {f:g} Hz clamped to {hi:g} Hz (0.45*sampleRate)")
        return hi
    return f


def _coef(ms, sr):
    """One-pole coefficient for a time constant in ms."""
    return math.exp(-1000.0 / (max(ms, 1e-3) * sr))


def _onepole(x, a, zi):
    """y[n] = (1-a) x[n] + a y[n-1] along the last axis. zi shape (..., 1) or None. Returns y, zf."""
    if zi is None:
        zi = np.zeros(x.shape[:-1] + (1,))
    return lfilter([1 - a], [1, -a], x, axis=-1, zi=zi)


class Stage:
    name = ""

    def lookbehind(self) -> int:
        return 0

    def process(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SosStage(Stage):
    def __init__(self, sos, lb):
        self.sos, self.zi, self.lb = np.asarray(sos, np.float64), None, lb

    def lookbehind(self):
        return self.lb

    def process(self, x):
        if self.zi is None:
            self.zi = np.zeros((self.sos.shape[0], x.shape[0], 2))
        y, self.zi = sosfilt(self.sos, x, axis=-1, zi=self.zi)
        return y


def _iir_lookbehind(sr, fmin, order=2):
    # ringing of an order-n IIR at fmin dies to < 1e-7 within roughly 6*order cycles
    return int(sr * min(2.0, max(0.05, 6.0 * max(order, 2) / max(fmin, 20.0))))


def _notch(f0, sr, q):
    w0 = 2 * np.pi * f0 / sr
    cw, alpha = np.cos(w0), np.sin(w0) / (2 * q)
    b, a = np.array([1, -2 * cw, 1]), np.array([1 + alpha, -2 * cw, 1 - alpha])
    return np.concatenate([b / a[0], a / a[0]])[None, :]


def cut_stage(cfg, sr, kind, warns):
    if not cfg["enabled"]:
        return None
    f = _clampf(cfg["freqHz"], sr, f"{kind}.freqHz", warns)
    order = int(min(8, max(1, round(cfg["slopeDbOct"] / 6))))
    sos = butter(order, f, btype="highpass" if kind == "lowCut" else "lowpass", fs=sr, output="sos")
    s = SosStage(sos, _iir_lookbehind(sr, f, order))
    s.name = kind
    return s


def eq_stage(cfg, sr, warns):
    rows, fmin = [], 1e9
    for i, b in enumerate(cfg["bands"]):
        if not b["enabled"] or (b["type"] != "notch" and b["gainDb"] == 0):
            continue
        f = _clampf(b["freqHz"], sr, f"eq.bands[{i}].freqHz", warns)
        rows.append(_notch(f, sr, b["q"]) if b["type"] == "notch" else _biquad(b["type"], f, sr, b["gainDb"], b["q"]))
        fmin = min(fmin, f / max(b["q"], 0.5) if b["type"] in ("peak", "notch") else f)
    if not rows:
        return None
    s = SosStage(np.concatenate(rows), _iir_lookbehind(sr, fmin, 2) * 2)
    s.name = "eq"
    return s


class Compressor(Stage):
    """Stereo-linked feed-forward compressor. Detector: one-pole mean-square of max-channel power (attack time
    constant); static curve with soft knee; release: gain reduction G = max(gr, onepole_release(gr)).
    Only linear filters carry state, so history converges exponentially (lookbehind = 14 time constants)."""
    name = "compressor"

    def __init__(self, c, sr):
        self.sr, self.t, self.r, self.k, self.mk = sr, c["thresholdDb"], c["ratio"], c["kneeDb"], c["makeupDb"]
        self.aa, self.ar = _coef(c["attackMs"], sr), _coef(c["releaseMs"], sr)
        self.za = self.zr = None
        self.lb = int(14 * (c["attackMs"] + c["releaseMs"]) * sr / 1000)

    def lookbehind(self):
        return self.lb

    def gain_db(self, x):
        p = np.max(x * x, axis=0)
        ms, self.za = _onepole(p, self.aa, self.za)
        lvl = 10 * np.log10(np.maximum(ms, 1e-12))
        over, s, k = lvl - self.t, 1 - 1 / self.r, self.k
        if k > 0:
            gr = np.where(over <= -k / 2, 0.0, np.where(over >= k / 2, s * over, s * (over + k / 2) ** 2 / (2 * k)))
        else:
            gr = np.maximum(over, 0) * s
        rel, self.zr = _onepole(gr, self.ar, self.zr)
        return self.mk - np.maximum(gr, rel)

    def process(self, x):
        return x * 10 ** (self.gain_db(x) / 20)


class Transient(Stage):
    """Envelope-difference transient shaper: fast (1 ms) vs slow (40 ms) envelopes of the max-abs channel.
    attackDb applies where fast > slow (onsets), sustainDb where slow > fast (tails). Gain is bounded by +/-12 dB."""
    name = "transient"

    def __init__(self, c, sr):
        self.at, self.su = c["attackDb"], c["sustainDb"]
        self.af, self.asl, self.ag = _coef(1.0, sr), _coef(40.0, sr), _coef(2.0, sr)
        self.z = [None, None, None]
        self.lb = int(14 * 43 * sr / 1000)

    def lookbehind(self):
        return self.lb

    def process(self, x):
        m = np.max(np.abs(x), axis=0)
        ef, self.z[0] = _onepole(m, self.af, self.z[0])
        es, self.z[1] = _onepole(m, self.asl, self.z[1])
        d = (ef - es) / (ef + es + 1e-9)
        g = self.at * np.clip(2 * d, 0, 1) + self.su * np.clip(-2 * d, 0, 1)
        g, self.z[2] = _onepole(g, self.ag, self.z[2])
        return x * 10 ** (np.clip(g, -12, 12) / 20)


SAT_FUNCS = {"tanh": np.tanh, "tape": lambda u: (2 / np.pi) * np.arctan((np.pi / 2) * u), "tube": np.tanh}


class Saturation(Stage):
    """wet = (f(g*x + b) - f(b)) / (f(g + b) - f(b)): unity at full scale, zero at zero. b = 0.4*asymmetry
    (+0.15 for tube) adds even harmonics; the DC this creates is removed by a 5 Hz one-pole high-pass on the wet."""
    name = "saturation"

    def __init__(self, c, sr):
        self.f = SAT_FUNCS[c["type"]]
        self.g = 10 ** (c["driveDb"] / 20)
        self.b = 0.4 * c["asymmetry"] + (0.15 if c["type"] == "tube" else 0.0)
        self.mix = c["mix"]
        self.norm = float(self.f(self.g + self.b) - self.f(self.b))
        self.adc, self.zx, self.zy = _coef(1000 / (2 * np.pi * 5), sr), None, None
        self.lb = int(14 * 1000 / (2 * np.pi * 5) * sr / 1000) if self.b else 0

    def lookbehind(self):
        return self.lb

    def process(self, x):
        w = (self.f(self.g * x + self.b) - self.f(self.b)) / self.norm
        if self.b:   # DC blocker y = w - lowpass(w)
            lp, self.zx = _onepole(w, self.adc, self.zx)
            w = w - lp
        return x + self.mix * (w - x)


class Width(Stage):
    """Mid/side. w <= 1: S' = w*S (w=0 -> exact mono, w=1 -> exact identity).
    w > 1 with monoBelowHz > 0: Linkwitz-Riley 4th-order split of S at monoBelowHz, S' = LP(S) + w*HP(S), so bass
    is never widened (LP+HP is flat/allpass; the side gets the LR4 phase shift only when widening).
    Mono input passes through."""
    name = "width"

    def __init__(self, c, sr, warns):
        self.w = c["widthPct"] / 100.0
        f = c["monoBelowHz"]
        self.lp = None
        if f > 0 and self.w > 1:
            f = _clampf(f, sr, "width.monoBelowHz", warns)
            lp, hp = (butter(2, f, k, fs=sr, output="sos") for k in ("lowpass", "highpass"))
            lb = _iir_lookbehind(sr, f, 4)
            self.lp, self.hp = SosStage(np.concatenate([lp, lp]), lb), SosStage(np.concatenate([hp, hp]), lb)

    def lookbehind(self):
        return self.lp.lb if self.lp else 0

    def process(self, x):
        if x.shape[0] < 2:
            return x
        m, s = (x[0] + x[1]) / 2, (x[0] - x[1]) / 2
        if self.lp is not None:
            s = self.lp.process(s[None, :])[0] + self.w * self.hp.process(s[None, :])[0]
        else:
            s = self.w * s
        return np.stack([m + s, m - s] + list(x[2:]))


class Gain(Stage):
    name = "output"

    def __init__(self, db):
        self.g = 10 ** (db / 20)

    def process(self, x):
        return x * self.g


def build_stage(section: str, cfg: dict, sr: int, warns: list) -> Stage | None:
    if section in ("lowCut", "highCut"):
        return cut_stage(cfg, sr, section, warns)
    if section == "eq":
        return eq_stage(cfg, sr, warns)
    if section == "compressor":
        return Compressor(cfg, sr) if cfg["enabled"] else None
    if section == "transient":
        return Transient(cfg, sr) if cfg["enabled"] and (cfg["attackDb"] or cfg["sustainDb"]) else None
    if section == "saturation":
        return Saturation(cfg, sr) if cfg["enabled"] and cfg["driveDb"] > 0 and cfg["mix"] > 0 else None
    if section == "width":
        return Width(cfg, sr, warns) if cfg["enabled"] and cfg["widthPct"] != 100 else None
    if section == "output":
        return Gain(cfg["gainDb"]) if cfg["gainDb"] != 0 else None
    raise KeyError(section)


class ToneChain:
    """Stateful freestyle tone chain (TONE_ORDER). process() on consecutive blocks == process() on the whole."""

    def __init__(self, params: dict, sr: int):
        self.warnings: list[str] = []
        self.stages = [s for sec in TONE_ORDER if (s := build_stage(sec, params[sec], sr, self.warnings))]

    @property
    def names(self):
        return [s.name for s in self.stages]

    def lookbehind(self):
        return sum(s.lookbehind() for s in self.stages)

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, np.float64)
        for s in self.stages:
            y = s.process(y)
        if not np.all(np.isfinite(y)):  # cannot happen with clamped params; last-resort guard
            self.warnings.append("non-finite samples replaced with 0")
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return y
