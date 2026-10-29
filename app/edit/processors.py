"""Lane processor registry ("plugin-style" chain).

A processor is stateless with respect to block boundaries: `wet(x)` receives the lane signal including
`lookbehind()` samples of history and must produce output that is valid for every index >= lookbehind().
That is what makes render_block() gapless and seek-safe (concatenated blocks == full render).

Every processor gets a fade envelope (fadeInMs / fadeOutMs). Default (base class): the envelope crossfades
between bypass (0) and fully processed (1):  y = x + env * (wet(x) - x).
Delay overrides this so the envelope scales only the delayed section:  y = (1-mix) x + env * mix * d.
The envelope ramps in from the start of the lane's content and ramps out ending at content end + tail().

Add a processor:
    @register("reverb")
    class Reverb(BaseProcessor):
        def wet(self, x): ...
"""
from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
from scipy.signal import sosfilt

from ..dsp import tone as tone_dsp
from .model import Processor, Snap

REGISTRY: dict[str, type["BaseProcessor"]] = {}


class ProcessorError(ValueError):
    pass


def register(key: str):
    def deco(cls):
        if key in REGISTRY:
            raise ValueError(f"processor '{key}' already registered")
        cls.type_key = key
        REGISTRY[key] = cls
        return cls
    return deco


def build(spec: Processor, sr: int, snap: Snap) -> "BaseProcessor":
    cls = REGISTRY.get(spec.type)
    if cls is None:
        raise ProcessorError(f"unknown processor type '{spec.type}' (known: {sorted(REGISTRY)})")
    return cls(spec, sr, snap)


def _num(params: dict, key: str, default: float, lo: float, hi: float) -> float:
    v = params.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ProcessorError(f"param '{key}' must be a number")
    if not lo <= v <= hi:
        raise ProcessorError(f"param '{key}'={v} out of range [{lo}, {hi}]")
    return float(v)


class BaseProcessor:
    type_key: ClassVar[str] = ""
    PARAMS: ClassVar[tuple[str, ...]] = ()
    LENIENT: ClassVar[bool] = False  # freestyle tone processors ignore unknown params instead of rejecting them

    def __init__(self, spec: Processor, sr: int, snap: Snap):
        unknown = set(spec.params) - set(self.PARAMS)
        if unknown and not self.LENIENT:
            raise ProcessorError(f"{spec.type}: unknown params {sorted(unknown)}")
        self.spec, self.sr, self.snap = spec, sr, snap
        self.fade_in = int(round(spec.fadeInMs * sr / 1000))
        self.fade_out = int(round(spec.fadeOutMs * sr / 1000))
        self.parse(spec.params)

    def parse(self, params: dict[str, Any]) -> None:  # override
        pass

    def lookbehind(self) -> int:
        return 0

    def tail(self) -> int:
        return 0

    def wet(self, x: np.ndarray) -> np.ndarray:  # override; x is (2, m) float64
        return x

    def envelope(self, t0: int, m: int, span_start: int, span_end: int) -> np.ndarray:
        t = np.arange(t0, t0 + m, dtype=np.float64)
        env = np.ones(m)
        if self.fade_in > 0:
            env *= np.clip((t - span_start + 1) / self.fade_in, 0.0, 1.0)
        if self.fade_out > 0:
            env *= np.clip((span_end + self.tail() - t) / self.fade_out, 0.0, 1.0)
        return env

    def apply(self, x: np.ndarray, t0: int, span_start: int, span_end: int) -> np.ndarray:
        if not self.spec.enabled:
            return x
        w = self.wet(x)
        if self.fade_in == 0 and self.fade_out == 0:
            return w
        env = self.envelope(t0, x.shape[1], span_start, span_end)
        return x + env * (w - x)


@register("delay")
class Delay(BaseProcessor):
    """Sample-accurate feedback delay. out = (1-mix)*x + mix*d,  d[n] = sum_k fb^(k-1) * x[n - k*D].

    params: timeMs (1..4000), feedback (0..0.95), mix (0..1), syncToSnap (bool), beats (>0, in snap units:
    beats when snap.type=='beats', snap.value-second steps when snap.type=='seconds').
    The feedback series is truncated once fb^K < 1e-4 (-80 dB) or K*D > 30 s, which keeps it block-exact.
    """
    PARAMS = ("timeMs", "feedback", "mix", "syncToSnap", "beats")
    MAX_DELAY_S = 4.0
    MAX_TAIL_S = 30.0

    def parse(self, p):
        self.feedback = _num(p, "feedback", 0.0, 0.0, 0.95)
        self.mix = _num(p, "mix", 0.5, 0.0, 1.0)
        self.sync = bool(p.get("syncToSnap", False))
        if self.sync:
            beats = _num(p, "beats", 1.0, 1e-6, 64.0)
            self.delay = delay_samples_synced(beats, self.snap, self.sr)
        else:
            self.delay = int(round(_num(p, "timeMs", 250.0, 1.0, 4000.0) * self.sr / 1000.0))
        if not 1 <= self.delay <= self.MAX_DELAY_S * self.sr:
            raise ProcessorError(f"delay of {self.delay} samples is outside 1..{self.MAX_DELAY_S}s")
        k = 1 if self.feedback == 0 else int(math.ceil(math.log(1e-4) / math.log(self.feedback)))
        self.taps = max(1, min(k, int(self.MAX_TAIL_S * self.sr // self.delay)))

    def lookbehind(self):
        return self.taps * self.delay

    def tail(self):
        return self.taps * self.delay

    def echoes(self, x):
        """The delayed (wet) section only: d[n] = sum_k fb^(k-1) x[n - kD]."""
        m = x.shape[1]
        d = np.zeros_like(x)
        g = 1.0
        for k in range(1, self.taps + 1):
            s = k * self.delay
            if s >= m:
                break
            d[:, s:] += g * x[:, : m - s]
            g *= self.feedback
        return d

    def wet(self, x):
        return (1.0 - self.mix) * x + self.mix * self.echoes(x)

    def apply(self, x, t0, span_start, span_end):
        """The fade envelope scales the delayed section only (output time); the dry part is never faded:
            y = (1 - mix) * x + env(t) * mix * d
        env ramps 0->1 over fadeInMs from the lane's first contributing sample and 1->0 over fadeOutMs ending
        at the end of the delay tail."""
        if not self.spec.enabled:
            return x
        d = self.echoes(x)
        if self.fade_in or self.fade_out:
            d *= self.envelope(t0, x.shape[1], span_start, span_end)
        return (1.0 - self.mix) * x + self.mix * d


def delay_samples_synced(beats: float, snap: Snap, sr: int) -> int:
    """Tempo-sync: `beats` snap units -> integer samples (rounded to nearest)."""
    return int(round(beats * snap.unit_samples(sr)))


@register("eq")
class Eq(BaseProcessor):
    """Wraps app/dsp/tone.py (3-band RBJ EQ + optional tanh drive) to prove the registry is extensible.

    params: lowGainDb/midGainDb/highGainDb (-24..24), lowFreq, midFreq, highFreq, midQ, drive (0..24 dB), driveMix.
    IIR state is rebuilt from `lookbehind` (0.5 s) of history (float64), so blocks match the full render to <1e-9.
    """
    PARAMS = ("lowGainDb", "midGainDb", "highGainDb", "lowFreq", "midFreq", "highFreq", "midQ", "drive", "driveMix")

    def parse(self, p):
        ny = self.sr / 2 * 0.95
        self.kw = dict(low_gain_db=_num(p, "lowGainDb", 0, -24, 24), mid_gain_db=_num(p, "midGainDb", 0, -24, 24),
                       high_gain_db=_num(p, "highGainDb", 0, -24, 24), low_freq=_num(p, "lowFreq", 120, 20, ny),
                       mid_freq=_num(p, "midFreq", 1000, 20, ny), high_freq=_num(p, "highFreq", 6000, 20, ny),
                       mid_q=_num(p, "midQ", 1.0, 0.1, 10))
        self.drive = _num(p, "drive", 0, 0, 24)
        self.drive_mix = _num(p, "driveMix", 1.0, 0, 1)

    def lookbehind(self):
        return int(0.5 * self.sr)

    def tail(self):
        return int(0.05 * self.sr)

    def wet(self, x):
        # Same filters as tone_dsp.eq / tone_dsp.saturate, run in float64 so block renders match the full render.
        k = self.kw
        sos = [tone_dsp._biquad(kind, f, self.sr, g, *q) for kind, f, g, q in (
            ("lowshelf", k["low_freq"], k["low_gain_db"], ()), ("peak", k["mid_freq"], k["mid_gain_db"], (k["mid_q"],)),
            ("highshelf", k["high_freq"], k["high_gain_db"], ())) if g]
        y = sosfilt(np.concatenate(sos).astype(np.float64), x, axis=1) if sos else x.copy()
        if self.drive > 0 and self.drive_mix > 0:
            g = 10 ** (self.drive / 20)
            y = self.drive_mix * np.tanh(g * y) / np.tanh(g) + (1 - self.drive_mix) * y
        return y


# ---------------------------------------------------------------- freestyle tone stages as lane processors
# Lenient like freestyle: values are clamped to app/tone/profile.SPEC ranges (warnings kept on .warnings),
# unknown params are ignored. Each runs with fresh state over `lookbehind()` history => block-exact (~1e-6).
from ..tone import dsp as _tdsp  # noqa: E402
from ..tone.profile import SPEC as _TSPEC, parse_profile as _parse_tone  # noqa: E402


class _ToneStageProcessor(BaseProcessor):
    SECTION: ClassVar[str] = ""
    LENIENT = True
    FORCE: ClassVar[dict] = {}

    def parse(self, p):
        prof = _parse_tone({"mode": "freestyle", "params": {self.SECTION: {"enabled": True, **p, **self.FORCE}}})
        self.cfg = prof.params[self.SECTION]
        self.warnings = list(prof.warnings)
        self._probe = self._make()

    def _make(self):
        return _tdsp.build_stage(self.SECTION, self.cfg, self.sr, self.warnings)

    def lookbehind(self):
        return self._probe.lookbehind() if self._probe else 0

    def tail(self):
        return 0

    def wet(self, x):
        st = self._make()
        return st.process(x) if st else x


def _tone_proc(key, section, force=None):
    cls = type(f"Tone_{key}", (_ToneStageProcessor,), {"SECTION": section, "FORCE": force or {},
                                                         "PARAMS": tuple(_TSPEC[section])})
    return register(key)(cls)


_tone_proc("width", "width")
_tone_proc("compressor", "compressor")
_tone_proc("transient", "transient")
_tone_proc("saturation", "saturation")
_tone_proc("saturation_tape", "saturation", {"type": "tape"})
_tone_proc("saturation_tube", "saturation", {"type": "tube"})
_tone_proc("lowcut", "lowCut")
_tone_proc("highcut", "highCut")


@register("tone")
class ToneFreestyle(BaseProcessor):
    """A whole freestyle tone chain on a lane: params = freestyle `params` object (lowCut..output).
    denoise/pitchTempo are not allowed per lane (they change timing/aren't block-exact) and are ignored."""
    LENIENT = True

    def parse(self, p):
        prof = _parse_tone({"mode": "freestyle", "params": {k: v for k, v in p.items()
                                                            if k not in ("denoise", "pitchTempo")}})
        self.params = prof.params
        self.warnings = list(prof.warnings)
        self._probe = _tdsp.ToneChain(self.params, self.sr)
        self.warnings += self._probe.warnings

    def lookbehind(self):
        return self._probe.lookbehind()

    def wet(self, x):
        return _tdsp.ToneChain(self.params, self.sr).process(x)
