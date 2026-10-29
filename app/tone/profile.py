"""Tone profiles: mode "preset" (strict, narrow ranges, errors) or "freestyle" (any value from the UI, clamped to safe
DSP ranges, with warnings instead of errors; unknown keys are ignored with a warning).

Canonical params (every section is always present after sanitizing; defaults are an exact identity):
    denoise, pitchTempo           pre-steps (run before tone; see app/tone/chain.py)
    lowCut, highCut, eq{bands[]}, compressor, transient, saturation, width, output   (tone, in this order)
`SPEC` below is the single source of truth for ranges/defaults/units: GET /tone/schema serves it, the Kotlin presets
embed a generated copy (schema/kotlin-presets/ToneRanges.kt), and the sanitizer clamps to it.
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = 1
MAX_BANDS = 32           # freestyle
PRESET_MAX_BANDS = 6     # preset mode
MAX_STEMS = 16
STEM_RE = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")
NYQUIST_FRACTION = 0.45  # filter frequencies are additionally clamped to 0.45*sampleRate at render time


def N(lo, hi, default, unit="", step=None, scale="linear", preset=None, desc=""):
    d = {"kind": "number", "min": lo, "max": hi, "default": default, "unit": unit, "scale": scale, "description": desc}
    if step:
        d["step"] = step
    d["presetMin"], d["presetMax"] = preset if preset else (lo, hi)
    return d


def B(default, desc=""):
    return {"kind": "bool", "default": default, "description": desc}


def E(options, default, desc=""):
    return {"kind": "enum", "options": list(options), "default": default, "description": desc}


BAND = {
    "enabled": B(True),
    "type": E(["peak", "lowshelf", "highshelf", "notch"], "peak"),
    "freqHz": N(20, 20000, 1000, "Hz", scale="log", desc="clamped to 0.45*sampleRate at render"),
    "gainDb": N(-30, 30, 0, "dB", 0.1, preset=(-12, 12), desc="ignored by notch"),
    "q": N(0.1, 18, 0.707, "", 0.01, "log", preset=(0.3, 4)),
}

SPEC: dict[str, dict[str, Any]] = {
    "denoise": {"enabled": B(False, "spectral gating (noisereduce), runs first"),
                "preset": E(["hiss", "air", "wind"], "hiss"),
                "strength": N(0, 1, 1, "", 0.01, desc="prop_decrease; preset default when omitted")},
    "pitchTempo": {"enabled": B(False, "Rubber Band, runs after denoise"),
                   "semitones": N(-24, 24, 0, "st", 0.01, preset=(-12, 12)),
                   "tempo": N(0.25, 4, 1, "x", 0.01, "log", preset=(0.5, 2))},
    "lowCut": {"enabled": B(False), "freqHz": N(20, 2000, 80, "Hz", scale="log", preset=(20, 500)),
               "slopeDbOct": N(6, 48, 24, "dB/oct", 6, desc="rounded to a multiple of 6 (Butterworth order 1..8)")},
    "highCut": {"enabled": B(False), "freqHz": N(1000, 20000, 16000, "Hz", scale="log", preset=(3000, 20000)),
                "slopeDbOct": N(6, 48, 24, "dB/oct", 6)},
    "eq": {"bands": {"kind": "list", "maxItems": MAX_BANDS, "presetMaxItems": PRESET_MAX_BANDS, "item": BAND,
                     "default": []}},
    "compressor": {"enabled": B(False),
                   "thresholdDb": N(-60, 0, -18, "dB", 0.1, preset=(-40, 0)),
                   "ratio": N(1, 20, 4, ":1", 0.1, "log", preset=(1, 8)),
                   "attackMs": N(0.1, 200, 10, "ms", 0.1, "log", preset=(1, 100)),
                   "releaseMs": N(5, 2000, 150, "ms", 1, "log", preset=(20, 1000)),
                   "kneeDb": N(0, 24, 6, "dB", 0.5),
                   "makeupDb": N(0, 24, 0, "dB", 0.1, preset=(0, 12))},
    "transient": {"enabled": B(False), "attackDb": N(-12, 12, 0, "dB", 0.1, preset=(-6, 6)),
                  "sustainDb": N(-12, 12, 0, "dB", 0.1, preset=(-6, 6))},
    "saturation": {"enabled": B(False), "type": E(["tanh", "tape", "tube"], "tanh"),
                   "driveDb": N(0, 36, 0, "dB", 0.1, preset=(0, 18)),
                   "mix": N(0, 1, 1, "", 0.01),
                   "asymmetry": N(0, 1, 0, "", 0.01, desc="even harmonics; tube adds a fixed 0.15 bias")},
    "width": {"enabled": B(False), "widthPct": N(0, 200, 100, "%", 1, preset=(50, 150),
                                                 desc="0 = mono, 100 = unchanged, 200 = double side"),
              "monoBelowHz": N(0, 500, 120, "Hz", 1,
                               desc="below this the side signal is never widened (bass stays mono-safe); 0 = off")},
    "output": {"gainDb": N(-36, 12, 0, "dB", 0.1, preset=(-24, 6))},
}
TONE_ORDER = ("lowCut", "highCut", "eq", "compressor", "transient", "saturation", "width", "output")
CHAIN_ORDER = ("denoise", "pitchTempo", "tone", "laneBus")


class ProfileError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def defaults(spec=None) -> dict:
    spec = SPEC if spec is None else spec
    return {k: ({kk: copy.deepcopy(vv["default"]) for kk, vv in v.items()} if "kind" not in v else
                copy.deepcopy(v["default"])) for k, v in spec.items()}


class _Ctx:
    def __init__(self, strict: bool):
        self.strict, self.warnings, self.errors = strict, [], []

    def bad(self, msg):
        (self.errors if self.strict else self.warnings).append(msg)


def _value(sp: dict, raw: Any, path: str, ctx: _Ctx):
    k = sp["kind"]
    if k == "bool":
        if isinstance(raw, bool):
            return raw
        if raw in (0, 1, "true", "false", "0", "1"):
            ctx.warnings.append(f"{path}: coerced {raw!r} to bool")
            return raw in (1, "true", "1")
        ctx.bad(f"{path}: expected bool, got {raw!r}; using {sp['default']}")
        return sp["default"]
    if k == "enum":
        if isinstance(raw, str) and raw.lower() in sp["options"]:
            return raw.lower()
        ctx.bad(f"{path}: {raw!r} not in {sp['options']}; using {sp['default']!r}")
        return sp["default"]
    # number
    v = raw
    if isinstance(v, str):
        try:
            v = float(v.strip())
        except ValueError:
            v = None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        ctx.bad(f"{path}: {raw!r} is not a finite number; using default {sp['default']}")
        return sp["default"]
    lo, hi = (sp["presetMin"], sp["presetMax"]) if ctx.strict else (sp["min"], sp["max"])
    if v < lo or v > hi:
        c = min(max(v, lo), hi)
        if ctx.strict:
            ctx.errors.append(f"{path}: {v} outside preset range [{lo}, {hi}] (use mode=freestyle to unlock)")
        else:
            ctx.warnings.append(f"{path}: {v} clamped to {c} (safe range [{lo}, {hi}])")
        v = c
    return float(v)


def _object(spec: dict, raw: Any, base: dict, path: str, ctx: _Ctx) -> dict:
    out = copy.deepcopy(base)
    if raw is None:
        return out
    if not isinstance(raw, dict):
        ctx.bad(f"{path}: expected an object, got {type(raw).__name__}; ignored")
        return out
    for key, val in raw.items():
        sp = spec.get(key)
        p = f"{path}.{key}" if path else key
        if sp is None:
            ctx.warnings.append(f"{p}: unknown key ignored")
            continue
        if "kind" not in sp:            # nested section
            out[key] = _object(sp, val, out[key], p, ctx)
        elif sp["kind"] == "list":
            out[key] = _list(sp, val, p, ctx)
        else:
            out[key] = _value(sp, val, p, ctx)
    return out


def _list(sp, raw, path, ctx):
    if not isinstance(raw, list):
        ctx.bad(f"{path}: expected a list; ignored")
        return []
    cap = sp["presetMaxItems"] if ctx.strict else sp["maxItems"]
    if len(raw) > cap:
        ctx.bad(f"{path}: {len(raw)} items, only the first {cap} are used")
        raw = raw[:cap]
    item_defaults = {k: v["default"] for k, v in sp["item"].items()}
    return [_object(sp["item"], r, item_defaults, f"{path}[{i}]", ctx) for i, r in enumerate(raw)]


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


@dataclass
class ToneProfile:
    mode: str = "freestyle"
    params: dict = field(default_factory=defaults)
    stems: dict[str, dict] = field(default_factory=dict)   # stem name -> FULL canonical params (base + override)
    warnings: list[str] = field(default_factory=list)
    name: Optional[str] = None

    def for_stem(self, stem: Optional[str]) -> dict:
        return self.stems.get(stem, self.params) if stem else self.params

    def to_json(self) -> dict:
        return {"version": SCHEMA_VERSION, "mode": self.mode, "name": self.name, "params": self.params,
                "stems": self.stems}


ROOT_KEYS = {"version", "schemaVersion", "mode", "params", "stems", "name", "id", "createdAt", "updatedAt"}


def parse_profile(raw: Any) -> ToneProfile:
    """Accepts {mode, params, stems?} (also a phone FreestylePreset export, whose extra metadata keys are allowed).
    Freestyle: never raises for values; preset: raises ProfileError listing every violation."""
    if not isinstance(raw, dict):
        raise ProfileError(["toneProfile must be a JSON object"])
    mode = raw.get("mode", "freestyle")
    if mode not in ("preset", "freestyle"):
        raise ProfileError([f"mode must be preset|freestyle, got {mode!r}"])
    ctx = _Ctx(strict=(mode == "preset"))
    for k in raw:
        if k not in ROOT_KEYS:
            ctx.warnings.append(f"{k}: unknown key ignored")
    v = raw.get("version", raw.get("schemaVersion", SCHEMA_VERSION))
    if v != SCHEMA_VERSION:
        ctx.warnings.append(f"version {v!r} != {SCHEMA_VERSION}; interpreted as v{SCHEMA_VERSION}")
    params_raw = raw.get("params") or {}
    params = _object(SPEC, params_raw, defaults(), "params", ctx)
    stems = {}
    sraw = raw.get("stems") or {}
    if not isinstance(sraw, dict):
        ctx.bad("stems: expected an object keyed by stem name; ignored")
        sraw = {}
    for i, (name, ov) in enumerate(sraw.items()):
        if i >= MAX_STEMS or not STEM_RE.match(str(name)):
            ctx.bad(f"stems.{name}: invalid stem name or more than {MAX_STEMS} stems; ignored")
            continue
        merged = _merge(params_raw if isinstance(params_raw, dict) else {}, ov if isinstance(ov, dict) else {})
        sub = _Ctx(ctx.strict)
        stems[name] = _object(SPEC, merged, defaults(), f"stems.{name}", sub)
        base_msgs = {m.replace("params.", f"stems.{name}.", 1) for m in ctx.warnings + ctx.errors}
        ctx.warnings += [m for m in sub.warnings if m not in base_msgs]
        ctx.errors += [m for m in sub.errors if m not in base_msgs]
    if ctx.errors:
        raise ProfileError(ctx.errors)
    nm = raw.get("name")
    return ToneProfile(mode, params, stems, list(dict.fromkeys(ctx.warnings)), nm if isinstance(nm, str) else None)


def legacy_profile(low=0.0, mid=0.0, high=0.0, low_freq=120.0, mid_freq=1000.0, high_freq=6000.0, mid_q=1.0,
                   drive=0.0, mix=0.0) -> dict:
    """The old /process/tone form fields expressed as a freestyle body (for documentation/clients)."""
    return {"mode": "freestyle", "params": {
        "eq": {"bands": [{"type": "lowshelf", "freqHz": low_freq, "gainDb": low},
                         {"type": "peak", "freqHz": mid_freq, "gainDb": mid, "q": mid_q},
                         {"type": "highshelf", "freqHz": high_freq, "gainDb": high}]},
        "saturation": {"enabled": drive > 0 and mix > 0, "driveDb": drive, "mix": mix}}}


def schema_doc() -> dict:
    return {"version": SCHEMA_VERSION, "chainOrder": list(CHAIN_ORDER), "toneOrder": list(TONE_ORDER),
            "nyquistFraction": NYQUIST_FRACTION, "maxStems": MAX_STEMS, "stemNamePattern": STEM_RE.pattern,
            "modes": {"freestyle": "values clamped to min/max with warnings; unknown keys ignored with warnings",
                      "preset": "values must be within presetMin/presetMax; violations are 422 errors"},
            "sections": SPEC, "defaults": defaults()}
