"""Generate schema/kotlin-dsp/test_vectors.json from the Python engine (the reference implementation).

The Kotlin reference DSP in schema/kotlin-dsp/ must reproduce these numbers (EditDspTest.kt checks them).
    python scripts/gen_kotlin_vectors.py [--check]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.edit.engine import CEILING, Renderer, curves, db_to_gain, item_geom  # noqa: E402
from app.edit.model import LaneItem, Processor, Project, Snap  # noqa: E402
from app.edit.processors import Delay, delay_samples_synced  # noqa: E402

OUT = ROOT / "schema" / "kotlin-dsp" / "test_vectors.json"
SR = 1000  # tiny sample rate keeps vectors short and human-checkable


def r(a, nd=9):
    return [round(float(v), nd) for v in np.asarray(a).ravel()]


def crossfade_vectors():
    w = np.linspace(0, 1, 11)
    out = []
    for curve in ("equal_power", "linear"):
        go, gi = curves(w, curve)
        out.append({"curve": curve, "w": r(w), "gainOut": r(go), "gainIn": r(gi)})
    return out


def item_vectors():
    """Weight (crossfade position) and envelope per timeline sample for a few items; plus a lane mix."""
    rng = np.random.default_rng(5)
    old = r(rng.uniform(-0.5, 0.5, 160), 6)
    new = r(rng.uniform(-0.5, 0.5, 160), 6)
    cases = []
    for mode in ("replace", "add"):
        for curve in ("equal_power", "linear"):
            it = LaneItem.model_validate({"id": "b", "sourceClipId": "B", "range": {"startSample": 20, "endSample": 100},
                                          "laneStartSample": 40, "mode": mode, "gainDb": -3, "fadeInMs": 10,
                                          "fadeOutMs": 20, "crossfade": {"lengthMs": 50, "curve": curve}})
            p = Project.model_validate({"id": "v", "sampleRate": 8000, "clips": [{"id": "A"}, {"id": "B"}], "lanes": [{
                "id": "l", "items": [{"id": "a", "sourceClipId": "A", "range": {"startSample": 0, "endSample": 160},
                                      "laneStartSample": 0, "crossfade": {"lengthMs": 50, "curve": curve}},
                                     it.model_dump()]}]}).model_copy(update={"sampleRate": SR})  # 1 kHz: short vectors
            rend = Renderer(p, {"A": np.array([old, old]), "B": np.array([new, new])})
            g = item_geom(it, SR)
            t = np.arange(0, 180, dtype=np.float64)
            lane = rend.compose(rend.lanes[0], 0, 180)[0]
            cases.append({"mode": mode, "curve": curve, "sampleRate": SR, "item": it.model_dump(mode="json"),
                          "geom": {"P": g.P, "L": g.L, "X": g.X, "h": g.h, "fadeIn": g.fi, "fadeOut": g.fo,
                                   "start": g.start, "end": g.end, "gain": round(g.gain, 12)},
                          "weight": r(g.weight(t)), "envelope": r(g.envelope(t)),
                          "underlay": {"clip": "A", "range": [0, 160], "laneStartSample": 0},
                          "sourceA": old, "sourceB": new, "laneOut": r(lane)})
    return cases


def delay_vectors():
    out = []
    for params, fi, fo in (({"timeMs": 7, "feedback": 0.5, "mix": 0.5}, 0, 0),
                           ({"timeMs": 5, "feedback": 0.7, "mix": 1.0}, 40, 30),
                           ({"syncToSnap": True, "beats": 0.5, "feedback": 0.0, "mix": 0.3}, 10, 0)):
        snap = Snap(type="beats", value=0.05, bpm=300)  # 1 snap unit = 0.05 beat at 300 bpm = 10 samples at 1 kHz
        d = Delay(Processor(id="d", type="delay", params=params, fadeInMs=fi, fadeOutMs=fo), SR, snap)
        x = np.zeros((2, 200))
        x[:, 3] = 1.0
        x[:, 50:55] = [0.2, -0.4, 0.6, -0.4, 0.2]
        span = (0, 60)
        y = d.apply(x, 0, *span)
        out.append({"params": params, "fadeInMs": fi, "fadeOutMs": fo, "sampleRate": SR,
                    "snap": snap.model_dump(), "delaySamples": d.delay, "taps": d.taps, "tail": d.tail(),
                    "spanStart": span[0], "spanEnd": span[1], "input": r(x[0]), "output": r(y[0])})
    return out


def bus_vectors():
    out = []
    x = np.array([[0.5, -0.25, 0.1], [0.4, -0.2, 0.3]])
    for gain_db, pan in ((0, 0), (-6, 0), (0, -1), (0, 1), (3, 0.5), (-12, -0.3)):
        gl = np.cos((pan + 1) * np.pi / 4) * np.sqrt(2) * db_to_gain(gain_db)
        gr = np.sin((pan + 1) * np.pi / 4) * np.sqrt(2) * db_to_gain(gain_db)
        out.append({"gainDb": gain_db, "pan": pan, "gainL": round(gl, 12), "gainR": round(gr, 12),
                    "inL": r(x[0]), "inR": r(x[1]), "outL": r(x[0] * gl), "outR": r(x[1] * gr)})
    solo = [{"lanes": [{"mute": False, "solo": True}, {"mute": True, "solo": True}, {"mute": False, "solo": False}],
             "audible": [True, False, False]},
            {"lanes": [{"mute": False, "solo": False}, {"mute": True, "solo": False}], "audible": [True, False]}]
    return {"gainPan": out, "soloMute": solo}


def softclip_vectors():
    T = db_to_gain(-1.0)
    x = np.array([0.0, 0.5, T, 0.9, 0.95, 1.0, 1.5, 3.0, -0.95, -2.0])
    a = np.abs(x)
    y = np.where(a > T, np.sign(x) * (T + (CEILING - T) * np.tanh((a - T) / (CEILING - T))), x)
    return {"thresholdDb": -1.0, "threshold": T, "ceiling": CEILING, "input": r(x), "output": r(y)}


def tempo_vectors():
    out = []
    for snap, beats in (({"type": "beats", "value": 1, "bpm": 120}, 1), ({"type": "beats", "value": 1, "bpm": 97}, 0.5),
                        ({"type": "beats", "value": 0.5, "bpm": 90}, 3), ({"type": "seconds", "value": 0.1}, 3)):
        out.append({"snap": snap, "beats": beats, "sampleRate": 44100,
                    "delaySamples": delay_samples_synced(beats, Snap.model_validate(snap), 44100)})
    return out


def build():
    return {"generatedBy": "scripts/gen_kotlin_vectors.py", "tolerance": 1e-6,
            "crossfadeCurves": crossfade_vectors(), "items": item_vectors(), "delay": delay_vectors(),
            "laneBus": bus_vectors(), "softClip": softclip_vectors(), "tempoSync": tempo_vectors()}


def text():
    return json.dumps(build(), indent=1) + "\n"


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(0 if OUT.exists() and OUT.read_text() == text() else 1)
    OUT.write_text(text())
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
