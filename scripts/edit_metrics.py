"""Measure splice click metrics on synthetic drums + plucks (prints a table; used for the report)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.edit.engine import Renderer  # noqa: E402
from app.edit.model import Project  # noqa: E402
from tests.edit_signals import SR, drums_and_plucks, plucks, splice_metrics  # noqa: E402

A, B = drums_and_plucks(), plucks()
P, RS, L = int(2.375 * SR), int(1.25 * SR), int(1.5 * SR)


def project(curve="equal_power", ms=75.0):
    return Project.model_validate({
        "id": "p", "sampleRate": SR, "clips": [{"id": "A"}, {"id": "B"}],
        "lanes": [{"id": "l1", "items": [
            {"id": "a", "sourceClipId": "A", "range": {"startSample": 0, "endSample": A.shape[1]}, "laneStartSample": 0},
            {"id": "b", "sourceClipId": "B", "range": {"startSample": RS, "endSample": RS + L}, "laneStartSample": P,
             "mode": "replace", "crossfade": {"lengthMs": ms, "curve": curve}}]}]})


def hard_splice():
    y = A.copy()
    y[:, P:P + L] = B[:, RS:RS + L]
    return y


def metrics(y, half=int(0.06 * SR)):
    return [dict(splice=name, **splice_metrics(y, [(A, t), (B, tb)], t, half))
            for name, t, tb in (("start", P, RS), ("end", P + L, RS + L))]


def run():
    out = {"hard_cut": metrics(hard_splice())}
    for curve in ("equal_power", "linear"):
        for ms in (50.0, 75.0, 100.0):
            out[f"{curve}_{int(ms)}ms"] = metrics(Renderer(project(curve, ms), {"A": A, "B": B}).render("main_plus_lanes"))
    return out


if __name__ == "__main__":
    for k, rows in run().items():
        for r in rows:
            print(f"{k:18s} {r['splice']:5s} slopeJump={r['slopeJump']:.4f} (ref {r['refSlopeJump']:.4f}, "
                  f"x{r['slopeJump'] / r['refSlopeJump']:.2f})  hfPeak={r['hfPeakDb']:.1f} dB (ref {r['refHfPeakDb']:.1f})  "
                  f"hfRise={r['hfRiseDb']:.1f} dB (ref {r['refHfRiseDb']:.1f})  rmsStep={r['rmsStepDb']:.2f} dB "
                  f"(ref {r['refRmsStepDb']:.2f})")
