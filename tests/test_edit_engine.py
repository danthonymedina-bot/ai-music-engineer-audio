"""Edit model + engine: splices, add/replace, summing, delay, tempo sync, gapless blocks, undo."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from app.edit import history as H
from app.edit.engine import Renderer, extract_slice, item_geom, render, render_block
from app.edit.model import LaneItem, Project
from app.edit.processors import REGISTRY, BaseProcessor, ProcessorError, delay_samples_synced, register
from app.edit.model import Snap
from tests.edit_signals import SR, drums_and_plucks, plucks, splice_metrics

ROOT = Path(__file__).resolve().parents[1]
A, B = drums_and_plucks(), plucks()


def item(id, clip, start, end, at, mode="add", **kw):
    return {"id": id, "sourceClipId": clip, "range": {"startSample": start, "endSample": end},
            "laneStartSample": at, "mode": mode, **kw}


def proj(lanes, clips=("A", "B"), **kw):
    return Project.model_validate({"id": "p", "sampleRate": SR, "clips": [{"id": c} for c in clips],
                                   "lanes": lanes, **kw})


def dc(n, v):
    return np.full((2, n), v)


# ---------------------------------------------------------------- splices / clicks
P, RS, L = int(2.375 * SR), int(1.25 * SR), int(1.5 * SR)
HALF = int(0.06 * SR)


def splice_project(curve, ms):
    return proj([{"id": "l1", "items": [
        item("a", "A", 0, A.shape[1], 0),
        item("b", "B", RS, RS + L, P, "replace", crossfade={"lengthMs": ms, "curve": curve})]}])


def _metrics(y):
    return [splice_metrics(y, [(A, t), (B, tb)], t, HALF) for t, tb in ((P, RS), (P + L, RS + L))]


@pytest.mark.parametrize("curve", ["equal_power", "linear"])
@pytest.mark.parametrize("ms", [50.0, 75.0, 100.0])
def test_replace_splices_are_click_free(curve, ms):
    y = Renderer(splice_project(curve, ms), {"A": A, "B": B}).render()
    for m in _metrics(y):
        assert m["slopeJump"] <= 1.05 * m["refSlopeJump"], m
        assert m["hfPeakDb"] <= m["refHfPeakDb"] + 1.0, m
        assert m["hfRiseDb"] <= m["refHfRiseDb"] + 1.0, m
        assert m["rmsStepDb"] <= m["refRmsStepDb"] + 0.5, m


def test_metrics_detect_a_hard_cut():
    """Sanity check that the metrics above would catch a click."""
    y = A.copy()
    y[:, P:P + L] = B[:, RS:RS + L]
    for m in _metrics(y):
        assert m["slopeJump"] > 5 * m["refSlopeJump"]
        assert m["hfPeakDb"] > m["refHfPeakDb"] + 10


@pytest.mark.parametrize("mode", ["add", "replace"])
def test_split_is_lossless(mode):
    """Contiguous seam rule: splitting mid-note leaves the render bit-identical (no crossfade inside a take)."""
    base = proj([{"id": "l1", "items": [item("a", "A", 0, A.shape[1], 0),
                                        item("b", "B", RS, RS + L, P, mode, fadeInMs=30, fadeOutMs=40)]}])
    ref = Renderer(base, {"A": A, "B": B}).render()
    p = H.split_item(base, "b", P + L // 2 + 17, "b2")
    p = H.split_item(p, "b2", P + L // 2 + 4000, "b3")
    assert len(p.lanes[0].items) == 4
    assert np.array_equal(Renderer(p, {"A": A, "B": B}).render(), ref)


def test_add_from_silence_is_click_free():
    p = proj([{"id": "l1", "items": [item("b", "B", RS, RS + L, P)]}])
    y = Renderer(p, {"A": A, "B": B}).render()
    g = item_geom(p.lanes[0].items[0], SR)
    assert np.all(y[:, : g.P - g.h] == 0) and np.any(y[:, g.P - g.h] != 0)  # silent until the centred crossfade starts
    for t, tb in ((P, RS), (P + L, RS + L)):
        m = splice_metrics(y, [(B, tb)], t, HALF)
        assert m["slopeJump"] <= 1.05 * m["refSlopeJump"] and m["hfPeakDb"] <= m["refHfPeakDb"] + 1.0, m


# ---------------------------------------------------------------- exact lengths / add vs replace

def test_exact_slice_lengths_and_split():
    p = proj([{"id": "l1", "items": [item("a", "A", 1234, 1234 + 50_001, 777)]}])
    it = p.lanes[0].items[0]
    assert extract_slice(A, it).shape == (2, 50_001)
    p2 = H.split_item(p, "a", 777 + 20_000, "a2")
    l, r = p2.lanes[0].items
    assert (l.range.length, r.range.length) == (20_000, 30_001)
    assert l.range.endSample == r.range.startSample and r.laneStartSample == l.laneEndSample


@pytest.mark.parametrize("ms", [50.0, 75.0, 100.0])
def test_add_contributes_exactly_range_length_samples(ms):
    """With linear crossfades the weights integrate to exactly L samples, wherever the crossfade is centred."""
    n, L2, at = 100_000, 33_333, 20_000
    src = dc(n, 0.25)
    p = proj([{"id": "l1", "items": [item("x", "A", 30_000, 30_000 + L2, at,
                                         crossfade={"lengthMs": ms, "curve": "linear"})]}], clips=("A",))
    y = Renderer(p, {"A": src}).render().astype(np.float64)
    assert np.sum(y[0]) / 0.25 == pytest.approx(L2, abs=1e-2)
    g = item_geom(p.lanes[0].items[0], SR)
    assert np.all(y[:, g.P - g.h + g.X: g.P + g.L - g.h] == np.float32(0.25))  # full weight in the body
    assert np.all(y[:, :g.P - g.h] == 0) and np.all(y[:, g.P + g.L - g.h + g.X:] == 0)


@pytest.mark.parametrize("curve", ["linear", "equal_power"])
def test_add_vs_replace(curve):
    n, s, L2 = 200_000, 60_000, 50_000
    srcs = {"A": dc(n, 0.5), "B": dc(n, 0.25)}
    def run(mode):
        p = proj([{"id": "l1", "items": [item("a", "A", 0, n, 0),
                                         item("b", "B", 0, L2, s, mode, crossfade={"curve": curve})]}])
        return Renderer(p, srcs).render()[0].astype(np.float64), item_geom(p.lanes[0].items[1], SR)
    add, g = run("add")
    rep, _ = run("replace")
    body = slice(g.P - g.h + g.X, g.P + g.L - g.h)
    assert np.allclose(add[body], 0.75) and np.allclose(rep[body], 0.25)
    assert np.allclose(add[10_000:g.P - g.h], 0.5) and np.allclose(rep[10_000:g.P - g.h], 0.5)
    assert np.allclose(rep[g.P + g.L - g.h + g.X: n - 10_000], 0.5)
    # crossfade is centred on the boundary: half-way point of the start window is the boundary sample
    w = g.weight(np.array([g.P - 0.5]))
    assert w[0] == pytest.approx(0.5, abs=1 / g.X)
    # replace never exceeds the louder of old/new with linear; equal power peaks at 1/sqrt2*(0.5+0.25)*... bounded
    assert rep.max() <= (0.5 if curve == "linear" else 0.5 * np.sqrt(2)) + 1e-9


def test_item_fades_are_independent_of_lane_gain():
    n = 100_000
    def run(gain):
        p = proj([{"id": "l1", "gainDb": gain, "items": [item("a", "A", 0, 50_000, 10_000, fadeInMs=200, fadeOutMs=300)]}],
                 clips=("A",))
        return Renderer(p, {"A": dc(n, 0.3)}).render().astype(np.float64)
    y0, y6 = run(0.0), run(-6.0)
    nz = np.abs(y0) > 1e-6
    assert np.allclose(y6[nz] / y0[nz], 10 ** (-6 / 20), rtol=1e-5)  # envelope shape unchanged by gain


# ---------------------------------------------------------------- summing / no clipping

def delayed_copy_project(limiter="soft_clip"):
    n = A.shape[1]
    return proj([
        {"id": "l1", "name": "dry", "items": [item("d", "A", 0, n, 0)]},
        {"id": "l2", "name": "delayed copy", "order": 1, "items": [item("w", "A", 0, n, 0)],
         "processorChain": [{"id": "dl", "type": "delay", "params": {"timeMs": 125, "feedback": 0.4, "mix": 1.0},
                             "fadeInMs": 50, "fadeOutMs": 500}]}],
        clips=("A",), mainClipId="A", masterLimiter=limiter)


@pytest.mark.parametrize("limiter", ["soft_clip", "gain_compensation"])
def test_lane2_delayed_copy_over_lane1_does_not_clip(limiter):
    p = delayed_copy_project(limiter)
    r = Renderer(p, {"A": A})
    pre = r._premaster(0, r.length, "main_plus_lanes")
    assert np.max(np.abs(pre)) > 1.5  # the naive sum would clip hard
    y, meta = render(p, {"A": A})
    assert np.max(np.abs(y)) < 1.0
    assert meta["master"]["peakDbfs"] < 0 and meta["master"]["headroomDb"] > 0
    assert {l["laneId"] for l in meta["lanes"]} == {"l1", "l2"}
    if limiter == "soft_clip":
        T = 10 ** (-1 / 20)
        below = np.abs(pre) <= T
        assert np.array_equal(y[below], pre[below].astype(np.float32))  # bit-transparent below threshold
        assert meta["master"]["softClippedSamples"] > 0
    else:
        assert meta["master"]["peakDbfs"] == pytest.approx(-1.0, abs=0.01)
        assert meta["master"]["gainCompensationDb"] < -3


def test_quiet_mix_is_untouched_and_lane_gain_pan():
    q = A * 0.1
    p = proj([{"id": "l1", "gainDb": -6.0, "pan": 1.0, "items": [item("a", "A", 0, q.shape[1], 0)]}], clips=("A",))
    y, meta = render(p, {"A": q})
    assert meta["master"]["softClippedSamples"] == 0
    assert np.allclose(y[0], 0, atol=1e-7)  # hard right: constant-power, sqrt2-normalised
    # body (outside crossfades) is exactly src * gain * sqrt2
    s = slice(5000, q.shape[1] - 5000)
    assert np.allclose(y[1, s], (q[1, s] * 10 ** (-6 / 20) * np.sqrt(2)).astype(np.float32), atol=1e-6)
    assert meta["lanes"][0]["peakDbfs"] == pytest.approx(20 * np.log10(np.max(np.abs(y))), abs=0.01)


def test_mute_solo_and_selection_ignored():
    lanes = [{"id": "l1", "items": [item("a", "A", 0, 40_000, 0)]},
             {"id": "l2", "items": [item("b", "B", 0, 40_000, 0)], "solo": True}]
    p = proj(lanes)
    y = Renderer(p, {"A": A, "B": B}).render()
    only_b = Renderer(proj([lanes[1]]), {"A": A, "B": B}).render()
    assert np.array_equal(y, only_b)
    sel = proj(lanes, selection={"sourceClipId": "A", "range": {"startSample": 5, "endSample": 9}})
    assert np.array_equal(Renderer(sel, {"A": A, "B": B}).render(), y)


# ---------------------------------------------------------------- delay / tempo sync

def impulse(n, at):
    x = np.zeros((2, n))
    x[:, at] = 1.0
    return x


def test_delay_offset_is_sample_exact():
    n, imp, at = 120_000, 10_000, 5_000
    p = proj([{"id": "l1", "items": [item("i", "A", 0, n, at)],
               "processorChain": [{"id": "d", "type": "delay", "params": {"timeMs": 100, "feedback": 0.5, "mix": 0.5}}]}],
             clips=("A",))
    y = Renderer(p, {"A": impulse(n, imp)}).render()[0].astype(np.float64)
    D = 4410
    t0 = at + imp
    nz = np.flatnonzero(np.abs(y) > 1e-9)
    assert nz[0] == t0 and y[t0] == pytest.approx(0.5)
    for k in range(1, 6):
        assert y[t0 + k * D] == pytest.approx(0.5 * 0.5 ** (k - 1), rel=1e-6)
        assert np.all(np.abs(y[t0 + (k - 1) * D + 1: t0 + k * D]) < 1e-9)


@pytest.mark.parametrize("snap,beats,expected", [
    ({"type": "beats", "value": 1, "bpm": 120}, 1, 22_050),
    ({"type": "beats", "value": 1, "bpm": 120}, 0.75, 16_538),     # 0.375 s -> 16537.5 -> round half even
    ({"type": "beats", "value": 1, "bpm": 97}, 0.5, 13_639),       # 0.30928 s
    ({"type": "beats", "value": 0.5, "bpm": 90}, 3, 44_100),       # 3 x eighth notes at 90 = 1 s
    ({"type": "seconds", "value": 0.1}, 3, 13_230),                # 3 snap steps of 0.1 s
    ({"type": "seconds", "value": 0.1}, 1, 4_410),
])
def test_tempo_sync_math(snap, beats, expected):
    s = Snap.model_validate(snap)
    assert delay_samples_synced(beats, s, SR) == expected
    n, at = 120_000, 1_000
    p = proj([{"id": "l1", "items": [item("i", "A", 0, n, 0)],
               "processorChain": [{"id": "d", "type": "delay",
                                   "params": {"syncToSnap": True, "beats": beats, "mix": 1.0}}]}],
             clips=("A",), snap=snap)
    y = Renderer(p, {"A": impulse(n, 10_000)}).render()[0]
    assert int(np.argmax(np.abs(y))) == 10_000 + expected


def test_beats_snap_requires_bpm_and_bad_params_rejected():
    with pytest.raises(ValidationError):
        Snap.model_validate({"type": "beats", "value": 1})
    bad = proj([{"id": "l1", "items": [item("i", "A", 0, 100, 0)],
                 "processorChain": [{"id": "d", "type": "delay", "params": {"feedback": 1.5}}]}], clips=("A",))
    with pytest.raises(ProcessorError):
        Renderer(bad, {"A": dc(1000, 0.1)})
    with pytest.raises(ProcessorError):
        Renderer(proj([{"id": "l1", "processorChain": [{"id": "x", "type": "nope"}]}], clips=("A",)), {"A": dc(10, 0)})


def test_processor_fade_envelope():
    n = 200_000
    p = proj([{"id": "l1", "items": [item("i", "A", 0, n, 0, crossfade={"lengthMs": 50})],
               "processorChain": [{"id": "e", "type": "eq", "params": {"lowGainDb": -24, "midGainDb": -24, "highGainDb": -24},
                                   "fadeInMs": 1000}]}], clips=("A",))
    y = Renderer(p, {"A": dc(n, 0.5)}).render()[0].astype(np.float64)
    # bypass at the start of the content, fully processed (heavily cut) after the 1 s fade-in
    assert y[3000] > 0.4 and abs(y[60_000]) < 0.05


def test_registry_is_extensible():
    @register("gain_test")
    class Gain(BaseProcessor):
        PARAMS = ("db",)
        def parse(self, p):
            self.g = 10 ** (p.get("db", 0) / 20)
        def wet(self, x):
            return x * self.g
    try:
        p = proj([{"id": "l1", "items": [item("i", "A", 0, 50_000, 0)],
                   "processorChain": [{"id": "g", "type": "gain_test", "params": {"db": -20}}]}], clips=("A",))
        y = Renderer(p, {"A": dc(50_000, 0.5)}).render()[0]
        assert y[25_000] == pytest.approx(0.05, rel=1e-5)
    finally:
        REGISTRY.pop("gain_test")
    assert {"delay", "eq"} <= set(REGISTRY)


# ---------------------------------------------------------------- gapless blocks / views

def full_project():
    n = A.shape[1]
    return proj([
        {"id": "l1", "items": [item("a", "A", 0, n, 0), item("b", "B", RS, RS + L, P, "replace")],
         "processorChain": [{"id": "eq1", "type": "eq", "params": {"lowGainDb": 3, "highGainDb": -4, "drive": 6}}]},
        {"id": "l2", "pan": -0.4, "gainDb": -3, "items": [item("c", "B", 1000, 90_000, 30_000, fadeInMs=20)],
         "processorChain": [{"id": "d2", "type": "delay", "params": {"syncToSnap": True, "beats": 0.5, "feedback": 0.6},
                             "fadeInMs": 100, "fadeOutMs": 800}]},
    ], mainClipId="A", mainGainDb=-2, snap={"type": "beats", "value": 1, "bpm": 128})


@pytest.mark.parametrize("view", ["main", "main_plus_lanes"])
def test_block_concat_equals_full_render(view):
    p = full_project()
    r = Renderer(p, {"A": A, "B": B})
    full = r.render(view)
    rng = np.random.default_rng(3)
    blocks, s = [], 0
    while s < r.length:
        n = int(rng.choice([1, 7, 64, 511, 4096, 44_101, 100_003]))
        blocks.append(render_block(p, {"A": A, "B": B}, s, min(n, r.length - s), view))
        s += n
    cat = np.concatenate(blocks, axis=1)
    assert cat.shape == full.shape
    assert np.max(np.abs(cat - full)) < 1e-6  # EQ's IIR state is rebuilt from 0.5 s lookbehind


def test_block_concat_is_bit_exact_without_iir():
    p = delayed_copy_project()
    r = Renderer(p, {"A": A})
    full = r.render()
    cat = np.concatenate([r.render_block(s, min(3001, r.length - s)) for s in range(0, r.length, 3001)], axis=1)
    assert np.array_equal(cat, full)


def test_views_share_length_and_alignment():
    p = full_project()
    r = Renderer(p, {"A": A, "B": B})
    main, both = r.render("main"), r.render("main_plus_lanes")
    assert main.shape == both.shape and main.shape[1] == r.length > A.shape[1]  # includes delay tail
    g = 10 ** (-2 / 20)
    assert np.allclose(main[:, : A.shape[1]], (A * g).astype(np.float32), atol=1e-6)  # main starts at sample 0
    assert np.all(main[:, A.shape[1]:] == 0)
    # switching views mid-stream is gapless: any sample of either view is the same whether rendered alone or in a block
    s = 123_457
    assert np.array_equal(r.render_block(s, 1, "main"), main[:, s:s + 1])
    assert np.array_equal(r.render_block(s, 1, "main_plus_lanes"), both[:, s:s + 1])


# ---------------------------------------------------------------- model / undo / schema

def test_history_undo_redo_and_op_log():
    p0 = proj([{"id": "l1"}], clips=("A",))
    h = H.new_history(p0)
    h = H.commit(h, H.add_item(H.current(h), "l1", LaneItem.model_validate(item("i1", "A", 0, 1000, 0))), "addItem", ["i1"])
    h = H.commit(h, H.move_item(H.current(h), "i1", 4410), "moveItem", ["i1"])
    h = H.commit(h, H.set_lane(H.current(h), "l1", gainDb=-3.0), "setLane", ["l1"])
    op = h.ops[1]
    assert op.type == "moveItem" and op.before["i1"]["laneStartSample"] == 0 and op.after["i1"]["laneStartSample"] == 4410
    assert h.ops[0].before["i1"] is None
    u = H.undo(H.undo(h))
    assert H.current(u).lanes[0].items[0].laneStartSample == 0 and H.current(u).lanes[0].gainDb == 0
    assert H.current(H.redo(u)).lanes[0].items[0].laneStartSample == 4410
    b = H.commit(u, H.delete_item(H.current(u), "i1"), "deleteItem", ["i1"])  # new branch drops redo
    assert not H.can_redo(b) and len(b.ops) == 2 and b.ops[-1].after["i1"] is None
    assert p0.lanes[0].items == ()  # snapshots are immutable
    with pytest.raises(ValidationError):
        p0.lanes[0].items = ()


def test_model_validation():
    with pytest.raises(ValidationError):
        proj([{"id": "l1", "items": [item("i", "Z", 0, 10, 0)]}])  # unknown clip
    with pytest.raises(ValidationError):
        proj([{"id": "l1", "items": [item("i", "A", 10, 10, 0)]}])  # empty range
    with pytest.raises(ValidationError):
        proj([{"id": "x", "items": [item("x", "A", 0, 10, 0)]}])  # duplicate id
    with pytest.raises(ValidationError):
        proj([{"id": "l1", "items": [item("i", "A", 0, 10, 0, crossfade={"lengthMs": 20})]}])
    p = full_project()
    assert Project.model_validate_json(p.model_dump_json()) == p


def test_schema_file_is_current_and_validates_examples():
    assert subprocess.run([sys.executable, str(ROOT / "scripts/gen_schema.py"), "--check"]).returncode == 0, \
        "schema/edit_model.json is stale: run python scripts/gen_schema.py"
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "schema/edit_model.json").read_text())
    jsonschema.validate(json.loads(full_project().model_dump_json()), schema)
    example = json.loads((ROOT / "schema/example_project.json").read_text())
    jsonschema.validate(example, schema)
    Project.model_validate(example)


# ---------------------------------------------------------------- equal power / delay envelope / bus / meters

def test_equal_power_is_true_sin_cos_and_differs_from_linear():
    from app.edit.engine import curves
    w = np.linspace(0, 1, 101)
    g_out, g_in = curves(w, "equal_power")
    assert np.allclose(g_out, np.cos(w * np.pi / 2)) and np.allclose(g_in, np.sin(w * np.pi / 2))
    assert np.allclose(g_out ** 2 + g_in ** 2, 1.0)  # constant power
    l_out, l_in = curves(w, "linear")
    assert np.allclose(l_out + l_in, 1.0)
    assert g_in[50] == pytest.approx(np.sqrt(0.5)) and l_in[50] == pytest.approx(0.5)


def test_equal_power_has_no_rms_dip_on_uncorrelated_audio_but_linear_does():
    n = 4 * SR
    rng = np.random.default_rng(11)
    old, new = 0.2 * rng.standard_normal((2, n)), 0.2 * rng.standard_normal((2, n))
    res = {}
    for curve in ("equal_power", "linear"):
        p = proj([{"id": "l1", "items": [item("a", "A", 0, n, 0),
                                         item("b", "B", SR, 3 * SR, SR, "replace",
                                              crossfade={"lengthMs": 100, "curve": curve})]}])
        y = Renderer(p, {"A": old, "B": new}).render().astype(np.float64)
        g = item_geom(p.lanes[0].items[1], SR)
        # samples in the start window are exactly old*g_out + new*g_in
        t = np.arange(g.P - g.h, g.P - g.h + g.X)
        w = (t - (g.P - g.h) + 0.5) / g.X
        go, gi = (np.cos(w * np.pi / 2), np.sin(w * np.pi / 2)) if curve == "equal_power" else (1 - w, w)
        assert np.allclose(y[:, t], (old[:, t] * go + new[:, t] * gi).astype(np.float32), atol=1e-6)
        mid = g.P - g.h + g.X // 2
        win = 441  # 10 ms around the crossfade midpoint vs 10 ms well before it
        rms = lambda a: np.sqrt(np.mean(y[:, a:a + win] ** 2))
        res[curve] = 20 * np.log10(rms(mid - win // 2) / rms(SR // 2))
    assert abs(res["equal_power"]) < 0.75, res      # no dip
    assert res["linear"] < -2.0, res                 # ~-3 dB dip at the midpoint
    assert res["equal_power"] - res["linear"] > 2.0


def test_delay_fade_envelopes_apply_to_the_wet_section_only():
    n, imp, D, mix, fb = 400_000, 20_000, 2_205, 0.5, 0.9
    fi, fo = 1.0, 0.5  # seconds
    p = proj([{"id": "l1", "items": [item("i", "A", 0, n, 0, crossfade={"lengthMs": 50})],
               "processorChain": [{"id": "d", "type": "delay", "params": {"timeMs": 50, "feedback": fb, "mix": mix},
                                   "fadeInMs": fi * 1000, "fadeOutMs": fo * 1000}]}], clips=("A",))
    r = Renderer(p, {"A": impulse(n, imp)})
    y = r.render()[0].astype(np.float64)
    g = r.lanes[0].geoms[0]
    proc = r.lanes[0].procs[0]
    span0, span1 = r.lanes[0].span
    assert y[imp] == pytest.approx(1 - mix)  # dry is never faded
    end = span1 + proc.tail()
    for k in range(1, proc.taps + 1):
        t = imp + k * D
        env = min(1, (t - span0 + 1) / (fi * SR)) * min(1, max(0, (end - t) / (fo * SR)))
        assert y[t] == pytest.approx(mix * fb ** (k - 1) * env, abs=1e-7), k
    assert y[imp + D] < 0.6 * mix  # first echo is inside the fade-in, clearly attenuated
    # without fades the same echo is at full level
    p0 = proj([{"id": "l1", "items": [item("i", "A", 0, n, 0, crossfade={"lengthMs": 50})],
                "processorChain": [{"id": "d", "type": "delay", "params": {"timeMs": 50, "feedback": fb, "mix": mix}}]}],
              clips=("A",))
    assert Renderer(p0, {"A": impulse(n, imp)}).render()[0, imp + D] == pytest.approx(mix)


def test_lane_bus_gain_is_applied_once_post_chain():
    n = 60_000
    def run(gain):
        p = proj([{"id": "l1", "gainDb": gain, "items": [item("a", "A", 0, n, 0), item("b", "A", 0, n, 10_000)],
                   "processorChain": [{"id": "e", "type": "eq", "params": {"drive": 18}}]}], clips=("A",))
        r = Renderer(p, {"A": 0.4 * A[:, :n]})
        return r.lane_block(r.lanes[0], 0, r.length)  # the lane bus output, before the master
    y0, y12 = run(0.0), run(-12.0)
    # the nonlinear drive sees the same input either way: bus gain is a pure scale on the lane sum
    assert np.allclose(y12, y0 * 10 ** (-12 / 20), rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("pan", [-1.0, -0.5, 0.0, 0.3, 1.0])
def test_lane_pan_is_constant_power(pan):
    x = A[:1].repeat(2, axis=0) * 0.3
    p = proj([{"id": "l1", "pan": pan, "items": [item("a", "A", 0, x.shape[1], 0)]}], clips=("A",))
    y = Renderer(p, {"A": x}).render().astype(np.float64)
    s = slice(10_000, x.shape[1] - 10_000)
    assert np.allclose(y[0, s] ** 2 + y[1, s] ** 2, 2 * x[0, s] ** 2, rtol=1e-4, atol=1e-9)


def test_solo_any_logic():
    lanes = [{"id": f"l{k}", "items": [item(f"i{k}", "A", 0, 1000, 0)], **kw}
             for k, kw in enumerate([{"solo": True}, {"solo": True, "mute": True}, {}, {"mute": True}])]
    r = Renderer(proj(lanes, clips=("A",)), {"A": dc(1000, 0.1)})
    assert [rt.audible for rt in r.lanes] == [True, False, False, False]
    r2 = Renderer(proj([{**l, "solo": False} for l in lanes], clips=("A",)), {"A": dc(1000, 0.1)})
    assert [rt.audible for rt in r2.lanes] == [True, False, True, False]


def test_meters_match_numpy():
    p = full_project()
    r = Renderer(p, {"A": A, "B": B})
    from app.edit.engine import RenderStats
    st = RenderStats()
    y = r.render("main_plus_lanes", st).astype(np.float64)
    meta = st.as_dict(r, "main_plus_lanes")
    assert meta["master"]["peakDbfs"] == pytest.approx(20 * np.log10(np.abs(y).max()), abs=1e-3)
    assert meta["master"]["rmsDbfs"] == pytest.approx(20 * np.log10(np.sqrt(np.mean(y ** 2))), abs=1e-3)
    assert meta["master"]["headroomDb"] == pytest.approx(-meta["master"]["peakDbfs"], abs=1e-3)
    for rt, lm in zip(r.lanes, meta["lanes"]):
        lane = r.lane_block(rt, 0, r.length)
        assert lm["peakDbfs"] == pytest.approx(20 * np.log10(np.abs(lane).max()), abs=1e-3)
        assert lm["rmsDbfs"] == pytest.approx(20 * np.log10(np.sqrt(np.mean(lane ** 2))), abs=1e-3)


def test_kotlin_vectors_are_current():
    assert subprocess.run([sys.executable, str(ROOT / "scripts/gen_kotlin_vectors.py"), "--check"]).returncode == 0, \
        "schema/kotlin-dsp/test_vectors.json is stale: run python scripts/gen_kotlin_vectors.py"
