import io
import json
import math
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app import main
from app.dsp import separation
from app.edit.engine import Renderer
from app.edit.model import Project
from app.tone import chain as tchain
from app.tone.dsp import ToneChain
from app.tone.profile import SPEC, ProfileError, defaults, parse_profile, schema_doc

client = TestClient(main.app, headers={"X-API-Key": "test-key-123"})
anon = TestClient(main.app)
SR = 44100
RNG = np.random.default_rng(7)


def stereo(seconds=1.0, amp=0.25):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    L = amp * np.sin(2 * np.pi * 220 * t) + 0.05 * RNG.standard_normal(n)
    R = amp * np.sin(2 * np.pi * 330 * t) + 0.05 * RNG.standard_normal(n)
    return np.stack([L, R])


def wav_bytes(x, sr=SR):
    b = io.BytesIO()
    sf.write(b, x.T, sr, format="WAV", subtype="FLOAT")
    return b.getvalue()


def fetch(url):
    r = client.get(url.replace("http://testserver", ""))
    assert r.status_code == 200
    return sf.read(io.BytesIO(r.content), always_2d=True)[0].T


EVERYTHING = {
    "lowCut": {"enabled": True, "freqHz": 60, "slopeDbOct": 24},
    "highCut": {"enabled": True, "freqHz": 15000, "slopeDbOct": 12},
    "eq": {"bands": [{"type": "lowshelf", "freqHz": 100, "gainDb": 4}, {"freqHz": 1000, "gainDb": -3, "q": 2},
                     {"type": "notch", "freqHz": 3000, "q": 8}, {"type": "highshelf", "freqHz": 10000, "gainDb": 3}]},
    "compressor": {"enabled": True, "thresholdDb": -24, "ratio": 4, "attackMs": 5, "releaseMs": 120, "makeupDb": 3},
    "transient": {"enabled": True, "attackDb": 4, "sustainDb": -3},
    "saturation": {"enabled": True, "type": "tube", "driveDb": 12, "mix": 0.5, "asymmetry": 0.4},
    "width": {"enabled": True, "widthPct": 160, "monoBelowHz": 150},
    "output": {"gainDb": -2},
}

# ---------------------------------------------------------------- arbitrary values are clamped, never crash
CRAZY = [1e308, -1e308, float("inf"), float("-inf"), float("nan"), "abc", "12", None, True, [], {}, -0.0, 0, 7]


def _crazy_params(v):
    return {"lowCut": {"enabled": True, "freqHz": v, "slopeDbOct": v},
            "highCut": {"enabled": True, "freqHz": v, "slopeDbOct": v},
            "eq": {"bands": [{"type": "peak", "freqHz": v, "gainDb": v, "q": v}, {"type": v, "freqHz": v}]},
            "compressor": {"enabled": True, "thresholdDb": v, "ratio": v, "attackMs": v, "releaseMs": v,
                           "kneeDb": v, "makeupDb": v},
            "transient": {"enabled": True, "attackDb": v, "sustainDb": v},
            "saturation": {"enabled": True, "type": "tube", "driveDb": v, "mix": v, "asymmetry": v},
            "width": {"enabled": True, "widthPct": v, "monoBelowHz": v},
            "output": {"gainDb": v}, "whatIsThis": v}


@pytest.mark.parametrize("v", CRAZY, ids=[repr(v) for v in CRAZY])
def test_arbitrary_values_are_clamped_not_crashed(v):
    p = parse_profile({"mode": "freestyle", "params": _crazy_params(v)})
    for sec, fields in SPEC.items():          # every value inside its documented safe range
        for k, sp in fields.items():
            if sp["kind"] == "number":
                assert sp["min"] <= p.params[sec][k] <= sp["max"], (sec, k)
    for b in p.params["eq"]["bands"]:
        assert 20 <= b["freqHz"] <= 20000 and -30 <= b["gainDb"] <= 30 and 0.1 <= b["q"] <= 18
    assert any("whatIsThis" in w for w in p.warnings)
    for sr in (8000, 44100):                 # bands above 0.45*sr are clamped at render time too
        y = ToneChain(p.params, sr).process(stereo(0.3)[:, : int(0.3 * sr)])
        assert np.all(np.isfinite(y)) and np.max(np.abs(y)) < 50


def test_warnings_not_errors_and_unknown_keys_ignored():
    p = parse_profile({"mode": "freestyle", "foo": 1, "params": {"width": {"widthPct": 900, "zz": 1}, "bar": {}}})
    assert p.params["width"]["widthPct"] == 200
    assert any("clamped to 200" in w for w in p.warnings) and any("foo" in w for w in p.warnings)
    assert any("params.bar" in w for w in p.warnings) and any("width.zz" in w for w in p.warnings)


def test_too_many_bands_truncated_with_warning():
    p = parse_profile({"params": {"eq": {"bands": [{"gainDb": 1}] * 40}}})
    assert len(p.params["eq"]["bands"]) == 32 and any("first 32" in w for w in p.warnings)


def test_preset_mode_is_strict():
    with pytest.raises(ProfileError) as e:
        parse_profile({"mode": "preset", "params": {"width": {"widthPct": 190}, "eq": {"bands": [{"gainDb": 20}]}}})
    assert len(e.value.errors) == 2
    ok = parse_profile({"mode": "preset", "params": {"width": {"enabled": True, "widthPct": 140}}})
    assert ok.params["width"]["widthPct"] == 140
    with pytest.raises(ProfileError):
        parse_profile({"mode": "jazz"})


# ---------------------------------------------------------------- identities
def test_flat_freestyle_is_exact_identity():
    x = stereo()
    for body in ({}, {"mode": "freestyle", "params": {}}, {"params": defaults()},
                 {"params": {"width": {"enabled": True, "widthPct": 100}, "output": {"gainDb": 0},
                             "eq": {"bands": [{"freqHz": 500, "gainDb": 0}]},
                             "saturation": {"enabled": True, "driveDb": 0}}}):
        c = ToneChain(parse_profile(body).params, SR)
        assert c.names == [] and np.array_equal(c.process(x), x)


def test_width_zero_is_mono_and_100_is_identity():
    x = stereo()
    for mono_below in (0, 120, 500):
        y0 = ToneChain(parse_profile({"params": {"width": {"enabled": True, "widthPct": 0,
                                                           "monoBelowHz": mono_below}}}).params, SR).process(x)
        assert np.array_equal(y0[0], y0[1])
        np.testing.assert_allclose(y0[0], (x[0] + x[1]) / 2, atol=1e-15)
        y1 = ToneChain(parse_profile({"params": {"width": {"enabled": True, "widthPct": 100,
                                                           "monoBelowHz": mono_below}}}).params, SR).process(x)
        assert np.array_equal(y1, x)


def test_width_200_keeps_bass_mono_safe():
    n = SR
    t = np.arange(n) / SR
    bass_side = 0.2 * np.sin(2 * np.pi * 50 * t)
    hi_side = 0.2 * np.sin(2 * np.pi * 3000 * t)
    s = bass_side + hi_side
    x = np.stack([s, -s])                       # pure side
    y = ToneChain(parse_profile({"params": {"width": {"enabled": True, "widthPct": 200,
                                                      "monoBelowHz": 200}}}).params, SR).process(x)
    side = (y[0] - y[1])[SR // 2:] / 2
    amp = lambda sig, f: 2 * abs(np.dot(sig, np.exp(-2j * np.pi * f * np.arange(len(sig)) / SR))) / len(sig)
    assert amp(side, 50) == pytest.approx(0.2, rel=0.05)      # bass side not widened
    assert amp(side, 3000) == pytest.approx(0.4, rel=0.05)    # highs doubled


def test_saturation_types_bounded_and_unity_at_full_scale():
    x = np.stack([np.linspace(-1, 1, 2001)] * 2)
    for typ in ("tanh", "tape", "tube"):
        p = parse_profile({"params": {"saturation": {"enabled": True, "type": typ, "driveDb": 24, "mix": 1}}}).params
        y = ToneChain(p, SR).process(x)
        assert np.max(np.abs(y)) < 1.5 and np.all(np.isfinite(y))
        if typ == "tanh":
            assert y[0, -1] == pytest.approx(1.0) and y[0, 1000] == pytest.approx(0.0, abs=1e-12)


def test_compressor_reduces_loud_and_transient_shaper_moves_onsets():
    t = np.arange(SR) / SR
    loud = np.stack([0.9 * np.sin(2 * np.pi * 440 * t)] * 2)
    p = parse_profile({"params": {"compressor": {"enabled": True, "thresholdDb": -20, "ratio": 10,
                                                 "attackMs": 1, "releaseMs": 50}}}).params
    y = ToneChain(p, SR).process(loud)
    assert np.max(np.abs(y[:, SR // 2:])) < 0.3
    hits = np.zeros((2, SR))
    for k in range(4):
        a = k * SR // 4
        hits[:, a:a + 4000] = np.exp(-np.arange(4000) / 800) * np.sin(2 * np.pi * 200 * np.arange(4000) / SR) * 0.5
    up = ToneChain(parse_profile({"params": {"transient": {"enabled": True, "attackDb": 9}}}).params, SR).process(hits)
    assert np.max(np.abs(up[:, :200])) > 1.5 * np.max(np.abs(hits[:, :200]))


def test_streaming_blocks_equal_whole():
    p = parse_profile({"params": EVERYTHING}).params
    x = stereo(2.0)
    whole = ToneChain(p, SR).process(x)
    c = ToneChain(p, SR)
    blocks = np.concatenate([c.process(x[:, a:a + 7919]) for a in range(0, x.shape[1], 7919)], axis=1)
    assert np.max(np.abs(whole - blocks)) < 1e-12


# ---------------------------------------------------------------- per-stem overrides
def test_per_stem_overrides_merge_and_apply():
    p = parse_profile({"params": {"output": {"gainDb": -6}, "width": {"enabled": True, "widthPct": 50}},
                       "stems": {"vocals": {"output": {"gainDb": 3}}, "bass": {"width": {"widthPct": 0}},
                                 "bad name!": {"output": {"gainDb": 1}}}})
    assert p.for_stem("vocals")["output"]["gainDb"] == 3 and p.for_stem("vocals")["width"]["widthPct"] == 50
    assert p.for_stem("bass")["width"]["widthPct"] == 0 and p.for_stem("bass")["output"]["gainDb"] == -6
    assert p.for_stem("drums") == p.params and "bad name!" not in p.stems
    assert any("bad name!" in w for w in p.warnings)


def test_separate_applies_tone_profile_per_stem(monkeypatch):
    x = stereo(1.0)

    def fake(src, out_dir, model):
        out_dir.mkdir(parents=True, exist_ok=True)
        res = {}
        for s in ("vocals", "drums", "bass", "other"):
            res[s] = out_dir / f"{s}.wav"
            sf.write(res[s], x.T, SR, subtype="FLOAT")
        return res
    monkeypatch.setattr(separation, "separate", fake)
    prof = {"mode": "freestyle", "params": {"output": {"gainDb": -6}, "huh": 1},
            "stems": {"vocals": {"output": {"gainDb": -12}}, "bass": {"width": {"enabled": True, "widthPct": 0}},
                      "piano": {"output": {"gainDb": 1}}}}
    r = client.post("/separate", files={"file": ("a.wav", wav_bytes(x), "audio/wav")},
                    data={"toneProfile": json.dumps(prof)})
    assert r.status_code == 202
    j = client.get(f"/jobs/{r.json()['job_id']}").json()
    assert j["status"] == "done", j
    assert set(j["urls"]) == {s + suf for s in ("vocals", "drums", "bass", "other") for suf in ("", "_raw")}
    raw = fetch(j["urls"]["drums_raw"])
    ratio = lambda a, b: 20 * np.log10(np.sqrt(np.mean(a ** 2)) / np.sqrt(np.mean(b ** 2)))
    assert ratio(fetch(j["urls"]["drums"]), raw) == pytest.approx(-6, abs=0.05)
    assert ratio(fetch(j["urls"]["vocals"]), raw) == pytest.approx(-12, abs=0.05)
    b = fetch(j["urls"]["bass"])
    assert np.max(np.abs(b[0] - b[1])) < 1e-4
    tp = j["meta"]["toneProfile"]
    assert tp["unmatchedStemOverrides"] == ["piano"] and any("huh" in w for w in tp["warnings"])


def test_separate_rejects_bad_profile_before_work():
    r = client.post("/separate", files={"file": ("a.wav", wav_bytes(stereo(0.2)), "audio/wav")},
                    data={"toneProfile": "{not json"})
    assert r.status_code == 422
    r = client.post("/separate", files={"file": ("a.wav", wav_bytes(stereo(0.2)), "audio/wav")},
                    data={"toneProfile": json.dumps({"mode": "preset", "params": {"output": {"gainDb": 11}}})})
    assert r.status_code == 422 and "errors" in r.json()["detail"]


# ---------------------------------------------------------------- chain order
def test_chain_order_denoise_pitch_tone(monkeypatch, tmp_path):
    calls = []

    def spy(name, real):
        def f(src, dst, params):
            calls.append(name)
            return real(src, dst, params)
        return f
    monkeypatch.setattr(tchain, "STEPS", [(n, spy(n, fn)) for n, fn in tchain.STEPS])
    src = tmp_path / "in.wav"
    sf.write(src, stereo(1.5).T, SR, subtype="FLOAT")
    p = parse_profile({"params": {"denoise": {"enabled": True, "preset": "hiss", "strength": 0.5},
                                  "pitchTempo": {"enabled": True, "tempo": 1.25},
                                  "output": {"gainDb": -1}}}).params
    meta = tchain.run_chain(src, tmp_path / "out.wav", p, tmp_path / "w")
    assert calls == ["denoise", "pitchTempo", "tone"] and meta["steps"] == ["denoise", "pitchTempo", "tone"]
    n_out = sf.info(str(tmp_path / "out.wav")).frames
    assert n_out == pytest.approx(1.5 * SR / 1.25, rel=0.02)          # tempo ran before tone, length changed
    assert schema_doc()["chainOrder"] == ["denoise", "pitchTempo", "tone", "laneBus"]
    assert not list((tmp_path / "w").glob("_*"))                     # intermediates cleaned up


def test_lane_processors_registered_and_before_bus():
    x = stereo(1.0, amp=0.3)
    sf_ = {"main": x.astype(np.float32)}
    proj = Project.model_validate({
        "id": "fs", "sampleRate": SR, "clips": [{"id": "main"}],
        "lanes": [{"id": "l1", "gainDb": -6, "items": [{"id": "i", "sourceClipId": "main",
                                                         "range": {"startSample": 0, "endSample": SR},
                                                         "laneStartSample": 0}],
                   "processorChain": [
                       {"id": "w", "type": "width", "params": {"widthPct": 0}},
                       {"id": "c", "type": "compressor", "params": {"thresholdDb": -30, "ratio": 3}},
                       {"id": "s", "type": "saturation_tape", "params": {"driveDb": 9999, "unknown": 1}},
                       {"id": "t", "type": "tone", "params": EVERYTHING}]}]})
    r = Renderer(proj, sf_)
    full = r.render_block(0, r.length, "main_plus_lanes")
    blocks = np.concatenate([r.render_block(s, min(30011, r.length - s), "main_plus_lanes")
                             for s in range(0, r.length, 30011)], axis=1)
    assert np.max(np.abs(full - blocks)) < 1e-6 and np.all(np.isfinite(full))
    # bus gain/pan come after the chain: lane sum == chain output scaled by the bus gain
    from app.edit.processors import build
    procs = [build(p, SR, proj.snap) for p in proj.lanes[0].processorChain]
    assert procs[2].cfg["driveDb"] == 36 and procs[2].cfg["type"] == "tape"
    rt = r.lanes[0]
    y = r.compose(rt, rt.span[0], rt.span[1])
    for p in procs:
        y = p.apply(y, rt.span[0], *rt.span)
    lane = r.lane_block(rt, rt.span[0], rt.span[1] - rt.span[0])
    np.testing.assert_allclose(lane, y * 10 ** (-6 / 20), atol=1e-9)


# ---------------------------------------------------------------- API
def test_process_tone_freestyle_and_legacy():
    x = stereo(1.0)
    up = ("a.wav", wav_bytes(x), "audio/wav")
    body = {"mode": "freestyle", "params": {"width": {"enabled": True, "widthPct": 0},
                                            "eq": {"bands": [{"freqHz": 1e7, "gainDb": 99}]}, "zap": 1},
            "stems": {"vocals": {"output": {"gainDb": -6}}}}
    r = client.post("/process/tone", files={"file": up}, data={"profile": json.dumps(body)})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["meta"]["steps"] == ["tone"] and j["meta"]["stemOverrideApplied"] is False
    assert any("clamped" in w for w in j["meta"]["warnings"]) and any("zap" in w for w in j["meta"]["warnings"])
    assert any("0.45*sampleRate" in w for w in j["meta"]["tone"]["warnings"])  # 20 kHz > 0.45*44.1k = 19845
    y = fetch(j["urls"]["output"])
    assert np.max(np.abs(y[0] - y[1])) < 1e-4
    r2 = client.post("/process/tone", files={"file": up}, data={"toneProfile": json.dumps(body), "stem": "vocals"})
    assert r2.json()["meta"]["stemOverrideApplied"] is True
    flat = client.post("/process/tone", files={"file": up}, data={"profile": "{}"}).json()
    np.testing.assert_allclose(fetch(flat["urls"]["output"]), x, atol=1 / 32767)
    legacy = client.post("/process/tone", files={"file": up}, data={"low_gain_db": 3, "drive": 6, "mix": 0.5})
    assert legacy.status_code == 200 and "output" in legacy.json()["urls"]
    assert client.post("/process/tone", files={"file": up}, data={"low_gain_db": 30}).status_code == 422
    assert client.post("/process/tone", files={"file": up},
                       data={"profile": json.dumps({"mode": "preset", "params": {"width": {"widthPct": 10}}})}
                       ).status_code == 422
    assert anon.post("/process/tone", files={"file": up}, data={"profile": "{}"}).status_code == 401


def test_tone_schema_endpoint():
    assert anon.get("/tone/schema").status_code == 401
    s = client.get("/tone/schema").json()
    assert s["sections"]["width"]["widthPct"] == {**SPEC["width"]["widthPct"]}
    assert s["sections"]["eq"]["bands"]["maxItems"] == 32 and s["defaults"]["output"]["gainDb"] == 0
    for sec in s["sections"].values():
        for sp in sec.values():
            if sp["kind"] == "number":
                assert sp["min"] <= sp["presetMin"] <= sp["default"] <= sp["presetMax"] <= sp["max"] or \
                    sp["min"] <= sp["default"] <= sp["max"]


def test_bundled_schema_files_current():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    assert subprocess.run([sys.executable, str(root / "scripts/gen_tone_schema.py"), "--check"]).returncode == 0


# ---------------------------------------------------------------- memory
def test_memory_5min_stereo_full_tone_chain(tmp_path):
    """5 min stereo 44.1 kHz through every tone stage, file -> file, in a fresh interpreter: peak RSS stays small."""
    src = tmp_path / "long.wav"
    with sf.SoundFile(str(src), "w", SR, 2, subtype="PCM_16") as f:
        for _ in range(30):
            f.write((stereo(10.0, amp=0.3) * 0.8).T)
    code = textwrap.dedent(f"""
        import resource, json
        from pathlib import Path
        from app.tone.chain import run_chain
        from app.tone.profile import parse_profile
        base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        p = parse_profile({{"params": json.loads({json.dumps(json.dumps(EVERYTHING))})}}).params
        run_chain(Path({str(src)!r}), Path({str(tmp_path / 'out.wav')!r}), p, Path({str(tmp_path / 'w')!r}))
        print(base, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    """)
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    base, peak = map(int, out.stdout.split())
    assert sf.info(str(tmp_path / "out.wav")).frames == 300 * SR
    assert (peak - base) / 1024 < 120, f"tone chain grew RSS by {(peak - base) / 1024:.0f} MB"


def test_phone_preset_export_is_accepted_as_tone_profile():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    raw = json.loads((root / "schema/kotlin-presets/example_preset.json").read_text())
    p = parse_profile(raw)
    assert p.warnings == [] and p.name == "Warm Vocal" and len(p.params["eq"]["bands"]) == 5
    assert p.for_stem("vocals")["compressor"]["ratio"] == 5 and p.params["compressor"]["ratio"] == 3
    parse_profile({**raw, "mode": "preset"})          # the bundled example also fits the strict preset ranges


def test_editmodel_kt_lists_every_registered_processor():
    import re
    from app.edit.processors import REGISTRY
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    kt = re.findall(r'const val \w+ = "(\w+)"', (root / "schema/EditModel.kt").read_text())
    assert sorted(kt) == sorted(REGISTRY)
