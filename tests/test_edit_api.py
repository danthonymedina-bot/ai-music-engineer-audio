import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app import main
from app.edit.engine import render
from app.edit.model import Project
from tests.edit_signals import SR, drums_and_plucks, plucks

client = TestClient(main.app, headers={"X-API-Key": "test-key-123"})
anon = TestClient(main.app)
A, B = drums_and_plucks(seconds=3.0), plucks(seconds=3.0)

PROJECT = {
    "id": "api-test", "sampleRate": SR, "clips": [{"id": "main"}, {"id": "take2"}], "mainClipId": "main",
    "snap": {"type": "seconds", "value": 0.1},
    "lanes": [
        {"id": "l1", "name": "fix", "items": [{"id": "i1", "sourceClipId": "take2",
                                                 "range": {"startSample": 22050, "endSample": 66150},
                                                 "laneStartSample": 44100, "mode": "add"}]},
        {"id": "l2", "name": "echo", "order": 1, "gainDb": -3,
         "items": [{"id": "i2", "sourceClipId": "main", "range": {"startSample": 0, "endSample": 88200},
                    "laneStartSample": 0}],
         "processorChain": [{"id": "dl", "type": "delay", "params": {"syncToSnap": True, "beats": 2, "feedback": 0.3}}]},
    ],
}


def wav(x):
    b = io.BytesIO()
    sf.write(b, x.T.astype(np.float32), SR, format="WAV", subtype="FLOAT")
    return b.getvalue()


def files(**clips):
    return [(f"clip:{k}", (f"{k}.wav", wav(v), "audio/wav")) for k, v in clips.items()]


def test_edit_render_wav_matches_engine():
    r = client.post("/edit/render", data={"project": json.dumps(PROJECT)}, files=files(main=A, take2=B))
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "audio/wav"
    meta = json.loads(r.headers["x-edit-meta"])
    y, sr = sf.read(io.BytesIO(r.content), always_2d=True)
    assert sr == SR and y.shape[0] == int(r.headers["x-edit-length-samples"]) == meta["lengthSamples"]
    ref, ref_meta = render(Project.model_validate(PROJECT), {"main": A.astype(np.float32), "take2": B.astype(np.float32)})
    assert y.shape[0] == ref.shape[1]
    assert np.max(np.abs(y.T - ref)) < 2 / 32768  # 16-bit output
    assert float(r.headers["x-edit-master-peak-dbfs"]) == meta["master"]["peakDbfs"] < 0
    assert [l["laneId"] for l in meta["lanes"]] == ["l1", "l2"] and meta["master"]["headroomDb"] > 0
    assert abs(meta["master"]["peakDbfs"] - ref_meta["master"]["peakDbfs"]) < 0.01


def test_edit_render_views_same_length_and_json_mode():
    lengths = []
    for view in ("main", "main_plus_lanes"):
        r = client.post("/edit/render", data={"project": json.dumps(PROJECT), "view": view, "response": "json"},
                        files=files(main=A, take2=B))
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["status"] == "done" and j["meta"]["view"] == view and "output" in j["urls"]
        a = client.get(j["urls"]["output"].replace("http://testserver", ""))
        assert a.status_code == 200
        lengths.append(sf.info(io.BytesIO(a.content)).frames)
    assert lengths[0] == lengths[1]


def test_edit_render_auth_and_validation():
    assert anon.post("/edit/render", data={"project": json.dumps(PROJECT)}).status_code == 401
    r = client.post("/edit/render", data={"project": json.dumps(PROJECT)}, files=files(main=A))
    assert r.status_code == 422 and "take2" in r.text
    assert client.post("/edit/render", data={"project": "{nope"}, files=files(main=A)).status_code == 422
    bad = json.loads(json.dumps(PROJECT))
    bad["lanes"][1]["processorChain"][0]["type"] = "flanger9000"
    r = client.post("/edit/render", data={"project": json.dumps(bad)}, files=files(main=A, take2=B))
    assert r.status_code == 422 and "flanger9000" in r.text
    long_range = json.loads(json.dumps(PROJECT))
    long_range["lanes"][0]["items"][0]["range"]["endSample"] = 10 ** 7
    r = client.post("/edit/render", data={"project": json.dumps(long_range)}, files=files(main=A, take2=B))
    assert r.status_code == 422 and "range ends" in r.text


def test_edit_render_is_rate_limited(monkeypatch):
    from app import security
    monkeypatch.setitem(security.LIMITS, "process", (1, 3600))
    ok = client.post("/edit/render", data={"project": json.dumps(PROJECT)}, files=files(main=A, take2=B))
    assert ok.status_code == 200
    r = client.post("/edit/render", data={"project": json.dumps(PROJECT)}, files=files(main=A, take2=B))
    assert r.status_code == 429


def test_edit_levels_endpoint():
    r = client.post("/edit/levels", data={"project": json.dumps(PROJECT)}, files=files(main=A, take2=B))
    assert r.status_code == 200, r.text
    m = r.json()
    ref, ref_meta = render(Project.model_validate(PROJECT), {"main": A.astype(np.float32), "take2": B.astype(np.float32)})
    for k in ("peakDbfs", "rmsDbfs", "headroomDb"):
        assert m["master"][k] == pytest.approx(ref_meta["master"][k], abs=0.01)
        assert [l[k] for l in m["lanes"]] == pytest.approx([l[k] for l in ref_meta["lanes"]], abs=0.01)
    part = client.post("/edit/levels", data={"project": json.dumps(PROJECT), "startSample": "44100",
                                             "numSamples": "22050"}, files=files(main=A, take2=B)).json()
    seg = ref[:, 44100:66150].astype(np.float64)
    assert part["measuredSamples"] == 22050
    assert part["master"]["rmsDbfs"] == pytest.approx(20 * np.log10(np.sqrt(np.mean(seg ** 2))), abs=0.01)
    assert anon.post("/edit/levels", data={"project": json.dumps(PROJECT)}).status_code == 401


def test_max_lanes_env_default_and_override():
    import subprocess, sys, os
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    code = "from app import config; print(config.MAX_LANES)"
    env = {k: v for k, v in os.environ.items() if k != "AUDIO_ENGINE_MAX_LANES"}
    assert subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True).stdout.strip() == "32"
    env["AUDIO_ENGINE_MAX_LANES"] = "5"
    assert subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True).stdout.strip() == "5"
    assert main.EDIT_LIMITS["lanes"] == int(os.environ.get("AUDIO_ENGINE_MAX_LANES", "32"))


def test_lane_limit_enforced(monkeypatch):
    monkeypatch.setitem(main.EDIT_LIMITS, "lanes", 1)
    r = client.post("/edit/levels", data={"project": json.dumps(PROJECT)}, files=files(main=A, take2=B))
    assert r.status_code == 422 and "limits" in r.text


def test_32_lane_project_renders_within_memory(tmp_path):
    """32 lanes (each: an item + delay + compressor + saturation) over 60 s FLAC sources, through the real
    /edit/render endpoint in a fresh interpreter. Peak RSS growth must stay far below the 512 MB instance."""
    import subprocess, sys, textwrap
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    code = textwrap.dedent(f"""
        import io, json, resource, time
        import numpy as np, soundfile as sf
        from fastapi.testclient import TestClient
        from app import main
        sr, n = 44100, 60 * 44100
        rng = np.random.default_rng(1)
        def flac(seed):
            t = np.arange(n) / sr
            x = 0.2 * np.sin(2 * np.pi * (110 + 40 * seed) * t) + 0.02 * rng.standard_normal(n)
            b = io.BytesIO(); sf.write(b, np.stack([x, x]).T, sr, format="FLAC"); return b.getvalue()
        srcs = {{"main": flac(0), "b": flac(1)}}
        lanes = []
        for i in range(32):
            a = (i * 37 * sr // 32) % (50 * sr)
            lanes.append({{"id": f"l{{i}}", "order": i, "gainDb": -18, "pan": (i % 5 - 2) / 2,
                "items": [{{"id": f"i{{i}}", "sourceClipId": "b" if i % 2 else "main",
                           "range": {{"startSample": a, "endSample": a + 8 * sr}}, "laneStartSample": a,
                           "mode": "replace" if i % 3 == 0 else "add"}}],
                "processorChain": [{{"id": f"d{{i}}", "type": "delay", "params": {{"timeMs": 180 + i, "feedback": 0.4}}}},
                                   {{"id": f"c{{i}}", "type": "compressor", "params": {{"thresholdDb": -30}}}},
                                   {{"id": f"s{{i}}", "type": "saturation_tube", "params": {{"driveDb": 6}}}}]}})
        proj = {{"id": "big", "sampleRate": sr, "clips": [{{"id": "main"}}, {{"id": "b"}}], "mainClipId": "main", "lanes": lanes}}
        c = TestClient(main.app, headers={{"X-API-Key": "test-key-123"}})
        base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t = time.time()
        r = c.post("/edit/render", data={{"project": json.dumps(proj), "response": "json"}},
                   files=[(f"clip:{{k}}", (f"{{k}}.flac", v, "audio/flac")) for k, v in srcs.items()])
        assert r.status_code == 200, r.text
        meta = r.json()["meta"]
        assert len(meta["lanes"]) == 32 and all(l["audible"] for l in meta["lanes"])
        print(base // 1024, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, round(time.time() - t, 1))
    """)
    env = {**__import__("os").environ, "AUDIO_ENGINE_STORAGE": str(tmp_path), "AUDIO_ENGINE_API_KEY": "test-key-123"}
    env.pop("AUDIO_ENGINE_MAX_LANES", None)
    out = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    base, peak, secs = out.stdout.split()
    print(f"32 lanes x 60 s: base {base} MB, peak {peak} MB, {secs} s")
    assert int(peak) - int(base) < 250 and int(peak) < 450
