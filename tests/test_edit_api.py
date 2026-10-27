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
