import io
import json
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient
from tests.conftest import sine, SR
from app import main
from app.dsp import separation

client = TestClient(main.app, headers={"X-API-Key": "test-key-123"})
anon = TestClient(main.app)


def wav_bytes(x, sr=SR):
    b = io.BytesIO()
    sf.write(b, x, sr, format="WAV")
    return b.getvalue()


def up(x=None, name="a.wav"):
    return (name, wav_bytes(sine(440) if x is None else x), "audio/wav")


def fetch(url):
    r = client.get(url.replace("http://testserver", ""))
    assert r.status_code == 200
    return sf.read(io.BytesIO(r.content))


def test_health():
    assert client.get("/health").json()["ok"]


def test_separate_mocked(monkeypatch, tmp_path):
    def fake(src, out_dir, model):
        out_dir.mkdir(parents=True, exist_ok=True)
        res = {}
        for s in ["vocals", "drums", "bass", "other"]:
            p = out_dir / f"{s}.wav"
            sf.write(p, sine(220), SR)
            res[s] = p
        return res
    monkeypatch.setattr(separation, "separate", fake)
    r = client.post("/separate", files={"file": up()}, data={"model": "htdemucs"})
    assert r.status_code == 202
    jid = r.json()["job_id"]
    j = client.get(f"/jobs/{jid}").json()  # TestClient runs background tasks before returning
    assert j["status"] == "done" and set(j["urls"]) == {"vocals", "drums", "bass", "other"}
    data, sr = fetch(j["urls"]["vocals"])
    assert sr == SR and len(data) == SR


def test_separate_backend_failure(monkeypatch):
    monkeypatch.setattr(separation, "separate", lambda *a: (_ for _ in ()).throw(RuntimeError("no backend")))
    jid = client.post("/separate", files={"file": up()}).json()["job_id"]
    j = client.get(f"/jobs/{jid}").json()
    assert j["status"] == "error" and "no backend" in j["error"]
    assert client.get(f"/stems/{jid}/vocals").status_code == 409


def test_bad_model_and_type():
    assert client.post("/separate", files={"file": up()}, data={"model": "x"}).status_code == 422
    assert client.post("/separate", files={"file": ("a.txt", b"hi", "text/plain")}).status_code == 415
    assert client.post("/separate", files={"file": ("a.wav", b"not audio", "audio/wav")}).status_code == 415


def test_unknown_job():
    assert client.get("/jobs/doesnotexist").status_code == 404
    assert client.get("/stems/../etc").status_code == 404


def test_pitch_tempo_ep():
    r = client.post("/process/pitch-tempo", files={"file": up()}, data={"semitones": 2, "tempo": 1.5, "engine": "librosa"})
    assert r.status_code == 200, r.text
    data, _ = fetch(r.json()["urls"]["output"])
    assert abs(len(data) - SR / 1.5) < SR * 0.05
    assert client.post("/process/pitch-tempo", files={"file": up()}, data={"tempo": 10}).status_code == 422


def test_tone_ep():
    r = client.post("/process/tone", files={"file": up()}, data={"low_gain_db": 3, "drive": 6, "mix": 0.5})
    assert r.status_code == 200, r.text
    assert r.json()["params"]["mix"] == 0.5
    assert client.post("/process/tone", files={"file": up()}, data={"low_gain_db": 99}).status_code == 422


def test_denoise_ep():
    rng = np.random.default_rng(1)
    x = sine(440, 2) + 0.05 * rng.standard_normal(2 * SR).astype(np.float32)
    r = client.post("/process/denoise", files={"file": up(x)}, data={"preset": "wind"})
    assert r.status_code == 200, r.text
    assert client.post("/process/denoise", files={"file": up()}, data={"preset": "x"}).status_code == 422


def test_mix_ep():
    files = [("files", up(sine(440, amp=0.9), "a.wav")), ("files", up(sine(660, amp=0.9), "b.wav"))]
    tracks = [{"gain_db": 0, "pan": -0.5}, {"gain_db": -3, "pan": 0.5, "offset_ms": 250}]
    r = client.post("/mix", files=files, data={"tracks": json.dumps(tracks), "ceiling_db": -1})
    assert r.status_code == 200, r.text
    data, sr = fetch(r.json()["urls"]["output"])
    assert data.shape[1] == 2 and len(data) == SR + SR // 4
    assert np.max(np.abs(data)) <= 10 ** (-1 / 20) + 1e-3
    bad = client.post("/mix", files=files, data={"tracks": json.dumps([{}])})
    assert bad.status_code == 422


def test_auth_required():
    assert anon.get("/health").status_code == 200
    assert anon.get("/jobs/abc").status_code == 401
    assert anon.post("/process/tone", files={"file": up()}).status_code == 401
    bad = TestClient(main.app, headers={"X-API-Key": "wrong"})
    assert bad.post("/separate", files={"file": up()}).status_code == 401
    assert bad.get("/stems/a/b").status_code == 401


def test_missing_server_key(monkeypatch):
    monkeypatch.delenv("AUDIO_ENGINE_API_KEY")
    assert client.get("/jobs/abc").status_code == 503


def test_rate_limit_separate(monkeypatch):
    from app import security
    monkeypatch.setitem(security.LIMITS, "separate", (3, 3600))
    monkeypatch.setattr(separation, "separate", lambda *a: {})
    codes = [client.post("/separate", files={"file": up()}).status_code for _ in range(4)]
    assert codes == [202, 202, 202, 429]
    r = client.post("/separate", files={"file": up()})
    assert r.status_code == 429 and int(r.headers["Retry-After"]) > 0
    # other client (different IP) unaffected
    other = TestClient(main.app, headers={"X-API-Key": "test-key-123", "X-Forwarded-For": "1.2.3.4"})
    assert other.post("/separate", files={"file": up()}).status_code == 202


def test_rate_limit_process_bucket_separate_from_separate(monkeypatch):
    from app import security
    monkeypatch.setitem(security.LIMITS, "process", (2, 3600))
    for _ in range(2):
        assert client.post("/process/tone", files={"file": up()}).status_code == 200
    assert client.post("/process/denoise", files={"file": up()}).status_code == 429
    monkeypatch.setattr(separation, "separate", lambda *a: {})
    assert client.post("/separate", files={"file": up()}).status_code == 202


def test_default_limits():
    from app.security import LIMITS
    assert LIMITS["separate"] == (10, 3600) and LIMITS["process"] == (60, 3600)


def test_upload_size_limit(monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_MB", 0)
    assert client.post("/process/tone", files={"file": up()}).status_code == 413


def test_duration_limit(monkeypatch):
    monkeypatch.setattr(main, "MAX_DURATION_S", 0.5)
    r = client.post("/process/tone", files={"file": up()})  # 1 s clip
    assert r.status_code == 413 and "longer" in r.json()["detail"]
