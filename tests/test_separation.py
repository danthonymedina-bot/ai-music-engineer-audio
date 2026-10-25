import numpy as np
import pytest
from tests.conftest import sine
from app.dsp import separation
from app.dsp.dsp_separation import separate_dsp, hpss_masks
from app.dsp.onnx_separation import onnx_available, separate_onnx

SR = 22050


def _mix(sr=SR, dur=3.0):
    t = np.arange(int(dur * sr)) / sr
    vocal = sum(np.sin(2 * np.pi * 440 * k * t) / k for k in range(1, 6)).astype(np.float32) * 0.2
    clicks = np.zeros_like(t, dtype=np.float32)
    clicks[:: int(0.25 * sr)] = 1.0
    drums = np.convolve(clicks, np.exp(-np.arange(200) / 20.0).astype(np.float32) * np.random.default_rng(0).standard_normal(200).astype(np.float32), "same") * 0.3
    bass = (0.3 * np.sin(2 * np.pi * 60 * t)).astype(np.float32)
    side = (0.2 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)  # hard-left instrument
    L = vocal + drums + bass + side
    R = vocal + drums + bass
    return np.stack([L, R]).astype(np.float32), dict(vocal=vocal, drums=drums, bass=bass)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def test_hpss_masks_sum_to_one():
    m = np.abs(np.random.default_rng(0).standard_normal((64, 50))).astype(np.float32)
    h, p = hpss_masks(m)
    assert np.allclose(h + p, 1, atol=1e-4)


def test_dsp_separation_shapes_and_sum():
    x, _ = _mix()
    st = separate_dsp(x, SR, chunk_s=1.0, pad_s=0.25)  # forces chunking
    assert set(st) == {"vocals", "drums", "bass", "other"}
    for v in st.values():
        assert v.shape == x.shape
    recon = sum(st.values())
    assert np.allclose(recon, x, atol=1e-3)  # "other" is the residual


def test_dsp_separation_routes_sources():
    x, src = _mix()
    st = separate_dsp(x, SR)
    assert corr(st["bass"][1], src["bass"]) > 0.9
    assert corr(st["vocals"][1], src["vocal"]) > 0.6
    assert corr(st["drums"][1], src["drums"]) > 0.4
    # hard-panned 700 Hz instrument should mostly not end up in vocals (left channel)
    left_side_leak = np.std(st["vocals"][0] - st["vocals"][1])
    assert left_side_leak < 0.1


def test_dsp_mono_input():
    st = separate_dsp(sine(440, 1.0, sr=SR), SR)
    assert st["vocals"].shape == (2, SR)


def test_backend_resolution(monkeypatch):
    monkeypatch.setenv("SEPARATION_BACKEND", "dsp")
    assert separation.resolve_backend("auto") == "dsp"
    monkeypatch.setenv("SEPARATION_BACKEND", "auto")
    monkeypatch.setattr(separation, "onnx_available", lambda: False)
    monkeypatch.setattr(separation, "demucs_available", lambda: False)
    assert separation.resolve_backend("auto") == "dsp"
    monkeypatch.setattr(separation, "onnx_available", lambda: True)
    assert separation.resolve_backend("auto") == "onnx"
    monkeypatch.setattr(separation, "onnx_available", lambda: False)
    monkeypatch.setenv("SEPARATION_BACKEND", "onnx")
    with pytest.raises(RuntimeError):
        separation.resolve_backend("auto")


def test_separate_writes_files_dsp(monkeypatch, tmp_path):
    import soundfile as sf
    monkeypatch.setenv("SEPARATION_BACKEND", "dsp")
    x, _ = _mix()
    src = tmp_path / "in.wav"
    sf.write(src, x.T, SR)
    out = separation.separate(src, tmp_path / "out", "auto")
    assert set(out) == {"vocals", "drums", "bass", "other"}
    d, sr = sf.read(out["vocals"])
    assert sr == SR and d.shape == (x.shape[1], 2)


@pytest.mark.skipif(not onnx_available(), reason="ONNX model files not present")
def test_onnx_separation_runs():
    sr = 44100
    t = np.arange(int(8 * sr)) / sr  # spans >1 chunk at chunk_s=6
    x = np.stack([np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 220 * t)]).astype(np.float32) * 0.3
    st = separate_onnx(x, sr)
    assert set(st) == {"vocals", "drums", "bass", "other"}
    recon = sum(st.values())
    assert recon.shape == x.shape
    # soft masks sum to 1 -> stems sum back to the mixture
    assert np.max(np.abs(recon - x)) < 1e-2
