import numpy as np
import pytest
from tests.conftest import sine, band_energy, dominant_freq, SR
from app.dsp.tone import eq, saturate, tone
from app.dsp.mix import Track, mix, pan_gains, limiter
from app.dsp.pitch_tempo import pitch_tempo
from app.dsp.denoise import denoise, highpass


def db(r):
    return 20 * np.log10(r)


@pytest.mark.parametrize("kw,freq,expected", [
    (dict(low_gain_db=6), 50, 6), (dict(high_gain_db=-6), 9000, -6), (dict(mid_gain_db=9), 1000, 9)])
def test_eq_bands(kw, freq, expected):
    x = sine(freq, 2)
    y = eq(x, SR, **kw)
    g = db(band_energy(y, SR, freq) / band_energy(x, SR, freq))
    assert abs(g - expected) < 1.0


def test_eq_flat_is_identity():
    x = sine(440)
    assert np.allclose(eq(x, SR), x, atol=1e-6)


def test_eq_stereo_shape():
    x = np.stack([sine(100), sine(200)])
    assert eq(x, SR, low_gain_db=3).shape == x.shape


def test_saturation_bounded_and_harmonics():
    x = sine(440, amp=1.0)
    y = saturate(x, drive=20, mix=1)
    assert np.max(np.abs(y)) <= 1.0001
    assert band_energy(y, SR, 1320) > 10 * band_energy(x, SR, 1320)  # odd harmonic added
    assert np.allclose(saturate(x, drive=20, mix=0), x)


def test_tone_chain():
    assert tone(sine(440), SR, low=3, drive=6, mix=0.5).shape == (SR,)


def test_pan_constant_power():
    for p in np.linspace(-1, 1, 9):
        l, r = pan_gains(p)
        assert abs(l * l + r * r - 1) < 1e-9


def test_mix_offset_gain_pan():
    a = sine(440, 0.5)
    out = mix([Track(a, gain_db=-6, pan=-1, offset_ms=100)], SR, limit=False)
    off = int(0.1 * SR)
    assert out.shape == (2, off + len(a))
    assert np.allclose(out[:, :off], 0)
    assert np.allclose(out[1], 0, atol=1e-6)  # hard left
    expected_peak = 0.5 * 10 ** (-6 / 20) * np.sqrt(2)
    assert abs(out[0].max() - expected_peak) < 1e-3


def test_mix_center_unity_and_sum():
    a = sine(440, 0.5, amp=0.2)
    out = mix([Track(a), Track(a)], SR, limit=False)
    assert np.allclose(out[0], 2 * a, atol=1e-5)


def test_limiter_ceiling():
    x = np.stack([sine(100, amp=3.0)] * 2)
    y = limiter(x, -1.0, sr=SR)
    assert np.max(np.abs(y)) <= 10 ** (-1 / 20) + 1e-6
    q = np.stack([sine(100, amp=0.1)] * 2)
    assert np.allclose(limiter(q, -1.0, sr=SR), q)


def test_pitch_shift_octave_librosa():
    y, eng = pitch_tempo(sine(440, 1), SR, semitones=12, engine="librosa")
    assert eng == "librosa"
    assert abs(dominant_freq(y, SR) - 880) < 15


def test_time_stretch_length():
    y, _ = pitch_tempo(sine(440, 1), SR, tempo=2.0, engine="librosa")
    assert abs(len(y) - SR / 2) < SR * 0.05
    assert abs(dominant_freq(y, SR) - 440) < 15


def test_stereo_pitch_tempo():
    x = np.stack([sine(440), sine(440)])
    y, _ = pitch_tempo(x, SR, tempo=0.5, engine="librosa")
    assert y.shape[0] == 2 and y.shape[1] > 1.9 * SR


def test_highpass_kills_rumble():
    x = sine(30, 2) + sine(1000, 2)
    y = highpass(x, SR, 100)
    assert band_energy(y, SR, 30) < 0.1 * band_energy(x, SR, 30)
    assert band_energy(y, SR, 1000) > 0.9 * band_energy(x, SR, 1000)


@pytest.mark.parametrize("preset", ["hiss", "air", "wind"])
def test_denoise_improves_snr(preset):
    rng = np.random.default_rng(0)
    # musical-ish: 440 Hz note bursts (0.4 s on / 0.4 s off) after a noise-only lead-in
    clean = sum(sine(220 * k, 4, amp=0.3 / k) for k in range(1, 8))  # harmonic 'note'
    t = np.arange(len(clean)) / SR
    clean *= ((t > 1.0) & ((t % 0.8) < 0.4)).astype(np.float32)
    noisy = clean + 0.05 * rng.standard_normal(len(clean)).astype(np.float32)
    y = denoise(noisy, SR, preset)
    n_before = np.std(noisy[: SR // 2])
    n_after = np.std(y[: SR // 2])
    assert n_after < 0.7 * n_before
    assert band_energy(y, SR, 440) > 0.5 * band_energy(clean, SR, 440)  # 2nd harmonic survives
    # overall SNR improvement
    ref = highpass(clean, SR, 100) if preset == "wind" else clean  # HPF phase shift is expected, not noise
    snr = lambda z: 10 * np.log10(np.sum(ref ** 2) / np.sum((z - ref) ** 2))
    assert snr(y) > snr(noisy if preset != 'wind' else highpass(noisy, SR, 100)) + (3 if preset == 'hiss' else 1.5)


def test_denoise_bad_preset():
    with pytest.raises(ValueError):
        denoise(sine(440), SR, "nope")
