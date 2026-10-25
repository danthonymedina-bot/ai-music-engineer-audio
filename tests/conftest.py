import os, tempfile
os.environ.setdefault("AUDIO_ENGINE_STORAGE", tempfile.mkdtemp(prefix="ae-test-"))
os.environ["AUDIO_ENGINE_API_KEY"] = "test-key-123"
import numpy as np
import pytest

SR = 22050


@pytest.fixture
def sr():
    return SR


def sine(f, dur=1.0, sr=SR, amp=0.5):
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def band_energy(x, sr, f, bw=0.05):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    fr = np.fft.rfftfreq(len(x), 1 / sr)
    m = (fr > f * (1 - bw)) & (fr < f * (1 + bw))
    return X[m].max()


def dominant_freq(x, sr):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / sr)[np.argmax(X)]


@pytest.fixture(autouse=True)
def _reset_limiter():
    from app.security import limiter
    limiter.reset()
    yield
