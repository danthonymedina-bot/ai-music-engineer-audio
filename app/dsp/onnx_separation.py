"""Lightweight ML separation: Open-Unmix (umxhq) exported to ONNX, int8, run with onnxruntime CPU in chunks.

Model dir ($ONNX_MODEL_DIR) must contain <target>.onnx for vocals/drums/bass/other, each mapping
magnitude (1, 2, 2049, T) -> estimated magnitude (1, 2, 2049, T) at 44.1 kHz (see ondevice/export_onnx.py).
Final stems = soft masks (est / sum est) applied to the mixture STFT (mixture phase), single-iteration Wiener-like.
"""
import os
from pathlib import Path

import numpy as np

SR, N_FFT, HOP = 44100, 4096, 1024
MODEL_FRAMES = 512  # LSTM graph traced at 512 frames (~11.9 s); windows are zero-padded to this
TARGETS = ("vocals", "drums", "bass", "other")
MODEL_DIR = Path(os.environ.get("ONNX_MODEL_DIR", Path(__file__).resolve().parents[2] / "models" / "umxhq"))


def onnx_available(model_dir: Path = MODEL_DIR) -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return all((model_dir / f"{t}.onnx").exists() for t in TARGETS)


_sessions = {}


def _session(path: Path):
    import onnxruntime as ort
    if path not in _sessions:
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("ONNX_THREADS", "1"))
        so.enable_cpu_mem_arena = False  # lower peak RSS on 512 MB hosts
        so.enable_mem_pattern = False
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        _sessions[path] = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    return _sessions[path]


def _run_model(sess, mag: np.ndarray) -> np.ndarray:
    """mag (2, 2049, F). Model has a fixed frame count (static LSTM export); window + zero-pad."""
    inp = sess.get_inputs()[0]
    T = inp.shape[3] if isinstance(inp.shape[3], int) else MODEL_FRAMES
    F = mag.shape[2]
    out = np.zeros_like(mag)
    for f0 in range(0, F, T):
        w = mag[:, :, f0:f0 + T]
        k = w.shape[2]
        if k < T:
            w = np.pad(w, ((0, 0), (0, 0), (0, T - k)))
        out[:, :, f0:f0 + k] = sess.run(None, {inp.name: w[None].astype(np.float32)})[0][0][:, :, :k]
    return np.maximum(out, 0)


def iter_onnx(x: np.ndarray, sr: int, model_dir: Path = MODEL_DIR, chunk_s: float = float(os.environ.get("ONNX_CHUNK_S", "6"))):
    """x must be 44.1 kHz. Yields dict[stem -> (2, n_chunk)]."""
    from scipy.signal import istft, stft
    if sr != SR:
        raise ValueError("resample to 44.1 kHz first")
    x = np.atleast_2d(np.asarray(x, dtype=np.float32))
    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)
    x = x[:2]
    n = x.shape[1]
    chunk, pad = int(chunk_s * SR), 2 * N_FFT
    sessions = [_session(model_dir / f"{t}.onnx") for t in TARGETS]
    for start in range(0, n, chunk):
        a, b = max(0, start - pad), min(n, start + chunk + pad)
        seg = x[:, a:b]
        # scipy scales by 1/sum(hann)=2/N_FFT; model was trained on unnormalised torch.stft magnitudes
        S = stft(seg, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="even", padded=True)[2].astype(np.complex64)
        mag = (np.abs(S) * (N_FFT / 2)).astype(np.float32)  # (2, 2049, F)
        ests = [_run_model(s, mag) ** 2 for s in sessions]  # power estimates
        del mag
        total = np.add.reduce(ests) + np.float32(1e-10)
        lo = start - a
        hi = lo + min(chunk, n - start)
        res = {}
        for e in ests:
            e /= total
        # residual mask -> "other" so stems always sum to the mixture (bins where the model predicts ~0 everywhere)
        ests[TARGETS.index("other")] = np.maximum(
            1 - sum(e for t, e in zip(TARGETS, ests) if t != "other"), 0).astype(np.float32)
        for t, e in zip(TARGETS, ests):
            y = istft(S * e, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=True)[1]
            y = np.pad(y, ((0, 0), (0, max(0, seg.shape[1] - y.shape[1]))))[:, :seg.shape[1]]
            res[t] = y[:, lo:hi].astype(np.float32)
            del y
        del S, ests, total
        yield res


def separate_onnx(x, sr, **kw) -> dict:
    parts = list(iter_onnx(x, sr, **kw))
    return {k: np.concatenate([p[k] for p in parts], axis=1) for k in parts[0]}
