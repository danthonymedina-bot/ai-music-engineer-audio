"""Per-stem processing chain, file -> file, constant memory except where noted:
    denoise (spectral gating, in memory, chunked) -> pitchTempo (Rubber Band CLI, streaming) -> tone (freestyle,
    streamed in blocks with carried filter state) ; the lane bus (gain/pan/mute/solo) is applied later by the
    edit engine. Steps are looked up from STEPS in this fixed order (tests monkeypatch STEPS to check the order).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .dsp import ToneChain

BLOCK = 1 << 16


def _readable(src: Path, work: Path) -> Path:
    try:
        sf.info(str(src))
        return src
    except Exception:
        out = work / f"{src.stem}_decoded.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:a", "pcm_f32le", str(out)],
                       check=True)
        return out


def step_denoise(src: Path, dst: Path, params: dict):
    c = params["denoise"]
    if not c["enabled"]:
        return None
    from ..audio_io import load
    from ..dsp.denoise import denoise
    x, sr = load(src)
    sf.write(str(dst), denoise(x, sr, c["preset"], c["strength"]).T, sr, subtype="FLOAT")
    return {"preset": c["preset"], "strength": c["strength"]}


def step_pitch_tempo(src: Path, dst: Path, params: dict):
    c = params["pitchTempo"]
    if not c["enabled"] or (c["semitones"] == 0 and c["tempo"] == 1):
        return None
    from ..dsp.pitch_tempo import pitch_tempo, rubberband_cli_available, rubberband_file
    if rubberband_cli_available():
        rubberband_file(str(src), str(dst), c["semitones"], c["tempo"])
        return {"engine": "rubberband-cli", "semitones": c["semitones"], "tempo": c["tempo"]}
    from ..audio_io import load
    x, sr = load(src)
    y, eng = pitch_tempo(x, sr, c["semitones"], c["tempo"])
    sf.write(str(dst), np.asarray(y).T, sr, subtype="FLOAT")
    return {"engine": eng, "semitones": c["semitones"], "tempo": c["tempo"]}


def step_tone(src: Path, dst: Path, params: dict):
    info = sf.info(str(src))
    chain = ToneChain(params, info.samplerate)
    peak, clipped = 0.0, 0
    with sf.SoundFile(str(dst), "w", info.samplerate, info.channels, subtype="FLOAT") as out:
        for blk in sf.blocks(str(src), blocksize=BLOCK, dtype="float64", always_2d=True):
            y = chain.process(blk.T)
            peak = max(peak, float(np.max(np.abs(y))) if y.size else 0.0)
            clipped += int(np.count_nonzero(np.abs(y) > 1.0))
            out.write(y.T)
    return {"stages": chain.names, "warnings": chain.warnings, "peak": peak, "samplesOver0dBFS": clipped}


STEPS = [("denoise", step_denoise), ("pitchTempo", step_pitch_tempo), ("tone", step_tone)]


def run_chain(src: Path, out_wav: Path, params: dict, work: Path) -> dict:
    """Runs the steps in order and writes a 16-bit WAV (samples above 0 dBFS are clipped and counted)."""
    work.mkdir(parents=True, exist_ok=True)
    cur, tmp, meta = _readable(Path(src), work), [], {"steps": []}
    try:
        for name, fn in STEPS:
            dst = work / f"_{out_wav.stem}_{name}.wav"
            m = fn(cur, dst, params)
            if m is None:
                continue
            meta["steps"].append(name)
            meta[name] = m
            tmp.append(dst)
            cur = dst
        with sf.SoundFile(str(out_wav), "w", sf.info(str(cur)).samplerate, sf.info(str(cur)).channels,
                          subtype="PCM_16") as out:
            for blk in sf.blocks(str(cur), blocksize=BLOCK, dtype="float32", always_2d=True):
                out.write(np.clip(blk, -1, 1))
    finally:
        for t in tmp:
            t.unlink(missing_ok=True)
        if cur != Path(src) and cur not in tmp:
            Path(cur).unlink(missing_ok=True)
    return meta
