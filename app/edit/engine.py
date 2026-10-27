"""Sample-accurate, stateless (block-exact) render engine for the edit model.

Signal flow per lane (all integer-sample math, float64 internally):
    items (list order) -> [item fade envelope + crossfade]  ->  lane composite
    composite -> processorChain (each with its own fade envelope)  ->  lane gain  ->  pan  ->  mute/solo
master = main clip (from timeline 0, mainGainDb) + sum(lanes)  ->  master limiter (transparent below threshold).

Replace/add geometry for an item of length L placed at P with crossfade X samples (X = min(lengthMs, L)),
h = X // 2 (the crossfade is centred on each boundary):
    start window [P-h, P-h+X): new fades in (w: 0->1), old (replace only) fades out
    body         [P-h+X, P+L-h): new at full weight; replace mutes the old content
    end window   [P+L-h, P+L-h+X): new fades out (w: 1->0), old fades back in
    w[i] = (i + 0.5) / X inside a window, so sum(w) over the item is exactly L samples.
New audio in the crossfade windows comes from the source just outside the range ("handles"); zero if the
source has no audio there. The item's fadeIn/fadeOut envelope is applied on [P, P+L) only to the new audio,
before the crossfade weights, and both happen before lane gain/pan, so none of them interact.

Everything is a pure function of the timeline sample, so render_block(start, n) for any start/n
concatenates to exactly the full render and both views share one length and alignment.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional

import numpy as np

from .model import Lane, LaneItem, Project
from .processors import BaseProcessor, build

View = Literal["main", "main_plus_lanes"]
DB_FLOOR = -150.0
CEILING = 10 ** (-0.1 / 20)  # soft clip asymptote: master output never reaches 0 dBFS


class EditError(ValueError):
    pass


# ---------------------------------------------------------------- sources

def as_stereo_view(a: np.ndarray) -> np.ndarray:
    """(N,), (1,N), (2,N) or (N,2)-memmap-transposed -> something indexable as [ch, a:b] with 2 channels."""
    if isinstance(a, (np.memmap, DiskSource)):
        return a
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[None, :]
    if a.shape[0] > 2:
        a = a[:2]
    return a


def read_source(src: np.ndarray, a: int, b: int) -> np.ndarray:
    """Samples [a, b) of a (C, N) source as (2, b-a) float64; zeros outside [0, N)."""
    out = np.zeros((2, max(0, b - a)), dtype=np.float64)
    n = src.shape[1]
    lo, hi = max(a, 0), min(b, n)
    if hi > lo:
        seg = np.asarray(src[:, lo:hi], dtype=np.float64)
        out[:, lo - a: hi - a] = seg if seg.shape[0] == 2 else np.repeat(seg[:1], 2, axis=0)
    return out


def decode_to_disk(path: str, sr: int, workdir: str) -> np.ndarray:
    """Decode any ffmpeg-readable file to float32 stereo at `sr` on disk and read it on demand (constant RAM)."""
    raw = os.path.join(workdir, os.path.basename(path) + ".f32")
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", path, "-f", "f32le", "-acodec",
                            "pcm_f32le", "-ac", "2", "-ar", str(sr), raw], capture_output=True)
        if r.returncode != 0:
            raise EditError(f"could not decode {os.path.basename(path)}")
    else:  # pragma: no cover - ffmpeg is in the image
        from scipy.signal import resample_poly
        from ..audio_io import load
        x, fsr = load(path)
        if fsr != sr:
            g = math.gcd(fsr, sr)
            x = resample_poly(x, sr // g, fsr // g, axis=1).astype(np.float32)
        x = x if x.shape[0] == 2 else np.repeat(x[:1], 2, axis=0)
        np.ascontiguousarray(x.T, dtype="<f4").tofile(raw)
    frames = os.path.getsize(raw) // 8
    if frames == 0:
        raise EditError(f"{os.path.basename(path)} decoded to zero samples")
    return DiskSource(raw, frames)


class DiskSource:
    """Read-on-demand float32 stereo source (interleaved f32le file). Unlike np.memmap, pages that were read do
    not stay in the process RSS, so a render's memory is bounded by the block size, not by the clip lengths."""

    def __init__(self, path: str, frames: int):
        self.path, self.shape, self.ndim = path, (2, frames), 2

    def __getitem__(self, key):
        ch, sl = key
        lo, hi, _ = sl.indices(self.shape[1])
        if hi <= lo:
            return np.zeros((2, 0), dtype=np.float32)[ch]
        a = np.fromfile(self.path, dtype="<f4", count=(hi - lo) * 2, offset=lo * 8)
        return a.reshape(-1, 2).T[ch]


# ---------------------------------------------------------------- geometry

def db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


def to_db(peak: float) -> float:
    return round(20 * math.log10(peak), 3) if peak > 0 else DB_FLOOR


@dataclass(frozen=True)
class ItemGeom:
    item: LaneItem
    P: int
    L: int
    X: int
    h: int
    fi: int
    fo: int
    gain: float
    curve: str
    hard_in: bool = False   # contiguous seam with the previous piece of the same source: no crossfade
    hard_out: bool = False

    @property
    def start(self) -> int:  # first sample where the item contributes
        return self.P if self.hard_in else self.P - self.h

    @property
    def end(self) -> int:  # one past the last contributing sample
        return self.P + self.L if self.hard_out else self.P + self.L - self.h + self.X

    def weight(self, t: np.ndarray) -> np.ndarray:
        """Crossfade position w(t) in [0, 1] (1 = new content fully in)."""
        X = max(self.X, 1)
        up = np.where(t >= self.P, 1.0, 0.0) if self.hard_in else (t - (self.P - self.h) + 0.5) / X
        down = (np.where(t < self.P + self.L, 1.0, 0.0) if self.hard_out
                else ((self.P + self.L - self.h + X) - t - 0.5) / X)
        return np.clip(np.minimum(up, down), 0.0, 1.0)

    def envelope(self, t: np.ndarray) -> np.ndarray:
        """Item fadeIn/fadeOut on [P, P+L); outside that range it holds its edge value (0 if a fade is set)."""
        e = np.ones(t.shape, dtype=np.float64)
        if self.fi > 0:
            e *= np.clip((t - self.P + 0.5) / self.fi, 0.0, 1.0)
        if self.fo > 0:
            e *= np.clip((self.P + self.L - t - 0.5) / self.fo, 0.0, 1.0)
        return e * self.gain


def item_geom(it: LaneItem, sr: int) -> ItemGeom:
    L = it.range.length
    X = min(int(round(it.crossfade.lengthMs * sr / 1000.0)), L)
    return ItemGeom(item=it, P=it.laneStartSample, L=L, X=X, h=X // 2,
                    fi=min(int(round(it.fadeInMs * sr / 1000.0)), L), fo=min(int(round(it.fadeOutMs * sr / 1000.0)), L),
                    gain=db_to_gain(it.gainDb), curve=it.crossfade.curve)


def mark_seams(geoms: list[ItemGeom]) -> list[ItemGeom]:
    """Contiguous seam rule: if item B continues item A exactly (same source, mode, gain and curve; A's range end ==
    B's range start; A's lane end == B's lane start) the boundary is not a splice, so neither side crossfades there.
    This is what makes Split lossless: split + render == render, bit for bit."""
    from dataclasses import replace
    out = list(geoms)
    for i, a in enumerate(geoms):
        for j, b in enumerate(geoms):
            ia, ib = a.item, b.item
            if i != j and ia.sourceClipId == ib.sourceClipId and ia.mode == ib.mode and ia.gainDb == ib.gainDb \
                    and ia.crossfade == ib.crossfade and ia.range.endSample == ib.range.startSample \
                    and ia.laneEndSample == ib.laneStartSample:
                out[i] = replace(out[i], hard_out=True)
                out[j] = replace(out[j], hard_in=True)
    return out


def curves(w: np.ndarray, curve: str):
    if curve == "linear":
        return 1.0 - w, w
    return np.cos(w * (np.pi / 2)), np.sin(w * (np.pi / 2))


def extract_slice(src: np.ndarray, it: LaneItem) -> np.ndarray:
    """The raw (un-faded) audio of an item: exactly range.length samples."""
    return read_source(src, it.range.startSample, it.range.endSample)


# ---------------------------------------------------------------- renderer

@dataclass
class _LaneRT:
    lane: Lane
    geoms: list[ItemGeom]
    procs: list[BaseProcessor]
    span: tuple[int, int]
    lookbehind: int
    tail: int
    gl: float
    gr: float
    audible: bool


@dataclass
class _Meter:
    peak: float = 0.0
    sumsq: float = 0.0

    def add(self, buf: np.ndarray):
        if buf.size:
            self.peak = max(self.peak, float(np.max(np.abs(buf))))
            self.sumsq += float(np.dot(buf.ravel(), buf.ravel()))

    def report(self, n: int) -> dict:
        rms = math.sqrt(self.sumsq / (2 * n)) if n else 0.0  # RMS over both channels and the whole rendered span
        return {"peakDbfs": to_db(self.peak), "rmsDbfs": to_db(rms), "headroomDb": round(-to_db(self.peak), 3)}


@dataclass
class RenderStats:
    """Meters measured on what is actually heard: lanes after their bus (gain, pan, mute/solo), master after the
    limiter. RMS is over the full rendered span (silence counts), both channels."""
    lanes: dict[str, _Meter] = field(default_factory=dict)
    main: _Meter = field(default_factory=_Meter)
    master_pre: _Meter = field(default_factory=_Meter)
    master: _Meter = field(default_factory=_Meter)
    samples: int = 0
    clipped_samples: int = 0  # samples the soft clip touched

    def lane(self, lane_id: str) -> _Meter:
        return self.lanes.setdefault(lane_id, _Meter())

    def as_dict(self, renderer: "Renderer", view: str, start: int = 0) -> dict:
        n = self.samples
        lanes = [{"laneId": rt.lane.id, "audible": rt.audible, **self.lane(rt.lane.id).report(n)}
                 for rt in renderer.lanes]
        return {
            "view": view, "sampleRate": renderer.sr, "lengthSamples": renderer.length,
            "measuredStartSample": start, "measuredSamples": n,
            "lanes": lanes,
            "main": self.main.report(n),
            "master": {**self.master.report(n), "prePeakDbfs": to_db(self.master_pre.peak),
                       "limiter": renderer.project.masterLimiter,
                       "gainCompensationDb": round(20 * math.log10(renderer.comp_gain), 3),
                       "softClipThresholdDb": renderer.project.softClipThresholdDb,
                       "ceilingDbfs": round(20 * math.log10(CEILING), 3),
                       "softClippedSamples": self.clipped_samples},
        }


class Renderer:
    def __init__(self, project: Project, sources: Mapping[str, np.ndarray]):
        self.project = project
        self.sr = project.sampleRate
        missing = {c.id for c in project.clips} - set(sources)
        if missing:
            raise EditError(f"missing audio for clips: {sorted(missing)}")
        self.sources = {k: as_stereo_view(v) for k, v in sources.items()}
        self.main = self.sources[project.mainClipId] if project.mainClipId else None
        self.main_gain = 0.0 if project.mainMute else db_to_gain(project.mainGainDb)
        any_solo = any(l.solo for l in project.lanes)
        self.lanes: list[_LaneRT] = []
        for lane in project.ordered_lanes():
            geoms = []
            for it in lane.items:
                n = self.sources[it.sourceClipId].shape[1]
                if it.range.endSample > n:
                    raise EditError(f"item '{it.id}' range ends at {it.range.endSample} but clip "
                                    f"'{it.sourceClipId}' has {n} samples")
                geoms.append(item_geom(it, self.sr))
            geoms = mark_seams(geoms)
            procs = [build(p, self.sr, project.snap) for p in lane.processorChain]
            active = [p for p in procs if p.spec.enabled]
            span = (min(g.start for g in geoms), max(g.end for g in geoms)) if geoms else (0, 0)
            gl, gr = (math.cos((lane.pan + 1) * math.pi / 4) * math.sqrt(2),
                      math.sin((lane.pan + 1) * math.pi / 4) * math.sqrt(2))
            g = db_to_gain(lane.gainDb)
            self.lanes.append(_LaneRT(lane, geoms, active, span, sum(p.lookbehind() for p in active),
                                      sum(p.tail() for p in active), gl * g, gr * g,
                                      bool(geoms) and not lane.mute and (lane.solo or not any_solo)))
        ends = [self.main.shape[1] if self.main is not None else 0]
        ends += [rt.span[1] + rt.tail for rt in self.lanes if rt.geoms]
        self.length = max(0, max(ends))
        self.thr = db_to_gain(project.softClipThresholdDb)
        self.comp_gain = 1.0
        self._comp_ready = project.masterLimiter != "gain_compensation"

    # -- lanes
    def compose(self, rt: _LaneRT, t0: int, t1: int) -> np.ndarray:
        buf = np.zeros((2, t1 - t0), dtype=np.float64)
        for g in rt.geoms:
            a, b = max(t0, g.start), min(t1, g.end)
            if a >= b:
                continue
            t = np.arange(a, b, dtype=np.float64)
            w = g.weight(t)
            g_old, g_new = curves(w, g.curve)
            src = self.sources[g.item.sourceClipId]
            rs = g.item.range.startSample
            new = read_source(src, rs + (a - g.P), rs + (b - g.P)) * (g.envelope(t) * g_new)
            seg = buf[:, a - t0: b - t0]
            if g.item.mode == "replace":
                seg *= g_old
            seg += new
        return buf

    def lane_block(self, rt: _LaneRT, start: int, n: int) -> np.ndarray:
        lb = rt.lookbehind
        x = self.compose(rt, start - lb, start + n)
        for p in rt.procs:
            x = p.apply(x, start - lb, rt.span[0], rt.span[1])
        x = x[:, lb:]
        x[0] *= rt.gl
        x[1] *= rt.gr
        return x

    def _active(self, rt: _LaneRT, start: int, n: int) -> bool:
        return rt.audible and start < rt.span[1] + rt.tail and start + n > rt.span[0]

    # -- master
    def soft_clip(self, y: np.ndarray, stats: Optional[RenderStats] = None) -> np.ndarray:
        T, C = self.thr, CEILING
        a = np.abs(y)
        over = a > T
        if over.any():
            y = y.copy()
            y[over] = np.sign(y[over]) * (T + (C - T) * np.tanh((a[over] - T) / (C - T)))
            if stats is not None:
                stats.clipped_samples += int(over.sum())
        return y

    def _prepare(self):
        if self._comp_ready:
            return
        self._comp_ready = True
        peak = 0.0
        for view in ("main", "main_plus_lanes"):  # both views share one gain so A/B switching is level-matched
            for s in range(0, self.length, BLOCK):
                peak = max(peak, float(np.max(np.abs(self._premaster(s, min(BLOCK, self.length - s), view)))))
        self.comp_gain = min(1.0, self.thr / peak) if peak > 0 else 1.0

    def _premaster(self, start: int, n: int, view: View, stats: Optional[RenderStats] = None) -> np.ndarray:
        y = np.zeros((2, n), dtype=np.float64)
        if self.main is not None and self.main_gain > 0:
            m = read_source(self.main, start, start + n) * self.main_gain
            if stats is not None:
                stats.main.add(m)
            y += m
        if view == "main_plus_lanes":
            for rt in self.lanes:
                if self._active(rt, start, n):
                    lb = self.lane_block(rt, start, n)
                    if stats is not None:
                        stats.lane(rt.lane.id).add(lb)
                    y += lb
        return y

    def render_block(self, start: int, n: int, view: View = "main_plus_lanes",
                     stats: Optional[RenderStats] = None) -> np.ndarray:
        """Master output for timeline samples [start, start+n) as (2, n) float32. Pure function of (start, n)."""
        if view not in ("main", "main_plus_lanes"):
            raise EditError(f"unknown view '{view}'")
        if start < 0 or n < 0:
            raise EditError("start and n must be >= 0")
        self._prepare()
        y = self._premaster(start, n, view, stats)
        if self.comp_gain != 1.0:
            y *= self.comp_gain
        if stats is not None:
            stats.master_pre.add(y)
        y = self.soft_clip(y, stats)
        if stats is not None:
            stats.master.add(y)
            stats.samples += n
        return y.astype(np.float32)

    def render(self, view: View = "main_plus_lanes", stats: Optional[RenderStats] = None) -> np.ndarray:
        return self.render_block(0, self.length, view, stats)

    def iter_blocks(self, view: View = "main_plus_lanes", stats: Optional[RenderStats] = None, block: int = 0):
        block = block or BLOCK
        for s in range(0, self.length, block):
            yield self.render_block(s, min(block, self.length - s), view, stats)


BLOCK = 1 << 17  # ~3 s at 44.1 kHz: keeps /edit/render well under 512 MB


def render_block(project: Project, sources: Mapping[str, np.ndarray], start: int, n: int,
                 view: View = "main_plus_lanes") -> np.ndarray:
    return Renderer(project, sources).render_block(start, n, view)


def render(project: Project, sources: Mapping[str, np.ndarray], view: View = "main_plus_lanes"):
    """Full render -> (audio (2, N) float32, metadata dict)."""
    r = Renderer(project, sources)
    st = RenderStats()
    y = r.render(view, st)
    return y, st.as_dict(r, view)


def render_to_wav(project: Project, sources: Mapping[str, np.ndarray], path: str,
                  view: View = "main_plus_lanes", subtype: str = "PCM_16") -> dict:
    """Streams blocks to a WAV file (constant memory). Returns metadata."""
    import soundfile as sf
    r = Renderer(project, sources)
    st = RenderStats()
    with sf.SoundFile(path, "w", samplerate=r.sr, channels=2, subtype=subtype) as f:
        for blk in r.iter_blocks(view, st):
            f.write(blk.T)
    return st.as_dict(r, view)
