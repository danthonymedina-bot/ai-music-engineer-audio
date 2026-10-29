"""AI Music Engineer - audio processing backend."""
import json
import subprocess
import re
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from . import jobs
from .audio_io import duration_s, load, save
from .config import DEFAULT_MODEL, MAX_DURATION_S, MAX_UPLOAD_MB
from .security import check_key, rate_limit
from .dsp import separation
from .dsp.denoise import PRESETS, denoise
from .dsp.mix import Track, mix
from .dsp.pitch_tempo import pitch_tempo, rubberband_available, rubberband_cli_available, rubberband_file
from .dsp.tone import tone
from .tone.chain import run_chain
from .tone.profile import ProfileError, parse_profile, schema_doc

import threading
SEPARATION_LOCK = threading.Lock()
TONE_SLOTS = threading.BoundedSemaphore(2)  # chain = denoise (chunked) + streaming; 2 at once fits 512 MB
app = FastAPI(title="AI Music Engineer Audio Engine", version="0.2.0")
AUTH = [Depends(check_key)]
SEP = AUTH + [Depends(rate_limit("separate"))]
PROC = AUTH + [Depends(rate_limit("process"))]
SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")
ALLOWED_EXT = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"}


def _save_upload(f: UploadFile, dest_dir: Path, name: str = "input") -> Path:
    ext = Path(f.filename or "").suffix.lower() or ".wav"
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, f"unsupported file type {ext}")
    dest = dest_dir / f"{name}{ext}"
    cap, total = MAX_UPLOAD_MB * 1024 * 1024, 0
    with dest.open("wb") as out:
        while chunk := f.file.read(1 << 20):
            total += len(chunk)
            if total > cap:
                out.close()
                dest.unlink()
                raise HTTPException(413, f"file exceeds {MAX_UPLOAD_MB} MB")
            out.write(chunk)
    dur = duration_s(dest)
    if dur is None:
        dest.unlink()
        raise HTTPException(415, "could not decode audio")
    if dur > MAX_DURATION_S:
        dest.unlink()
        raise HTTPException(413, f"audio longer than {MAX_DURATION_S / 60:g} min")
    return dest


def _job_view(j: dict, request: Request) -> dict:
    base = str(request.base_url).rstrip("/")
    pub = {k: v for k, v in j.items() if k != "outputs"}  # don't leak server paths
    return {**pub, "urls": {k: f"{base}/stems/{j['id']}/{k}" for k in j["outputs"]}}


def _run(jid: str, fn):
    jobs.update(jid, status="running")
    try:
        outputs, meta = fn()
        jobs.update(jid, status="done", outputs={k: str(v) for k, v in outputs.items()}, meta=meta)
    except Exception as e:  # noqa: BLE001
        jobs.update(jid, status="error", error=f"{type(e).__name__}: {e}")


@app.get("/health")
def health():
    try:
        b = separation.resolve_backend("auto")
    except RuntimeError:
        b = None
    return {"ok": True, "separation_backend": b, "quality": separation.QUALITY.get(b),
            "demucs": separation.demucs_available(), "onnx": separation.onnx_available(),
            "rubberband": rubberband_available()}


@app.post("/separate", status_code=202, dependencies=SEP)
def separate_ep(bg: BackgroundTasks, request: Request, file: UploadFile = File(...), model: str = Form(DEFAULT_MODEL),
                toneProfile: Optional[str] = Form(None)):
    """Optional toneProfile (JSON, freestyle or preset, or a phone FreestylePreset export) is applied to every stem
    after separation (stems.<name> overrides per stem). Processed stems keep their names; dry stems are <name>_raw."""
    prof = _parse_profile_field(toneProfile)
    if model not in separation.MODELS:
        raise HTTPException(422, f"unknown model {model}")
    try:
        backend = separation.resolve_backend(model)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    jid = jobs.create("separate", {"model": model, "backend": backend, "quality": separation.QUALITY[backend]})
    d = jobs.job_dir(jid)
    src = _save_upload(file, d)

    def work():
        with SEPARATION_LOCK:  # one separation (+ its streaming tone pass) at a time: stays within 512 MB
            stems = separation.separate(src, d / "stems", model)
            if not stems:
                raise RuntimeError("separation produced no stems")
            meta = {"model": model, "backend": backend, "quality": separation.QUALITY[backend]}
            if prof is not None:
                out, tmeta = {}, {}
                for name, path in stems.items():
                    raw = path.with_name(f"{name}_raw.wav")
                    path.rename(raw)
                    tmeta[name] = run_chain(raw, path, prof.for_stem(name), d / "work")
                    out[name], out[f"{name}_raw"] = path, raw
                stems = out
                meta["toneProfile"] = {"mode": prof.mode, "warnings": prof.warnings, "stems": tmeta,
                                       "unmatchedStemOverrides": sorted(set(prof.stems) - set(tmeta))}
        return stems, meta
    bg.add_task(_run, jid, work)
    return {"job_id": jid, "status": "queued", "status_url": f"{str(request.base_url).rstrip('/')}/jobs/{jid}"}


@app.get("/jobs/{job_id}", dependencies=AUTH)
def job_status(job_id: str, request: Request):
    if not SAFE.match(job_id) or not (j := jobs.get(job_id)):
        raise HTTPException(404, "job not found")
    return _job_view(j, request)


@app.get("/stems/{job_id}/{stem}", dependencies=AUTH)
def get_stem(job_id: str, stem: str):
    if not SAFE.match(job_id) or not SAFE.match(stem) or not (j := jobs.get(job_id)):
        raise HTTPException(404, "not found")
    if j["status"] != "done":
        raise HTTPException(409, f"job is {j['status']}")
    p = j["outputs"].get(stem)
    if not p or not Path(p).exists():
        raise HTTPException(404, "stem not found")
    return FileResponse(p, media_type="audio/wav", filename=f"{stem}.wav")


def _sync_job(kind, params, file, request, fn):
    """Short DSP ops run inline (BackgroundTasks would also work) but still produce a job record + URL."""
    jid = jobs.create(kind, params)
    d = jobs.job_dir(jid)
    src = _save_upload(file, d)
    x, sr = load(src)

    def work():
        y, meta = fn(x, sr)
        out = d / "output.wav"
        save(out, y, sr)
        return {"output": out}, meta
    _run(jid, work)
    j = jobs.get(jid)
    if j["status"] == "error":
        raise HTTPException(422, j["error"])
    return _job_view(j, request)


@app.post("/process/pitch-tempo", dependencies=PROC)
def pitch_tempo_ep(request: Request, file: UploadFile = File(...), semitones: float = Form(0.0),
                   tempo: float = Form(1.0), engine: str = Form("auto")):
    if not -24 <= semitones <= 24 or not 0.25 <= tempo <= 4:
        raise HTTPException(422, "semitones in [-24,24], tempo in [0.25,4]")
    if engine in ("auto", "rubberband") and rubberband_cli_available():
        # file-to-file through the CLI: constant memory, best quality
        jid = jobs.create("pitch-tempo", dict(semitones=semitones, tempo=tempo))
        d = jobs.job_dir(jid)
        src = _save_upload(file, d)

        def work():
            wav = d / "input_decoded.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), str(wav)], check=True)
            out = d / "output.wav"
            rubberband_file(str(wav), str(out), semitones, tempo)
            wav.unlink(missing_ok=True)
            return {"output": out}, {"engine": "rubberband-cli"}
        _run(jid, work)
        j = jobs.get(jid)
        if j["status"] == "error":
            raise HTTPException(422, j["error"])
        return _job_view(j, request)
    return _sync_job("pitch-tempo", dict(semitones=semitones, tempo=tempo), file, request,
                     lambda x, sr: (lambda r: (r[0], {"engine": r[1]}))(pitch_tempo(x, sr, semitones, tempo, engine)))


def _parse_profile_field(raw: Optional[str]):
    if raw is None or raw.strip() == "":
        return None
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(422, "toneProfile/profile must be JSON")
    try:
        return parse_profile(body)
    except ProfileError as e:
        raise HTTPException(422, {"errors": e.errors})


@app.get("/tone/schema", dependencies=AUTH)
def tone_schema():
    """Parameter ranges/defaults/units/steps for building sliders dynamically (see schema/FREESTYLE.md)."""
    return schema_doc()


@app.post("/process/tone", dependencies=PROC)
def tone_ep(request: Request, file: UploadFile = File(...), low_gain_db: float = Form(0.0),
            mid_gain_db: float = Form(0.0), high_gain_db: float = Form(0.0), low_freq: float = Form(120.0),
            mid_freq: float = Form(1000.0), high_freq: float = Form(6000.0), mid_q: float = Form(1.0),
            drive: float = Form(0.0), mix_amount: float = Form(0.0, alias="mix"),
            profile: Optional[str] = Form(None), toneProfile: Optional[str] = Form(None),
            stem: Optional[str] = Form(None)):
    """Two bodies: legacy form fields (unchanged), or profile/toneProfile = JSON {mode, params, stems?} which runs
    the full chain denoise -> pitchTempo -> tone. `stem` selects a stems.<name> override."""
    prof = _parse_profile_field(profile if profile is not None else toneProfile)
    if prof is not None:
        if stem is not None and not SAFE.match(stem):
            raise HTTPException(422, "bad stem name")
        jid = jobs.create("tone", {"mode": prof.mode, "stem": stem})
        d = jobs.job_dir(jid)
        src = _save_upload(file, d)

        def work():
            out = d / "output.wav"
            m = run_chain(src, out, prof.for_stem(stem), d / "work")
            return {"output": out}, {"mode": prof.mode, "stem": stem, "stemOverrideApplied": stem in prof.stems,
                                     "warnings": prof.warnings, **m}
        with TONE_SLOTS:
            _run(jid, work)
        j = jobs.get(jid)
        if j["status"] == "error":
            raise HTTPException(422, j["error"])
        return _job_view(j, request)
    for g in (low_gain_db, mid_gain_db, high_gain_db):
        if abs(g) > 24:
            raise HTTPException(422, "gains must be within ±24 dB")
    if not 0 <= mix_amount <= 1 or not 0 <= drive <= 40:
        raise HTTPException(422, "mix in [0,1], drive in [0,40] dB")
    p = dict(low=low_gain_db, mid=mid_gain_db, high=high_gain_db, drive=drive, mix=mix_amount)
    return _sync_job("tone", p, file, request, lambda x, sr: (tone(
        x, sr, low_gain_db, mid_gain_db, high_gain_db, drive, mix_amount,
        low_freq=low_freq, mid_freq=mid_freq, high_freq=high_freq, mid_q=mid_q), {}))


@app.post("/process/denoise", dependencies=PROC)
def denoise_ep(request: Request, file: UploadFile = File(...), preset: str = Form("hiss"),
               strength: Optional[float] = Form(None)):
    if preset not in PRESETS:
        raise HTTPException(422, f"preset must be one of {list(PRESETS)}")
    return _sync_job("denoise", dict(preset=preset), file, request,
                     lambda x, sr: (denoise(x, sr, preset, strength), {"preset": preset}))


@app.post("/mix", dependencies=PROC)
def mix_ep(request: Request, files: List[UploadFile] = File(...), tracks: str = Form("[]"),
           ceiling_db: float = Form(-1.0)):
    """tracks: JSON list aligned with files: [{"gain_db":0,"pan":0,"offset_ms":0}, ...]."""
    try:
        cfg = json.loads(tracks) or [{}] * len(files)
    except json.JSONDecodeError:
        raise HTTPException(422, "tracks must be JSON")
    if len(cfg) != len(files):
        raise HTTPException(422, "tracks length must match files")
    jid = jobs.create("mix", {"tracks": cfg, "ceiling_db": ceiling_db})
    d = jobs.job_dir(jid)
    loaded, sr0 = [], None
    for i, (f, c) in enumerate(zip(files, cfg)):
        x, sr = load(_save_upload(f, d, f"track{i}"))
        if sr0 is None:
            sr0 = sr
        elif sr != sr0:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=sr0)
        loaded.append(Track(x, float(c.get("gain_db", 0)), float(c.get("pan", 0)), float(c.get("offset_ms", 0))))

    def work():
        out = d / "output.wav"
        save(out, mix(loaded, sr0, ceiling_db), sr0)
        return {"output": out}, {"sr": sr0}
    _run(jid, work)
    return _job_view(jobs.get(jid), request)


# ---------------------------------------------------------------- edit model render
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from .edit.engine import BLOCK, EditError, Renderer, RenderStats, decode_to_disk  # noqa: E402
from .edit.model import Project  # noqa: E402
from .edit.processors import ProcessorError  # noqa: E402

EDIT_LIMITS = {"clips": 16, "lanes": 12, "items_per_lane": 256, "processors_per_lane": 8}
EDIT_RENDER_SLOTS = threading.BoundedSemaphore(2)  # each render streams ~3 s blocks; 2 at once stays << 512 MB


def _edit_render_sync(project: Project, uploads: dict, view: str, d: Path, write: bool = True,
                      start: int = 0, count: Optional[int] = None):
    with EDIT_RENDER_SLOTS:
        sources = {}
        for cid, f in uploads.items():
            p = _save_upload(f, d, f"clip_{cid}")
            sources[cid] = decode_to_disk(str(p), project.sampleRate, str(d))
            p.unlink(missing_ok=True)
        try:
            r = Renderer(project, sources)
            if r.length > (MAX_DURATION_S + 60) * r.sr:
                raise HTTPException(413, f"timeline longer than {MAX_DURATION_S / 60:g} min + 60 s")
            st = RenderStats()
            out = d / "output.wav"
            if write:
                import soundfile as sf
                with sf.SoundFile(str(out), "w", samplerate=r.sr, channels=2, subtype="PCM_16") as fh:
                    for blk in r.iter_blocks(view, st):
                        fh.write(blk.T)
            else:  # levels only: meter [start, start+count) block by block, keep nothing
                end = r.length if count is None else min(r.length, start + count)
                for s0 in range(start, end, BLOCK):
                    r.render_block(s0, min(BLOCK, end - s0), view, st)
            return out, st.as_dict(r, view, start)
        finally:
            del sources
            for f32 in d.glob("*.f32"):
                f32.unlink(missing_ok=True)


async def _edit_inputs(request: Request):
    form = await request.form(max_files=EDIT_LIMITS["clips"] + 1)
    raw = form.get("project")
    if not isinstance(raw, str):
        raise HTTPException(422, "form field 'project' (JSON) is required")
    try:
        project = Project.model_validate_json(raw)
    except ValidationError as e:
        raise HTTPException(422, json.loads(e.json(include_url=False, include_input=False)))
    view = form.get("view") or "main_plus_lanes"
    if view not in ("main", "main_plus_lanes"):
        raise HTTPException(422, "view must be main|main_plus_lanes")
    if len(project.clips) > EDIT_LIMITS["clips"] or len(project.lanes) > EDIT_LIMITS["lanes"] or any(
            len(l.items) > EDIT_LIMITS["items_per_lane"] or len(l.processorChain) > EDIT_LIMITS["processors_per_lane"]
            for l in project.lanes):
        raise HTTPException(422, f"project exceeds limits {EDIT_LIMITS}")
    uploads = {k[5:]: v for k, v in form.multi_items() if k.startswith("clip:") and hasattr(v, "filename")}
    wanted = {c.id for c in project.clips}
    if set(uploads) != wanted:
        raise HTTPException(422, f"upload exactly one 'clip:<id>' part per clip; missing "
                                 f"{sorted(wanted - set(uploads))}, unexpected {sorted(set(uploads) - wanted)}")
    return form, project, view, uploads


@app.post("/edit/levels", dependencies=PROC)
async def edit_levels(request: Request):
    """Same multipart inputs as /edit/render (plus optional startSample / numSamples). Returns only the meters:
    per-lane, main and master peak dBFS, RMS dBFS and headroom. Nothing is stored."""
    form, project, view, uploads = await _edit_inputs(request)
    try:
        start = int(form.get("startSample") or 0)
        count = int(form["numSamples"]) if form.get("numSamples") else None
    except ValueError:
        raise HTTPException(422, "startSample/numSamples must be integers")
    if start < 0 or (count is not None and count <= 0):
        raise HTTPException(422, "startSample must be >= 0 and numSamples > 0")
    import tempfile
    jobs.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(jobs.STORAGE_DIR)) as td:
        try:
            _, meta = await run_in_threadpool(_edit_render_sync, project, uploads, view, Path(td), False, start, count)
        except (EditError, ProcessorError) as e:
            raise HTTPException(422, str(e))
    return meta


@app.post("/edit/render", dependencies=PROC)
async def edit_render(request: Request):
    """multipart/form-data:
      project  = Project JSON (schema/edit_model.json)
      view     = main | main_plus_lanes (default)
      response = wav (default: audio/wav body + X-Edit-* headers) | json (job record + url + metadata)
      clip:<clipId> = one audio file per Project.clips entry
    """
    form, project, view, uploads = await _edit_inputs(request)
    mode = form.get("response") or "wav"
    if mode not in ("wav", "json"):
        raise HTTPException(422, "response must be wav|json")
    jid = jobs.create("edit_render", {"projectId": project.id, "view": view})
    d = jobs.job_dir(jid)
    try:
        out, meta = await run_in_threadpool(_edit_render_sync, project, uploads, view, d)
    except (EditError, ProcessorError) as e:
        jobs.update(jid, status="error", error=str(e))
        raise HTTPException(422, str(e))
    jobs.update(jid, status="done", outputs={"output": str(out)}, meta=meta)
    if mode == "json":
        return _job_view(jobs.get(jid), request)
    m = meta["master"]
    headers = {
        "X-Edit-Job-Id": jid,
        "X-Edit-Length-Samples": str(meta["lengthSamples"]),
        "X-Edit-Sample-Rate": str(meta["sampleRate"]),
        "X-Edit-Master-Peak-Dbfs": str(m["peakDbfs"]),
        "X-Edit-Master-Headroom-Db": str(m["headroomDb"]),
        "X-Edit-Master-Rms-Dbfs": str(m["rmsDbfs"]),
        "X-Edit-Meta": json.dumps(meta, separators=(",", ":")),
    }
    return FileResponse(out, media_type="audio/wav", filename=f"{project.id}-{view}.wav", headers=headers)
