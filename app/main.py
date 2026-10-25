"""AI Music Engineer - audio processing backend."""
import json
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
from .dsp.pitch_tempo import pitch_tempo, rubberband_available
from .dsp.tone import tone

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
    return {"ok": True, "demucs": separation.demucs_available(), "rubberband": rubberband_available()}


@app.post("/separate", status_code=202, dependencies=SEP)
def separate_ep(bg: BackgroundTasks, request: Request, file: UploadFile = File(...), model: str = Form(DEFAULT_MODEL)):
    if model not in separation.DEMUCS_MODELS:
        raise HTTPException(422, f"unknown model {model}")
    jid = jobs.create("separate", {"model": model})
    d = jobs.job_dir(jid)
    src = _save_upload(file, d)

    def work():
        stems = separation.separate(src, d / "stems", model)
        if not stems:
            raise RuntimeError("separation produced no stems")
        return stems, {"model": model}
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
    return _sync_job("pitch-tempo", dict(semitones=semitones, tempo=tempo), file, request,
                     lambda x, sr: (lambda r: (r[0], {"engine": r[1]}))(pitch_tempo(x, sr, semitones, tempo, engine)))


@app.post("/process/tone", dependencies=PROC)
def tone_ep(request: Request, file: UploadFile = File(...), low_gain_db: float = Form(0.0),
            mid_gain_db: float = Form(0.0), high_gain_db: float = Form(0.0), low_freq: float = Form(120.0),
            mid_freq: float = Form(1000.0), high_freq: float = Form(6000.0), mid_q: float = Form(1.0),
            drive: float = Form(0.0), mix_amount: float = Form(0.0, alias="mix")):
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
