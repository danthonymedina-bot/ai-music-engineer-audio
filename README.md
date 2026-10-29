# AI Music Engineer — Audio Engine (server)

FastAPI backend for the Android "AI Music Engineer" app: stem separation, pitch/tempo, tone (EQ + saturation), denoise and multi-track mixing.
Android integration guide: **[INTEGRATION.md](INTEGRATION.md)**. On-device fallback: **[ondevice/](ondevice/README.md)**.

## Layout
```
app/main.py            FastAPI routes
app/jobs.py            in-process job store (JSON per job under $AUDIO_ENGINE_STORAGE/jobs/<id>)
app/audio_io.py        load (soundfile, ffmpeg fallback for mp3/m4a/aac/opus) / save (16-bit WAV)
app/dsp/tone.py        RBJ biquad low-shelf / peaking / high-shelf + tanh soft-clip saturation
app/dsp/pitch_tempo.py Rubber Band (pyrubberband + rubberband CLI) → librosa fallback
app/dsp/denoise.py     noisereduce spectral gating, presets wind / air / hiss
app/dsp/mix.py         gain dB, constant-power pan, offset ms, peak limiter
app/dsp/separation.py  backend selection: SEPARATION_BACKEND=auto|onnx|dsp|demucs
app/dsp/onnx_separation.py  Open-Unmix umxhq int8 ONNX, chunked (default on free tier)
app/dsp/dsp_separation.py   classical HPSS + centre/side separation (LOWER QUALITY fallback, scipy only)
models/umxhq/          4 × ~9 MB int8 ONNX models (see its README for provenance/license)
app/security.py        X-API-Key auth + in-memory per-key/IP rate limiter
app/tone/              freestyle/preset tone profiles (clamp+warn), stateful DSP stages, per-stem chain
app/edit/              edit model (Pydantic) + sample-accurate block render engine + processor registry + undo history
schema/                JSON Schema, EditModel.kt, kotlin-dsp reference + test vectors, README/MIGRATION for Android
tests/                 pytest, synthetic signals; separation is mocked
```
All DSP functions are pure: `(np.ndarray[(C,)N], sr, params) -> np.ndarray`.

## Run locally
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt          # light deps
pip install -r requirements-ml.txt           # optional: demucs + torch (large)
sudo apt install ffmpeg rubberband-cli       # optional but recommended
uvicorn app.main:app --reload
pytest -q
```
Env: `AUDIO_ENGINE_API_KEY` (**required**), `AUDIO_ENGINE_STORAGE` (default `/tmp/audio-engine`), `SEPARATION_BACKEND` (auto),
`AUDIO_ENGINE_MAX_UPLOAD_MB` (40), `AUDIO_ENGINE_MAX_DURATION_S` (300), `RATE_SEPARATE_PER_HOUR` (10), `RATE_PROCESS_PER_HOUR` (60),
`ONNX_CHUNK_S` (6), `ONNX_THREADS` (1), `ONNX_MODEL_DIR` (models/umxhq).
Local: `export AUDIO_ENGINE_API_KEY=$(cat .secrets/api_key.txt)` (file is gitignored, chmod 600).

## Auth & limits
- Every endpoint except `/health` (and `/docs`) requires header **`X-API-Key: <key>`** → otherwise **401**. If the server has no key configured → **503**.
- The beta key lives in `.secrets/api_key.txt` (never committed). On Render it is the env var `AUDIO_ENGINE_API_KEY` (`sync: false` in `render.yaml`, so you paste it in the dashboard).
- The Android app ships the key via `local.properties` → `BuildConfig.AUDIO_ENGINE_API_KEY` (see INTEGRATION.md). **It is extractable from the APK** — acceptable for a closed beta; before public launch move to per-user tokens (e.g. Firebase ID token verified server-side) and rotate this key.
- Rate limits (in-memory, per key+client IP, sliding 1 h): **10 `/separate`**, **60 process calls** (`/process/*` + `/mix`) → **429** with `Retry-After`. Uploads max **40 MB** and **5 min** audio → **413**. Undecodable → **415**.
- Limits reset on restart and aren't shared across instances — fine for one beta instance.

## API examples
```bash
B=http://localhost:8000
curl $B/health                        # no key needed
H="X-API-Key: $AUDIO_ENGINE_API_KEY"   # add -H "$H" to every call below
# separation (async → 202 + job_id)
curl -H "$H" -F file=@song.mp3 -F model=htdemucs $B/separate
curl -H "$H" $B/jobs/<job_id>                 # status: queued|running|done|error, urls.{vocals,drums,bass,other}
curl -H "$H" -o vocals.wav $B/stems/<job_id>/vocals
# pitch / tempo (sync; returns job with urls.output)
curl -H "$H" -F file=@vocals.wav -F semitones=-2 -F tempo=1.1 $B/process/pitch-tempo
# tone
curl -H "$H" -F file=@bass.wav -F low_gain_db=4 -F mid_gain_db=-2 -F mid_freq=400 -F high_gain_db=1 -F drive=12 -F mix=0.3 $B/process/tone
# denoise (preset wind|air|hiss, optional strength 0..1)
curl -H "$H" -F file=@take.m4a -F preset=wind $B/process/denoise
# mix
curl -H "$H" -F files=@vox.wav -F files=@gtr.wav \
  -F 'tracks=[{"gain_db":0,"pan":0},{"gain_db":-4,"pan":-0.6,"offset_ms":120}]' -F ceiling_db=-1 $B/mix
# freestyle tone (clamped, never rejected; see schema/FREESTYLE.md) + slider ranges
curl -H "$H" -F file=@vox.wav -F 'profile={"mode":"freestyle","params":{"width":{"enabled":true,"widthPct":160},"saturation":{"enabled":true,"type":"tape","driveDb":12}}}' $B/process/tone
curl -H "$H" $B/tone/schema
curl -H "$H" -F file=@song.mp3 -F "toneProfile=<my_preset.json" $B/separate   # tone applied per stem
# edit model render (WAV + X-Edit-* headers) / levels only (JSON meters); see schema/README.md
curl -H "$H" -F "project=<schema/example_project.json" -F view=main_plus_lanes \
  -F clip:main=@song.flac -F clip:vox=@vox.flac -o edit.wav -D - $B/edit/render
curl -H "$H" -F "project=<schema/example_project.json" -F clip:main=@song.flac -F clip:vox=@vox.flac $B/edit/levels
```
Model values: `auto` (default → backend from `SEPARATION_BACKEND`), `umxhq` (ONNX), `dsp` (classical), and — only if Demucs is installed (paid plans) — `htdemucs`, `htdemucs_ft`, `htdemucs_6s`, `mdx_extra`.
Job `meta` / `params` report `backend` and `quality` (`medium` = Open-Unmix, `low (classical DSP, not ML)`, `high` = Demucs).

## Separation backends
| backend | quality | RAM peak (whole server, 3-min song) | time (3-min song, 1 CPU thread, this box) | where |
|---|---|---|---|---|
| `onnx` Open-Unmix umxhq int8 | medium (published SDR vocals ≈ 6.3 dB on MUSDB18) | **336 MB** | ~14 s here; expect 1–3 min on Render free's 0.1 CPU | default on free tier |
| `dsp` HPSS + centre/side | **low** — "karaoke-style", vocals = centre-panned harmonic, drums = percussive, bass = <160 Hz | ~260 MB | ~16 s | automatic fallback if models missing |
| `demucs` htdemucs | high | ~1.5–2 GB | 0.5–1.5× song length on CPU | paid plan only (`--build-arg WITH_DEMUCS=1`) |
`auto` picks onnx → demucs → dsp. Only one separation runs at a time (lock) to stay within 512 MB.

Measured peak RSS of the running server (VmHWM) on a 3-min stereo 44.1 kHz file, one request at a time: idle 123 MB; separate (onnx) 336 MB;
pitch-tempo (rubberband CLI) 126 MB; tone 366 MB; denoise wind 372 MB. Without the rubberband CLI, the librosa fallback hit ~1.2 GB — the Docker image includes `rubberband-cli`, so don't remove it.

## Deploy (Render FREE tier beta)
1. Repo is public on GitHub; Render → **New → Blueprint** (reads `render.yaml`: Docker, `plan: free`, no disk) — or create via API.
2. Set **`AUDIO_ENGINE_API_KEY`** to the value in `.secrets/api_key.txt` (`sync: false`).
3. Build ~3–5 min (no torch). Check `https://<service>.onrender.com/health` → `"separation_backend":"onnx"`.

**Free-tier caveats:** 512 MB RAM, shared ~0.1 CPU (slow), **spins down after ~15 min idle → first request takes ~30–60 s** (app should show "waking up server…"), **no persistent disk**: results in `/tmp` vanish on restart/spin-down/redeploy, so the app must download stems promptly. Single worker; rate limits are in-memory and reset on restart.
Upgrade path: Render **standard** (2 GB, ~$25/mo) + `WITH_DEMUCS=1` + disk → high-quality Demucs (separation ~0.5–1.5× song length on CPU).

Docker locally: `docker build -t audio-engine . && docker run -p 8000:8000 -e AUDIO_ENGINE_API_KEY=$(cat .secrets/api_key.txt) audio-engine`.

## Licensing notes — time-stretch / pitch engines
| Engine | License | Notes |
|---|---|---|
| **Rubber Band** (used here, via CLI) | **GPL v2+** or **commercial** (Breakfast Quay / Particular Programs) | Server-side CLI use via subprocess is generally not "distribution" of the app, so the GPL doesn't reach the Android APK. If Rubber Band is ever **linked into the Android app** (NDK), you need the commercial license or must GPL the app. |
| **SoundTouch** — *on-device default* | **LGPL v2.1** | Default on-phone time-stretch/pitch fallback. Ship as a **separate shared library (`libsoundtouch.so`) loaded dynamically** via JNI so the app stays non-LGPL; include the license + offer source/relinking (keep the .so replaceable, don't statically link). Quality below Rubber Band but fine for practice/preview. |
| **Superpowered** — *paid upgrade* | Proprietary SDK, commercial license (confirm current pricing/terms) | Upgrade path for higher-quality, low-latency on-device pitch/tempo + effects via NDK. |
| librosa (fallback) | ISC | Phase-vocoder quality; fine for preview, audibly worse than Rubber Band on vocals. |
| Demucs | MIT (code + weights) | Paid-plan backend. noisereduce: MIT. |
| Open-Unmix umxhq | code MIT; weights: verify Zenodo record | Free-tier backend. `umxl` weights are non-commercial — not used. |
**Decided:** Rubber Band (GPL) server-only via CLI; SoundTouch (LGPL, dynamic .so) on-device by default; Superpowered as the paid upgrade.

## Denoise upgrade path (ML)
Current: `noisereduce` spectral gating (fast, CPU, no weights, can produce "musical noise" artifacts on heavy noise).
Planned upgrades behind the same `/process/denoise` endpoint (e.g. `preset=ml` / `engine=` param):
- **RNNoise** (BSD, Xiph) — tiny GRU, 48 kHz, speech-focused, real-time on CPU and on-device. Good for voice memos/vocals.
- **DeepFilterNet 2/3** (MIT/Apache) — full-band 48 kHz, much better quality on vocals and wind, CPU real-time; `pip install deepfilternet`, ONNX export available for Android.
Music-bed denoise (not just voice) remains hard; keep spectral gating for instruments.

## Verified vs not
- ✅ Unit tests for EQ/saturation/mix/limiter/pitch-tempo (librosa)/denoise + API tests with mocked separation.
- ✅ Auth (401/503), rate limits (429 + Retry-After), size/duration caps (413).
- ✅ ONNX (umxhq int8) and DSP separation on a synthetic 3-min song incl. memory measurements; rubberband CLI path.
- ❌ Real Demucs inference and separation quality on real music (only synthetic signals here).
