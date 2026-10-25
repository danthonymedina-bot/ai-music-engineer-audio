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
app/dsp/separation.py  demucs (htdemucs default; only cloud backend), lazy subprocess
app/security.py        X-API-Key auth + in-memory per-key/IP rate limiter
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
Env: `AUDIO_ENGINE_API_KEY` (**required**), `AUDIO_ENGINE_STORAGE` (default `./storage`), `AUDIO_ENGINE_MAX_UPLOAD_MB` (50),
`AUDIO_ENGINE_MAX_DURATION_S` (600), `RATE_SEPARATE_PER_HOUR` (10), `RATE_PROCESS_PER_HOUR` (60), `AUDIO_ENGINE_DEFAULT_MODEL` (htdemucs).
Local: `export AUDIO_ENGINE_API_KEY=$(cat .secrets/api_key.txt)` (file is gitignored, chmod 600).

## Auth & limits
- Every endpoint except `/health` (and `/docs`) requires header **`X-API-Key: <key>`** → otherwise **401**. If the server has no key configured → **503**.
- The beta key lives in `.secrets/api_key.txt` (never committed). On Render it is the env var `AUDIO_ENGINE_API_KEY` (`sync: false` in `render.yaml`, so you paste it in the dashboard).
- The Android app ships the key via `local.properties` → `BuildConfig.AUDIO_ENGINE_API_KEY` (see INTEGRATION.md). **It is extractable from the APK** — acceptable for a closed beta; before public launch move to per-user tokens (e.g. Firebase ID token verified server-side) and rotate this key.
- Rate limits (in-memory, per key+client IP, sliding 1 h): **10 `/separate`**, **60 process calls** (`/process/*` + `/mix`) → **429** with `Retry-After`. Uploads max **50 MB** and **10 min** audio → **413**. Undecodable → **415**.
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
```
Model values (Demucs only): `htdemucs` (default), `htdemucs_ft`, `htdemucs_6s`, `mdx_extra`.

## Deploy (Render, CPU-only beta)
1. Push this repo to GitHub/GitLab.
2. Render dashboard → **New → Blueprint** → select repo; it reads `render.yaml` (Docker, `standard` plan, 10 GB disk at `/data`).
3. When prompted, set **`AUDIO_ENGINE_API_KEY`** to the value in `.secrets/api_key.txt`.
4. Wait for build (~10 min first time: torch CPU + Demucs weights baked in), then `curl https://<service>.onrender.com/health`.
5. Put the URL into INTEGRATION.md (`BASE_URL` TODO) and the APK's `local.properties`.

**Sizing:** CPU-only. Separation takes roughly **0.5–1.5× the song length** (a 4-min song ≈ 2–6 min); the app must poll. Render **free tier (512 MB RAM) is too small** for Demucs (OOM) and has no persistent disk — use **standard (2 GB RAM)**. A GPU host would cut separation to seconds if the beta grows.
Run **one instance/worker**: jobs, rate limits and storage are in-process/local. No file TTL yet — the 10 GB disk needs periodic cleanup.

Docker locally: `docker build -t audio-engine . && docker run -p 8000:8000 -e AUDIO_ENGINE_API_KEY=$(cat .secrets/api_key.txt) -v $PWD/data:/data audio-engine` (`--build-arg WITH_DEMUCS=0` for a slim image). `railway.json` is kept only as an unused alternative.

## Licensing notes — time-stretch / pitch engines
| Engine | License | Notes |
|---|---|---|
| **Rubber Band** (used here, via CLI) | **GPL v2+** or **commercial** (Breakfast Quay / Particular Programs) | Server-side CLI use via subprocess is generally not "distribution" of the app, so the GPL doesn't reach the Android APK. If Rubber Band is ever **linked into the Android app** (NDK), you need the commercial license or must GPL the app. |
| **SoundTouch** — *on-device default* | **LGPL v2.1** | Default on-phone time-stretch/pitch fallback. Ship as a **separate shared library (`libsoundtouch.so`) loaded dynamically** via JNI so the app stays non-LGPL; include the license + offer source/relinking (keep the .so replaceable, don't statically link). Quality below Rubber Band but fine for practice/preview. |
| **Superpowered** — *paid upgrade* | Proprietary SDK, commercial license (confirm current pricing/terms) | Upgrade path for higher-quality, low-latency on-device pitch/tempo + effects via NDK. |
| librosa (fallback) | ISC | Phase-vocoder quality; fine for preview, audibly worse than Rubber Band on vocals. |
| Demucs | MIT (code + weights) | Only cloud separation backend. noisereduce: MIT. |
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
- ❌ Real Demucs inference, rubberband path, Docker build and Render deploy not exercised here.
