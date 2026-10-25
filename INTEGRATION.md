# Android ↔ Audio Engine integration

For the Android (Jetpack Compose) team and the main bot. All endpoints are plain HTTP + multipart.

> **BASE_URL = `https://TODO-ai-music-engineer-audio.onrender.com/`** — **TODO:** replace with the real Render URL after deploy.

## Auth
Every request except `GET /health` must send **`X-API-Key: <key>`** (401 otherwise). The key is set server-side as the Render env var `AUDIO_ENGINE_API_KEY` (`sync: false`, entered in the dashboard) and is shared out-of-band — never commit it.
Bake it into the APK from the untracked `local.properties`:
```properties
# local.properties (gitignored)
audioEngine.url=https://TODO-ai-music-engineer-audio.onrender.com/
audioEngine.apiKey=<paste key>
```
```kotlin
// app/build.gradle.kts
val props = java.util.Properties().apply { rootProject.file("local.properties").takeIf { it.exists() }?.inputStream()?.use(::load) }
android { buildFeatures { buildConfig = true }
  defaultConfig {
    buildConfigField("String", "AUDIO_ENGINE_API_KEY", "\"${props.getProperty("audioEngine.apiKey", "")}\"")
  }
}
```
⚠️ A key in an APK **can be extracted** (apktool/strings). Acceptable for the closed beta; for public launch switch to per-user auth (Firebase ID token) and rotate the key. Enable R8 to at least avoid trivially readable code.

## Limits
- 10 `/separate` per hour and 60 processing calls (`/process/*` + `/mix`) per hour, per key + client IP → **429** with `Retry-After` (seconds). Show "try again in N min".
- Upload ≤ **50 MB** and ≤ **10 min** of audio → **413**.
- CPU beta: separation takes ~0.5–1.5× song length.

## Base URL
Configure per build type, never hard-code:
```kotlin
// app/build.gradle.kts
buildTypes {
  debug   { buildConfigField("String", "AUDIO_ENGINE_URL", "\"http://10.0.2.2:8000/\"") } // emulator → host
  release { buildConfigField("String", "AUDIO_ENGINE_URL", "\"${props.getProperty("audioEngine.url", "https://TODO-ai-music-engineer-audio.onrender.com/")}\"") } // TODO real URL
}
```
Debug over plain HTTP needs `android:usesCleartextTraffic` / a network-security-config for 10.0.2.2 only.

## Endpoints
| Method | Path | Body (multipart/form-data) | Response |
|---|---|---|---|
| GET | `/health` | – | `{ok, demucs, rubberband}` capabilities (no key needed) |
| POST | `/separate` | `file`, `model?` htdemucs (default) / htdemucs_ft / htdemucs_6s / mdx_extra | **202** `{job_id, status, status_url}` |
| GET | `/jobs/{id}` | – | `Job` (below) |
| GET | `/stems/{id}/{stem}` | – | `audio/wav` |
| POST | `/process/pitch-tempo` | `file`, `semitones` [-24,24], `tempo` [0.25,4], `engine?` auto/rubberband/librosa | **200** `Job` (done) |
| POST | `/process/tone` | `file`, `low_gain_db`,`mid_gain_db`,`high_gain_db` (±24), `low_freq`,`mid_freq`,`high_freq`,`mid_q`, `drive` (0–40 dB), `mix` (0–1) | **200** `Job` |
| POST | `/process/denoise` | `file`, `preset` wind/air/hiss, `strength?` 0–1 | **200** `Job` |
| POST | `/mix` | `files` (repeat), `tracks` JSON `[{gain_db,pan(-1..1),offset_ms}]` same order, `ceiling_db?` | **200** `Job` |

```json
// Job
{"id":"9f…","kind":"separate","status":"done","params":{"model":"htdemucs"},
 "error":null,"meta":{"model":"htdemucs"},
 "created_at":1761400000.1,"updated_at":1761400042.7,
 "urls":{"vocals":"https://…/stems/9f…/vocals","drums":"…","bass":"…","other":"…"}}
```
Processing endpoints return `urls.output`. Download any URL with GET → 16-bit WAV at the input sample rate.
Accepted uploads: wav, mp3, flac, m4a, aac, ogg, opus (≤ 50 MB, ≤ 10 min).

## Flow
1. Upload with `/separate` → get `job_id`.
2. Poll `/jobs/{id}` every 2 s (back off to 5 s after 30 s). Separation on CPU ≈ 0.5–1.5× track length; GPU ≈ 5–10 s per song.
3. On `done`, download `urls[stem]` to app cache; on `error`, show `error`.
4. Per-stem edits: upload the stem WAV to `/process/*` (synchronous; keep the call off the main thread, timeout ≥ 120 s).
5. Final bounce: `/mix` with all stems/takes.

## Error codes
| Code | Meaning | Client action |
|---|---|---|
| 202 | job accepted | start polling |
| 401 | missing/invalid `X-API-Key` | check BuildConfig key |
| 429 | rate limit hit | wait `Retry-After` seconds |
| 503 | server has no API key configured | ops issue |
| 404 | unknown job / stem | drop cached id |
| 409 | stem requested before job `done` | keep polling |
| 413 | upload > 50 MB or audio > 10 min | compress / trim |
| 415 | unsupported extension or undecodable audio | convert to wav/m4a |
| 422 | bad params, or DSP failed (`detail` has message) | show message |
| 5xx | server error / restart (jobs in progress are lost on restart) | retry with backoff, re-submit |
Job-level failure is `status:"error"` with HTTP 200 on `/jobs/{id}`.

## Retrofit / OkHttp
```kotlin
interface AudioEngineApi {
  @Multipart @POST("separate")
  suspend fun separate(@Part file: MultipartBody.Part, @Part("model") model: RequestBody): SubmitResponse

  @GET("jobs/{id}") suspend fun job(@Path("id") id: String): Job

  @Multipart @POST("process/pitch-tempo")
  suspend fun pitchTempo(@Part file: MultipartBody.Part,
                         @Part("semitones") semitones: RequestBody,
                         @Part("tempo") tempo: RequestBody): Job

  @Multipart @POST("process/tone")
  suspend fun tone(@Part file: MultipartBody.Part, @PartMap params: Map<String, @JvmSuppressWildcards RequestBody>): Job

  @Multipart @POST("process/denoise")
  suspend fun denoise(@Part file: MultipartBody.Part, @Part("preset") preset: RequestBody): Job

  @Multipart @POST("mix")
  suspend fun mix(@Part files: List<MultipartBody.Part>, @Part("tracks") tracksJson: RequestBody): Job

  @Streaming @GET suspend fun download(@Url url: String): ResponseBody
}

@Serializable data class SubmitResponse(@SerialName("job_id") val jobId: String, val status: String)
@Serializable data class Job(val id: String, val status: String, val error: String? = null,
                             val urls: Map<String, String> = emptyMap())

val client = OkHttpClient.Builder()
  .addInterceptor { chain ->
    chain.proceed(chain.request().newBuilder().header("X-API-Key", BuildConfig.AUDIO_ENGINE_API_KEY).build())
  }
  .connectTimeout(15, TimeUnit.SECONDS).writeTimeout(120, TimeUnit.SECONDS).readTimeout(180, TimeUnit.SECONDS)
  .build()
val api = Retrofit.Builder().baseUrl(BuildConfig.AUDIO_ENGINE_URL).client(client)
  .addConverterFactory(Json { ignoreUnknownKeys = true }.asConverterFactory("application/json".toMediaType()))
  .build().create(AudioEngineApi::class.java)

fun File.asPart(name: String = "file") =
  MultipartBody.Part.createFormData(name, this.name, asRequestBody("audio/*".toMediaType()))
fun String.text() = toRequestBody("text/plain".toMediaType())

suspend fun separateAndWait(f: File): Job {
  val id = api.separate(f.asPart(), "htdemucs".text()).jobId
  var delayMs = 2000L
  while (true) {
    val j = api.job(id)
    when (j.status) { "done", "error" -> return j }
    delay(delayMs); delayMs = (delayMs * 1.2).toLong().coerceAtMost(5000)
  }
}

suspend fun download(url: String, dest: File) = withContext(Dispatchers.IO) {
  api.download(url).byteStream().use { input -> dest.outputStream().use { input.copyTo(it) } }
}
```
Use a `WorkManager` `CoroutineWorker` for separation so it survives the app going to background. Content-URIs from the picker must be copied to a cache `File` (or wrapped in a custom `RequestBody`) before upload.

## Notes for the main bot
- Server is single-process; jobs are lost on restart; stems are kept on disk until cleaned (no TTL yet).
- `/health` tells the app whether separation is available; if `demucs:false`, fall back to the on-device path (`ondevice/`).
