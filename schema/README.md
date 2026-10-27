# Edit model: source of truth (for the Android UI Bot)

The edit model describes a non-destructive arrangement: a **main** clip plus up to 12 **lanes** of items cut from
source clips. Each lane has its own processor chain and a gain/pan bus. The backend renders it with
sample-accurate math, and the same math is ported to Kotlin in `kotlin-dsp/` for on-device playback.

| File | What it is |
|---|---|
| `edit_model.json` | JSON Schema (draft 2020-12) generated from `app/edit/model.py` (`scripts/gen_schema.py`; CI checks it with `--check`). `x-schemaVersion` matches `Project.version` (currently `1`). |
| `EditModel.kt` | kotlinx.serialization mirror of the schema (package `com.david.aimusicengineer.edit`), plus `EditHistory` (undo/redo). Verified: it decodes and re-encodes `example_project.json` to a Project equal to the Python one, and the output validates against the schema. |
| `example_project.json` | A complete project: two clips, an `add` lane with a delay and EQ, and a `replace` lane. |
| `kotlin-dsp/EditDsp.kt` | Reference DSP: crossfade curves, item geometry, lane composer, delay with its wet envelope, lane bus with solo/mute, soft clip, and tempo sync. |
| `kotlin-dsp/EditDspTest.kt` + `test_vectors.json` | JUnit4 tests against vectors produced by the Python engine (`scripts/gen_kotlin_vectors.py`). Tolerance is 1e-6. All 6 tests pass on JDK 21 / Kotlin 2.0.21. |
| `MIGRATION.md` | How to swap the UI's stand-in model for `EditModel.kt`, and how each UI control maps to a field. |

## Model at a glance

```
Project { version=1, id, sampleRate, snap{type seconds|beats, value=0.1, bpm}, clips[SourceClip],
          mainClipId, mainGainDb, mainMute, lanes[Lane], selection?, masterLimiter soft_clip|gain_compensation,
          softClipThresholdDb=-1 }
Lane    { id, name, order, colorArgb (0xAARRGGBB as Long), gainDb (BUS), pan -1..1, mute, solo,
          items[LaneItem], processorChain[Processor] }
LaneItem{ id, sourceClipId, range{startSample,endSample}, laneStartSample, mode add|replace, gainDb,
          fadeInMs, fadeOutMs, crossfade{lengthMs 50..100 =75, curve equal_power|linear} }
Processor{ id, type ("delay" | "eq" | ...), enabled, params{}, fadeInMs, fadeOutMs }
EditHistory{ snapshots[Project] (= ops+1), ops[EditOp{opId,type,targetIds,before,after}], cursor }
```

- All positions are **integer samples at `Project.sampleRate`**. Ranges are half-open: `[startSample, endSample)`.
- Ids are unique across lanes, items and processors (the validator rejects duplicates).
- `selection` is UI-only state; the renderer ignores it.
- Unknown fields are rejected (`extra="forbid"` / `ignoreUnknownKeys=false`). Bump `version` for breaking changes.

## Render semantics (identical in Python and Kotlin)

**Lane composite.** Items are processed in list order. Let `L = range length`, `P = laneStartSample`,
`X = min(round(lengthMs*sr/1000), L)` (round-half-even, which is `Math.rint` in Kotlin) and `h = X/2` (integer division).
- Start window `[P-h, P-h+X)`, body, then end window `[P+L-h, P+L-h+X)`. The crossfade is *centred* on each
  boundary. In a window, `w = (i+0.5)/X`.
- Curves: `equal_power` gives `out = cos(w·π/2)`, `in = sin(w·π/2)`; `linear` gives `out = 1-w`, `in = w`.
  Use equal_power for different material and linear for identical or correlated material.
- `replace`: `buf = buf·g_out + new·env·g_in`, so the old content is muted under the body. `add`:
  `buf += new·env·g_in`.
- Audio inside the crossfade windows comes from the source just outside the range ("handles"), or zero when the
  source has nothing there.
- Item `fadeInMs`/`fadeOutMs` form a linear envelope on `[P, P+L)` and are multiplied by `dbToGain(item.gainDb)`.
- **Seam rule:** when an item continues the same source exactly where the previous one ended (same mode, gain and
  crossfade), that boundary is a hard edge with no crossfade. This keeps Split bit-lossless.

**Lane chain.** composite → enabled processors in order → **bus**: `dbToGain(lane.gainDb)` is applied once to the
lane sum, then constant-power pan with `L = cos((pan+1)π/4)·√2` and `R = sin((pan+1)π/4)·√2` (unity at centre)
→ audibility. Mute always wins. If any lane is soloed, only soloed, unmuted lanes play. Solo does not affect
the main clip.

**Processors.** Each processor's envelope ramps in from the start of the lane's content and ramps out, finishing at
content end + the processor tail.
- `delay` params: `timeMs` 1..4000 (default 250), `feedback` 0..0.95 (default 0), `mix` 0..1 (default 0.5),
  `syncToSnap`, `beats` (snap units).
  - Output: `y = (1-mix)·x + env·mix·d`, so the fade envelope shapes **only the echoes** and the dry signal is never
    faded.
  - Echo signal: `d[n] = Σ_k fb^(k-1)·x[n-kD]`, truncated once fb^k < 1e-4 or after 30 s.
  - Sync: `D = rint(beats · unit)`, where unit is `60/bpm·sr·value` for a beats snap and `value·sr` for a
    seconds snap.
- `eq` params: `lowGainDb`/`midGainDb`/`highGainDb` (±24), `lowFreq` (120), `midFreq` (1000), `highFreq` (6000),
  `midQ` (1), `drive` (0..24 dB) and `driveMix`. It is the same EQ as `/tone`.
  - Generic envelope: `y = x + env·(wet - x)`.
  - This one is server-side only for now; there is no Kotlin port.
- Unknown params or types give a 422. New processors are added server-side with `@register("name")`.

**Master.** The main clip plays from sample 0, multiplied by `mainGainDb` (or silenced by `mainMute`), plus the
lanes. Then:
- `soft_clip` (default) is transparent below the threshold (−1 dBFS). Above it,
  `T + (C−T)·tanh((|x|−T)/(C−T))` with ceiling `C = −0.1 dBFS`.
- `gain_compensation` pre-scans both views and scales everything down by one fixed gain when needed.

**Gapless.** Everything is a pure function of the timeline sample. `render_block(start, n)` concatenates to exactly
the full render (verified bit-exact with delay + EQ). The `main` and `main_plus_lanes` views have the same length
(`max(main end, lane content end + chain tail)`) and the same alignment, so the UI can A/B them at the same playhead.

## HTTP API

Both endpoints need the `X-API-Key` header and use the `process` rate limit. Requests are `multipart/form-data`:

| Field | Value |
|---|---|
| `project` | Project JSON |
| `view` | `main` \| `main_plus_lanes` (default) |
| `clip:<clipId>` | One audio file per `Project.clips` entry; exactly one each, or 422 |
| `response` (`/edit/render` only) | `wav` (default) \| `json` |
| `startSample`, `numSamples` (`/edit/levels` only) | Optional metering window |

Limits: 16 clips, 12 lanes, 256 items per lane, 8 processors per lane, and timeline ≤ the server's max duration
+ 60 s. Each upload is capped at 40 MB, so send FLAC/MP3 for long sources.

- `POST /edit/render` returns a 16-bit stereo WAV at `sampleRate` with these headers: `X-Edit-Length-Samples`,
  `X-Edit-Sample-Rate`, `X-Edit-Master-Peak-Dbfs`, `X-Edit-Master-Rms-Dbfs`, `X-Edit-Master-Headroom-Db`,
  `X-Edit-Job-Id` and `X-Edit-Meta` (compact JSON of the metadata below). With `response=json` it returns the job
  record and `urls.output`.
- `POST /edit/levels` renders and meters without storing anything, and returns only the metadata:

```json
{"view":"main_plus_lanes","sampleRate":44100,"lengthSamples":7938000,"measuredStartSample":0,"measuredSamples":7938000,
 "lanes":[{"laneId":"lane-1","audible":true,"peakDbfs":-9.1,"rmsDbfs":-24.3,"headroomDb":9.1}],
 "main":{"peakDbfs":-1.6,"rmsDbfs":-14.2,"headroomDb":1.6},
 "master":{"peakDbfs":-1.0,"rmsDbfs":-13.8,"headroomDb":1.0,"prePeakDbfs":0.4,"limiter":"soft_clip",
           "gainCompensationDb":0.0,"softClipThresholdDb":-1.0,"ceilingDbfs":-0.1,"softClippedSamples":312}}
```

Lane meters are post-bus (after gain and pan). Silence reports −150 dBFS.

## Mapping from the current Android `timeline/Arrangement.kt`

The Android project was read but not modified.

| Arrangement.kt | Edit model | Notes |
|---|---|---|
| `Clip.id: Long` | `LaneItem.id: String` | Use `"item-$id"`. |
| `Clip.sourceId` | `LaneItem.sourceClipId` → `SourceClip.id` | Each source file is uploaded once as `clip:<id>`. |
| `Clip.sourceStart`, `Clip.length` (frames) | `range.startSample`, `range.endSample = sourceStart+length` | Same units, provided the frames are at `Project.sampleRate`. |
| `Clip.timelineStart` | `laneStartSample` | |
| `Clip.lane: Int` | Containment in `Lane.items`; `Lane.order` = the old index | `Lane.id = "lane-$index"`. |
| `Clip.gainDb` | `LaneItem.gainDb` | Per item, applied before the bus. Lane `gainDb` is the new bus fader. |
| `Clip.fadeIn/fadeOut` (frames) | `fadeInMs/fadeOutMs` | `ms = frames·1000/sr`. |
| `ClipMath.AUTO_FADE` 4 ms linear | `crossfade {75 ms, equal_power}` | Centred crossfade with handles replaces the 4 ms auto-fade. |
| (none: add only) | `mode add\|replace` | `replace` is new. |
| `Clip.fx: FxSettings` (per clip) | `Lane.processorChain` | FX are per lane now; see the open questions. |
| `Arrangement.lanes: Int` (≤12) | `lanes: List<Lane>` (≤12) | |
| `muteMain` | `mainMute` | |
| `mainSource "mix"/"master"` | `mainClipId` | Upload whichever file is the main clip. |
| snap (bpm beats, else 0.1 s grid) | `Snap{type, value, bpm}` | Snap rounding: use `rint` to match the server. |
| History / undo stack | `EditHistory` | The server mirror is `app/edit/history.py`. |
| `ClipOps.create/move/split/delete/setGain/addLane` | `add_item / move_item / split_item / delete_item / set_item / add_lane` | Split: the left part keeps the id with fadeOut=0; the right part gets a new id with fadeIn=0. |

## Open questions for the Android UI Bot / David

1. **LoopMode (OFF/COUNT/UNTIL_NEXT):** there is no loop field yet. One option is to expand loops into repeated
   items, which the seam rule makes gapless. The other is to add `LaneItem.loop` in schema v2.
2. **`mainCuts: List<Pair<Int,Int>>`:** these could become `replace` items on a lane, using a silent source or a
   per-item gain of −60 dB. Alternatively, a `Project.mainMutes` list could be added. Which one should we use?
3. **Per-clip `FxSettings` vs per-lane chain:** is lane-level FX acceptable, or does the UI need per-item FX?
4. **`compare` / `abOriginal`:** the plan is to map these to `view=main` vs `main_plus_lanes`, which are
   sample-aligned. Is that the intended behaviour?
5. **Units:** the UI uses frames and the model uses ms for fades. Are frames always at `Project.sampleRate`, or do
   they need resampling?
6. **`ClipOps.MIN_LEN` 64 frames:** the server accepts any non-empty range. Should 64 become a schema minimum?
7. **On-device EQ:** should the EQ biquads also be ported to Kotlin, or is EQ server-only?
