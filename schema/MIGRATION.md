# Migrating the Android UI from its stand-in model to `EditModel.kt`

The Android project was not modified. These steps are for the Android UI Bot.

## 1. Add the model

1. Copy `schema/EditModel.kt` to `app/src/main/java/com/david/aimusicengineer/edit/EditModel.kt`.
   - Optionally, copy `schema/kotlin-dsp/EditDsp.kt` to `.../edit/dsp/` for on-device playback.
2. Gradle (Kotlin 2.x):
   ```kotlin
   plugins { kotlin("plugin.serialization") version "<your kotlin version>" }
   dependencies {
       implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
       testImplementation("junit:junit:4.13.2")
   }
   ```
3. Copy `schema/kotlin-dsp/EditDspTest.kt` to `app/src/test/...`, and copy `schema/kotlin-dsp/test_vectors.json`
   to `app/src/test/resources/edit/test_vectors.json`.
   - Run `./gradlew test`. All 6 tests must pass; they pass against JDK 21 / Kotlin 2.0.21.
4. Serialize with `project.toJson()` and `Project.fromJson(s)`. Both use `EditJson`, which encodes defaults and
   rejects unknown keys.
   - This is the exact body for the `project` form field of `/edit/render` and `/edit/levels`.

## 2. Replace the stand-in incrementally

1. Keep `Arrangement` as the ViewModel state for now and add an adapter `Arrangement.toProject()`, sketched below.
   Render and meter through the server with it.
2. Move the UI state to `EditHistory` (`history.current` is the `Project`). Every user action becomes
   `history.commit(newProject, EditOp(...))`, and undo/redo become `history.undo()` / `history.redo()`.
3. Delete `Arrangement`/`ClipOps` once every screen reads from `Project`. Keep `ClipMath` snap helpers, but switch
   them to `Math.rint`.

```kotlin
fun Arrangement.toProject(sr: Int, mainId: String, bpm: Double?): Project {
    val lanesOut = (0 until lanes).map { idx ->
        Lane(id = "lane-$idx", name = "Lane ${idx + 1}", order = idx,
             items = clips.filter { it.lane == idx }.sortedBy { it.timelineStart }.map { c ->
                 LaneItem(id = "item-${c.id}", sourceClipId = c.sourceId,
                          range = Range(c.sourceStart, c.sourceStart + c.length),
                          laneStartSample = c.timelineStart, mode = ItemMode.ADD,
                          gainDb = c.gainDb.toDouble(),
                          fadeInMs = c.fadeIn * 1000.0 / sr, fadeOutMs = c.fadeOut * 1000.0 / sr)
             })
    }
    val sources = (clips.map { it.sourceId } + mainId).distinct().map { SourceClip(id = it) }
    return Project(id = "local", sampleRate = sr, clips = sources, mainClipId = mainId, mainMute = muteMain,
                   snap = if (bpm != null) Snap(type = SnapType.BEATS, value = 1.0, bpm = bpm) else Snap(),
                   lanes = lanesOut)
}
```

Sample positions are `Int` in `EditModel.kt`, as they are in Arrangement.kt. Per-clip `FxSettings`,
loops and `mainCuts` have no direct field yet; see the open questions in `README.md`.

## 3. UI control → field

| Control | Field / op | Notes |
|---|---|---|
| In/out markers on the source waveform | `selection = Selection(sourceClipId, Range(in, out))` | UI only; snap the markers with `Snap` and `rint`. |
| "Add to lane" / drop on lane | add `LaneItem(range = selection.range, laneStartSample = playhead)` | |
| Add ⇄ Replace toggle | `LaneItem.mode` (`ADD` / `REPLACE`) | Replace mutes everything underneath (main and earlier items in the lane) with centred crossfades. |
| Crossfade slider 50–100 ms | `LaneItem.crossfade.lengthMs` (default 75) | Clamp in the UI; the server rejects values out of range. |
| Crossfade curve switch | `crossfade.curve` (`EQUAL_POWER` default, `LINEAR` for re-cutting the same take) | |
| Item fade handles | `LaneItem.fadeInMs / fadeOutMs` | |
| Item gain | `LaneItem.gainDb` (−60..12) | |
| Drag item / move to another lane | `laneStartSample`; move the item between `Lane.items` | |
| Split at playhead | Left keeps the id with `fadeOutMs=0`; right gets a new id with `fadeInMs=0`, `range.start += cut`, `laneStartSample = cut` | Seam rule makes it bit-lossless. |
| Lane fader | `Lane.gainDb` (bus, −60..12) | Applied once to the lane sum after FX. |
| Lane pan knob | `Lane.pan` (−1..1, constant power) | |
| Lane M / S buttons | `Lane.mute`, `Lane.solo` | Mute wins. With any solo, only soloed lanes play; main always plays unless `mainMute`. |
| Lane color / name / reorder | `colorArgb` (Long 0xAARRGGBB), `name`, `order` | |
| Main volume / mute | `mainGainDb`, `mainMute` | |
| Delay: time | `params.timeMs` (1..4000) | |
| Delay: sync toggle + note value | `params.syncToSnap = true`, `params.beats` (for example 0.75 = dotted eighth with a beats snap) | |
| Delay: feedback / mix | `params.feedback` (0..0.95), `params.mix` (0..1) | |
| Delay: fade in/out of echoes | `Processor.fadeInMs / fadeOutMs` | Shapes only the echoes. |
| EQ knobs | `params.lowGainDb/midGainDb/highGainDb/lowFreq/midFreq/highFreq/midQ/drive/driveMix` | |
| FX bypass | `Processor.enabled` | |
| Snap: grid/BPM | `Snap.type`, `Snap.value`, `Snap.bpm` | |
| A/B (original vs edit) | Render with `view=main` vs `view=main_plus_lanes` | Same length, sample-aligned. |
| Level meters | `POST /edit/levels` (optional `startSample`/`numSamples`), or the `X-Edit-Meta` header on renders | Per-lane, main and master peak/RMS/headroom. |
| Limiter mode | `masterLimiter`, `softClipThresholdDb` | |
| Undo / redo | `EditHistory.undo() / redo()`; `commit()` drops the redo branch and caps history at 200 | |

## 4. On-device playback with `kotlin-dsp`

To render a block `[t0, t0+n)`:
1. For each lane, compose with lookbehind:
   `LaneComposer.compose(geoms, sources, t0 - lookbehind, n + lookbehind)`, using
   `geoms = ItemGeom.markSeams(items.map { ItemGeom.of(it, sr) })`.
2. Run `DelayProcessor.apply(...)` with the lane span from `LaneComposer.span(geoms)`.
3. Drop the lookbehind.
4. Multiply by `LaneBus.gains(lane.gainDb, lane.pan)` if `LaneBus.audible(lanes)[i]`.
5. Sum with the main clip × `dbToGain(mainGainDb)`.
6. Apply `SoftClip.apply` per sample.

Blocks computed this way concatenate to exactly the full render. Keep the math in `Double`, because the vectors
assume it.

The server stays authoritative for EQ and for anything not yet ported. If on-device output and server output
disagree, the server and `test_vectors.json` win.
