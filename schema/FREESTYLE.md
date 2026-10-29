# Freestyle tone mode for the Android UI Bot (phone-only)

Freestyle is an unlocked tone sandbox, set parameter by parameter, with no preset grid. The phone builds every
slider from `GET /tone/schema` and sends whatever the user dialled in. The server **never rejects a value in
freestyle mode**. Each value is clamped to a safe DSP range and reported in `warnings`. Unknown keys are ignored,
also with a warning.

## 1. Build the sliders from `GET /tone/schema`

Send the `X-API-Key` header. The response has the same content as `schema/tone_schema.json`; a bundled offline copy
for Kotlin is `kotlin-presets/ToneRanges.kt`.

```json
{"version":1, "chainOrder":["denoise","pitchTempo","tone","laneBus"],
 "toneOrder":["lowCut","highCut","eq","compressor","transient","saturation","width","output"],
 "sections":{"width":{"enabled":{"kind":"bool","default":false},
                      "widthPct":{"kind":"number","min":0,"max":200,"default":100,"unit":"%","step":1,
                                  "scale":"linear","presetMin":50,"presetMax":150,"description":"..."}, ...}, ...},
 "defaults":{...}}
```

`kind` decides the control:

| kind | Control | How to build it |
|---|---|---|
| `number` | Slider | Range `min..max`. Use a log slider when `scale == "log"`. Show `unit`. Snap to `step` if present. Reset to `default` on double-tap. |
| `bool` | Switch | |
| `enum` | Segmented control | Options from `options`. |
| `list` (`eq.bands`) | Add-band button | Up to `maxItems` bands, each built from `item`. |

Draw the preset-mode range (`presetMin..presetMax`) as a shaded "safe zone" if you like. Freestyle may go beyond
it, up to `min..max`.

## 2. Ranges

Freestyle clamps to `min..max`. Preset mode requires `presetMin..presetMax` and returns 422 otherwise.

| Parameter | Freestyle range (default) | Preset range | Notes |
|---|---|---|---|
| `denoise.enabled` / `preset` / `strength` | off / hiss\|air\|wind / 0..1 (1) | same | Runs first; spectral gating. |
| `pitchTempo.semitones` | −24..24 (0) | −12..12 | Rubber Band; runs second. |
| `pitchTempo.tempo` | 0.25..4 × (1) | 0.5..2 | |
| `lowCut.freqHz` / `slopeDbOct` | 20..2000 Hz (80) / 6..48 (24) | 20..500 | Butterworth; slope rounded to a multiple of 6. |
| `highCut.freqHz` / `slopeDbOct` | 1000..20000 Hz (16000) / 6..48 | 3000..20000 | |
| `eq.bands[]` | ≤32 bands (preset: ≤6); type peak\|lowshelf\|highshelf\|notch | | |
| `eq.bands[].freqHz` | 20..20000 Hz | same | Every filter frequency is also clamped to 0.45 × sampleRate at render time, with a warning. |
| `eq.bands[].gainDb` | −30..30 dB (0) | −12..12 | Notch ignores gain. |
| `eq.bands[].q` | 0.1..18 (0.707) | 0.3..4 | |
| `compressor.thresholdDb` | −60..0 (−18) | −40..0 | Stereo-linked, RMS-ish detector, soft knee. |
| `compressor.ratio` | 1..20 (4) | 1..8 | |
| `compressor.attackMs` / `releaseMs` | 0.1..200 (10) / 5..2000 (150) | 1..100 / 20..1000 | |
| `compressor.kneeDb` / `makeupDb` | 0..24 (6) / 0..24 (0) | – / 0..12 | |
| `transient.attackDb` / `sustainDb` | −12..12 (0) | −6..6 | Gain is bounded to ±12 dB. |
| `saturation.type` | tanh\|tape\|tube | | |
| `saturation.driveDb` | 0..36 (0) | 0..18 | Unity at full scale. |
| `saturation.mix` / `asymmetry` | 0..1 (1) / 0..1 (0) | | Asymmetry adds even harmonics; DC is removed. |
| `width.widthPct` | 0..200 % (100) | 50..150 | 0 = exact mono; 100 = exact identity. |
| `width.monoBelowHz` | 0..500 Hz (120) | | When widening (>100 %), the side signal below this frequency is never widened (LR4 split). 0 = off. |
| `output.gainDb` | −36..12 (0) | −24..6 | WAV output clips at 0 dBFS; the count is in `meta.tone.samplesOver0dBFS`. |

How bad values are handled:
- Non-numbers, NaN and ±Infinity fall back to the **default**. Numeric strings such as `"3"` are accepted.
- Huge values clamp to the edge of the range.
- Bad enum values fall back to the default.
- Stem names must match `^[A-Za-z0-9_.-]{1,32}$`, with at most 16 stems.

A flat or default profile is an exact identity: a stage is skipped when it is disabled or set to its neutral value.

## 3. Request bodies

```json
{"mode": "freestyle",
 "params": {"eq": {"bands": [{"type":"peak","freqHz":3000,"gainDb":4,"q":1.2}]},
            "saturation": {"enabled": true, "type": "tube", "driveDb": 14, "mix": 0.4},
            "width": {"enabled": true, "widthPct": 170},
            "compressor": {"enabled": true, "thresholdDb": -24, "ratio": 3}},
 "stems": {"vocals": {"compressor": {"ratio": 6}}, "bass": {"width": {"widthPct": 0}}}}
```

- **`params`**: partial is fine; missing sections are filled with defaults.
- **`stems.<name>`**: partial overrides deep-merged onto `params` (lists such as `eq.bands` are replaced as a whole).
- **`mode: "preset"`**: same shape with strict preset ranges. Out-of-range values return 422
  `{"detail":{"errors":[...]}}`.

Endpoints (all need `X-API-Key`; the rate limits are unchanged):

| Call | Body | Result |
|---|---|---|
| `POST /process/tone` | multipart `file`, `profile` (or `toneProfile`) = the JSON above, optional `stem=vocals` to apply that stem's override | Job view. `meta.steps`, `meta.warnings` (from the profile), `meta.tone.{stages,warnings,peak,samplesOver0dBFS}` and `urls.output`. The old form fields (`low_gain_db`, …, `mix`) still work unchanged when `profile` is absent. |
| `POST /separate` | multipart `file`, `model`, optional `toneProfile` | After separation, every stem runs **denoise → pitchTempo → tone** with `for_stem(name)`. Processed stems keep their names; dry stems become `<name>_raw`. `meta.toneProfile.{warnings,stems,unmatchedStemOverrides}`. |
| `GET /tone/schema` | – | The ranges document above. |

The phone can also put freestyle stages on edit lanes (`EditModel.kt` `ProcessorTypes`):
`tone` (params = the whole freestyle `params` minus denoise/pitchTempo), `width`, `compressor`, `transient`,
`saturation` / `saturation_tape` / `saturation_tube`, `lowcut` and `highcut`. They run before the lane bus
(gain → pan → mute/solo), are block-exact (gapless) and clamp in the same way.

### Chain order (fixed)

**denoise (spectral gating) → pitch/tempo (Rubber Band) → tone (lowCut → highCut → EQ → compressor → transient →
saturation → width → output) → lane bus.**

The tone stage streams file to file in 64k-frame blocks with carried filter state. A 5-min stereo 44.1 kHz file
through all 8 tone stages took 3.1 s on the box, with no measurable RSS growth above the interpreter baseline.

## 4. Local custom presets (no backend storage)

Files are in `schema/kotlin-presets/` (package `com.david.aimusicengineer.presets`):

| File | Purpose |
|---|---|
| `FreestylePreset.kt` | Serializable preset with `id` (UUID), `name`, `createdAt`, `updatedAt` (epoch ms), `schemaVersion` and `mode="freestyle"`. `params` holds `eq` (fixed bands 100 Hz / 1 kHz / 4 kHz / 8 kHz peaks plus a 12 kHz "air" shelf), `saturation`, `width` and `compressor`. Optional `stems` overrides. `PresetValidator.sanitize()` clamps with the bundled ranges. |
| `ToneRanges.kt` | **Generated** offline copy of the `/tone/schema` numbers (`scripts/gen_tone_schema.py`; CI fails if it is stale). Same rule as the server: non-finite → default, otherwise clamp. |
| `PresetRepository.kt` | `PresetRepository` interface, `FilePresetRepository` (one JSON file per preset under `filesDir/presets`) and `PresetMigrations`. |
| `PresetRepositoryTest.kt` | 10 JUnit4 tests (temp dir) covering every operation, rename conflicts, corrupt-file recovery and migration. |
| `example_preset.json` | A real export. The server test proves it parses with zero warnings, in both freestyle and preset mode. |

**A preset file is already a valid `toneProfile`.** Send `repo.exportJson(id)` as `toneProfile` to `/separate` or as
`profile` to `/process/tone`. The server ignores `id`/`name`/`createdAt`/`updatedAt`/`schemaVersion` and does not
store presets.

Repository behaviour:
- **`save(name, params, stems, overwrite=false)`**: names are trimmed and whitespace-collapsed, 1..64 characters,
  and unique **case-insensitively**. A conflict throws `PresetNameConflictException(existingId)`;
  `overwrite = true` keeps the existing id and `createdAt` and bumps `updatedAt`.
- **`list()`** is sorted by name, case-insensitively.
- **Observing the list:** `presets: StateFlow<List<FreestylePreset>>` emits after every change.
- **Loading:** `load(id)` and `loadByName(name)`.
- **`rename(id, newName)`** gives the same conflict error; a case-only rename of the same preset is allowed.
- **`delete(id)`** returns a Boolean.
- **`exportJson(id)` / `importJson(json, overwrite=false)`**:
  - On import, an id collision gets a new UUID.
  - A name conflict gets " (2)", " (3)" and so on appended, or, with `overwrite`, replaces the preset of that name.
  - Bad JSON or a newer schema throws `InvalidPresetException`.
- **Atomic writes:** each write goes to a `.<id>.json.tmp` file, which is fsynced and then `ATOMIC_MOVE`d into place.
  Leftover temp files are deleted on load.
- **Corrupt files** are moved to `presets/corrupt/` and listed in `quarantined: StateFlow<List<String>>`, so the
  app keeps working. Show a one-time "1 preset couldn't be read" notice.
- **Migration:**
  - `schemaVersion` < current runs `PresetMigrations` step by step and rewrites the file. v0 is the early stand-in
    format `{eqGains[5], drive, widthPct}`.
  - Files newer than the app are kept, not quarantined, and hidden.
  - When you bump `CURRENT_SCHEMA_VERSION`, add a `n → n+1` entry.

Gradle: kotlinx-serialization-json 1.7.3 and kotlinx-coroutines-core 1.8.x (Android: -android). For tests, add
`junit:4.13.2` and copy `PresetRepositoryTest.kt` into `src/test`. The tests ran with JDK 21 / Kotlin 2.0.21.
`rangesMatchServerSchemaFile` runs only when `-DtoneSchema=<path to schema/tone_schema.json>` is set.

### Wiring the Save button

```kotlin
// Hilt/manual singleton
val repo = FilePresetRepository(File(context.filesDir, "presets"))

// ViewModel
val presets = repo.presets            // collectAsStateWithLifecycle() in Compose
init { viewModelScope.launch { repo.refresh() } }

fun onSaveClicked(name: String, ui: PresetParams, stems: Map<String, JsonObject>?) = viewModelScope.launch {
    try {
        repo.save(name, ui, stems)                               // success: list re-emits
        _events.emit(UiEvent.Saved(name))
    } catch (e: PresetNameConflictException) {
        _events.emit(UiEvent.AskOverwrite(name))                 // dialog: "Replace 'name'?" -> save(..., overwrite = true)
    } catch (e: InvalidPresetException) {
        _events.emit(UiEvent.Error(e.message))                   // empty / >64-char name
    }
}

fun onPresetPicked(p: FreestylePreset) { uiState.value = uiState.value.copy(params = p.params, stems = p.stems) }
fun onShare(id: String) = viewModelScope.launch { shareText(repo.exportJson(id)) }        // ACTION_SEND text/json
fun onImport(text: String) = viewModelScope.launch { repo.importJson(text) }
fun onApplyToSong(id: String) = viewModelScope.launch {
    api.separate(file, toneProfile = repo.exportJson(id))       // or api.processTone(file, profile = ...)
}
```

- Show `PresetValidator.sanitize(ui).warnings` inline. The sliders cannot exceed the ranges anyway, but imported or
  migrated files can.
- To edit an existing preset, call `save(sameName, newParams, overwrite = true)`.
- If you prefer Jetpack DataStore, keep the `PresetRepository` interface and its semantics. The tests are written
  against the interface plus a directory.

## 5. Open questions

1. **EQ in the preset model:** the presets have fixed bands (100/1k/4k/8k + air shelf, Q 1.0). Should users be
   able to move preset band frequencies or Q, or is a full N-band preset needed? The server already accepts up to 32
   bands.
2. **Preset contents:** presets cover EQ, saturation, width and compressor only. Should the cuts, transient,
   output gain, denoise and pitch/tempo also be saved in a preset?
3. **Lane processors:** there is no `denoise` or `pitchTempo` per lane, because both change timing and are not
   block-exact. Is per-stem (in `/separate`) enough?
4. **Output safety:** output above 0 dBFS is clipped in the 16-bit WAV and counted. Should freestyle get an
   optional output soft-clip or limiter?
5. **Bass when widening:** `monoBelowHz` only stops the bass being *widened*; below 100 % width the bass narrows
   with everything else. Should there be a "force mono bass" option?
6. **Demucs:** the free tier runs the umxhq ONNX backend, not Demucs. The tone profile applies the same way to any
   backend.
