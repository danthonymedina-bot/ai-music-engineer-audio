// Edit model v1 — Kotlin mirror of app/edit/model.py / schema/edit_model.json (source of truth: the Python model).
// Drop into the Android app as e.g. app/src/main/java/com/david/aimusicengineer/edit/EditModel.kt and add
// kotlinx.serialization (plugin "org.jetbrains.kotlin.plugin.serialization" + "kotlinx-serialization-json").
//
// All positions are Int samples at Project.sampleRate. Ranges are half-open [startSample, endSample).
// JSON field names are identical to the server's; encode with EditJson so defaults are always written.
package com.david.aimusicengineer.edit

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject

const val EDIT_SCHEMA_VERSION = 1

val EditJson = Json { encodeDefaults = true; explicitNulls = true; ignoreUnknownKeys = false }

/** White marker = startSample, orange marker = endSample (exclusive). */
@Serializable
data class Range(val startSample: Int, val endSample: Int) {
    init { require(startSample >= 0 && endSample > startSample) { "bad range $startSample..$endSample" } }
    val length: Int get() = endSample - startSample
}

@Serializable
enum class SnapType { @SerialName("seconds") SECONDS, @SerialName("beats") BEATS }

@Serializable
data class Snap(
    val type: SnapType = SnapType.SECONDS,
    val value: Double = 0.1,          // seconds (SECONDS) or beats (BEATS) per grid step
    val bpm: Double? = null,          // required when type == BEATS
) {
    init { require(type == SnapType.SECONDS || (bpm != null && bpm > 0)) { "snap.bpm required for beats" } }
    fun unitSamples(sampleRate: Int): Double =
        if (type == SnapType.BEATS) value * 60.0 / bpm!! * sampleRate else value * sampleRate
}

/** Immutable source buffer. Uploaded to /edit/render as multipart part "clip:<id>". */
@Serializable
data class SourceClip(val id: String, val name: String? = null, val numSamples: Int? = null, val channels: Int? = null)

@Serializable
enum class CrossfadeCurve { @SerialName("equal_power") EQUAL_POWER, @SerialName("linear") LINEAR }

@Serializable
data class Crossfade(val lengthMs: Double = 75.0, val curve: CrossfadeCurve = CrossfadeCurve.EQUAL_POWER) {
    init { require(lengthMs in 50.0..100.0) { "crossfade.lengthMs must be 50..100" } }
}

@Serializable
enum class ItemMode { @SerialName("add") ADD, @SerialName("replace") REPLACE }

@Serializable
data class LaneItem(
    val id: String,
    val sourceClipId: String,
    val range: Range,
    val laneStartSample: Int,          // timeline position of range.startSample
    val mode: ItemMode = ItemMode.ADD,
    val gainDb: Double = 0.0,          // per-item gain (pre-bus), -60..12
    val fadeInMs: Double = 0.0,
    val fadeOutMs: Double = 0.0,
    val crossfade: Crossfade = Crossfade(),
) { val laneEndSample: Int get() = laneStartSample + range.length }

/** type is a registry key: "delay", "eq" (more later: "reverb", "pitch"). params are type-specific. */
@Serializable
data class Processor(
    val id: String,
    val type: String,
    val enabled: Boolean = true,
    val params: JsonObject = JsonObject(emptyMap()),
    val fadeInMs: Double = 0.0,
    val fadeOutMs: Double = 0.0,
)

@Serializable
data class Lane(
    val id: String,
    val name: String = "",
    val order: Int = 0,
    val colorArgb: Long? = null,       // 0xAARRGGBB as unsigned (Long so 0xFF...... fits)
    val gainDb: Double = 0.0,          // lane BUS gain: once on the lane sum, after the processor chain
    val pan: Double = 0.0,             // -1..1, constant power
    val mute: Boolean = false,         // mute wins over solo
    val solo: Boolean = false,         // any solo => only soloed lanes play (main clip unaffected)
    val items: List<LaneItem> = emptyList(),
    val processorChain: List<Processor> = emptyList(),
)

/** UI-only; ignored by the renderer. */
@Serializable
data class Selection(val sourceClipId: String, val range: Range)

@Serializable
enum class MasterLimiter { @SerialName("soft_clip") SOFT_CLIP, @SerialName("gain_compensation") GAIN_COMPENSATION }

@Serializable
data class Project(
    val version: Int = EDIT_SCHEMA_VERSION,
    val id: String,
    val sampleRate: Int = 44100,
    val snap: Snap = Snap(),
    val clips: List<SourceClip> = emptyList(),
    val mainClipId: String? = null,    // plays from timeline sample 0 ("main" view)
    val mainGainDb: Double = 0.0,
    val mainMute: Boolean = false,
    val lanes: List<Lane> = emptyList(),
    val selection: Selection? = null,
    val masterLimiter: MasterLimiter = MasterLimiter.SOFT_CLIP,
    val softClipThresholdDb: Double = -1.0,
) {
    fun lane(id: String) = lanes.first { it.id == id }
    fun orderedLanes() = lanes.sortedWith(compareBy({ it.order }, { it.id }))
    fun toJson(): String = EditJson.encodeToString(serializer(), this)
    companion object { fun fromJson(s: String): Project = EditJson.decodeFromString(serializer(), s) }
}

/** One undoable edit; before/after hold each target's JSON (null = absent). */
@Serializable
data class EditOp(
    val opId: String,
    val type: String,                  // addItem, moveItem, splitItem, deleteItem, setLane, addProcessor, setProcessor, setSnap...
    val targetIds: List<String>,
    val before: Map<String, JsonElement?> = emptyMap(),
    val after: Map<String, JsonElement?> = emptyMap(),
)

/** Immutable snapshots + op log. snapshots.size == ops.size + 1; cursor indexes the current snapshot. */
@Serializable
data class EditHistory(val snapshots: List<Project>, val ops: List<EditOp> = emptyList(), val cursor: Int = 0) {
    init { require(snapshots.size == ops.size + 1 && cursor in snapshots.indices) }
    val current: Project get() = snapshots[cursor]
    val canUndo get() = cursor > 0
    val canRedo get() = cursor < snapshots.size - 1
    fun undo() = if (canUndo) copy(cursor = cursor - 1) else this
    fun redo() = if (canRedo) copy(cursor = cursor + 1) else this
    /** Drops the redo branch; keeps at most [max] ops. */
    fun commit(next: Project, op: EditOp, max: Int = 200): EditHistory {
        var s = snapshots.subList(0, cursor + 1) + next
        var o = ops.subList(0, cursor) + op
        if (o.size > max) { s = s.takeLast(max + 1); o = o.takeLast(max) }
        return EditHistory(s, o, s.size - 1)
    }
    companion object { fun of(p: Project) = EditHistory(listOf(p)) }
}

/** Processor.type values known to the server registry (app/edit/processors.py). Freestyle tone types take the same
 *  params as the matching section of a freestyle ToneProfile (GET /tone/schema) and are clamped, never rejected. */
object ProcessorTypes {
    const val DELAY = "delay"
    const val EQ = "eq"
    const val TONE = "tone"                    // whole freestyle tone chain: params = ToneProfile.params
    const val WIDTH = "width"
    const val COMPRESSOR = "compressor"
    const val TRANSIENT = "transient"
    const val SATURATION = "saturation"        // params.type = tanh|tape|tube
    const val SATURATION_TAPE = "saturation_tape"
    const val SATURATION_TUBE = "saturation_tube"
    const val LOW_CUT = "lowcut"
    const val HIGH_CUT = "highcut"
    val ALL = listOf(DELAY, EQ, TONE, WIDTH, COMPRESSOR, TRANSIENT, SATURATION, SATURATION_TAPE, SATURATION_TUBE, LOW_CUT, HIGH_CUT)
}
