package com.david.aimusicengineer.presets

import kotlinx.serialization.EncodeDefault
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject

/**
 * A user's Freestyle preset, stored on the phone only. Its JSON IS a valid freestyle ToneProfile body
 * ({"mode":"freestyle","params":{...},"stems":{...}} plus id/name/createdAt/updatedAt/schemaVersion, which the
 * server accepts and ignores), so the exported/stored JSON can be sent as `toneProfile` to /separate or as
 * `profile` to /process/tone unchanged.
 *
 * The preset UI has fixed EQ bands: 100 Hz, 1 kHz, 4 kHz, 8 kHz (peaking) + "air" high shelf (12 kHz).
 */
@OptIn(ExperimentalSerializationApi::class)
@Serializable
data class FreestylePreset(
    val id: String,                       // UUID v4 string
    val name: String,
    val createdAt: Long,                  // epoch millis
    val updatedAt: Long,
    @EncodeDefault val schemaVersion: Int = CURRENT_SCHEMA_VERSION,
    @EncodeDefault val mode: String = "freestyle",
    val params: PresetParams = PresetParams(),
    /** Optional per-stem overrides keyed by stem name (vocals, drums, bass, other, ...): partial `params` objects. */
    val stems: Map<String, JsonObject>? = null,
) {
    companion object { const val CURRENT_SCHEMA_VERSION = 1 }
}

@Serializable
data class PresetParams(
    val eq: Eq = Eq(),
    val saturation: Saturation = Saturation(),
    val width: Width = Width(),
    val compressor: Compressor = Compressor(),
)

@Serializable
enum class BandType { @SerialName("peak") PEAK, @SerialName("lowshelf") LOWSHELF,
    @SerialName("highshelf") HIGHSHELF, @SerialName("notch") NOTCH }

@Serializable
data class EqBand(val enabled: Boolean = true, val type: BandType = BandType.PEAK, val freqHz: Double,
                  val gainDb: Double = 0.0, val q: Double = 0.707)

@Serializable
data class Eq(val bands: List<EqBand> = Eq.fixedBands()) {
    val gain100 get() = bands[0].gainDb
    val gain1k get() = bands[1].gainDb
    val gain4k get() = bands[2].gainDb
    val gain8k get() = bands[3].gainDb
    val air get() = bands[4].gainDb

    /** Index 0..4 = 100 Hz, 1 kHz, 4 kHz, 8 kHz, air. */
    fun withGain(index: Int, gainDb: Double) = copy(bands = bands.mapIndexed { i, b -> if (i == index) b.copy(gainDb = gainDb) else b })

    companion object {
        val FIXED_FREQS = doubleArrayOf(100.0, 1000.0, 4000.0, 8000.0, 12000.0)
        const val AIR_INDEX = 4
        fun fixedBands(g100: Double = 0.0, g1k: Double = 0.0, g4k: Double = 0.0, g8k: Double = 0.0, air: Double = 0.0) =
            listOf(EqBand(freqHz = 100.0, gainDb = g100, q = 1.0), EqBand(freqHz = 1000.0, gainDb = g1k, q = 1.0),
                   EqBand(freqHz = 4000.0, gainDb = g4k, q = 1.0), EqBand(freqHz = 8000.0, gainDb = g8k, q = 1.0),
                   EqBand(type = BandType.HIGHSHELF, freqHz = 12000.0, gainDb = air, q = 0.707))
    }
}

@Serializable
enum class SatType { @SerialName("tanh") TANH, @SerialName("tape") TAPE, @SerialName("tube") TUBE }

@Serializable
data class Saturation(val enabled: Boolean = false, val type: SatType = SatType.TANH, val driveDb: Double = 0.0,
                      val mix: Double = 1.0, val asymmetry: Double = 0.0)

@Serializable
data class Width(val enabled: Boolean = false, val widthPct: Double = 100.0, val monoBelowHz: Double = 120.0)

@Serializable
data class Compressor(val enabled: Boolean = false, val thresholdDb: Double = -18.0, val ratio: Double = 4.0,
                      val attackMs: Double = 10.0, val releaseMs: Double = 150.0, val kneeDb: Double = 6.0,
                      val makeupDb: Double = 0.0)

/** Clamps every value to the bundled /tone/schema freestyle ranges (offline). Returns the value + warnings. */
object PresetValidator {
    data class Result(val params: PresetParams, val warnings: List<String>)

    fun sanitize(p: PresetParams): Result {
        val w = mutableListOf<String>()
        fun c(key: String, v: Double): Double {
            val out = ToneRanges.clamp(key, v)
            if (out != v) w += "$key: $v clamped to $out"
            return out
        }
        // the preset UI always has exactly the 5 fixed bands; missing ones are added flat, extra ones dropped
        val fixed = Eq.fixedBands()
        if (p.eq.bands.size != fixed.size) w += "eq.bands: expected ${fixed.size} fixed bands, got ${p.eq.bands.size}"
        val bands = fixed.mapIndexed { i, f ->
            val b = p.eq.bands.getOrNull(i) ?: f
            b.copy(type = f.type, freqHz = f.freqHz, gainDb = c("eq.bands.gainDb", b.gainDb), q = c("eq.bands.q", b.q))
        }
        val s = p.saturation
        val wd = p.width
        val k = p.compressor
        return Result(PresetParams(
            Eq(bands),
            s.copy(driveDb = c("saturation.driveDb", s.driveDb), mix = c("saturation.mix", s.mix),
                   asymmetry = c("saturation.asymmetry", s.asymmetry)),
            wd.copy(widthPct = c("width.widthPct", wd.widthPct), monoBelowHz = c("width.monoBelowHz", wd.monoBelowHz)),
            k.copy(thresholdDb = c("compressor.thresholdDb", k.thresholdDb), ratio = c("compressor.ratio", k.ratio),
                   attackMs = c("compressor.attackMs", k.attackMs), releaseMs = c("compressor.releaseMs", k.releaseMs),
                   kneeDb = c("compressor.kneeDb", k.kneeDb), makeupDb = c("compressor.makeupDb", k.makeupDb)),
        ), w)
    }
}

val PresetJson = Json { encodeDefaults = true; ignoreUnknownKeys = true; prettyPrint = true; explicitNulls = false }
