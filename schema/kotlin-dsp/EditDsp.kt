// Reference DSP for on-device playback of the edit model. Mirrors app/edit/engine.py + processors.py exactly;
// verified against test_vectors.json by EditDspTest.kt. Pure functions of the timeline sample => block-exact
// (any block size / seek gives the same samples as a full render), so it is safe on the audio thread.
// Rounding note: Python's round() is round-half-even; use Math.rint (NOT roundToInt) for every ms->samples step.
package com.david.aimusicengineer.edit.dsp

import com.david.aimusicengineer.edit.CrossfadeCurve
import com.david.aimusicengineer.edit.ItemMode
import com.david.aimusicengineer.edit.Lane
import com.david.aimusicengineer.edit.LaneItem
import com.david.aimusicengineer.edit.Processor
import com.david.aimusicengineer.edit.Snap
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonPrimitive
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sign
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tanh

fun msToSamples(ms: Double, sr: Int): Int = Math.rint(ms * sr / 1000.0).toInt()
fun dbToGain(db: Double): Double = 10.0.pow(db / 20.0)
private fun clamp01(x: Double) = if (x < 0.0) 0.0 else if (x > 1.0) 1.0 else x

/** Equal power: out = cos(w*pi/2), in = sin(w*pi/2). Linear: out = 1-w, in = w. Returns (gainOut, gainIn). */
object CrossfadeMath {
    fun gains(w: Double, curve: CrossfadeCurve): Pair<Double, Double> = when (curve) {
        CrossfadeCurve.EQUAL_POWER -> cos(w * PI / 2) to sin(w * PI / 2)
        CrossfadeCurve.LINEAR -> (1.0 - w) to w
    }
}

/**
 * Geometry of one lane item. Crossfade X samples centred on each boundary (h = X / 2):
 * start window [P-h, P-h+X) fades new in / old out; end window [P+L-h, P+L-h+X) the reverse.
 */
data class ItemGeom(
    val item: LaneItem, val P: Int, val L: Int, val X: Int, val h: Int, val fadeIn: Int, val fadeOut: Int,
    val gain: Double, val hardIn: Boolean = false, val hardOut: Boolean = false,
) {
    val start: Int get() = if (hardIn) P else P - h
    val end: Int get() = if (hardOut) P + L else P + L - h + X

    fun weight(t: Int): Double {
        val x = max(X, 1).toDouble()
        val up = if (hardIn) (if (t >= P) 1.0 else 0.0) else (t - (P - h) + 0.5) / x
        val down = if (hardOut) (if (t < P + L) 1.0 else 0.0) else ((P + L - h + max(X, 1)) - t - 0.5) / x
        return clamp01(min(up, down))
    }

    fun envelope(t: Int): Double {
        var e = 1.0
        if (fadeIn > 0) e *= clamp01((t - P + 0.5) / fadeIn)
        if (fadeOut > 0) e *= clamp01((P + L - t - 0.5) / fadeOut)
        return e * gain
    }

    companion object {
        fun of(it: LaneItem, sr: Int): ItemGeom {
            val L = it.range.length
            val X = min(msToSamples(it.crossfade.lengthMs, sr), L)
            return ItemGeom(it, it.laneStartSample, L, X, X / 2, min(msToSamples(it.fadeInMs, sr), L),
                min(msToSamples(it.fadeOutMs, sr), L), dbToGain(it.gainDb))
        }

        /** Contiguous seam rule (lossless Split): B continues A exactly => no crossfade at that boundary. */
        fun markSeams(g: List<ItemGeom>): List<ItemGeom> {
            val out = g.toMutableList()
            for (i in g.indices) for (j in g.indices) {
                val a = g[i].item; val b = g[j].item
                if (i != j && a.sourceClipId == b.sourceClipId && a.mode == b.mode && a.gainDb == b.gainDb &&
                    a.crossfade == b.crossfade && a.range.endSample == b.range.startSample &&
                    a.laneEndSample == b.laneStartSample) {
                    out[i] = out[i].copy(hardOut = true); out[j] = out[j].copy(hardIn = true)
                }
            }
            return out
        }
    }
}

/** A stereo source: channel arrays of equal length. */
class StereoSource(val left: FloatArray, val right: FloatArray) { val frames get() = left.size }

/** Composite of a lane's items over timeline [t0, t0+n) (stateless). Items apply in list order. */
object LaneComposer {
    fun compose(geoms: List<ItemGeom>, sources: Map<String, StereoSource>, t0: Int, n: Int): Array<DoubleArray> {
        val l = DoubleArray(n); val r = DoubleArray(n); val t1 = t0 + n
        for (g in geoms) {
            val a = max(t0, g.start); val b = min(t1, g.end)
            if (a >= b) continue
            val src = sources.getValue(g.item.sourceClipId)
            val rs = g.item.range.startSample
            for (t in a until b) {
                val (gOld, gNew) = CrossfadeMath.gains(g.weight(t), g.item.crossfade.curve)
                val si = rs + (t - g.P)
                val k = g.envelope(t) * gNew
                val nl = if (si in 0 until src.frames) src.left[si] * k else 0.0
                val nr = if (si in 0 until src.frames) src.right[si] * k else 0.0
                val i = t - t0
                if (g.item.mode == ItemMode.REPLACE) { l[i] *= gOld; r[i] *= gOld }
                l[i] += nl; r[i] += nr
            }
        }
        return arrayOf(l, r)
    }

    /** Lane content span (first..one-past-last contributing sample); null if the lane is empty. */
    fun span(geoms: List<ItemGeom>): Pair<Int, Int>? =
        if (geoms.isEmpty()) null else geoms.minOf { it.start } to geoms.maxOf { it.end }
}

object TempoSync {
    /** `beats` snap units -> samples (round half even, like the server). */
    fun delaySamples(beats: Double, snap: Snap, sr: Int): Int = Math.rint(beats * snap.unitSamples(sr)).toInt()
}

/**
 * Sample-accurate feedback delay, block-exact via a truncated feedback series:
 *   d[n] = sum_{k=1..taps} fb^(k-1) x[n - kD];   y = (1-mix) x + env(t) * mix * d
 * env (fadeIn/fadeOut) scales only the delayed section: 0->1 over fadeIn from spanStart, 1->0 over fadeOut ending
 * at spanEnd + tail. Feed it the lane composite starting `lookbehind` samples before the block you need.
 */
class DelayProcessor(val delay: Int, val feedback: Double, val mix: Double, val fadeIn: Int, val fadeOut: Int,
                     val enabled: Boolean = true, sr: Int) {
    val taps: Int = run {
        val k = if (feedback == 0.0) 1 else ceil(ln(1e-4) / ln(feedback)).toInt()
        max(1, min(k, floor(30.0 * sr / delay).toInt()))
    }
    val lookbehind get() = if (enabled) taps * delay else 0
    val tail get() = if (enabled) taps * delay else 0

    fun envelope(t: Int, spanStart: Int, spanEnd: Int): Double {
        var e = 1.0
        if (fadeIn > 0) e *= clamp01((t - spanStart + 1).toDouble() / fadeIn)
        if (fadeOut > 0) e *= clamp01((spanEnd + tail - t).toDouble() / fadeOut)
        return e
    }

    /** x: one channel starting at timeline sample t0. Returns a new array of the same length. */
    fun apply(x: DoubleArray, t0: Int, spanStart: Int, spanEnd: Int): DoubleArray {
        if (!enabled) return x.copyOf()
        val m = x.size
        val d = DoubleArray(m)
        var g = 1.0
        for (k in 1..taps) {
            val s = k * delay
            if (s >= m) break
            for (i in s until m) d[i] += g * x[i - s]
            g *= feedback
        }
        val fades = fadeIn > 0 || fadeOut > 0
        return DoubleArray(m) { i ->
            val e = if (fades) envelope(t0 + i, spanStart, spanEnd) else 1.0
            (1.0 - mix) * x[i] + mix * (d[i] * e)
        }
    }

    companion object {
        fun of(p: Processor, sr: Int, snap: Snap): DelayProcessor {
            require(p.type == "delay")
            fun num(k: String, def: Double) = p.params[k]?.jsonPrimitive?.doubleOrNull ?: def
            val sync = p.params["syncToSnap"]?.jsonPrimitive?.booleanOrNull ?: false
            val d = if (sync) TempoSync.delaySamples(num("beats", 1.0), snap, sr) else msToSamples(num("timeMs", 250.0), sr)
            return DelayProcessor(d, num("feedback", 0.0), num("mix", 0.5), msToSamples(p.fadeInMs, sr),
                msToSamples(p.fadeOutMs, sr), p.enabled, sr)
        }
    }
}

/** Lane bus: gain once on the lane sum (post chain), constant-power pan normalised to unity at centre. */
object LaneBus {
    fun gains(gainDb: Double, pan: Double): Pair<Double, Double> {
        val g = dbToGain(gainDb); val th = (pan.coerceIn(-1.0, 1.0) + 1) * PI / 4
        return cos(th) * sqrt(2.0) * g to sin(th) * sqrt(2.0) * g
    }
    /** Solo-any: if any lane is soloed only soloed lanes play; mute always wins. Empty lanes are not audible. */
    fun audible(lanes: List<Lane>): List<Boolean> {
        val anySolo = lanes.any { it.solo }
        return lanes.map { it.items.isNotEmpty() && !it.mute && (it.solo || !anySolo) }
    }
    fun audibleFlags(mute: List<Boolean>, solo: List<Boolean>): List<Boolean> {
        val anySolo = solo.any { it }
        return mute.indices.map { !mute[it] && (solo[it] || !anySolo) }
    }
}

/** Master: bit-transparent below threshold T; above, a tanh knee approaching ceiling C = -0.1 dBFS. */
object SoftClip {
    val CEILING = 10.0.pow(-0.1 / 20)
    fun apply(x: Double, thresholdDb: Double = -1.0): Double {
        val t = dbToGain(thresholdDb); val a = abs(x)
        return if (a <= t) x else sign(x) * (t + (CEILING - t) * tanh((a - t) / (CEILING - t)))
    }
}
