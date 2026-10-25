// Sketch: on-device Open-Unmix separation with ONNX Runtime Mobile.
// Gradle: implementation("com.microsoft.onnxruntime:onnxruntime-android:<latest>")
//         implementation("com.github.wendykierp:JTransforms:3.1")
package com.davidmedina.aimusicengineer.ondevice

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer

class AudioSeparator(context: Context, assetName: String = "vocals.int8.ort") : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(
        context.assets.open(assetName).readBytes(),
        OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            // addNnapi()  // benchmark first; LSTM support varies by device
        }
    )

    companion object { const val N_FFT = 4096; const val HOP = 1024; const val BINS = N_FFT / 2 + 1; const val CHUNK_FRAMES = 256 }

    /** pcm: [2][n] float stereo @44.1k. Returns separated target [2][n]. */
    suspend fun separate(pcm: Array<FloatArray>): Array<FloatArray> = withContext(Dispatchers.Default) {
        val spec = Stft.forward(pcm, N_FFT, HOP)            // complex [2][BINS][T] (re, im) — implement with JTransforms
        val mag = spec.magnitude()                           // [2][BINS][T]
        val est = Array(2) { Array(BINS) { FloatArray(spec.frames) } }
        var start = 0
        while (start < spec.frames) {                        // chunked inference; add overlap+crossfade in production
            val len = minOf(CHUNK_FRAMES, spec.frames - start)
            val buf = FloatBuffer.allocate(2 * BINS * len)
            for (c in 0 until 2) for (b in 0 until BINS) for (t in 0 until len) buf.put(mag[c][b][start + t])
            buf.rewind()
            OnnxTensor.createTensor(env, buf, longArrayOf(1, 2, BINS.toLong(), len.toLong())).use { input ->
                session.run(mapOf("mag" to input)).use { out ->
                    val o = (out[0].value as Array<Array<Array<FloatArray>>>)[0]
                    for (c in 0 until 2) for (b in 0 until BINS) for (t in 0 until len) est[c][b][start + t] = o[c][b][t]
                }
            }
            start += len
        }
        // Soft mask = est / (mag + eps), applied to complex spec keeps the mixture phase
        val masked = spec.applyMask { c, b, t -> est[c][b][t] / (mag[c][b][t] + 1e-8f) }
        Stft.inverse(masked, N_FFT, HOP, pcm[0].size)
    }

    override fun close() { session.close() }
}

// Stft / ComplexSpec helpers intentionally omitted from the sketch (Hann window, JTransforms FloatFFT_1D.realForward).
