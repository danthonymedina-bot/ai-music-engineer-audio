# On-device fallback (Android, ONNX Runtime Mobile)

Goal: offline / no-server separation + light DSP when `/health` is unreachable or `demucs:false`.

## Model choice
| Model | Params | ONNX friendliness | Quality | Notes |
|---|---|---|---|---|
| **Open-Unmix (umxl / umxhq)** | ~9 M per target | Good: LSTM on magnitude spectrogram; do STFT/iSTFT in Kotlin/C++ | OK | **Recommended first.** Export only the net; spectrogram I/O keeps graph simple. |
| HTDemucs | ~42 M | Hard: hybrid time+freq branches, complex STFT ops, transformer; needs opset ≥17 and STFT moved outside the graph | Best | Feasible but heavy (~170 MB fp32, ~45 MB int8); 30 s+ per song on flagship CPU. |

## Pipeline on device
1. Decode (MediaCodec/`MediaExtractor`) → float PCM 44.1 kHz stereo.
2. STFT (n_fft 4096, hop 1024) in Kotlin (JTransforms) or NDK (KissFFT).
3. ORT session: magnitude `[1, 2, 2049, T]` → masks/estimates per target, chunked ~6 s with overlap.
4. Apply mask to complex STFT, iSTFT, overlap-add, write WAV.

## Files
- `export_onnx.py` — exports Open-Unmix (default) or HTDemucs core to ONNX, then dynamic int8 quantization.
- `AudioSeparator.kt` — Kotlin wrapper sketch for ORT Mobile.

## Quantization notes
- Dynamic int8 (`quantize_dynamic`, weights only) works well for LSTM/Linear: ~4× smaller, ~1.5–2× faster, small SDR loss (~0.1–0.3 dB) — verify with museval on MUSDB18 samples.
- Static int8 (QDQ) needs calibration data; better for conv-heavy U-Nets / NNAPI.
- fp16 is an option for GPU/NNAPI execution providers.
- Use `--use_ort_format` conversion (`python -m onnxruntime.tools.convert_onnx_models_to_ort`) and a **reduced-op build** (`onnxruntime-mobile` custom build with the model's op config) to cut AAR size from ~15 MB to ~3–5 MB.
- Execution providers: CPU (XNNPACK) default; NNAPI is device-dependent for LSTM — benchmark before enabling.

## Alternative: MediaPipe / LiteRT
MediaPipe Audio tasks (LiteRT, ex-TFLite) offer classification out of the box but **no separation task**; you'd convert the model to `.tflite` (e.g. via `ai-edge-torch`) and run it with the LiteRT Interpreter (GPU delegate available). Pick this if the app already ships TFLite/MediaPipe; otherwise ORT keeps one export path from PyTorch.

## Light DSP on device
EQ/saturation/mix are trivial (biquads + tanh) — port `app/dsp/tone.py` / `mix.py` to Kotlin or use Oboe + C++. For pitch/tempo, **default = SoundTouch (LGPL, built as `libsoundtouch.so` and loaded dynamically via JNI)**; **Superpowered = paid upgrade**. Rubber Band stays server-only (GPL). See root README licensing table. For denoise: RNNoise or DeepFilterNet ONNX.

## Status
Sketch only — not run here (no torch/openunmix installed, no Android toolchain).
