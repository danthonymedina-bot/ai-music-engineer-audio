# Open-Unmix `umxhq` — int8 ONNX (vocals / drums / bass / other)

- Source: `openunmix` pretrained `umxhq` (Stöter et al., Inria), Zenodo DOI 10.5281/zenodo.3370489, trained on MUSDB18-HQ.
- Exported with `ondevice/export_onnx.py` (opset 17, static 512-frame input, `[1, 2, 2049, 512]` magnitude → magnitude, 44.1 kHz, n_fft 4096, hop 1024),
  then `onnxruntime.quantization.quantize_dynamic` (QInt8 weights). ~9 MB each.
- Code license: MIT. **Weights license: verify on the Zenodo record before commercial use** (MUSDB18 itself is research-oriented). Do NOT swap in `umxl` — its weights are CC BY-NC-SA 4.0 (non-commercial).
- int8 vs fp32 on a synthetic 3-min test: vocals SDR 3.92 vs 3.96 dB (negligible loss), ~2× faster, ~100 MB less RAM.
