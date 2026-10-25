"""Export a lightweight separation model to ONNX (+ int8) for ONNX Runtime Mobile.

    pip install torch openunmix onnx onnxruntime    # demucs too for --model htdemucs
    python export_onnx.py --model umxhq --target vocals --out vocals.onnx
    python -m onnxruntime.tools.convert_onnx_models_to_ort vocals.int8.onnx

UNTESTED sketch: run on a dev machine; validate with onnxruntime vs torch outputs (see check()).
"""
import argparse

import numpy as np
import torch


def export_umx(name: str, target: str, out: str, frames: int = 512):
    from openunmix import utils
    sep = utils.load_separator(name, targets=[target], device="cpu", pretrained=True)
    net = sep.target_models[target].eval()  # OpenUnmix: input (nb_samples, nb_channels, nb_bins, nb_frames) magnitude
    dummy = torch.rand(1, 2, 2049, frames)
    # Static frame count: the traced LSTM reshape bakes in T, so dynamic axes don't actually work.
    # Callers window + zero-pad to `frames` (see app/dsp/onnx_separation.py).
    torch.onnx.export(net, dummy, out, input_names=["mag"], output_names=["est"], opset_version=17, dynamo=False)
    return dummy


def export_htdemucs(out: str, seconds: float = 7.8):
    # Experimental: HTDemucs includes STFT inside forward(); torch.onnx STFT export is partial.
    # Practical route is to wrap the inner model so STFT/iSTFT happen outside (see demucs/htdemucs.py _spec/_ispec).
    from demucs.pretrained import get_model
    model = get_model("htdemucs").models[0].eval()
    dummy = torch.rand(1, 2, int(44100 * seconds))
    torch.onnx.export(model, dummy, out, input_names=["wav"], output_names=["stems"], opset_version=17)
    return dummy


def quantize(src: str) -> str:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    dst = src.replace(".onnx", ".int8.onnx")
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
    return dst


def check(path: str, dummy: torch.Tensor):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    y = s.run(None, {s.get_inputs()[0].name: dummy.numpy()})[0]
    print(path, "output", y.shape, "finite", np.isfinite(y).all())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="umxhq", help="umxhq (MIT code) | umxl (weights CC BY-NC-SA!) | htdemucs")
    ap.add_argument("--target", default="vocals")
    ap.add_argument("--out", default="model.onnx")
    ap.add_argument("--no-quant", action="store_true")
    a = ap.parse_args()
    d = export_htdemucs(a.out) if a.model == "htdemucs" else export_umx(a.model, a.target, a.out)
    check(a.out, d)
    if not a.no_quant:
        check(quantize(a.out), d)
