#!/usr/bin/env python3
"""Native Core ML export (Apple Neural Engine / GPU / CPU) and a small benchmark + parity check.

The model is identical to the ONNX export (same EMA weights, pos-embed baked for the canvas, softmax output).
Precision FP16, compute units ALL; batch 1 (latency) and batch 16 (throughput) variants.
  python -m rotlab.coreml_export CKPT OUT_PREFIX --canvas 224 [--parity views.npz --onnx model-fp32.onnx]
"""
import argparse, json, time

import numpy as np
import torch
import coremltools as ct

from rotlab.export import Prob
from rotlab.model import RotNet


def load(ckpt, canvas):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False); sd = dict(ck['ema'])
    if canvas != ck['config']['img_size']:
        from timm.layers import resample_abs_pos_embed
        sd['backbone.pos_embed'] = resample_abs_pos_embed(sd['backbone.pos_embed'], (canvas // 14, canvas // 14), num_prefix_tokens=1)
    m = RotNet(img_size=canvas, pretrained=False, arch=ck['config'].get('arch', 's')); m.load_state_dict(sd)
    return Prob(m.eval())


def convert(model, canvas, batch, out):
    x = torch.randn(batch, 3, canvas, canvas)
    traced = torch.jit.trace(model, x)
    ml = ct.convert(traced, inputs=[ct.TensorType(name='image', shape=x.shape)], outputs=[ct.TensorType(name='prob')],
                    convert_to='mlprogram', compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.ALL,
                    minimum_deployment_target=ct.target.macOS14)
    ml.save(out); return out


def bench(path, canvas, batch, units, runs=50):
    m = ct.models.MLModel(path, compute_units=units); x = np.random.rand(batch, 3, canvas, canvas).astype(np.float32)
    for _ in range(5): m.predict({'image': x})
    t = []
    for _ in range(runs):
        t0 = time.perf_counter(); m.predict({'image': x}); t.append((time.perf_counter() - t0) * 1000)
    t = sorted(t); return dict(p50_ms=round(t[len(t) // 2], 2), p95_ms=round(t[int(len(t) * .95) - 1], 2), throughput_img_s=round(batch * 1000 / t[len(t) // 2], 1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('ckpt'); ap.add_argument('out'); ap.add_argument('--canvas', type=int, default=224)
    ap.add_argument('--parity'); ap.add_argument('--onnx'); a = ap.parse_args()
    model = load(a.ckpt, a.canvas)
    p1 = convert(model, a.canvas, 1, f'{a.out}-b1.mlpackage'); p16 = convert(model, a.canvas, 16, f'{a.out}-b16.mlpackage')
    for units, name in ((ct.ComputeUnit.ALL, 'ALL'), (ct.ComputeUnit.CPU_AND_NE, 'CPU+ANE'), (ct.ComputeUnit.CPU_AND_GPU, 'CPU+GPU'), (ct.ComputeUnit.CPU_ONLY, 'CPU')):
        print(json.dumps(dict(model=a.out, units=name, canvas=a.canvas, batch1=bench(p1, a.canvas, 1, units), batch16=bench(p16, a.canvas, 16, units, 20))), flush=True)
    if a.parity:
        import onnxruntime as ort
        from rotlab.core import circ_err, decode
        z = np.load(a.parity); x = z['x'].astype(np.float32); th = z['t']
        ml = ct.models.MLModel(p1, compute_units=ct.ComputeUnit.ALL)
        pc = decode(np.concatenate([ml.predict({'image': x[i:i + 1]})['prob'] for i in range(len(x))]))
        s = ort.InferenceSession(a.onnx, providers=['CPUExecutionProvider'])
        po = decode(np.concatenate([s.run(None, {'image': x[i:i + 16]})[0] for i in range(0, len(x), 16)]))
        print(json.dumps(dict(parity_n=len(x), coreml_w10=float((circ_err(pc, th) <= 10).mean()), onnx_fp32_w10=float((circ_err(po, th) <= 10).mean()),
                              diff_gt2deg=float((circ_err(pc, po) > 2).mean()))), flush=True)


if __name__ == '__main__':
    main()
