"""Criterion 1 on a SHIPPED ONNX graph (lead, 28 Sep): argmax rho4 and distribution-level (soft) rho4 on the content-free
probe sets, computed from the ONNX CPU outputs (not from the PyTorch store).
  python -m rotlab.onnx_probe MODEL.onnx CANVAS [--sets trace_probe_G75,trace_probe_P75,trace_probe_PPNG] [--threads 16]"""
import argparse, io, pickle
import numpy as np
import onnxruntime as ort
from PIL import Image
from rotlab.core import decode, letterbox
from rotlab.camp_eval import view_file

ap = argparse.ArgumentParser(); ap.add_argument('onnx'); ap.add_argument('canvas', type=int)
ap.add_argument('--sets', default='trace_probe_G75,trace_probe_P75,trace_probe_PPNG'); ap.add_argument('--threads', type=int, default=16)
a = ap.parse_args()
so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
s = ort.InferenceSession(a.onnx, so, providers=['CPUExecutionProvider']); inp = s.get_inputs()[0].name
worst = -9.0
for n in a.sets.split(','):
    d = pickle.loads(view_file(n).read_bytes()); th = np.asarray(d['theta'], np.float64)
    pr = np.stack([s.run(None, {inp: letterbox(Image.open(io.BytesIO(b)).convert('RGB'), a.canvas)[None]})[0][0] for b in d['png']])
    p = pr.astype(np.float64); p = p / p.sum(1, keepdims=True) if (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-3) else np.exp(p - p.max(1, keepdims=True)) / np.exp(p - p.max(1, keepdims=True)).sum(1, keepdims=True)
    pred = decode(p)   # on probabilities (the graph may emit probabilities or logits)
    r4 = float(np.mean(np.cos(np.radians(4 * (pred - th)))))
    ang = np.arange(p.shape[1]) * 360.0 / p.shape[1]
    soft = float((p * np.cos(np.radians(4 * (ang[None] - th[:, None])))).sum(1).mean())
    worst = max(worst, r4)
    print(f'{a.onnx.split("/")[-1]} @{a.canvas} {n}: argmax rho4 {r4:+.3f}  soft {soft:+.3f}  top-bin {p.max(1).mean():.4f}', flush=True)
print(f'{a.onnx.split("/")[-1]} @{a.canvas} C1 {"PASS" if worst <= 0.20 else "FAIL"} (max argmax rho4 {worst:+.3f})')
