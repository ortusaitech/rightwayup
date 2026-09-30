"""Shipped-graph evaluation (lead, 28 Sep 17:25): an ONNX export on camp_eval view sets -> within-10, confidence (release
decoder: mass within +-10 deg), fraction ANSWERED at a tier threshold, within-10 of the answered, and for content-free probe
sets argmax / soft rho4 overall and among answered views. Saves DATA/rotlab/onnx-eval/<tag>/<set>.npz (prob as float16).
  python -m rotlab.onnx_eval MODEL.onnx CANVAS TAG --sets a,b --thr 0.765 [--threads 32]"""
import argparse, io, pickle
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image
from rotlab.core import DATA, circ_err, decode, letterbox
from rotlab.camp_eval import view_file

ap = argparse.ArgumentParser(); ap.add_argument('onnx'); ap.add_argument('canvas', type=int); ap.add_argument('tag')
ap.add_argument('--sets', required=True); ap.add_argument('--thr', type=float, required=True); ap.add_argument('--threads', type=int, default=32)
a = ap.parse_args()
so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
s = ort.InferenceSession(a.onnx, so, providers=['CPUExecutionProvider']); inp = s.get_inputs()[0].name
out = DATA / 'rotlab/onnx-eval' / a.tag; out.mkdir(parents=True, exist_ok=True)
for n in a.sets.split(','):
    d = pickle.loads(view_file(n).read_bytes()); th = np.asarray(d['theta'], np.float64)
    raw = np.stack([s.run(None, {inp: letterbox(Image.open(io.BytesIO(b)).convert('RGB'), a.canvas)[None]})[0][0] for b in d['png']]).astype(np.float64)
    p = raw / raw.sum(1, keepdims=True) if (raw >= 0).all() and np.allclose(raw.sum(1), 1, atol=1e-3) else \
        np.exp(raw - raw.max(1, keepdims=True)) / np.exp(raw - raw.max(1, keepdims=True)).sum(1, keepdims=True)
    pred = decode(p); k = p.argmax(1); idx = (k[:, None] + np.arange(-10, 11)[None]) % p.shape[1]
    conf = np.take_along_axis(p, idx, 1).sum(1); ans = conf >= a.thr; e = circ_err(pred, th)
    np.savez(out / f'{n}.npz', prob=p.astype(np.float16), pred=pred, conf=conf, theta=th, ids=np.asarray(d.get('ids', np.arange(len(th)))))
    line = (f'{a.tag} {n}: n {len(th)} w10 {100 * (e <= 10).mean():.2f}% | conf mean {conf.mean():.3f} max {conf.max():.3f} | '
            f'answered@{a.thr} {100 * ans.mean():.2f}% w10|answered {100 * (e[ans] <= 10).mean() if ans.any() else float("nan"):.2f}%')
    if n.startswith('trace_probe'):
        ang = np.arange(p.shape[1]) * 360.0 / p.shape[1]
        r4 = np.cos(np.radians(4 * (pred - th))); soft = (p * np.cos(np.radians(4 * (ang[None] - th[:, None])))).sum(1)
        line += f' | rho4 argmax {r4.mean():+.3f} soft {soft.mean():+.3f} | rho4 among answered {r4[ans].mean() if ans.any() else float("nan"):+.3f}'
    print(line, flush=True)
