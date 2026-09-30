"""ONNX export parity: FP32 / INT8 ONNX (CPU) vs the stored PyTorch camp_eval predictions of the same checkpoint and canvas.
  python -m rotlab.onnx_parity MODEL.onnx TAG CANVAS --sets origin,fair [--threads 32]"""
import argparse, io, pickle, time
import numpy as np
import onnxruntime as ort
from PIL import Image
from rotlab.core import DATA, circ_err, decode, letterbox
from rotlab.camp_eval import view_file
from rotlab.camp_store import load

ap = argparse.ArgumentParser(); ap.add_argument('onnx'); ap.add_argument('tag'); ap.add_argument('canvas', type=int)
ap.add_argument('--sets', default='origin,fair'); ap.add_argument('--threads', type=int, default=32)
a = ap.parse_args()
so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
s = ort.InferenceSession(a.onnx, so, providers=['CPUExecutionProvider']); inp = s.get_inputs()[0].name
rows, _ = load(DATA / 'rotlab/camp-eval', a.tag)
for n in a.sets.split(','):
    d = pickle.loads(view_file(n).read_bytes()); th = np.asarray(d['theta'], np.float64)
    t0 = time.time(); pr = []
    for b in d['png']:
        x = letterbox(Image.open(io.BytesIO(b)).convert('RGB'), a.canvas)[None]
        pr.append(s.run(None, {inp: x})[0][0])
    pr = np.stack(pr); p = decode(pr); ref = decode(rows[f'{n}_prob'].astype(np.float32))
    e, er = circ_err(p, th), circ_err(ref, th); dp = circ_err(p, ref)
    print(f'{a.onnx.split("/")[-1]} {n}: onnx w10 {(e <= 10).sum()} vs torch {(er <= 10).sum()} of {len(th)}; '
          f'flips {int(((e <= 10) != (er <= 10)).sum())}; median |pred diff| {np.median(dp):.3f} p99 {np.percentile(dp, 99):.2f}; '
          f'{time.time() - t0:.0f}s', flush=True)
