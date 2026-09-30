"""Per-format calibration predictions (lead, 29 Sep; FREEZE v2): run a SHIPPED ONNX file on the calibration views
(suite/views-cal-c<canvas>.npz, the same letterboxed tensors rotlab.cal_fill / suite --cal use) and write
suite/<NAME>-cal-c<canvas>.npz in rotlab.suite's format (<set>_p_pred, <set>_p_conf, <set>_theta), so the release
Tier.calibrate fits thresholds for that exact file and runtime. Calibration sets only; never final sets.
decode / confidence are verbatim numpy copies of rotlab.core.decode / rotlab.evaluate.confidence (this runs in an
onnxruntime-gpu environment without torch). --ref compares with an existing store (FP32 sanity check).
  python cal_onnx.py MODEL.onnx CANVAS NAME cpu|cuda [--threads 16] [--batch 16] [--ref STORE_STEM]
"""
import argparse, time
from pathlib import Path

import numpy as np
import onnxruntime as ort

SUITE = Path('/workspace/rotation-data/rotlab/suite')
CALS = ['meva_cal', 'poly_cal', 'oi_cal']


def decode(prob):
    k = prob.argmax(1)
    idx = (k[:, None] + np.arange(-10, 11)[None]) % 360
    w = np.take_along_axis(prob, idx, 1)
    off = (w * np.arange(-10, 11)[None]).sum(1) / np.maximum(w.sum(1), 1e-12)
    return (k + off) % 360


def confidence(prob, pred, width=10):
    k = np.arange(360)[None]
    d = np.abs((k - pred[:, None] + 180) % 360 - 180)
    return (prob * (d <= width)).sum(1)


def within10(p, t):
    e = np.abs((p - t) % 360); return int((np.minimum(e, 360 - e) <= 10).sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('onnx'); ap.add_argument('canvas', type=int); ap.add_argument('name')
    ap.add_argument('provider', choices=['cpu', 'cuda']); ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--batch', type=int, default=1); ap.add_argument('--ref')
    a = ap.parse_args()
    so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
    prov = ['CUDAExecutionProvider'] if a.provider == 'cuda' else ['CPUExecutionProvider']
    s = ort.InferenceSession(a.onnx, so, providers=prov); inp = s.get_inputs()[0]
    assert s.get_providers()[0] == prov[0], f'{prov[0]} not active'
    bs = a.batch if not isinstance(inp.shape[0], int) else 1
    z = np.load(SUITE / f'views-cal-c{a.canvas}.npz'); rows = {}; t0 = time.time()
    for c in CALS:
        x, th = z[f'{c}_x'], z[f'{c}_t']
        pr = np.concatenate([s.run(None, {inp.name: np.ascontiguousarray(x[i:i + bs])})[0] for i in range(0, len(x), bs)]).astype(np.float64)
        pr = pr / pr.sum(1, keepdims=True)
        pd = decode(pr)
        rows[f'{c}_p_pred'], rows[f'{c}_p_conf'], rows[f'{c}_theta'] = pd, confidence(pr, pd), th
        msg = f'{a.name} c{a.canvas} {c}: w10 {within10(pd, th)}/{len(th)}'
        if a.ref:
            r = np.load(SUITE / f'{a.ref}-cal-c{a.canvas}.npz')
            msg += (f' | ref w10 {within10(r[f"{c}_p_pred"], th)}; |dpred|>0.5deg {int((np.abs((pd - r[f"{c}_p_pred"] + 180) % 360 - 180) > 0.5).sum())};'
                    f' median |dconf| {np.median(np.abs(rows[f"{c}_p_conf"] - r[f"{c}_p_conf"])):.4f}, mean dconf {np.mean(rows[f"{c}_p_conf"] - r[f"{c}_p_conf"]):+.4f}')
        print(msg, flush=True)
    out = SUITE / f'{a.name}-cal-c{a.canvas}.npz'
    if out.exists():
        raise SystemExit(f'refusing to overwrite {out}')
    np.savez(out, **rows)
    print(f'{a.name} c{a.canvas} done {time.time() - t0:.0f}s -> {out.name}', flush=True)


if __name__ == '__main__':
    main()
