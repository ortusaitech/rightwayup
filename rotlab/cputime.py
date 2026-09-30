#!/usr/bin/env python3
"""Same-host CPU latency of ONNX/ORT graphs (batch 1, prepared tensor), ORT spinning disabled.
Models are timed INTERLEAVED (round-robin) so background load affects all equally.
  python cputime.py a.onnx b.onnx ... --threads 1,2,4 --runs 40
"""
import argparse, json, time
import numpy as np, onnxruntime as ort

ap = argparse.ArgumentParser(); ap.add_argument('models', nargs='+'); ap.add_argument('--threads', default='1,2,4'); ap.add_argument('--runs', type=int, default=40)
a = ap.parse_args()
for t in map(int, a.threads.split(',')):
    sess = []
    for m in a.models:
        so = ort.SessionOptions(); so.intra_op_num_threads = t; so.inter_op_num_threads = 1
        so.add_session_config_entry('session.intra_op.allow_spinning', '0')
        s = ort.InferenceSession(m, so, providers=['CPUExecutionProvider'])
        i = s.get_inputs()[0]; shape = [1 if not isinstance(d, int) else d for d in i.shape]
        x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
        for _ in range(3): s.run(None, {i.name: x})
        sess.append((m.split('/')[-1], s, i.name, x, shape, []))
    for _ in range(a.runs):
        for name, s, n, x, shape, ts in sess:
            t0 = time.perf_counter(); s.run(None, {n: x}); ts.append((time.perf_counter() - t0) * 1000)
    for name, s, n, x, shape, ts in sess:
        ts = np.array(ts)
        print(json.dumps(dict(model=name, threads=t, input=shape, mean_ms=round(ts.mean(), 1), p50_ms=round(np.percentile(ts, 50), 1), p95_ms=round(np.percentile(ts, 95), 1))), flush=True)
