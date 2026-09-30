#!/usr/bin/env python3
"""Risk vs coverage when accepting only confident predictions (single frame).
  python -m rotlab.selective EVAL_JSON"""
import json, sys
import numpy as np
from rotlab.core import circ_err, load_pack

rows = load_pack(); d = json.load(open(sys.argv[1]))
pred, conf = np.array(d['predictions']), np.array(d['confidence'])
tgt = np.array([r['target_degrees'] for r in rows]); err = circ_err(pred, tgt)
pan = np.array([r['panel'] for r in rows])
for p in ('common', 'fresh_diode', 'origin'):
    m = pan == p; e, c = err[m], conf[m]
    print(f'== {p} (n={m.sum()})')
    for cov in (1.0, 0.95, 0.9, 0.8, 0.7):
        thr = np.quantile(c, 1 - cov) if cov < 1 else -1
        acc = c > thr
        print(f'  coverage {acc.mean():.2f} thr {thr:.3f}: wrong>10 {int((e[acc] > 10).sum())} ({100 * (e[acc] > 10).mean():.1f}%)  >90 {int((e[acc] > 90).sum())}  >=150 {int((e[acc] >= 150).sum())}')
