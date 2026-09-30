#!/usr/bin/env python3
"""Two-stage cascade: fast model always; strong model only when fast confidence < threshold.
Reports accuracy vs fraction routed, on the pack (eval jsons) and the fair photo set (npz rows).
  python -m rotlab.cascade FAST_EVAL_JSON STRONG_EVAL_JSON [--fair FAST_NPZ STRONG_NPZ --fast-key model_c4 --strong-key model]"""
import argparse, json
import numpy as np
from rotlab.core import circ_err, load_pack

ap = argparse.ArgumentParser(); ap.add_argument('fast'); ap.add_argument('strong'); ap.add_argument('--fair', nargs=2)
ap.add_argument('--fast-key', default='model_tta'); ap.add_argument('--strong-key', default='model')
a = ap.parse_args()
rows = load_pack(); tgt = np.array([r['target_degrees'] for r in rows]); pan = np.array([r['panel'] for r in rows])
f = json.load(open(a.fast)); s = json.load(open(a.strong))
fp, fc, sp = np.array(f['predictions']), np.array(f['confidence']), np.array(s['predictions'])
sets = [(p, pan == p, tgt, fp, fc, sp) for p in ('common', 'fresh_diode', 'origin')]
if a.fair:
    F, S = np.load(a.fair[0]), np.load(a.fair[1])
    sets.append(('fair_photo', np.ones(len(F['theta']), bool), F['theta'], F[a.fast_key + '_pred'], F[a.fast_key + '_conf'], S[a.strong_key + '_pred']))
allc = np.concatenate([st[4][st[1]] for st in sets])
for route in (0.0, 0.1, 0.2, 0.3, 1.0):
    thr = np.quantile(allc, route) if 0 < route < 1 else (-1 if route == 0 else 2)
    line = []
    for name, m, t, p1, c1, p2 in sets:
        use2 = c1[m] < thr; pred = np.where(use2, p2[m], p1[m]); e = circ_err(pred, t[m])
        line.append(f'{name} {int((e <= 10).sum())}/{m.sum()} t150 {int((e >= 150).sum())} routed {use2.mean():.0%}')
    print(f'route~{route:.0%}: ' + ' | '.join(line))
