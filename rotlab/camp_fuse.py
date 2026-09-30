#!/usr/bin/env python3
"""H10 pre-registered SECONDARY pipeline: no-fit probability fusion p = 0.5 * (p_A + p_B) of two absolute-frame tags
(same views, same 224 letterbox), canonical decoder/confidence. No weights/thresholds are searched.
Inputs are read via camp_store.load (receipt binds exact NPZ bytes); both must be absolute-frame <set>_prob (residual-frame
H9 outputs are rejected), share view hashes, theta arrays and source ids. Output tag receipt binds both input NPZ hashes.
  python -m rotlab.camp_fuse DIR TAG_A TAG_B OUT_TAG [--sets a,b,...]
"""
import sys
from pathlib import Path

import numpy as np

from rotlab.camp_store import commit, load
from rotlab.core import decode
from rotlab.evaluate import confidence


def fuse(d, ta, tb, out, sets=None):
    from rotlab.camp_eval import DECODER
    za, ra = load(d, ta); zb, rb = load(d, tb)
    for r in (ra, rb):
        if 'h9_recipe' in r or r.get('decoder') != DECODER:
            raise SystemExit('fusion inputs must be absolute-frame outputs scored with the canonical decoder')
    names = sets or sorted(set(ra['sets']) & set(rb['sets']))
    rows, vs = {}, {}
    for s in names:
        if ra['sets'][s] != rb['sets'][s] or not np.array_equal(za[f'{s}_theta'], zb[f'{s}_theta']):
            raise SystemExit(f'{s}: views/angles differ')
        if (f'{s}_ids' in za) != (f'{s}_ids' in zb) or (f'{s}_ids' in za and not np.array_equal(za[f'{s}_ids'], zb[f'{s}_ids'])):
            raise SystemExit(f'{s}: source ids differ')
        p = 0.5 * (za[f'{s}_prob'].astype(np.float64) + zb[f'{s}_prob'].astype(np.float64))
        if not np.isfinite(p).all():
            raise SystemExit(f'{s}: non-finite probabilities')
        pred = decode(p)
        rows.update({f'{s}_prob': p.astype(np.float16), f'{s}_pred': pred, f'{s}_conf': confidence(p, pred), f'{s}_theta': za[f'{s}_theta']})
        if f'{s}_ids' in za:
            rows[f'{s}_ids'] = za[f'{s}_ids']
        vs[s] = ra['sets'][s]
    ident = dict(ckpt_sha256=f'fusion:{ra["ckpt_sha256"]}+{rb["ckpt_sha256"]}', pool=None, canvas=ra['canvas'], decoder=DECODER,
                 fusion=dict(rule='0.5*(p_A+p_B), canonical decode', A=ta, B=tb, A_npz_sha256=ra['npz_sha256'], B_npz_sha256=rb['npz_sha256']))
    if ra['canvas'] != rb['canvas']:
        raise SystemExit('inputs must share the input canvas')
    return commit(d, out, rows, ident, vs)


if __name__ == '__main__':
    a = sys.argv[1:]
    sets = a[a.index('--sets') + 1].split(',') if '--sets' in a else None
    fuse(Path(a[0]), a[1], a[2], a[3], sets); print('fused', a[3])
