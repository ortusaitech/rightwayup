#!/usr/bin/env python3
"""Full evaluation suite for a checkpoint at one or more input resolutions.

Sets: benchmark pack (common / fresh_diode / origin), fair photo (1,030, fixed crop), Open Images photo test
(fixed crop), MEVA dev (7 cameras). Per set and canvas it stores decoded predictions and confidence (plain and
180-degree TTA) in DATA/rotlab/suite/<run>-c<canvas>.npz, so cascades and risk-coverage can be computed offline.
  python -m rotlab.suite CKPT --canvases 140,168,196,224
  python -m rotlab.suite --table RUN [RUN ...]       # print the stored results
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from rotlab.core import DATA, circ_err, decode, letterbox
from rotlab.evaluate import PackCache, confidence, predict
from rotlab.model import RotNet

OUT = DATA / 'rotlab/suite'
PANELS = ('common', 'fresh_diode', 'origin')


def sets(canvas, cal=False):
    """name -> (x, theta) at this canvas. Views are deterministic (seeded) and cached per canvas.
    cal=True returns the calibration sets built by rotlab.calib (thresholds only)."""
    kind = 'cal' if cal is True else (cal or '')
    f = OUT / (f'views-{kind}-c{canvas}.npz' if kind else f'views-c{canvas}.npz')
    if kind == 'deg' and not f.exists():
        from rotlab.degraded_sets import build
        build(canvas)
    if cal:
        z = np.load(f); return {k[:-2]: (z[k], z[k[:-2] + '_t']) for k in z.files if k.endswith('_x')}
    if f.exists():
        z = np.load(f); return {k[:-2]: (z[k], z[k[:-2] + '_t']) for k in z.files if k.endswith('_x')}
    from rotlab.fair_photo import fair_views
    from rotlab.meva_test import views as meva_views
    from rotlab.oi_test import views as oi_views
    pc = PackCache(canvas); pan = np.array([r['panel'] for r in pc.rows]); out = {}
    for p in PANELS:
        out[p] = (np.asarray(pc.x)[pan == p], pc.target[pan == p])
    _, fv, ft = fair_views(7); out['fair'] = (np.stack([letterbox(v, canvas) for v in fv]), ft)
    ov, ot = oi_views('fixed'); out['oi'] = (np.stack([letterbox(v, canvas) for v in ov]), ot)
    mv = meva_views('dev'); out['meva'] = (np.stack([letterbox(v['view'], canvas) for v in mv]), np.array([v['theta'] for v in mv]))
    np.savez(f, **{f'{k}_x': x for k, (x, _) in out.items()}, **{f'{k}_t': t for k, (_, t) in out.items()})
    return out


def run(ckpt, canvases, cal=False):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False); cfg = ck['config']
    m = RotNet(img_size=cfg['img_size'], pretrained=False, dynamic=True, arch=cfg.get('arch', 's')).cuda().eval()
    m.load_state_dict(ck['ema']); name = Path(ckpt).parent.name
    for cv in canvases:
        rows = {}
        for s, (x, th) in sets(cv, cal).items():
            for tta, key in ((False, 'p'), (True, 'tta')):
                prob = predict(m, x, tta180=tta); p = decode(prob)
                rows[f'{s}_{key}_pred'] = p; rows[f'{s}_{key}_conf'] = confidence(prob, p)
            rows[f'{s}_theta'] = th
        np.savez(OUT / f'{name}-{(cal if isinstance(cal, str) else "cal") + "-" if cal else ""}c{cv}.npz', **rows)
        print(name, 'cal' if cal else '', cv, {s: int((circ_err(rows[f'{s}_p_pred'], rows[f'{s}_theta']) <= 10).sum()) for s in sets(cv, cal)}, flush=True)


def table(runs):
    names = ['common', 'fresh_diode', 'origin', 'fair', 'oi', 'meva']
    print('| run @ canvas | ' + ' | '.join(names) + ' |'); print('|---' * (len(names) + 1) + '|')
    for r in runs:
        for f in sorted(OUT.glob(f'{r}-c[0-9]*.npz'), key=lambda p: int(p.stem.split('-c')[-1])):
            z = np.load(f); cells = []
            for s in names:
                e = circ_err(z[f'{s}_p_pred'], z[f'{s}_theta']); et = circ_err(z[f'{s}_tta_pred'], z[f'{s}_theta'])
                cells.append(f'{(e <= 10).sum()} ({(e <= 10).mean() * 100:.1f}%) / tta {(et <= 10).sum()}')
            print(f'| {f.stem} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('items', nargs='+'); ap.add_argument('--canvases', default='224')
    ap.add_argument('--table', action='store_true'); ap.add_argument('--cal', action='store_true'); ap.add_argument('--deg', action='store_true', help='degraded photo sets (fair_deg, oi_deg)')
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if a.table:
        table(a.items)
    else:
        for c in a.items:
            run(c, [int(v) for v in a.canvases.split(',')], 'deg' if a.deg else a.cal)
