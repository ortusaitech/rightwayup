#!/usr/bin/env python3
"""Campaign 2026-09-27 offline analysis of camp_eval npz files (full 360-bin probabilities).

  python camp_analyze.py DIR table [TAG ...]          # metrics per set (decoder = stored)
  python camp_analyze.py DIR decoders TAG              # decoder ablation (argmax+local mean vs max window mass)
  python camp_analyze.py DIR cascade SMALL LARGE [--thr 0.836]   # frozen-threshold cascade, paired vs Max baseline
"""
import json, sys
from math import comb
from pathlib import Path

import numpy as np

SETS = ['origin', 'fair', 'oi', 'common', 'fresh_diode', 'meva', 'fair_deg', 'oi_deg',
        'newcoco_val_fixed', 'newcoco_val_maxarea', 'newcoco_val_fixed_deg', 'newoi_val_fixed', 'newoi_val_fixed_deg', 'rlivit_dev_rgb', 'rlivit_dev_thermal']


def circ_err(a, b):
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360); return np.minimum(d, 360 - d)


def dec_local(prob, w=10):
    k = prob.argmax(1); idx = (k[:, None] + np.arange(-w, w + 1)[None]) % 360
    ww = np.take_along_axis(prob, idx, 1); off = (ww * np.arange(-w, w + 1)[None]).sum(1) / np.maximum(ww.sum(1), 1e-12)
    return (k + off) % 360


def dec_maxwin(prob, w=10):
    """Centre of the +-w-bin circular window with maximum probability mass, refined by the window's circular mean."""
    c = np.cumsum(np.concatenate([prob, prob, prob], 1), 1)
    s = c[:, 360 + w:720 + w] - c[:, 360 - w - 1:720 - w - 1]     # window sums centred on bins 0..359
    k = s.argmax(1); idx = (k[:, None] + np.arange(-w, w + 1)[None]) % 360
    ww = np.take_along_axis(prob, idx, 1); off = (ww * np.arange(-w, w + 1)[None]).sum(1) / np.maximum(ww.sum(1), 1e-12)
    return (k + off) % 360


def conf(prob, pred, w=10):
    k = np.round(pred).astype(int) % 360; idx = (k[:, None] + np.arange(-w, w + 1)[None]) % 360
    return np.take_along_axis(prob, idx, 1).sum(1)


def met(e):
    return dict(n=len(e), w10=int((e <= 10).sum()), pct=round(100 * float((e <= 10).mean()), 2), mae=round(float(e.mean()), 2),
                med=round(float(np.median(e)), 2), p95=round(float(np.percentile(e, 95)), 1), t90=int((e > 90).sum()), t150=int((e >= 150).sum()))


def mcnemar(a, b):
    x = int((a & ~b).sum()); y = int((~a & b).sum()); n = x + y
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n)
    return x, y, p


def load(d, tag):
    """Verified read (camp_store: receipt binds the exact NPZ bytes returned)."""
    import sys as _s; _s.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rotlab.camp_store import load as sload
    rows, rec = sload(d, tag)
    if rec is None:
        raise SystemExit(f'missing {tag}')
    rows['__receipt__'] = rec
    return rows


def table(d, tags):
    print('| tag | ' + ' | '.join(SETS) + ' |'); print('|---' * (len(SETS) + 1) + '|')
    for t in tags:
        z = load(d, t); cells = []
        for s in SETS:
            cells.append(str(int((circ_err(z[f'{s}_pred'], z[f'{s}_theta']) <= 10).sum())) if f'{s}_pred' in z else '–')
        print(f'| {t} | ' + ' | '.join(cells) + ' |')


def decoders(d, tag):
    z = load(d, tag)
    for s in SETS:
        if f'{s}_prob' not in z:
            continue
        p = z[f'{s}_prob'].astype(np.float64); th = z[f'{s}_theta']
        base = circ_err(dec_local(p), th)
        row = dict(set=s, local=met(base))
        for w in (5, 10, 15):
            e = circ_err(dec_maxwin(p, w), th); x, y, pv = mcnemar(e <= 10, base <= 10)
            row[f'maxwin{w}'] = dict(**met(e), gain=x, loss=y, p=round(pv, 3))
        print(json.dumps(row))


def cascade(d, small, large, thr):
    zs, zl = load(d, small), load(d, large)
    base_l = load(d, 'soup-G7-G3-a0.5-c224') if (Path(d) / 'soup-G7-G3-a0.5-c224.npz').exists() else None
    for s in SETS:
        if f'{s}_pred' not in zs or f'{s}_pred' not in zl:
            continue
        for z in [zl] + ([base_l] if base_l is not None and f'{s}_pred' in base_l else []):   # alignment: same views, same angles
            if z['__receipt__']['sets'][s] != zs['__receipt__']['sets'][s] or not np.array_equal(z[f'{s}_theta'], zs[f'{s}_theta']):
                raise SystemExit(f'{s}: view hash or theta mismatch between receipts')
        th = zs[f'{s}_theta']; r = zs[f'{s}_conf'] < thr
        p = np.where(r, zl[f'{s}_pred'], zs[f'{s}_pred']); ok = circ_err(p, th) <= 10
        out = dict(set=s, route=round(float(r.mean()), 3), **met(circ_err(p, th)))
        if base_l is not None and f'{s}_pred' in base_l:
            pb = np.where(r, base_l[f'{s}_pred'], zs[f'{s}_pred']); okb = circ_err(pb, th) <= 10
            x, y, pv = mcnemar(ok, okb); out.update(max_baseline=int(okb.sum()), gain=x, loss=y, p=round(pv, 3))
        print(json.dumps(out))


if __name__ == '__main__':
    d, cmd = sys.argv[1], sys.argv[2]
    if cmd == 'table':
        table(d, sys.argv[3:])
    elif cmd == 'decoders':
        decoders(d, sys.argv[3])
    elif cmd == 'cascade':
        thr = float(sys.argv[sys.argv.index('--thr') + 1]) if '--thr' in sys.argv else 0.836
        cascade(d, sys.argv[3], sys.argv[4], thr)
