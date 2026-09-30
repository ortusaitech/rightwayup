#!/usr/bin/env python3
"""Campaign 2026-09-27: paired endpoint comparison of two prediction tags (pre-registered H8/H9/H10 endpoints).

Reads both tags through camp_store.load (receipt binds exact NPZ bytes); every compared set must have identical view
hashes and theta arrays in both receipts (and identical source ids where stored). Pooled endpoint = unweighted within-10
hits over the four new-val sets (COCO fixed, COCO fixed_deg, OI fixed, OI fixed_deg); diagonal stratum = views with
|theta mod 90 - 45| <= 15; paired 95% CI of (B - A) by bootstrap over SOURCE parents (all views of a parent together).
Other panels (dev, R-LiViT, max-area diagnostic) are reported per panel with the same metrics; view index is the unit
where no source ids exist (reused dev caches).
  python -m rotlab.camp_paired DIR TAG_A TAG_B [--json OUT]
"""
import json, sys
from pathlib import Path

import numpy as np

from rotlab.camp_store import load

POOLED = ['newcoco_val_fixed', 'newcoco_val_fixed_deg', 'newoi_val_fixed', 'newoi_val_fixed_deg']
OTHER = ['newcoco_val_maxarea', 'origin', 'fair', 'oi', 'common', 'fresh_diode', 'meva', 'fair_deg', 'oi_deg', 'rlivit_dev_rgb', 'rlivit_dev_thermal']


def err(p, t):
    e = np.abs((np.asarray(p) - np.asarray(t)) % 360); return np.minimum(e, 360 - e)


def met(e):
    return dict(n=int(len(e)), w10=int((e <= 10).sum()), pct=round(100 * float((e <= 10).mean()), 2), mae=round(float(e.mean()), 3),
                median=round(float(np.median(e)), 3), p95=round(float(np.percentile(e, 95)), 2), ge90=int((e >= 90).sum()), ge150=int((e >= 150).sum()))


def boot(diff, clusters, n=5000, seed=0):
    u, inv = np.unique(clusters, return_inverse=True); per = np.bincount(inv, weights=diff, minlength=len(u))
    b = per[np.random.default_rng(seed).integers(0, len(u), (n, len(u)))].sum(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    d, ta, tb = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    za, ra = load(d, ta); zb, rb = load(d, tb)
    out = dict(A=ta, B=tb, A_ckpt=ra.get('ckpt_sha256'), B_ckpt=rb.get('ckpt_sha256'), sets={})
    pool_a, pool_b, pool_src, diag = [], [], [], []
    for s in POOLED + OTHER:
        if f'{s}_pred' not in za or f'{s}_pred' not in zb:
            if s in POOLED:
                raise SystemExit(f'pooled set {s} missing from a tag')
            continue
        if ra['sets'][s] != rb['sets'][s] or not np.array_equal(za[f'{s}_theta'], zb[f'{s}_theta']):
            raise SystemExit(f'{s}: views/angles differ between tags')
        th = za[f'{s}_theta']; pa, pb = za[f'{s}_pred'], zb[f'{s}_pred']
        if not (len(pa) == len(pb) == len(th)) or not (np.isfinite(pa).all() and np.isfinite(pb).all() and np.isfinite(th).all()):
            raise SystemExit(f'{s}: length mismatch or non-finite predictions/angles')
        ea, eb = err(pa, th), err(pb, th)
        ids_a, ids_b = za.get(f'{s}_ids'), zb.get(f'{s}_ids')
        if s in POOLED:   # fail closed: source parents are REQUIRED to keep fixed/degraded views of a photo together
            if ids_a is None or ids_b is None or len(ids_a) != len(th) or not np.array_equal(ids_a, ids_b):
                raise SystemExit(f'{s}: source ids missing, wrong length or differently ordered in a tag')
            cl = np.asarray(ids_a)
        elif ids_a is not None and ids_b is not None and np.array_equal(ids_a, ids_b) and len(ids_a) == len(th):
            cl = np.asarray(ids_a)
        else:              # reused dev caches without ids: per-view units (stated limitation)
            cl = np.array([f'{s}:{i}' for i in range(len(th))])
        dg = np.abs(th % 90 - 45) <= 15
        dd = (eb <= 10).astype(float) - (ea <= 10).astype(float)
        out['sets'][s] = dict(A=met(ea), B=met(eb), diff_w10=int(dd.sum()), ci95=boot(dd, cl), cluster_unit='source parent' if not str(cl[0]).startswith(f'{s}:') else 'view (no ids; limitation)', diag_A=int((ea[dg] <= 10).sum()),
                              diag_B=int((eb[dg] <= 10).sum()), diag_n=int(dg.sum()))
        if s in POOLED:
            pool_a.append(ea); pool_b.append(eb); pool_src.append(cl); diag.append(dg)
    ea, eb, cl, dg = map(np.concatenate, (pool_a, pool_b, pool_src, diag))
    dd = (eb <= 10).astype(float) - (ea <= 10).astype(float)
    ci = boot(dd, cl)
    out['pooled'] = dict(estimand='micro-average within-10 over all views of the four new-val sets (each photo contributes fixed + degraded)',
                         A=met(ea), B=met(eb), diff_w10=int(dd.sum()), diff_pt=round(100 * float(dd.mean()), 3), ci95_views=ci,
                         ci95_pt=[round(100 * x / len(dd), 3) for x in ci], note='bootstrap conditional on the selected checkpoints',
                         sources=int(len(np.unique(cl))), diag_n=int(dg.sum()), diag_A=int((ea[dg] <= 10).sum()), diag_B=int((eb[dg] <= 10).sum()),
                         diag_ci95=boot(dd[dg], cl[dg]))
    txt = json.dumps(out, indent=1)
    if '--json' in sys.argv:
        Path(sys.argv[sys.argv.index('--json') + 1]).write_text(txt)
    print(txt)


if __name__ == '__main__':
    main()
