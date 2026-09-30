#!/usr/bin/env python3
"""Cascade and risk-coverage analysis from stored suite predictions (DATA/rotlab/suite/<run>-c<canvas>.npz).

  python -m rotlab.tiers cascade FAST_STEM STRONG_STEM [--route 0.1,0.2,0.3] [--tta-strong]
  python -m rotlab.tiers coverage STEM [--tta]

Routing thresholds here are pooled quantiles over all test sets: PROVISIONAL (test-derived). Release thresholds
must be fixed on calibration data only (MEVA G419/G299/G330 + Poly Haven calibration).
"""
import argparse
import numpy as np

from rotlab.core import DATA, circ_err

SUITE = DATA / 'rotlab/suite'
SETS = ('common', 'fresh_diode', 'origin', 'fair', 'oi', 'meva')


def load(stem):
    return np.load(SUITE / f'{stem}.npz')


def cell(e):
    return f'{(e <= 10).sum()} ({(e <= 10).mean() * 100:.1f}%)'


def cascade(a):
    f, s = load(a.fast), load(a.strong); sk = 'tta' if a.tta_strong else 'p'
    pooled = np.concatenate([f[f'{x}_p_conf'] for x in SETS])
    print(f'| routed (pooled) | ' + ' | '.join(SETS) + ' |'); print('|---' * (len(SETS) + 1) + '|')
    for q in [0.0] + [float(v) for v in a.route.split(',')] + [1.0]:
        t = np.quantile(pooled, q) if 0 < q < 1 else (-1 if q == 0 else 2)
        cells = []
        for x in SETS:
            r = f[f'{x}_p_conf'] < t
            p = np.where(r, s[f'{x}_{sk}_pred'], f[f'{x}_p_pred'])
            cells.append(f'{cell(circ_err(p, f[f"{x}_theta"]))} r{r.mean() * 100:.0f}%')
        print(f'| {q * 100:.0f}% | ' + ' | '.join(cells) + ' |')


def coverage(a):
    z = load(a.stem); k = 'tta' if a.tta else 'p'
    print('| coverage | ' + ' | '.join(f'{x} acc / >=150' for x in SETS) + ' |'); print('|---' * (len(SETS) + 1) + '|')
    for cov in (1.0, 0.95, 0.9, 0.8, 0.7):
        cells = []
        for x in SETS:
            c = z[f'{x}_{k}_conf']; e = circ_err(z[f'{x}_{k}_pred'], z[f'{x}_theta'])
            keep = c >= np.quantile(c, 1 - cov) if cov < 1 else np.ones_like(c, bool)
            cells.append(f'{(e[keep] <= 10).mean() * 100:.1f}% / {(e[keep] >= 150).sum()}')
        print(f'| {cov * 100:.0f}% | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('stems', nargs='+')
    ap.add_argument('--route', default='0.1,0.2,0.3'); ap.add_argument('--tta-strong', action='store_true'); ap.add_argument('--tta', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'cascade':
        a.fast, a.strong = a.stems; cascade(a)
    else:
        a.stem = a.stems[0]; coverage(a)
