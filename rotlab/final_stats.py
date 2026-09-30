#!/usr/bin/env python3
"""Tier statistics with calibrated routing/abstention, clean-vs-degraded breakdowns and the incumbent on identical views.

Thresholds come ONLY from the calibration sets (rotlab.calib, `<run>-cal-c<canvas>.npz`), each calibration set weighted
equally. They are then applied unchanged to the tests.
  python -m rotlab.final_stats --small S2-vits-multires --large soup-G7-G3-a0.5 > final-stats.md
"""
import argparse, json
from pathlib import Path

import numpy as np

from rotlab.core import DATA, circ_err, load_pack

S = DATA / 'rotlab/suite'
TESTS = ['common', 'fresh_diode', 'origin', 'fair', 'oi', 'meva']
DEG = ['fair_deg', 'oi_deg']
CALS = ['meva_cal', 'poly_cal', 'oi_cal']
LABEL = {'common': 'Common mixed', 'fresh_diode': 'Fresh DIODE', 'origin': 'Origin (leaky)', 'fair': 'Fair photos',
         'oi': 'Open Images photos', 'meva': 'MEVA unseen CCTV', 'fair_deg': 'Fair photos, degraded',
         'oi_deg': 'Open Images, degraded'}
GROUPS = {'blur / focus / resolution': {'defocus_blur', 'lens_smear', 'resolution_loss'},
          'lighting / exposure / weather': {'darkness', 'overexposure', 'glare', 'fog_low_contrast', 'color_cast'},
          'compression / noise / banding': {'jpeg_compression', 'macroblocking', 'noise_spike', 'horizontal_banding'},
          'occlusion': {'partial_occlusion'}}


def z(stem):
    return np.load(S / f'{stem}.npz')


def wquant(vals_by_set, q):
    """quantile over the union with every set weighted equally."""
    v = np.concatenate(vals_by_set); w = np.concatenate([np.full(len(x), 1 / len(x)) for x in vals_by_set])
    o = np.argsort(v); c = np.cumsum(w[o]) / w.sum()
    return float(v[o][np.searchsorted(c, q)])


class Tier:
    def __init__(self, name, small, large=None, route=0.0):
        self.name, self.small, self.large, self.route = name, small, large, route

    def pred(self, kind, s):
        a = z(f'{self.small}{kind}')
        p, c = a[f'{s}_p_pred'].copy(), a[f'{s}_p_conf'].copy()
        if self.large:
            b = z(f'{self.large}{kind}'); r = a[f'{s}_p_conf'] < self.t_route
            p[r], c[r] = b[f'{s}_p_pred'][r], b[f'{s}_p_conf'][r]
        return p, c, a[f'{s}_theta']

    def calibrate(self, small_cv, large_cv):
        self.kind_cal_small, self.kind_cal_large = small_cv, large_cv
        if self.large:
            self.t_route = wquant([z(f'{self.small}-cal-{small_cv}')[f'{s}_p_conf'] for s in CALS], self.route)
        cal = []
        for s in CALS:
            a = z(f'{self.small}-cal-{small_cv}'); p, c = a[f'{s}_p_pred'].copy(), a[f'{s}_p_conf'].copy()
            if self.large:
                b = z(f'{self.large}-cal-{large_cv}'); r = c < self.t_route; p[r], c[r] = b[f'{s}_p_pred'][r], b[f'{s}_p_conf'][r]
            cal.append((c, circ_err(p, a[f'{s}_theta']) > 10))
        self.t_cov90 = wquant([c for c, _ in cal], 0.10)
        # strict: lowest threshold whose answered error rate (sets weighted equally) is <= 1 %
        grid = np.linspace(0, 1, 401); self.t_strict = 1.0
        for t in grid:
            rates = [e[c >= t].mean() if (c >= t).any() else 0 for c, e in cal]
            if np.mean(rates) <= 0.01:
                self.t_strict = float(t); break


def pct(x): return f'{x * 100:.1f}%'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--small', required=True); ap.add_argument('--large', required=True)
    a = ap.parse_args()
    sm, lg = a.small, a.large
    tiers = [Tier('Nano (S @112)', f'{sm}-c112'.replace('-c112', ''), None), Tier('Fast (S @224)', sm, None),
             Tier('Balanced (S→L, 10%)', sm, lg, 0.10), Tier('Max (S→L, 20%)', sm, lg, 0.20)]
    cvs = [('c112', None), ('c224', None), ('c224', 'c224'), ('c224', 'c224')]
    for t, (scv, lcv) in zip(tiers, cvs):
        t.calibrate(scv, lcv or 'c224'); t.test_small_cv, t.test_large_cv = scv, lcv or 'c224'

    def tp(t, s, deg=False):
        kind_s = f'-{"deg-" if deg else ""}{t.test_small_cv}'
        a_ = z(f'{t.small}{kind_s}'); p, c = a_[f'{s}_p_pred'].copy(), a_[f'{s}_p_conf'].copy()
        if t.large:
            b = z(f'{t.large}-{"deg-" if deg else ""}{t.test_large_cv}'); r = c < t.t_route; p[r], c[r] = b[f'{s}_p_pred'][r], b[f'{s}_p_conf'][r]
            t.routed = getattr(t, 'routed', {}); t.routed[s] = r.mean()
        return p, c, a_[f'{s}_theta']

    # incumbent predictions on identical views
    rows = load_pack(); pan = np.array([r['panel'] for r in rows])
    inc = {p: np.array([r['faithful'] for r in rows])[pan == p] for p in ('common', 'fresh_diode', 'origin')}
    inc['fair'] = np.load(next((DATA / 'rotlab/fair-photo').glob('rows-seed7-*.npz')))['incumbent']
    inc['oi'] = np.load(DATA / 'rotlab/oi-test/incumbent-fixed.npz')['pred']
    inc['meva'] = np.load(DATA / 'rotlab/meva-test-v1/results-dev/incumbent.npz')['pred']
    di = np.load(S / 'incumbent-deg.npz'); inc.update({k: di[f'{k}_pred'] for k in DEG})

    print('# Tier statistics (provisional; thresholds fixed on calibration data only)\n')
    for t in tiers:
        extra = f', route threshold {t.t_route:.3f}' if t.large else ''
        print(f'- {t.name}: abstain thresholds 90%-coverage {t.t_cov90:.3f}, strict ≤1% error {t.t_strict:.3f}{extra}')
    print('\n## Accuracy within 10° (all images answered)\n')
    print('| Test (n) | Incumbent | ' + ' | '.join(t.name for t in tiers) + ' |'); print('|---' * (len(tiers) + 2) + '|')
    for s in TESTS + DEG:
        deg = s in DEG; th = None; cells = []
        for t in tiers:
            p, c, th = tp(t, s, deg); e = circ_err(p, th); cells.append(f'{pct((e <= 10).mean())} ({(e >= 150).sum()})')
        ie = circ_err(inc[s], th)
        print(f'| {LABEL[s]} ({len(th)}) | {pct((ie <= 10).mean())} ({(ie >= 150).sum()}) | ' + ' | '.join(cells) + ' |')
    print('\n(in brackets: upside-down errors ≥150°)\n')
    for key, name in (('t_cov90', 'standard (90% answered on calibration)'), ('t_strict', 'strict (≤1% wrong on calibration)')):
        print(f'\n## With abstention — {name}\n')
        print('| Test | ' + ' | '.join(f'{t.name} answered / correct / ≥150°' for t in tiers) + ' |'); print('|---' * (len(tiers) + 1) + '|')
        for s in TESTS + DEG:
            cells = []
            for t in tiers:
                p, c, th = tp(t, s, s in DEG); keep = c >= getattr(t, key); e = circ_err(p, th)
                cells.append(f'{pct(keep.mean())} / {pct((e[keep] <= 10).mean()) if keep.any() else "-"} / {(e[keep] >= 150).sum()}')
            print(f'| {LABEL[s]} | ' + ' | '.join(cells) + ' |')
    # clean vs degraded
    print('\n## Clean vs degraded (within 10°)\n')
    degmap = json.loads((S / 'pack-degradations.json').read_text())
    ids = np.array([r['id'] for r in rows])[pan == 'common']
    real = np.array([i in degmap for i in ids]); isdeg = np.array([bool(degmap.get(i)) for i in ids])
    print('| Slice (n) | Incumbent | ' + ' | '.join(t.name for t in tiers) + ' |'); print('|---' * (len(tiers) + 2) + '|')
    slices = [('MEVA CCTV, clean', 'meva', lambda n: np.arange(n) % 2 == 0), ('MEVA CCTV, degraded', 'meva', lambda n: np.arange(n) % 2 == 1),
              ('Real-camera common, clean', 'common', lambda n: real & ~isdeg), ('Real-camera common, degraded', 'common', lambda n: real & isdeg),
              ('Fair photos, clean', 'fair', None), ('Fair photos, degraded (same views)', 'fair_deg', None),
              ('Open Images, clean', 'oi', None), ('Open Images, degraded (same views)', 'oi_deg', None)]
    for name, s, mk in slices:
        cells = []
        for t in tiers:
            p, c, th = tp(t, s, s in DEG); m = mk(len(th)) if mk else np.ones(len(th), bool); e = circ_err(p, th)[m]
            cells.append(pct((e <= 10).mean()))
        ie = circ_err(inc[s], th)[m]
        print(f'| {name} ({m.sum()}) | {pct((ie <= 10).mean())} | ' + ' | '.join(cells) + ' |')
    print('\n## Real-camera common views by degradation group (within 10°)\n')
    fam = [set(d['family'] for d in degmap.get(i, [])) for i in ids]
    print('| Group (n) | Incumbent | ' + ' | '.join(t.name for t in tiers) + ' |'); print('|---' * (len(tiers) + 2) + '|')
    for g, fs in GROUPS.items():
        m = np.array([bool(f & fs) for f in fam])
        cells = []
        for t in tiers:
            p, c, th = tp(t, 'common'); cells.append(pct((circ_err(p, th)[m] <= 10).mean()))
        ie = circ_err(inc['common'], th)[m]
        print(f'| {g} ({m.sum()}) | {pct((ie <= 10).mean())} | ' + ' | '.join(cells) + ' |')
    rt = next(t for t in tiers if t.name.startswith('Max'))
    print('\nMax routed share per test: ' + ', '.join(f'{k} {v * 100:.0f}%' for k, v in rt.routed.items()))


if __name__ == '__main__':
    main()
