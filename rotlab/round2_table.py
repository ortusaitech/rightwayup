"""PREREG round 2 read-out (lead 30 Sep 2026): K3 deltas vs N70L with paired McNemar, K1' (corner-induced flips) per panel,
with and without the R2-2 repaint. Reads camp-eval stores and corner-panel per-item caches; prints one table."""
import os
import numpy as np
from pathlib import Path
from scipy.stats import binomtest

DATA = Path(os.environ.get('ROTLAB_DATA', '/workspace/rotation-data')) / 'rotlab'
CE, CP = DATA / 'camp-eval', DATA / 'corner-panel'
K3 = ['gf_dev_all_cam', 'gf_photo2_clean', 'gf_dev_diode_cam']
EXTRA = ['gf_dev_diode_clean', 'gf_dev_all_clean', 'oi', 'oi_deg', 'newoi_val_fixed', 'newoi_val_fixed_deg']
FIVE = [f'coco2014_faithful_seed{i}' for i in range(5)]
MODELS = [('N70L', 'PICO-N70L-fill-c70', 'N70L', 'pico'), ('N70C', 'PICO-N70C-fill-c70', 'N70C', 'pico'),
          ('S50', 'PICO-N70S50-fill-c70', 'N70S50', 'pico'), ('S70', 'PICO-N70S70-fill-c70', 'N70S70', 'pico'),
          ('S80', 'PICO-N70S80-fill-c70', 'N70S80', 'pico'), ('S90', 'PICO-N70S90-fill-c70', 'N70S90', 'pico'),
          ('Nano', None, 'N70L', 'nano')]


def ok(z, s):
    d = np.abs(z[s + '_pred'] - z[s + '_theta']) % 360
    return np.minimum(d, 360 - d) <= 10


def err(z):
    d = np.abs(z['angle'] - z['theta']) % 360
    return np.minimum(d, 360 - d)


ref = np.load(CE / 'PICO-N70L-fill-c70.npz')
print('K3 / dev (fill mode @70): % within 10 deg, change vs N70L in views, McNemar p')
for nm, st, _, _ in [m for m in MODELS if m[1] and m[0] != 'N70L']:
    f = CE / f'{st}.npz'
    if not f.exists():
        print(nm, 'no store'); continue
    z = np.load(f); row = []
    for s in K3 + EXTRA:
        a, b = ok(ref, s), ok(z, s); lost, gained = int((a & ~b).sum()), int((~a & b).sum())
        p = binomtest(lost, lost + gained).pvalue if lost + gained else 1.0
        row.append(f'{s} {100 * b.mean():.2f} ({gained - lost:+d}, p {p:.2f})')
    a = np.concatenate([ok(ref, s) for s in FIVE]); b = np.concatenate([ok(z, s) for s in FIVE])
    row.append(f'FIVE {100 * b.mean():.2f} ({int(b.sum() - a.sum()):+d})')
    k3 = all(100 * (ok(z, s).mean() - ok(ref, s).mean()) >= -1.0 for s in K3) and 100 * (b.mean() - a.mean()) >= -1.0
    print(f'{nm}: K3 {"PASS" if k3 else "FAIL"} | ' + ' | '.join(row))
print("\nK1' corner-induced flips (P2 >=150 and q90 <=10), P2 within-10 / >=150; +repaint = R2-2 detector at inference")
for s in ['gf_dev_all_clean', 'gf_photo2_clean']:
    for nm, _, tag, tier in MODELS:
        g = lambda v: CP / f'items-{tag}-fp32-cuda-{s}_upright{v}-{tier}.npz'
        if not g('').exists():
            continue
        e2, eq = err(np.load(g(''))), err(np.load(g('-q90')))
        line = f"{s:17s} {nm:5s} K1' {100 * ((e2 >= 150) & (eq <= 10)).mean():5.2f}  P2 {100 * (e2 <= 10).mean():5.2f} / {100 * (e2 >= 150).mean():5.2f}"
        rp = CP / f'items-{tag}-fp32-cuda-{s}_upright+repaint-{tier}.npz'
        if rp.exists():
            er = err(np.load(rp))
            line += f"  | +repaint K1' {100 * ((er >= 150) & (eq <= 10)).mean():5.2f}  P2 {100 * (er <= 10).mean():5.2f} / {100 * (er >= 150).mean():5.2f}"
        print(line)
