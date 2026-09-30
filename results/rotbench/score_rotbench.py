"""RotBench v2 scorer (release/results/rotbench-v2/PREREG.md). Step 1 reproduces the 24 Sep comparator rows from the
stored predictions; step 2 scores the six v2 tiers the same way. Writes RESULTS.md.
   python3 score_rotbench.py v2-rotbench-predictions.npz"""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / 'protected'
SIGNS = HERE.parent.parent / 'benchmarks/competitor-signs'
TIERS = [('Pico', 'pico'), ('Nano', 'nano'), ('Fast', 'fast'), ('Balanced', 'balanced'), ('Pro', 'pro'), ('Max', 'max')]
COMPS = [('Woehrer 2026', 'woehrer'), ('Deep-OAD', 'deepoad'), ('GeoCalib', 'geocalib')]
PUBLISHED_SMALL = [('Humans (published)', (0.99, 0.99, 0.99, 0.97)), ('GPT-5 (published)', (1.00, 0.41, 0.81, 0.59)),
                   ('Gemini 2.5 Pro (published)', (1.00, 0.50, 0.72, 0.40)), ('o3 (published)', (1.00, 0.45, 0.70, 0.48))]
V1_MAX = {'small': (1.00, 1.00, 0.98, 1.00), 'large': (1.00, 0.99, 0.99, 1.00)}   # COMPETITORS-v1-2026-09-24.md §1


def circ(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % 360
    return np.minimum(d, 360 - d)


def row(p, th):
    q = (np.round(p / 90) % 4) * 90; ok = circ(q, th) < 1
    per = [ok[th == t].mean() for t in (0, 270, 180, 90)]            # counter-clockwise 0 / 90 / 180 / 270
    return per, ok.mean(), (circ(p, th) <= 10).mean(), int((circ(p, th) >= 150).sum())


def main(path):
    z = np.load(path, allow_pickle=True)
    L = ['# RotBench with the v2 release models (see PREREG.md)', '',
         f'Predictions `{Path(path).name}`; scorer `score_rotbench.py`. 4-way accuracy per counter-clockwise rotation (prediction',
         'snapped to the nearest quarter turn), mean, within 10° and upside-down errors (≥150°); every image answered.', '']
    comps = {}
    for n, m in COMPS:
        c = np.load(V1 / f'comparator-{m}-all-sets.npz'); sg = json.load(open(SIGNS / f'{m}-sign.json'))['sign']
        comps[n] = {rb: ((sg * c[f'{rb}_pred']) % 360, c[f'{rb}_theta']) for rb in ('rotbench', 'rotbench_full')}
    for rb in ('rotbench', 'rotbench_full'):
        th, split = z[f'{rb}_theta'], z[f'{rb}_split']
        for n in comps:
            assert np.array_equal(comps[n][rb][1], th), f'{n} {rb}: view order differs from the 24 Sep run'
    L += ['View order and angles equal the stored 24 Sep comparator files for both variants (checked).', '']
    for rb, title in (('rotbench', 'official protocol (PIL rotate, no expand)'), ('rotbench_full', 'full-frame variant (expand=True)')):
        th, split = z[f'{rb}_theta'], z[f'{rb}_split']
        for sp in ('small', 'large'):
            m = split == sp
            L += [f'## RotBench-{sp.capitalize()} ({m.sum() // 4} photos), {title}', '',
                  '| Model | 0° | 90° | 180° | 270° | mean | within 10° | ≥150° |', '|---|---|---|---|---|---|---|---|']
            if rb == 'rotbench' and sp == 'small':
                for n, v in PUBLISHED_SMALL:
                    L.append(f'| {n} | ' + ' | '.join(f'{x:.2f}' for x in v) + ' | — | — | — |')
            for n in comps:
                per, mean, w10, t150 = row(comps[n][rb][0][m], th[m])
                L.append(f'| {n} (24 Sep predictions) | ' + ' | '.join(f'{x:.2f}' for x in per) + f' | {mean:.2f} | {100 * w10:.1f}% | {t150} |')
            if rb == 'rotbench':
                L.append('| v1 Max (24 Sep, reference) | ' + ' | '.join(f'{x:.2f}' for x in V1_MAX[sp]) + ' | — | — | — |')
            for tn, tk in TIERS:
                per, mean, w10, t150 = row(z[f'{rb}_{tk}_pred'][m], th[m])
                b = '**' if tk == 'max' else ''
                L.append(f'| {b}{tn}{b} | ' + ' | '.join(f'{x:.2f}' for x in per) + f' | {mean:.2f} | {100 * w10:.1f}% | {t150} |')
            L.append('')
    (HERE / 'RESULTS.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main(sys.argv[1])
