"""A9 scorer (release/results/protected-v2/PREREG.md): v2 release predictions on the 24 Sep held-out views vs the stored
24 Sep comparator predictions. Step 1 reproduces the 24 Sep comparator numbers (pipeline check). Writes RESULTS.md.
   python3 score_a9.py v2-predictions.npz"""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / 'protected'
SIGNS = HERE.parent.parent / 'benchmarks/competitor-signs'
TIERS = [('Pico', 'pico'), ('Nano', 'nano'), ('Fast', 'fast'), ('Balanced', 'balanced'), ('Pro', 'pro'), ('Max', 'max')]
COMPS = [('Woehrer 2026', 'woehrer'), ('Deep-OAD', 'deepoad'), ('GeoCalib', 'geocalib')]
DEV_CAMS = {'G331', 'G639'}                     # v2 development cameras (gf_dev_meva); see PREREG
rng = np.random.default_rng(20260930)


def err(p, t):
    d = np.abs(np.asarray(p) - np.asarray(t)) % 360
    return np.minimum(d, 360 - d)


def pct(x):
    return f'{100 * x:.1f}%'


def comp_preds(set_):
    out = {}
    for name, m in COMPS:
        z = np.load(V1 / f'comparator-{m}-all-sets.npz'); sg = json.load(open(SIGNS / f'{m}-sign.json'))['sign']
        out[name] = ((sg * z[f'{set_}_pred']) % 360, z[f'{set_}_theta'])
    return out


def paired(a_ok, b_ok, clusters, n_boot=10000):
    """Paired difference in correct views (a - b) with a 95% cluster bootstrap interval."""
    u, inv = np.unique(clusters, return_inverse=True)
    da = np.bincount(inv, weights=a_ok.astype(float)); db = np.bincount(inv, weights=b_ok.astype(float))
    diff = da - db; k = len(u)
    boots = np.array([diff[rng.integers(0, k, k)].sum() for _ in range(n_boot)])
    return int(round(diff.sum())), int(np.floor(np.percentile(boots, 2.5))), int(np.ceil(np.percentile(boots, 97.5)))


def main(pred_path):
    z = np.load(pred_path, allow_pickle=True)
    views = {'meva_frozen': [json.loads(l) for l in open(HERE / 'meva-frozen-views.jsonl')],
             'v1_frozen': [json.loads(l) for l in open(HERE / 'clean-v1-frozen-views.jsonl')]}
    L = ['# A9: v2 release models on the 24 Sep held-out sets (second use; see PREREG.md)', '',
         f'Predictions: `{Path(pred_path).name}`; scorer: `score_a9.py`. Within 10°, every view answered; (≥150° errors).', '']
    # ---- step 1: pipeline check against the 24 Sep report
    L += ['## Pipeline check: 24 Sep comparator numbers reproduced from the stored predictions', '']
    for s in ['meva_frozen', 'v1_frozen']:
        cp = comp_preds(s)
        assert np.allclose(cp['Woehrer 2026'][1], z[f'{s}_theta']), 'comparator views differ'
        L.append(f'- {s}: ' + ', '.join(f'{n} {pct((err(p, t) <= 10).mean())} ({int((err(p, t) >= 150).sum())})' for n, (p, t) in cp.items()))
    L.append('')
    for s, title in [('meva_frozen', 'MEVA test v1, frozen split: unseen CCTV cameras'), ('v1_frozen', 'Clean v1 frozen test: DIODE scene 12, MEVA G339, Poly Haven')]:
        th, var = z[f'{s}_theta'], z[f'{s}_variant']; n = len(th)
        if s == 'meva_frozen':
            clusters = np.array([v['frame_id'] for v in views[s]]); cams = np.array([v['frame_id'].split('/')[0] for v in views[s]])
            held = ~np.isin(cams, list(DEV_CAMS))
            slices = [('Held out: 4 cameras G329 / G420 / G421 / G474', held), ('All 6 cameras (G331, G639 = v2 development cameras)', np.ones(n, bool)),
                      ('Held out, clean', held & (var == 'clean')), ('Held out, degraded', held & (var == 'degraded'))] + \
                     [(f'camera {c}' + (' (v2 dev camera)' if c in DEV_CAMS else ''), cams == c) for c in sorted(set(cams))]
        else:
            clusters = np.array([v['parent_id'] for v in views[s]]); grp = np.array([v['group'] for v in views[s]]); held = np.ones(n, bool)
            slices = [('All (held out)', held), ('Clean', var == 'clean'), ('Degraded', var == 'degraded')] + [(g, grp == g) for g in sorted(set(grp))]
        cp = comp_preds(s)
        L += [f'## {title} ({n} views)', '', '### Within 10°, every view answered (≥150° errors)', '',
              '| Slice (n) | ' + ' | '.join(n_ for n_, _ in COMPS) + ' | ' + ' | '.join(t for t, _ in TIERS) + ' |',
              '|---|' + '---|' * (len(COMPS) + len(TIERS))]
        for name, m in slices:
            cells = []
            for cn, (p, t) in cp.items():
                e = err(p, t)[m]; cells.append(f'{pct((e <= 10).mean())} ({int((e >= 150).sum())})')
            for tn, tk in TIERS:
                e = err(z[f'{s}_{tk}_pred'], th)[m]; cells.append(f'{pct((e <= 10).mean())} ({int((e >= 150).sum())})')
            L.append(f'| {name} ({int(m.sum())}) | ' + ' | '.join(cells) + ' |')
        L += ['', '### Max vs comparators on the held-out views: paired difference in correct views, 95% cluster bootstrap', '']
        mx = err(z[f'{s}_max_pred'], th) <= 10
        for cn, (p, t) in cp.items():
            if cn == 'GeoCalib':
                continue
            d, lo, hi = paired(mx[held], (err(p, t) <= 10)[held], clusters[held])
            L.append(f'- Max − {cn}: {d:+d} views [{lo:+d}, {hi:+d}] of {int(held.sum())} ({"significant" if lo > 0 or hi < 0 else "not significant"})')
        L += ['', '### Abstention on the held-out views (thresholds as shipped, fp32)', '',
              '| Tier | standard: answered | correct among answered | ≥150° answered | strict: answered | correct among answered | ≥150° answered |',
              '|---|---|---|---|---|---|---|']
        for tn, tk in TIERS:
            e = err(z[f'{s}_{tk}_pred'], th); row = [tn]
            for mode in ['standard', 'strict']:
                keep = held & ~z[f'{s}_{tk}_abstain_{mode}']
                row += [pct(keep.sum() / held.sum()), pct((e[keep] <= 10).mean()) if keep.any() else '—', str(int((e[keep] >= 150).sum()))]
            L.append('| ' + ' | '.join(row) + ' |')
        L.append('')
    out = HERE / 'RESULTS.md'; out.write_text('\n'.join(L) + '\n'); print('\n'.join(L)); print('->', out)


if __name__ == '__main__':
    main(sys.argv[1])
