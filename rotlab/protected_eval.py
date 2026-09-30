#!/usr/bin/env python3
"""Protected final evaluation: run ONCE, on the frozen weights recorded in release/FREEZE.md.

Sets (never trained on, never scored before):
  meva_frozen  MEVA test v1 frozen split (whole unseen cameras; rotlab.meva_test views, seed 11)
  v1_frozen    clean v1 frozen test: DIODE scene 12, MEVA G339, Poly Haven (clean-v1-splits.json 'frozen_test')
Each image gives a clean and a CCTV-degraded view at seeded angles with the fixed 4:3 crop, as in the development tests.

  python -m rotlab.protected_eval dump --i-am-releasing     # views -> cviews/{meva_frozen,v1_frozen}.pkl (lossless PNG)
  python -m rotlab.protected_eval ours --i-am-releasing     # frozen tiers (checks checkpoint SHA-256; refuses to re-run)
  python -m rotlab.competitors run {woehrer,deepoad,geocalib} meva_frozen,v1_frozen    # comparators, same pixels
  python -m rotlab.protected_eval report > RESULTS.md
"""
import argparse, collections, hashlib, io, json, pickle, random, sys
from pathlib import Path

import numpy as np
from PIL import Image

from rotlab.core import DATA, circ_err, decode, degrade, letterbox, render

ROOT = DATA / 'rotlab/protected'
VC = DATA / 'rotlab/cviews'
COMP = DATA / 'rotlab/competitors'
V1 = DATA / 'archives/ortus-rotation-hybrid-v1-2026-08-27/frozen-dataset'
SPLITS = DATA / 'rotlab/clean-v1-splits.json'
SMALL = DATA / 'rotlab/runs/S2-vits-multires/final.pt'
LARGE = DATA / 'rotlab/runs/soup-G7-G3-a0.5/final.pt'
FROZEN_SHA = {SMALL: 'de9cad7675f77c68efd6b91693484c4e9f75a41d579a87b5ec35910ee6229284',
              LARGE: 'ae53431c92acfd123ea5b5a08c1dcc4850e88772e1d1e5fbb0fd6886d42276f9'}
# (name, small canvas, uses large, route share) -- thresholds are recomputed from calibration data and must match FREEZE.md
TIERS = [('Nano', 112, False, 0.0), ('Fast', 224, False, 0.0), ('Balanced', 224, True, 0.10), ('Max', 224, True, 0.20)]
FROZEN_THRESH = {'Nano': (None, 0.485, 0.785), 'Fast': (None, 0.706, 0.743), 'Balanced': (0.706, 0.778, 0.750),
                 'Max': (0.836, 0.808, 0.677)}   # (route, abstain standard, abstain strict), as recorded in FREEZE.md
SETS = ('meva_frozen', 'v1_frozen')


def guard(a):
    if not a.i_am_releasing:
        raise SystemExit('protected sets: run once, at release, with --i-am-releasing (see release/FREEZE.md)')


def dump(a):
    guard(a)
    for s in SETS:
        if (VC / f'{s}.pkl').exists():
            raise SystemExit(f'{s} views already exist; the protected views are built once')
    from rotlab.meva_test import views as meva_views
    mv = meva_views('frozen')
    sets = {'meva_frozen': ([v['view'] for v in mv], [v['theta'] for v in mv], [f"meva:{v['cam']}" for v in mv],
                            [v['variant'] for v in mv])}
    keep = set(json.loads(SPLITS.read_text())['frozen_test'])
    parents = [p for p in map(json.loads, open(V1 / 'metadata/parents-materialized.jsonl')) if p['parent_id'] in keep]
    rng = random.Random('protected:v1_frozen'); vs, th, gr, va = [], [], [], []
    for p in sorted(parents, key=lambda p: p['archive_parent_path']):
        src = Image.open(V1 / p['archive_parent_path']).convert('RGB'); W, H = src.size; d = min(W, H) - 4; cw, ch = 0.8 * d, 0.6 * d
        grp = p['source_family'] + (':' + p['parent_group'].split(':')[-1] if p['source_family'] != 'poly_haven' else '')
        for variant in ('clean', 'degraded'):
            t = rng.uniform(0, 360); v = render(src, t, cw, ch, round(cw), round(ch))
            if variant == 'degraded':
                v = degrade(v, rng)
            vs.append(v); th.append((t + float(p['base_orientation_degrees'])) % 360); gr.append(grp); va.append(variant)
    sets['v1_frozen'] = (vs, th, gr, va)
    for s, (vs, th, gr, va) in sets.items():
        png = []
        for v in vs:
            b = io.BytesIO(); v.save(b, format='PNG'); png.append(b.getvalue())
        (VC / f'{s}.pkl').write_bytes(pickle.dumps(dict(png=png, theta=np.asarray(th), group=gr, variant=va)))
        print(s, len(png), 'views', dict(collections.Counter(gr)), flush=True)


def load(s):
    d = pickle.loads((VC / f'{s}.pkl').read_bytes())
    return [Image.open(io.BytesIO(b)).convert('RGB') for b in d['png']], d['theta'], np.array(d['group']), np.array(d['variant'])


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def thresholds():
    """Recompute route/abstain thresholds from the calibration predictions exactly as rotlab.final_stats does."""
    from rotlab.final_stats import Tier
    out = {}
    for name, cv, big, route in TIERS:
        t = Tier(name, SMALL.parent.name, LARGE.parent.name if big else None, route)
        t.calibrate(f'c{cv}', 'c224')
        out[name] = (getattr(t, 't_route', None), t.t_cov90, t.t_strict)
        want = FROZEN_THRESH[name]
        assert all((x is None and y is None) or abs(x - y) < 5e-4 for x, y in zip(out[name], want)), (name, out[name], want)
    return out


def ours(a):
    guard(a)
    import torch
    from rotlab.evaluate import confidence, predict
    from rotlab.model import RotNet
    f = ROOT / 'ours.npz'
    if f.exists():
        raise SystemExit('protected evaluation of the frozen weights already ran; results are in ' + str(ROOT))
    for p, want in FROZEN_SHA.items():
        got = sha(p); assert got == want, f'{p} does not match FREEZE.md ({got})'
    th = thresholds(); ROOT.mkdir(parents=True, exist_ok=True)
    nets = {}
    for key, ck in (('small', SMALL), ('large', LARGE)):
        c = torch.load(ck, map_location='cpu', weights_only=False)
        m = RotNet(img_size=c['config']['img_size'], pretrained=False, dynamic=True, arch=c['config'].get('arch', 's')).cuda().eval()
        m.load_state_dict(c['ema']); nets[key] = m
    rows = {}
    for s in SETS:
        vs, theta, _, _ = load(s); rows[f'{s}_theta'] = theta
        for key, cv in (('small', 112), ('small', 224), ('large', 224)):
            x = np.stack([letterbox(v, cv) for v in vs]); prob = predict(nets[key], x, tta180=False); p = decode(prob)
            rows[f'{s}_{key}{cv}_pred'] = p; rows[f'{s}_{key}{cv}_conf'] = confidence(prob, p)
    np.savez(f, **rows)
    (ROOT / 'thresholds.json').write_text(json.dumps(th, indent=1))
    print('scored', {s: len(rows[f'{s}_theta']) for s in SETS})


def tier_pred(z, s, name, th):
    _, cv, big, _ = next(t for t in TIERS if t[0] == name)
    p, c = z[f'{s}_small{cv}_pred'].copy(), z[f'{s}_small{cv}_conf'].copy(); r = np.zeros(len(p), bool)
    if big:
        r = c < th[name][0]; p[r], c[r] = z[f'{s}_large224_pred'][r], z[f'{s}_large224_conf'][r]
    return p, c, r


def pct(x): return f'{100 * x:.1f}%'


def report(a):
    z = np.load(ROOT / 'ours.npz'); th = {k: tuple(v) for k, v in json.loads((ROOT / 'thresholds.json').read_text()).items()}
    comps = []
    for n, m in (('Previous SOTA (Woehrer 2026)', 'woehrer'), ('Deep-OAD', 'deepoad'), ('GeoCalib', 'geocalib')):
        if (COMP / f'{m}.npz').exists():
            # native outputs -> clockwise chart with the sign fixed once on calibration photos (rotlab.competitors sign)
            sg = json.loads((COMP / f'{m}-sign.json').read_text())['sign']; cz = dict(np.load(COMP / f'{m}.npz'))
            comps.append((n, {k: (sg * v) % 360 if k.endswith('_pred') else v for k, v in cz.items()}))
    names = [t[0] for t in TIERS]
    print('# Protected final evaluation (frozen weights, run once)\n')
    print('Weights and thresholds as recorded in `FREEZE.md`; views built once, identical pixels for every model.\n')
    for s, title in (('meva_frozen', 'MEVA test v1, frozen split: unseen CCTV cameras'),
                     ('v1_frozen', 'Clean v1 frozen test: DIODE scene 12, MEVA G339, Poly Haven')):
        _, theta, gr, va = load(s); n = len(theta)
        print(f'## {title} ({n} views)\n')
        print('### Accuracy within 10°, all views answered (upside-down errors ≥150° in brackets)\n')
        cols = [c for c, _ in comps] + names
        print('| Slice (n) | ' + ' | '.join(cols) + ' |'); print('|---' * (len(cols) + 1) + '|')
        slices = [('All', np.ones(n, bool)), ('Clean', va == 'clean'), ('Degraded', va == 'degraded')] + \
                 [(g, gr == g) for g in sorted(set(gr))]
        for label, m in slices:
            cells = []
            for _, cz in comps:
                if f'{s}_pred' in cz:
                    e = circ_err(cz[f'{s}_pred'], theta)[m]; cells.append(f'{pct((e <= 10).mean())} ({(e >= 150).sum()})')
                else:
                    cells.append('—')
            for nm in names:
                p, _, _ = tier_pred(z, s, nm, th); e = circ_err(p, theta)[m]; cells.append(f'{pct((e <= 10).mean())} ({(e >= 150).sum()})')
            print(f'| {label} ({m.sum()}) | ' + ' | '.join(cells) + ' |')
        scope = np.abs(((theta + 180) % 360) - 180) <= 45
        print(f'\n### Within ±45° true roll only (calibration-model scope, n = {scope.sum()})\n')
        print('| ' + ' | '.join(cols) + ' |'); print('|---' * len(cols) + '|')
        cells = []
        for _, cz in comps:
            cells.append(pct((circ_err(cz[f'{s}_pred'], theta)[scope] <= 10).mean()) if f'{s}_pred' in cz else '—')
        for nm in names:
            p, _, _ = tier_pred(z, s, nm, th); cells.append(pct((circ_err(p, theta)[scope] <= 10).mean()))
        print('| ' + ' | '.join(cells) + ' |')
        for idx, label in ((1, 'standard (90% answered on calibration)'), (2, 'strict (≤1% wrong on calibration)')):
            print(f'\n### With abstention: {label}\n')
            print('| Tier | answered | correct within 10° among answered | upside-down among answered |'); print('|---|---|---|---|')
            for nm in names:
                p, c, _ = tier_pred(z, s, nm, th); keep = c >= th[nm][idx]; e = circ_err(p, theta)
                print(f'| {nm} | {pct(keep.mean())} | {pct((e[keep] <= 10).mean()) if keep.any() else "—"} | {(e[keep] >= 150).sum()} |')
        routed = {nm: tier_pred(z, s, nm, th)[2].mean() for nm in ('Balanced', 'Max')}
        print(f"\nRouted to the large model: Balanced {pct(routed['Balanced'])}, Max {pct(routed['Max'])}.\n")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['dump', 'ours', 'report'])
    ap.add_argument('--i-am-releasing', action='store_true')
    a = ap.parse_args()
    {'dump': dump, 'ours': ours, 'report': report}[a.cmd](a)
