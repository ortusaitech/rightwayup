#!/usr/bin/env python3
"""MEVA real-CCTV test set v1: 15 cameras never trained on (see rotlab.meva_frames), split by whole
camera into dev (inspect freely) and frozen (protected, scored once at release). Each frame gives a
clean and a CCTV-degraded view at independent seeded angles, rendered with the angle-independent
fixed 4:3 crop (diagonal = min(W,H)) used by the fair photo set.
  python -m rotlab.meva_test --split dev CKPT [CKPT ...]
"""
import argparse, collections, hashlib, json, random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rotlab.core import DATA, circ_err, decode, degrade, letterbox, metrics, render
from rotlab.evaluate import confidence, predict
from rotlab.fair_photo import incumbent
from rotlab.model import RotNet

ROOT = DATA / 'rotlab/meva-test-v1'
THERMAL = {'G474', 'G475', 'G476', 'G479'}
# Owner decision 2026-09-23: training had no thermal footage, so two thermal cameras move to training (whole
# cameras, never clips of a test camera). G475 came from dev; G476 from the never-scored frozen split.
TO_TRAIN = {'G475', 'G476'}


def split_of(cam):
    """Hash-ranked within stratum (thermal / visible); first half of each stratum -> dev. TO_TRAIN -> 'train'."""
    if cam in TO_TRAIN:
        return 'train'
    for group in (sorted(THERMAL), sorted({'G326', 'G329', 'G331', 'G336', 'G340', 'G341', 'G420', 'G421', 'G423', 'G508', 'G639'})):
        if cam in group:
            ranked = sorted(group, key=lambda c: hashlib.md5(f'meva-test-v1:{c}'.encode()).hexdigest())
            return 'dev' if ranked.index(cam) < (len(ranked) + 1) // 2 else 'frozen'
    raise KeyError(cam)


def views(split, seed=11):
    rows = [json.loads(l) for l in open(ROOT / 'frames.jsonl')]
    rows = [r for r in rows if split_of(r['cam']) == split]
    rng = random.Random(f'{seed}:{split}'); out = []
    for r in rows:
        src = Image.open(ROOT / 'frames' / r['cam'] / Path(r['path']).name).convert('RGB'); W, H = src.size
        d = min(W, H) - 4; cw, ch = 0.8 * d, 0.6 * d
        for variant in ('clean', 'degraded'):
            th = rng.uniform(0, 360)
            v = render(src, th, cw, ch, round(cw), round(ch))
            if variant == 'degraded':
                v = degrade(v, rng)
            out.append(dict(cam=r['cam'], variant=variant, theta=(th + r['base_roll_cw']) % 360, view=v))
    return out


def summary(err, vs):
    res = {'all': metrics(err)}
    for key in ('variant', 'cam'):
        g = collections.defaultdict(list)
        for e, v in zip(err, vs):
            g[v[key]].append(e)
        res[key] = {k: metrics(np.array(e)) for k, e in sorted(g.items())}
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('ckpts', nargs='*'); ap.add_argument('--split', default='dev')
    ap.add_argument('--i-am-releasing', action='store_true', help='required to score the frozen split')
    ap.add_argument('--threads', type=int, default=16)
    a = ap.parse_args()
    if a.split == 'frozen' and not a.i_am_releasing:
        raise SystemExit('frozen MEVA split is protected: score it once, at release, with --i-am-releasing')
    vs = views(a.split); th = np.array([v['theta'] for v in vs])
    cams = collections.Counter(v['cam'] for v in vs)
    print(f'{a.split}: {len(vs)} views, {len(cams)} cameras', dict(sorted(cams.items())), flush=True)
    out = ROOT / f'results-{a.split}'; out.mkdir(exist_ok=True)
    inc_f = out / 'incumbent.npz'
    if not inc_f.exists():
        inc = incumbent(a.threads); np.savez(inc_f, pred=np.array([inc(v['view']) for v in vs]), theta=th)
    pi = np.load(inc_f)['pred']
    res = {'incumbent': summary(circ_err(pi, th), vs)}
    for c in a.ckpts:
        ck = torch.load(c, map_location='cpu', weights_only=False); cv = ck['config'].get('canvas', 224)
        m = RotNet(img_size=224, pretrained=False, dynamic=cv != 224, arch=ck['config'].get('arch', 's')).cuda().eval()
        m.load_state_dict(ck['ema'])
        x = np.stack([letterbox(v['view'], cv) for v in vs]); name = Path(c).parent.name
        rows = {}
        for tta, key in ((False, 'model'), (True, 'model_tta')):
            prob = predict(m, x, tta180=tta); p = decode(prob)
            res[f'{name}:{key}'] = summary(circ_err(p, th), vs); rows[f'{key}_pred'] = p; rows[f'{key}_conf'] = confidence(prob, p)
        np.savez(out / f'rows-{name}.npz', theta=th, incumbent=pi, **rows)
        del m; torch.cuda.empty_cache()
    (out / 'summary.json').write_text(json.dumps(res, indent=1))
    n = len(vs)
    print(f'| model | all ({n}) w10 | clean | degraded | >=150 |')
    for k, r in res.items():
        print(f"| {k} | {r['all']['w10']} ({r['all']['w10_pct']}%) | {r['variant']['clean']['w10']} | {r['variant']['degraded']['w10']} | {r['all']['t150']} |")
    print('per camera w10:', {k: {c: m['w10'] for c, m in r['cam'].items()} for k, r in res.items()})


if __name__ == '__main__':
    main()
