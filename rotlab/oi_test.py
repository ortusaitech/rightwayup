#!/usr/bin/env python3
"""Open Images photo test: eligible Open Images *validation* photos (same rights/orientation screening as
training, never trained on by us or by the previous SOTA), rendered at seeded random angles with
(a) the angle-independent fixed 4:3 crop and (b) the previous SOTA's max-area crop (reported separately).
Also re-scores the MEVA dev split. Writes DATA/rotlab/oi-test/summary-<policy>.json and rows-<policy>-<run>.npz.
  python -m rotlab.oi_test CKPT [CKPT ...]
"""
import argparse, json, random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rotlab.core import DATA, circ_err, decode, largest_rotated_rect, letterbox, metrics, render
from rotlab.evaluate import confidence, predict
from rotlab.fair_photo import incumbent
from rotlab.model import RotNet

SRC = DATA / 'rotlab/openimages/validation'
OUT = DATA / 'rotlab/oi-test'


def views(policy, seed=13):
    recs = sorted((json.loads(p.read_text()) for p in (SRC / 'records').glob('*.json')), key=lambda r: r['id'])
    from rotlab.openimages import candidates
    test_ids = {r['ImageID'] for r in candidates('validation', 4000)}     # ranks 0-3999 = photo test; 4000-7999 = calibration
    recs = [r for r in recs if r['status'] == 'eligible' and r['id'] in test_ids]
    rng = random.Random(f'{seed}:{policy}'); out, th = [], []
    for r in recs:
        src = Image.open(SRC / 'img' / f"{r['id']}.jpg").convert('RGB'); W, H = src.size
        t = rng.uniform(0, 360)
        if policy == 'fixed':
            d = min(W, H) - 4; cw, ch = 0.8 * d, 0.6 * d
        else:
            cw, ch = largest_rotated_rect(W, H, t); cw, ch = cw - 2, ch - 2
        out.append(render(src, t, cw, ch, max(8, round(cw)), max(8, round(ch)))); th.append(t)
    return out, np.array(th)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('ckpts', nargs='*'); ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    models = []
    for c in a.ckpts:
        ck = torch.load(c, map_location='cpu', weights_only=False); cv = ck['config'].get('canvas', 224)
        m = RotNet(img_size=224, pretrained=False, dynamic=cv != 224, arch=ck['config'].get('arch', 's')).cuda().eval()
        m.load_state_dict(ck['ema']); models.append((Path(c).parent.name, m, cv))
    for policy in ('fixed', 'maxarea'):
        vs, th = views(policy)
        f = OUT / f'incumbent-{policy}.npz'
        if not f.exists():
            inc = incumbent(a.threads); np.savez(f, pred=np.array([inc(v) for v in vs]), theta=th)
        pi = np.load(f)['pred']; res = {'n': len(vs), 'incumbent': metrics(circ_err(pi, th))}
        for name, m, cv in models:
            x = np.stack([letterbox(v, cv) for v in vs]); rows = {}
            for tta, key in ((False, 'model'), (True, 'model_tta')):
                prob = predict(m, x, tta180=tta); p = decode(prob)
                res[f'{name}:{key}'] = metrics(circ_err(p, th)); rows[f'{key}_pred'] = p; rows[f'{key}_conf'] = confidence(prob, p)
            np.savez(OUT / f'rows-{policy}-{name}.npz', theta=th, incumbent=pi, **rows)
        (OUT / f'summary-{policy}.json').write_text(json.dumps(res, indent=1))
        print(f'## {policy} crop, n={len(vs)}')
        for k, r in res.items():
            if k != 'n':
                print(f"| {k} | {r['w10']} ({r['w10_pct']}%) | mean {r['mean']} | >=150 {r['t150']} |", flush=True)


if __name__ == '__main__':
    main()
