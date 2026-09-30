#!/usr/bin/env python3
"""Evaluate a checkpoint on the fixed 2,530-row benchmark pack (common/fresh_diode/origin).

Reports per panel and per source, plus the faithful/installed incumbents on identical rows,
and the origin ∩ COCO-val2017 subset (115 rows). Optional 180-degree test-time averaging.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rotlab.core import BUCKETS, PACK, bucket_of, circ_err, decode, letterbox, load_pack, metrics
from rotlab.model import RotNet

VAL17 = Path(__file__).with_name('origin_val2017_ids.json')
C4_SIGN = 1


class PackCache:
    """Preprocessed pack tensors, built once per canvas size and cached on disk."""

    def __init__(self, size=224, buckets=False):
        self.rows = load_pack(); self.buckets = buckets
        if not buckets:
            cache = PACK.parent / f'rotlab-pack-cache-{size}.npy'
            if cache.exists():
                self.x = np.load(cache, mmap_mode='r')
            else:
                x = np.stack([letterbox(Image.open(r['image']['path']).convert('RGB'), size) for r in self.rows])
                np.save(cache, x); self.x = x
        else:
            cache = PACK.parent / 'rotlab-pack-cache-buckets.npz'
            if not cache.exists():
                ims = [Image.open(r['image']['path']).convert('RGB') for r in self.rows]
                bk = np.array([bucket_of(*im.size) for im in ims])
                arrs = {b: np.stack([letterbox(im, BUCKETS[b]) for im, k in zip(ims, bk) if k == b]) for b in BUCKETS if (bk == b).any()}
                np.savez(cache, bk=bk, **arrs)
            z = np.load(cache)
            self.bk = z['bk']; self.x = {b: z[b] for b in BUCKETS if b in z}
        self.target = np.array([r['target_degrees'] for r in self.rows], np.float64)
        v = set(json.loads(VAL17.read_text())['in_val2017']) if VAL17.exists() else set()
        self.val17 = np.array([r['panel'] == 'origin' and _coco_id(r) in v for r in self.rows]) if v else None


def _coco_id(r):
    m = re.search(r'val2014_0*(\d+)', r['image']['path'])
    return int(m.group(1)) if m else -1


@torch.no_grad()
def predict(model, x, bs=128, tta180=False, device='cuda', pc=None):
    if isinstance(x, dict):  # bucketed pack: predict per bucket, restore original row order
        out = np.zeros((len(pc.bk), 360), np.float32)
        for b, xb in x.items():
            out[pc.bk == b] = predict(model, xb, bs, tta180, device)
        return out
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        b = torch.from_numpy(np.ascontiguousarray(x[i:i + bs])).to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p = model(b).float().softmax(1)
            if tta180 == 'c4':   # quarter-turn views (square canvas only); SIGN = +1 or -1 fixed empirically
                acc = p.clone()
                for k in (1, 2, 3):
                    q = model(torch.rot90(b, k, (2, 3))).float().softmax(1)
                    acc += torch.roll(q, C4_SIGN * 90 * k, 1)
                p = acc / 4
            elif tta180:
                q = model(torch.rot90(b, 2, (2, 3))).float().softmax(1)
                p = (p + torch.roll(q, -180, 1)) / 2
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def report(pc: PackCache, prob: np.ndarray, label: str):
    pred = decode(prob)
    err = circ_err(pred, pc.target)
    res = dict(label=label, panels={}, sources={})
    fam = np.array([r['source_family'] for r in pc.rows]); pan = np.array([r['panel'] for r in pc.rows])
    for p in ('common', 'fresh_diode', 'origin'):
        m = pan == p
        res['panels'][p] = metrics(err[m])
        for arm in ('faithful', 'installed'):
            e = circ_err(np.array([r[arm] for r, k in zip(pc.rows, m) if k]), pc.target[m])
            res['panels'][p][arm] = metrics(e)
    for f in sorted(set(fam[pan == 'common'])):
        m = (pan == 'common') & (fam == f)
        res['sources'][f] = metrics(err[m])
    if pc.val17 is not None and pc.val17.any():
        res['panels']['origin_val2017'] = metrics(err[pc.val17])
        res['panels']['origin_val2017']['faithful'] = metrics(circ_err(np.array([r['faithful'] for r, k in zip(pc.rows, pc.val17) if k]), pc.target[pc.val17]))
    return res, pred


def confidence(prob, pred, width=10):
    """Posterior mass within +-width degrees of the decoded prediction."""
    k = np.arange(360)[None]
    d = np.abs((k - pred[:, None] + 180) % 360 - 180)
    return (prob * (d <= width)).sum(1)


def table(res):
    lines = [f"### {res['label']}", '', '| Panel | model within10 / mean / >90 / ≥150 | faithful within10 / ≥150 |', '|---|---|---|']
    for p, m in res['panels'].items():
        f = m.get('faithful', {})
        lines.append(f"| {p} ({m['n']}) | **{m['w10']}** / {m['mean']}° / {m['gt90']} / {m['t150']} | {f.get('w10','')} / {f.get('t150','')} |")
    lines.append('')
    lines.append('| Common source | within10 / mean / ≥150 |'); lines.append('|---|---|')
    for s, m in res['sources'].items():
        lines.append(f"| {s} | {m['w10']}/{m['n']} / {m['mean']}° / {m['t150']} |")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint', type=Path)
    ap.add_argument('--tta180', action='store_true')
    ap.add_argument('--ema', action='store_true')
    ap.add_argument('--canvas', type=int, default=0, help='evaluate at this square size (multi-resolution checkpoints)')
    a = ap.parse_args()
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['config']
    canvas = a.canvas or cfg.get('canvas', cfg['img_size'])
    model = RotNet(img_size=cfg['img_size'], pretrained=False, hidden=cfg.get('hidden', 1024), dynamic=True, arch=cfg.get('arch', 's')).cuda()
    model.load_state_dict(ck['ema' if a.ema and 'ema' in ck else 'model'])
    pc = PackCache(canvas, cfg.get('buckets', False))
    res, pred = report(pc, predict(model, pc.x, tta180=a.tta180, pc=pc), f"{a.checkpoint.parent.name}/{a.checkpoint.name}{' ema' if a.ema else ''}{' tta180' if a.tta180 else ''}")
    print(table(res))
    out = a.checkpoint.with_suffix(f".eval{'-ema' if a.ema else ''}{'-tta' if a.tta180 else ''}{f'-c{a.canvas}' if a.canvas else ''}.json")
    prob = predict(model, pc.x, tta180=a.tta180, pc=pc)
    conf = confidence(prob, pred)
    out.write_text(json.dumps(dict(result=res, predictions=[float(v) for v in pred], confidence=[round(float(c), 5) for c in conf]), indent=1))


if __name__ == '__main__':
    main()
