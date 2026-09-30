#!/usr/bin/env python3
"""Campaign 2026-09-27 evaluation harness (GPU). Views come from lossless view caches (PNG + clockwise theta):
DATA/rotlab/cviews/<set>.pkl (reused development panels) and DATA/rotlab/newdata/views/<set>.pkl (new val sets).
Stores per set: prob (float16, 360 bins), pred (decode), conf (+-10 deg mass), theta ->
DATA/rotlab/camp-eval/<tag>.npz with keys <set>_{prob,pred,conf,theta}.

  python -m rotlab.camp_eval CKPT --canvases 168,224 [--sets dev,newval] [--pool K:G] [--tag NAME]
--pool K:G = static spatial token pooling after block K to a GxG grid (area weights as a fixed matrix; CLS kept).
"""
import argparse, hashlib, io, math, os, pickle, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from rotlab.core import DATA, circ_err, decode, letterbox
from rotlab.evaluate import confidence
from rotlab.model_x import RotNet   # lead: all SPECS archs; s/b/l identical to rotlab.model.RotNet

CV = DATA / 'rotlab/cviews'
NV = DATA / 'rotlab/newdata/views'
OUT = DATA / 'rotlab/camp-eval'
DEV = ['origin', 'fair', 'oi', 'common', 'fresh_diode', 'meva', 'fair_deg', 'oi_deg']
NEWVAL = ['newcoco_val_fixed', 'newcoco_val_maxarea', 'newcoco_val_fixed_deg', 'newoi_val_fixed', 'newoi_val_fixed_deg']


DECODER = 'argmax + local circular mean over +-10 bins (core.decode); conf = mass within +-10 deg'
FROZEN = DATA / 'rotlab/camp-frozen/FROZEN.json'


def view_file(name):
    return (NV if name.startswith(('new', 'rlivit', 'aalborg')) else CV) / f'{name}.pkl'


def sha256_file(f):
    import hashlib
    side = Path(str(f) + '.sha256')
    if side.exists() and side.stat().st_mtime >= Path(f).stat().st_mtime:
        return side.read_text().split()[0]
    h = hashlib.sha256()
    with open(f, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 24), b''):
            h.update(b)
    side.write_text(h.hexdigest() + '\n')
    return h.hexdigest()


def load_views(name, with_ids=False):
    d = pickle.loads(view_file(name).read_bytes())
    out = [Image.open(io.BytesIO(b)).convert('RGB') for b in d['png']], np.asarray(d['theta'], np.float64)
    if with_ids:   # stable source-image ids (new sets); reused dev caches have none -> None
        return out + (np.asarray(d['ids']) if 'ids' in d else None,)
    return out


def area_matrix(g, p):
    """(p*p, g*g) matrix averaging a g x g grid into p x p cells with exact area weights."""
    def one_d(g, p):
        m = np.zeros((p, g))
        for i in range(p):
            a, b = i * g / p, (i + 1) * g / p
            for j in range(g):
                m[i, j] = max(0.0, min(b, j + 1) - max(a, j))
            m[i] /= m[i].sum()
        return m
    m = one_d(g, p)
    return torch.tensor(np.kron(m, m), dtype=torch.float32)


class Pooled(torch.nn.Module):
    """RotNet with static token pooling after block k (grid g -> p). Same weights, no new parameters."""

    def __init__(self, net: RotNet, k, p):
        super().__init__(); self.net, self.k, self.p = net, k, p; self._W = {}

    def forward(self, x):
        bb = self.net.backbone
        t = bb.patch_embed(x); t = bb._pos_embed(t); t = bb.patch_drop(t); t = bb.norm_pre(t)
        for i, blk in enumerate(bb.blocks):
            if i == self.k:
                n = t.shape[1] - 1; g = int(round(math.sqrt(n)))
                key = (g, t.device)
                if key not in self._W:
                    self._W[key] = area_matrix(g, self.p).to(t.device)
                W = self._W[key].to(t.dtype)
                t = torch.cat([t[:, :1], torch.einsum('qn,bnd->bqd', W, t[:, 1:])], 1)
            t = blk(t)
        t = bb.norm(t)
        f = torch.cat([t[:, 0], t[:, 1:].mean(1)], 1)
        return self.net.head(f)


def load_model(ckpt, pool=None):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False); cfg = ck['config']
    m = RotNet(img_size=cfg['img_size'], pretrained=False, dynamic=True, arch=cfg.get('arch', 's'))
    m.load_state_dict(ck['ema']); m = m.cuda().eval()
    if cfg.get('pool') and not pool:
        m.pool = tuple(map(int, cfg['pool'].split(':')))
    if pool:
        if m.arch not in ('s', 'b', 'l'):   # Pooled assumes 1 prefix token and bypasses the input adapter
            raise SystemExit(f'--pool is implemented for s/b/l only, not {m.arch}')
        k, p = map(int, pool.split(':')); m = Pooled(m, k, p).cuda().eval()
    return m


@torch.no_grad()
def probs(m, x, bs=128):
    out = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out.append(m(xb).float().softmax(1).cpu().numpy())
    return np.concatenate(out)


def main():
    import json
    ap = argparse.ArgumentParser(); ap.add_argument('ckpt'); ap.add_argument('--canvases', default='224')
    ap.add_argument('--sets', default='dev'); ap.add_argument('--pool'); ap.add_argument('--tag')
    ap.add_argument('--features', action='store_true', help='also store the head input (concat CLS, mean patch) as <set>_feat (float16)')
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    names = []
    for s in a.sets.split(','):
        names += DEV if s == 'dev' else NEWVAL if s == 'newval' else [s]
    missing = [n for n in names if not view_file(n).exists()]
    if missing:
        raise SystemExit(f'missing requested view sets: {missing}')
    ck_sha = sha256_file(a.ckpt)
    from rotlab.camp_final import check_score, is_final
    if any(is_final(n) for n in names):
        if a.tag or a.features:
            raise SystemExit('refusing: final sets are scored only under the default tag, without extra outputs')
        from rotlab.camp_final import sha as fresh_sha, verify_views
        ck_sha = fresh_sha(a.ckpt)                           # fresh bytes, no sidecar cache, for final scoring
        ck_pool0 = torch.load(a.ckpt, map_location='cpu', weights_only=False)['config'].get('pool')
        for cv in map(int, a.canvases.split(',')):
            check_score(ck_sha, cv, a.pool or ck_pool0)      # exact frozen stage tuple + opened + approved
        for n in names:
            if is_final(n):
                verify_views(n, view_file(n))
    vsha = {n: sha256_file(view_file(n)) for n in names}
    m = load_model(a.ckpt, a.pool)
    ck_pool = torch.load(a.ckpt, map_location='cpu', weights_only=False)['config'].get('pool')
    run = Path(a.ckpt).parent.name
    canvases = list(map(int, a.canvases.split(',')))
    from rotlab.camp_store import commit
    # STREAMING: one view set at a time; PNG bytes kept, decoded + letterboxed per batch of 256; only probabilities kept.
    for n in names:
        raw = view_file(n).read_bytes()
        if hashlib.sha256(raw).hexdigest() != vsha[n]:      # hash of the exact bytes used
            raise SystemExit(f'{n}: view cache bytes do not match the recorded hash')
        d = pickle.loads(raw); del raw
        pngs, th = d['png'], np.asarray(d['theta'], np.float64); ids = np.asarray(d['ids']) if 'ids' in d else None; del d
        for cv in canvases:
            tag = (a.tag or f'{run}{"-pool" + a.pool.replace(":", "x") if a.pool else ""}') + f'-c{cv}'
            t0 = time.time(); prs = []; feats = []
            hk = m.head[0].register_forward_hook(lambda mod, inp, out: feats.append(inp[0].float().cpu().numpy().astype(np.float16))) if a.features else None
            for i in range(0, len(pngs), 256):
                x = np.stack([letterbox(Image.open(io.BytesIO(b)).convert('RGB'), cv) for b in pngs[i:i + 256]])
                prs.append(probs(m, x)); del x
            if hk is not None:
                hk.remove()
            pr = np.concatenate(prs); p = decode(pr)
            rows = {f'{n}_prob': pr.astype(np.float16), f'{n}_pred': p, f'{n}_conf': confidence(pr, p), f'{n}_theta': th}
            if ids is not None:
                rows[f'{n}_ids'] = ids
            if a.features:
                rows[f'{n}_feat'] = np.concatenate(feats)
            commit(OUT, tag, rows, dict(ckpt_sha256=ck_sha, pool=a.pool or ck_pool, canvas=cv, decoder=DECODER, features='head input concat(CLS, mean patch) pre-LayerNorm' if a.features else None), {n: vsha[n]})
            print(tag, n, f'{time.time() - t0:.0f}s', int((circ_err(p, th) <= 10).sum()), '/', len(th), flush=True)
        del pngs


if __name__ == '__main__':
    main()
