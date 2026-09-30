#!/usr/bin/env python3
"""[lead-push] Token-subset RotNet: E0 fill-free tokens, E1 Focus-L, E5 confidence-chosen budget (claude-lead/research/
EFFICIENCY.md). Same weights as RotNet; the only new parameter is a zero-init per-level scale embedding.

E0 (modes fill / fillmask): patch tokens whose 14-px cell is pure letterbox fill are dropped after patch-embed + pos-embed
   (CLS kept), so attention and the mean pool see content tokens only. 'fill' runs each distinct keep-mask of a batch as
   its own dense sub-batch (fast); 'fillmask' keeps the full grid and masks fill KEYS in every block and fill tokens in the
   pool (fill queries are computed and discarded). Both are mathematically exact (content-token outputs identical to
   dropping); they differ only by float reduction order. --sinks N keeps N fill tokens (register-like sinks).
E1 (mode focus): the full-depth backbone runs on a small multi-scale token set built from ONE letterbox render. Level l
   cells are 14*2^l px; a cell is area-downsampled to 14x14 and embedded with the model's own patch embed; its position
   embedding is the area-average of the native PE it covers, plus scale_emb[l] (zero-init, so zero-shot inputs stay in
   the pretrained PE manifold). Plans:
     quad    - quadtree: split the k0 most salient 56-px cells into 28-px cells, then k1 of those into native patches;
               non-overlapping cover, tokens = n56 + 3(k0+k1) (224: 16 + 3k, e.g. 34/40/49/58/64)
     glance  - all 56-px cells + the (budget - n56) most salient native patches (overlapping cover)
     uniform - image resized (bilinear, antialias) to G*14 px, G*G native-PE tokens: control arm U (--uniform G)
     full    - all native patches (== RotNet; parity reference)
   Saliency (per native patch; fill patches are never refined): attn = S2 last-block CLS->patch attention (mean over
   heads; one extra K projection), grad = S2 angle saliency |d log p(argmax)/dt_i . t_i| on the post-PE patch tokens (one
   S2 backward; GPU eval/training only, the teacher for a future distilled scorer), pixel = content-only finite-difference
   energy (no S2; always-on shape B), random.
E5: --budget-by-conf 0.70:49,1.01:34 picks the budget per view from S2's +-10 deg confidence (first thr with conf < thr).

  PYTHONPATH=/workspace/code ROTLAB_DATA=/workspace/rotation-data
  python -m rotlab.focus eval CKPT --canvases 224 --sets dev,newval --mode fill            # tag <run>-fill-c224
  python -m rotlab.focus eval CKPT --mode focus --plan quad --budget 40 --scorer attn --s2 S2.pt
  python -m rotlab.focus train --run NAME --init G7.pt --s2 S2.pt --scorer attn --budgets 34,40,49,58,64 --mix ...
  python -m rotlab.focus export CKPT|s|b|l --out FL.onnx --plan quad --budget 40 --scorer attn [--s2 S2.pt --s2-out S.onnx]
  python -m rotlab.focus bench --threads 4 --rounds 12      # random weights, CPU FP32 ORT, ratios vs L@224 / L@140
"""
from __future__ import annotations
import argparse, copy, hashlib, io, json, math, os, pickle, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from rotlab.core import FILL, MEAN, STD, circ_err, decode, letterbox
from rotlab.model import RotNet, param_groups

P = 14
FILL_NORM = ((np.asarray(FILL, np.float32) / 255 - MEAN) / STD).astype(np.float32)   # same float32 ops as core.letterbox
TOL = 1e-3          # normalised units (1 grey level ~ 0.017): absorbs resize rounding, never a real colour step
PLANS = ('full', 'fill', 'fillmask', 'quad', 'glance', 'uniform')


# ---------------------------------------------------------------- fill geometry
def fill_mask(x, tol=TOL, patch=P):
    """(B,3,H,W) normalised input -> (B, H/patch, W/patch) bool, True where EVERY pixel of the patch is within tol of the
    normalised fill colour in all channels. (Letterbox fill is bit-exact in the float32 eval path; content patches that
    are flat fill colour are indistinguishable from fill before the PE and are dropped too; eval counts them.)"""
    f = torch.as_tensor(FILL_NORM, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    notfill = ((x - f).abs().amax(1, keepdim=True) > tol).to(x.dtype)
    return F.max_pool2d(notfill, patch)[:, 0] == 0


def letterbox_fill_mask(w, h, canvas, patch=P):
    """Geometric reference for core.letterbox (same rounding): (g, g) bool, True where the patch is entirely padding."""
    s = min(canvas / w, canvas / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    x0, y0 = (canvas - nw) // 2, (canvas - nh) // 2
    e = np.arange(canvas // patch) * patch
    col = (e + patch <= x0) | (e >= x0 + nw)
    row = (e + patch <= y0) | (e >= y0 + nh)
    return row[:, None] | col[None, :]


def sink_mask(fill, n):
    """(B, N) fill flags -> (B, N) bool marking up to n fill tokens per sample (evenly spaced in raster order) to keep."""
    out = torch.zeros_like(fill)
    for b in range(len(fill)):
        i = fill[b].nonzero()[:, 0]
        if len(i):
            out[b, i[torch.linspace(0, len(i) - 1, min(n, len(i))).round().long()]] = True
    return out


def pixel_saliency(x, tol=TOL, patch=P):
    """Cheap S2-free scorer: per patch, mean |horizontal| + |vertical| finite difference (summed over channels) over pixel
    pairs that are both content, so the letterbox edge does not attract tokens. (B,3,H,W) -> (B, H/patch, W/patch)."""
    f = torch.as_tensor(FILL_NORM, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    c = ((x - f).abs().amax(1, keepdim=True) > tol).to(x.dtype)
    dx = (x[..., :, 1:] - x[..., :, :-1]).abs().sum(1, keepdim=True) * c[..., :, 1:] * c[..., :, :-1]
    dy = (x[..., 1:, :] - x[..., :-1, :]).abs().sum(1, keepdim=True) * c[..., 1:, :] * c[..., :-1, :]
    return F.avg_pool2d(F.pad(dx, (0, 1)) + F.pad(dy, (0, 0, 0, 1)), patch)[:, 0]


# ---------------------------------------------------------------- token selection (static counts: topk/gather only)
def quad_counts(budget, n2, split=0.5):
    """(k0, k1) = (# 56-px cells split, # 28-px cells split) for a quad budget n2 + 3(k0 + k1); k1 <= 4 k0."""
    s, r = divmod(budget - n2, 3)
    if r or s < 0:
        raise ValueError(f'quad budget must be {n2} + 3k tokens, got {budget}')
    k0 = min(n2, max(math.ceil(s * split), math.ceil(s / 5)))
    k1 = s - k0
    if k1 > 4 * k0:
        raise ValueError(f'quad budget {budget} exceeds the full grid')
    return k0, k1


def _norm_sal(s, fill):
    """Non-negative, per-sample max-normalised saliency with fill patches zeroed. (B, g, g) -> (B, g, g) float32."""
    s = s.float().clamp_min(0) * (~fill).float()
    return s / (s.flatten(1).amax(1).view(-1, 1, 1) + 1e-12)


def _tb(n, device):
    """Tiny index tie-break so torch and ONNX Runtime pick the same cells among equal scores (lower index wins)."""
    return -1e-6 * torch.arange(n, device=device, dtype=torch.float32) / n


def _children(p, gp, gc):
    """Row-major indices of the 2x2 children on grid gc (= 2 gp) of parent cells p on grid gp. (B, k) -> (B, 4k)."""
    r = torch.div(p, gp, rounding_mode='floor')
    base = 2 * r * gc + 2 * (p - r * gp)
    return torch.stack([base, base + 1, base + gc, base + gc + 1], 2).flatten(1)


def _complement(sel, n, k):
    """Ascending indices in [0, n) not in sel (B, n - k); static count k."""
    mark = torch.zeros(sel.shape[0], n, device=sel.device).scatter(1, sel, 1.0)
    return (mark * n + torch.arange(n, device=sel.device)).topk(k, 1, largest=False).indices


def select_quad(sal, fill, budget, split=0.5):
    """-> (B, budget) indices into the token bank [native g*g | 28-px (g/2)^2 | 56-px (g/4)^2] (quadtree leaves)."""
    B, g = sal.shape[0], int(sal.shape[-1])       # int(): shapes are traced tensors during ONNX export
    g1, g2 = g // 2, g // 4
    n0, n1, n2 = g * g, g1 * g1, g2 * g2
    k0, k1 = quad_counts(budget, n2, split)
    coarse = torch.arange(n2, device=sal.device).expand(B, -1)
    if k0 == 0:
        return n0 + n1 + coarse
    s = _norm_sal(sal, fill)[:, None]
    s1 = F.avg_pool2d(s, 2).flatten(1) + _tb(n1, sal.device)
    s2 = F.avg_pool2d(s, 4).flatten(1) + _tb(n2, sal.device)
    top0 = s2.topk(k0, 1).indices                                  # 56-px cells split
    mid = _children(top0, g2, g1)                                  # (B, 4 k0) their 28-px children
    parts = []
    if k1:
        j1 = s1.gather(1, mid).topk(k1, 1).indices
        parts.append(_children(mid.gather(1, j1), g1, g))          # native children of the split 28-px cells
        parts.append(n0 + mid.gather(1, _complement(j1, 4 * k0, 4 * k0 - k1)))
    else:
        parts.append(n0 + mid)
    if k0 < n2:
        parts.append(n0 + n1 + _complement(top0, n2, n2 - k0))
    return torch.cat(parts, 1)


def select_glance(sal, fill, budget):
    """-> (B, budget) indices: every 56-px cell + the (budget - n56) most salient non-fill native patches."""
    B, g = sal.shape[0], int(sal.shape[-1])
    n0, n1, n2 = g * g, (g // 2) ** 2, (g // 4) ** 2
    sc = (_norm_sal(sal, fill) - 2 * fill.float()).flatten(1) + _tb(n0, sal.device)
    fine = sc.topk(budget - n2, 1).indices
    return torch.cat([fine, n0 + n1 + torch.arange(n2, device=sal.device).expand(B, -1)], 1)


# ---------------------------------------------------------------- model
def patch_pe(bb, gh, gw):
    """(gh, gw, D) patch position embeddings exactly as timm's dynamic _pos_embed uses them for a gh x gw grid."""
    from timm.layers import resample_abs_pos_embed
    pe = resample_abs_pos_embed(bb.pos_embed, new_size=(gh, gw), old_size=bb.patch_embed.grid_size, num_prefix_tokens=1)
    return pe[0, 1:].reshape(gh, gw, -1)


class FocusNet(nn.Module):
    """RotNet backbone + head on a token subset (see module doc). forward(x, sal=None, budget=None) -> logits."""
    LEVELS = 3

    def __init__(self, net: RotNet, plan='full', budget=40, quad_split=0.5, uniform=7, pool='mean', sinks=0, tol=TOL):
        super().__init__()
        bb = net.backbone
        assert not bb.no_embed_class and bb.reg_token is None and bb.cls_token is not None, 'expects DINOv2-style ViT'
        assert net.pool is None, 'pooled RotNet checkpoints are not supported'
        assert plan in PLANS and pool in ('mean', 'area')
        self.net = net
        self.scale_emb = nn.Parameter(torch.zeros(self.LEVELS, bb.embed_dim))
        self.plan, self.budget, self.quad_split, self.uniform, self.pool, self.sinks, self.tol = plan, budget, quad_split, uniform, pool, sinks, tol
        self.grad_ckpt = False
        self.export = False          # batch-1 traceable fill path (NonZero) for ONNX
        self._pe = {}                # frozen PE grids (export only; never during training)

    def config(self):
        return dict(plan=self.plan, budget=self.budget, quad_split=self.quad_split, uniform=self.uniform, pool=self.pool,
                    sinks=self.sinks, tol=self.tol)

    def _grid_pe(self, gh, gw):
        if self.export:
            if (gh, gw) not in self._pe:
                self._pe[(gh, gw)] = patch_pe(self.net.backbone, gh, gw).detach()
            return self._pe[(gh, gw)]
        return patch_pe(self.net.backbone, gh, gw)

    def embed(self, x, level=0):
        """(B,3,H,W) -> (B, n, D) tokens of 14*2^level-px cells (row-major) incl. area-averaged PE + scale_emb[level]."""
        pe = self._grid_pe(int(x.shape[-2]) // P, int(x.shape[-1]) // P)
        f = 2 ** level
        if f > 1:   # exact area downsample; normalisation is a per-channel affine map, so pooling commutes with it
            x = F.avg_pool2d(x, f)
            pe = F.avg_pool2d(pe.permute(2, 0, 1)[None], f)[0].permute(1, 2, 0)
        pm = self.net.backbone.patch_embed
        e = pm.norm(pm.proj(x).flatten(2).transpose(1, 2))
        return e + pe.reshape(1, -1, pe.shape[-1]) + self.scale_emb[level]

    def encode(self, tok, weights=None, attn_mask=None):
        """(B, T, D) patch tokens (with PE) -> logits: CLS + its PE prepended, all blocks, final norm, head on
        concat(CLS, pooled patches). weights (B, T) gives a weighted mean pool; attn_mask (B,1,1,1+T) bool masks keys."""
        bb = self.net.backbone
        cls = (bb.cls_token + bb.pos_embed[:, :1]).expand(len(tok), -1, -1)
        t = bb.norm_pre(bb.patch_drop(bb.pos_drop(torch.cat([cls, tok], 1))))
        for blk in bb.blocks:
            if self.grad_ckpt and self.training:
                t = torch.utils.checkpoint.checkpoint(blk, t, attn_mask, use_reentrant=False)
            else:
                t = blk(t) if attn_mask is None else blk(t, attn_mask=attn_mask)
        t = bb.norm(t)
        p = t[:, 1:]
        f = p.mean(1) if weights is None else (p * weights[..., None]).sum(1) / weights.sum(1, keepdim=True)
        return self.net.head(torch.cat([t[:, 0], f], 1))

    # -- E0
    def keep_mask(self, x):
        fill = fill_mask(x, self.tol).flatten(1)
        keep = ~fill
        if self.sinks:
            keep = keep | sink_mask(fill, self.sinks)
        return keep | (keep.sum(1, keepdim=True) == 0)          # an all-fill input keeps everything

    def forward_fill(self, x):
        """Exact drop: each distinct keep-mask of the batch runs as its own dense sub-batch."""
        keep, tok = self.keep_mask(x), self.embed(x, 0)
        if self.export:              # batch 1, traced: dynamic token count via NonZero
            return self.encode(tok[:, keep[0].nonzero()[:, 0]])
        uniq, inv = torch.unique(keep.to(torch.uint8), dim=0, return_inverse=True)
        out = None
        for u in range(len(uniq)):
            sel = (inv == u).nonzero()[:, 0]
            o = self.encode(tok[sel][:, uniq[u].bool().nonzero()[:, 0]])
            out = o.new_empty(len(x), o.shape[1]) if out is None else out
            out[sel] = o
        return out

    def forward_fillmask(self, x):
        """Attention-mask variant: full grid, fill keys masked in every block, fill tokens excluded from the pool."""
        keep, tok = self.keep_mask(x), self.embed(x, 0)
        m = torch.cat([keep.new_ones(len(x), 1), keep], 1)[:, None, None, :]
        return self.encode(tok, weights=keep.to(tok.dtype), attn_mask=m)

    # -- E1
    def bank(self, x):
        """Token bank [native | 28 px | 56 px] (B, n0+n1+n2, D) and per-token area in native patches (n0+n1+n2,)."""
        toks = [self.embed(x, l) for l in range(self.LEVELS)]
        area = torch.cat([torch.full((t.shape[1],), 4.0 ** l, device=x.device) for l, t in enumerate(toks)])
        return torch.cat(toks, 1), area

    def select(self, x, sal, budget):
        g = int(x.shape[-1]) // P
        assert int(x.shape[-2]) == int(x.shape[-1]) and g % 4 == 0, 'focus plans need a square canvas that is a multiple of 56 px'
        sal = sal.reshape(len(x), g, g)
        fill = fill_mask(x, self.tol)
        if self.plan == 'quad':
            return select_quad(sal, fill, budget, self.quad_split)
        return select_glance(sal, fill, budget)

    def forward(self, x, sal=None, budget=None):
        if self.plan == 'full':
            return self.encode(self.embed(x, 0))
        if self.plan == 'fill':
            return self.forward_fill(x)
        if self.plan == 'fillmask':
            return self.forward_fillmask(x)
        if self.plan == 'uniform':
            g = self.uniform
            return self.encode(self.embed(F.interpolate(x, size=(g * P, g * P), mode='bilinear', antialias=True, align_corners=False), 0))
        bank, area = self.bank(x)
        idx = self.select(x, sal, budget or self.budget)
        tok = bank.gather(1, idx[..., None].expand(-1, -1, bank.shape[-1]))
        return self.encode(tok, weights=area[idx].to(tok.dtype) if self.pool == 'area' else None)


class S2Saliency(nn.Module):
    """Frozen small model -> (logits, saliency (B, gh, gw)) on the SAME render. kind: 'attn' (last-block CLS attention,
    mean over heads) or 'grad' (angle saliency; runs its own backward, so not for export)."""

    def __init__(self, net: RotNet, kind='attn'):
        super().__init__()
        assert kind in ('attn', 'grad')
        self.net, self.kind, self._y = net, kind, None
        net.backbone.blocks[-1].norm1.register_forward_hook(self._grab)

    def _grab(self, mod, inp, out):
        self._y = out

    def forward(self, x):
        gh, gw = int(x.shape[-2]) // P, int(x.shape[-1]) // P
        if self.kind == 'grad':
            return self._grad(x, gh, gw)
        logits = self.net(x)
        y, self._y = self._y, None
        a = self.net.backbone.blocks[-1].attn
        D, H = a.qkv.weight.shape[0] // 3, a.num_heads
        W, b = a.qkv.weight, a.qkv.bias
        q = F.linear(y[:, :1], W[:D], None if b is None else b[:D])            # CLS query only
        k = F.linear(y, W[D:2 * D], None if b is None else b[D:2 * D])
        B, N, _ = k.shape
        q = a.q_norm(q.view(B, 1, H, D // H).transpose(1, 2))
        k = a.k_norm(k.view(B, N, H, D // H).transpose(1, 2))
        w = ((q * a.scale) @ k.transpose(-2, -1)).float().softmax(-1)[:, :, 0, 1:].mean(1)
        return logits, w.reshape(B, gh, gw)

    def _grad(self, x, gh, gw):
        bb, net = self.net.backbone, self.net
        with torch.enable_grad():
            t = bb._pos_embed(bb.patch_embed(x)).detach().requires_grad_(True)
            u = bb.norm(bb.blocks(bb.norm_pre(bb.patch_drop(t))))
            logits = net.head(torch.cat([u[:, 0], u[:, 1:].mean(1)], 1))
            g, = torch.autograd.grad(F.log_softmax(logits.float(), 1).max(1).values.sum(), t)
        self._y = None
        return logits.detach(), (g * t).sum(-1)[:, 1:].abs().detach().float().reshape(len(x), gh, gw)


# ---------------------------------------------------------------- checkpoints
def load_ckpt(path, drop_path=0.0):
    """-> (FocusNet with checkpoint weights (EMA), config). Plain RotNet checkpoints get a zero scale_emb."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ck['config']
    if cfg.get('pool'):
        raise SystemExit(f'{path}: pooled checkpoints are not supported')
    net = RotNet(img_size=cfg['img_size'], drop_path=drop_path, hidden=cfg.get('hidden', 1024), pretrained=False, dynamic=True,
                 arch=cfg.get('arch', 's'))
    fc = cfg.get('focus')
    m = FocusNet(net, **(fc['model'] if fc else {}))
    (m if fc else net).load_state_dict(ck['ema'] if 'ema' in ck else ck['model'])
    return m, cfg


def load_s2(path, kind):
    m, cfg = load_ckpt(path)
    if cfg.get('focus'):
        raise SystemExit('the saliency model must be a plain RotNet checkpoint')
    return S2Saliency(m.net.eval().requires_grad_(False), kind)


def parse_policy(s):
    """'0.70:49,1.01:34' -> [(0.70, 49), (1.01, 34)] (ascending thresholds)."""
    pol = sorted((float(t), int(k)) for t, k in (p.split(':') for p in s.split(',')))
    return pol


def policy_budget(conf, pol):
    out = np.full(len(conf), pol[-1][1])
    for thr, k in reversed(pol):
        out[conf < thr] = k
    return out


# ---------------------------------------------------------------- eval (mirrors camp_eval; same caches/decoder/store)
def infer(m, s2, x, scorer, budget, policy, gen):
    """One letterboxed batch (on device) -> (probs float32 numpy (B, 360), per-view budget or None)."""
    from rotlab.evaluate import confidence
    ac = torch.autocast('cuda', dtype=torch.bfloat16) if x.is_cuda else torch.autocast('cpu', enabled=False)
    focus = m.plan in ('quad', 'glance')
    with torch.no_grad(), ac:
        sal, bud = None, (np.full(len(x), budget) if focus else None)
        if s2 is not None:
            lg, sal = s2(x)
            if policy and focus:
                p = lg.float().softmax(1).cpu().numpy()
                bud = policy_budget(confidence(p, decode(p)), policy)
        if scorer == 'pixel':
            sal = pixel_saliency(x, m.tol)
        elif scorer == 'random':
            sal = torch.rand(len(x), x.shape[-2] // P, x.shape[-1] // P, generator=gen).to(x.device)
        if bud is None:
            out = m(x).float()
        else:
            out = torch.empty(len(x), 360, device=x.device)
            for b in np.unique(bud):
                sel = torch.from_numpy(np.flatnonzero(bud == b)).to(x.device)
                out[sel] = m(x[sel], sal[sel], int(b)).float()
    return out.softmax(1).cpu().numpy(), bud


def cmd_eval(argv):
    from PIL import Image
    from rotlab.camp_eval import DECODER, DEV, NEWVAL, OUT, sha256_file, view_file
    from rotlab.camp_final import is_final
    from rotlab.camp_store import commit
    from rotlab.evaluate import confidence
    ap = argparse.ArgumentParser(prog='rotlab.focus eval')
    ap.add_argument('ckpt'); ap.add_argument('--canvases', default='224'); ap.add_argument('--sets', default='dev')
    ap.add_argument('--mode', default='fill', choices=['full', 'fill', 'fillmask', 'focus'])
    ap.add_argument('--sinks', type=int, default=0, help='fill modes: keep N fill tokens as attention sinks')
    ap.add_argument('--plan', choices=['quad', 'glance', 'uniform'], help='focus mode (default: checkpoint focus config, else quad)')
    ap.add_argument('--budget', type=int, help='focus patch tokens excl. CLS (default: checkpoint, else 40)')
    ap.add_argument('--budget-by-conf', default='', help='E5: e.g. 0.70:49,1.01:34 (needs --s2)')
    ap.add_argument('--scorer', choices=['attn', 'grad', 'pixel', 'random'], help='focus saliency (default: checkpoint, else attn)')
    ap.add_argument('--s2', type=Path, help='small-model checkpoint for attn/grad saliency and --budget-by-conf')
    ap.add_argument('--quad-split', type=float); ap.add_argument('--uniform', type=int); ap.add_argument('--pool', choices=['mean', 'area'])
    ap.add_argument('--tol', type=float, default=TOL); ap.add_argument('--tag'); ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args(argv)
    names = []
    for s in a.sets.split(','):
        names += DEV if s == 'dev' else NEWVAL if s == 'newval' else [s]
    final = any(is_final(n) for n in names)
    if final:   # FREEZE v2: only the shipped fill-drop mode, with the checkpoint's own sinks/tol, behind the camp_final gate
        ck_fd = torch.load(a.ckpt, map_location='cpu', weights_only=False)['config'].get('filldrop') or {}
        if a.mode != 'fill' or a.tag or a.budget_by_conf or a.sinks != ck_fd.get('sinks', 0) or a.tol != ck_fd.get('tol', TOL):
            raise SystemExit('refusing: final sets are scored only in the frozen fill mode (checkpoint sinks/tol, default tag)')
    missing = [n for n in names if not view_file(n).exists()]
    if missing:
        raise SystemExit(f'missing requested view sets: {missing}')
    if final:
        from rotlab.camp_final import check_score, sha as fresh_sha, verify_views
        ck_fresh = fresh_sha(a.ckpt)                         # fresh bytes, no sidecar cache
        for cv in map(int, a.canvases.split(',')):
            check_score(ck_fresh, cv, 'fill')                # exact frozen stage tuple + opened + approved
        for n in names:
            if is_final(n):
                verify_views(n, view_file(n))
    m, cfg = load_ckpt(a.ckpt)
    fc = (cfg.get('focus') or {})
    fm = fc.get('model', {})
    scorer = None
    if a.mode == 'focus':
        m.plan = a.plan or (fm.get('plan') if fm.get('plan') in ('quad', 'glance', 'uniform') else 'quad')
        m.budget = a.budget or fm.get('budget', 40)
        m.quad_split = a.quad_split if a.quad_split is not None else fm.get('quad_split', 0.5)
        m.uniform = a.uniform or fm.get('uniform', 7)
        m.pool = a.pool or fm.get('pool', 'mean')
        scorer = None if m.plan == 'uniform' else (a.scorer or fc.get('scorer', 'attn'))
        if m.plan == 'uniform' and a.budget_by_conf:
            raise SystemExit('--budget-by-conf does not apply to the uniform plan')
    else:
        m.plan, m.sinks = a.mode, a.sinks
        if a.budget_by_conf:
            raise SystemExit('--budget-by-conf needs --mode focus')
    m.tol = a.tol
    policy = parse_policy(a.budget_by_conf) if a.budget_by_conf else None
    need_s2 = scorer in ('attn', 'grad') or policy is not None
    if need_s2 and not a.s2:
        raise SystemExit('--s2 is required for attn/grad saliency and --budget-by-conf')
    dev = torch.device(a.device)
    m = m.to(dev).eval()
    s2 = load_s2(a.s2, 'grad' if scorer == 'grad' else 'attn').to(dev).eval() if need_s2 else None
    ck_sha = ck_fresh if final else sha256_file(a.ckpt)
    s2_sha = sha256_file(a.s2) if need_s2 else None
    vsha = {n: sha256_file(view_file(n)) for n in names}
    run = Path(a.ckpt).parent.name
    if a.mode == 'focus':
        blab = 'conf' + hashlib.sha256(a.budget_by_conf.encode()).hexdigest()[:6] if policy else str(m.budget)
        suffix = f'focus-uniform{m.uniform}' if m.plan == 'uniform' else f'focus-{m.plan}{blab}-{scorer}'
        suffix += ('-area' if m.pool == 'area' else '') + (f'-split{m.quad_split:g}' if m.plan == 'quad' and m.quad_split != 0.5 else '')
    else:
        suffix = {'full': 'tokfull', 'fill': 'fill', 'fillmask': 'fillmask'}[a.mode] + (f'sink{a.sinks}' if a.sinks else '')
    if a.tol != TOL:
        suffix += f'-tol{a.tol:g}'
    ident_focus = dict(mode=a.mode, **m.config(), scorer=scorer, budget_by_conf=a.budget_by_conf or None, s2_sha256=s2_sha, seed=a.seed,
                       code='claude-lead rotlab.focus')
    gen = torch.Generator().manual_seed(a.seed)
    for n in names:
        raw = view_file(n).read_bytes()
        if hashlib.sha256(raw).hexdigest() != vsha[n]:
            raise SystemExit(f'{n}: view cache bytes do not match the recorded hash')
        d = pickle.loads(raw); del raw
        pngs, th = d['png'], np.asarray(d['theta'], np.float64)
        ids = np.asarray(d['ids']) if 'ids' in d else None
        del d
        for cv in map(int, a.canvases.split(',')):
            tag = (a.tag or f'{run}-{suffix}') + f'-c{cv}'
            t0 = time.time(); prs, buds, ntok, geo_diff = [], [], [], 0
            for i in range(0, len(pngs), 256):
                ims = [Image.open(io.BytesIO(b)).convert('RGB') for b in pngs[i:i + 256]]
                xb = torch.from_numpy(np.stack([letterbox(im, cv) for im in ims]))
                for j in range(0, len(xb), a.bs):
                    x = xb[j:j + a.bs].to(dev)
                    pr, bud = infer(m, s2, x, scorer, m.budget, policy, gen)
                    prs.append(pr)
                    if bud is not None:
                        buds.append(bud)
                    if a.mode in ('fill', 'fillmask'):
                        fmk = fill_mask(x, a.tol).cpu().numpy()
                        ntok.append(m.keep_mask(x).sum(1).cpu().numpy())
                        geo = np.stack([letterbox_fill_mask(*im.size, cv) for im in ims[j:j + a.bs]])
                        geo_diff += int((fmk != geo).any((1, 2)).sum())
            pr = np.concatenate(prs); p = decode(pr)
            rows = {f'{n}_prob': pr.astype(np.float16), f'{n}_pred': p, f'{n}_conf': confidence(pr, p), f'{n}_theta': th}
            if ids is not None:
                rows[f'{n}_ids'] = ids
            if buds:
                rows[f'{n}_budget'] = np.concatenate(buds).astype(np.int16)
            if ntok:
                rows[f'{n}_ntok'] = np.concatenate(ntok).astype(np.int16)
            commit(OUT, tag, rows, dict(ckpt_sha256=ck_sha, pool='fill' if final else None, canvas=cv, decoder=DECODER, features=None, focus=ident_focus), {n: vsha[n]})
            extra = f' ntok_mean {np.mean(rows[f"{n}_ntok"]):.1f} geo_mismatch_views {geo_diff}' if ntok else ''
            extra += f' budget_mean {np.mean(rows[f"{n}_budget"]):.1f}' if buds else ''
            print(tag, n, f'{time.time() - t0:.0f}s', int((circ_err(p, th) <= 10).sum()), '/', len(th), extra, flush=True)
        del pngs


# ---------------------------------------------------------------- training (Focus-L arms F / P / U; data = train_ddp)
def focus_loss(logits, target, t_logits=None, alpha=0.7, temp=1.0):
    """CGD cross-entropy, optionally mixed with KL(teacher || student) on the 360-bin posteriors (as train.py --teacher)."""
    loss = torch.sum(-target * F.log_softmax(logits.float(), 1), 1).mean()
    if t_logits is not None:
        pt = F.softmax(t_logits.float() / temp, 1)
        kd = torch.sum(pt * (torch.log(pt + 1e-12) - F.log_softmax(logits.float() / temp, 1)), 1).mean() * temp * temp
        loss = (1 - alpha) * loss + alpha * kd
    return loss


def saliency_for(scorer, s2, x, tol, step_seed):
    if scorer in ('attn', 'grad'):
        return s2(x)[1]
    if scorer == 'pixel':
        return pixel_saliency(x, tol)
    if scorer == 'random':
        g = torch.Generator(device=x.device).manual_seed(step_seed)
        return torch.rand(len(x), x.shape[-2] // P, x.shape[-1] // P, generator=g, device=x.device)
    return None


def cmd_train(argv):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    try:
        from rotlab.train_ddp import RUNS, Views, cgd_batch
    except ImportError:      # local checkout without the lead copy
        from rotlab.train import RUNS, Views, cgd_batch
    ap = argparse.ArgumentParser(prog='rotlab.focus train')
    ap.add_argument('--run', required=True); ap.add_argument('--mix', required=True)
    ap.add_argument('--init', type=Path, required=True, help='RotNet (e.g. G7) or focus checkpoint; EMA weights are used')
    ap.add_argument('--steps', type=int, default=7813); ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2.5e-5); ap.add_argument('--head-lr', type=float, default=5e-4)
    ap.add_argument('--scale-lr', type=float, default=1e-4, help='zero-init per-level scale embedding (input layer; no layer decay)')
    ap.add_argument('--lld', type=float, default=0.8); ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--warmup', type=int, default=300); ap.add_argument('--sigma', type=float, default=6.0)
    ap.add_argument('--canvas', type=int, default=224); ap.add_argument('--drop-path', type=float, default=0.1)
    ap.add_argument('--ema', type=float, default=0.9995); ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--save-every', type=int, default=2000); ap.add_argument('--keep-milestones', action='store_true')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--data-frac', type=float, default=1.0)
    ap.add_argument('--plan', default='quad', choices=['quad', 'glance', 'uniform'])
    ap.add_argument('--budgets', default='34,40,49,58,64', help='per-step random budget (FlexiViT-style); one value = fixed K')
    ap.add_argument('--quad-split', type=float, default=0.5); ap.add_argument('--uniform', type=int, default=7)
    ap.add_argument('--pool', default='mean', choices=['mean', 'area'])
    ap.add_argument('--scorer', default='attn', choices=['attn', 'grad', 'pixel', 'random'])
    ap.add_argument('--s2', type=Path, help='frozen small model for attn/grad saliency')
    ap.add_argument('--teacher', type=Path, help='full-token RotNet teacher (EMA), run live on the identical view')
    ap.add_argument('--distill-alpha', type=float, default=0.5); ap.add_argument('--distill-temp', type=float, default=1.0)
    ap.add_argument('--grad-ckpt', action='store_true')
    a = ap.parse_args(argv)
    budgets = [int(v) for v in a.budgets.split(',')]
    ddp = 'LOCAL_RANK' in os.environ and int(os.environ.get('WORLD_SIZE', '1')) > 1
    rank, world, lrank = 0, 1, 0
    if ddp:
        dist.init_process_group('nccl'); rank, world = dist.get_rank(), dist.get_world_size(); lrank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(lrank)
        assert a.batch % world == 0, 'global batch must divide by world size'
    main0 = rank == 0; gbatch = a.batch; a.batch = gbatch // world
    mix = {k: float(v) for k, v in (x.split('=') for x in a.mix.split(','))}
    out = RUNS / a.run
    if main0:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'config.json').write_text(json.dumps(vars(a) | dict(mix=mix, world=world, global_batch=gbatch), default=str, indent=1))
    torch.manual_seed(a.seed); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    model, icfg = load_ckpt(a.init, drop_path=a.drop_path)
    model.plan, model.quad_split, model.uniform, model.pool = a.plan, a.quad_split, a.uniform, a.pool
    model.budget = 40 if 40 in budgets else budgets[0]
    if a.plan == 'quad':
        for b in budgets:
            quad_counts(b, (a.canvas // 56) ** 2, a.quad_split)   # fail before any GPU work
    model.grad_ckpt = a.grad_ckpt
    model = model.cuda().train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    fwd = DDP(model, device_ids=[lrank], broadcast_buffers=False, find_unused_parameters=a.plan == 'uniform') if ddp else model
    s2 = None
    if a.plan != 'uniform' and a.scorer in ('attn', 'grad'):
        if not a.s2:
            raise SystemExit('--s2 is required for attn/grad saliency')
        s2 = load_s2(a.s2, a.scorer).cuda().eval()
    teacher = None
    if a.teacher:
        teacher = load_ckpt(a.teacher)[0].cuda().eval().requires_grad_(False)
        teacher.plan = 'full'
    groups = param_groups(model.net, a.lr, a.head_lr, a.wd, a.lld) + [dict(params=[model.scale_emb], lr=a.scale_lr, weight_decay=0.0)]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), fused=True)
    base = [g['lr'] for g in opt.param_groups]
    trainable = [p for p in model.parameters() if p.requires_grad]
    ds = Views(mix, a.canvas, a.seed + 7919 * rank, frac=a.data_frac, maxarea_p=0.0)   # angle-independent crops only
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    cfg = dict(img_size=icfg['img_size'], canvas=a.canvas, hidden=icfg.get('hidden', 1024), buckets=False, arch=icfg.get('arch', 's'),
               multires=[], pool=None,
               focus=dict(model=model.config(), budgets=budgets, scorer=None if a.plan == 'uniform' else a.scorer,
                          s2=str(a.s2) if s2 is not None else None, init=str(a.init), teacher=str(a.teacher) if a.teacher else None))
    log = (out / 'log.jsonl').open('a') if main0 else open(os.devnull, 'w')
    t0 = time.time(); seen = 0; hist = []

    def save(step, tag=None):
        if main0:
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), config=cfg, step=step, presentations=seen, args=vars(a)),
                       out / f'{tag or f"ckpt-{step:06d}"}.pt')

    it = iter(dl); step = 0
    while step < a.steps:
        x, th = next(it)
        x = x.cuda(non_blocking=True); th = th.cuda(non_blocking=True)
        b = budgets[random.Random(a.seed * 7919 + step).randrange(len(budgets))]      # same on every rank
        lr_f = min(1.0, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps)))
        for g, lr0 in zip(opt.param_groups, base):
            g['lr'] = lr0 * lr_f
        with torch.autocast('cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                sal = saliency_for(a.scorer if a.plan != 'uniform' else None, s2, x, model.tol, a.seed * 100003 + step * world + rank)
                t_logits = teacher(x) if teacher is not None else None
            logits = fwd(x, sal, b)
        loss = focus_loss(logits, cgd_batch(th, a.sigma), t_logits, a.distill_alpha, a.distill_temp)
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); step += 1; seen += len(x) * world
        with torch.no_grad():
            dcy = min(a.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - dcy)
        if step % 10 == 0:
            hist.append(loss.item())
        if step % 200 == 0 and main0:
            el = time.time() - t0
            rec = dict(step=step, loss=round(float(np.mean(hist[-20:])), 4), gn=round(float(gn), 3), budget=b, img_s=round(seen / el, 1), min=round(el / 60, 1))
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
        if step % a.save_every == 0 and step != a.steps:
            save(step, 'last')
            if a.keep_milestones:
                save(step)
    save(step, 'final')
    if ddp:
        dist.barrier(); dist.destroy_process_group()


# ---------------------------------------------------------------- ONNX export + CPU micro-benchmark
class ProbGraph(nn.Module):
    """Export wrapper -> softmax probabilities. scorer 'pixel' computes saliency in-graph (one input: image); 'attn'
    takes S2's saliency as a second input (1, g, g); non-focus plans take the image only."""

    def __init__(self, m: FocusNet, scorer=None):
        super().__init__(); self.m, self.scorer = m, scorer

    def forward(self, x, sal=None):
        if self.scorer == 'pixel':
            sal = pixel_saliency(x, self.m.tol)
        return self.m(x, sal).softmax(1)


class S2Graph(nn.Module):
    def __init__(self, s2: S2Saliency):
        super().__init__(); self.s2 = s2

    def forward(self, x):
        lg, sal = self.s2(x)
        return lg.softmax(1), sal


def fix_topk_k(model):
    """ORT rejects a 0-D TopK K (torch 2.8 TorchScript export can emit one): reshape such constants to [1]."""
    import onnx
    from onnx import numpy_helper
    ks = {n.input[1] for n in model.graph.node if n.op_type == 'TopK'}
    for init in model.graph.initializer:
        if init.name in ks and len(init.dims) == 0:
            init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).reshape(1), init.name))
    for n in model.graph.node:
        if n.op_type == 'Constant' and n.output[0] in ks:
            for at in n.attribute:
                if at.name == 'value' and len(at.t.dims) == 0:
                    at.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(at.t).reshape(1)))
    return model


def export_onnx(module, inputs, names, path=None, outputs=('prob',)):
    """Trace-export (opset 18, like rotlab.export) -> serialized bytes (and optionally a file). Batch 1, static shapes."""
    import onnx
    buf = io.BytesIO()
    module.eval()
    gh, gw = int(inputs[0].shape[-2]) // P, int(inputs[0].shape[-1]) // P
    for mm in module.modules():
        if isinstance(mm, FocusNet):
            mm.export = True
            # lead 28 Sep: bake the grid PE OUTSIDE the trace (at non-native canvases, e.g. 112, timm's resample uses
            # antialiased bicubic, which has no ONNX opset-18 export); inside the trace it is then a constant.
            with torch.no_grad():
                mm._pe[(gh, gw)] = patch_pe(mm.net.backbone, gh, gw).detach().clone()
    with torch.no_grad():
        torch.onnx.export(module, tuple(inputs), buf, input_names=list(names), output_names=list(outputs), opset_version=18, dynamo=False)
    for mm in module.modules():
        if isinstance(mm, FocusNet):
            mm.export = False; mm._pe = {}
    model = fix_topk_k(onnx.load_from_string(buf.getvalue()))
    data = model.SerializeToString()
    if path:
        Path(path).write_bytes(data)
    return data


def ort_session(data, threads=4):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    so.add_session_config_entry('session.intra_op.allow_spinning', '0')
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(data, so, providers=['CPUExecutionProvider'])


def sample_input(canvas, aspect=4 / 3, seed=0):
    """A realistic letterboxed input: smooth random content of the given aspect (so fill modes and scorers see fill)."""
    from PIL import Image
    r = np.random.default_rng(seed)
    w = 320; h = max(8, round(w / aspect))
    im = Image.fromarray(r.integers(0, 255, (h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)).resize((w, h), Image.BICUBIC)
    return torch.from_numpy(letterbox(im, canvas))[None]


def cmd_export(argv):
    ap = argparse.ArgumentParser(prog='rotlab.focus export')
    ap.add_argument('model', help="checkpoint path, or s/b/l for random weights")
    ap.add_argument('--out', required=True); ap.add_argument('--canvas', type=int, default=224)
    ap.add_argument('--plan', default='quad', choices=list(PLANS)); ap.add_argument('--budget', type=int, default=40)
    ap.add_argument('--scorer', default='attn', choices=['attn', 'pixel']); ap.add_argument('--quad-split', type=float, default=0.5)
    ap.add_argument('--uniform', type=int, default=7); ap.add_argument('--pool', default='mean', choices=['mean', 'area'])
    ap.add_argument('--s2', help='also export the small model with a saliency output: checkpoint path or s'); ap.add_argument('--s2-out')
    a = ap.parse_args(argv)
    if a.plan == 'uniform':      # aten::_upsample_bilinear2d_aa has no opset-18 export
        raise SystemExit('uniform: deploy it as a plain RotNet at canvas G*14 (rotlab.export --canvas), letterboxed at that size')
    if a.model in ('s', 'b', 'l'):
        m = FocusNet(RotNet(img_size=224, pretrained=False, dynamic=True, arch=a.model))
    else:
        m = load_ckpt(a.model)[0]
    m.plan, m.budget, m.quad_split, m.uniform, m.pool = a.plan, a.budget, a.quad_split, a.uniform, a.pool
    focus = a.plan in ('quad', 'glance')
    x = sample_input(a.canvas)
    g = a.canvas // P
    sal = torch.rand(1, g, g)
    two = focus and a.scorer == 'attn'
    mod = ProbGraph(m.eval(), a.scorer if focus else None)
    with torch.no_grad():
        ref = mod(x, sal if two else None).numpy()
    data = export_onnx(mod, [x, sal] if two else [x], ['image', 'saliency'] if two else ['image'], a.out)
    got = ort_session(data).run(None, {'image': x.numpy(), **({'saliency': sal.numpy()} if two else {})})[0]
    print(json.dumps(dict(out=a.out, plan=a.plan, budget=a.budget if focus else None, scorer=a.scorer if focus else None,
                          inputs=['image', 'saliency'] if two else ['image'], max_abs_diff_vs_torch=float(np.abs(got - ref).max()))))
    if a.s2:
        s2 = S2Saliency(RotNet(img_size=224, pretrained=False, dynamic=True, arch='s')) if a.s2 == 's' else load_s2(a.s2, 'attn')
        sg = S2Graph(s2.eval())
        with torch.no_grad():
            rp, rs = sg(x)
        data = export_onnx(sg, [x], ['image'], a.s2_out or a.out.replace('.onnx', '') + '-s2sal.onnx', outputs=('prob', 'saliency'))
        gp, gs = ort_session(data).run(None, {'image': x.numpy()})
        print(json.dumps(dict(s2_out=a.s2_out, max_abs_diff_prob=float(np.abs(gp - rp.numpy()).max()), max_abs_diff_sal=float(np.abs(gs - rs.numpy()).max()))))


# canonical-equivalent anchors (FP32, i7-1260P, ORT, 4 threads, batch 1): L224 p50 598 ms (REPORT §2 decomposition);
# L140 ~234 ms (EFFICIENCY §2: L-tokens-49 / L140 = 0.543 <-> 127 ms); S224 ~52 ms.
ANCHORS = {'L224': 598.0, 'L140': 234.0}


def cmd_bench(argv):
    """Phased, paired timing: an L@140 anchor session stays resident; each target graph is built, timed round-robin
    against the anchor in the same rounds (rotating start), then freed (a ViT-L FP32 session holds ~2.8 GB). Ratios vs
    L@224 go through the L224 phase: r(X/L224) = r(X/L140) / r(L224/L140). Random weights: latency only."""
    import gc
    ap = argparse.ArgumentParser(prog='rotlab.focus bench')
    ap.add_argument('--threads', type=int, default=4); ap.add_argument('--rounds', type=int, default=12)
    ap.add_argument('--warmup', type=int, default=3); ap.add_argument('--reps', type=int, default=3, help='runs per graph per round')
    ap.add_argument('--arch', default='l'); ap.add_argument('--focus', default='quad40-attn,quad49-attn,quad49-pixel',
                    help='comma list of <plan><budget>-<scorer>, plan quad|glance, scorer attn|pixel')
    ap.add_argument('--small', action='store_true', help='also time S@224 plain / +attn saliency output / fill-free (4:3, 16:9)')
    ap.add_argument('--json', help='write the summary here')
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    x224, x140, sal = sample_input(224), sample_input(140), torch.rand(1, 16, 16)

    def build(name, module, inputs, names, outputs=('prob',)):
        t = time.time()
        data = export_onnx(module, inputs, names, outputs=outputs)
        s = ort_session(data, a.threads)
        print(f'built {name} ({len(data) / 1e6:.0f} MB, {time.time() - t:.0f}s)', flush=True)
        return s, {k: v.numpy() for k, v in zip(names, inputs)}

    def phase(graphs):
        names = list(graphs)
        for _ in range(a.warmup):
            for n in names:
                graphs[n][0].run(None, graphs[n][1])
        times, rmean = {n: [] for n in names}, {n: [] for n in names}
        for r in range(a.rounds):
            for n in names[r % len(names):] + names[:r % len(names)]:
                s, feeds = graphs[n]; ts = []
                for _ in range(a.reps):
                    t0 = time.perf_counter(); s.run(None, feeds); ts.append((time.perf_counter() - t0) * 1e3)
                times[n] += ts; rmean[n].append(float(np.mean(ts)))
        return times, rmean

    l140 = RotNet(img_size=140, pretrained=False, dynamic=False, arch=a.arch).eval()
    anchor = build('L140', ProbGraph(FocusNet(l140)), [x140], ['image'])
    del l140; gc.collect()
    big = RotNet(img_size=224, pretrained=False, dynamic=True, arch=a.arch).eval()
    targets = [('L224', lambda: (ProbGraph(FocusNet(big)), [x224], ['image']))]
    for spec in filter(None, a.focus.split(',')):
        pb, scorer = spec.split('-')
        plan = 'quad' if pb.startswith('quad') else 'glance'
        m = FocusNet(big, plan=plan, budget=int(pb[len(plan):]))
        two = scorer == 'attn'
        targets.append((f'FL-{spec}', lambda m=m, scorer=scorer, two=two: (
            ProbGraph(m, scorer), [x224, sal] if two else [x224], ['image', 'saliency'] if two else ['image'])))
    if a.small:
        sm = RotNet(img_size=224, pretrained=False, dynamic=True, arch='s').eval()
        targets.append(('S224', lambda: (ProbGraph(FocusNet(sm)), [x224], ['image'])))
        targets.append(('S224-attnsal', lambda: (S2Graph(S2Saliency(copy.deepcopy(sm))), [x224], ['image'], ('prob', 'saliency'))))
        for asp, lab in ((4 / 3, '4:3'), (16 / 9, '16:9')):
            targets.append((f'S224-fill-{lab}', lambda asp=asp: (ProbGraph(FocusNet(sm, plan='fill')), [sample_input(224, asp)], ['image'])))
            targets.append((f'S224-{lab}', lambda asp=asp: (ProbGraph(FocusNet(sm)), [sample_input(224, asp)], ['image'])))
    rows, anchor_t = {}, []
    for name, make in targets:
        g = build(name, *make())
        times, rmean = phase({'L140': anchor, name: g})
        del g; gc.collect()
        t = np.array(times[name]); anchor_t += times['L140']
        rows[name] = dict(mean=round(float(t.mean()), 1), p90=round(float(np.percentile(t, 90)), 1),
                          ratio_L140=round(float(np.mean(np.array(rmean[name]) / np.array(rmean['L140']))), 4),
                          anchor_mean_in_phase=round(float(np.mean(times['L140'])), 1))
        print(name, json.dumps(rows[name]), f'loadavg {os.getloadavg()[0]:.1f}', flush=True)
    del big
    at = np.array(anchor_t)
    rows = dict(L140=dict(mean=round(float(at.mean()), 1), p90=round(float(np.percentile(at, 90)), 1), ratio_L140=1.0), **rows)
    r224 = rows['L224']['ratio_L140']
    for v in rows.values():
        v['ratio_L224'] = round(v['ratio_L140'] / r224, 4)
        v['canon_ms_via_L224'] = round(v['ratio_L224'] * ANCHORS['L224'], 1)
        v['canon_ms_via_L140'] = round(v['ratio_L140'] * ANCHORS['L140'], 1)
    summary = dict(threads=a.threads, rounds=a.rounds, reps=a.reps, loadavg=[round(v, 2) for v in os.getloadavg()],
                   anchors_ms=ANCHORS, rows=rows)
    print('| graph | mean ms | p90 ms | / L140 (paired) | / L224 | canonical ms via L224 / via L140 |\n|---|---|---|---|---|---|')
    for n, v in rows.items():
        print(f"| {n} | {v['mean']} | {v['p90']} | {v['ratio_L140']} | {v['ratio_L224']} | {v['canon_ms_via_L224']} / {v['canon_ms_via_L140']} |")
    print(json.dumps(summary))
    if a.json:
        Path(a.json).write_text(json.dumps(summary, indent=1))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cmds = dict(eval=cmd_eval, train=cmd_train, export=cmd_export, bench=cmd_bench)
    if not argv or argv[0] not in cmds:
        raise SystemExit(f'usage: python -m rotlab.focus {{{"|".join(cmds)}}} ...')
    cmds[argv[0]](argv[1:])


if __name__ == '__main__':
    main()
