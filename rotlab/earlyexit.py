#!/usr/bin/env python3
"""[lead-push] Early-exit RotNet: ONE DINOv2 ViT (arch l) with light 360-bin exit heads after intermediate blocks.
At inference the blocks run in order; after each exit block the exit's confidence (posterior mass within +-10 deg of its
decoded angle, the release confidence) is compared with that exit's threshold. Easy views stop there; hard views continue
through the SAME network from the tokens already computed (no recomputation, unlike the S2 -> L cascade). The final exit is
RotNet's own final norm + head, so a RotNet checkpoint (e.g. G7) loads unchanged and gives identical final outputs.

  train  (GPU, torchrun for DDP)  python -m rotlab.earlyexit train --run NAME --arch l --init CKPT --exits 6,12,18 --mix ... \
                                      [--freeze-steps N] [--exit-weights ..] [--kd-alpha 1] [--exit-grad 1]
  eval   (GPU)  python -m rotlab.earlyexit eval CKPT --canvases 168,224 --sets dev,newval
                -> camp-eval/<run>-eKK-c<canvas>.{npz,json}, one camp_store tag per exit block KK (final = e24 for ViT-L)
  policy (CPU)  python -m rotlab.earlyexit policy RUN --canvas 168 --sets dev,newval (--thresholds .9,.9,.9 | --sweep)
  strip  (CPU)  python -m rotlab.earlyexit strip CKPT OUT.pt      # final exit only = plain RotNet checkpoint (camp_eval etc.)
  export (CPU)  python -m rotlab.earlyexit export CKPT|random OUTDIR --canvas 168 [--exits 6,12,18] [--full]
  bench  (CPU)  python -m rotlab.earlyexit bench OUTDIR --thresholds .9,.9,.9 [--n 100] [--replay RUN --sets ...]

Training loss (per batch): final_w * CGD(final) + sum_k w_k * (CGD(exit_k) + alpha * T^2 * KL(p_final.detach() || p_exit_k)),
with an optional heads-only first phase (--freeze-steps: backbone + final head frozen in eval mode, exits trained on fixed
features; the final output stays exactly the init checkpoint's) followed by joint fine-tuning.
"""
from __future__ import annotations
import argparse, contextlib, copy, hashlib, io, itertools, json, math, os, pickle, random, re, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from rotlab.core import circ_err, decode, letterbox
from rotlab.evaluate import confidence
from rotlab.model import RotNet, param_groups
from rotlab.train import cgd_batch   # identical in train_ddp; the data pipeline (Views) is imported from train_ddp in train()

# ---------------------------------------------------------------- latency calibration (research/EFFICIENCY.md, REPORT.md)
# model-only FP32 p50, i7-1260P, ORT 1.24.4 CPU EP, 4 intra-op threads, batch 1. Per-block cost ~ total / depth.
CAL_MS = {'l': {140: 234.0, 168: 347.0, 224: 598.0}}
E2E_SCALE, E2E_ADD = 1.10, 2.0   # e2e ~ 1.10 x model p50 + ~2 ms (routed L224 stage 659 ~ 1.10 x 598; S2 path 58.5 ~ 1.10 x 52 + ~1)
P90_JITTER = 1.05                # the simulated per-view cost has no timing noise; measured unimodal p90 ~ 1.05-1.1 x mean
W_MEAN, W_P90 = 164.5, 196.6     # Woehrer 2026 canonical e2e, best workload set (means 171.2/164.5/173.0, p90 202.6/196.6/212.4)
DECODER = 'argmax + local circular mean over +-10 bins (core.decode); conf = mass within +-10 deg'


def ms_full(arch, canvas):
    """Calibrated full-depth model-only ms at a square canvas (token-linear interpolation between calibration points)."""
    cal = CAL_MS.get(arch)
    if not cal:
        raise SystemExit(f'no latency calibration for arch {arch!r}; pass --ms-full')
    if canvas in cal:
        return cal[canvas]
    tok = lambda c: (c // 14) ** 2 + 1
    xs = sorted(cal)
    if not xs[0] <= canvas <= xs[-1]:
        raise SystemExit(f'canvas {canvas} outside the calibrated range {xs}; pass --ms-full')
    return float(np.interp(tok(canvas), [tok(c) for c in xs], [cal[c] for c in xs]))


# ---------------------------------------------------------------- model
def _key(k):
    return f'b{k:02d}'


class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s):
        ctx.s = s
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.s, None


class ExitHead(nn.Module):
    """Fresh per-token LayerNorm -> concat(CLS, mean patch) -> LayerNorm -> Linear -> GELU -> Linear(bins).
    Same shape as the final path (backbone.norm -> pool -> RotNet.head) but with its own parameters: the backbone's final norm
    is NOT shared (its affine is fitted to block-24 statistics, and exit gradients through it would move the final exit)."""

    def __init__(self, d, hidden, bins):
        super().__init__()
        self.norm = nn.LayerNorm(d, eps=1e-6)
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, hidden), nn.GELU(), nn.Linear(hidden, bins))

    def forward(self, t):
        t = self.norm(t)
        return self.head(torch.cat([t[:, 0], t[:, 1:].mean(1)], 1))


class EarlyExitRotNet(RotNet):
    """RotNet + exit heads after blocks `exits` (1-based block counts, each < depth). State-dict keys backbone.*/head.* are
    RotNet's; the new ones are exits.bKK.*.  forward(x) is RotNet.forward (final exit only);  forward(x, all_exits=True)
    returns [logits after block k for k in exits] + [final logits], all from one pass."""

    def __init__(self, exits=(6, 12, 18), exit_hidden=1024, exit_grad=1.0, img_size=224, drop_path=0.1, hidden=1024, bins=360,
                 pretrained=True, dynamic=False, arch='l'):
        super().__init__(img_size=img_size, drop_path=drop_path, hidden=hidden, bins=bins, pretrained=pretrained, dynamic=dynamic, arch=arch)
        n, d = len(self.backbone.blocks), self.backbone.embed_dim
        self.exit_at = tuple(sorted({int(k) for k in exits}))
        if not all(0 < k < n for k in self.exit_at):
            raise ValueError(f'exits must be block counts in [1, {n - 1}], got {exits}')
        self.depth, self.exit_hidden, self.exit_grad, self.grad_ckpt = n, exit_hidden, exit_grad, False
        self.exits = nn.ModuleDict({_key(k): ExitHead(d, exit_hidden, bins) for k in self.exit_at})

    @property
    def blocks_at(self):
        return list(self.exit_at) + [self.depth]

    def embed(self, x):
        bb = self.backbone
        t = bb.patch_embed(x); t = bb._pos_embed(t); t = bb.patch_drop(t)
        return bb.norm_pre(t)

    def run_blocks(self, t, lo, hi):
        for blk in self.backbone.blocks[lo:hi]:
            if self.grad_ckpt and torch.is_grad_enabled() and (t.requires_grad or blk.norm1.weight.requires_grad):
                t = torch.utils.checkpoint.checkpoint(blk, t, use_reentrant=False)
            else:
                t = blk(t)
        return t

    def final(self, t):
        t = self.backbone.norm(t)
        return self.head(torch.cat([t[:, 0], t[:, 1:].mean(1)], 1))

    def exit_logits(self, t, k):
        if k == self.depth:
            return self.final(t)
        if torch.is_grad_enabled() and self.exit_grad != 1.0:
            t = t.detach() if self.exit_grad == 0 else _GradScale.apply(t, self.exit_grad)
        return self.exits[_key(k)](t)

    def forward(self, x, all_exits=False):
        if not all_exits:
            return super().forward(x)
        if self.pool is not None:
            raise ValueError('token pooling is not supported with early exits')
        t, lo, outs = self.embed(x), 0, []
        for k in self.blocks_at:
            t = self.run_blocks(t, lo, k); lo = k
            outs.append(self.exit_logits(t, k))
        return outs

    def init_exits_from_final(self, only=None):
        """Warm start: copy the final norm into each exit's per-token norm and the final head into its MLP (needs
        exit_hidden == final hidden). Parameters stay separate."""
        for k in (only if only is not None else self.exit_at):
            e = self.exits[_key(k)]
            e.norm.load_state_dict(self.backbone.norm.state_dict()); e.head.load_state_dict(self.head.state_dict())


def ee_config(m: EarlyExitRotNet):
    return dict(exits=list(m.exit_at), exit_hidden=m.exit_hidden, depth=m.depth)


def load_init(model: EarlyExitRotNet, sd):
    """Load a RotNet or EarlyExitRotNet state dict. backbone.* and head.* must load completely; exit heads load when their
    block is in model.exit_at (others are ignored/left fresh). -> (exit blocks loaded, exit blocks fresh, ignored keys)."""
    own = model.state_dict()
    bad = [k for k in own if not k.startswith('exits.') and (k not in sd or sd[k].shape != own[k].shape)]
    bad += [k for k in sd if not k.startswith('exits.') and k not in own]
    if bad:
        raise SystemExit(f'init does not match the backbone/head: {bad[:8]}')
    keep = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}
    ignored = sorted(k for k in sd if k not in keep)
    model.load_state_dict(keep, strict=False)
    loaded = [k for k in model.exit_at if all(f'exits.{_key(k)}.{n}' in keep for n, _ in model.exits[_key(k)].named_parameters())]
    return loaded, [k for k in model.exit_at if k not in loaded], ignored


def build_from_ckpt(ck, weights='ema', img_size=None, exits=None, dynamic=True):
    """EarlyExitRotNet from a saved checkpoint dict (EE or plain RotNet). img_size != saved: pos-embed resampled (static
    export). exits: subset of the checkpoint's exits (default all)."""
    cfg = ck['config']; ee = cfg.get('early_exit') or {}
    sd = dict(ck[weights] if weights in ck else ck['model'])
    ex = tuple(exits) if exits is not None else tuple(ee.get('exits', ()))
    if not ex:
        raise SystemExit('checkpoint has no exit heads; pass --exits (they would be untrained)')
    img = img_size or cfg['img_size']
    if img != cfg['img_size']:
        from timm.layers import resample_abs_pos_embed
        sd['backbone.pos_embed'] = resample_abs_pos_embed(sd['backbone.pos_embed'], (img // 14, img // 14), num_prefix_tokens=1)
    m = EarlyExitRotNet(exits=ex, exit_hidden=ee.get('exit_hidden', 1024), img_size=img, pretrained=False, dynamic=dynamic,
                        arch=cfg.get('arch', 'l'), hidden=cfg.get('hidden', 1024))
    loaded, fresh, _ = load_init(m, sd)
    if fresh and ee:
        raise SystemExit(f'exits {fresh} are not in the checkpoint (trained exits: {ee.get("exits")})')
    return m.eval(), cfg


# ---------------------------------------------------------------- loss / phases
def ee_loss(outs, theta, sigma=6.0, weights=None, final_weight=1.0, alpha=1.0, temp=1.0):
    """outs = [exit logits..., final logits]. -> (loss, parts); parts = {'final': cgd, 'cgd': [...], 'kl': [...]} as detached
    0-d tensors (no host sync; convert when logging)."""
    tgt = cgd_batch(theta, sigma)
    zf = outs[-1].float()
    ce = lambda z: torch.sum(-tgt * F.log_softmax(z, 1), 1).mean()
    cf = ce(zf)
    loss = final_weight * cf
    parts = dict(final=cf.detach(), cgd=[], kl=[])
    lpt = F.log_softmax(zf.detach() / temp, 1); pt = lpt.exp()
    weights = [1.0] * (len(outs) - 1) if weights is None else weights
    for w, z in zip(weights, outs[:-1]):
        z = z.float(); c = ce(z)
        kl = torch.sum(pt * (lpt - F.log_softmax(z / temp, 1)), 1).mean() * temp * temp
        loss = loss + w * (c + alpha * kl)
        parts['cgd'].append(c.detach()); parts['kl'].append(kl.detach())
    return loss, parts


def set_phase(model: EarlyExitRotNet, heads_only):
    """heads_only: only exits.* train; backbone + final head frozen and in eval mode (no drop-path on the fixed features)."""
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith('exits.') or not heads_only)
    model.train()
    if heads_only:
        model.backbone.eval(); model.head.eval()
    return [p for p in model.parameters() if p.requires_grad]


def train(argv):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from rotlab.train_ddp import RUNS, Views

    ap = argparse.ArgumentParser(prog='python -m rotlab.earlyexit train')
    ap.add_argument('--run', required=True)
    ap.add_argument('--mix', required=True)
    ap.add_argument('--steps', type=int, default=16000)
    ap.add_argument('--batch', type=int, default=128, help='global batch (divided over DDP ranks)')
    ap.add_argument('--lr', type=float, default=1e-4, help='top backbone block (layer-wise decay below)')
    ap.add_argument('--head-lr', type=float, default=5e-4, help='final head')
    ap.add_argument('--exit-lr', type=float, default=0.0, help='exit heads (default: --head-lr)')
    ap.add_argument('--lld', type=float, default=0.8)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--warmup', type=int, default=500, help='linear warm-up steps, restarted at each phase')
    ap.add_argument('--sigma', type=float, default=6.0)
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--canvas', type=int, default=0, help='input canvas if different from img-size (pos-embed interpolated)')
    ap.add_argument('--drop-path', type=float, default=0.1)
    ap.add_argument('--ema', type=float, default=0.9995)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--save-every', type=int, default=4000)
    ap.add_argument('--keep-milestones', action='store_true', help='save ckpt-NNNNNN.pt at every --save-every (no overwrite)')
    ap.add_argument('--init', type=Path, help='RotNet (e.g. G7) or early-exit checkpoint; its EMA weights')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--arch', default='l', help='rotlab.model.ARCH key (s, b, l)')
    ap.add_argument('--maxarea-p', type=float, default=0.0, help='DIAGNOSTIC ONLY, never for release candidates (owner decision 2026-09-23)')
    ap.add_argument('--data-frac', type=float, default=1.0)
    ap.add_argument('--multires', default='', help='e.g. 140,168,224: each step trains at one of these square sizes (batch resized)')
    ap.add_argument('--grad-ckpt', action='store_true')
    ap.add_argument('--exits', default='6,12,18', help='intermediate exit blocks (1-based); the final exit is always added')
    ap.add_argument('--exit-weights', default='', help='one loss weight per intermediate exit (default 1 each)')
    ap.add_argument('--final-weight', type=float, default=1.0)
    ap.add_argument('--kd-alpha', type=float, default=1.0, help='per-exit weight of KL(final posterior, detached || exit)')
    ap.add_argument('--kd-temp', type=float, default=1.0)
    ap.add_argument('--exit-hidden', type=int, default=1024)
    ap.add_argument('--exit-init', choices=['fresh', 'final'], default='fresh', help='new exit heads: fresh, or copies of the final norm+head')
    ap.add_argument('--exit-grad', type=float, default=1.0, help='scale of exit-loss gradients entering the backbone (0 = detached)')
    ap.add_argument('--freeze-steps', type=int, default=0, help='phase 1: the first N steps train only the exit heads')
    ap.add_argument('--cpu-smoke', action='store_true', help=argparse.SUPPRESS)   # tests only: run the loop on CPU
    a = ap.parse_args(argv)
    from rotlab.model import ARCH
    if a.arch not in ARCH:
        raise SystemExit(f'unknown arch {a.arch}')
    dev = 'cpu' if a.cpu_smoke else 'cuda'
    amp = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if dev == 'cuda' else contextlib.nullcontext
    ddp = 'LOCAL_RANK' in os.environ and int(os.environ.get('WORLD_SIZE', '1')) > 1
    rank, world, lrank = 0, 1, 0
    if ddp:
        dist.init_process_group('nccl'); rank, world = dist.get_rank(), dist.get_world_size(); lrank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(lrank)
        assert a.batch % world == 0, 'global batch must divide by world size'
    main0 = rank == 0; gbatch = a.batch; a.batch = gbatch // world
    mix = {k: float(v) for k, v in (x.split('=') for x in a.mix.split(','))}
    canvas = a.canvas or a.img_size; sizes = [int(v) for v in a.multires.split(',')] if a.multires else []
    exits = [int(v) for v in a.exits.split(',')]
    w_exit = [float(v) for v in a.exit_weights.split(',')] if a.exit_weights else [1.0] * len(exits)
    assert len(w_exit) == len(exits), '--exit-weights needs one value per exit'
    exit_lr = a.exit_lr or a.head_lr; F_ = min(a.freeze_steps, a.steps)
    out = RUNS / a.run; out.mkdir(parents=True, exist_ok=True)
    if main0:
        (out / 'config.json').write_text(json.dumps(vars(a) | dict(mix=mix, world=world, global_batch=gbatch), default=str, indent=1))
    torch.manual_seed(a.seed); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    model = EarlyExitRotNet(exits=exits, exit_hidden=a.exit_hidden, exit_grad=a.exit_grad, img_size=a.img_size, drop_path=a.drop_path,
                            dynamic=True, arch=a.arch, pretrained=a.init is None)
    fresh = list(model.exit_at)
    if a.init:
        ck = torch.load(a.init, map_location='cpu', weights_only=False)
        loaded, fresh, ignored = load_init(model, ck['ema'] if 'ema' in ck else ck['model']); del ck
        print(json.dumps(dict(init=str(a.init), exits_loaded=loaded, exits_fresh=fresh, ignored_keys=len(ignored))), flush=True)
    if a.exit_init == 'final' and fresh:
        model.init_exits_from_final(fresh)
    model = model.to(dev)
    if a.grad_ckpt:
        model.grad_ckpt = True; model.backbone.set_grad_checkpointing(True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    groups = param_groups(model, a.lr, a.head_lr, a.wd, a.lld)
    for nd in (False, True):
        ps = [p for n, p in model.exits.named_parameters() if (p.ndim == 1 or 'norm' in n) == nd]
        groups.append(dict(params=ps, lr=exit_lr, lr_scale=exit_lr, weight_decay=0.0 if nd else a.wd))
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), fused=dev == 'cuda')
    base = [g['lr'] for g in opt.param_groups]

    ds = Views(mix, canvas, a.seed + 7919 * rank, frac=a.data_frac, maxarea_p=a.maxarea_p)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=dev == 'cuda', persistent_workers=True, prefetch_factor=4)
    cfg = dict(img_size=a.img_size, canvas=canvas, hidden=1024, buckets=False, arch=a.arch, multires=sizes, pool=None,
               early_exit=ee_config(model))
    log = (out / 'log.jsonl').open('a') if main0 else open(os.devnull, 'w')
    t0 = time.time(); seen = 0; hist = []

    def save(step, tag=None):
        if main0:
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), config=cfg, step=step, presentations=seen, args=vars(a)),
                       out / f'{tag or f"ckpt-{step:06d}"}.pt')

    def lr_factor(step):
        s, n = (step, F_) if step < F_ else (step - F_, a.steps - F_)
        return min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, n))))

    it = iter(dl); step = 0; phase = None; fwd = model; trainable = []
    while step < a.steps:
        heads_only = step < F_
        if heads_only != phase:
            if phase is True:
                save(step, 'phase1')
            phase = heads_only; trainable = set_phase(model, heads_only)
            fwd = DDP(model, device_ids=[lrank], broadcast_buffers=False) if (ddp and not heads_only) else model
            if main0:
                print(json.dumps(dict(step=step, phase='heads-only' if heads_only else 'joint',
                                      trainable=sum(p.numel() for p in trainable))), flush=True)
        x, th = next(it)
        x = x.to(dev, non_blocking=True); th = th.to(dev, non_blocking=True)
        if sizes:
            sz = random.Random(a.seed * 7919 + step).choice(sizes)
            if sz != x.shape[-1]:
                x = F.interpolate(x, size=(sz, sz), mode='bilinear', antialias=True, align_corners=False)
        f = lr_factor(step)
        for g, b in zip(opt.param_groups, base):
            g['lr'] = b * f
        with amp():
            outs = fwd(x, all_exits=True)
        loss, parts = ee_loss(outs, th, a.sigma, w_exit, a.final_weight, a.kd_alpha, a.kd_temp)
        opt.zero_grad(set_to_none=True); loss.backward()
        if ddp and heads_only:   # no DDP wrapper while the backbone is frozen: average the (small) exit-head grads by hand
            gs = [p.grad for p in trainable]
            flat = torch._utils._flatten_dense_tensors(gs); dist.all_reduce(flat); flat /= world
            for g_, s_ in zip(gs, torch._utils._unflatten_dense_tensors(flat, gs)):
                g_.copy_(s_)
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); step += 1; seen += len(x) * world
        with torch.no_grad():
            d = min(a.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        if step % 10 == 0:
            hist.append([float(v) for v in [loss.detach(), parts['final']] + parts['cgd'] + parts['kl']])
        if step % 200 == 0 and main0:
            el = time.time() - t0; h = np.mean(hist[-20:], 0); J = len(exits)
            rec = dict(step=step, phase='heads' if heads_only else 'joint', loss=round(h[0], 4), final=round(h[1], 4),
                       exit_cgd={k: round(v, 4) for k, v in zip(exits, h[2:2 + J])}, exit_kl={k: round(v, 4) for k, v in zip(exits, h[2 + J:])},
                       gn=round(float(gn), 3), img_s=round(seen / el, 1), min=round(el / 60, 1))
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
        if step % a.save_every == 0 and step != a.steps:
            save(step, 'last')
            if a.keep_milestones:
                save(step)
    save(step, 'final')
    if ddp:
        dist.barrier(); dist.destroy_process_group()


# ---------------------------------------------------------------- eval (all exits -> camp_store tags)
@torch.no_grad()
def exit_probs(m: EarlyExitRotNet, x, bs=128, device='cpu'):
    """x: (N,3,H,W) float32 numpy -> list over m.blocks_at of (N,360) float32 probabilities."""
    out = [[] for _ in m.blocks_at]
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(np.ascontiguousarray(x[i:i + bs])).to(device)
        with (torch.autocast('cuda', dtype=torch.bfloat16) if device != 'cpu' else contextlib.nullcontext()):
            zs = m(xb, all_exits=True)
        for o, z in zip(out, zs):
            o.append(z.float().softmax(1).cpu().numpy())
    return [np.concatenate(o) for o in out]


def exit_tag(run, k, canvas):
    return f'{run}-e{k:02d}-c{canvas}'


def eval_views(m, pngs, theta, ids, name, vsha, ck_sha, canvases, run, out, weights='ema', bs=128, device='cpu'):
    """Score one view set at each canvas; commit one camp_store tag per exit (same row keys as camp_eval)."""
    from PIL import Image
    from rotlab.camp_store import commit
    blocks = m.blocks_at; res = {}
    for cv in canvases:
        t0 = time.time(); prs = [[] for _ in blocks]
        for i in range(0, len(pngs), 256):
            x = np.stack([letterbox(Image.open(io.BytesIO(b)).convert('RGB'), cv) for b in pngs[i:i + 256]])
            for o, p in zip(prs, exit_probs(m, x, bs, device)):
                o.append(p)
            del x
        line = []
        for k, o in zip(blocks, prs):
            pr = np.concatenate(o); p = decode(pr)
            rows = {f'{name}_prob': pr.astype(np.float16), f'{name}_pred': p, f'{name}_conf': confidence(pr, p), f'{name}_theta': theta}
            if ids is not None:
                rows[f'{name}_ids'] = ids
            commit(out, exit_tag(run, k, cv), rows, dict(ckpt_sha256=ck_sha, pool=None, canvas=cv, decoder=DECODER, exit_block=k,
                                                         exits=list(blocks), weights=weights), {name: vsha})
            w10 = int((circ_err(p, theta) <= 10).sum()); res[(k, cv)] = w10; line.append(f'e{k:02d} {w10}')
        print(f'{run} {name} c{cv} {time.time() - t0:.0f}s', ' | '.join(line), '/', len(theta), flush=True)
    return res


def expand_sets(spec):
    from rotlab.camp_eval import DEV, NEWVAL
    names = []
    for s in spec.split(','):
        names += DEV if s == 'dev' else NEWVAL if s == 'newval' else [s]
    return names


def evaluate(argv):
    from rotlab.camp_eval import OUT, sha256_file, view_file
    from rotlab.camp_final import is_final
    ap = argparse.ArgumentParser(prog='python -m rotlab.earlyexit eval'); ap.add_argument('ckpt')
    ap.add_argument('--canvases', default='224'); ap.add_argument('--sets', default='dev'); ap.add_argument('--tag')
    ap.add_argument('--weights', default='ema', choices=['ema', 'model']); ap.add_argument('--bs', type=int, default=128)
    a = ap.parse_args(argv)
    names = expand_sets(a.sets)
    missing = [n for n in names if not view_file(n).exists()]
    if missing:
        raise SystemExit(f'missing requested view sets: {missing}')
    if any(is_final(n) for n in names):
        raise SystemExit('refusing: sealed final sets are scored only through camp_final/camp_eval (strip the checkpoint first)')
    vsha = {n: sha256_file(view_file(n)) for n in names}
    ck_sha = sha256_file(a.ckpt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m, _ = build_from_ckpt(torch.load(a.ckpt, map_location='cpu', weights_only=False), a.weights)
    m = m.to(device).eval()
    run = a.tag or Path(a.ckpt).parent.name; canvases = [int(v) for v in a.canvases.split(',')]
    for n in names:
        raw = view_file(n).read_bytes()
        if hashlib.sha256(raw).hexdigest() != vsha[n]:
            raise SystemExit(f'{n}: view cache bytes do not match the recorded hash')
        d = pickle.loads(raw); del raw
        ids = np.asarray(d['ids']) if 'ids' in d else None
        eval_views(m, d['png'], np.asarray(d['theta'], np.float64), ids, n, vsha[n], ck_sha, canvases, run, OUT, a.weights, a.bs, device)
        del d


# ---------------------------------------------------------------- policy simulator
def load_exits(out, run, canvas, sets, use=None):
    """Stored exits of RUN at CANVAS -> (blocks, conf (J,N), err (J,N), pred (J,N), set label per view). The final exit is
    always included; `use` selects intermediate exits."""
    from rotlab.camp_store import load
    pat = re.compile(re.escape(run) + r'-e(\d{2})-c' + str(canvas) + r'\.json$')
    found = sorted(int(m.group(1)) for f in Path(out).glob(f'{run}-e??-c{canvas}.json') if (m := pat.search(f.name)))
    if not found:
        raise SystemExit(f'no stored exits for {run} at canvas {canvas} in {out}')
    _, rec = load(out, exit_tag(run, found[-1], canvas))
    depth = rec['exits'][-1]
    if found[-1] != depth:
        raise SystemExit(f'final exit e{depth:02d} missing for {run} c{canvas}')
    blocks = [k for k in found if k != depth and (use is None or k in use)] + [depth]
    if use is not None and len(blocks) - 1 != len(set(use)):
        raise SystemExit(f'requested exits {sorted(use)} not all stored (stored: {found})')
    conf, err, pred, lab = [], [], [], None
    for k in blocks:
        rows, rec = load(out, exit_tag(run, k, canvas))
        miss = [s for s in sets if f'{s}_pred' not in rows]
        if miss:
            raise SystemExit(f'{exit_tag(run, k, canvas)}: sets not stored: {miss}')
        conf.append(np.concatenate([rows[f'{s}_conf'] for s in sets]))
        pred.append(np.concatenate([rows[f'{s}_pred'] for s in sets]))
        err.append(np.concatenate([circ_err(rows[f'{s}_pred'], rows[f'{s}_theta']) for s in sets]))
        lab = np.concatenate([np.full(len(rows[f'{s}_pred']), s) for s in sets])
    return blocks, np.stack(conf), np.stack(err), np.stack(pred), lab


def exit_index(conf, thresholds, pred=None, agree=False):
    """First intermediate exit j with conf[j] >= thresholds[j] (and, with agree, |pred[j] - pred[j-1]| <= 10 deg; the first
    exit cannot agree) else the final exit (last row)."""
    J, N = conf.shape
    ok = conf[:-1] >= np.asarray(thresholds, np.float64)[:, None]
    if agree:
        ok[0] = False
        if J > 2:
            ok[1:] &= circ_err(pred[1:-1], pred[:-2]) <= 10
    ok = np.vstack([ok, np.ones((1, N), bool)])
    return ok.argmax(0)


def cum_ms_of(blocks, ms_block, exit_ms=0.0):
    """Model-only ms to reach and read exit j: blocks[j] * ms_block + (j + 1) exit reads."""
    return np.asarray(blocks, np.float64) * ms_block + exit_ms * (np.arange(len(blocks)) + 1)


def summarize(ex, err, cum_ms, e2e_scale=E2E_SCALE, e2e_add=E2E_ADD):
    J, N = err.shape
    e = err[ex, np.arange(N)]; model = np.asarray(cum_ms)[ex]; e2e = e2e_scale * model + e2e_add
    q = lambda v: float(np.quantile(v, 0.9, method='inverted_cdf'))   # nearest-rank p90
    return dict(n=N, w10=int((e <= 10).sum()), w10_pct=round(100 * float((e <= 10).mean()), 3), gt90=int((e > 90).sum()),
                t150=int((e >= 150).sum()), shares=[round(float((ex == j).mean()), 4) for j in range(J)],
                model_mean=round(float(model.mean()), 1), model_p90=round(q(model), 1),
                e2e_mean=round(float(e2e.mean()), 1), e2e_p90=round(q(e2e), 1))


def simulate(conf, err, blocks, thresholds, ms_block=None, cum_ms=None, pred=None, agree=False, exit_ms=0.0, **kw):
    """Policy arithmetic on stored exits. conf/err: (J,N) for exit blocks `blocks` (final last). Cost per view is the
    cumulative model-only ms of its exit (cum_ms, or blocks * ms_block + exit reads)."""
    cum = np.asarray(cum_ms, np.float64) if cum_ms is not None else cum_ms_of(blocks, ms_block, exit_ms)
    ex = exit_index(conf, thresholds, pred, agree)
    return ex, summarize(ex, err, cum, **kw)


def sweep(conf, err, blocks, grid, cum_ms, pred=None, agree=False, prefix=(), max_combos=200000):
    """Every per-exit threshold combination from `grid` (inf = exit disabled) -> list of (thresholds, summary). `prefix`: fixed
    thresholds for leading rows (e.g. a first-stage model), not swept."""
    J = conf.shape[0] - 1 - len(prefix)
    if len(grid) ** J > max_combos:
        raise SystemExit(f'{len(grid)}^{J} combinations > {max_combos}; use a coarser --grid or fewer --exits')
    return [(tuple(prefix) + thr, simulate(conf, err, blocks, tuple(prefix) + thr, cum_ms=cum_ms, pred=pred, agree=agree)[1])
            for thr in itertools.product(grid, repeat=J)]


def best_policy(results, max_mean=W_MEAN, max_p90=W_P90, jitter=P90_JITTER):
    ok = [(t, s) for t, s in results if s['e2e_mean'] < max_mean and s['e2e_p90'] * jitter < max_p90]
    return max(ok, key=lambda r: (r[1]['w10'], -r[1]['e2e_mean'])) if ok else None


def pareto(results):
    """Non-dominated (w10 up, e2e_mean down) points, cheapest first."""
    front, best = [], -1
    for t, s in sorted(results, key=lambda r: (r[1]['e2e_mean'], -r[1]['w10'])):
        if s['w10'] > best:
            front.append((t, s)); best = s['w10']
    return front


def _fmt_thr(t):
    return ','.join('off' if math.isinf(v) else f'{v:g}' for v in t)


def policy(argv):
    from rotlab.camp_eval import OUT
    ap = argparse.ArgumentParser(prog='python -m rotlab.earlyexit policy'); ap.add_argument('run')
    ap.add_argument('--canvas', type=int, default=224); ap.add_argument('--sets', default='dev,newval')
    ap.add_argument('--select-sets', default='', help='choose thresholds on these sets, report on --sets (default: same)')
    ap.add_argument('--exits', default='', help='subset of stored intermediate exits to use (default all)')
    ap.add_argument('--thresholds', default='', help='one per used intermediate exit (off = never exit there)')
    ap.add_argument('--sweep', action='store_true'); ap.add_argument('--grid', default='0.5,0.6,0.7,0.75,0.8,0.82,0.84,0.86,0.88,0.9,off',
                                                     help='per-exit threshold grid; conf saturates near 0.904 (the +-10 deg mass of the sigma-6 CGD target)')
    ap.add_argument('--agree', action='store_true', help='also require the exit to agree (<= 10 deg) with the previous exit')
    ap.add_argument('--arch', default='l'); ap.add_argument('--ms-full', type=float, default=0.0, help='override calibrated full-depth ms')
    ap.add_argument('--exit-ms', type=float, default=0.0, help='model-only ms per exit read (head + decode); see bench')
    ap.add_argument('--bench-json', help='use measured canonical cumulative exit costs from `bench` (same canvas/exits)')
    ap.add_argument('--max-mean', type=float, default=W_MEAN); ap.add_argument('--max-p90', type=float, default=W_P90)
    ap.add_argument('--first', help='optional first stage: a stored camp_store tag (e.g. the S2 tag at 224); views run it first and '
                    'enter the early-exit network only when its conf < --first-thr (hybrid S2 -> early-exit L)')
    ap.add_argument('--first-thr', type=float, default=0.836); ap.add_argument('--first-ms', type=float, default=52.0,
                                                                               help='first-stage model-only ms (S2@224 ~50-54)')
    ap.add_argument('--store', default=str(OUT)); ap.add_argument('--json', help='write the result here')
    a = ap.parse_args(argv)
    use = [int(v) for v in a.exits.split(',')] if a.exits else None
    sets = expand_sets(a.sets); sel = expand_sets(a.select_sets) if a.select_sets else sets
    blocks, conf, err, pred, lab = load_exits(a.store, a.run, a.canvas, sets, use)
    depth = blocks[-1]
    if a.bench_json:
        b = json.loads(Path(a.bench_json).read_text())
        if b['canvas'] != a.canvas or b['blocks'] != blocks:
            raise SystemExit(f'bench json is for canvas {b["canvas"]} exits {b["blocks"]}, need {a.canvas} {blocks}')
        cum = np.asarray(b['canonical_cum_ms']); src = f'measured ({a.bench_json})'
    else:
        full = a.ms_full or ms_full(a.arch, a.canvas); cum = cum_ms_of(blocks, full / depth, a.exit_ms)
        src = f'calibrated {full:g} ms / {depth} blocks'
    parse = lambda s: [math.inf if v == 'off' else float(v) for v in s.split(',')]
    res = dict(run=a.run, canvas=a.canvas, blocks=blocks, latency=src, cum_model_ms=[round(float(c), 1) for c in cum],
               e2e=f'{E2E_SCALE} x model + {E2E_ADD} ms; p90 gate x{P90_JITTER}', bars=dict(mean=a.max_mean, p90=a.max_p90))
    Jee = len(blocks)
    res['standalone'] = {f'e{k:02d}': summarize(np.full(conf.shape[1], j), err, cum) for j, k in enumerate(blocks)}
    prefix = []
    if a.first:   # prepend the first stage as row 0 (cost first_ms); early-exit rows then cost first_ms + their own
        from rotlab.camp_store import load
        fr, _ = load(a.store, a.first); er, _ = load(a.store, exit_tag(a.run, depth, a.canvas))
        for s in sets:
            if f'{s}_pred' not in fr or not np.array_equal(fr[f'{s}_theta'], er[f'{s}_theta']):
                raise SystemExit(f'{a.first}: set {s} missing or in a different view order')
        c0 = np.concatenate([fr[f'{s}_conf'] for s in sets]); p0 = np.concatenate([fr[f'{s}_pred'] for s in sets])
        e0 = np.concatenate([circ_err(fr[f'{s}_pred'], fr[f'{s}_theta']) for s in sets])
        conf, err, pred = np.vstack([c0[None], conf]), np.vstack([e0[None], err]), np.vstack([p0[None], pred])
        cum = np.concatenate([[a.first_ms], a.first_ms + cum]); prefix = [a.first_thr]
        res['standalone']['first'] = summarize(np.zeros(conf.shape[1], int), err, cum)
        res['first'] = dict(tag=a.first, thr=a.first_thr, ms=a.first_ms); res['cum_model_ms'] = [round(float(c), 1) for c in cum]
    J = conf.shape[0]
    print(f'{a.run} c{a.canvas} exits {blocks}{" after " + a.first if a.first else ""}  latency: {src}; cumulative model ms {res["cum_model_ms"]}')
    for k, s in res['standalone'].items():
        print(f'  {k} alone: w10 {s["w10"]}/{s["n"]} ({s["w10_pct"]}%) gt90 {s["gt90"]} t150 {s["t150"]}  e2e {s["e2e_mean"]} ms (always-on)')
    if a.sweep:
        m = np.isin(lab, sel)
        results = sweep(conf[:, m], err[:, m], blocks, parse(a.grid), cum, pred[:, m], a.agree, prefix)
        front = pareto(results); best = best_policy(results, a.max_mean, a.max_p90)
        print(f'pareto (selection sets {",".join(sel) if a.select_sets else "= report sets"}): thresholds | w10 | e2e mean/p90 | shares')
        for t, s in front[-15:]:
            print(f'  {_fmt_thr(t):>28} | {s["w10"]:>6} {s["w10_pct"]:>7}% | {s["e2e_mean"]:>6}/{s["e2e_p90"]:<6} | {s["shares"]}')
        res['pareto'] = [dict(thresholds=_fmt_thr(t), **s) for t, s in front]
        if best is None:
            print(f'no policy meets e2e mean < {a.max_mean} and p90 x{P90_JITTER} < {a.max_p90}'); thr = None
        else:
            thr = list(best[0]); print(f'best under the bars: {_fmt_thr(thr)}')
    else:
        thr = parse(a.thresholds) if a.thresholds else [math.inf] * (Jee - 1)
        if len(thr) != Jee - 1:
            raise SystemExit(f'--thresholds needs {Jee - 1} values (exits {blocks[:-1]})')
        thr = prefix + thr
    if thr is not None:
        res['thresholds'] = _fmt_thr(thr); res['sets'] = {}
        for s in sets + ['pooled']:
            m = lab == s if s != 'pooled' else np.ones(len(lab), bool)
            _, r = simulate(conf[:, m], err[:, m], blocks, thr, cum_ms=cum, pred=pred[:, m], agree=a.agree)
            fin = summarize(np.full(int(m.sum()), J - 1), err[:, m], cum)
            r['final_only_w10'] = fin['w10']; res['sets'][s] = r
            if a.first:
                r['cascade_first_to_final'] = simulate(conf[:, m], err[:, m], blocks, prefix + [math.inf] * (Jee - 1), cum_ms=cum)[1]
            print(f'  {s:>24}: w10 {r["w10"]}/{r["n"]} (final-only {fin["w10"]}, {r["w10"] - fin["w10"]:+d}) gt90 {r["gt90"]} t150 {r["t150"]} '
                  f'e2e {r["e2e_mean"]}/{r["e2e_p90"]} ms shares {r["shares"]}')
        p = res['sets']['pooled']
        res['beats_w'] = bool(p['e2e_mean'] < a.max_mean and p['e2e_p90'] * P90_JITTER < a.max_p90)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return res


# ---------------------------------------------------------------- strip (final exit only -> plain RotNet checkpoint)
def strip(argv):
    ap = argparse.ArgumentParser(prog='python -m rotlab.earlyexit strip'); ap.add_argument('ckpt'); ap.add_argument('out')
    a = ap.parse_args(argv)
    if Path(a.out).exists():
        raise SystemExit(f'refusing to overwrite {a.out}')
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    for w in ('model', 'ema'):
        if w in ck:
            ck[w] = {k: v for k, v in ck[w].items() if not k.startswith('exits.')}
    ck['config'] = {k: v for k, v in ck['config'].items() if k != 'early_exit'}
    ck['stripped_from'] = dict(path=str(a.ckpt), sha256=_sha(a.ckpt))
    torch.save(ck, a.out); print('stripped ->', a.out)


# ---------------------------------------------------------------- ONNX: one graph per segment, tokens passed between them
class Segment(nn.Module):
    """Blocks lo+1..hi (the first segment starts from the image). Outputs (tokens, prob) or, for the last, prob."""

    def __init__(self, net: EarlyExitRotNet, lo, hi):
        super().__init__(); self.net, self.lo, self.hi = net, lo, hi

    def forward(self, x):
        t = self.net.embed(x) if self.lo == 0 else x
        t = self.net.run_blocks(t, self.lo, self.hi)
        p = self.net.exit_logits(t, self.hi).softmax(1)
        return p if self.hi == self.net.depth else (t, p)


class _FinalOnly(nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net

    def forward(self, x):
        return self.net(x).softmax(1)


def _sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def export_segments(net: EarlyExitRotNet, outdir, canvas, full=False, ckpt_sha=None):
    """Static-canvas ONNX FP32 (opset 18) segment graphs + manifest.json (and optionally the unsegmented final-only graph)."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True); net = net.eval()
    segs, lo = [], 0
    x = torch.randn(1, 3, canvas, canvas)
    with torch.no_grad():
        for k in net.blocks_at:
            last = k == net.depth; f = outdir / f'seg-{lo:02d}-{k:02d}.onnx'
            inp = x if lo == 0 else net.run_blocks(net.embed(x), 0, lo)
            iname = 'image' if lo == 0 else 'tokens_in'; onames = ['prob'] if last else ['tokens', 'prob']
            torch.onnx.export(Segment(net, lo, k), inp, str(f), input_names=[iname], output_names=onames, opset_version=18, dynamo=False,
                              dynamic_axes={n: {0: 'batch'} for n in [iname] + onames})
            segs.append(dict(file=f.name, lo=lo, hi=k, input=iname, outputs=onames, sha256=_sha(f))); lo = k
        man = dict(canvas=canvas, depth=net.depth, blocks=net.blocks_at, segments=segs, ckpt_sha256=ckpt_sha, decoder=DECODER,
                   confidence='posterior mass within +-10 deg of the decoded angle', opset=18)
        if full:
            f = outdir / 'full.onnx'
            torch.onnx.export(_FinalOnly(net), x, str(f), input_names=['image'], output_names=['prob'], opset_version=18, dynamo=False,
                              dynamic_axes={'image': {0: 'batch'}, 'prob': {0: 'batch'}})
            man['full'] = dict(file=f.name, sha256=_sha(f))
    (outdir / 'manifest.json').write_text(json.dumps(man, indent=1))
    return man


def export(argv):
    ap = argparse.ArgumentParser(prog='python -m rotlab.earlyexit export'); ap.add_argument('ckpt', help="checkpoint or 'random'")
    ap.add_argument('outdir'); ap.add_argument('--canvas', type=int, default=224)
    ap.add_argument('--exits', default='', help='subset of the checkpoint exits to cut at (random: default 6,12,18)')
    ap.add_argument('--arch', default='l'); ap.add_argument('--weights', default='ema', choices=['ema', 'model'])
    ap.add_argument('--exit-hidden', type=int, default=1024); ap.add_argument('--full', action='store_true', help='also the unsegmented graph')
    a = ap.parse_args(argv)
    ex = [int(v) for v in a.exits.split(',')] if a.exits else None
    if a.ckpt == 'random':
        torch.manual_seed(0)
        net = EarlyExitRotNet(exits=ex or (6, 12, 18), exit_hidden=a.exit_hidden, img_size=a.canvas, pretrained=False, arch=a.arch)
        sha = None
    else:
        net, _ = build_from_ckpt(torch.load(a.ckpt, map_location='cpu', weights_only=False), a.weights, a.canvas, ex, dynamic=False)
        sha = _sha(a.ckpt)
    man = export_segments(net, a.outdir, a.canvas, a.full, sha)
    print(json.dumps(dict(outdir=a.outdir, blocks=man['blocks'], files=[s['file'] for s in man['segments']])))


class EarlyExitORT:
    """CPU runtime over the segment graphs: run segments in order, stop at the first exit whose confidence passes."""

    def __init__(self, outdir, threads=4):
        import onnxruntime as ort
        self.ort = ort; outdir = Path(outdir)
        self.man = json.loads((outdir / 'manifest.json').read_text())
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
        so.add_session_config_entry('session.intra_op.allow_spinning', '0')
        self.sess = [ort.InferenceSession(str(outdir / s['file']), so, providers=['CPUExecutionProvider']) for s in self.man['segments']]
        self.blocks = [s['hi'] for s in self.man['segments']]
        self.full = ort.InferenceSession(str(outdir / self.man['full']['file']), so, providers=['CPUExecutionProvider']) if 'full' in self.man else None

    def __call__(self, x, thresholds, conf_override=None, agree=False, stop_at=None):
        """x: (1,3,H,W) float32. -> (prob (1,360), exit block, conf, pred). conf_override[j] replaces the computed confidence
        in the decision (latency benchmarks with random weights); stop_at forces the exit index."""
        v = self.ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(x, np.float32)); name = 'image'; prev = None
        for j, s in enumerate(self.sess):
            last = j == len(self.sess) - 1
            if last:
                prob = s.run_with_ort_values(['prob'], {name: v})[0].numpy()
            else:
                v, pv = s.run_with_ort_values(['tokens', 'prob'], {name: v}); prob = pv.numpy(); name = 'tokens_in'
            pred = decode(prob); conf = float(confidence(prob, pred)[0])
            c = conf if conf_override is None else conf_override[j]
            ok = c >= thresholds[j] if not last else True
            if agree and not last:
                ok = ok and prev is not None and float(circ_err(pred, prev)[0]) <= 10
            if stop_at is not None:
                ok = j == stop_at or last
            if ok:
                return prob, self.blocks[j], conf, float(pred[0])
            prev = pred


def synthetic_conf(n, J, seed=0, a0=0.5, slope=2.5, b=1.5):
    """Correlated synthetic confidences (J exits incl. final): logit_j = a0 + slope*(j+1)/J + b*z + 0.5*e (z per view)."""
    r = np.random.default_rng(seed); z = r.standard_normal(n)
    lg = a0 + slope * (np.arange(J)[:, None] + 1) / J + b * z[None] + 0.5 * r.standard_normal((J, n))
    return 1 / (1 + np.exp(-lg))


def bench(argv):
    ap = argparse.ArgumentParser(prog='python -m rotlab.earlyexit bench'); ap.add_argument('outdir')
    ap.add_argument('--thresholds', default='0.85,0.85,0.85', help='one per intermediate exit (off = never)')
    ap.add_argument('--n', type=int, default=100, help='synthetic views for the policy run'); ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--threads', type=int, default=4); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--syn', default='0.5,2.5,1.5', help='synthetic confidence model a0,slope,b (see synthetic_conf)')
    ap.add_argument('--replay', help='use stored confidences of this eval RUN (same canvas/exits) instead of synthetic draws')
    ap.add_argument('--sets', default='dev'); ap.add_argument('--store', default='')
    ap.add_argument('--agree', action='store_true'); ap.add_argument('--arch', default='l'); ap.add_argument('--json')
    a = ap.parse_args(argv)
    rt = EarlyExitORT(a.outdir, a.threads); cv = rt.man['canvas']; blocks = rt.blocks; J = len(blocks)
    thr = [math.inf if v == 'off' else float(v) for v in a.thresholds.split(',')]
    assert len(thr) == J - 1, f'--thresholds needs {J - 1} values'
    x = np.random.default_rng(a.seed).standard_normal((1, 3, cv, cv)).astype(np.float32)
    for _ in range(3):   # warm-up
        rt(x, thr, stop_at=J - 1)
        if rt.full is not None:
            rt.full.run(None, {'image': x})
    # 1) cumulative cost of each exit depth (and the unsegmented graph), round-robin so background load hits all equally
    per = [[] for _ in range(J)]; full = []
    for _ in range(a.rounds):
        for j in range(J):
            t0 = time.perf_counter(); rt(x, thr, stop_at=j); per[j].append((time.perf_counter() - t0) * 1000)
        if rt.full is not None:
            t0 = time.perf_counter(); rt.full.run(None, {'image': x}); full.append((time.perf_counter() - t0) * 1000)
    cum = [float(np.median(p)) for p in per]
    # same-round ratios to the unsegmented graph (robust to background-load drift, as in research/EFFICIENCY.md)
    den = np.array(full[:a.rounds]) if rt.full is not None else np.array(per[-1])
    ratio = [np.array(p) / den for p in per]
    # 2) the policy on confidence draws, interleaving one full-depth reference every 10 views
    if a.replay:
        from rotlab.camp_eval import OUT
        _, conf, _, _, _ = load_exits(a.store or OUT, a.replay, cv, expand_sets(a.sets), blocks[:-1])
        idx = np.random.default_rng(a.seed).choice(conf.shape[1], min(a.n, conf.shape[1]), replace=False); conf = conf[:, idx]
    else:
        conf = synthetic_conf(a.n, J, a.seed, *map(float, a.syn.split(',')))
    lat, ex = [], []
    for i in range(conf.shape[1]):
        t0 = time.perf_counter(); _, k, _, _ = rt(x, thr, conf_override=conf[:, i], agree=False)
        lat.append((time.perf_counter() - t0) * 1000); ex.append(blocks.index(k))
        if rt.full is not None and i % 10 == 9:
            t0 = time.perf_counter(); rt.full.run(None, {'image': x}); full.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat); ex = np.array(ex)
    q = lambda v: float(np.quantile(v, 0.9, method='inverted_cdf'))
    ref = float(np.median(full)) if full else cum[-1]
    try:
        cal = ms_full(a.arch, cv)
    except SystemExit:
        cal = None
    scale = (cal or ref) / ref   # canonical-equivalent: the unsegmented graph (or the full chain) == calibration
    ov = ratio[-1] - 1
    res = dict(canvas=cv, blocks=blocks, threads=a.threads, rounds=a.rounds, thresholds=a.thresholds, conf_source=a.replay or f'synthetic {a.syn}',
               raw_cum_ms=[round(c, 1) for c in cum], raw_full_ms=round(ref, 1),
               cum_over_full=[round(float(np.median(r)), 4) for r in ratio],
               segmentation_overhead_pct=dict(median=round(100 * float(np.median(ov)), 2), q25=round(100 * float(np.quantile(ov, .25)), 2),
                                              q75=round(100 * float(np.quantile(ov, .75)), 2)) if rt.full is not None else None,
               canonical_scale=round(scale, 4), canonical_cum_ms=[round(float(np.median(r)) * (cal or ref), 1) for r in ratio],
               calibrated_cum_ms=[round(cal * k / blocks[-1], 1) for k in blocks] if cal else None,
               shares=[round(float((ex == j).mean()), 3) for j in range(J)],
               policy_raw=dict(mean=round(float(lat.mean()), 1), p90=round(q(lat), 1)),
               policy_canonical=dict(mean=round(float(lat.mean()) * scale, 1), p90=round(q(lat) * scale, 1)),
               policy_from_ratios=dict(mean=round(float(np.mean(cc := np.array([np.median(r) for r in ratio])[ex] * (cal or ref))), 1),
                                       p90=round(q(cc), 1)),   # exit shares x same-round cumulative costs (load-robust)
               loadavg=os.getloadavg()[0])
    print(json.dumps(res))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return res


if __name__ == '__main__':
    cmds = dict(train=train, eval=evaluate, policy=policy, strip=strip, export=export, bench=bench)
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        raise SystemExit(__doc__)
    cmds[sys.argv[1]](sys.argv[2:])
