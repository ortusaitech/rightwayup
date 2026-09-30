#!/usr/bin/env python3
"""[lead-push] Fill-drop training (E0, trained): RotNet trained with its letterbox-fill patch tokens DROPPED, exactly as
`rotlab.focus eval --mode fill` scores it, plus the eval passthrough, the deployment ONNX export and a CPU benchmark.

Forward (FillDropNet = focus.FocusNet, plan 'fill'; same weights as RotNet, no new trainable parameter):
  patch embed + position embedding (timm's dynamic grid, exactly) -> drop every patch token whose 14-px cell is pure
  letterbox fill (focus.fill_mask: every pixel within tol of the normalised fill colour) -> CLS + kept tokens through all
  blocks -> final norm -> head on concat(CLS, mean over KEPT patch tokens). --sinks N also keeps N fill tokens per image
  (evenly spaced in raster order; the same tokens focus.sink_mask picks), which are attended and pooled, as in focus eval.
  An image without fill equals RotNet exactly. An all-fill image keeps everything (focus semantics).
  --batching group (default): each distinct keep-mask of the batch runs as its own dense sub-batch. Training letterboxes
    come in a handful of mask shapes (about 4-6 per batch of 128), so this means a few dense passes with flash
    attention and no padding waste.
  --batching pad: the kept tokens of every image are gathered to the front (raster order) and padded to the batch's max
    kept count. Padded KEYS are masked in every block (SDPA bool mask) and padded tokens are excluded from the pool. It
    is one dense pass, but it saves nothing whenever an image of the batch is square.
  Both are exact: equal to per-image processing up to float reduction order (test_filldrop).
Checkpoints are PLAIN RotNet state dicts (model / ema) plus config.filldrop, so every existing tool loads them:
  rotlab.focus eval --mode fill [--sinks N] (the intended score), rotlab.camp_eval (full-token score), soups, export.

  PYTHONPATH=/workspace/code ROTLAB_DATA=/workspace/rotation-data
  python -m rotlab.filldrop train --run NAME --arch s --init CKPT --mix ... [any rotlab.train_ddp flag] [--sinks N]
  torchrun --nproc_per_node N -m rotlab.filldrop train --run-name NAME ...  # DDP (torchrun's own parser rejects --run)
  python -m rotlab.filldrop eval CKPT --canvases 224 --sets dev,newval      # = focus eval --mode fill --sinks <ckpt's>
  python -m rotlab.filldrop export CKPT|s|b|l --out FD.onnx [--sinks N]     # batch-1 ONNX, dynamic token count
  python -m rotlab.filldrop bench --archs s,l --aspects 4:3,16:9,1:1        # CPU ORT: fill-drop vs plain, same process
"""
from __future__ import annotations
import argparse, copy, gc, json, math, os, random, re, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rotlab.focus import TOL, FocusNet, ProbGraph, export_onnx, fill_mask, ort_session, sample_input

MAX_SINKS = 15      # sink_select replicates CPU torch.linspace's scalar path, which covers steps < 16 (tested exhaustively)
ARCHS = ('s', 'b', 'l', 's4', 's5', 's6', 's7', 's8', 's9', 's10')   # Pico lane 29 Sep: truncated DINOv2-S (rotlab.model.DEPTH)


# ---------------------------------------------------------------- keep mask
def sink_select(fill, n):
    """(B, N) bool fill flags -> (B, N) bool: up to n fill tokens per image to keep as sinks. These are the SAME tokens
    focus.sink_mask picks: ranks round(torch.linspace(0, nf - 1, k)) among the fill tokens in raster order, where
    k = min(n, nf). The float32 arithmetic of CPU linspace is replicated (start + step*i below halfway, end - step*(k-1-i)
    from halfway on; round half to even). Vectorised, with no host sync, and ONNX-traceable."""
    f = fill.to(torch.int64)
    nf = f.sum(1, keepdim=True)                                            # (B, 1)
    rank = f.cumsum(1) - 1                                                 # rank of each token among the fill tokens
    k = torch.clamp(nf, max=n)
    j = torch.arange(n, device=fill.device, dtype=torch.int64)[None]       # (1, n)
    end = (nf - 1).to(torch.float32)
    step = end / (k - 1).clamp(min=1).to(torch.float32)
    val = torch.where(j < torch.div(k, 2, rounding_mode='floor'), step * j.to(torch.float32),
                      end - step * (k - 1 - j).to(torch.float32))
    val = torch.where(k == 1, torch.zeros_like(val), val)                  # linspace(.., steps=1) == [start]
    tgt = torch.where(j < k, val.round().to(torch.int64), torch.full_like(val, -1, dtype=torch.int64))
    return fill & (rank[:, :, None] == tgt[:, None, :]).any(-1)


class FillDropNet(FocusNet):
    """RotNet (arch s/b/l) whose forward drops letterbox-fill patch tokens exactly (module doc). forward(x) -> logits."""

    def __init__(self, net, sinks=0, tol=TOL, batching='group'):
        assert getattr(net, 'arch', 's') in ARCHS, f'fill-drop supports DINOv2 s/b/l, not {net.arch}'
        assert 0 <= sinks <= MAX_SINKS, f'--sinks must be in [0, {MAX_SINKS}]'
        assert batching in ('group', 'pad')
        super().__init__(net, plan='fill', sinks=sinks, tol=tol)
        self.scale_emb.requires_grad_(False)     # stays exactly 0 (added to level-0 tokens): the weights remain plain RotNet
        self.batching = batching
        self.stats = None                        # (keep (B, N) bool, number of dense passes) of the last forward

    def keep_mask(self, x):
        fill = fill_mask(x, self.tol).flatten(1)
        keep = ~fill
        if self.sinks:
            keep = keep | sink_select(fill, self.sinks)
        return keep | (keep.sum(1, keepdim=True) == 0)          # an all-fill input keeps everything

    def forward(self, x, sal=None, budget=None):
        return self.forward_fill(x) if (self.export or self.batching == 'group') else self.forward_pad(x)

    def forward_fill(self, x):
        """'group': every distinct keep-mask of the batch is one dense sub-batch (export: batch 1, NonZero)."""
        keep, tok = self.keep_mask(x), self.embed(x, 0)
        if self.export:
            return self.encode(tok[:, keep[0].nonzero()[:, 0]])
        uniq, inv = torch.unique(keep.to(torch.uint8), dim=0, return_inverse=True)
        self.stats = (keep.detach(), len(uniq))
        outs, sels = [], []
        for u in range(len(uniq)):
            sel = (inv == u).nonzero()[:, 0]
            outs.append(self.encode(tok[sel][:, uniq[u].bool().nonzero()[:, 0]]))
            sels.append(sel)
        if len(outs) == 1:
            return outs[0]
        return torch.cat(outs)[torch.argsort(torch.cat(sels))]

    def forward_pad(self, x):
        """'pad': one dense pass over [CLS | kept tokens (raster order) | padding]; padded keys masked in every block."""
        keep, tok = self.keep_mask(x), self.embed(x, 0)
        self.stats = (keep.detach(), 1)
        n = keep.sum(1)
        L = int(n.max())
        order = torch.sort((~keep).to(torch.uint8), dim=1, stable=True).indices[:, :L]      # kept first, raster order
        valid = torch.arange(L, device=x.device)[None] < n[:, None]
        t = tok.gather(1, order[..., None].expand(-1, -1, tok.shape[-1]))
        if bool(valid.all()):
            return self.encode(t)
        m = torch.cat([valid.new_ones(len(x), 1), valid], 1)[:, None, None, :]
        return self.encode(t, weights=valid.to(t.dtype), attn_mask=m)


def ckpt_config(path):
    """Config dict of a checkpoint without materialising its weights (mmap)."""
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)['config']


def build_net(arch, img_size=224, drop_path=0.0, pretrained=False, init=None, hidden=1024):
    """rotlab.model_x RotNet (s/b/l byte-identical to rotlab.model.RotNet); init: checkpoint path (EMA weights)."""
    from rotlab.model_x import RotNet
    net = RotNet(img_size=img_size, drop_path=drop_path, hidden=hidden, pretrained=pretrained, dynamic=True, arch=arch)
    if init is not None:
        from rotlab.model import truncate_state_dict
        ck = torch.load(init, map_location='cpu', weights_only=False)
        net.load_state_dict(truncate_state_dict(ck['ema'] if 'ema' in ck else ck['model'], arch))   # Pico: full-depth warm start
        del ck
    return net


# ---------------------------------------------------------------- training (train_ddp CLI + data pipeline)
def train_parser():
    ap = argparse.ArgumentParser(prog='rotlab.filldrop train', description='rotlab.train_ddp with fill tokens dropped')
    ap.add_argument('--run', '--run-name', dest='run', required=True,
                    help='run dir name; under torchrun use --run-name (torchrun rejects --run as an ambiguous prefix of --run-path)')
    ap.add_argument('--mix', required=True)
    ap.add_argument('--steps', type=int, default=16000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-4)       # top backbone block
    ap.add_argument('--head-lr', type=float, default=5e-4)
    ap.add_argument('--lld', type=float, default=0.8)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--sigma', type=float, default=6.0)
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--canvas', type=int, default=0, help='input canvas if different from img-size (pos-embed interpolated)')
    ap.add_argument('--drop-path', type=float, default=0.1)
    ap.add_argument('--ema', type=float, default=0.9995)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--save-every', type=int, default=4000)
    ap.add_argument('--init', type=Path, help='resume weights (EMA if present) from a plain RotNet checkpoint')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--buckets', action='store_true', help='aspect-bucketed canvases (L 168x294, S 224, P 294x168)')
    ap.add_argument('--arch', default='s', choices=ARCHS)
    ap.add_argument('--teacher', type=Path, help='distil from this checkpoint (its EMA weights), run live per batch on FULL tokens')
    ap.add_argument('--distill-alpha', type=float, default=0.7, help='weight of teacher KL vs ground-truth CGD')
    ap.add_argument('--distill-temp', type=float, default=1.0)
    ap.add_argument('--teacher-canvas', type=int, default=0, help='lead: teacher sees the batch at this canvas (render at --canvas >= it; the student trains on --multires sizes resized from the canvas); 0 = teacher sees the student input')
    ap.add_argument('--maxarea-p', type=float, default=0.0, help='DIAGNOSTIC ONLY (angle-leaking crop), as train_ddp')
    ap.add_argument('--scrub', default='', help='lead: trace-scrubbed rendering, e.g. ss=1,decoy=0.35,cam=0.4 (rotlab/scrub.py)')
    ap.add_argument('--data-frac', type=float, default=1.0)
    ap.add_argument('--multires', default='', help='e.g. 112,140,168,196,224: each step trains at one of these square sizes')
    ap.add_argument('--grad-ckpt', action='store_true', help='activation checkpointing per block')
    ap.add_argument('--pack-eval', action='store_true', help='in-training eval on the legacy benchmark pack (fill-drop forward)')
    ap.add_argument('--train-blocks', type=int, default=0, help='if >0, train only the last N backbone blocks + final norm + head')
    ap.add_argument('--pool', default='', help='refused: static token pooling needs the full grid')
    ap.add_argument('--keep-milestones', action='store_true', help='save ckpt-NNNNNN.pt at every --save-every (no overwrite)')
    ap.add_argument('--sinks', type=int, default=0, help=f'keep N fill tokens per image as attention sinks (0..{MAX_SINKS})')
    ap.add_argument('--tol', type=float, default=TOL, help='fill detection tolerance (normalised units; eval must use the same)')
    ap.add_argument('--batching', default='group', choices=['group', 'pad'])
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='cpu: CPU smoke tests only (fp32, gloo DDP)')
    return ap


def cmd_train(argv):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from rotlab.train_ddp import RUNS, Views, cgd_batch
    from rotlab.model_x import SPECS, param_groups
    a = train_parser().parse_args(argv)
    if a.pool:
        raise SystemExit('--pool (static token pooling) is incompatible with dropping fill tokens')
    if not 0 <= a.sinks <= MAX_SINKS:
        raise SystemExit(f'--sinks must be in [0, {MAX_SINKS}]')
    if a.buckets and a.multires:
        raise SystemExit('--buckets with --multires would resize bucket canvases to squares; pick one')
    if a.init:
        icfg = ckpt_config(a.init)
        if icfg.get('focus') or icfg.get('pool'):
            raise SystemExit(f'{a.init}: focus / pooled checkpoints are not valid inits (plain RotNet only)')
        from rotlab.model import DEPTH
        ia = icfg.get('arch', 's')   # Pico: a truncated s<k> may warm-start from a full-depth 's' checkpoint
        if not (ia == a.arch or (ia == 's' and a.arch in DEPTH)) or icfg.get('img_size', 224) != a.img_size:
            raise SystemExit(f'{a.init}: arch/img_size {icfg.get("arch", "s")}/{icfg.get("img_size")} != --arch {a.arch} --img-size {a.img_size}')
    cuda = a.device == 'cuda'
    ddp = 'LOCAL_RANK' in os.environ and int(os.environ.get('WORLD_SIZE', '1')) > 1
    rank, world, lrank = 0, 1, 0
    if ddp:
        dist.init_process_group('nccl' if cuda else 'gloo'); rank, world = dist.get_rank(), dist.get_world_size(); lrank = int(os.environ['LOCAL_RANK'])
        if cuda:
            torch.cuda.set_device(lrank)
        assert a.batch % world == 0, 'global batch must divide by world size'
    dev = torch.device(f'cuda:{lrank}' if cuda else 'cpu')
    main0 = rank == 0; gbatch = a.batch; a.batch = gbatch // world
    mix = {k: float(v) for k, v in (x.split('=') for x in a.mix.split(','))}
    canvas = a.canvas or a.img_size; sizes = [int(v) for v in a.multires.split(',')] if a.multires else []
    if any(s % 14 for s in sizes + [canvas]) and not a.buckets:
        raise SystemExit('canvas and --multires sizes must be multiples of 14 (fill tokens are 14-px cells)')
    out = RUNS / a.run
    if main0:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'config.json').write_text(json.dumps(vars(a) | dict(mix=mix, world=world, global_batch=gbatch), default=str, indent=1))
    torch.manual_seed(a.seed); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    net = build_net(a.arch, a.img_size, a.drop_path, pretrained=a.init is None, init=a.init)
    if a.train_blocks:
        nb = len(net.backbone.blocks)
        for n_, p_ in net.backbone.named_parameters():
            p_.requires_grad = n_.startswith('norm.') or (n_.startswith('blocks.') and int(n_.split('.')[1]) >= nb - a.train_blocks)
        print('trainable params', sum(p_.numel() for p_ in net.parameters() if p_.requires_grad), flush=True)
    model = FillDropNet(net, sinks=a.sinks, tol=a.tol, batching=a.batching).to(dev)
    model.grad_ckpt = a.grad_ckpt
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    fwd = DDP(model, device_ids=[lrank] if cuda else None, broadcast_buffers=False) if ddp else model
    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    teacher = None
    if a.teacher:
        tc = ckpt_config(a.teacher)
        teacher = build_net(tc.get('arch', 's'), tc['img_size'], hidden=tc.get('hidden', 1024), init=a.teacher).to(dev).eval().requires_grad_(False)
    opt = torch.optim.AdamW(param_groups(model.net, a.lr, a.head_lr, a.wd, a.lld), betas=(0.9, 0.999), fused=cuda)
    base = [g['lr'] for g in opt.param_groups]

    lkw = dict(num_workers=a.workers, pin_memory=cuda, persistent_workers=a.workers > 0)
    if a.buckets:   # each worker assembles whole same-bucket batches; rank-offset seed (train_ddp reuses a.seed on every rank)
        ds = Views(mix, canvas, a.seed + 7919 * rank, bucket_batch=a.batch, frac=a.data_frac)
        dl = DataLoader(ds, batch_size=None, prefetch_factor=2 if a.workers else None, **lkw)
    else:
        ds = Views(mix, canvas, a.seed + 7919 * rank, frac=a.data_frac, maxarea_p=a.maxarea_p, scrub=a.scrub)
        dl = DataLoader(ds, batch_size=a.batch, prefetch_factor=4 if a.workers else None, **lkw)
    pc = None
    if a.pack_eval and main0:
        from rotlab.evaluate import PackCache
        pc = PackCache(canvas, a.buckets)
    cfg = dict(img_size=a.img_size, canvas=canvas, hidden=net.head[1].out_features, buckets=a.buckets, arch=a.arch, multires=sizes,
               pool=None, backbone=dict(timm=SPECS[a.arch].timm, revision=SPECS[a.arch].revision, sha256=SPECS[a.arch].sha256,
                                        status=SPECS[a.arch].status),
               filldrop=dict(sinks=a.sinks, tol=a.tol, batching=a.batching, init=str(a.init) if a.init else None,
                             teacher=str(a.teacher) if a.teacher else None, code='rotlab.filldrop',
                             eval='python -m rotlab.filldrop eval CKPT  (== rotlab.focus eval --mode fill' +
                                  (f' --sinks {a.sinks}' if a.sinks else '') + (f' --tol {a.tol:g}' if a.tol != TOL else '') + ')'))
    log = (out / 'log.jsonl').open('a') if main0 else open(os.devnull, 'w')
    t0 = time.time(); seen = 0; hist = []; kept = []; groups = []
    if main0:
        print(json.dumps(dict(run=a.run, arch=a.arch, canvas=canvas, multires=sizes, sinks=a.sinks, batching=a.batching, world=world,
                              global_batch=gbatch, device=str(dev))), flush=True)

    def evaluate(step):
        if pc is None:
            return
        from rotlab.evaluate import predict, report, table
        rows = {}
        for name, m in (('model', model), ('ema', ema)):
            res, _ = report(pc, predict(m, pc.x, pc=pc), f'{a.run} step {step} {name} (fill-drop)')
            rows[name] = res
        model.train()
        rec = dict(step=step, presentations=seen, eval=rows, minutes=round((time.time() - t0) / 60, 2))
        log.write(json.dumps(rec) + '\n'); log.flush()
        (out / f'eval-{step:06d}.md').write_text(table(rows['model']) + '\n\n' + table(rows['ema']))

    def save(step, tag=None):
        if not main0:
            return
        torch.save(dict(model=model.net.state_dict(), ema=ema.net.state_dict(), config=cfg, step=step, presentations=seen, args=vars(a)),
                   out / f'{tag or f"ckpt-{step:06d}"}.pt')

    from rotlab import scrub as _scrubp
    leanpen = _scrubp.parse(a.scrub).get('leanpen', 0.0) if a.scrub else 0.0   # lead 28 Sep 21:00
    model.train(); it = iter(dl); step = 0
    while step < a.steps:
        bt = next(it)
        x, th = bt[0].to(dev, non_blocking=True), bt[1].to(dev, non_blocking=True)
        xc = bt[2].to(dev, non_blocking=True) if len(bt) > 2 else None   # lead 28 Sep: clean teacher view (--scrub dual=1)
        x_full = x
        if sizes:
            sz = random.Random(a.seed * 7919 + step).choice(sizes)
            if sz != x.shape[-1]:
                x = F.interpolate(x, size=(sz, sz), mode='bilinear', antialias=True, align_corners=False)
            if xc is not None and sz != xc.shape[-1]:
                xc = F.interpolate(xc, size=(sz, sz), mode='bilinear', antialias=True, align_corners=False)
        lr_f = min(1.0, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps)))
        for g, b in zip(opt.param_groups, base):
            g['lr'] = b * lr_f
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cuda):
            logits = fwd(x)
            if teacher is not None:
                with torch.no_grad():
                    xt = xc if xc is not None else x if not a.teacher_canvas else (x_full if x_full.shape[-1] == a.teacher_canvas else F.interpolate(
                        x_full, size=(a.teacher_canvas, a.teacher_canvas), mode='bilinear', antialias=True, align_corners=False))
                    t_logits = teacher(xt)
        logp = F.log_softmax(logits.float(), 1)
        pmark = torch.nan_to_num(th) >= 2000.0   # lead 28 Sep 21:00: probes with a known cue angle (scrub leanpen)
        probe = torch.isnan(th) | pmark   # lead 28 Sep: content-free shortcut stimuli (--scrub probe=p) -> uniform target, no KD
        cue = (torch.nan_to_num(th) >= 1000.0) & ~pmark   # lead 28 Sep 20:45: cue-KD samples (scrub cuekd): KD only
        th_raw = th
        th = torch.where(cue, th - 1000.0, torch.where(probe, torch.zeros_like(th), th))
        tgt = cgd_batch(torch.nan_to_num(th), a.sigma)
        if probe.any():
            tgt = torch.where(probe[:, None], torch.full_like(tgt, 1.0 / tgt.shape[1]), tgt)
        ce_i = torch.sum(-tgt * logp, 1)
        loss = ce_i.mean()
        if teacher is not None:
            T = a.distill_temp
            pt = F.softmax(t_logits.float() / T, 1)
            kd_i = torch.sum(pt * (torch.log(pt + 1e-12) - F.log_softmax(logits.float() / T, 1)), 1) * T * T
            if cue.any():
                w_ce = torch.where(cue, torch.zeros_like(ce_i), torch.full_like(ce_i, 1 - a.distill_alpha))
                w_kd = torch.where(probe, torch.zeros_like(kd_i), torch.where(cue, torch.ones_like(kd_i), torch.full_like(kd_i, a.distill_alpha)))
                loss = (w_ce * ce_i + w_kd * kd_i).mean()
            else:
                keep = (~probe).float()
                kd = (kd_i * keep).sum() / keep.sum().clamp(min=1.0)
                loss = (1 - a.distill_alpha) * loss + a.distill_alpha * kd
        if pmark.any() and leanpen > 0:
            from rotlab.train_ddp import lean_penalty
            loss = loss + leanpen * lean_penalty(logp, th_raw, pmark)
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); step += 1; seen += len(x) * world
        with torch.no_grad():
            d = min(a.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        if step % 10 == 0:
            keep, ng = model.stats
            hist.append(loss.item()); kept.append(keep.float().mean().item()); groups.append(ng)
        if step % 200 == 0 and main0:
            el = time.time() - t0
            rec = dict(step=step, loss=round(float(np.mean(hist[-20:])), 4), gn=round(float(gn), 3), img_s=round(seen / el, 1),
                       min=round(el / 60, 1), kept=round(float(np.mean(kept[-20:])), 4), passes=round(float(np.mean(groups[-20:])), 2))
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
        if step % a.eval_every == 0 or step == a.steps:
            evaluate(step)
        if step % a.save_every == 0 or step == a.steps:
            save(step, 'last' if step != a.steps else 'final')
            if a.keep_milestones and step != a.steps:
                save(step)
    if ddp:
        dist.barrier(); dist.destroy_process_group()


# ---------------------------------------------------------------- eval passthrough (focus eval --mode fill)
def eval_argv(argv):
    """focus-eval argv for a fill-drop checkpoint: --mode fill, the checkpoint's --sinks / --tol unless given, and for a
    milestone ckpt-NNNNNN.pt a tag <run>-sNNNNNN-fill[sinkN] (focus names tags after the run dir, so milestones collide)."""
    if not argv or argv[0].startswith('-'):
        raise SystemExit('usage: python -m rotlab.filldrop eval CKPT [rotlab.focus eval flags]')
    ck, rest = argv[0], []
    for r in argv[1:]:                    # --flag=value -> --flag value
        rest += r.split('=', 1) if r.startswith('--') and '=' in r else [r]
    fd = ckpt_config(ck).get('filldrop') or {}
    has = lambda f: f in rest      # noqa: E731
    if has('--mode') and rest[rest.index('--mode') + 1] not in ('fill', 'fillmask'):
        raise SystemExit('filldrop eval scores the fill modes only; use rotlab.focus / rotlab.camp_eval directly for others')
    if not has('--mode'):
        rest += ['--mode', 'fill']
    sinks = fd.get('sinks', 0)
    if not has('--sinks') and sinks:
        rest += ['--sinks', str(sinks)]
    if not has('--tol') and fd.get('tol', TOL) != TOL:
        rest += ['--tol', repr(fd['tol'])]
    stem = Path(ck).stem
    if not has('--tag') and stem != 'final':
        mm = re.fullmatch(r'ckpt-(\d+)', stem)
        if not mm:
            raise SystemExit(f'{ck}: pass --tag for a {stem}.pt checkpoint (focus eval tags by run dir only)')
        mode = rest[rest.index('--mode') + 1]
        s = int(rest[rest.index('--sinks') + 1]) if has('--sinks') else 0
        rest += ['--tag', f'{Path(ck).parent.name}-s{mm.group(1)}-{mode}' + (f'sink{s}' if s else '')]
    return [ck] + rest


def cmd_eval(argv):
    from rotlab import focus
    argv = eval_argv(argv)
    print('focus eval', ' '.join(argv), flush=True)
    focus.cmd_eval(argv)


# ---------------------------------------------------------------- deployment export + CPU benchmark
def export_filldrop(m: FillDropNet, canvas=224, path=None):
    """Batch-1 ONNX (opset 18, TorchScript trace, FP32): input 'image' (1, 3, canvas, canvas) letterboxed as today,
    output 'prob' (1, 360). The fill mask, the sink choice and the token gather (NonZero) are computed in-graph, so ONE
    graph serves every aspect ratio with a data-dependent token count."""
    return export_onnx(ProbGraph(m.eval()), [sample_input(canvas, 4 / 3)], ['image'], path)


def parse_aspect(s):
    w, h = (float(v) for v in s.split(':'))
    return w / h


def cmd_export(argv):
    ap = argparse.ArgumentParser(prog='rotlab.filldrop export')
    ap.add_argument('model', help='checkpoint path, or s/b/l for random weights')
    ap.add_argument('--out', required=True); ap.add_argument('--canvas', type=int, default=224)
    ap.add_argument('--sinks', type=int, help='default: the checkpoint config (0 for random weights)')
    ap.add_argument('--tol', type=float); ap.add_argument('--model-weights', action='store_true', help='raw weights, not EMA')
    ap.add_argument('--int8', action='store_true', help='also write <out>-int8.onnx (rotlab.export dynamic-quantisation settings)')
    a = ap.parse_args(argv)
    if a.model in ARCHS:
        net, fd = build_net(a.model), {}
    else:
        ck = torch.load(a.model, map_location='cpu', weights_only=False); cfg = ck['config']
        if cfg.get('focus') or cfg.get('pool'):
            raise SystemExit('plain RotNet / fill-drop checkpoints only')
        net = build_net(cfg.get('arch', 's'), cfg['img_size'], hidden=cfg.get('hidden', 1024))
        net.load_state_dict(ck['model' if a.model_weights else 'ema']); fd = cfg.get('filldrop') or {}
        del ck
    m = FillDropNet(net, sinks=a.sinks if a.sinks is not None else fd.get('sinks', 0),
                    tol=a.tol if a.tol is not None else fd.get('tol', TOL)).eval()
    data = export_filldrop(m, a.canvas, a.out)
    s = ort_session(data)
    rows = {}
    for lab in ('4:3', '16:9', '1:1', '2:3', '21:9'):
        x = sample_input(a.canvas, parse_aspect(lab), seed=11)
        with torch.no_grad():
            ref = m(x).softmax(1).numpy()
        rows[lab] = dict(kept=int(m.keep_mask(x).sum()), max_abs_prob_diff=float(np.abs(s.run(None, {'image': x.numpy()})[0] - ref).max()))
    print(json.dumps(dict(out=a.out, sinks=m.sinks, tol=m.tol, canvas=a.canvas, mb=round(len(data) / 1e6, 1), parity=rows)))
    if a.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        q = a.out.replace('.onnx', '') + '-int8.onnx'
        quantize_dynamic(a.out, q, weight_type=QuantType.QInt8, per_channel=True, reduce_range=True, op_types_to_quantize=['MatMul', 'Gemm'])
        print('int8', q)


def cmd_bench(argv):
    """Paired CPU timing, random weights (latency only): per arch, the plain deployed graph (model_x.export_onnx, static
    canvas) and the fill-drop graph are resident together; for each aspect both run on the SAME letterboxed input,
    round-robin (rotating start), reps per round. Ratio = mean over rounds of (fill round mean / plain round mean);
    p90 ratio = p90(fill) / p90(plain)."""
    from rotlab.model_x import export_onnx as export_plain
    ap = argparse.ArgumentParser(prog='rotlab.filldrop bench')
    ap.add_argument('--archs', default='s,l'); ap.add_argument('--aspects', default='4:3,16:9,1:1')
    ap.add_argument('--sinks', default='0', help='comma list of sink counts to time (each is its own graph)')
    ap.add_argument('--canvas', type=int, default=224); ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--rounds', type=int, default=12); ap.add_argument('--reps', type=int, default=3); ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--json')
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    sinks = [int(v) for v in a.sinks.split(',')]
    res = dict(threads=a.threads, rounds=a.rounds, reps=a.reps, canvas=a.canvas, batch=1, rows=[],
               versions=dict(torch=torch.__version__, ort=__import__('onnxruntime').__version__))
    for arch in a.archs.split(','):
        net = build_net(arch).eval()
        t = time.time()
        graphs = {'plain': ort_session(export_plain(net, a.canvas), a.threads)}
        for s in sinks:
            graphs[f'fill{s}'] = ort_session(export_filldrop(FillDropNet(net, sinks=s), a.canvas), a.threads)
        print(f'built {arch} graphs in {time.time() - t:.0f}s', flush=True)
        for lab in a.aspects.split(','):
            x = sample_input(a.canvas, parse_aspect(lab), seed=3)
            feeds = {'image': x.numpy()}
            with torch.no_grad():
                ref = {'plain': net(x).softmax(1).numpy()}
                for s in sinks:
                    ref[f'fill{s}'] = FillDropNet(net, sinks=s).eval()(x).softmax(1).numpy()
            names = list(graphs)
            parity = {n: float(np.abs(graphs[n].run(None, feeds)[0] - ref[n]).max()) for n in names}
            for _ in range(a.warmup):
                for n in names:
                    graphs[n].run(None, feeds)
            times, rmean = {n: [] for n in names}, {n: [] for n in names}
            for r in range(a.rounds):
                for n in names[r % len(names):] + names[:r % len(names)]:
                    ts = []
                    for _ in range(a.reps):
                        t0 = time.perf_counter(); graphs[n].run(None, feeds); ts.append((time.perf_counter() - t0) * 1e3)
                    times[n] += ts; rmean[n].append(float(np.mean(ts)))
            pl = np.array(times['plain'])
            for n in names:
                tt = np.array(times[n])
                row = dict(arch=arch, aspect=lab, graph=n, kept=int(FillDropNet(net, sinks=int(n[4:])).keep_mask(x).sum()) if n != 'plain' else
                           (a.canvas // 14) ** 2, mean=round(float(tt.mean()), 1), p90=round(float(np.percentile(tt, 90)), 1),
                           ratio_mean=round(float(np.mean(np.array(rmean[n]) / np.array(rmean['plain']))), 4),
                           ratio_p90=round(float(np.percentile(tt, 90) / np.percentile(pl, 90)), 4), max_abs_prob_diff_vs_torch=parity[n],
                           loadavg=round(os.getloadavg()[0], 1))
                res['rows'].append(row)
                print(json.dumps(row), flush=True)
        del graphs, net; gc.collect()
    print('| arch | aspect | graph | patch tokens | mean ms | p90 ms | mean ratio (paired) | p90 ratio | ORT-torch max abs prob diff |')
    print('|---|---|---|---|---|---|---|---|---|')
    for r in res['rows']:
        print(f"| {r['arch']} | {r['aspect']} | {r['graph']} | {r['kept']} | {r['mean']} | {r['p90']} | {r['ratio_mean']} | {r['ratio_p90']} | {r['max_abs_prob_diff_vs_torch']:.1e} |")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    argv = argv[1:] if argv[:1] == ['--'] else argv         # torchrun ... -m rotlab.filldrop -- train ...
    cmds = dict(train=cmd_train, eval=cmd_eval, export=cmd_export, bench=cmd_bench)
    if not argv or argv[0] not in cmds:
        raise SystemExit(f'usage: python -m rotlab.filldrop {{{"|".join(cmds)}}} ...')
    cmds[argv[0]](argv[1:])


if __name__ == '__main__':
    main()
