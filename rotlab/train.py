#!/usr/bin/env python3
"""Train the single-pass directed 360-bin model. Fresh random angle for every presentation.

  python -m rotlab.train --run NAME --mix pass=0.5,coco=0.15,diode=0.12,meva=0.12,poly_haven=0.11 \
      --steps 16000 --batch 128 --eval-every 2000

Checkpoints/evals land in /workspace/rotation-data/rotlab/runs/NAME/.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from rotlab.core import BUCKET_ASPECTS, BUCKET_P, BUCKETS, DATA, Blob, degrade, letterbox, sample_view
from rotlab.evaluate import PackCache, predict, report, table
from rotlab.model import RotNet, param_groups

RUNS = DATA / 'rotlab/runs'
HYBRID = ('diode', 'meva', 'poly_haven', 'poly_haven_blender_direct_v2')
GROUPED = HYBRID + ('diode2', 'meva2')   # subset by scene/camera group in --data-frac


def keep_frac(r, frac):
    """Deterministic nested subset (frac 0.25 is inside 0.5). Hybrid families subset by scene/camera
    group so a fraction removes diversity, not just frames; photo families subset by image id."""
    if frac >= 1:
        return True
    key = r['group'] if r['family'] in GROUPED else r['id']
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def blobs_for(mix, frac=1.0):
    """family name -> Blob. hybrid families are filtered views of hybrid.bin.
    Sources near-duplicate to any origin-benchmark photo (sources/exclude.json) are dropped."""
    exf = DATA / 'rotlab/sources/exclude.json'
    ex = set(json.loads(exf.read_text())['ids']) if exf.exists() else set()
    out = {}
    for fam in mix:
        if fam in HYBRID:
            out[fam] = Blob('hybrid', filt=lambda r, f=fam: r['family'] == f and r['id'] not in ex and keep_frac(r, frac))
        else:
            out[fam] = Blob(fam, filt=lambda r: r['id'] not in ex and keep_frac(r, frac))
        print(f'data {fam}: {len(out[fam])} images, {len({r["group"] for r in out[fam].rows})} groups', flush=True)
    return out


class Views(IterableDataset):
    """Yields single samples (square mode) or whole same-bucket batches (bucket mode)."""

    def __init__(self, mix, size, seed, degrade_p=1.0, bucket_batch=0, frac=1.0, maxarea_p=0.0):
        self.mix, self.size, self.seed, self.degrade_p, self.bucket_batch = mix, size, seed, degrade_p, bucket_batch
        self.maxarea_p = maxarea_p
        self.blobs = blobs_for(mix, frac)
        self.fams = list(mix); self.w = [mix[f] for f in self.fams]

    def __iter__(self):
        wi = get_worker_info()
        # deterministic per (seed, worker); runs before 2026-09-24 also mixed in int(time.time())
        rng = random.Random(self.seed * 1000 + (wi.id if wi else 0) + (int(time.time()) if os.environ.get("ROTLAB_LEGACY_TIME_SEED") else 0))
        if not self.bucket_batch:
            while True:
                yield self.one(rng, self.size, None)
        while True:
            b = rng.choices(list(BUCKET_P), list(BUCKET_P.values()))[0]
            xs, ts = zip(*(self.one(rng, BUCKETS[b], BUCKET_ASPECTS[b]) for _ in range(self.bucket_batch)))
            yield torch.stack(xs), torch.stack(ts)

    def one(self, rng, canvas, aspects):
        while True:
            fam = rng.choices(self.fams, self.w)[0]
            b = self.blobs[fam]
            src, row = b.image(rng.randrange(len(b)))
            theta = rng.uniform(0, 360)
            try:
                im = sample_view(src, row['base_roll_cw'], theta, rng, aspects=aspects, maxarea_p=self.maxarea_p)
            except ValueError:
                continue
            if rng.random() < self.degrade_p:
                im = degrade(im, rng)
            return torch.from_numpy(letterbox(im, canvas)), torch.tensor(theta, dtype=torch.float32)


def cgd_batch(theta, sigma, bins=360):
    k = torch.arange(bins, device=theta.device, dtype=torch.float32)
    d = torch.remainder(k[None] - theta[:, None] + 180, 360) - 180
    t = torch.exp(-0.5 * (d / sigma) ** 2)
    return t / t.sum(1, keepdim=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
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
    ap.add_argument('--init', type=Path, help='resume weights (model/ema) from a checkpoint')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--buckets', action='store_true', help='aspect-bucketed canvases (L 168x294, S 224, P 294x168)')
    ap.add_argument('--arch', default='s', choices=['s', 'b', 'l'])
    ap.add_argument('--teacher', type=Path, help='distil from this checkpoint (its EMA weights), run live per batch')
    ap.add_argument('--distill-alpha', type=float, default=0.7, help='weight of teacher KL vs ground-truth CGD')
    ap.add_argument('--distill-temp', type=float, default=1.0)
    ap.add_argument('--maxarea-p', type=float, default=0.0, help='DIAGNOSTIC ONLY, never for release candidates (owner decision 2026-09-23): share of views using the incumbent-benchmark max-area crop, which leaks the angle')
    ap.add_argument('--data-frac', type=float, default=1.0, help='train on a deterministic nested subset of every family')
    ap.add_argument('--multires', default='', help='e.g. 140,168,196,224: each step trains at one of these square sizes (batch resized from the canvas)')
    ap.add_argument('--grad-ckpt', action='store_true', help='activation checkpointing (large backbones)')
    ap.add_argument('--pack-eval', action='store_true', help='in-training eval on the legacy benchmark pack (needs the pack archive)')
    ap.add_argument('--train-blocks', type=int, default=0, help='campaign: if >0, train only the last N backbone blocks + final norm + head')
    ap.add_argument('--pool', default='', help='campaign: K:P static token pooling after block K to a PxP grid')
    a = ap.parse_args()
    mix = {k: float(v) for k, v in (x.split('=') for x in a.mix.split(','))}
    canvas = a.canvas or a.img_size; sizes = [int(v) for v in a.multires.split(',')] if a.multires else []
    dyn = a.buckets or canvas != a.img_size or bool(sizes)
    out = RUNS / a.run; out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(vars(a) | dict(mix=mix), default=str, indent=1))
    torch.manual_seed(a.seed); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    model = RotNet(img_size=a.img_size, drop_path=a.drop_path, dynamic=dyn, arch=a.arch, pretrained=a.init is None).cuda()
    if a.init:
        ck = torch.load(a.init, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['ema'] if 'ema' in ck else ck['model'])
    if a.pool:
        model.pool = tuple(map(int, a.pool.split(':')))
    if a.train_blocks:
        nb = len(model.backbone.blocks)
        for n_, p_ in model.backbone.named_parameters():
            p_.requires_grad = n_.startswith('norm.') or (n_.startswith('blocks.') and int(n_.split('.')[1]) >= nb - a.train_blocks)
        print('trainable params', sum(p_.numel() for p_ in model.parameters() if p_.requires_grad), flush=True)
    if a.grad_ckpt:
        model.backbone.set_grad_checkpointing(True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    teacher = None
    if a.teacher:
        tk = torch.load(a.teacher, map_location='cpu', weights_only=False)
        teacher = RotNet(img_size=tk['config']['img_size'], pretrained=False, arch=tk['config'].get('arch', 's'), dynamic=True).cuda().eval().requires_grad_(False)
        teacher.load_state_dict(tk['ema']); del tk
    opt = torch.optim.AdamW(param_groups(model, a.lr, a.head_lr, a.wd, a.lld), betas=(0.9, 0.999), fused=True)
    base = [g['lr'] for g in opt.param_groups]

    if a.buckets:  # each worker assembles whole same-bucket batches
        ds = Views(mix, canvas, a.seed, bucket_batch=a.batch, frac=a.data_frac)
        dl = DataLoader(ds, batch_size=None, num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    else:
        ds = Views(mix, canvas, a.seed, frac=a.data_frac, maxarea_p=a.maxarea_p)
        dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    pc = PackCache(canvas, a.buckets) if a.pack_eval else None
    cfg = dict(img_size=a.img_size, canvas=canvas, hidden=1024, buckets=a.buckets, arch=a.arch, multires=sizes, pool=a.pool or None)
    log = (out / 'log.jsonl').open('a')
    t0 = time.time(); seen = 0; hist = []

    def evaluate(step):
        if pc is None:
            return
        rows = {}
        for name, m in (('model', model), ('ema', ema)):
            res, _ = report(pc, predict(m, pc.x, pc=pc), f'{a.run} step {step} {name}')
            rows[name] = res
        model.train()
        rec = dict(step=step, presentations=seen, eval=rows, minutes=round((time.time() - t0) / 60, 2))
        log.write(json.dumps(rec) + '\n'); log.flush()
        (out / f'eval-{step:06d}.md').write_text(table(rows['model']) + '\n\n' + table(rows['ema']))
        e = rows['ema']['panels']
        print(json.dumps(dict(step=step, ema=dict(common=e['common']['w10'], fresh=e['fresh_diode']['w10'], origin=e['origin']['w10'],
                                              origin_val17=e.get('origin_val2017', {}).get('w10'), t150=[e[p]['t150'] for p in ('common', 'fresh_diode', 'origin')]))), flush=True)

    def save(step, tag=None):
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), config=cfg, step=step, presentations=seen, args=vars(a)),
                   out / f'{tag or f"ckpt-{step:06d}"}.pt')

    model.train(); it = iter(dl); step = 0
    while step < a.steps:
        x, th = next(it)
        x = x.cuda(non_blocking=True); th = th.cuda(non_blocking=True)
        if sizes:
            sz = random.Random(a.seed * 7919 + step).choice(sizes)
            if sz != x.shape[-1]:
                x = F.interpolate(x, size=(sz, sz), mode='bilinear', antialias=True, align_corners=False)
        lr_f = min(1.0, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps)))
        for g, b in zip(opt.param_groups, base):
            g['lr'] = b * lr_f
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(x)
        logp = F.log_softmax(logits.float(), 1)
        loss = torch.sum(-cgd_batch(th, a.sigma) * logp, 1).mean()
        if teacher is not None:
            T = a.distill_temp
            pt = F.softmax(t_logits.float() / T, 1)
            kd = torch.sum(pt * (torch.log(pt + 1e-12) - F.log_softmax(logits.float() / T, 1)), 1).mean() * T * T
            loss = (1 - a.distill_alpha) * loss + a.distill_alpha * kd
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); step += 1; seen += len(x)
        with torch.no_grad():
            d = min(a.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        hist.append(loss.item()) if step % 10 == 0 else None
        if step % 200 == 0:
            el = time.time() - t0
            print(json.dumps(dict(step=step, loss=round(float(np.mean(hist[-20:])), 4), gn=round(float(gn), 3), img_s=round(seen / el, 1), min=round(el / 60, 1))), flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            evaluate(step)
        if step % a.save_every == 0 or step == a.steps:
            save(step, 'last' if step != a.steps else 'final')
    save(step, 'final')


if __name__ == '__main__':
    main()
