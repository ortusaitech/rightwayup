#!/usr/bin/env python3
"""[lead-push DDP copy of rotlab.train; single-GPU behaviour unchanged; launch with torchrun --nproc_per_node N]
Train the single-pass directed 360-bin model. Fresh random angle for every presentation.

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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from rotlab.core import BUCKET_ASPECTS, BUCKET_P, BUCKETS, DATA, Blob, degrade, letterbox, sample_view
from rotlab.evaluate import PackCache, predict, report, table
from rotlab.model_x import SPECS, RotNet, param_groups   # s/b/l unchanged; adds b_reg, pe_s, tips_b (research-only)

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

    def __init__(self, mix, size, seed, degrade_p=1.0, bucket_batch=0, frac=1.0, maxarea_p=0.0, scrub=''):
        self.mix, self.size, self.seed, self.degrade_p, self.bucket_batch = mix, size, seed, degrade_p, bucket_batch
        self.maxarea_p = maxarea_p
        from rotlab import scrub as _scrub
        self.scrub = _scrub.parse(scrub) if scrub else None   # lead 28 Sep: trace-scrubbed rendering (rotlab/scrub.py)
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
            sc = self.scrub
            if sc and sc['probe'] > 0 and rng.random() < sc['probe']:
                from rotlab import scrub as _scrub   # content-free shortcut stimulus, label NaN = uniform target
                if sc.get('leanpen', 0.0) > 0:   # lead 28 Sep 21:00: probe label = 2000 + cue angle (lean penalty)
                    im_, ang_ = _scrub.probe_image(rng, return_angle=True); lab = torch.tensor(2000.0 + ang_, dtype=torch.float32)
                else:
                    im_ = _scrub.probe_image(rng); lab = torch.tensor(float('nan'))
                x = torch.from_numpy(letterbox(im_, canvas))
                return (x, lab, x) if sc['dual'] else (x, lab)
            cue = False
            try:
                if sc:
                    from rotlab import scrub as _scrub
                    cue = sc.get('cuekd', 0.0) > 0 and rng.random() < sc['cuekd']   # lead 28 Sep: cue-KD sample
                    clean = _scrub.render_view(sample_view, src, row['base_roll_cw'], theta, rng, sc, nodown=cue, aspects=aspects,
                                               maxarea_p=self.maxarea_p)
                    im = _scrub.clutter(clean, rng, sc)
                    if cue:
                        clean = _scrub.downscale(clean, sc['down'])
                else:
                    im = sample_view(src, row['base_roll_cw'], theta, rng, aspects=aspects, maxarea_p=self.maxarea_p)
            except ValueError:
                continue
            if rng.random() < self.degrade_p:
                im = degrade(im, rng)
            x, t = torch.from_numpy(letterbox(im, canvas)), torch.tensor(theta + (1000.0 if sc and cue else 0.0), dtype=torch.float32)
            if sc and sc['dual']:   # lead 28 Sep: the teacher sees the clean view (no decoy / camera JPEG / degradation)
                return x, t, torch.from_numpy(letterbox(clean, canvas))
            return x, t


def lean_penalty(logp, th, pmark):
    """lead 28 Sep 21:00: mean over marked probes of soft^2, soft = sum_phi p(phi) cos 4(phi - t), t = label - 2000."""
    p = logp[pmark].exp(); t = th[pmark] - 2000.0
    ang = torch.arange(p.shape[1], device=p.device, dtype=p.dtype) * (360.0 / p.shape[1])
    soft = (p * torch.cos(torch.deg2rad(4.0 * (ang[None] - t[:, None])))).sum(1)
    return (soft ** 2).mean()


LEANPEN = 0.0


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
    ap.add_argument('--arch', default='s', choices=sorted(SPECS))
    ap.add_argument('--teacher', type=Path, help='distil from this checkpoint (its EMA weights), run live per batch')
    ap.add_argument('--distill-alpha', type=float, default=0.7, help='weight of teacher KL vs ground-truth CGD')
    ap.add_argument('--distill-temp', type=float, default=1.0)
    ap.add_argument('--teacher-canvas', type=int, default=0, help='lead: teacher sees the batch at this canvas (render at --canvas >= it; the student trains on --multires sizes resized from the canvas); 0 = teacher sees the student input')
    ap.add_argument('--scrub', default='', help='lead: trace-scrubbed rendering, e.g. ss=1,decoy=0.35,cam=0.4 (rotlab/scrub.py)')
    ap.add_argument('--maxarea-p', type=float, default=0.0, help='DIAGNOSTIC ONLY, never for release candidates (owner decision 2026-09-23): share of views using the incumbent-benchmark max-area crop, which leaks the angle')
    ap.add_argument('--data-frac', type=float, default=1.0, help='train on a deterministic nested subset of every family')
    ap.add_argument('--multires', default='', help='e.g. 140,168,196,224: each step trains at one of these square sizes (batch resized from the canvas)')
    ap.add_argument('--grad-ckpt', action='store_true', help='activation checkpointing (large backbones)')
    ap.add_argument('--qat', action='store_true', help='lead 28 Sep: INT8-aware fine-tuning (rotlab/qat.py: ORT dynamic-INT8 fake quant)')
    ap.add_argument('--pack-eval', action='store_true', help='in-training eval on the legacy benchmark pack (needs the pack archive)')
    ap.add_argument('--train-blocks', type=int, default=0, help='campaign: if >0, train only the last N backbone blocks + final norm + head')
    ap.add_argument('--pool', default='', help='campaign: K:P static token pooling after block K to a PxP grid')
    ap.add_argument('--keep-milestones', action='store_true', help='lead: save ckpt-NNNNNN.pt at every --save-every (no overwrite)')
    a = ap.parse_args()
    ddp = 'LOCAL_RANK' in os.environ and int(os.environ.get('WORLD_SIZE', '1')) > 1
    rank, world, lrank = 0, 1, 0
    if ddp:
        dist.init_process_group('nccl'); rank, world = dist.get_rank(), dist.get_world_size(); lrank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(lrank)
        assert a.batch % world == 0, 'global batch must divide by world size'
    main0 = rank == 0; gbatch = a.batch; a.batch = gbatch // world
    mix = {k: float(v) for k, v in (x.split('=') for x in a.mix.split(','))}
    canvas = a.canvas or a.img_size; sizes = [int(v) for v in a.multires.split(',')] if a.multires else []
    dyn = a.buckets or canvas != a.img_size or bool(sizes)
    out = RUNS / a.run; out.mkdir(parents=True, exist_ok=True)
    if main0:
        (out / 'config.json').write_text(json.dumps(vars(a) | dict(mix=mix, world=world, global_batch=gbatch), default=str, indent=1))
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
    global LEANPEN
    if a.scrub:
        from rotlab import scrub as _scrubp
        LEANPEN = _scrubp.parse(a.scrub).get('leanpen', 0.0)
    if a.scrub and 'cuekd' in a.scrub:
        raise SystemExit('scrub cuekd is implemented only in rotlab.filldrop (its loss decodes the cue-KD label marker)')
    if a.qat:
        from rotlab.qat import apply_qat
        print('qat: patched', apply_qat(model), 'Linear layers', flush=True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    fwd = DDP(model, device_ids=[lrank], broadcast_buffers=False) if ddp else model
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
        ds = Views(mix, canvas, a.seed + 7919 * rank, frac=a.data_frac, maxarea_p=a.maxarea_p, scrub=a.scrub)
        dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    pc = PackCache(canvas, a.buckets) if (a.pack_eval and main0) else None
    cfg = dict(img_size=a.img_size, canvas=canvas, hidden=1024, buckets=a.buckets, arch=a.arch, multires=sizes, pool=a.pool or None,
               backbone=dict(timm=SPECS[a.arch].timm, revision=SPECS[a.arch].revision, sha256=SPECS[a.arch].sha256, status=SPECS[a.arch].status))
    log = (out / 'log.jsonl').open('a') if main0 else open(os.devnull, 'w')
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
        if not main0:
            return
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), config=cfg, step=step, presentations=seen, args=vars(a)),
                   out / f'{tag or f"ckpt-{step:06d}"}.pt')

    model.train(); it = iter(dl); step = 0
    while step < a.steps:
        bt = next(it)
        x, th = bt[0].cuda(non_blocking=True), bt[1].cuda(non_blocking=True)
        xc = bt[2].cuda(non_blocking=True) if len(bt) > 2 else None   # lead 28 Sep: clean teacher view (--scrub dual=1)
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
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = fwd(x)
            if teacher is not None:
                with torch.no_grad():
                    xt = xc if xc is not None else x if not a.teacher_canvas else (x_full if x_full.shape[-1] == a.teacher_canvas else F.interpolate(
                        x_full, size=(a.teacher_canvas, a.teacher_canvas), mode='bilinear', antialias=True, align_corners=False))
                    t_logits = teacher(xt)
        logp = F.log_softmax(logits.float(), 1)
        pmark = torch.nan_to_num(th) >= 2000.0   # lead 28 Sep 21:00: probes with a known cue angle (scrub leanpen)
        probe = torch.isnan(th) | pmark   # lead 28 Sep: content-free shortcut stimuli (--scrub probe=p) -> uniform target, no KD
        tgt = cgd_batch(torch.nan_to_num(torch.where(probe, torch.zeros_like(th), th)), a.sigma)
        if probe.any():
            tgt = torch.where(probe[:, None], torch.full_like(tgt, 1.0 / tgt.shape[1]), tgt)
        loss = torch.sum(-tgt * logp, 1).mean()
        if pmark.any() and LEANPEN > 0:
            loss = loss + LEANPEN * lean_penalty(logp, th, pmark)
        if teacher is not None:
            T = a.distill_temp
            pt = F.softmax(t_logits.float() / T, 1)
            kd_i = torch.sum(pt * (torch.log(pt + 1e-12) - F.log_softmax(logits.float() / T, 1)), 1) * T * T
            keep = (~probe).float()
            kd = (kd_i * keep).sum() / keep.sum().clamp(min=1.0)
            loss = (1 - a.distill_alpha) * loss + a.distill_alpha * kd
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); step += 1; seen += len(x) * world
        with torch.no_grad():
            d = min(a.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        hist.append(loss.item()) if step % 10 == 0 else None
        if step % 200 == 0 and main0:
            el = time.time() - t0
            print(json.dumps(dict(step=step, loss=round(float(np.mean(hist[-20:])), 4), gn=round(float(gn), 3), img_s=round(seen / el, 1), min=round(el / 60, 1))), flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            evaluate(step)
        if step % a.save_every == 0 or step == a.steps:
            save(step, 'last' if step != a.steps else 'final')
            if a.keep_milestones and step != a.steps:
                save(step)
    save(step, 'final')
    if ddp:
        dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    main()
