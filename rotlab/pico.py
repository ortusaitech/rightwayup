#!/usr/bin/env python3
"""Pico lane (29 Sep 2026): shared-stream multi-student fill-drop distillation for truncated DINOv2-S students.

One data stream (rotlab.train_ddp.Views with the scrub recipe), ONE teacher forward per batch, K students (e.g. s6 and s8,
rotlab.model.DEPTH) that each have their own optimizer, EMA and run directory. Per student the loss is exactly
rotlab.filldrop.cmd_train's (CGD CE + KD, probes -> uniform target and no KD, cue-KD samples KD only, lean penalty on marked
probes). Differences from filldrop.cmd_train (both recorded in research/pico/PREREG.md):
  * several students share each batch (the data loader is the bottleneck on a 24-CPU pod; the GPU is not);
  * --teacher-canvas with --scrub dual=1: the teacher sees the CLEAN view at the render canvas (e.g. 224, where the Max is
    accurate) while the students train on --multires sizes (e.g. 84-112) resized from the same render (filldrop resizes the
    clean view to the student size, i.e. the teacher would see 84-112 px).
Checkpoints are plain RotNet state dicts (config.arch = s6/s8, config.filldrop) like filldrop's, so rotlab.filldrop eval /
export and rotlab.focus eval (with this code tree) load them.

  PYTHONPATH=/workspace/code-pico ROTLAB_DATA=/workspace/rotation-data
  python -m rotlab.pico train --runs PICO-D6,PICO-D8 --archs s6,s8 --init CKPT --teacher CKPT --teacher-canvas 224 \\
      --img-size 224 --multires 84,84,98,112,112 --scrub ... --mix ... --steps N --lr LR --workers 12
"""
from __future__ import annotations
import argparse, copy, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parser():
    ap = argparse.ArgumentParser(prog='rotlab.pico train')
    ap.add_argument('--runs', required=True, help='comma list of run dir names, one per student')
    ap.add_argument('--archs', required=True, help='comma list of student archs (s4..s10), same length as --runs')
    ap.add_argument('--mix', required=True)
    ap.add_argument('--steps', type=int, required=True)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--head-lr', type=float, default=5e-4)
    ap.add_argument('--lld', type=float, default=0.8)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--sigma', type=float, default=6.0)
    ap.add_argument('--img-size', type=int, default=224, help='model img_size (pos-embed grid; dynamic) and render canvas')
    ap.add_argument('--drop-path', type=float, default=0.1)
    ap.add_argument('--ema', type=float, default=0.9995)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--save-every', type=int, default=5000)
    ap.add_argument('--keep-milestones', action='store_true')
    ap.add_argument('--init', type=Path, required=True, help='full-depth s checkpoint (EMA weights); first k blocks kept')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--teacher', type=Path, required=True)
    ap.add_argument('--distill-alpha', type=float, default=0.5)
    ap.add_argument('--distill-temp', type=float, default=1.0)
    ap.add_argument('--teacher-canvas', type=int, default=0, help='teacher input size (clean view resized if needed); 0 = student size')
    ap.add_argument('--scrub', default='')
    ap.add_argument('--data-frac', type=float, default=1.0)
    ap.add_argument('--multires', required=True, help="student sizes (multiples of 14); repeats weight the choice; ';' separates "
                    "per-student lists (one list = all students); students with the same list get the same size each step")
    ap.add_argument('--tol', type=float, default=None)
    ap.add_argument('--max-minutes', type=float, default=0.0, help='safety: save final and stop after this wall time (0 = off)')
    ap.add_argument('--reload-every', type=int, default=6000,
                    help='recreate the loader workers every N steps (fresh fork -> bounded copy-on-write memory creep); the data '
                         'seed of segment k = seed + 104729 k, so no segment replays another (0 = off)')
    ap.add_argument('--resume-step', type=int, default=0, help='resume every student that has <run>/ckpt-NNNNNN.pt (model, EMA, '
                    'optimizer); a student without one starts fresh at this step with its own schedule of steps - resume-step')
    ap.add_argument('--init-blocks', default='', help="per-student ';' list of parent block indices for the warm start, e.g. "
                    "'0,1,3,5,7,11' (student block j <- parent block idx[j]); empty entry = the first k blocks")
    ap.add_argument('--feat-kd', default='0', help="per-student ';' weights of the feature-distillation term (0 = off): "
                    "gamma * (1 - cos(LN(f_student), LN(f_parent))) on the head input, parent = --feat-teacher on the clean view")
    ap.add_argument('--feat-teacher', type=Path, help='full-depth s checkpoint for --feat-kd (fill-drop forward, same token set)')
    ap.add_argument('--init-ckpts', default='', help="per-student ';' warm-start checkpoints (empty entry = --init)")
    ap.add_argument('--lr-scale', default='1', help="per-student ';' multiplier of --lr and --head-lr")
    ap.add_argument('--kd-src', default='teacher', help="per-student ';' KD source: 'teacher' (--teacher at --teacher-canvas) or "
                    "'parent' (the --feat-teacher network on the clean view at the student's size: a teacher-assistant)")
    return ap


def featnet_class():
    from rotlab.filldrop import FillDropNet

    class FeatNet(FillDropNet):
        """FillDropNet whose forward returns cat(logits, head input) when with_feat is set (per fill-drop group, so the
        rows follow FillDropNet's own regrouping/reordering). Weights and state-dict keys are unchanged."""
        with_feat = False

        def encode(self, tok, weights=None, attn_mask=None):
            if not self.with_feat:
                return super().encode(tok, weights, attn_mask)
            cap = {}
            h = self.net.head.register_forward_hook(lambda m, i, o: cap.__setitem__('f', i[0]))
            try:
                logits = super().encode(tok, weights, attn_mask)
            finally:
                h.remove()
            return torch.cat([logits, cap['f'].to(logits.dtype)], 1)
    return FeatNet


def cmd_train(argv):
    from torch.utils.data import DataLoader
    from rotlab.filldrop import FillDropNet, build_net, ckpt_config, TOL
    from rotlab.model import DEPTH
    from rotlab.model_x import SPECS, param_groups
    from rotlab.train_ddp import RUNS, Views, cgd_batch, lean_penalty
    from rotlab import scrub as _scrubp
    a = parser().parse_args(argv)
    tol = TOL if a.tol is None else a.tol
    runs, archs = a.runs.split(','), a.archs.split(',')
    assert len(runs) == len(archs) and all(x in DEPTH or x == 's' for x in archs), (runs, archs)   # 's' = full 12 blocks
    icfg = ckpt_config(a.init)
    assert icfg.get('arch', 's') == 's' and icfg.get('img_size', 224) == a.img_size and not icfg.get('focus') and not icfg.get('pool'), icfg
    mls = a.multires.split(';')
    mls = mls * len(runs) if len(mls) == 1 else mls
    assert len(mls) == len(runs), 'one --multires list, or one per student'
    msizes = [[int(v) for v in m.split(',')] for m in mls]
    ibs = a.init_blocks.split(';') if a.init_blocks else [''] * len(runs)
    fks = [float(v) for v in a.feat_kd.split(';')]; fks = fks * len(runs) if len(fks) == 1 else fks
    assert len(ibs) == len(runs) and len(fks) == len(runs), 'one --init-blocks / --feat-kd entry per student'
    ics = a.init_ckpts.split(';') if a.init_ckpts else [''] * len(runs)
    lrs = [float(v) for v in a.lr_scale.split(';')]; lrs = lrs * len(runs) if len(lrs) == 1 else lrs
    assert len(lrs) == len(runs), 'one --lr-scale entry per student'
    assert len(ics) == len(runs), 'one --init-ckpts entry per student'
    kds = a.kd_src.split(';'); kds = kds * len(runs) if len(kds) == 1 else kds
    assert len(kds) == len(runs) and set(kds) <= {'teacher', 'parent'}, kds
    assert not (any(fks) or 'parent' in kds) or a.feat_teacher, '--feat-kd / --kd-src parent need --feat-teacher'
    if any(s % 14 for s in sum(msizes, []) + [a.img_size]):
        raise SystemExit('canvas and --multires sizes must be multiples of 14')
    dev = torch.device('cuda')
    mix = {k: float(v) for k, v in (x.split('=') for x in a.mix.split(','))}
    torch.manual_seed(a.seed); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    sc = _scrubp.parse(a.scrub) if a.scrub else {}
    leanpen = sc.get('leanpen', 0.0)

    tc = ckpt_config(a.teacher)
    def load_sd(path, arch, key=None):   # mmap: the parent keeps no anonymous copy of the checkpoint (forked workers inherit it)
        from rotlab.model import truncate_state_dict
        ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        sd = ck[key] if key else (ck['ema'] if 'ema' in ck else ck['model'])
        return truncate_state_dict(sd, arch), ck
    teacher = build_net(tc.get('arch', 's'), tc['img_size'], hidden=tc.get('hidden', 1024))
    teacher.load_state_dict(load_sd(a.teacher, tc.get('arch', 's'))[0]); teacher = teacher.to(dev).eval().requires_grad_(False)

    def remap_blocks(sd, idx, pre='backbone.blocks.'):   # student block j <- parent block idx[j]; other parent blocks dropped
        out = {}
        for n, v in sd.items():
            if n.startswith(pre):
                i = int(n[len(pre):].split('.')[0])
                if i in idx:
                    out[pre + str(idx.index(i)) + n[len(pre) + len(str(i)):]] = v
            else:
                out[n] = v
        return out

    FeatNet = featnet_class()
    P = None
    if any(fks) or 'parent' in kds:
        pn = build_net('s', a.img_size, 0.0)
        pn.load_state_dict(load_sd(a.feat_teacher, 's')[0])
        P = FeatNet(pn, sinks=0, tol=tol, batching='group').to(dev).eval().requires_grad_(False); P.with_feat = True

    S = []
    for run, arch, sizes, ib, fk, kd, ic, lrsc in zip(runs, archs, msizes, ibs, fks, kds, ics, lrs):
        ic = Path(ic) if ic else a.init
        out = RUNS / run; out.mkdir(parents=True, exist_ok=True)
        rck = out / f'ckpt-{a.resume_step:06d}.pt' if a.resume_step else None
        rck = rck if rck is not None and rck.exists() else None
        if rck:   # a resumed student keeps the schedule it started with (recorded in its checkpoint config)
            s0 = int(torch.load(rck, map_location='cpu', weights_only=False, mmap=True)['config']['filldrop'].get('schedule_start', 0))
        else:
            s0 = a.resume_step
        idx = [int(v) for v in ib.split(',')] if ib else list(range(DEPTH.get(arch, 12)))
        assert len(idx) == DEPTH.get(arch, 12), (run, idx)
        (out / ('config.json' if not rck else f'config-resume-{a.resume_step:06d}.json')).write_text(json.dumps(
            vars(a) | dict(run=run, arch=arch, multires=','.join(map(str, sizes)), init_blocks=idx, init_ckpt=str(ic), lr_scale=lrsc, feat_kd=fk, kd_src=kd, schedule_start=s0,
                           schedule_steps=a.steps - s0, mix=mix, world=1, global_batch=a.batch, trainer='rotlab.pico (shared stream)',
                           co_students=runs), default=str, indent=1))
        net = build_net(arch, a.img_size, a.drop_path, pretrained=False)
        if rck:
            net.load_state_dict(load_sd(rck, arch, 'model')[0])
        else:
            full = torch.load(ic, map_location='cpu', weights_only=False, mmap=True)
            net.load_state_dict(remap_blocks(full['ema'] if 'ema' in full else full['model'], idx)); del full
        model = FeatNet(net, sinks=0, tol=tol, batching='group').to(dev); model.with_feat = fk > 0
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        opt = torch.optim.AdamW(param_groups(model.net, a.lr, a.head_lr, a.wd, a.lld), betas=(0.9, 0.999), fused=True)
        base0 = [g['lr'] for g in opt.param_groups]    # intended per-group base LRs (before any optimizer state is loaded)
        lr_factor = 1.0
        if rck:
            esd, ck = load_sd(rck, arch, 'ema'); ema.net.load_state_dict(esd)
            if 'opt' in ck:
                opt.load_state_dict(ck['opt'])   # NB: this also restores each group's last scheduled 'lr'
                # 16:25 fix: before, base = the loaded (already scheduled) lr, so every resume compounded the cosine. Keep
                # the trajectory the student is on (no LR jump): factor = saved lr / (intended base x schedule at the save).
                sl = a.resume_step - 1 - s0; nloc = a.steps - s0
                f_save = min(1.0, (sl + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, sl / nloc)))
                lr_factor = float(ck['opt']['param_groups'][0]['lr']) / (base0[0] * f_save)
                print(f'{run}: LR continuity factor {lr_factor:.4f} (effective base = intended x factor)', flush=True)
            assert ck['step'] == a.resume_step, (rck, ck['step'])
            print(f'{run}: resumed model/EMA/{"optimizer" if "opt" in ck else "NO optimizer"} from {rck}', flush=True)
            del ck
        cfg = dict(img_size=a.img_size, canvas=a.img_size, hidden=net.head[1].out_features, buckets=False, arch=arch, multires=sizes, pool=None,
                   backbone=dict(timm=SPECS[arch].timm, revision=SPECS[arch].revision, sha256=SPECS[arch].sha256, status=SPECS[arch].status,
                                 depth=DEPTH.get(arch, 12)),
                   filldrop=dict(sinks=0, tol=tol, batching='group', init=str(ic), init_blocks=idx, teacher=str(a.teacher), teacher_canvas=a.teacher_canvas,
                                 lr_scale=lrsc, feat_kd=fk, feat_teacher=str(a.feat_teacher) if (fk or kd == 'parent') else None, kd_src=kd,
                                 schedule_start=s0, schedule_steps=a.steps - s0,
                                 code='rotlab.pico (shared-stream multi-student)',
                                 eval='python -m rotlab.filldrop eval CKPT  (PYTHONPATH=/workspace/code-pico)'))
        S.append(dict(run=run, arch=arch, sizes=sizes, s0=s0, n=a.steps - s0, fk=fk, kd=kd, out=out, model=model, ema=ema, opt=opt, base=[b * lr_factor * lrsc for b in base0],
                      trainable=[p for p in model.parameters() if p.requires_grad], cfg=cfg, log=(out / 'log.jsonl').open('a'),
                      hist=[], kept=[], groups=[], nparam=sum(p.numel() for p in model.net.parameters())))
        print(json.dumps(dict(run=run, arch=arch, blocks=len(net.backbone.blocks), init_blocks=idx, feat_kd=fk, kd_src=kd, schedule=[s0, a.steps],
                              resumed=bool(rck), params_M=round(S[-1]['nparam'] / 1e6, 2))), flush=True)

    import gc
    ds = Views(mix, a.img_size, a.seed, frac=a.data_frac, maxarea_p=0.0, scrub=a.scrub)

    def loader(step):
        k = step // a.reload_every if a.reload_every else 0
        ds.seed = a.seed + 104729 * k + (31 * (step % a.reload_every) if a.reload_every else 0)   # a mid-segment resume gets its own seed
        dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True, persistent_workers=False,
                        prefetch_factor=4 if a.workers else None)
        gc.collect(); gc.freeze()                # parent objects -> permanent generation: the children's GC never writes them
        print(f'loader segment {k} (data seed {ds.seed}) at step {step}', flush=True)
        return iter(dl)

    def save(s, step, tag, with_opt=False):
        d = dict(model=s['model'].net.state_dict(), ema=s['ema'].net.state_dict(), config=s['cfg'], step=step, presentations=seen,
                 args=vars(a) | dict(run=s['run'], arch=s['arch']))
        if with_opt:
            d['opt'] = s['opt'].state_dict()
        torch.save(d, s['out'] / f'{tag}.pt')

    for s in S:
        s['model'].train()
    step = a.resume_step; seen = step * a.batch; t0 = time.time(); seen0 = seen; it = loader(step); stop = False
    while step < a.steps and not stop:
        if a.reload_every and step > a.resume_step and step % a.reload_every == 0:
            del it; gc.unfreeze(); gc.collect()
            it = loader(step)
        bt = next(it)
        x, th = bt[0].to(dev, non_blocking=True), bt[1].to(dev, non_blocking=True)
        xc = bt[2].to(dev, non_blocking=True) if len(bt) > 2 else None
        rs = lambda t, n: t if n == t.shape[-1] else F.interpolate(t, size=(n, n), mode='bilinear', antialias=True, align_corners=False)  # noqa: E731
        szs = [random.Random(a.seed * 7919 + step).choice(s['sizes']) for s in S]
        xin = {n: rs(x, n) for n in set(szs)}
        src = xc if xc is not None else x
        assert a.teacher_canvas or len(set(szs)) == 1, '--teacher-canvas 0 needs one student size per step'
        tcv = a.teacher_canvas or szs[0]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            t_logits = teacher(rs(src, tcv))
        # targets (identical to filldrop.cmd_train)
        pmark = torch.nan_to_num(th) >= 2000.0
        probe = torch.isnan(th) | pmark
        cue = (torch.nan_to_num(th) >= 1000.0) & ~pmark
        th_raw = th
        th = torch.where(cue, th - 1000.0, torch.where(probe, torch.zeros_like(th), th))
        tgt = cgd_batch(torch.nan_to_num(th), a.sigma)
        if probe.any():
            tgt = torch.where(probe[:, None], torch.full_like(tgt, 1.0 / tgt.shape[1]), tgt)
        T = a.distill_temp
        pt = F.softmax(t_logits.float() / T, 1)
        pout = {}
        for s, sz in zip(S, szs):
            xs = xin[sz]
            s['sz'] = sz
            loc = step - s['s0']
            lr_f = min(1.0, (loc + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, loc / s['n'])))
            for g, b in zip(s['opt'].param_groups, s['base']):
                g['lr'] = b * lr_f
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out_ = s['model'](xs)
            logits, fs = (out_[:, :360], out_[:, 360:]) if s['fk'] > 0 else (out_, None)
            logp = F.log_softmax(logits.float(), 1)
            ce_i = torch.sum(-tgt * logp, 1)
            if s['kd'] == 'parent':
                if sz not in pout:
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        pout[sz] = P(rs(src, sz)).float()
                pts = F.softmax(pout[sz][:, :360] / T, 1)
            else:
                pts = pt
            kd_i = torch.sum(pts * (torch.log(pts + 1e-12) - F.log_softmax(logits.float() / T, 1)), 1) * T * T
            if cue.any():
                w_ce = torch.where(cue, torch.zeros_like(ce_i), torch.full_like(ce_i, 1 - a.distill_alpha))
                w_kd = torch.where(probe, torch.zeros_like(kd_i), torch.where(cue, torch.ones_like(kd_i), torch.full_like(kd_i, a.distill_alpha)))
                loss = (w_ce * ce_i + w_kd * kd_i).mean()
            else:
                keep = (~probe).float()
                kd = (kd_i * keep).sum() / keep.sum().clamp(min=1.0)
                loss = (1 - a.distill_alpha) * ce_i.mean() + a.distill_alpha * kd
            if pmark.any() and leanpen > 0:
                loss = loss + leanpen * lean_penalty(logp, th_raw, pmark)
            if s['fk'] > 0:   # feature KD: parent's head input on the clean view at the student's size, probes excluded
                if sz not in pout:
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        pout[sz] = P(rs(src, sz)).float()
                fs, fp = fs.float(), pout[sz][:, 360:]
                cosd = 1 - F.cosine_similarity(F.layer_norm(fs, fs.shape[-1:]), F.layer_norm(fp, fp.shape[-1:]), dim=1)
                keepf = (~probe).float()
                fl = (cosd * keepf).sum() / keepf.sum().clamp(min=1.0)
                loss = loss + s['fk'] * fl; s['fl'] = fl.item() if (step + 1) % 10 == 0 else s.get('fl', 0.0)
            s['opt'].zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(s['trainable'], 1.0)
            s['opt'].step()
            with torch.no_grad():
                d = min(a.ema, (2 + loc) / (11 + loc))       # == filldrop's (1 + step') / (10 + step') with step' = local step + 1
                for pe, pm in zip(s['ema'].parameters(), s['model'].parameters()):
                    pe.lerp_(pm, 1 - d)
            s['gn'] = gn
            if (step + 1) % 10 == 0:
                keepm, ng = s['model'].stats
                s['hist'].append(loss.item()); s['kept'].append(keepm.float().mean().item()); s['groups'].append(ng)
        step += 1; seen += len(x)
        el = time.time() - t0
        if step % 200 == 0:
            for s in S:
                rec = dict(step=step, loss=round(float(np.mean(s['hist'][-20:])), 4), gn=round(float(s['gn']), 3), img_s=round((seen - seen0) / el, 1),
                           min=round(el / 60, 1), kept=round(float(np.mean(s['kept'][-20:])), 4), passes=round(float(np.mean(s['groups'][-20:])), 2),
                           sz_last=s['sz'], local=step - s['s0'], **({'feat': round(s.get('fl', 0.0), 4)} if s['fk'] else {}))
                print(s['run'], json.dumps(rec), flush=True); s['log'].write(json.dumps(rec) + '\n'); s['log'].flush()
        if a.max_minutes and el / 60 > a.max_minutes:
            print(f'max-minutes reached at step {step}: saving final', flush=True); stop = True
        if step % a.save_every == 0 or step == a.steps or stop:
            for s in S:
                save(s, step, 'final' if (step == a.steps or stop) else 'last')
                if a.keep_milestones and step != a.steps and not stop:
                    save(s, step, f'ckpt-{step:06d}', with_opt=True)
    print(json.dumps(dict(done=True, step=step, minutes=round((time.time() - t0) / 60, 1))), flush=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] != 'train':
        raise SystemExit('usage: python -m rotlab.pico train ...')
    cmd_train(argv[1:])


if __name__ == '__main__':
    main()
