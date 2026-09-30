#!/usr/bin/env python3
"""H8 (pre-registered, campaign 2026-09-27): supervised non-orthogonal rotation consistency, matched control.

Init G7 (EMA). Trainable: head + ALL LayerNorm affine parameters of the backbone (norm1/norm2 per block + final norm);
every linear/attention/LayerScale/patch/pos weight frozen. Input 224 (letterbox), G7 mix, CGD sigma 6, core.degrade.
Paired views per parent: theta1 ~ U[0,360), s = +-1, theta2 = theta1 + 45 s. Both rendered DIRECTLY from the source with
the same centre, aspect and angle-independent extent (the crop's circumscribed circle lies inside the image, so no
padding at any angle), same zoom, and an IDENTICAL degradation draw (same seeded RNG). Residual content difference:
the rotated rectangle covers slightly different corner regions of the same disc (documented, unavoidable).
  control:   loss = mean(CGD(view1), CGD(view2))
  treatment: loss = control + lam * JS(roll(p1, 45 s bins), p2)
Screening dose: 512 updates, 64 parent pairs (128 views) per update, seed 0, fixed endpoint, no EMA (final weights).
  python -m rotlab.camp_h8 --arm control|treatment --run NAME [--steps 512 --pairs 64 --lam 0.1 --lr 5e-5]
"""
import argparse, json, math, os, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from rotlab.core import ASPECTS, DATA, degrade, letterbox, render
from rotlab.model import RotNet
from rotlab.train import blobs_for, cgd_batch

RUNS = DATA / 'rotlab/runs'
G7MIX = dict(pass_=0.45)


def shift_idx(s, delta=45, bins=360, device='cpu'):
    """Index so that p.gather(1, idx) == p rolled by +delta*s bins (element k takes p[k - delta*s])."""
    k = torch.arange(bins, device=device)[None]
    return (k - delta * s[:, None]) % bins


def js(p, q, eps=1e-8):
    m = 0.5 * (p + q)
    return 0.5 * (p * (torch.log(p + eps) - torch.log(m + eps))).sum(1) + 0.5 * (q * (torch.log(q + eps) - torch.log(m + eps))).sum(1)


def self_test():
    th = torch.tensor([10.0, 350.0, 200.0]); s = torch.tensor([1, -1, 1])
    p1 = cgd_batch(th, 6.0); p2 = cgd_batch((th + 45 * s) % 360, 6.0)
    shifted = p1.gather(1, shift_idx(s))
    assert torch.allclose(shifted, p2, atol=1e-6), 'circular shift sign wrong'
    j = js(shifted, p2); assert torch.isfinite(j).all() and j.max() < 1e-5
    u = torch.full((1, 360), 1 / 360); assert torch.isfinite(js(u, p2[:1])).all()
    return True


class Pairs(IterableDataset):
    def __init__(self, mix, seed):
        self.mix, self.seed = mix, seed; self.blobs = blobs_for(mix); self.fams = list(mix); self.w = [mix[f] for f in self.fams]

    def __iter__(self):
        wi = get_worker_info(); rng = random.Random(self.seed * 1000 + (wi.id if wi else 0))
        while True:
            fam = rng.choices(self.fams, self.w)[0]; b = self.blobs[fam]
            src, row = b.image(rng.randrange(len(b))); W, H = src.size
            th1 = rng.uniform(0, 360); s = rng.choice((-1, 1)); th2 = (th1 + 45 * s) % 360
            a = rng.choices([x for x, _ in ASPECTS], [p for _, p in ASPECTS])[0]
            dmax = min(W, H) - 4; z = rng.uniform(0.65, 1.0) if rng.random() < 0.5 else 1.0; D = z * dmax
            r = rng.uniform(0, (dmax - D) / 2); phi = rng.uniform(0, 2 * math.pi)
            cx, cy = W / 2 + r * math.cos(phi), H / 2 + r * math.sin(phi)
            cw, ch = D * a / math.sqrt(1 + a * a), D / math.sqrt(1 + a * a)
            ow = max(8, min(round(cw), 448)); oh = max(8, round(ow / a))
            dseed = rng.getrandbits(64); dp = rng.random() < 1.0     # degrade_p = 1 as train.py (degrade itself skips 35%)
            xs = []
            for th in (th1, th2):
                v = render(src, (th - row['base_roll_cw']) % 360, cw, ch, ow, oh, cx, cy)
                if dp:
                    v = degrade(v, random.Random(dseed))          # identical draw for both views
                xs.append(torch.from_numpy(letterbox(v, 224)))
            yield xs[0], xs[1], torch.tensor(th1, dtype=torch.float32), torch.tensor(th2, dtype=torch.float32), torch.tensor(s)


def provenance(init, mix):
    """Binding receipt for one arm: init checkpoint SHA-256, admitted pack index/binary hashes, exclusions, code, runtime."""
    import hashlib, platform, PIL, timm
    def sha(p):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(1 << 24), b''):
                h.update(b)
        return h.hexdigest()
    src = DATA / 'rotlab/sources'; packs = sorted({'hybrid' if f in ('diode', 'meva', 'poly_haven', 'poly_haven_blender_direct_v2') else f for f in mix})
    verified = {}
    vf = Path('/workspace/ops/big-files.sha256')          # SHA-256 verified at staging (sha256sum -c OK)
    if vf.exists():
        for line in vf.read_text().splitlines():
            h, n = line.split(maxsplit=1); verified[n.strip()] = h
    code = Path(__file__).parent
    return dict(init_sha256=sha(init), packs={p: dict(jsonl_sha256=sha(src / f'{p}.jsonl'), bin_sha256_verified_at_fetch=verified.get(f'rotlab/sources/{p}.bin'),
                                                        bin_bytes=(src / f'{p}.bin').stat().st_size) for p in packs},
                exclude_sha256=sha(src / 'exclude.json'), code_sha256={f: sha(code / f) for f in ('camp_h8.py', 'core.py', 'model.py', 'train.py')},
                runtime=dict(python=platform.python_version(), torch=torch.__version__, timm=timm.__version__, numpy=np.__version__, pillow=PIL.__version__,
                             cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--arm', required=True, choices=['control', 'treatment']); ap.add_argument('--run', required=True)
    ap.add_argument('--init', default=str(RUNS / 'G7-grecipe-coco2-140/final.pt')); ap.add_argument('--steps', type=int, default=512)
    ap.add_argument('--pairs', type=int, default=64); ap.add_argument('--lam', type=float, default=0.1); ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--warmup', type=int, default=50); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    assert self_test()
    if a.self_test:
        print('self-test ok'); return
    mix = {'pass': 0.45, 'coco': 0.06, 'coco2': 0.09, 'diode': 0.15, 'meva': 0.15, 'poly_haven': 0.08, 'poly_haven_blender_direct_v2': 0.02}
    out = RUNS / a.run; out.mkdir(parents=True, exist_ok=True)
    prov = provenance(a.init, mix)
    (out / 'config.json').write_text(json.dumps(vars(a) | dict(mix=mix, trainable='head + all backbone LayerNorm affines', provenance=prov), indent=1))
    torch.manual_seed(a.seed); torch.backends.cuda.matmul.allow_tf32 = True
    ck = torch.load(a.init, map_location='cpu', weights_only=False)
    m = RotNet(img_size=224, pretrained=False, dynamic=True, arch='l', drop_path=0.0); m.load_state_dict(ck['ema']); m = m.cuda()
    for n, p in m.named_parameters():
        p.requires_grad = n.startswith('head.') or ('norm' in n.split('.')[-2] if n.startswith('backbone.') and n.count('.') >= 2 else False) or n.startswith('backbone.norm.')
    tr = [p for p in m.parameters() if p.requires_grad]
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert all(n.startswith('head.') or '.norm' in n for n in names), names[:5]
    print(json.dumps(dict(trainable_tensors=len(tr), trainable_params=sum(p.numel() for p in tr))), flush=True)
    m.backbone.set_grad_checkpointing(True); m.train()
    opt = torch.optim.AdamW(tr, lr=a.lr, weight_decay=0.0)
    dl = DataLoader(Pairs(mix, a.seed), batch_size=a.pairs, num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    it = iter(dl); t0 = time.time(); log = []
    for step in range(1, a.steps + 1):
        x1, x2, t1, t2, s = [v.cuda(non_blocking=True) for v in next(it)]
        f = min(1.0, step / a.warmup) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        for g in opt.param_groups:
            g['lr'] = a.lr * f
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg = m(torch.cat([x1, x2]))
        lp = F.log_softmax(lg.float(), 1); n = len(x1)
        sup = 0.5 * ((-cgd_batch(t1, 6.0) * lp[:n]).sum(1).mean() + (-cgd_batch(t2, 6.0) * lp[n:]).sum(1).mean())
        p = lp.exp(); jsd = js(p[:n].gather(1, shift_idx(s, device=p.device)), p[n:]).mean()
        loss = sup + (a.lam * jsd if a.arm == 'treatment' else 0.0)
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(tr, 1.0); opt.step()
        if step % 50 == 0 or step == a.steps:
            rec = dict(step=step, sup=round(sup.item(), 4), js=round(jsd.item(), 5), img_s=round(step * 2 * n / (time.time() - t0), 1))
            print(json.dumps(rec), flush=True); log.append(rec)
    sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    torch.save(dict(model=sd, ema=sd, config=dict(ck['config'], img_size=224, canvas=224, h8=vars(a), provenance=prov), step=a.steps), out / 'final.pt')
    print('saved', out / 'final.pt', flush=True)


if __name__ == '__main__':
    main()
