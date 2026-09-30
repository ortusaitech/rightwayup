#!/usr/bin/env python3
"""H13 (PREPARATION ONLY; no GPU training authorised): fresh MambaOut-Small (timm/mambaout_small.in1k rev 6dab1289,
research-only ImageNet-1k lineage) on the admitted S2 expanded mix, crop-safe sample_view + core.degrade, 224 letterbox,
CGD sigma 6, batch 128, FIXED endpoint 60,000 steps (7.68M presentations); descriptive EMA exports at 8k/24k/40k; full
recovery state every 4,000 steps. Secondary fixed endpoint (not in this module): 0.5 p_Small + 0.5 p_S2 via camp_fuse.

Data stream (seekable, full-byte, versioned):
  * sample i of GLOBAL batch b draws from random.Random(seed_of(seed, b, i)) with a stable SHA-256 derivation (never
    Python hash()); every sample is independent of worker count and of the resume point.
  * resume-relative worker assignment: a loader started at batch s gives worker w the batches s + w + j*W, so with the
    in-order DataLoader the k-th consumed batch is exactly s + k for ANY s (4000 % 24 = 16 included). Each batch
    carries its global id; the loop asserts id == step. Prefetched-but-unconsumed batches are never counted.
  * digest chain (version H13v1): chain_{k+1} = sha256(chain_k || sha256(version, id, shapes, dtypes, x bytes,
    theta bytes)). The chain hex + count is the serialisable resume state.
Recovery: every 4,000 completed updates an atomic checkpoint (tmp + fsync + rename, then a sha256 sidecar; the previous
verified checkpoint is kept) binds model, EMA, optimizer, group order, schedule, EMA counter, cursor, digest chain and
CPU/CUDA/Python/NumPy RNG state of the SAME completed update. Resume is explicit only (--resume PATH --resume-sha256 SHA
--resume-authority TEXT), refuses truncated/mismatched state or config, and appends a segment record: retained
trajectory dose (final step x 128) and physical executed presentations are recorded separately. No automatic replay or
restart; no examples beyond the fixed endpoint; a partial run is never "completed".
Provenance (bound into the config hash at admission and recomputed + compared on resume; fail closed): source-pack
manifests + verified binary hashes (camp_h8.provenance + staged-extra receipt), REQUIRED exclusion file hash (missing ->
refuse), ordered filtered row identities per family, code hashes (camp_h13/core/train/camp_h8/model) and runtime.
Exports (8k/24k/40k/final EMA): serialised in memory, sha bound into the recovery state saved FIRST, then published
atomically (tmp + fsync + rename); an existing export must be byte-identical or the run refuses (never overwritten).
Physical work: durable progress receipts (fsync) every `receipt_every` updates plus segment start/end records; an abrupt
death leaves a BOUNDED UNKNOWN [last receipt, last receipt + receipt_every] (inclusive: a crash may fall between an optimizer
update that completes a receipt boundary and its receipt) rather than an invented count. A forecast abort writes an exact end
record. On resume, recorded exports are reconciled BEFORE any step: existing files must match their recorded sha; a missing
export for the checkpoint's own step is reconstructed from the checkpoint EMA and atomically published (sha must equal the
recorded one); a missing EARLIER milestone fails closed (never fabricated from later weights).
Forecast: after 200 updates of EVERY segment (fresh or resumed) and again every 2,000 updates; exit 3 if the forecast
finish passes --limit. The external job wall clock remains the enforced stop.
  python -m rotlab.camp_h13 selftest
  python -m rotlab.camp_h13 train --arch mo_s|tiny --run NAME --limit EPOCH [--resume ...] [--stop-after N (tests)]
  python -m rotlab.camp_h13 eval CKPT --canvases 224 --sets newval,dev,rlivit_dev_rgb,rlivit_dev_thermal
"""
import argparse, copy, hashlib, io, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from rotlab.core import DATA, degrade, letterbox, sample_view

VERSION = 'H13v1'
MO_S = ('mambaout_small.in1k', '/workspace/rotation-data/quarantine/external-weights/mambaout_small-in1k-6dab1289/model.safetensors')
MO_S_SHA256 = '48b5046af3a41cb18c46dfad3c462e468966ac0ace33fcc5e3ca6608f9bdcc5f'   # rev 6dab12890168444e9e114f91ccf8b568b73f7228
EXMIX = {'pass': 0.15, 'pass2': 0.10, 'coco': 0.05, 'coco2': 0.08, 'oi': 0.22, 'diode': 0.12, 'diode2': 0.05, 'meva': 0.10,
         'meva2': 0.05, 'poly_haven': 0.06, 'poly_haven_blender_direct_v2': 0.02}
PROTOCOL = dict(steps=60000, batch=128, lr=1e-4, head_lr=5e-4, stage_decay=0.8, wd=0.05, warmup=1500, ema=0.9995, clip=1.0,
                drop_path=0.1, sigma=6.0, canvas=224, seed=0, workers=24, milestones=(8000, 24000, 40000), recovery_every=4000)
EXPECTED_MISSING = {'head.fc.weight', 'head.fc.bias'}


# ------------------------------------------------------------------ stream
def seed_of(seed, b, i):
    return int.from_bytes(hashlib.sha256(f'{VERSION}:{seed}:{b}:{i}'.encode()).digest()[:8], 'big')


def batch_digest(bid, x, th):
    h = hashlib.sha256(); h.update(f'{VERSION}|{bid}|{tuple(x.shape)}|{x.dtype}|{tuple(th.shape)}|{th.dtype}|'.encode())
    h.update(memoryview(x.contiguous().numpy()).cast('B')); h.update(memoryview(th.contiguous().numpy()).cast('B'))
    return h.digest()


def chain_step(chain_hex, d):
    return hashlib.sha256(bytes.fromhex(chain_hex) + d).hexdigest()


CHAIN0 = hashlib.sha256(VERSION.encode()).hexdigest()


class SeekableViews(IterableDataset):
    """Whole batches (x uint8->float letterboxed, theta, global id). Worker w of W renders batches start + w + j*W."""

    def __init__(self, mix, canvas, seed, batch, start, blobs=None):
        self.mix, self.canvas, self.seed, self.batch, self.start = mix, canvas, seed, batch, start
        self.fams = list(mix); self.w = [mix[f] for f in self.fams]
        if blobs is None:
            from rotlab.train import blobs_for
            blobs = blobs_for(mix)
        self.blobs = blobs

    def one(self, rng):   # identical logic to rotlab.train.Views.one (square mode, degrade_p 1, no max-area)
        while True:
            fam = rng.choices(self.fams, self.w)[0]
            b = self.blobs[fam]
            src, row = b.image(rng.randrange(len(b)))
            theta = rng.uniform(0, 360)
            try:
                im = sample_view(src, row['base_roll_cw'], theta, rng)
            except ValueError:
                continue
            im = degrade(im, rng)
            return torch.from_numpy(letterbox(im, self.canvas)), torch.tensor(theta, dtype=torch.float32)

    def make(self, bid):
        xs, ts = zip(*(self.one(random.Random(seed_of(self.seed, bid, i))) for i in range(self.batch)))
        return torch.stack(xs), torch.stack(ts), torch.tensor(bid, dtype=torch.int64)

    def __iter__(self):
        wi = get_worker_info(); w, W = (wi.id, wi.num_workers) if wi else (0, 1)
        bid = self.start + w
        while True:
            yield self.make(bid); bid += W


def loader(ds, workers, pin):
    g = torch.Generator(); g.manual_seed(0)   # dedicated: loader base-seed draw never consumes the model's CPU RNG
    return DataLoader(ds, batch_size=None, num_workers=workers, pin_memory=pin, persistent_workers=False,
                      prefetch_factor=2 if workers else None, generator=g)


# ------------------------------------------------------------------ models
class MoS(nn.Module):
    def __init__(self, drop_path=0.0, bins=360):
        super().__init__()
        import timm
        self.backbone = timm.create_model(MO_S[0], pretrained=False, num_classes=bins, drop_path_rate=drop_path)

    def forward(self, x):
        return self.backbone(x)


class Tiny(nn.Module):
    """CPU test stand-in with the same parameter naming (backbone.stages.N / backbone.head) and stochastic layers."""

    def __init__(self, drop_path=0.1, bins=360):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.stages = nn.ModuleList([nn.Sequential(nn.Conv2d(3, 8, 5, 4), nn.GELU()), nn.Sequential(nn.Conv2d(8, 8, 3, 2), nn.GELU())])
        self.backbone.head = nn.Sequential(nn.Dropout(drop_path), nn.Linear(8, bins))

    def forward(self, x):
        for s in self.backbone.stages:
            x = s(x)
        return self.backbone.head(x.mean((2, 3)))


def load_pretrained_small(model, path, want_sha):
    """Exact key accounting: missing == {head.fc.weight, head.fc.bias}, unexpected == {} (else refuse)."""
    from safetensors.torch import load_file
    if sha_file(path) != want_sha:
        raise SystemExit('refusing: MambaOut-Small weights sha256 mismatch')
    sd = {k: v for k, v in load_file(path).items() if not k.startswith('head.fc.')}
    res = model.backbone.load_state_dict(sd, strict=False)
    if set(res.missing_keys) != EXPECTED_MISSING or res.unexpected_keys:
        raise SystemExit(f'refusing: key accounting missing={sorted(res.missing_keys)} unexpected={sorted(res.unexpected_keys)}')
    return dict(missing=sorted(res.missing_keys), unexpected=[], loaded=len(sd))


def mo_groups(model, lr, head_lr, wd, decay):
    """Same rule as camp_h10.mo_groups: stages.3 x1, stages.2 x0.8, stages.1 x0.64, stages.0/stem x0.512; head at head_lr.
    Groups are returned in a deterministic order (sorted by (lr, no_decay)) and named for checkpoint validation."""
    groups = {}
    for n, p in model.named_parameters():
        if n.startswith('backbone.head.'):
            scale = head_lr
        elif n.startswith('backbone.stages.'):
            scale = lr * decay ** (3 - int(n.split('.')[2])) if len([1 for _ in model.backbone.stages]) == 4 else lr
        else:
            scale = lr * decay ** 3
        nd = p.ndim == 1 or 'norm' in n
        g = groups.setdefault((scale, nd), dict(params=[], names=[], lr=scale, weight_decay=0.0 if nd else wd)); g['params'].append(p); g['names'].append(n)
    out = [groups[k] for k in sorted(groups)]
    return out


# ------------------------------------------------------------------ recovery
def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def config_sha(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def rng_state(device):
    st = dict(torch_cpu=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state())
    if device == 'cuda':
        st['torch_cuda'] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st, device):
    torch.set_rng_state(st['torch_cpu']); random.setstate(st['python']); np.random.set_state(st['numpy'])
    if device == 'cuda':
        torch.cuda.set_rng_state_all(st['torch_cuda'])


def save_recovery(out, step, model, ema, opt, groups, chain, cfg, device, exports):
    d = out / 'recovery'; d.mkdir(exist_ok=True)
    state = dict(version=VERSION, step=step, cursor=step, ema_counter=step, chain=chain, chain_count=step,
                 config=cfg, config_sha256=config_sha(cfg), group_names=[g['names'] for g in groups],
                 schedule=dict(base_lrs=[g['base_lr'] for g in groups], warmup=cfg['warmup'], steps=cfg['steps']),
                 model={k: v.detach().cpu() for k, v in model.state_dict().items()}, ema={k: v.detach().cpu() for k, v in ema.state_dict().items()},
                 opt=opt.state_dict(), rng=rng_state(device), exports=dict(exports))
    p = d / f'step-{step:06d}.pt'; tmp = d / f'.step-{step:06d}.pt.tmp'
    with open(tmp, 'wb') as f:
        torch.save(state, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)
    (d / f'step-{step:06d}.pt.sha256').write_text(sha_file(p) + '\n')
    kept = sorted(d.glob('step-*.pt'))
    for old in kept[:-2]:   # keep the newest two verified states (the previous one is retained)
        old.with_suffix('.pt.sha256').unlink(missing_ok=True); old.unlink()
    return p


def load_recovery(path, want_sha, cfg):
    path = Path(path); side = path.with_suffix('.pt.sha256')
    got = sha_file(path)
    if got != want_sha or not side.exists() or side.read_text().strip() != got:
        raise SystemExit('refusing: recovery state sha256 mismatch (truncated/altered or wrong sidecar)')
    st = torch.load(path, map_location='cpu', weights_only=False)
    if st.get('version') != VERSION or st['config_sha256'] != config_sha(cfg) or st['cursor'] != st['step'] or st['chain_count'] != st['step']:
        raise SystemExit('refusing: recovery state version/config/cursor mismatch')
    return st


# ------------------------------------------------------------------ provenance
def row_identity(blobs):
    h = hashlib.sha256(); counts = {}
    for fam in sorted(blobs):
        rows = getattr(blobs[fam], 'rows', None)
        if rows is None:
            raise SystemExit(f'refusing: family {fam} has no row identities')
        counts[fam] = len(rows)
        for r in rows:
            h.update(f"{fam}|{r['id']}|{r.get('offset')}|{r.get('length')}\n".encode())
    return dict(ordered_filtered_rows_sha256=h.hexdigest(), counts=counts)


def admission_provenance(blobs, fake):
    import platform, PIL, timm
    code = Path(__file__).parent
    base = dict(code_sha256={f: sha_file(code / f) for f in ('camp_h13.py', 'core.py', 'train.py', 'camp_h8.py', 'model.py')},
                runtime=dict(python=platform.python_version(), torch=torch.__version__, timm=timm.__version__, numpy=np.__version__, pillow=PIL.__version__))
    if fake:
        return dict(base, data='TEST-FAKE-BLOBS (CPU tests only; refused for mo_s)', rows=row_identity(blobs))
    exf = DATA / 'rotlab/sources/exclude.json'
    if not exf.exists():
        raise SystemExit('refusing: REQUIRED exclusion file rotlab/sources/exclude.json missing (train.blobs_for would silently exclude nothing)')
    from rotlab.camp_h8 import provenance
    prov = provenance(MO_S[1], EXMIX)
    extra = {}
    xf = Path('/workspace/ops/staged-extra.sha256')
    if xf.exists():
        for line in xf.read_text().splitlines():
            h, n = line.split(maxsplit=1); extra[n.strip()] = h
    for pk, v in prov['packs'].items():
        if v.get('bin_sha256_verified_at_fetch'):
            continue
        key = f'rotlab/sources/{pk}.bin'; f = DATA / key
        if key not in extra or f.stat().st_size > 100_000_000 or sha_file(f) != extra[key]:
            raise SystemExit(f'refusing: pack {pk} has no verified staged binary hash')
        v['bin_sha256_verified_at_fetch'] = extra[key]; v['bin_hash_source'] = 'ops/staged-extra.sha256 (re-hashed at admission)'
    prov['runtime'].pop('gpu', None)   # the device may differ across a (separately authorised) migration; recorded per segment
    return dict(base, packs=prov['packs'], exclude_sha256=prov['exclude_sha256'], init_sha256=prov['init_sha256'], rows=row_identity(blobs))


# ------------------------------------------------------------------ durable records
def append_durable(path, rec):
    with open(path, 'a') as f:
        f.write(json.dumps(rec) + '\n'); f.flush(); os.fsync(f.fileno())


def publish_export(md, blob, want_sha):
    """Atomic publication; an existing export must be byte-identical (verify) or the run refuses (never overwrite)."""
    md.mkdir(parents=True, exist_ok=True); p = md / 'final.pt'
    if p.exists():
        if sha_file(p) != want_sha:
            raise SystemExit(f'refusing: existing export {p} differs from the recomputed one (never overwritten)')
        return 'verified-existing'
    tmp = md / '.final.pt.tmp'
    with open(tmp, 'wb') as f:
        f.write(blob); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)
    if sha_file(p) != want_sha:
        raise SystemExit(f'export readback mismatch {p}')
    return 'published'


# ------------------------------------------------------------------ training
def cgd(theta, sigma, bins=360):
    k = torch.arange(bins, device=theta.device, dtype=torch.float32)
    d = torch.remainder(k[None] - theta[:, None] + 180, 360) - 180
    t = torch.exp(-0.5 * (d / sigma) ** 2)
    return t / t.sum(1, keepdim=True)


def lr_factor(step, warmup, steps):   # step = 1-based index of the update being made
    return min(1.0, step / warmup) * 0.5 * (1 + math.cos(math.pi * step / steps))


def train(a, blobs=None):
    fake = blobs is not None
    if a.arch == 'mo_s' and (fake or os.environ.get('H13_TEST_BLOBS')):
        raise SystemExit('refusing: test/fake inputs can never enter mo_s training')
    cfg = dict(PROTOCOL, arch=a.arch, mix=EXMIX, run=a.run, receipt_every=200)
    for k in ('steps', 'batch', 'warmup', 'workers', 'recovery_every', 'canvas', 'receipt_every'):
        if getattr(a, k, None) is not None:
            cfg[k] = getattr(a, k)
    if a.milestones is not None:
        cfg['milestones'] = tuple(int(x) for x in a.milestones.split(',') if x)
    device = 'cuda' if a.arch == 'mo_s' else 'cpu'
    out = Path(a.out_root or (DATA / 'rotlab/runs')) / a.run
    if blobs is None:
        from rotlab.train import blobs_for
        blobs = blobs_for(EXMIX)
    cfg['provenance'] = admission_provenance(blobs, fake)   # recomputed on resume; any change -> config sha mismatch -> refuse
    torch.manual_seed(cfg['seed']); random.seed(cfg['seed']); np.random.seed(cfg['seed'])
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
    model = (MoS(cfg['drop_path']) if a.arch == 'mo_s' else Tiny(cfg['drop_path']))
    keys = load_pretrained_small(model, MO_S[1], MO_S_SHA256) if a.arch == 'mo_s' else None
    model = model.to(device)
    if device == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    groups = mo_groups(model, cfg['lr'], cfg['head_lr'], cfg['wd'], cfg['stage_decay'])
    for g in groups:
        g['base_lr'] = g['lr']
    opt = torch.optim.AdamW([dict(params=g['params'], lr=g['lr'], weight_decay=g['weight_decay']) for g in groups], betas=(0.9, 0.999))
    step, chain, exports = 0, CHAIN0, {}
    seg = dict(kind='start', seg_id=time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + f'-{os.getpid()}', start_step=0, resumed_from=None)
    if a.resume:
        if not a.resume_authority:
            raise SystemExit('refusing: resume needs an explicit --resume-authority (root decision); no automatic restart')
        st = load_recovery(a.resume, a.resume_sha256, cfg)
        if st['group_names'] != [g['names'] for g in groups]:
            raise SystemExit('refusing: optimizer group order differs')
        model.load_state_dict(st['model']); ema.load_state_dict(st['ema']); opt.load_state_dict(st['opt'])
        step, chain, exports = st['step'], st['chain'], dict(st['exports']); set_rng_state(st['rng'], device)
        for name, esha in exports.items():   # reconcile recorded exports BEFORE any step
            md = out.parent / name; ep = md / 'final.pt'
            if ep.exists():
                if sha_file(ep) != esha:
                    raise SystemExit(f'refusing: existing export {ep} differs from its recorded sha')
                continue
            est = cfg['steps'] if name.endswith('-final') else int(name.rsplit('-s', 1)[1])
            if est != step:
                raise SystemExit(f'refusing: recorded export {name} (step {est}) is missing and the checkpoint is at step {step}; never fabricated')
            sd = {k: v.detach().cpu() for k, v in st['ema'].items()}
            buf = io.BytesIO(); torch.save(dict(model=sd, ema=sd, step=step, presentations=step * cfg['batch'], config=dict(img_size=224, canvas=224, arch=a.arch, hidden=None, h13=cfg, stream_chain=chain)), buf)
            if hashlib.sha256(buf.getvalue()).hexdigest() != esha:
                raise SystemExit(f'refusing: reconstructed export {name} does not match its recorded sha')
            print('export', name, publish_export(md, buf.getvalue(), esha), '(reconciled on resume)', flush=True)
        seg.update(start_step=step, resumed_from=str(a.resume), resume_sha256=a.resume_sha256, authority=a.resume_authority)
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out / 'config.json').write_text(json.dumps(dict(cfg, config_sha256=config_sha(cfg), keys=keys), indent=1, default=str))
    seg['device'] = torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'
    append_durable(out / 'segments.jsonl', seg)
    ds = SeekableViews(EXMIX, cfg['canvas'], cfg['seed'], cfg['batch'], step, blobs=blobs)
    it = iter(loader(ds, cfg['workers'], device == 'cuda'))
    trainable = [p for p in model.parameters() if p.requires_grad]
    model.train(); t0 = time.time(); seg_t0 = step
    while step < cfg['steps']:
        x, th, bid = next(it)
        if int(bid) != step:
            raise SystemExit(f'stream order violated: got batch {int(bid)} at step {step}')
        chain = chain_step(chain, batch_digest(int(bid), x, th))
        x = x.to(device, non_blocking=True); th = th.to(device, non_blocking=True)
        if device == 'cuda':
            x = x.contiguous(memory_format=torch.channels_last)
        f = lr_factor(step + 1, cfg['warmup'], cfg['steps'])
        for g, pg in zip(groups, opt.param_groups):
            pg['lr'] = g['base_lr'] * f
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            lg = model(x)
        loss = (-cgd(th, cfg['sigma']) * F.log_softmax(lg.float(), 1)).sum(1).mean()
        if not torch.isfinite(loss):
            raise SystemExit(f'non-finite loss at step {step + 1}')
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, cfg['clip']); opt.step()
        step += 1
        with torch.no_grad():
            d = min(cfg['ema'], (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        if os.environ.get('H13_TEST_KILL_BEFORE_RECEIPT') and step == int(os.environ['H13_TEST_KILL_BEFORE_RECEIPT']) and a.arch == 'tiny':
            import signal
            os.kill(os.getpid(), signal.SIGKILL)   # TEST ONLY: death between an update and its receipt
        if step % cfg['receipt_every'] == 0:
            append_durable(out / 'segments.jsonl', dict(kind='progress', seg_id=seg['seg_id'], step=step))
        done = step - seg['start_step']
        if a.limit and (done == 200 or (done > 200 and done % 2000 == 0)):   # every segment, then periodically
            rate = done * cfg['batch'] / (time.time() - t0)
            end = time.time() + (cfg['steps'] - step) * cfg['batch'] / rate + a.eval_min * 60
            print(json.dumps(dict(step=step, segment_updates=done, forecast_img_s=round(rate, 1), forecast_end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(end)))), flush=True)
            if end > a.limit:
                append_durable(out / 'segments.jsonl', dict(kind='end', seg_id=seg['seg_id'], start_step=seg['start_step'], end_step=step,
                                                            executed_updates=step - seg['start_step'], completed=False, aborted='FORECAST_ABORT', chain=chain))
                print('FORECAST_ABORT', flush=True); sys.exit(3)
        pending = None
        if step in cfg['milestones'] or step == cfg['steps']:
            name = f'{a.run}-s{step:06d}' if step != cfg['steps'] else f'{a.run}-final'
            sd = {k: v.detach().cpu() for k, v in ema.state_dict().items()}
            buf = io.BytesIO(); torch.save(dict(model=sd, ema=sd, step=step, presentations=step * cfg['batch'], config=dict(img_size=224, canvas=224, arch=a.arch, hidden=None, h13=cfg, stream_chain=chain)), buf)
            blob = buf.getvalue(); esha = hashlib.sha256(blob).hexdigest()
            if name in exports and exports[name] != esha:
                raise SystemExit(f'refusing: recomputed export {name} differs from the recorded one')
            exports[name] = esha; pending = (out.parent / name, blob, esha)
        if step % cfg['recovery_every'] == 0 or step == cfg['steps'] or pending:
            save_recovery(out, step, model, ema, opt, groups, chain, cfg, device, exports)   # recovery FIRST, then the export
        if os.environ.get('H13_TEST_KILL_BEFORE_PUBLISH') and pending and step == int(os.environ['H13_TEST_KILL_BEFORE_PUBLISH']) and a.arch == 'tiny':
            import signal
            os.kill(os.getpid(), signal.SIGKILL)   # TEST ONLY: death after the recovery save, before publication
        if pending:
            print('export', pending[0].name, publish_export(*pending), pending[2][:16], flush=True)
        if os.environ.get('H13_TEST_KILL_AT') and step == int(os.environ['H13_TEST_KILL_AT']) and a.arch == 'tiny':
            import signal
            os.kill(os.getpid(), signal.SIGKILL)   # TEST ONLY: abrupt death with live prefetching workers
        if a.stop_after and step >= a.stop_after:   # TEST ONLY: simulated crash (no checkpoint written here)
            break
        if step % 200 == 0:
            print(json.dumps(dict(step=step, loss=round(loss.item(), 4), img_s=round((step - seg_t0) * cfg['batch'] / (time.time() - t0), 1))), flush=True)
    append_durable(out / 'segments.jsonl', dict(kind='end', seg_id=seg['seg_id'], start_step=seg['start_step'], end_step=step,
                                                executed_updates=step - seg['start_step'], completed=step == cfg['steps'], chain=chain,
                                                ts=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    return dict(step=step, chain=chain)


def dose_summary(run_dir, batch, receipt_every):
    """Retained trajectory dose (final end step x batch) vs physical executed updates per segment: exact when an end record
    exists, otherwise a BOUNDED UNKNOWN [last durable receipt, last receipt + receipt_every] (abrupt death; inclusive)."""
    recs = [json.loads(l) for l in open(Path(run_dir) / 'segments.jsonl')]
    segs = {}
    for r in recs:
        s = segs.setdefault(r['seg_id'], dict(start=None, last=None, end=None))
        if r['kind'] == 'start':
            s['start'] = r['start_step']; s['last'] = r['start_step']
        elif r['kind'] == 'progress':
            s['last'] = r['step']
        else:
            s['end'] = r
    lo = hi = 0; detail = []
    for sid, s in segs.items():
        if s['end'] is not None:
            n = s['end']['executed_updates']; lo += n; hi += n; detail.append(dict(seg=sid, exact=n))
        else:
            a_, b_ = s['last'] - s['start'], s['last'] - s['start'] + receipt_every; lo += a_; hi += b_
            detail.append(dict(seg=sid, bounded_unknown=[a_, b_], last_durable_step=s['last']))
    ends = [s['end'] for s in segs.values() if s['end'] is not None]
    final = ends[-1] if ends else None
    return dict(retained_presentations=(final['end_step'] if final else 0) * batch, physical_updates_range=[lo, hi],
                physical_presentations_range=[lo * batch, hi * batch], segments=detail, completed=bool(final and final['completed']))


def evaluate(argv):
    from rotlab import camp_eval

    def load_model(ckpt, pool=None):
        assert pool is None
        ck = torch.load(ckpt, map_location='cpu', weights_only=False); assert ck['config']['arch'] == 'mo_s'
        m = MoS(0.0); m.load_state_dict(ck['ema']); return m.cuda().eval()
    camp_eval.load_model = load_model
    sys.argv = ['camp_eval'] + argv
    camp_eval.main()


def parser():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--arch', choices=['mo_s', 'tiny'], required=True); ap.add_argument('--run', required=True)
    ap.add_argument('--limit', type=float, default=0); ap.add_argument('--eval-min', type=float, default=20.0)
    ap.add_argument('--resume'); ap.add_argument('--resume-sha256'); ap.add_argument('--resume-authority')
    ap.add_argument('--stop-after', type=int, default=0, help='TEST ONLY: simulated crash')
    ap.add_argument('--out-root'); ap.add_argument('--milestones')
    for k in ('steps', 'batch', 'warmup', 'workers', 'recovery_every', 'canvas', 'receipt_every'):
        ap.add_argument('--' + k.replace('_', '-'), type=int)
    return ap


if __name__ == '__main__':
    if sys.argv[1] == 'eval':
        evaluate(sys.argv[2:])
    else:
        a = parser().parse_args()
        if a.arch == 'mo_s' and (any(getattr(a, k) is not None for k in ('steps', 'batch', 'warmup', 'workers', 'recovery_every', 'canvas', 'milestones', 'receipt_every'))
                                 or a.stop_after or os.environ.get('H13_TEST_BLOBS') or os.environ.get('H13_TEST_KILL_AT')):
            raise SystemExit('refusing: the mo_s protocol is fixed; test overrides/fake inputs/kill hooks are CPU-test only')
        if os.environ.get('H13_TEST_BLOBS'):
            from rotlab.test_h13 import fake_blobs
            train(a, fake_blobs())
        else:
            train(a)
