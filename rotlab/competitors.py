#!/usr/bin/env python3
"""Head-to-head against external orientation / roll estimators on our test sets (same pixels for everyone).

Each external model runs in its own environment and writes native outputs; the sign convention is fixed ONCE on
calibration images (Open Images calibration photos turned +15 degrees clockwise) before any test set is scored.
  python -m rotlab.competitors sign  MODEL          # -> DATA/rotlab/competitors/MODEL-sign.json
  python -m rotlab.competitors run   MODEL [SETS]   # -> DATA/rotlab/competitors/MODEL.npz (native angle, confidence)
  python -m rotlab.competitors score                # tables incl. our tiers and the previous SOTA
Models: deepoad, geocalib, perspective, orientany, g3t. The previous SOTA (woehrer) and our tiers are read from
existing prediction files. Licences: see release/SOTA-LANDSCAPE.md (Perspective Fields and G3T: internal only).
"""
import json, os, random, sys, tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from rotlab.core import DATA, circ_err, render

OUT = DATA / 'rotlab/competitors'
DEV = os.environ.get('COMP_DEVICE', 'cuda')   # 'cpu' for CPU speed runs (4 threads)
SETS = ['meva', 'common', 'fresh_diode', 'fair', 'oi', 'fair_deg', 'oi_deg', 'origin', 'rotbench', 'rotbench_full']


# ------------------------------------------------------------------ views (identical to our evaluation sets)
VCACHE = Path(os.environ.get('COMP_VCACHE', '/workspace/rotation-data/rotlab/cviews'))


def views(name):
    """-> (list of PIL images, clockwise ground-truth angles). Reads the lossless cache written by `dump` when present,
    so every model (in its own environment) sees exactly the same pixels."""
    f = VCACHE / f'{name}.pkl'
    if f.exists():
        import io, pickle
        d = pickle.loads(f.read_bytes())
        return [Image.open(io.BytesIO(b)).convert('RGB') for b in d['png']], np.asarray(d['theta'])
    r = _views(name); return r[0], r[1]


def cmd_dump():
    """Render every test set once (our environment) and store PNG bytes + angles."""
    import io, pickle
    VCACHE.mkdir(parents=True, exist_ok=True)
    for s in (sys.argv[2].split(',') if len(sys.argv) > 2 else SETS):
        r = _views(s); vs, th = r[0], r[1]; png = []
        for v in vs:
            b = io.BytesIO(); v.save(b, format='PNG'); png.append(b.getvalue())
        extra = dict(split=r[2]) if len(r) > 2 else {}
        (VCACHE / f'{s}.pkl').write_bytes(pickle.dumps(dict(png=png, theta=np.asarray(th), **extra)))
        print('dumped', s, len(png), flush=True)


def _views(name):
    if name in ('rotbench', 'rotbench_full'):
        # RotBench (arXiv 2508.13968; HF tianyin/RotBench, Apache-2.0; images from Spatial-MM). Official protocol
        # (create_datasets.py): PIL img.rotate(deg) for deg in 0/90/180/270, counter-clockwise, WITHOUT expand (canvas keeps
        # the original size; corners clipped). 'rotbench_full' = our CCTV-realistic variant (expand=True, whole frame).
        # Large (300) and Small (50) are reported separately; the split labels are stored in the view cache.
        out, th, split = [], [], []
        for sp, im in rotbench_images():
            for ccw in (0, 90, 180, 270):
                out.append(im.rotate(ccw, expand=(name == 'rotbench_full')) if ccw else im); th.append((-ccw) % 360); split.append(sp)
        return out, np.array(th, float), np.array(split)
    if name == 'fair':
        from rotlab.fair_photo import fair_views
        _, v, t = fair_views(7); return v, np.asarray(t)
    if name == 'oi':
        from rotlab.oi_test import views as oi_views
        v, t = oi_views('fixed'); return v, np.asarray(t)
    if name == 'meva':
        from rotlab.meva_test import views as meva_views
        mv = meva_views('dev'); return [x['view'] for x in mv], np.array([x['theta'] for x in mv])
    if name in ('fair_deg', 'oi_deg'):
        from rotlab.degraded_sets import views as deg_views
        v, t = deg_views()[name]; return v, np.asarray(t)
    from rotlab.core import load_pack
    rows = [r for r in load_pack() if r['panel'] == name]
    return [Image.open(r['image']['path']).convert('RGB') for r in rows], np.array([r['target_degrees'] for r in rows])


def rotbench_images():
    import io, glob
    import pandas as pd
    from huggingface_hub import snapshot_download
    d = snapshot_download('tianyin/RotBench', repo_type='dataset'); out = []
    for f in sorted(glob.glob(f'{d}/**/*.parquet', recursive=True)):
        sp = 'small' if 'small' in f.lower() else 'large'
        for rec in pd.read_parquet(f)['image']:
            out.append((sp, Image.open(io.BytesIO(rec['bytes'])).convert('RGB')))
    return out


def sign_views(n=20, turn=15.0):
    from rotlab.calib import sources
    src = sources()['oi_cal'][:n]; out = []
    for path, _ in src:
        im = Image.open(path).convert('RGB'); W, H = im.size; d = min(W, H) - 4; cw, ch = 0.8 * d, 0.6 * d
        out.append(render(im, turn, cw, ch, round(cw), round(ch)))
    return out, turn


# ------------------------------------------------------------------ adapters: PIL -> (native angle deg, confidence or nan)
class DeepOAD:
    """Deep-OAD OAD-360 (ViT-B/16 regression), community ONNX conversion; output = counter-clockwise degrees."""
    def __init__(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        p = hf_hub_download('Chuckame/deep-image-orientation-angle-detection', 'deep-image-orientation-angle-detection.onnx')
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        self.s = ort.InferenceSession(p, so, providers=(['CUDAExecutionProvider'] if DEV == 'cuda' else []) + ['CPUExecutionProvider']); self.i = self.s.get_inputs()[0]
        self.nhwc = self.i.shape[-1] == 3

    def __call__(self, im):
        x = np.asarray(im.convert('RGB').resize((224, 224), Image.BILINEAR), np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = x[None] if self.nhwc else x.transpose(2, 0, 1)[None]
        y = float(np.ravel(self.s.run(None, {self.i.name: x.astype(np.float32)})[0])[0])
        return y, float('nan')


class GeoCalibM:
    """GeoCalib (pinhole weights); roll in radians, roll_uncertainty used as (inverse) confidence."""
    def __init__(self):
        import torch
        from geocalib import GeoCalib
        self.t = torch; self.m = GeoCalib(weights='pinhole').to(DEV).eval()

    def __call__(self, im):
        x = self.t.from_numpy(np.asarray(im.convert('RGB'), np.float32) / 255.0).permute(2, 0, 1).to(DEV)
        with self.t.no_grad():
            r = self.m.calibrate(x)
        roll = float(r['gravity'].roll.squeeze().item()); unc = float(r['roll_uncertainty'].squeeze().item())
        return float(np.degrees(roll)), -unc


class Perspective:
    """Perspective Fields ParamNet (360Cities+EDINA, centred principal point); INTERNAL ONLY (Adobe research licence)."""
    def __init__(self):
        from perspective2d import PerspectiveFields
        self.m = PerspectiveFields('Paramnet-360Cities-edina-centered').eval().to(DEV)

    def __call__(self, im):
        import torch
        with torch.no_grad():
            p = self.m.inference(img_bgr=np.asarray(im.convert('RGB'))[:, :, ::-1].copy())
        return float(p['pred_roll'].item() if hasattr(p['pred_roll'], 'item') else p['pred_roll']), float('nan')


class OrientAnything:
    """Orient Anything V1 (DINOv2-L, released demo weights ronormsigma1), whole image, no background removal."""
    def __init__(self):
        import torch
        sys.path.insert(0, '/root/crepos/Orient-Anything')
        from vision_tower import DINOv2_MLP
        from transformers import AutoImageProcessor
        from huggingface_hub import hf_hub_download
        from paths import DINO_LARGE
        ck = hf_hub_download('Viglong/Orient-Anything', 'ronormsigma1/dino_weight.pt')
        self.m = DINOv2_MLP(dino_mode='large', in_dim=1024, out_dim=360 + 180 + 360 + 2, evaluate=True, mask_dino=False, frozen_back=False)
        self.m.load_state_dict(torch.load(ck, map_location='cpu')); self.m = self.m.to(DEV).eval()
        self.pre = AutoImageProcessor.from_pretrained(DINO_LARGE); self.t = torch

    def __call__(self, im):
        x = self.pre(images=im.convert('RGB'))
        x['pixel_values'] = self.t.from_numpy(np.array(x['pixel_values'])).to(DEV)
        with self.t.no_grad():
            p = self.m(x)
        rot = float(self.t.argmax(p[:, 540:900], -1).item()) - 180.0
        conf = float(self.t.softmax(p[:, -2:], -1)[0][0].item())
        return rot, conf


class G3T:
    """G3T (VGGT-based gravity model), single view, local camera-to-gravity head; roll about the optical axis.
    Treat as non-commercial / internal only."""
    def __init__(self):
        import torch
        sys.path.insert(0, '/root/crepos/g3t')
        from vggt.utils.inference_utils import setup_model, run_g3t_model
        from vggt.utils.load_fn import load_and_preprocess_images_square
        self.t, self.run_model, self.load = torch, run_g3t_model, load_and_preprocess_images_square
        self.m = setup_model(None).eval().to(DEV)
        self.tmp = Path(tempfile.mkdtemp())

    def __call__(self, im):
        import torch.nn.functional as F
        f = self.tmp / 'v.png'; im.convert('RGB').save(f)
        x, _ = self.load([str(f)], 1024); x = F.interpolate(x.to(DEV), size=(518, 518), mode='bilinear', align_corners=False)
        _, _, _, _, _, c2g = self.run_model(images=x, model=self.m, points_source='point_head')
        R = c2g[0, :3, :3].float().cpu().numpy()
        return float(np.degrees(np.arctan2(-R[0, 1], R[0, 0]))), float('nan')


class Woehrer:
    """Previous SOTA: the released MambaOut-B CGD checkpoint via the validated faithful adapter (clockwise output)."""
    def __init__(self):
        from rotlab.fair_photo import incumbent
        self.f = incumbent(4)

    def __call__(self, im):
        return float(self.f(im)), float('nan')


class Ours:
    """Our released tiers' building blocks (EMA weights, letterbox input, decoded angle + confidence); clockwise output."""
    CFG = dict(ours_nano=('S2-vits-multires', 112), ours_fast=('S2-vits-multires', 224), ours_large=('soup-G7-G3-a0.5', 224))

    def __init__(self, name):
        import torch
        from rotlab.model import RotNet
        run, self.cv = self.CFG[name]
        ck = torch.load(DATA / f'rotlab/runs/{run}/final.pt', map_location='cpu', weights_only=False)
        self.m = RotNet(img_size=224, pretrained=False, dynamic=True, arch=ck['config'].get('arch', 's')).to(DEV).eval()
        self.m.load_state_dict(ck['ema'])

    def __call__(self, im):
        from rotlab.core import decode, letterbox
        from rotlab.evaluate import confidence, predict
        prob = predict(self.m, letterbox(im, self.cv)[None], device=DEV); p = decode(prob)
        return float(p[0]), float(confidence(prob, p)[0])


ADAPTERS = dict(deepoad=DeepOAD, geocalib=GeoCalibM, perspective=Perspective, orientany=OrientAnything, g3t=G3T, woehrer=Woehrer,
                ours_nano=lambda: Ours('ours_nano'), ours_fast=lambda: Ours('ours_fast'), ours_large=lambda: Ours('ours_large'))


# ------------------------------------------------------------------ commands
def cmd_sign(model):
    m = ADAPTERS[model](); vs, turn = sign_views()
    nat = np.array([m(v)[0] for v in vs])
    ep, en = circ_err(nat % 360, np.full(len(nat), turn)), circ_err((-nat) % 360, np.full(len(nat), turn))
    s = 1 if np.median(ep) <= np.median(en) else -1
    rec = dict(model=model, sign=s, turn_cw=turn, n=len(nat), median_err_plus=float(np.median(ep)), median_err_minus=float(np.median(en)),
               native=[round(float(v), 2) for v in nat])
    OUT.mkdir(parents=True, exist_ok=True); (OUT / f'{model}-sign.json').write_text(json.dumps(rec, indent=1)); print(json.dumps(rec))


def cmd_run(model, sets, shard=None):
    """shard 'k/n': process images i with i % n == k of each set into MODEL__SET__k-of-n.npz (merged by the scorer)."""
    assert (OUT / f'{model}-sign.json').exists(), 'run the sign check first'
    m = ADAPTERS[model](); f = OUT / f'{model}.npz'
    res = dict(np.load(f)) if f.exists() else {}
    for s in sets:
        if f'{s}_pred' in res:
            continue
        vs, th = views(s)
        if shard:
            k, n = (int(v) for v in shard.split('/')); idx = np.arange(len(vs))[k::n]
            sf = OUT / f'{model}__{s}__{k}-of-{n}.npz'
            if sf.exists(): continue
            out = [m(vs[i]) for i in idx]
            np.savez(sf, idx=idx, pred=np.array([o[0] for o in out]), conf=np.array([o[1] for o in out]), theta=th[idx], n=len(vs))
            print(model, s, f'shard {k}/{n}', len(idx), flush=True); continue
        out = [m(v) for v in vs]
        res[f'{s}_pred'] = np.array([o[0] for o in out]); res[f'{s}_conf'] = np.array([o[1] for o in out]); res[f'{s}_theta'] = th
        np.savez(f, **res); print(model, s, len(vs), flush=True)


def merge_shards(model):
    """Fold complete shard sets into MODEL.npz."""
    import glob, re
    f = OUT / f'{model}.npz'; res = dict(np.load(f)) if f.exists() else {}
    groups = {}
    for p in glob.glob(str(OUT / f'{model}__*__*-of-*.npz')):
        mm = re.search(r'__(.+)__(\d+)-of-(\d+)\.npz$', p); groups.setdefault((mm.group(1), int(mm.group(3))), []).append(p)
    for (s, n), ps in groups.items():
        if f'{s}_pred' in res or len(ps) != n: continue
        zs = [np.load(p) for p in ps]; N = int(zs[0]['n'])
        pred, conf, th = np.zeros(N), np.zeros(N), np.zeros(N)
        for z in zs:
            pred[z['idx']] = z['pred']; conf[z['idx']] = z['conf']; th[z['idx']] = z['theta']
        res[f'{s}_pred'], res[f'{s}_conf'], res[f'{s}_theta'] = pred, conf, th
    if res: np.savez(f, **res)


def cmd_speed(model, runs=50):
    """End-to-end single-image latency (batch 1, incl. the model's own preprocessing) on this machine's GPU, after
    warm-up; peak CUDA memory via torch where the model uses torch. Input: a fixed 640x480 view from the fair-photo set."""
    import time
    vs, _ = views('fair'); im = vs[0].resize((640, 480))
    if DEV == 'cpu':
        runs = int(os.environ.get('COMP_RUNS', 10))
    try:
        import torch
        if DEV == 'cpu': torch.set_num_threads(4)
        else: torch.cuda.reset_peak_memory_stats()
        has_t = DEV == 'cuda'
    except Exception:
        has_t = False
    m = ADAPTERS[model]()
    for _ in range(2 if DEV == 'cpu' else 5): m(im)
    t = []
    for _ in range(runs):
        if has_t: torch.cuda.synchronize()
        a = time.perf_counter(); m(im)
        if has_t: torch.cuda.synchronize()
        t.append((time.perf_counter() - a) * 1000)
    vram = None
    if DEV == 'cuda':   # total per-process GPU memory (nvidia-smi), comparable across PyTorch and ONNX Runtime models
        try:
            import subprocess
            out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True).stdout
            vram = sum(int(mb) for p_, mb in (l.split(',') for l in out.splitlines()) if int(p_) == os.getpid())
        except Exception:
            pass
    t = sorted(t); gpu = ''
    try:
        import subprocess
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    if DEV == 'cpu':
        try: gpu = 'CPU ' + next(l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')) + ' (4 threads)'
        except Exception: gpu = 'CPU'
    rec = dict(model=model, device=DEV, runs=runs, hw=gpu, p50_ms=round(t[len(t) // 2], 2), mean_ms=round(sum(t) / len(t), 2), p95_ms=round(t[max(0, int(len(t) * .95) - 1)], 2),
               peak_torch_cuda_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20) if has_t else None, process_vram_mb=vram)
    with open(OUT / 'speed.jsonl', 'a') as f: f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec), flush=True)


def ours(stem):
    z = np.load(DATA / f'rotlab/suite/{stem}.npz'); return {k: z[k] for k in z.files}


def woehrer():
    from rotlab.core import load_pack
    rows = load_pack(); pan = np.array([r['panel'] for r in rows]); out = {}
    for p in ('common', 'fresh_diode', 'origin'):
        out[p] = (np.array([r['faithful'] for r in rows])[pan == p], np.array([r['target_degrees'] for r in rows])[pan == p])
    fz = np.load(next((DATA / 'rotlab/fair-photo').glob('rows-seed7-*.npz'))); out['fair'] = (fz['incumbent'], fz['theta'])
    oz = np.load(DATA / 'rotlab/oi-test/incumbent-fixed.npz'); out['oi'] = (oz['pred'], oz['theta'])
    mz = np.load(DATA / 'rotlab/meva-test-v1/results-dev/incumbent.npz'); out['meva'] = (mz['pred'], mz['theta'])
    dz = np.load(DATA / 'rotlab/suite/incumbent-deg.npz')
    for k in ('fair_deg', 'oi_deg'): out[k] = (dz[f'{k}_pred'], dz[f'{k}_theta'])
    return out


def cmd_score():
    from rotlab.final_stats import Tier
    for mdl in ADAPTERS: merge_shards(mdl)
    rows = {}
    rows['Previous SOTA (Woehrer 2026)'] = woehrer()
    if (OUT / 'woehrer.npz').exists():
        z = np.load(OUT / 'woehrer.npz')
        for rb in ('rotbench', 'rotbench_full'):
            if f'{rb}_pred' in z.files: rows['Previous SOTA (Woehrer 2026)'][rb] = (z[f'{rb}_pred'] % 360, z[f'{rb}_theta'])
    for model in ('deepoad', 'geocalib', 'perspective', 'orientany', 'g3t'):
        f = OUT / f'{model}.npz'
        if not f.exists(): continue
        s = json.loads((OUT / f'{model}-sign.json').read_text())['sign']; z = np.load(f)
        rows[{'deepoad': 'Deep-OAD (OAD-360)', 'geocalib': 'GeoCalib', 'perspective': 'Perspective Fields',
              'orientany': 'Orient Anything V1', 'g3t': 'G3T'}[model]] = {k: ((s * z[f'{k}_pred']) % 360, z[f'{k}_theta']) for k in SETS if f'{k}_pred' in z.files}
    for name, sm, lg, route, scv in (('Ours: Fast', 'S2-vits-multires', None, 0, 'c224'), ('Ours: Max', 'S2-vits-multires', 'soup-G7-G3-a0.5', 0.20, 'c224')):
        t = Tier(name, sm, lg, route); t.calibrate(scv, 'c224'); a = ours(f'{sm}-c224'); ad = ours(f'{sm}-deg-c224'); r = {}
        for k in SETS:
            src = ad if k.endswith('_deg') else a
            if f'{k}_p_pred' not in src: continue
            p = src[f'{k}_p_pred'].copy(); c = src[f'{k}_p_conf']
            if lg:
                b = ours(f'{lg}-{"deg-" if k.endswith("_deg") else ""}c224'); m = c < t.t_route; p[m] = b[f'{k}_p_pred'][m]
            r[k] = (p, src[f'{k}_theta'])
        fz, lz = OUT / 'ours_fast.npz', OUT / 'ours_large.npz'
        for rb in ('rotbench', 'rotbench_full'):
            if fz.exists() and f'{rb}_pred' in np.load(fz).files and (lg is None or lz.exists()):
                a2 = np.load(fz); p = a2[f'{rb}_pred'].copy()
                if lg:
                    m = a2[f'{rb}_conf'] < t.t_route; p[m] = np.load(lz)[f'{rb}_pred'][m]
                r[rb] = (p % 360, a2[f'{rb}_theta'])
        rows[name] = r
    d0 = lambda a: np.abs(((a + 180) % 360) - 180)
    for rb in ('rotbench', 'rotbench_full'):
        if not any(rb in r for r in rows.values()): continue
        import pickle
        split = pickle.loads((VCACHE / f'{rb}.pkl').read_bytes())['split']
        title = 'official protocol (PIL rotate, no expand)' if rb == 'rotbench' else 'our full-frame variant (expand=True)'
        for sp in ('large', 'small'):
            m0 = split == sp
            print(f'## RotBench-{sp.capitalize()} ({m0.sum() // 4} photos), {title}: 4-way accuracy per counter-clockwise rotation\n')
            print('| Model | 0° | 90° | 180° | 270° | mean | within 10° |'); print('|---|---|---|---|---|---|---|')
            for name, r in rows.items():
                if rb not in r: continue
                p, th = r[rb]; p, th = p[m0], th[m0]; q = (np.round(p / 90) % 4) * 90; ok = circ_err(q, th) < 1
                cells = [f'{ok[th == t].mean():.2f}' for t in (0, 270, 180, 90)]
                print(f'| {name} | ' + ' | '.join(cells) + f' | {ok.mean():.2f} | {(circ_err(p, th) <= 10).mean() * 100:.1f}% |')
            print()
    print('## Within 10° — full circle (upside-down errors ≥150° in brackets)\n')
    print('| Model | ' + ' | '.join(SETS) + ' |'); print('|---' * (len(SETS) + 1) + '|')
    for name, r in rows.items():
        cells = []
        for k in SETS:
            if k not in r: cells.append('—'); continue
            e = circ_err(r[k][0], r[k][1]); cells.append(f'{(e <= 10).mean() * 100:.1f}% ({(e >= 150).sum()})')
        print(f'| {name} | ' + ' | '.join(cells) + ' |')
    print('\n## Within 10° — images whose true roll is within ±45° (scope of calibration models)\n')
    print('| Model | ' + ' | '.join(SETS) + ' |'); print('|---' * (len(SETS) + 1) + '|')
    for name, r in rows.items():
        cells = []
        for k in SETS:
            if k not in r: cells.append('—'); continue
            m = d0(r[k][1]) <= 45; e = circ_err(r[k][0][m], r[k][1][m]); cells.append(f'{(e <= 10).mean() * 100:.1f}% (n={m.sum()})')
        print(f'| {name} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'sign': cmd_sign(sys.argv[2])
    elif cmd == 'run': cmd_run(sys.argv[2], sys.argv[3].split(',') if len(sys.argv) > 3 else SETS, sys.argv[4] if len(sys.argv) > 4 else None)
    elif cmd == 'merge': merge_shards(sys.argv[2])
    elif cmd == 'speed': cmd_speed(sys.argv[2])
    elif cmd == 'dump': cmd_dump()
    else: cmd_score()
