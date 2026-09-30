#!/usr/bin/env python3
"""[lead-push] Extended RotNet factory: the accepted DINOv2 lineage (s/b/l, byte-identical behaviour to rotlab.model.RotNet)
plus RESEARCH-ONLY backbones (owner decision 2026-09-27: research runs allowed; NOT shipped unless the owner accepts them):

  key     timm model (pinned HF revision / weight SHA-256 in SPECS)   status
  s       vit_small_patch14_dinov2.lvd142m                            accepted (7 Sep 2026)
  b       vit_base_patch14_dinov2.lvd142m                             conditionally accepted (23 Sep 2026)
  l       vit_large_patch14_dinov2.lvd142m                            conditionally accepted (23 Sep 2026)
  b_reg   vit_base_patch14_reg4_dinov2.lvd142m                        research-only (registers not in the 23 Sep decision)
  pe_s    vit_pe_spatial_small_patch16_512.fb   (Meta PE-Spatial S/16, Apache-2.0)   research-only
  tips_b  vit_base_patch14_reg1_tipsv2.webli    (Google TIPSv2 B/14, Apache-2.0)     research-only

All six are defined by timm (Apache-2.0) code; no model code is copied from the PE or TIPS repositories.

Head input, defined the same way for every backbone:
  concat(g, mean of PATCH tokens) -> LayerNorm -> Linear(2D, 1024) -> GELU -> Linear(1024, 360)
  g = the CLS token if the backbone has one (all six do); else the pretrained attention-pool (MAP) output if it has one;
  else the token-wise max over patch tokens (so the two halves never duplicate). Prefix tokens (CLS + registers:
  b_reg 1+4, tips_b 1+1) are excluded from the patch mean (rotlab.model.RotNet's t[:, 1:] would average registers in).
  Tokens are per-token LayerNormed before pooling: DINOv2/TIPS by their pretrained final norm; PE-Spatial has no final
  norm (timm use_post_transformer_norm=False), so a fresh LayerNorm `tok_norm` (weight 1, bias 0; trained at head LR)
  is added. s/b/l have no extra module, buffer or op: state-dict keys, forward and ONNX graph equal rotlab.model.RotNet.
Input (data pipeline unchanged: letterboxed square canvas, ImageNet mean/std = rotlab.core.MEAN/STD):
  the wrapper maps it to the backbone's pretraining normalisation with a fixed per-channel affine (non-persistent
  buffers, not in the state dict): pe_s (x*std_in + mean_in - 0.5)/0.5 (PE: Normalize([.5]*3, [.5]*3)); tips_b
  x*std_in + mean_in ([0, 1], "no ImageNet normalization", TIPSv2 model card); DINOv2 keys: no op.
  Canvases that are not a multiple of the patch (PE /16 at 140/168/196/252) are resized in the wrapper, bilinear without
  antialias (exportable at opset 18), to the nearest multiple, half up: 140->144, 168->176, 196->192, 252->256. Train
  pe_s on multiples of 16 (--multires 112,144,176,208,224) so training never hits the resize.
Dynamic size: dynamic=True resamples the absolute pos-embed per input grid (timm, bicubic) and, for PE, recomputes the
  2D RoPE for the grid (timm ref_feat_shape 32x32 scaling). Export bakes the resampled pos-embed (static_copy), as
  rotlab.export does; PE exports with timm's non-fused attention (the fused path fails TorchScript export).

  python -m rotlab.model_x describe [--canvases 112,140,168,196,224]     # layout, params, GMACs (random weights)
  python -m rotlab.model_x bench [--threads 4 --rounds 12 --targets pe_s@224,tips_b@140,...] [--json OUT]
  python -m rotlab.model_x memest [--batch 128]                           # bf16 activation-memory estimate (CPU)
  python -m rotlab.model_x export CKPT|ARCH OUT_PREFIX --canvas 224 [--int8]  # ONNX (use instead of rotlab.export)
Local weights root: /workspace/rotation-data/quarantine/external-weights (override: ROTLAB_EXT_WEIGHTS).
Weights are refused unless size and SHA-256 equal the pin (fetch with rotlab.fetch_weights, which writes receipts).
"""
from __future__ import annotations

import argparse, hashlib, io, json, os, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers import resample_abs_pos_embed

from rotlab.core import MEAN as IN_MEAN, STD as IN_STD
from rotlab.model import ARCH as ARCH0, RotNet as RotNet0

EXT0 = '/workspace/rotation-data/quarantine/external-weights'
EXT = os.environ.get('ROTLAB_EXT_WEIGHTS', EXT0)
IMNET = (tuple(float(v) for v in IN_MEAN), tuple(float(v) for v in IN_STD))


def _ext(p):
    """Pod path -> path under the (possibly overridden) external-weights root."""
    return str(Path(EXT) / Path(p).relative_to(EXT0)) if p.startswith(EXT0) else p


@dataclass(frozen=True)
class Spec:
    timm: str                    # timm model name incl. pretrained tag
    weights: str                 # local weights file (pod path; ROTLAB_EXT_WEIGHTS relocates it)
    repo: str                    # Hugging Face repo the weights file is fetched from
    revision: str                # pinned commit
    sha256: str                  # pinned SHA-256 of the weights file (== HF LFS oid at `revision`)
    size: int
    status: str
    patch: int = 14
    mean: tuple = IMNET[0]       # normalisation the backbone was pretrained with
    std: tuple = IMNET[1]
    export_unfused: bool = False  # TorchScript ONNX export needs timm's non-fused attention (RoPE ViTs)
    kwargs: tuple = ()            # extra timm.create_model kwargs as (name, value) pairs


_ACC_B = 'conditionally accepted (owner 2026-09-23; ship only if better AND faster than the incumbent)'
SPECS = {
    's': Spec('vit_small_patch14_dinov2.lvd142m', ARCH0['s'][1], 'timm/vit_small_patch14_dinov2.lvd142m',
              '4610ca143709d58a633b6397a74412c2c3842454',
              '04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081', 88240510, 'accepted (owner 2026-09-07)'),
    'b': Spec('vit_base_patch14_dinov2.lvd142m', ARCH0['b'][1], 'timm/vit_base_patch14_dinov2.lvd142m',
              '4685c99dabffe5affac90bd99dbffd25801ae58d',
              '55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b', 346334872, _ACC_B),
    'l': Spec('vit_large_patch14_dinov2.lvd142m', ARCH0['l'][1], 'timm/vit_large_patch14_dinov2.lvd142m',
              '4741e1cafbf45415e77074bb0cb42dba76c8684a',
              '0424a5d1b515278cba3c6640ccbeaacc41de59d3a93df0dd5e494285eea2b355', 1217502758, _ACC_B),
    'b_reg': Spec('vit_base_patch14_reg4_dinov2.lvd142m', f'{EXT0}/dinov2_base_reg4-lvd142m-3b06466a/model.safetensors',
                  'timm/vit_base_patch14_reg4_dinov2.lvd142m', '3b06466a5548c52b8b98822e1390987695fcbf82',
                  'c24ecfb4a1d8ca79193f6b9efcffc461872a09ddac43a0931357e4802931a006', 346344168,
                  'research-only (DINOv2 reg4: same Apache-2.0 release, not named in the 2026-09-23 decision)'),
    'pe_s': Spec('vit_pe_spatial_small_patch16_512.fb', f'{EXT0}/pe_spatial_small_patch16_512-fb-bde8c2a1/model.safetensors',
                 'timm/vit_pe_spatial_small_patch16_512.fb', 'bde8c2a1ca579f414140672d311c3a4fcf1058a7',
                 'ed334e352f8490867ca875ce13f8db766497f32a30f88b3a6c87da89a2649229', 87946528,
                 'research-only (owner 2026-09-27)', patch=16, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), export_unfused=True),
    'tips_b': Spec('vit_base_patch14_reg1_tipsv2.webli', f'{EXT0}/tipsv2_base_patch14_reg1-webli-85ea5ee3/model.safetensors',
                   'timm/vit_base_patch14_reg1_tipsv2.webli', '85ea5ee34f6e0fb867e8161465d735097cc11189',
                   '943854100ad2e688d8c8b77403358680fa5f9a4920c5c76fc5efeaa188f93cdc', 345275112,
                   'research-only (owner 2026-09-27)', mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
}
# Pico lane (29 Sep 2026): truncated DINOv2-S, same pinned weights as 's', first k blocks kept (rotlab.model.DEPTH)
from dataclasses import replace as _replace
from rotlab.model import DEPTH, truncate_blocks, truncate_state_dict  # noqa: E402
for _k, _d in DEPTH.items():
    SPECS[_k] = _replace(SPECS['s'], status=f'accepted DINOv2-S weights, first {_d} of 12 blocks (Pico research, 29 Sep 2026)')
ARCH = {k: (s.timm, _ext(s.weights)) for k, s in SPECS.items()}      # rotlab.model.ARCH-compatible view


def align(n, p):
    """Nearest multiple of the patch, half up (140 -> 144 for /16); multiples pass unchanged."""
    return max(p, (int(n) + p // 2) // p * p)


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def verify_weights(arch):
    """Fail closed unless the local weights file has the pinned size and SHA-256."""
    sp = SPECS[arch]; f = Path(_ext(sp.weights))
    if not f.is_file():
        raise FileNotFoundError(f'{arch}: pinned weights missing at {f} (python -m rotlab.fetch_weights fetch {arch})')
    if f.stat().st_size != sp.size or sha256_file(f) != sp.sha256:
        raise RuntimeError(f'{arch}: {f} does not match the pinned size/SHA-256 ({sp.repo}@{sp.revision[:8]}); refusing')
    return str(f)


def renorm_coeffs(arch):
    """x_backbone = x_pipeline * scale + shift, per channel (pipeline = ImageNet mean/std)."""
    sp = SPECS[arch]
    m0, s0 = np.asarray(IMNET[0]), np.asarray(IMNET[1]); m1, s1 = np.asarray(sp.mean), np.asarray(sp.std)
    return s0 / s1, (m0 - m1) / s1


def set_fused_attn(model, on):
    for mod in model.modules():
        if hasattr(mod, 'fused_attn'):
            mod.fused_attn = bool(on)
    return model


class RotNetX(RotNet0):
    """Drop-in for rotlab.model.RotNet (same constructor signature, `pool`, `backbone`, `head`) over SPECS."""

    def __init__(self, img_size=224, drop_path=0.1, hidden=1024, bins=360, pretrained=True, dynamic=False, arch='s'):
        nn.Module.__init__(self)
        if arch not in SPECS:
            raise ValueError(f'unknown arch {arch!r}; choose from {sorted(SPECS)}')
        sp = SPECS[arch]
        self.arch, self.patch = arch, sp.patch
        kw = dict(pretrained=pretrained, img_size=align(img_size, sp.patch), num_classes=0, drop_path_rate=drop_path,
                  dynamic_img_size=dynamic, **dict(sp.kwargs))
        if pretrained:
            kw['pretrained_cfg_overlay'] = dict(file=verify_weights(arch))
        self.backbone = bb = truncate_blocks(timm.create_model(sp.timm, **kw), arch)
        d = bb.embed_dim
        self.npre = int(getattr(bb, 'num_prefix_tokens', 0))
        self.global_token = ('cls' if getattr(bb, 'cls_token', None) is not None else
                             'map' if getattr(bb, 'attn_pool', None) is not None else 'max')
        self.tok_norm = nn.LayerNorm(d, eps=1e-6) if isinstance(getattr(bb, 'norm', None), nn.Identity) else None
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, hidden), nn.GELU(), nn.Linear(hidden, bins))
        self.pool = None     # campaign: optional (k, p) static token pooling after block k (VisionTransformer only)
        scale, shift = renorm_coeffs(arch)
        self.renorm = not (np.allclose(scale, 1) and np.allclose(shift, 0))
        if self.renorm:
            self.register_buffer('in_scale', torch.tensor(scale, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
            self.register_buffer('in_shift', torch.tensor(shift, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

    # ------------------------------------------------------------------ forward
    def adapt(self, x):
        """Pipeline canvas -> backbone input (patch-multiple size, backbone normalisation). No op for s/b/l/b_reg."""
        H, W = int(x.shape[-2]), int(x.shape[-1]); p = self.patch
        if H % p or W % p:
            x = F.interpolate(x, size=(align(H, p), align(W, p)), mode='bilinear', align_corners=False)
        if self.renorm:
            x = x * self.in_scale + self.in_shift
        return x

    def tokens(self, x):
        x = self.adapt(x)
        return self.backbone.forward_features(x) if self.pool is None else self._pooled_features(x)

    def pool_tokens(self, t):
        """(B, npre + N, D) backbone tokens -> (B, 2D) head input."""
        if self.tok_norm is not None:
            t = self.tok_norm(t)
        g = t[:, 0] if self.global_token == 'cls' else None     # same op order as rotlab.model.RotNet (identical ONNX graph)
        p = t[:, self.npre:]
        if self.global_token == 'map':
            g = self.backbone.attn_pool(t)
        elif self.global_token == 'max':
            g = p.amax(1)
        return torch.cat([g, p.mean(1)], 1)

    def features(self, x):
        return self.pool_tokens(self.tokens(x))

    def forward(self, x):
        return self.head(self.features(x))

    def _pooled_features(self, x):
        from timm.models.vision_transformer import VisionTransformer
        from rotlab.camp_eval import area_matrix
        bb, (k, p), n0 = self.backbone, self.pool, self.npre
        if not isinstance(bb, VisionTransformer):
            raise NotImplementedError(f'token pooling is implemented for timm VisionTransformer only, not {type(bb).__name__}')
        t = bb.patch_embed(x); t = bb._pos_embed(t); t = bb.patch_drop(t); t = bb.norm_pre(t)
        for i, blk in enumerate(bb.blocks):
            if i == k:
                g = round((t.shape[1] - n0) ** 0.5)
                W = area_matrix(g, p).to(device=t.device, dtype=t.dtype)
                t = torch.cat([t[:, :n0], torch.matmul(W, t[:, n0:])], 1)
            t = blk(t)
        return bb.norm(t)

    def describe(self):
        sp = SPECS[self.arch]; sc, sh = renorm_coeffs(self.arch)
        return dict(arch=self.arch, timm=sp.timm, status=sp.status, backbone_class=type(self.backbone).__name__,
                    patch=self.patch, prefix_tokens=self.npre, global_token=self.global_token,
                    token_norm='pretrained final norm' if self.tok_norm is None else 'fresh LayerNorm (tok_norm)',
                    input_map=('identity' if not self.renorm else f'x*{np.round(sc, 4).tolist()} + {np.round(sh, 4).tolist()}'),
                    head_input=f'concat({self.global_token}, mean(patch tokens[{self.npre}:])) = {2 * self.backbone.embed_dim}',
                    pin=dict(repo=sp.repo, revision=sp.revision, sha256=sp.sha256, size=sp.size))


RotNet = RotNetX   # `from rotlab.model_x import RotNet, param_groups` is the whole integration


def param_groups(model, lr, head_lr, wd, lld=0.75):
    """rotlab.model.param_groups generalised: layer-wise decay over backbone blocks (pre-block params incl. PE's
    norm_pre at depth 0, post-block params at the top); every non-backbone param (head, tok_norm) at head_lr;
    no weight decay on 1-D params, pos/cls/reg tokens. Identical groups to rotlab.model.param_groups for s/b/l."""
    bb = model.backbone; n = len(bb.blocks); groups = {}
    first = ('patch_embed', 'cls_token', 'pos_embed', 'reg_token', 'mask_token', 'norm_pre', 'rope')

    def add(p, name, scale):
        nd = p.ndim == 1 or name.endswith(('pos_embed', 'cls_token', 'reg_token')) or 'norm' in name
        g = groups.setdefault((scale, nd), dict(params=[], lr_scale=scale, weight_decay=0.0 if nd else wd))
        g['params'].append(p)

    for name, p in bb.named_parameters():
        layer = int(name.split('.')[1]) + 1 if name.startswith('blocks.') else 0 if name.startswith(first) else n
        add(p, name, lr * lld ** (n - layer))
    for name, p in model.named_parameters():
        if not name.startswith('backbone.'):
            add(p, name, head_lr)
    out = []
    for g in groups.values():
        g['lr'] = g['lr_scale']; out.append(g)
    return out


# ---------------------------------------------------------------- export helpers
def static_copy(model, canvas):
    """Eval copy at a fixed square canvas with the runtime pos-embed resample baked in (as rotlab.export does)."""
    h = model.head
    m = RotNetX(img_size=canvas, drop_path=0.0, hidden=h[1].out_features, bins=h[3].out_features, pretrained=False,
                dynamic=False, arch=model.arch)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bb, sb = model.backbone, m.backbone
    if getattr(bb, 'pos_embed', None) is not None and tuple(sd['backbone.pos_embed'].shape) != tuple(sb.pos_embed.shape):
        npt = 0 if getattr(bb, 'no_embed_class', False) else bb.num_prefix_tokens
        sd['backbone.pos_embed'] = resample_abs_pos_embed(sd['backbone.pos_embed'], new_size=sb.patch_embed.grid_size,
                                                          old_size=bb.patch_embed.grid_size, num_prefix_tokens=npt)
    m.load_state_dict(sd); m.pool = model.pool
    return m.eval()


class Prob(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        return self.m(x).softmax(1)


def export_onnx(model, canvas, path=None):
    """ONNX FP32, opset 18, TorchScript exporter, output softmax 'prob' (like rotlab.export), batch axis dynamic.
    Returns the serialized model bytes (and writes `path` if given)."""
    m = static_copy(model, canvas)
    if SPECS[m.arch].export_unfused:
        set_fused_attn(m, False)
    buf = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(Prob(m), torch.randn(1, 3, canvas, canvas), buf, input_names=['image'], output_names=['prob'],
                          opset_version=18, dynamo=False, dynamic_axes={'image': {0: 'batch'}, 'prob': {0: 'batch'}})
    data = buf.getvalue()
    if path:
        Path(path).write_bytes(data)
    return data


def ort_session(data, threads=4):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    so.add_session_config_entry('session.intra_op.allow_spinning', '0')
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(data, so, providers=['CPUExecutionProvider'])


def gmacs(model, canvas):
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model.eval()(torch.randn(1, 3, canvas, canvas))
    return fc.get_total_flops() / 2e9


# ---------------------------------------------------------------- CLI
def cmd_describe(argv):
    ap = argparse.ArgumentParser(prog='rotlab.model_x describe')
    ap.add_argument('--archs', default='s,b,b_reg,pe_s,tips_b'); ap.add_argument('--canvases', default='112,140,168,196,224')
    a = ap.parse_args(argv)
    for arch in a.archs.split(','):
        m = RotNetX(img_size=224, pretrained=False, dynamic=True, arch=arch, drop_path=0.0).eval()
        d = m.describe()
        d['params_M_backbone'] = round(sum(p.numel() for p in m.backbone.parameters()) / 1e6, 2)
        d['params_M_head'] = round(sum(p.numel() for n, p in m.named_parameters() if not n.startswith('backbone.')) / 1e6, 3)
        d['per_canvas'] = {}
        for cv in map(int, a.canvases.split(',')):
            with torch.no_grad():
                t = m.tokens(torch.randn(1, 3, cv, cv))
            d['per_canvas'][cv] = dict(backbone_px=align(cv, m.patch), patch_tokens=t.shape[1] - m.npre, gmacs=round(gmacs(m, cv), 3))
        print(json.dumps(d), flush=True)


ANCHOR_MS = 52.0   # DINOv2 ViT-S/14 @224, FP32 ORT 4 threads batch 1, i7-1260P: measured 50-54 ms (BACKBONES.md §1)
DEFAULT_TARGETS = 's@224,pe_s@224,pe_s@256,pe_s@176,pe_s@144,pe_s@112,s@168,tips_b@140,tips_b@168,tips_b@224,b@140,b@168,b@224,b_reg@140'


def cmd_bench(argv):
    """Paired CPU micro-benchmark (random weights: latency only). An S@224 anchor session stays resident; each target
    graph is exported, timed round-robin against the anchor in the same rounds (rotating start), then freed.
    Ratio = mean over rounds of (target round mean / anchor round mean)."""
    import gc
    ap = argparse.ArgumentParser(prog='rotlab.model_x bench')
    ap.add_argument('--threads', type=int, default=4); ap.add_argument('--rounds', type=int, default=12)
    ap.add_argument('--reps', type=int, default=3); ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--targets', default=DEFAULT_TARGETS); ap.add_argument('--json')
    a = ap.parse_args(argv)
    torch.manual_seed(0)

    def build(arch, cv):
        t0 = time.time()
        data = export_onnx(RotNetX(img_size=224, pretrained=False, dynamic=True, arch=arch, drop_path=0.0).eval(), cv)
        import onnx
        nodes = len(onnx.load_from_string(data).graph.node)
        s = ort_session(data, a.threads)
        x = np.random.default_rng(0).standard_normal((1, 3, cv, cv)).astype(np.float32)
        print(f'built {arch}@{cv} ({len(data) / 1e6:.0f} MB, {nodes} nodes, {time.time() - t0:.0f}s)', flush=True)
        return s, {'image': x}, nodes

    anchor = build('s', 224)
    rows, anchor_all = {}, []
    for spec in filter(None, a.targets.split(',')):
        arch, cv = spec.split('@'); cv = int(cv)
        tgt = build(arch, cv)
        graphs = {'anchor': anchor, spec: tgt}; names = list(graphs)
        for _ in range(a.warmup):
            for n in names:
                graphs[n][0].run(None, graphs[n][1])
        times, rmean = {n: [] for n in names}, {n: [] for n in names}
        for r in range(a.rounds):
            for n in names[r % 2:] + names[:r % 2]:
                s, feeds, _ = graphs[n]; ts = []
                for _ in range(a.reps):
                    t0 = time.perf_counter(); s.run(None, feeds); ts.append((time.perf_counter() - t0) * 1e3)
                times[n] += ts; rmean[n].append(float(np.mean(ts)))
        t = np.array(times[spec]); anchor_all += times['anchor']
        ratio = float(np.mean(np.array(rmean[spec]) / np.array(rmean['anchor'])))
        rows[spec] = dict(arch=arch, canvas=cv, backbone_px=align(cv, SPECS[arch].patch), nodes=tgt[2],
                          mean=round(float(t.mean()), 1), p50=round(float(np.median(t)), 1), p90=round(float(np.percentile(t, 90)), 1),
                          anchor_mean_in_phase=round(float(np.mean(times['anchor'])), 1), ratio_S224=round(ratio, 4),
                          canon_ms=round(ratio * ANCHOR_MS, 1))
        print(spec, json.dumps(rows[spec]), f'loadavg {os.getloadavg()[0]:.1f}', flush=True)
        del tgt, graphs; gc.collect()
    at = np.array(anchor_all)
    summary = dict(threads=a.threads, rounds=a.rounds, reps=a.reps, anchor='s@224', anchor_mean=round(float(at.mean()), 1),
                   anchor_canonical_ms=ANCHOR_MS, loadavg=[round(v, 2) for v in os.getloadavg()],
                   versions=dict(torch=torch.__version__, timm=timm.__version__, ort=__import__('onnxruntime').__version__), rows=rows)
    print('| graph | backbone px | ONNX nodes | mean ms | p90 ms | / S@224 (paired) | canonical ms (S@224 = 52) |\n|---|---|---|---|---|---|---|')
    for n, v in rows.items():
        print(f"| {n} | {v['backbone_px']} | {v['nodes']} | {v['mean']} | {v['p90']} | {v['ratio_S224']} | {v['canon_ms']} |")
    print(json.dumps(summary))
    if a.json:
        Path(a.json).write_text(json.dumps(summary, indent=1))


def cmd_memest(argv):
    """Training memory estimate: bytes autograd saves in one bf16-autocast forward (CPU, random weights, no grad
    checkpointing) at two probe batches -> per-sample slope + batch-independent part (bf16 weight casts), scaled to
    --batch; plus fp32 weights, grads, AdamW m/v and the EMA copy. CUDA flash attention saves similar per-token state;
    allocator slack and cuDNN/cuBLAS workspaces are not included (hence the 80% budget)."""
    ap = argparse.ArgumentParser(prog='rotlab.model_x memest')
    ap.add_argument('--archs', default='s,pe_s,b,b_reg,tips_b,l'); ap.add_argument('--canvases', default='224,140')
    ap.add_argument('--batch', type=int, default=128); ap.add_argument('--gpu-gb', type=float, default=32.0)
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    for arch in a.archs.split(','):
        m = RotNetX(img_size=224, pretrained=False, dynamic=True, arch=arch, drop_path=0.1).train()
        n_par = sum(p.numel() for p in m.parameters())
        fixed = n_par * 4 * 5 / 2 ** 30          # fp32 weights, grads, Adam m and v, EMA copy

        def saved(b, cv):
            seen = {}

            def pack(t):
                seen[(t.data_ptr(), t.dtype, tuple(t.shape))] = t.numel() * t.element_size()
                return t
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t), torch.autocast('cpu', dtype=torch.bfloat16):
                m(torch.randn(b, 3, cv, cv))
            return sum(seen.values()) / 2 ** 30
        for cv in map(int, a.canvases.split(',')):
            s1, s2 = saved(1, cv), saved(3, cv)
            per = (s2 - s1) / 2; const = max(0.0, s1 - per)
            act = per * a.batch + const
            fit = int((a.gpu_gb * 0.8 - fixed - const) / per) if per > 0 else 0
            print(json.dumps(dict(arch=arch, canvas=cv, params_M=round(n_par / 1e6, 1), fixed_GB=round(fixed + const, 2),
                                  act_GB_per_sample=round(per, 4), batch=a.batch, act_GB_at_batch=round(act, 1),
                                  total_GB_at_batch=round(act + fixed, 1), max_batch_at_80pct=fit)), flush=True)


def cmd_export(argv):
    """Checkpoint (EMA by default) -> <out>-fp32.onnx (+ -int8.onnx with rotlab.export's quantisation settings).
    Use this instead of rotlab.export for b_reg/tips_b/pe_s: rotlab.export bakes the pos-embed with patch 14 and one
    prefix token, which is wrong for no_embed_class (reg) models and for PE's /16 grid."""
    ap = argparse.ArgumentParser(prog='rotlab.model_x export')
    ap.add_argument('ckpt', help='checkpoint path, or an arch key for random weights'); ap.add_argument('out')
    ap.add_argument('--canvas', type=int, default=224); ap.add_argument('--model', action='store_true', help='raw weights, not EMA')
    ap.add_argument('--int8', action='store_true')
    a = ap.parse_args(argv)
    if a.ckpt in SPECS:
        m = RotNetX(img_size=224, pretrained=False, dynamic=True, arch=a.ckpt, drop_path=0.0)
    else:
        ck = torch.load(a.ckpt, map_location='cpu', weights_only=False); cfg = ck['config']
        m = RotNetX(img_size=cfg['img_size'], hidden=cfg.get('hidden', 1024), pretrained=False, dynamic=True, arch=cfg.get('arch', 's'), drop_path=0.0)
        m.load_state_dict(ck['model' if a.model else 'ema'])
        if cfg.get('pool'):
            m.pool = tuple(map(int, cfg['pool'].split(':')))
    m.eval()
    data = export_onnx(m, a.canvas, a.out + '-fp32.onnx')
    x = torch.randn(2, 3, a.canvas, a.canvas)
    with torch.no_grad():
        ref = m(x).softmax(1).numpy()
    got = ort_session(data, 4).run(None, {'image': x.numpy()})[0]
    print(json.dumps(dict(out=a.out + '-fp32.onnx', arch=m.arch, canvas=a.canvas, max_abs_prob_diff=float(np.abs(got - ref).max()))))
    if a.int8:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(a.out + '-fp32.onnx', a.out + '-int8.onnx', weight_type=QuantType.QInt8, per_channel=True, reduce_range=True,
                         op_types_to_quantize=['MatMul', 'Gemm'])
        print('int8', a.out + '-int8.onnx')


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else argv
    cmds = dict(describe=cmd_describe, bench=cmd_bench, memest=cmd_memest, export=cmd_export)
    if not argv or argv[0] not in cmds:
        raise SystemExit(__doc__)
    cmds[argv[0]](argv[1:])


if __name__ == '__main__':
    main()
