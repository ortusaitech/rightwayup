"""Single-pass directed 360-bin orientation model on the owner-accepted DINOv2 ViT-S/14.

Accepted base (7 Sep owner decision): timm/vit_small_patch14_dinov2.lvd142m, revision
4610ca14..., weight SHA 04d27f34... (no registers).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import timm

DINO = '/workspace/rotation-data/quarantine/external-weights/e3ai/dinov2_small-lvd142m-4610ca14/model.safetensors'
ARCH = {'s': ('vit_small_patch14_dinov2.lvd142m', DINO),
        # research-only until owner extends the DINOv2 acceptance to ViT-B (same Apache-2.0 publisher grant)
        'b': ('vit_base_patch14_dinov2.lvd142m', '/workspace/rotation-data/quarantine/external-weights/dinov2_base-lvd142m/model.safetensors'),
        'l': ('vit_large_patch14_dinov2.lvd142m', '/workspace/rotation-data/quarantine/external-weights/dinov2_large-lvd142m/model.safetensors')}


class RotNet(nn.Module):
    def __init__(self, img_size=224, drop_path=0.1, hidden=1024, bins=360, pretrained=True, dynamic=False, arch='s'):
        super().__init__()
        kw = dict(pretrained=pretrained, img_size=img_size, num_classes=0, drop_path_rate=drop_path, dynamic_img_size=dynamic)
        name, weights = ARCH[arch]
        if pretrained:
            kw['pretrained_cfg_overlay'] = dict(file=weights)
        self.backbone = timm.create_model(name, **kw)
        d = self.backbone.embed_dim
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, hidden), nn.GELU(), nn.Linear(hidden, bins))

        self.pool = None   # campaign 2026-09-27: optional (k, p) static token pooling after block k to a p x p grid

    def forward(self, x):
        if self.pool is None:
            t = self.backbone.forward_features(x)          # B, 1+N, D (final norm applied)
        else:
            t = self._pooled_features(x)
        f = torch.cat([t[:, 0], t[:, 1:].mean(1)], 1)
        return self.head(f)

    def _pooled_features(self, x):
        from rotlab.camp_eval import area_matrix
        bb, (k, p) = self.backbone, self.pool
        t = bb.patch_embed(x); t = bb._pos_embed(t); t = bb.patch_drop(t); t = bb.norm_pre(t)
        for i, blk in enumerate(bb.blocks):
            if i == k:
                g = round((t.shape[1] - 1) ** 0.5)
                W = area_matrix(g, p).to(device=t.device, dtype=t.dtype)
                t = torch.cat([t[:, :1], torch.matmul(W, t[:, 1:])], 1)
            t = blk(t)
        return bb.norm(t)


def param_groups(model: RotNet, lr, head_lr, wd, lld=0.75):
    """Layer-wise lr decay over the ViT blocks; no wd on norms/biases/pos/cls."""
    bb = model.backbone
    n = len(bb.blocks)
    groups = {}

    def add(p, name, scale):
        nd = p.ndim == 1 or name.endswith(('pos_embed', 'cls_token', 'reg_token')) or 'norm' in name
        key = (scale, nd)
        g = groups.setdefault(key, dict(params=[], lr_scale=scale, weight_decay=0.0 if nd else wd))
        g['params'].append(p)

    for name, p in bb.named_parameters():
        if name.startswith('blocks.'):
            layer = int(name.split('.')[1]) + 1
        elif name.startswith(('patch_embed', 'cls_token', 'pos_embed', 'reg_token', 'mask_token')):
            layer = 0
        else:
            layer = n
        add(p, name, lr * lld ** (n - layer))
    for name, p in model.head.named_parameters():
        add(p, 'head.' + name, head_lr)
    out = []
    for g in groups.values():
        g['lr'] = g['lr_scale']
        out.append(g)
    return out
