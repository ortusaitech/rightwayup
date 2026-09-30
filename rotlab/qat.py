"""INT8-aware fine-tuning (lead, 28 Sep 20:00). The release's CPU INT8 graph is onnxruntime dynamic quantization: per-channel
int8 weights (reduce_range: 7-bit) and per-tensor uint8 ACTIVATIONS on the inputs of the 98 MatMuls it quantizes (qkv, proj,
fc1, fc2 of every block + the two head Gemms; the attention q@k / attn@v stay float). That activation quantization amplified
the grid-fix Max's residual probe lean (soft rho4 0.012 -> 0.110); weight-only 8-bit did not. apply_qat() makes exactly those
Linear layers see the same quantization in training (straight-through estimator), so the probe loss and KD are optimised for
the quantized graph. Checkpoints stay plain float weights; export and quantization are unchanged."""
import types
import torch
import torch.nn as nn
import torch.nn.functional as F


def fq_act(x):
    """Per-sample (ORT runs batch 1) per-tensor asymmetric uint8, range forced to include 0 (DynamicQuantizeLinear)."""
    xf = x.float(); b = xf.shape[0]; flat = xf.reshape(b, -1)
    mn = flat.amin(1).clamp(max=0); mx = flat.amax(1).clamp(min=0)
    s = ((mx - mn) / 255.0).clamp(min=1e-8); z = torch.round(-mn / s).clamp(0, 255)
    sh = (b,) + (1,) * (xf.dim() - 1); s, z = s.view(sh), z.view(sh)
    xq = (torch.clamp(torch.round(xf / s) + z, 0, 255) - z) * s
    return (xf + (xq - xf).detach()).to(x.dtype)


def fq_w(w, reduce_range=True):
    """Per-output-channel symmetric int8 as onnxruntime quantize_dynamic(per_channel, reduce_range): scale = absmax / 64,
    q in [-64, 64] (measured on onnxruntime 1.24.4: absmax/scale == 64 for every channel)."""
    wf = w.float(); qmax = 64.0 if reduce_range else 127.0
    s = (wf.abs().amax(1, keepdim=True) / qmax).clamp(min=1e-12)
    wq = torch.clamp(torch.round(wf / s), -qmax, qmax) * s
    return (wf + (wq - wf).detach()).to(w.dtype)


def _qforward(self, x):
    return F.linear(fq_act(x), fq_w(self.weight), self.bias)


def apply_qat(model):
    """Patch the Linear layers that onnxruntime dynamic INT8 quantizes. Returns the number patched (98 for ViT-L + head)."""
    n = 0
    for name, m in model.named_modules():
        leaf = name.split('.')[-1]
        if isinstance(m, nn.Linear) and (('blocks.' in name and leaf in ('qkv', 'proj', 'fc1', 'fc2')) or name.startswith('head')):
            m.forward = types.MethodType(_qforward, m); n += 1
    return n
