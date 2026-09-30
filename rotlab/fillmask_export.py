"""Batch-capable fill-drop exports (lead, 29 Sep; owner decision: ship them and report their throughput). The FILLMASK
formulation, identical to the shipped Core ML packages (devices lane `v2cm.make_fillmask_module`, mask 'add', pool
'sum'): every patch token is computed, letterbox-fill KEYS get an additive -1e4 in every attention block and fill
tokens are left out of the pool, so the answer equals the batch-1 fill-drop graph up to float rounding while the input
has a dynamic batch dimension. The fill mask is computed from the float32 input (tol 1e-3) in every precision.
  python -m rotlab.fillmask_export CKPT --canvas 112 --out PREFIX      # PREFIX-fp32.onnx, PREFIX-fp32-int8.onnx, PREFIX-fp16.onnx
  python -m rotlab.fillmask_export --parity PREFIX-fp32.onnx FILL-fp32.onnx CANVAS SETS   # vs the batch-1 fill-drop file
"""
import argparse, copy, io, json, sys

import numpy as np

NEG = -1e4


def build(ckpt, canvas):
    import torch
    from rotlab.filldrop import FillDropNet, build_net
    from rotlab.focus import P, TOL, patch_pe
    ck = torch.load(ckpt, map_location='cpu', weights_only=False); cfg = ck['config']
    net = build_net(cfg.get('arch', 's'), cfg['img_size'], hidden=cfg.get('hidden', 1024)); net.load_state_dict(ck['ema'])
    fd = cfg.get('filldrop') or {}
    m = FillDropNet(net, sinks=fd.get('sinks', 0), tol=fd.get('tol', TOL)).eval()
    g = canvas // P
    with torch.no_grad():
        pe = patch_pe(m.net.backbone, g, g).detach().clone()
    return m, pe


def module(m, pe, half=False):
    import torch

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = copy.deepcopy(m).half() if half else m
            self.register_buffer('pe', pe.reshape(1, -1, pe.shape[-1]).clone().to(torch.float16 if half else torch.float32))

        def forward(self, x):
            mm = self.m
            keep = mm.keep_mask(x)                                   # float32 input: exact fill mask (tol 1e-3)
            kf = keep.to(torch.float32)
            am = ((1.0 - torch.cat([torch.ones_like(kf[:, :1]), kf], 1)) * NEG)[:, None, None, :]
            xi = x.half() if half else x
            pm = mm.net.backbone.patch_embed
            tok = pm.norm(pm.proj(xi).flatten(2).transpose(1, 2)) + self.pe + mm.scale_emb[0]
            if half:
                kf, am = kf.half(), am.half()
            bb = mm.net.backbone   # = FocusNet.encode, but the CLS expand uses tok.size(0) (len() would trace a constant batch)
            cls = (bb.cls_token + bb.pos_embed[:, :1]).expand(tok.size(0), -1, -1)
            t = bb.norm_pre(bb.patch_drop(bb.pos_drop(torch.cat([cls, tok], 1))))
            for blk in bb.blocks:
                t = blk(t, attn_mask=am)
            t = bb.norm(t); p = t[:, 1:]
            f = (p * kf[..., None]).sum(1) / kf.sum(1, keepdim=True)
            return mm.net.head(torch.cat([t[:, 0], f], 1)).float().softmax(1)
    return M().eval()


def export(mod, canvas, path, device):
    import onnx
    import torch
    from rotlab.focus import fix_topk_k
    x = torch.randn(2, 3, canvas, canvas, device=device)
    buf = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(mod.to(device), (x,), buf, input_names=['image'], output_names=['prob'], opset_version=18, dynamo=False,
                          dynamic_axes={'image': {0: 'batch'}, 'prob': {0: 'batch'}})
    open(path, 'wb').write(fix_topk_k(onnx.load_from_string(buf.getvalue())).SerializeToString())


def parity(fm, fill, canvas, sets):
    """ORT CPU: fillmask file at batch 16 vs the shipped batch-1 fill-drop file on real views."""
    import pickle
    import onnxruntime as ort
    from PIL import Image
    from rotlab.camp_eval import view_file
    from rotlab.core import decode, letterbox
    so = ort.SessionOptions(); so.intra_op_num_threads = 16
    a = ort.InferenceSession(fm, so, providers=['CPUExecutionProvider']); b = ort.InferenceSession(fill, so, providers=['CPUExecutionProvider'])
    out = {}
    for n in sets.split(','):
        d = pickle.loads(view_file(n).read_bytes()); th = np.asarray(d['theta'], float)
        x = np.stack([letterbox(Image.open(io.BytesIO(p)).convert('RGB'), canvas) for p in d['png']])
        pa = np.concatenate([a.run(None, {'image': x[i:i + 16]})[0] for i in range(0, len(x), 16)])
        pb = np.concatenate([b.run(None, {'image': x[i:i + 1]})[0] for i in range(len(x))])
        da, db = decode(pa), decode(pb); dd = np.abs((da - db + 180) % 360 - 180)
        e = lambda p: int((np.minimum(np.abs((p - th) % 360), 360 - np.abs((p - th) % 360)) <= 10).sum())
        out[n] = dict(n=len(th), w10_fillmask=e(da), w10_fill=e(db), max_abs_dprob=float(np.abs(pa - pb).max()), max_dangle=float(dd.max()))
    print(json.dumps(dict(fillmask=fm, fill=fill, parity=out)))


def main():
    if sys.argv[1:2] == ['--parity']:
        return parity(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
    import torch
    ap = argparse.ArgumentParser(prog='rotlab.fillmask_export'); ap.add_argument('ckpt'); ap.add_argument('--canvas', type=int, required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    m, pe = build(a.ckpt, a.canvas)
    export(module(m, pe), a.canvas, a.out + '-fp32.onnx', 'cpu')
    from onnxruntime.quantization import QuantType, quantize_dynamic   # same settings as rotlab.filldrop export --int8
    quantize_dynamic(a.out + '-fp32.onnx', a.out + '-fp32-int8.onnx', weight_type=QuantType.QInt8, per_channel=True, reduce_range=True,
                     op_types_to_quantize=['MatMul', 'Gemm'])
    if torch.cuda.is_available():
        export(module(m, pe, half=True), a.canvas, a.out + '-fp16.onnx', 'cuda')
    print(json.dumps(dict(out=a.out, canvas=a.canvas, fp16=torch.cuda.is_available())))


if __name__ == '__main__':
    main()
