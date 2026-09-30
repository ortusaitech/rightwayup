"""FP16 fill-drop ONNX (lead, 29 Sep; FREEZE v2 exports). Same IO contract as rotlab.export's FP16 copy: float32
'image' (1, 3, canvas, canvas) letterboxed as today -> float32 'prob' (1, 360). The letterbox-fill mask is computed on
the float32 input (the fill test uses a 1e-3 tolerance that FP16 cannot resolve), then patch embedding, blocks and head
run in FP16 on the kept tokens (NonZero gather, batch 1), softmax in float32. The grid position table is resampled in
float32 before the cast. Exported on CUDA; parity vs the float32 PyTorch fill-drop model on 5 aspect ratios, run with
ONNX Runtime CUDA.
  python -m rotlab.fp16_fill CKPT --canvas 112 --out GF-SOUP-SV2-nano-s112-fill-fp16.onnx
"""
import argparse, io, json

import numpy as np

try:   # the parity step runs in an onnxruntime-gpu environment without torch
    import torch
    import torch.nn as nn
    from rotlab.filldrop import FillDropNet, build_net, parse_aspect
    from rotlab.focus import P, TOL, fix_topk_k, patch_pe, sample_input
except ImportError:
    torch = None


class HalfFill(torch.nn.Module if torch else object):
    def __init__(self, fm):
        super().__init__(); self.m = fm

    def forward(self, x):
        keep = self.m.keep_mask(x)                                  # float32 input: exact fill mask
        tok = self.m.embed(x.half(), 0)
        return self.m.encode(tok[:, keep[0].nonzero()[:, 0]]).float().softmax(1)


def main():
    import sys
    if sys.argv[1:2] == ['--parity']:
        return parity(sys.argv[2])
    import onnx
    ap = argparse.ArgumentParser(prog='rotlab.fp16_fill'); ap.add_argument('ckpt'); ap.add_argument('--canvas', type=int, required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False); cfg = ck['config']
    if cfg.get('focus') or cfg.get('pool'):
        raise SystemExit('plain RotNet / fill-drop checkpoints only')
    net = build_net(cfg.get('arch', 's'), cfg['img_size'], hidden=cfg.get('hidden', 1024)); net.load_state_dict(ck['ema'])
    fd = cfg.get('filldrop') or {}
    ref = FillDropNet(net, sinks=fd.get('sinks', 0), tol=fd.get('tol', TOL)).eval().cuda()
    g = a.canvas // P
    with torch.no_grad():
        pe32 = patch_pe(ref.net.backbone, g, g).detach().clone()   # float32 resample, before any cast
    import copy
    m16 = copy.deepcopy(ref).half().eval()
    m16.export = True; m16._pe[(g, g)] = pe32.half()
    h = HalfFill(m16).eval()
    x0 = sample_input(a.canvas, 4 / 3).cuda()
    buf = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(h, (x0,), buf, input_names=['image'], output_names=['prob'], opset_version=18, dynamo=False)
    data = fix_topk_k(onnx.load_from_string(buf.getvalue())).SerializeToString()
    open(a.out, 'wb').write(data)
    xs, refs, kept = {}, {}, {}
    for lab in ('4:3', '16:9', '1:1', '2:3', '21:9'):
        x = sample_input(a.canvas, parse_aspect(lab), seed=11)
        with torch.no_grad():
            refs[lab] = ref(x.cuda()).softmax(1).float().cpu().numpy()
        xs[lab] = x.numpy(); kept[lab] = int(ref.keep_mask(x.cuda()).sum())
    np.savez(a.out + '.parity-ref.npz', **{f'x_{k}': v for k, v in xs.items()}, **{f'ref_{k}': v for k, v in refs.items()})
    print(json.dumps(dict(out=a.out, canvas=a.canvas, mb=round(len(data) / 1e6, 1), kept=kept,
                          parity='run: <python with onnxruntime-gpu> -m rotlab.fp16_fill --parity ' + a.out)))


def parity(path):
    """ONNX Runtime CUDA run of the FP16 file on the saved inputs vs the float32 PyTorch reference."""
    import onnxruntime as ort
    z = np.load(path + '.parity-ref.npz')
    s = ort.InferenceSession(path, providers=['CUDAExecutionProvider'])
    assert s.get_providers()[0] == 'CUDAExecutionProvider', 'CUDA provider not active'
    rows = {}
    for k in [f[2:] for f in z.files if f.startswith('x_')]:
        o = s.run(None, {'image': z[f'x_{k}']})[0]; r = z[f'ref_{k}']
        rows[k] = dict(max_abs_prob_diff=float(np.abs(o - r).max()), argmax_equal=bool(o.argmax() == r.argmax()))
    print(json.dumps(dict(file=path, parity=rows)))

if __name__ == '__main__':
    main()
