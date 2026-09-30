"""Zero-shot input-resolution probe: evaluate a square-canvas checkpoint at smaller canvases
(positional embeddings interpolated). Lower bound for a low-res fine-tune.
  python -m rotlab.res_probe CKPT 112 140 168 224"""
import sys, torch
import rotlab.evaluate as ev
from rotlab.model import RotNet
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
m = RotNet(img_size=224, pretrained=False, dynamic=True, arch=ck['config'].get('arch', 's')).cuda().eval()
m.load_state_dict(ck['ema'])
for size in map(int, sys.argv[2:]):
    pc = ev.PackCache(size)
    res, _ = ev.report(pc, ev.predict(m, pc.x), f'{size}')
    print(size, {p: (v['w10'], v['t150']) for p, v in res['panels'].items()}, flush=True)
