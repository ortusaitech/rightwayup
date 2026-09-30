"""Compare no-TTA / 180 / C4 (both roll signs) for a checkpoint on the pack."""
import sys, torch, numpy as np
import rotlab.evaluate as ev
from rotlab.model import RotNet
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
m = RotNet(img_size=224, pretrained=False, arch=ck['config'].get('arch', 's')).cuda().eval(); m.load_state_dict(ck['ema'])
pc = ev.PackCache(224)
for name, t, sign in (('none', False, 1), ('180', True, 1), ('c4+', 'c4', 1), ('c4-', 'c4', -1)):
    ev.C4_SIGN = sign
    res, _ = ev.report(pc, ev.predict(m, pc.x, tta180=t), name)
    print(name, {p: (v['w10'], v['t150']) for p, v in res['panels'].items()})
