"""Uniform (or weighted) EMA weight soup of N checkpoints sharing one ancestry: python soupn.py OUT_DIR ckpt[:w] ckpt[:w] ..."""
import sys, torch
from pathlib import Path
out = Path(sys.argv[1]); parts = [(p.rsplit(':', 1)[0], float(p.rsplit(':', 1)[1])) if ':' in p else (p, 1.0) for p in sys.argv[2:]]
tot = sum(w for _, w in parts); acc, first = None, None
for p, w in parts:
    ck = torch.load(p, map_location='cpu', weights_only=False); e = ck['ema']
    if acc is None:
        first, acc = ck, {k: (v.float() * (w / tot) if v.is_floating_point() else v) for k, v in e.items()}
    else:
        assert e.keys() == acc.keys(), p
        for k, v in e.items():
            if v.is_floating_point():
                acc[k] += v.float() * (w / tot)
m = {k: v.to(first['ema'][k].dtype) for k, v in acc.items()}
out.mkdir(parents=True, exist_ok=True)
torch.save(dict(ema=m, model=m, config=first['config'], soup=dict(parts=[[p, w] for p, w in parts])), out / 'final.pt'); print('soup', out, parts)
