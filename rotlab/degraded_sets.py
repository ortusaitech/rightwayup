#!/usr/bin/env python3
"""Degraded photo tests: the fair-photo views (1,030) and the Open Images photo-test views (3,345), every view passed
through the CCTV degradation model (core.degrade, force=True: resolution loss, blur, IR/greyscale, exposure, noise,
JPEG), seeded. Same angles and crops as the clean sets, so clean-vs-degraded is a paired comparison.
Also scores the incumbent on identical pixels -> DATA/rotlab/suite/incumbent-deg.npz.
  python -m rotlab.degraded_sets CANVAS [CANVAS ...]
"""
import random, sys
import numpy as np

from rotlab.core import DATA, degrade, letterbox

OUT = DATA / 'rotlab/suite'


def views():
    from rotlab.fair_photo import fair_views
    from rotlab.oi_test import views as oi_views
    _, fv, ft = fair_views(7); ov, ot = oi_views('fixed')
    out = {}
    for name, vs, th in (('fair_deg', fv, ft), ('oi_deg', ov, ot)):
        rng = random.Random(f'deg:{name}')
        out[name] = ([degrade(v, rng, force=True) for v in vs], np.asarray(th))
    return out


def build(canvas, score_incumbent=True):
    vs = views()
    np.savez(OUT / f'views-deg-c{canvas}.npz', **{f'{k}_x': np.stack([letterbox(v, canvas) for v in ims]) for k, (ims, _) in vs.items()},
             **{f'{k}_t': th for k, (_, th) in vs.items()})
    f = OUT / 'incumbent-deg.npz'
    if score_incumbent and not f.exists():
        from rotlab.fair_photo import incumbent
        inc = incumbent(16)
        np.savez(f, **{f'{k}_pred': np.array([inc(v) for v in ims]) for k, (ims, _) in vs.items()}, **{f'{k}_theta': th for k, (_, th) in vs.items()})
    print('built degraded views', canvas, {k: len(v[0]) for k, v in vs.items()}, flush=True)


if __name__ == '__main__':
    for c in sys.argv[1:]:
        build(int(c))
