"""Calibration predictions with the FILL-DROP forward (deployment graph for fill-drop models), in rotlab.suite's file format.

rotlab.suite --cal scores a plain full-token RotNet. A fill-drop model ships as the fill-drop graph (letterbox fill tokens
dropped in-graph), so its routing/abstention thresholds must be fitted on fill-drop predictions of the same calibration
views. Writes DATA/rotlab/suite/<run>-fill-cal-c<canvas>.npz with keys <set>_{p,tta}_{pred,conf} and <set>_theta, exactly
like suite.run(cal=True), so final_stats / protected_eval.thresholds() can read it as run '<run>-fill'.
  python -m rotlab.cal_fill CKPT --canvases 112,224 [--sinks N] [--tol T]
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from rotlab.core import DATA, circ_err, decode
from rotlab.evaluate import confidence
from rotlab.focus import TOL, infer, load_ckpt

OUT = DATA / 'rotlab/suite'


def main():
    ap = argparse.ArgumentParser(prog='rotlab.cal_fill')
    ap.add_argument('ckpt'); ap.add_argument('--canvases', default='224')
    ap.add_argument('--sinks', type=int, default=0); ap.add_argument('--tol', type=float, default=TOL)
    ap.add_argument('--bs', type=int, default=128)
    a = ap.parse_args()
    m, cfg = load_ckpt(a.ckpt)
    m.plan, m.sinks, m.tol = 'fill', a.sinks, a.tol
    m = m.cuda().eval(); name = Path(a.ckpt).parent.name
    for cv in map(int, a.canvases.split(',')):
        z = np.load(OUT / f'views-cal-c{cv}.npz')
        sets = {k[:-2]: (z[k], z[k[:-2] + '_t']) for k in z.files if k.endswith('_x')}
        rows = {}
        for s, (x, th) in sets.items():
            p0, p1 = [], []
            for i in range(0, len(x), a.bs):
                b = torch.from_numpy(np.ascontiguousarray(x[i:i + a.bs])).cuda()
                p0.append(infer(m, None, b, None, None, None, None)[0])
                q = infer(m, None, torch.rot90(b, 2, (2, 3)), None, None, None, None)[0]
                p1.append(np.roll(q, -180, 1))
            pr = np.concatenate(p0); pt = (pr + np.concatenate(p1)) / 2
            for key, prob in (('p', pr), ('tta', pt)):
                pd = decode(prob); rows[f'{s}_{key}_pred'] = pd; rows[f'{s}_{key}_conf'] = confidence(prob, pd)
            rows[f'{s}_theta'] = th
        np.savez(OUT / f'{name}-fill-cal-c{cv}.npz', **rows)
        print(name, 'fill cal', cv, {s: int((circ_err(rows[f'{s}_p_pred'], rows[f'{s}_theta']) <= 10).sum()) for s in sets},
              flush=True)


if __name__ == '__main__':
    main()
