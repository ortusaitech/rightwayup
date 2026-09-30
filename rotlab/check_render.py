"""Verify rotlab.core.render reproduces archived v3 derivatives (angle chart + affine)."""
import json
import numpy as np
from PIL import Image
from rotlab.core import render, DATA
A = DATA / 'archives/ortus-rotation-hybrid-screen-v3-2026-08-27'
n = 0
for line in (A / 'metadata/derivatives-train-materialized.jsonl').open():
    d = json.loads(line)
    if d['degradation_recipe'].get('steps') or d.get('width') != 320:
        continue
    g = d['geometry']; src = Image.open(d['source_path']).convert('RGB')
    if src.size != (g['source_width'], g['source_height']):
        continue
    ref = np.asarray(Image.open(A / d['archive_path']).convert('RGB'), np.float32)
    cx, cy = d['centre_xy']
    out = {}
    for sgn in (1, -1):
        im = render(src, sgn * d['applied_clockwise_degrees'], g['crop_width'], g['crop_height'], 320, 192, cx, cy)
        out[sgn] = float(np.abs(np.asarray(im, np.float32) - ref).mean())
    print(d['source_family'], round(d['applied_clockwise_degrees'], 1), 'MAD +:', round(out[1], 2), ' -:', round(out[-1], 2))
    n += 1
    if n >= 8:
        break
