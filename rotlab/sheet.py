#!/usr/bin/env python3
"""Contact sheet of the worst benchmark rows for an evaluated checkpoint.
  python -m rotlab.sheet RUN_DIR/final.eval-ema.json --panel common --family diode --n 48 --out sheet.jpg
Each tile: the benchmark image, and the same image rotated by the model's correction
(so a correct model shows it upright). Caption: target / predicted / error.
"""
import argparse, json
import numpy as np
from PIL import Image, ImageDraw
from rotlab.core import circ_err, load_pack


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('eval_json'); ap.add_argument('--panel'); ap.add_argument('--family')
    ap.add_argument('--n', type=int, default=48); ap.add_argument('--min-err', type=float, default=90); ap.add_argument('--out', default='sheet.jpg')
    a = ap.parse_args()
    rows = load_pack(); pred = np.array(json.load(open(a.eval_json))['predictions'])
    tgt = np.array([r['target_degrees'] for r in rows]); err = circ_err(pred, tgt)
    idx = [i for i in np.argsort(-err) if err[i] >= a.min_err and (not a.panel or rows[i]['panel'] == a.panel)
           and (not a.family or rows[i]['source_family'] == a.family)][:a.n]
    T = 180; cols = 6
    sheet = Image.new('RGB', (cols * 2 * T, ((len(idx) + cols - 1) // cols) * (T + 16)), (40, 40, 40))
    d = ImageDraw.Draw(sheet)
    for k, i in enumerate(idx):
        im = Image.open(rows[i]['image']['path']).convert('RGB')
        up = im.rotate(pred[i], expand=True, fillcolor=(60, 60, 60))  # PIL rotates CCW; CCW by pred restores upright
        for j, x in enumerate((im, up)):
            x = x.copy(); x.thumbnail((T, T))
            sheet.paste(x, ((k % cols) * 2 * T + j * T, (k // cols) * (T + 16)))
        d.text(((k % cols) * 2 * T + 2, (k // cols) * (T + 16) + T), f"{(rows[i]['source_group'] or rows[i]['id'])[-14:]} t{tgt[i]:.0f} p{pred[i]:.0f} e{err[i]:.0f}", fill=(255, 255, 0))
    sheet.save(a.out, quality=85); print(a.out, len(idx))


if __name__ == '__main__':
    main()
