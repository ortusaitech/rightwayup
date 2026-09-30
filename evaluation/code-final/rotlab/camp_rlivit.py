#!/usr/bin/env python3
"""Campaign 2026-09-27: R-LiViT RGB-T (Zenodo 16356714, CC BY 4.0; Augsburg traffic intersections) as a NEW-VIEWPOINT
evaluation panel (one physical rig; not distinct sensor hardware) (evaluation only; never trained on).

Split (whole INFERRED intersections: recording-date clusters + shared visual landmarks; site identity not proven;
see reports/rlivit-panel.md):
  final (SEALED): inferred intersection A = locations 6, 7 (recorded 4-5 May 2024)
  dev:            intersection B = location 4 (17 May); intersection C = locations 0, 1, 2, 3, 5 (23-31 May)
One frame per sequence (position 26, mid-sequence; frames within a sequence span ~10 s and are not independent).
Valid region: per location and modality, footprint = pixels non-black in ANY frame of that location (RGB: the top
undistortion arc is black everywhere; thermal: projected into the RGB frame and zero-padded); then the largest inscribed
axis-aligned rectangle (after a 6-px erosion). View = seeded angle, fixed 4:3 crop with diagonal = min(rect) - 4 centred
on the rectangle (inscribed circle inside the valid region at any angle: no padding can enter). RGB and thermal of one
pair share the angle (pairing preserved). LABELS ARE RELATIVE TO THE STORED RASTER (synthetic rotation, base roll assumed
0): the source roll is NOT validated. A least-squares vertical-VP roll estimate is recorded per location as an unvalidated
diagnostic only (it is unreliable: 3-7 deg and one -37 deg outlier); no location is excluded or relabelled from it.
One physical RGB/thermal rig at 7 viewpoints / 3 intersections: this measures unseen scene/viewpoint generalisation,
not distinct camera hardware. Report within 10 deg and a within 15 deg sensitivity; no fine-angle claims.
  python -m rotlab.camp_rlivit prepare          # manifest + masks + roll report (no views)
  python -m rotlab.camp_rlivit views dev        # DATA/rotlab/newdata/views/rlivit_dev_{rgb,thermal}.pkl
  python -m rotlab.camp_rlivit views final --i-am-final
"""
import hashlib, io, json, pickle, random, re, sys, zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from rotlab.core import DATA, render

ZIP = DATA / 'raw/rlivit/R-LiViT_RGB-T.zip'
ND = DATA / 'rotlab/newdata'
ROOT = 'R-LiViT_RGB-T/'
SPLIT = {'6': 'final', '7': 'final', '4': 'dev', '0': 'dev', '1': 'dev', '2': 'dev', '3': 'dev', '5': 'dev'}
INTERSECTION = {'6': 'A', '7': 'A', '4': 'B', '0': 'C', '1': 'C', '2': 'C', '3': 'C', '5': 'C'}
POS = '26'


def seqs(z):
    x = z.read(ROOT + 'sequences.xml').decode()
    return [dict(seq=s, daytime=d, location=l) for s, d, l in
            re.findall(r'<rgbt_seq_id>(\d+)</rgbt_seq_id>\s*<daytime>(\w+)</daytime>\s*<location>(\d+)</location>', x)]


def img(z, mod, s, pos=POS):
    return Image.open(io.BytesIO(z.read(f'{ROOT}{mod}/{s}/{pos}.png'))).convert('RGB')


def largest_rect(mask):
    """Largest axis-aligned all-True rectangle (histogram method). Returns x0, y0, x1, y1 (exclusive)."""
    h, w = mask.shape; heights = np.zeros(w, int); best = (0, 0, 0, 0, 0)
    for y in range(h):
        heights = np.where(mask[y], heights + 1, 0); st = []
        for x in range(w + 1):
            cur = heights[x] if x < w else 0
            while st and heights[st[-1]] >= cur:
                hh = heights[st.pop()]; x0 = st[-1] + 1 if st else 0; a = hh * (x - x0)
                if a > best[0]:
                    best = (a, x0, y - hh + 1, x, y + 1)
            st.append(x)
    return best[1:]


def footprint(z, mod, items):
    acc = None
    for r in items:
        for pos in ('02', '26', '46'):
            a = np.asarray(img(z, mod, r['seq'], pos)).max(2) > 12
            acc = a if acc is None else (acc | a)
    from scipy.ndimage import binary_erosion
    return binary_erosion(acc, iterations=6)


def roll_estimate(z, items):
    import cv2
    segs = []
    for r in items[:6]:
        g = np.asarray(img(z, 'rgb', r['seq']).convert('L'))
        lines = cv2.createLineSegmentDetector().detect(g)[0]
        if lines is None:
            continue
        for x1, y1, x2, y2 in lines[:, 0]:
            L = np.hypot(x2 - x1, y2 - y1)
            if L > 60 and abs(np.degrees(np.arctan2(x2 - x1, y2 - y1)) % 180 - 90) > 70:   # within 20 deg of vertical
                segs.append((x1, y1, x2, y2, L))
    if len(segs) < 10:
        return None, len(segs)
    # least-squares vanishing point of the near-vertical segments (lines a x + b y = c), weighted by length
    A, c, wts = [], [], []
    for x1, y1, x2, y2, L in segs:
        a, b = y2 - y1, x1 - x2; n = np.hypot(a, b); A.append([a / n, b / n]); c.append((a * x1 + b * y1) / n); wts.append(L)
    A, c, W = np.array(A), np.array(c), np.sqrt(np.array(wts))
    vp = np.linalg.lstsq(A * W[:, None], c * W, rcond=None)[0]
    cx, cy = 640, 360
    roll = np.degrees(np.arctan2(vp[0] - cx, vp[1] - cy))   # 0 when the VP lies straight below (or above) the centre
    if vp[1] < cy:
        roll = (roll + 180 + 180) % 360 - 180
    return float(roll), len(segs)


def cmd_prepare():
    z = zipfile.ZipFile(ZIP); S = seqs(z); out = {'sequences': [], 'locations': {}}
    for l in sorted({r['location'] for r in S}):
        items = [r for r in S if r['location'] == l]
        rects = {}
        for mod in ('rgb', 'thermal'):
            m = footprint(z, mod, items); x0, y0, x1, y1 = largest_rect(m); rects[mod] = [int(x0), int(y0), int(x1), int(y1)]
        roll, nseg = roll_estimate(z, [r for r in items if r['daytime'] == 'day'] or items)
        out['locations'][l] = dict(intersection=INTERSECTION[l], split=SPLIT[l], n_seq=len(items), rect=rects, roll_deg=roll, n_vertical_segments=nseg,
                                   roll_note='unvalidated diagnostic; not used for labels or exclusion')
        print(l, out['locations'][l], flush=True)
    for r in S:
        rec = dict(id=f'rlivit:{r["seq"]}:{POS}', seq=r['seq'], location=r['location'], intersection=INTERSECTION[r['location']],
                   split=SPLIT[r['location']], daytime=r['daytime'])
        for mod in ('rgb', 'thermal'):
            rec[f'sha256_{mod}'] = hashlib.sha256(z.read(f'{ROOT}{mod}/{r["seq"]}/{POS}.png')).hexdigest()
        out['sequences'].append(rec)
    out['source'] = dict(zenodo=16356714, doi='10.5281/zenodo.16356714', licence='CC-BY-4.0', file='R-LiViT_RGB-T.zip',
                         md5='88e3db20698705017e4f06e4bbb2dde6', bytes=3798751628, attribution='R-LiViT (arXiv:2503.17122)')
    f = ND / 'rlivit.json'; f.write_text(json.dumps(out, indent=1))
    print('manifest', hashlib.sha256(f.read_bytes()).hexdigest(), flush=True)


def cmd_views(split, final=False):
    if split == 'final':
        assert final, 'final views only with --i-am-final'
        from rotlab.camp_newdata import open_holdout
        open_holdout()
    man = json.loads((ND / 'rlivit.json').read_text()); z = zipfile.ZipFile(ZIP)
    acc = {m: dict(png=[], theta=[], ids=[], location=[], daytime=[]) for m in ('rgb', 'thermal')}; support = []
    for r in man['sequences']:
        if r['split'] != split:
            continue
        t = random.Random(f'rwu27:rlivit:{r["id"]}').uniform(0, 360)
        srcs, ok = {}, True
        for mod in ('rgb', 'thermal'):   # support check on THIS frame: inscribed circle must be free of padding (exact zeros)
            raw = z.read(f'{ROOT}{mod}/{r["seq"]}/{POS}.png')
            if hashlib.sha256(raw).hexdigest() != r[f'sha256_{mod}']:
                raise SystemExit(f'{r["id"]} {mod}: source PNG bytes differ from the manifest')
            src = Image.open(io.BytesIO(raw)).convert('RGB'); x0, y0, x1, y1 = man['locations'][r['location']]['rect'][mod]
            d = min(x1 - x0, y1 - y0) - 4; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            a = np.asarray(src); yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
            circ = (xx - cx) ** 2 + (yy - cy) ** 2 <= (d / 2) ** 2
            zero_frac = float((a[circ].max(1) == 0).mean())
            support.append(dict(id=r['id'], mod=mod, zero_frac=zero_frac))
            ok &= zero_frac <= 0.001; srcs[mod] = (src, d, cx, cy)
        if not ok:            # exclude the PAIR (pairing preserved), with a receipt
            continue
        for mod in ('rgb', 'thermal'):
            src, d, cx, cy = srcs[mod]; x0, y0, x1, y1 = cx - 1, cy - 1, cx + 1, cy + 1
            v = render(src, t, 0.8 * d, 0.6 * d, round(0.8 * d), round(0.6 * d), (x0 + x1) / 2, (y0 + y1) / 2)
            b = io.BytesIO(); v.save(b, format='PNG')
            for k, val in (('png', b.getvalue()), ('theta', t), ('ids', r['id']), ('location', r['location']), ('daytime', r['daytime'])):
                acc[mod][k].append(val)
    (ND / 'views').mkdir(parents=True, exist_ok=True)
    (ND / f'rlivit_{split}_support.json').write_text(json.dumps(dict(max_zero_frac=max(x['zero_frac'] for x in support),
        excluded=sorted({x['id'] for x in support if x['zero_frac'] > 0.001}), frames=support), indent=1))
    for mod, dd in acc.items():
        dd['theta'] = np.array(dd['theta']); f = ND / 'views' / f'rlivit_{split}_{mod}.pkl'
        f.write_bytes(pickle.dumps(dd)); print('views', f.name, len(dd['png']), hashlib.sha256(f.read_bytes()).hexdigest()[:16], flush=True)
        if split == 'final':
            from rotlab.camp_final import register_views
            register_views(f.stem, f, dd['ids'])


if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'prepare':
        cmd_prepare()
    elif c == 'views':
        cmd_views(sys.argv[2], '--i-am-final' in sys.argv)
