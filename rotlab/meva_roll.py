#!/usr/bin/env python3
"""Per-camera roll audit from scene geometry (independent of any learned model).

Near-vertical line segments from every frame of a static camera are pooled; their common
vanishing point (length-weighted least squares, RANSAC) gives the vertical direction, and roll is
the angle between that direction and the image's vertical axis (clockwise positive). A contact
sheet per camera is written for visual review.
  python -m rotlab.meva_roll FRAMES_DIR   (run where cv2 is available)
"""
import collections, json, math, random, sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def segments(path, max_tilt=30):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE); h, w = g.shape
    e = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 60, 160)
    ls = cv2.HoughLinesP(e, 1, np.pi / 720, 60, minLineLength=0.08 * h, maxLineGap=4)
    out = []
    for x1, y1, x2, y2 in (ls[:, 0] if ls is not None else []):
        if abs(math.degrees(math.atan2(x2 - x1, y2 - y1 if y2 != y1 else 1e-9))) % 180 in ():
            pass
        dx, dy = x2 - x1, y2 - y1
        tilt = math.degrees(math.atan2(dx, dy)); tilt = (tilt + 90) % 180 - 90
        if abs(tilt) <= max_tilt:
            out.append((x1 / w, y1 / h, x2 / w, y2 / h, math.hypot(dx / w, dy / h)))
    return out, (w, h)


def vp_roll(segs, aspect, iters=400, tol=1.5):
    """segments in normalised coords -> (roll_deg, inliers). Lines in pixel-proportional coords."""
    L = []
    for x1, y1, x2, y2, n in segs:
        p1 = np.array([x1 * aspect, y1, 1.0]); p2 = np.array([x2 * aspect, y2, 1.0])
        l = np.cross(p1, p2); l /= np.linalg.norm(l[:2]); L.append((l, n))
    if len(L) < 8:
        return None, 0
    Ls = np.array([l for l, _ in L]); W = np.array([n for _, n in L]); c = np.array([aspect / 2, 0.5])
    rng = random.Random(0); best = (None, -1, None)

    def angdist(vp):  # angle (deg) between each segment and the direction to vp from its midpoint
        return None

    mids = []
    for x1, y1, x2, y2, _ in segs:
        mids.append(((x1 + x2) / 2 * aspect, (y1 + y2) / 2, (x2 - x1) * aspect, y2 - y1))
    mids = np.array(mids)

    def residual(vp):
        # vp homogeneous; direction from midpoint to vp vs segment direction
        dx = vp[0] - mids[:, 0] * vp[2]; dy = vp[1] - mids[:, 1] * vp[2]
        a1 = np.arctan2(dx, dy); a2 = np.arctan2(mids[:, 2], mids[:, 3])
        d = np.degrees(np.abs((a1 - a2 + np.pi / 2) % np.pi - np.pi / 2))
        return d

    for _ in range(iters):
        i, j = rng.sample(range(len(L)), 2)
        vp = np.cross(Ls[i], Ls[j])
        if not np.all(np.isfinite(vp)) or np.linalg.norm(vp) == 0:
            continue
        vp = vp / np.linalg.norm(vp)
        inl = residual(vp) < tol; s = W[inl].sum()
        if s > best[1]:
            best = (vp, s, inl)
    vp, _, inl = best
    # refine: smallest singular vector of weighted inlier lines
    A = Ls[inl] * W[inl, None]
    vp = np.linalg.svd(A)[2][-1]
    dx, dy = vp[0] - c[0] * vp[2], vp[1] - c[1] * vp[2]
    if dy * np.sign(vp[2] if vp[2] != 0 else 1) < 0 or (vp[2] == 0 and dy < 0):
        dx, dy = -dx, -dy
    if dy < 0:
        dx, dy = -dx, -dy
    roll = math.degrees(math.atan2(dx, dy))   # + means the true vertical leans right going down
    return roll, int(inl.sum())


def sheet(paths, out, n=12, tile=320):
    paths = paths[:: max(1, len(paths) // n)][:n]
    cols = 4; rows = math.ceil(len(paths) / cols)
    S = Image.new('RGB', (cols * tile, rows * int(tile * 0.5625)), 'black')
    for k, p in enumerate(paths):
        im = Image.open(p).convert('RGB'); im.thumbnail((tile, tile))
        S.paste(im, ((k % cols) * tile, (k // cols) * int(tile * 0.5625)))
    S.save(out, quality=85)


def main():
    root = Path(sys.argv[1]); rows = [json.loads(l) for l in open(root / 'frames.jsonl')]
    by = collections.defaultdict(list)
    for r in rows:
        by[r['cam']].append(root / 'frames' / r['cam'] / Path(r['path']).name)
    res = {}
    (root / 'sheets').mkdir(exist_ok=True)
    for cam, ps in sorted(by.items()):
        segs = []; size = None
        for p in ps:
            s, size = segments(p); segs += s
        roll, n = vp_roll(segs, size[0] / size[1])
        per = []
        for p in ps[:: max(1, len(ps) // 8)]:
            s, sz = segments(p); r, k = vp_roll(s, sz[0] / sz[1])
            if r is not None:
                per.append(round(r, 2))
        res[cam] = dict(frames=len(ps), segments=len(segs), inliers=n, roll_deg=None if roll is None else round(roll, 2),
                        per_frame_rolls=per)
        sheet(ps, root / 'sheets' / f'{cam}.jpg')
        print(cam, res[cam], flush=True)
    (root / 'roll-audit.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
