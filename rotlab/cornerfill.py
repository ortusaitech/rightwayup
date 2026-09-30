"""Prototype (lead 30 Sep 2026): detect the flat-colour corner triangles that a rotation leaves (PIL/OpenCV/editor rotate with
an expanded or kept canvas) and repaint them with the letterbox fill colour, so fill-drop models ignore them exactly as they
ignore letterbox padding.

Signature checked (natural photos essentially never show it): at >= 3 image corners, a region of ONE flat colour whose runs
along the two borders (legs) define a right triangle; the triangle interior is flat; opposite corners have the same
hypotenuse angle and adjacent corners complementary angles (the sides of a rotated rectangle are perpendicular).
"""
import math
import numpy as np

FILL = np.array([124, 116, 104], np.uint8)


def _run(line):
    """Length of the leading run of True values."""
    idx = np.flatnonzero(~line)
    return int(idx[0]) if idx.size else int(line.size)


def detect(a, tol=10, min_leg=4, angle_tol=3.0, interior=0.98, min_corners=3):
    """a: HxWx3 uint8. -> (mask HxW bool or None, info dict)."""
    H, W = a.shape[:2]
    if H < 16 or W < 16:
        return None, dict(reason='small')
    cs = np.stack([a[0, 0], a[0, W - 1], a[H - 1, 0], a[H - 1, W - 1]]).astype(np.int16)
    c = np.median(cs, 0)
    if (np.abs(cs - c).max(1) <= tol).sum() < min_corners:
        return None, dict(reason='corner colours differ')
    flat_full = np.abs(a.astype(np.int16) - c).max(-1) <= tol
    rows = np.flatnonzero(~flat_full.all(1)); cols = np.flatnonzero(~flat_full.all(0))
    if rows.size < 16 or cols.size < 16:
        return None, dict(reason='flat image')
    Y1, Y2, X1, X2 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1     # fully flat border rows/columns are fill as well
    flat = flat_full[Y1:Y2, X1:X2]; H, W = flat.shape
    tri = {}
    # (name, row of the horizontal border, column of the vertical border, x direction, y direction)
    for name, y0, x0, sx, sy in [('tl', 0, 0, 1, 1), ('tr', 0, W - 1, -1, 1), ('bl', H - 1, 0, 1, -1), ('br', H - 1, W - 1, -1, -1)]:
        row = flat[y0, ::sx] if sx > 0 else flat[y0, ::-1]
        col = flat[::sy, x0] if sy > 0 else flat[::-1, x0]
        hl, vl = _run(row), _run(col)
        if hl < min_leg or vl < min_leg or hl >= W - 1 or vl >= H - 1:
            continue
        # interior check: pixels with x/hl + y/vl <= 0.85 (local coordinates from this corner)
        yy, xx = np.mgrid[0:vl, 0:hl]
        inside = xx / hl + yy / vl <= 0.85
        ys = y0 + sy * yy[inside]; xs = x0 + sx * xx[inside]
        if inside.sum() and flat[ys, xs].mean() < interior:
            continue
        tri[name] = (hl, vl, math.degrees(math.atan2(vl, hl)))
    if len(tri) < min_corners:
        return None, dict(reason=f'{len(tri)} triangles', tri=tri)
    ang = {k: v[2] for k, v in tri.items()}
    checks = []
    for p, q in [('tl', 'br'), ('tr', 'bl')]:              # opposite corners: same angle
        if p in ang and q in ang:
            checks.append(abs(ang[p] - ang[q]))
    for p, q in [('tl', 'tr'), ('tr', 'br'), ('br', 'bl'), ('bl', 'tl')]:   # adjacent corners: complementary
        if p in ang and q in ang:
            checks.append(abs(ang[p] + ang[q] - 90.0))
    if not checks or max(checks) > angle_tol:
        return None, dict(reason='angles inconsistent', tri=tri, checks=checks)
    mask = np.zeros((H, W), bool)
    for name, (hl, vl, _) in tri.items():
        y0, x0, sx, sy = dict(tl=(0, 0, 1, 1), tr=(0, W - 1, -1, 1), bl=(H - 1, 0, 1, -1), br=(H - 1, W - 1, -1, -1))[name]
        yy, xx = np.mgrid[0:vl + 1, 0:hl + 1]
        t = xx / max(hl, 1) + yy / max(vl, 1) <= 1.0
        ys = np.clip(y0 + sy * yy[t], 0, H - 1); xs = np.clip(x0 + sx * xx[t], 0, W - 1)
        mask[ys, xs] = True
    mask &= flat                                           # repaint only the flat pixels of the triangles
    full = flat_full.copy(); full[Y1:Y2, X1:X2] = mask      # + the fully flat border rows / columns
    return full, dict(reason='ok', tri=tri, checks=checks, frac=float(full.mean()), bbox=(int(Y1), int(Y2), int(X1), int(X2)))


def repaint(im):
    """PIL image -> (PIL image with detected rotation corners set to FILL, info)."""
    from PIL import Image
    a = np.asarray(im.convert('RGB'))
    m, info = detect(a)
    if m is None:
        return im, info
    b = a.copy(); b[m] = FILL
    return Image.fromarray(b), info
