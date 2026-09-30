"""rotlab core: sources, rendering, degradations, preprocessing, targets, metrics, benchmark pack.

Angle chart (same as the hybrid corpus and the benchmark pack): theta in [0,360) is the
CLOCKWISE rotation of image content relative to upright. Rotating the image counter-
clockwise by theta restores it. Pack `target_degrees` uses this chart for every panel.
"""
from __future__ import annotations
import os
import io, json, math, mmap, random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 64 * 1024 * 1024   # lead 28 Sep: grid-free PNG packs may carry large ICC profiles (metadata only)

DATA = Path(os.environ.get('ROTLAB_DATA', '/workspace/rotation-data'))   # data root; override with ROTLAB_DATA
SRC_DIR = DATA / 'rotlab/sources'
PACK = DATA / 'evaluation/rotation-benchmark-pack-v1'
CANVAS = 224
FILL = (124, 116, 104)  # ImageNet mean in uint8; letterbox padding
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ---------------------------------------------------------------- sources
class Blob:
    """Random access to one packed family (<name>.bin + <name>.jsonl)."""

    def __init__(self, name, src_dir=SRC_DIR, filt=None):
        self.name = name
        self.rows = [json.loads(l) for l in (src_dir / f'{name}.jsonl').open()]
        if filt:
            self.rows = [r for r in self.rows if filt(r)]
        self._path = src_dir / f'{name}.bin'
        self._mm = None

    def __len__(self):
        return len(self.rows)

    def image(self, i):
        if self._mm is None:  # lazily per process (fork-safe)
            f = self._path.open('rb')
            self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        r = self.rows[i]
        return Image.open(io.BytesIO(self._mm[r['offset']:r['offset'] + r['length']])).convert('RGB'), r


# ---------------------------------------------------------------- geometry
def max_rect(w, h, applied_cw, aspect):
    """Largest centred axis-aligned (in the output frame) rect of given aspect (w/h) that
    fits inside a w x h source rotated by applied_cw degrees. Returns (crop_w, crop_h)."""
    t = math.radians(applied_cw % 180.0)
    c, s = abs(math.cos(t)), abs(math.sin(t))
    g = 2.0  # resampling guard, as legacy renderer
    hh = min((w - 2 * g) / (aspect * c + s), (h - 2 * g) / (aspect * s + c))
    return aspect * hh, hh


def largest_rotated_rect(w, h, applied_cw):
    """Maximal-AREA axis-aligned rect inside a w x h source rotated by applied_cw degrees (free aspect).
    Same formula as the incumbent's rotation_utils.largest_rotated_rect, i.e. its training AND benchmark crop."""
    a = math.radians(applied_cw)
    long_w = w >= h
    side_long, side_short = (w, h) if long_w else (h, w)
    sa, ca = abs(math.sin(a)), abs(math.cos(a))
    if side_short <= 2 * sa * ca * side_long or abs(sa - ca) < 1e-10:
        x = 0.5 * side_short
        return (x / sa, x / ca) if long_w else (x / ca, x / sa)
    c2 = ca * ca - sa * sa
    return (w * ca - h * sa) / c2, (h * ca - w * sa) / c2


def render(src: Image.Image, applied_cw: float, crop_w: float, crop_h: float, out_w: int, out_h: int,
           cx: float | None = None, cy: float | None = None) -> Image.Image:
    """Output = content rotated clockwise by applied_cw, cropped (crop_w x crop_h source px),
    centred at source point (cx, cy) in rotated frame coordinates, rendered at out_w x out_h.
    Same affine convention as legacy materialize_post_convergence_fresh_stage0.transform_image."""
    W, H = src.size
    cx = W / 2 if cx is None else cx
    cy = H / 2 if cy is None else cy
    sx, sy = crop_w / out_w, crop_h / out_h
    r = math.radians(applied_cw)
    co, si = math.cos(r), math.sin(r)
    a, b = co * sx, si * sy
    d, e = -si * sx, co * sy
    c0 = cx - a * (out_w / 2) - b * (out_h / 2)
    f0 = cy - d * (out_w / 2) - e * (out_h / 2)
    return src.transform((out_w, out_h), Image.Transform.AFFINE, (a, b, c0, d, e, f0),
                         resample=Image.Resampling.BICUBIC, fillcolor=FILL)


ASPECTS = [(16 / 9, .25), (5 / 3, .20), (4 / 3, .15), (3 / 2, .10), (1.0, .10), (3 / 4, .12), (2 / 3, .08)]

# Aspect buckets with (nearly) equal token budgets (patch 14): same CPU cost, more real pixels.
BUCKETS = {'L': (168, 294), 'S': (224, 224), 'P': (294, 168)}   # (height, width)
BUCKET_ASPECTS = {'L': [(16 / 9, .4), (5 / 3, .3), (3 / 2, .15), (4 / 3, .15)],
                  'S': [(1.0, .5), (5 / 4, .25), (4 / 5, .25)],
                  'P': [(3 / 4, .45), (2 / 3, .35), (9 / 16, .2)]}
BUCKET_P = {'L': .6, 'S': .1, 'P': .3}


def bucket_of(w, h):
    r = w / h
    return 'L' if r >= 1.25 else ('P' if r <= 0.8 else 'S')


def sample_view(src: Image.Image, base_cw: float, theta: float, rng: random.Random, zoom_p=0.5, aspects=None, maxarea_p=0.0):
    """Render a training view whose TOTAL orientation is theta (clockwise). With probability maxarea_p
    the view is instead the incumbent-benchmark crop: centred maximal-area rect (angle-dependent shape)."""
    W, H = src.size
    applied = (theta - base_cw) % 360
    if maxarea_p and rng.random() < maxarea_p:
        cw, ch = largest_rotated_rect(W, H, applied)
        cw, ch = max(8.0, cw - 4), max(8.0, ch - 4)
        s = min(1.0, 448 / max(cw, ch))
        return render(src, applied, cw, ch, max(8, round(cw * s)), max(8, round(ch * s)))
    aspects = aspects or ASPECTS
    aspect = rng.choices([a for a, _ in aspects], [p for _, p in aspects])[0]
    cw, ch = max_rect(W, H, applied, aspect)
    z = rng.uniform(0.65, 1.0) if rng.random() < zoom_p else 1.0
    zw, zh = cw * z, ch * z
    # random sub-rect inside the max rect: offset in output(rotated) frame -> source frame
    ox, oy = rng.uniform(-(cw - zw) / 2, (cw - zw) / 2), rng.uniform(-(ch - zh) / 2, (ch - zh) / 2)
    r = math.radians(applied)
    # output-frame x axis maps to source (cos, -sin); y axis to (sin, cos)
    scx = W / 2 + ox * math.cos(r) + oy * math.sin(r)
    scy = H / 2 - ox * math.sin(r) + oy * math.cos(r)
    # render at native crop resolution (<= source px), like CCTV/benchmark frames
    ow = max(8, min(round(zw), 448)); oh = max(8, round(ow / aspect))
    return render(src, applied, zw, zh, ow, oh, scx, scy)


# ---------------------------------------------------------------- degradations (CCTV-ish)
def degrade(im: Image.Image, rng: random.Random, force: bool = False) -> Image.Image:
    if rng.random() < 0.35 and not force:
        return im
    w, h = im.size
    if rng.random() < 0.35:  # resolution loss
        f = rng.uniform(0.35, 0.9)
        im = im.resize((max(8, int(w * f)), max(8, int(h * f))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if rng.random() < 0.25:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.6)))
    a = np.asarray(im).astype(np.float32)
    if rng.random() < 0.15:  # monochrome / IR
        g = a @ np.array([0.299, 0.587, 0.114], np.float32)
        a = np.repeat(g[..., None], 3, 2)
    if rng.random() < 0.5:  # exposure / contrast / gamma
        a = np.clip((a - 128) * rng.uniform(0.6, 1.3) + 128 + rng.uniform(-40, 30), 0, 255)
        gm = rng.uniform(0.7, 1.5)
        a = 255 * (a / 255) ** gm
    if rng.random() < 0.3:
        a = a + np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(2, 12), a.shape)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        b = io.BytesIO(); im.save(b, format='JPEG', quality=rng.randint(25, 90)); b.seek(0)
        im = Image.open(b).convert('RGB')
    return im


# ---------------------------------------------------------------- model input
def letterbox(im: Image.Image, size=CANVAS) -> np.ndarray:
    """Aspect-preserving fit into a canvas (int = square, or (height, width)), centred on
    mean-colour padding -> CHW float32."""
    ch, cw = (size, size) if isinstance(size, int) else size
    w, h = im.size
    s = min(cw / w, ch / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    im = im.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new('RGB', (cw, ch), FILL)
    canvas.paste(im, ((cw - nw) // 2, (ch - nh) // 2))
    a = (np.asarray(canvas, np.float32) / 255 - MEAN) / STD
    return a.transpose(2, 0, 1).copy()


# ---------------------------------------------------------------- targets / decoding / metrics
def cgd_target(theta: float, sigma=6.0, bins=360) -> np.ndarray:
    k = np.arange(bins, dtype=np.float64)
    d = (k - theta + 180) % 360 - 180
    t = np.exp(-0.5 * (d / sigma) ** 2)
    return (t / t.sum()).astype(np.float32)


def circ_err(a, b):
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360)
    return np.minimum(d, 360 - d)


def decode(prob: np.ndarray) -> np.ndarray:
    """Argmax bin refined by local circular mean over +-10 bins."""
    k = prob.argmax(1)
    idx = (k[:, None] + np.arange(-10, 11)[None]) % 360
    w = np.take_along_axis(prob, idx, 1)
    off = (w * np.arange(-10, 11)[None]).sum(1) / np.maximum(w.sum(1), 1e-12)
    return (k + off) % 360


def metrics(err: np.ndarray) -> dict:
    ax = np.minimum(err, 180 - err)
    n = len(err)
    return dict(n=n, mean=round(float(err.mean()), 3), median=round(float(np.median(err)), 3),
                w10=int((err <= 10).sum()), w10_pct=round(100 * float((err <= 10).mean()), 2),
                gt30=int((err > 30).sum()), gt90=int((err > 90).sum()), t150=int((err >= 150).sum()),
                axial_w10=int((ax <= 10).sum()))


# ---------------------------------------------------------------- benchmark pack
def load_pack():
    man = json.load((PACK / 'MANIFEST.json').open())['rows']
    refs = [json.loads(l) for l in (PACK / 'REFERENCES.jsonl').open()]
    assert [r['id'] for r in man] == [r['id'] for r in refs]
    for m, r in zip(man, refs):
        m['faithful'] = r['arms']['faithful']['prediction_degrees']
        m['installed'] = r['arms']['installed']['prediction_degrees']
    return man
