"""Trace-scrubbed view rendering (lead, 28 Sep 2026; motivated by research/trace-audit/REPORT.md).

The production renderer rotates a JPEG parent with bicubic interpolation at native resolution, which leaves two cues that a
physically rolled camera never produces: periodic interpolation traces aligned with the source pixel grid, and the source's
8x8 JPEG block grid rotated with the content. Both encode theta mod 90. This module removes or decorrelates them:

  ss_render    drop-in for core.render: parent crop upsampled 2x (Lanczos), same affine render at 2x (bicubic), antialiased
               Lanczos 2x downsample (= the audit's (b') renderer). Skipped when the view is already downscaled >= 2x,
               where the audit found the interpolation cue gone (E3).
  decoy_grid   re-encode the view as JPEG in a frame rotated by a random angle beta (rotate, encode, rotate back): adds a
               block grid and resampling traces that point at a random angle, so grid-like cues stop being informative.
  camera_jpeg  raster-aligned JPEG, like a camera encoder.

Usage in training: --scrub ss=1,decoy=0.35,cam=0.4 (train_ddp / filldrop), probabilities per view.
"""
from __future__ import annotations
import io, math, random
from PIL import Image

from rotlab import core
from rotlab.core import FILL

_RENDER = core.render   # the production bicubic renderer


def parse(spec: str) -> dict:
    """ss / decoy / cam / probe: per-view probabilities; dscale=1: decoy block scale randomised (x0.7-1.4);
    dual=1: Views also returns the CLEAN view (render only, no decoy/camera JPEG/degradation) for the teacher;
    down=f (f > 1): the rendered view is downscaled by f with an antialiasing Lanczos filter aligned with the OUTPUT raster
    (lead 28 Sep 14:25: a view rendered at ~1:1 from its parent keeps the parent's rotated spectral support, a theta-mod-90
    cue even after supersampling; at f >= ~1.42 the rotated parent band covers the output band and the cue is gone);
    cuekd=p (needs dual=1, down>1 and a teacher; lead 28 Sep 20:45): with probability p the STUDENT sees the ~1:1 (cue-bearing)
    view while the teacher sees the same view downscaled by `down` (cue-free), and that sample is trained by KD only (no hard
    label), so the rotated-spectral cue can no longer lower the loss -> direct pressure to unlearn it;
    leanpen=L (lead 28 Sep 21:00, owner: "penalise learning the cue"): probes carry their cue angle t (label 2000 + t) and the
    loss adds L * mean(soft^2), soft = sum_phi p(phi) cos 4(phi - t), i.e. the measured lean toward the cue on content-free
    stimuli, which keeps a gradient where the uniform-target CE has almost none;
    corner=p (lead 30 Sep 01:30, owner: fix Pico on rotation corners): with probability p the view is the WHOLE (sub)frame
    turned with flat-colour corners (corner_view) instead of a crop inside the rotated frame."""
    d = dict(ss=0.0, decoy=0.0, cam=0.0, probe=0.0, dscale=0.0, dual=0.0, down=0.0, cuekd=0.0, leanpen=0.0, corner=0.0)
    for kv in filter(None, (spec or '').split(',')):
        k, v = kv.split('='); assert k in d, k; d[k] = float(v)
    return d


def ss_render(src: Image.Image, applied_cw: float, crop_w: float, crop_h: float, out_w: int, out_h: int,
              cx: float | None = None, cy: float | None = None) -> Image.Image:
    W, H = src.size
    cx = W / 2 if cx is None else cx
    cy = H / 2 if cy is None else cy
    if max(crop_w / out_w, crop_h / out_h) >= 2.0:
        return _RENDER(src, applied_cw, crop_w, crop_h, out_w, out_h, cx, cy)
    # upsample only the source bounding box of the rotated crop (+ margin for the bicubic support)
    r = math.radians(applied_cw); c, s = abs(math.cos(r)), abs(math.sin(r))
    bx, by = crop_w / 2 * c + crop_h / 2 * s + 4, crop_w / 2 * s + crop_h / 2 * c + 4
    x0, y0 = max(0, int(math.floor(cx - bx))), max(0, int(math.floor(cy - by)))
    x1, y1 = min(W, int(math.ceil(cx + bx))), min(H, int(math.ceil(cy + by)))
    sub = src.crop((x0, y0, x1, y1))
    sub2 = sub.resize((2 * sub.size[0], 2 * sub.size[1]), Image.LANCZOS)
    v2 = _RENDER(sub2, applied_cw, 2 * crop_w, 2 * crop_h, 2 * out_w, 2 * out_h, 2 * (cx - x0), 2 * (cy - y0))
    return v2.resize((out_w, out_h), Image.LANCZOS)


def jpeg(im: Image.Image, q: int, subsampling: int) -> Image.Image:
    b = io.BytesIO(); im.save(b, format='JPEG', quality=q, subsampling=subsampling); b.seek(0)
    return Image.open(b).convert('RGB')


def decoy_grid(view: Image.Image, rng: random.Random, dscale: bool = False) -> Image.Image:
    """JPEG block grid (+ resampling traces) at a random angle beta; with dscale the block size in view pixels is
    8/r for r ~ U(0.7, 1.4), so no single grid period identifies the decoy."""
    beta = rng.uniform(5.0, 85.0) + 90.0 * rng.randrange(4)
    w, h = view.size
    v = view
    if dscale:
        r = rng.uniform(0.7, 1.4)
        v = view.resize((max(16, round(w * r)), max(16, round(h * r))), Image.LANCZOS)
    r1 = v.rotate(beta, resample=Image.BICUBIC, expand=True, fillcolor=FILL)
    if rng.random() < 0.5:   # lead 28 Sep (pilot): half the decoys are pure resampling decoys (conflicting interpolation
        r1 = jpeg(r1, rng.randint(55, 95), rng.choice([0, 2]))   # traces, no JPEG grid)
    r2 = r1.rotate(-beta, resample=Image.BICUBIC, expand=True, fillcolor=FILL)
    vw, vh = v.size
    l, t = round((r2.size[0] - vw) / 2), round((r2.size[1] - vh) / 2)
    out = r2.crop((l, t, l + vw, t + vh))
    return out.resize((w, h), Image.LANCZOS) if out.size != (w, h) else out


def camera_jpeg(view: Image.Image, rng: random.Random) -> Image.Image:
    return jpeg(view, rng.randint(70, 95), rng.choice([0, 2, 2]))


def downscale(v, f):
    return v.resize((max(16, round(v.size[0] / f)), max(16, round(v.size[1] / f))), Image.LANCZOS) if f > 1.0 else v


def corner_colour(rng: random.Random):
    u = rng.random()
    if u < 0.45:
        return (0, 0, 0)
    if u < 0.65:
        return (255, 255, 255)
    if u < 0.75:
        g = rng.randint(20, 235); return (g, g, g)
    return tuple(rng.randint(0, 255) for _ in range(3))


def corner_view(src: Image.Image, base_cw: float, theta: float, rng: random.Random) -> Image.Image:
    """The whole frame (or an unrotated sub-crop of it) turned to total orientation theta the way an editor or
    PIL/OpenCV rotation does it: canvas expanded to the rotated bounding box (70%) or kept at the frame size (30%), the
    uncovered corners filled with a flat colour (black 45%, white 20%, grey 10%, random 25%). Rendered supersampled like
    ss_render (source at 2x the output scale, bicubic turn, antialiased Lanczos 2x downsample), output side <= 448 like
    sample_view. Motivation (lead 30 Sep 01:10): PICO-N70L misread such images by ~180 deg with high confidence (training
    views were always crops inside the rotated frame); the larger tiers were not affected."""
    W, H = src.size
    if rng.random() < 0.4:   # framing / aspect variety: an unrotated sub-crop, then the whole sub-frame is turned
        fw, fh = rng.uniform(0.55, 1.0), rng.uniform(0.55, 1.0)
        cw_, ch_ = max(16, round(W * fw)), max(16, round(H * fh))
        x0, y0 = rng.randint(0, W - cw_), rng.randint(0, H - ch_)
        src = src.crop((x0, y0, x0 + cw_, y0 + ch_)); W, H = src.size
    applied = (theta - base_cw) % 360
    expand = rng.random() < 0.7
    r = math.radians(applied); c, s = abs(math.cos(r)), abs(math.sin(r))
    bw, bh = (W * c + H * s, W * s + H * c) if expand else (W, H)
    k = min(1.0, 448 / max(bw, bh))
    if rng.random() < 0.3:
        k *= rng.uniform(0.4, 1.0)   # smaller frames (CCTV-like resolutions)
    s2 = src.resize((max(16, round(2 * k * W)), max(16, round(2 * k * H))), Image.LANCZOS)
    v = s2.rotate(-applied, resample=Image.BICUBIC, expand=expand, fillcolor=corner_colour(rng))   # PIL: +angle = CCW
    return v.resize((max(8, round(v.size[0] / 2)), max(8, round(v.size[1] / 2))), Image.LANCZOS)


def render_view(sample_view, src, base_cw, theta, rng, cfg, nodown=False, **kw):
    """sample_view with the scrub renderer choice applied (patches core.render for this call only): the CLEAN view."""
    if cfg.get('corner', 0.0) > 0 and rng.random() < cfg['corner']:
        v = corner_view(src, base_cw, theta, rng)
        return v if nodown else downscale(v, cfg.get('down', 0.0))
    use_ss = cfg['ss'] > 0 and rng.random() < cfg['ss']
    core.render = ss_render if use_ss else _RENDER
    try:
        v = sample_view(src, base_cw, theta, rng, **kw)
    finally:
        core.render = _RENDER
    return v if nodown else downscale(v, cfg.get('down', 0.0))


def clutter(im, rng, cfg):
    """Uninformative grids on a clean view: decoy grid at a random angle and/or a raster-aligned camera JPEG."""
    if cfg['decoy'] > 0 and rng.random() < cfg['decoy']:
        im = decoy_grid(im, rng, cfg['dscale'] > 0)
    if cfg['cam'] > 0 and rng.random() < cfg['cam']:
        im = camera_jpeg(im, rng)
    return im


def view(sample_view, src, base_cw, theta, rng, cfg, **kw):
    return clutter(render_view(sample_view, src, base_cw, theta, rng, cfg, **kw), rng, cfg)


def probe_image(rng: random.Random, return_angle: bool = False):
    """Content-free shortcut stimulus (target = uniform): pink noise or flat grey + noise, JPEG-encoded at 0 deg, then
    rotated by a random angle with the PLAIN bicubic renderer, so it carries exactly the grid / interpolation cues and no
    content. Teaches that these cues carry no orientation."""
    import numpy as np
    g = np.random.default_rng(rng.getrandbits(32))
    W, H = rng.randint(260, 448), rng.randint(200, 448)
    if rng.random() < 0.5:
        fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.rfftfreq(W)[None, :]; f = np.sqrt(fx ** 2 + fy ** 2)
        amp = np.where((f > 0) & (f <= 0.5), 1.0 / np.maximum(f, 1e-9), 0.0)
        ch = []
        for _ in range(3):
            z = g.standard_normal(amp.shape) + 1j * g.standard_normal(amp.shape)
            x = np.fft.irfft2(z * amp, s=(H, W)); ch.append((x - x.mean()) / (x.std() + 1e-9))
        a = np.stack(ch, -1); a = rng.uniform(100, 156) + rng.uniform(20, 45) * a
    else:
        a = rng.uniform(40, 215) + g.normal(0, rng.uniform(1, 6), (H, W, 3))
    src = Image.fromarray(np.clip(np.round(a), 0, 255).astype(np.uint8))
    if rng.random() < 0.85:
        src = jpeg(src, rng.randint(60, 95), rng.choice([0, 2]))
    t = rng.uniform(0, 360); d = min(W, H) - 4
    cw, ch_ = 0.8 * d, 0.6 * d
    im = _RENDER(src, t, cw, ch_, round(cw), round(ch_))
    return (im, t) if return_angle else im
