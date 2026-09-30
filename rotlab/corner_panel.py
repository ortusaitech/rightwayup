"""Corner panel (lead 30 Sep 2026, research/pico/PREREG.md addendum K1): stored DEV views turned by a further seeded angle
with a canvas-expanding PIL rotation and flat-colour corners, scored through the RELEASE pipeline (package Orienter).

  build:  python -m rotlab.corner_panel build gf_photo2_clean,gf_dev_all_clean          (P1: double rotation)
  build2: python -m rotlab.corner_panel build2 gf_photo2_clean,gf_dev_all_clean         (P2: upright, then turned)
  score:  python -m rotlab.corner_panel score MODEL_DIR TIER[,TIER..] [--precision fp32] [--tag NAME]

Labels: the view's theta (total clockwise orientation) + delta. Views are never used for training, calibration or model
choice of the fix beyond the registered acceptance check.
"""
from __future__ import annotations
import io, json, os, pickle, random, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

DATA = Path(os.environ.get('ROTLAB_DATA', '/workspace/rotation-data'))
CV = DATA / 'rotlab/cviews'
OUT = DATA / 'rotlab/corner-panel'
SEED = 20260930


def fill_of(rng):
    u = rng.random()
    if u < 0.5:
        return 'black', (0, 0, 0)
    if u < 0.75:
        return 'white', (255, 255, 255)
    return 'random', tuple(rng.randint(0, 255) for _ in range(3))


def build(names):
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        f = OUT / f'{name}_corner.pkl'
        if f.exists():
            raise SystemExit(f'{f} exists (panels are built once)')
        d = pickle.loads((CV / f'{name}.pkl').read_bytes())
        rng = random.Random(f'{SEED}:{name}')
        png, theta, delta, fill = [], [], [], []
        for b, t in zip(d['png'], d['theta']):
            im = Image.open(io.BytesIO(b)).convert('RGB')
            dl = rng.uniform(0.0, 360.0); kind, col = fill_of(rng)
            v = im.rotate(-dl, resample=Image.BICUBIC, expand=True, fillcolor=col)   # PIL: +angle = counter-clockwise
            o = io.BytesIO(); v.save(o, format='PNG'); png.append(o.getvalue())
            theta.append(float((t + dl) % 360.0)); delta.append(dl); fill.append(kind)
        f.write_bytes(pickle.dumps(dict(png=png, theta=theta, delta=delta, fill=fill, source=name, seed=SEED)))
        import hashlib
        print(name, len(png), 'views ->', f, hashlib.sha256(f.read_bytes()).hexdigest()[:16], flush=True)


def upright(im, theta):
    """The view turned back to upright (its content's labelled orientation removed) and cut to the largest axis-aligned
    rectangle inside the turned frame, i.e. a corner-free image whose content is upright in its frame."""
    from rotlab.core import largest_rotated_rect
    w, h = im.size
    v = im.rotate(theta, resample=Image.BICUBIC, expand=True)   # content turned theta counter-clockwise -> upright
    cw, ch = largest_rotated_rect(w, h, theta)
    cw, ch = max(1.0, cw - 2), max(1.0, ch - 2)
    l, t = (v.size[0] - cw) / 2, (v.size[1] - ch) / 2
    return v.crop((round(l), round(t), round(l + cw), round(t + ch)))


FILL = (124, 116, 104)   # letterbox fill (rightwayup.core.FILL): pixels of this colour are fill-dropped


def build2(names, variant='corner'):
    """P2 (lead 30 Sep 01:48, PREREG deviation): the use case. Each dev view is first made upright and corner-free
    (upright()), then turned by a seeded uniform delta with rotate(expand=True) and flat corners; label = delta."""
    OUT.mkdir(parents=True, exist_ok=True)
    import hashlib
    for name in names:
        f = OUT / (f'{name}_upright_corner.pkl' if variant == 'corner' else f'{name}_upright-{variant}_corner.pkl')
        if f.exists():
            raise SystemExit(f'{f} exists (panels are built once)')
        d = pickle.loads((CV / f'{name}.pkl').read_bytes())
        rng = random.Random(f'{SEED}:up:{name}')
        png, theta, delta, fill, skipped = [], [], [], [], 0
        for b, t in zip(d['png'], d['theta']):
            dl = rng.uniform(0.0, 360.0); kind, col = fill_of(rng)
            up = upright(Image.open(io.BytesIO(b)).convert('RGB'), float(t))
            if min(up.size) < 32:
                skipped += 1; continue
            lab = dl
            if variant == 'fillc':    # control: same geometry, corners exactly the letterbox fill (dropped by fill-drop)
                v = up.rotate(-dl, resample=Image.BICUBIC, expand=True, fillcolor=FILL)
            elif variant == 'q90':    # control: same content, nearest quarter turn (exact, no corners, no resampling)
                q = int(round(dl / 90.0)) % 4; lab = 90.0 * q
                v = up.transpose([None, Image.ROTATE_270, Image.ROTATE_180, Image.ROTATE_90][q]) if q else up
            else:
                v = up.rotate(-dl, resample=Image.BICUBIC, expand=True, fillcolor=col)
            o = io.BytesIO(); v.save(o, format='PNG'); png.append(o.getvalue())
            theta.append(float(lab % 360.0)); delta.append(dl); fill.append(kind)
        src = name + ':upright' + ('' if variant == 'corner' else '-' + variant)
        f.write_bytes(pickle.dumps(dict(png=png, theta=theta, delta=delta, fill=fill, source=src, seed=SEED, skipped=skipped)))
        print(name, len(png), 'views (skipped', skipped, ') ->', f, hashlib.sha256(f.read_bytes()).hexdigest()[:16], flush=True)


def circ(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % 360.0
    return np.minimum(d, 360.0 - d)


def score(model_dir, tiers, precision='fp32', tag='', only='', device='cpu', repaint=False):
    os.environ['RIGHTWAYUP_MODEL_DIR'] = str(model_dir)
    import rightwayup as rw
    res = {}
    for f in sorted(OUT.glob('*_corner.pkl')):
        if only and only not in f.name:
            continue
        d = pickle.loads(f.read_bytes())
        ims = [Image.open(io.BytesIO(b)).convert('RGB') for b in d['png']]
        th = np.asarray(d['theta']); fill = np.asarray(d['fill'])
        if repaint:   # PREREG round 2, R2-2: detected rotation corners repainted with the letterbox fill
            from rotlab import cornerfill
            rp = [cornerfill.repaint(im) for im in ims]; ims = [x for x, _ in rp]
            print(tag, d['source'], 'repaint detected', sum(i['reason'] == 'ok' for _, i in rp), '/', len(rp), flush=True)
            d['source'] = d['source'] + "+repaint"
        for tier in tiers:
            o = rw.Orienter(tier, device=device, precision=precision)
            t0 = time.time(); rs = []
            for i in range(0, len(ims), 64):
                rs += o.predict_batch(ims[i:i + 64])
            e = circ([r.angle_cw for r in rs], th); ab = np.array([r.abstain for r in rs])
            m = dict(n=len(e), w10=float((e <= 10).mean() * 100), ge150=float((e >= 150).mean() * 100),
                     answered=float((~ab).mean() * 100), wrong_answered=float(((e > 10) & ~ab).sum() / max(1, (~ab).sum()) * 100),
                     by_fill={k: dict(n=int((fill == k).sum()), w10=float((e[fill == k] <= 10).mean() * 100),
                                      ge150=float((e[fill == k] >= 150).mean() * 100)) for k in ('black', 'white', 'random')},
                     secs=round(time.time() - t0, 1))
            res[f'{d["source"]}:{tier}'] = m
            np.savez(OUT / f'items-{tag}-{precision}-{device}-{d["source"].replace(":", "_")}-{tier}.npz', angle=np.array([r.angle_cw for r in rs]),
                     conf=np.array([r.confidence for r in rs]), abstain=ab, theta=th, delta=np.asarray(d['delta']), fill=fill,
                     size=np.array([im.size for im in ims]))
            print(tag, d['source'], tier, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items() if k != 'by_fill'}),
                  {k: (round(v['w10'], 1), round(v['ge150'], 1)) for k, v in m['by_fill'].items()}, flush=True)
    return res


def triggers(names):
    """Zero-trigger check for R2-2: the corner detector over stored views (no model)."""
    from rotlab import cornerfill
    for name in names:
        f = CV / f'{name}.pkl'
        f = f if f.exists() else DATA / 'rotlab/newdata/views' / f'{name}.pkl'
        d = pickle.loads(f.read_bytes())
        hits = [i for i, b in enumerate(d['png']) if cornerfill.detect(np.asarray(Image.open(io.BytesIO(b)).convert('RGB')))[0] is not None]
        print('triggers', name, len(hits), '/', len(d['png']), hits[:10], flush=True)


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if a[0] == 'build':
        build(a[1].split(','))
    elif a[0] == 'triggers':
        triggers(a[1].split(','))
    elif a[0] == 'build2':
        build2(a[1].split(','), a[a.index('--variant') + 1] if '--variant' in a else 'corner')
    elif a[0] == 'score':
        prec = a[a.index('--precision') + 1] if '--precision' in a else 'fp32'
        tag = a[a.index('--tag') + 1] if '--tag' in a else Path(a[1]).name
        only = a[a.index('--only') + 1] if '--only' in a else ''
        dev = a[a.index('--device') + 1] if '--device' in a else 'cpu'
        r = score(a[1], a[2].split(','), prec, tag, only, dev, '--repaint' in a)
        (OUT / f'score-{tag}-{prec}-{dev}{"-" + only if only else ""}{"-repaint" if "--repaint" in a else ""}.json').write_text(json.dumps(r, indent=1))


if __name__ == '__main__':
    main()
