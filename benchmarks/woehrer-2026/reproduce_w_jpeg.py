#!/usr/bin/env python3
"""Reproduce our benchmark notes on Woehrer 2026 (arXiv 2603.25351): its COCO rotation benchmark scored on the
views as published (lossless) and on the same views saved once as JPEG after rotation, as any camera or photo
pipeline would save them.

What it does
  1. Reads the 1,030 benchmark photos (COCO val2014, Fischer et al. 2015 labels 1 and 2) from --coco-dir, or
     downloads them from the COCO bucket (s3.amazonaws.com/images.cocodataset.org) with --download, and checks every file's SHA-256.
  2. Rebuilds the test views exactly as the upstream harness does: numpy.random.seed(seed); uniform(0, 360, 1030)
     angles in a fixed image order; OpenCV bilinear rotation with an expanded canvas; centre crop to the
     largest axis-aligned rectangle (upstream rotation_utils.py, MIT, copied below).
  3. Optionally re-encodes each view (JPEG at the given qualities, or a lossless PNG round trip).
  4. Runs Woehrer 2026 (the public checkpoint, or an ONNX export of it) with its own eval preprocessing
     (timm: shortest side to 224 bicubic, centre crop 224, ImageNet normalisation; argmax over 360 bins).
  5. Prints within-2/5/10 accuracy per seed and the five-seed mean, plus the share of predictions that land within
     2 degrees of 0/90/180/270, and writes a results JSON with per-view predictions.
  --protocol fair uses our angle-independent crop instead (fixed 4:3 rectangle whose diagonal is min(W,H) - 4, so
  the outline carries no angle information; one angle per photo from random.Random(7); PIL bicubic rendering).

Install (CPU is fine; about 0.2 to 0.3 s per view at 6 threads, so about 45 min for five seeds, clean + JPEG)
  pip install numpy pillow opencv-python-headless onnxruntime            # ONNX route
  pip install torch torchvision timm                                      # checkpoint route (--ckpt); torch and
                                                                          # torchvision must be matching builds
  For bit-exact views use opencv-python-headless==4.12.0.88 numpy==2.1.2 pillow==11.0.0; the script reports how many
  rendered views match our pixel hashes either way.

Model
  --ckpt   cgd_mambaout_base_coco2017.ckpt from https://huggingface.co/maxwoe/image-rotation-angle-estimation (MIT)
  --onnx   an ONNX export of that checkpoint with input [1,3,224,224] and 360 outputs (probabilities or logits)

Examples
  python reproduce_w_jpeg.py --coco-dir val2014 --download --ckpt cgd_mambaout_base_coco2017.ckpt \
      --seeds 0,1,2,3,4 --conditions clean,jpeg90 --out w-jpeg-results.json
  python reproduce_w_jpeg.py --coco-dir val2014 --onnx w.onnx --seeds 0 --conditions clean,png,jpeg95,jpeg90,jpeg75
  python reproduce_w_jpeg.py --coco-dir val2014 --onnx w.onnx --protocol fair --conditions clean,jpeg90

Angle conventions: the upstream harness rotates content counter-clockwise by the truth angle (cv2 convention) and
the model predicts that angle; error = circular |prediction - truth|. In --protocol fair the view is rendered with a
clockwise rotation th, so truth = (-th) mod 360 in the model's convention.
Script by ORTUS AI (Apache-2.0). The functions marked "upstream" are from maxwoe/image-rotation-angle-estimation
@ e4e35610, Copyright (c) 2026 Maximilian Woehrer, MAXI solutions e.U., MIT License.
"""
import argparse, hashlib, io, json, math, os, platform, random, sys, time, urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
import PIL

DEFS = Path(__file__).with_name('w_benchmark_views.json')
COCO_URL = 'https://s3.amazonaws.com/images.cocodataset.org/val2014/{}'
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ------------------------------------------------------------------ upstream rotation_utils.py (MIT, Woehrer 2026)
def rotate_image(img, angle, rotation_center=None, expand=False):
    import cv2
    h, w = img.shape[:2]
    if rotation_center is None:
        rotation_center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(rotation_center, angle, 1.0)
    if expand:
        abs_cos, abs_sin = abs(M[0, 0]), abs(M[0, 1])
        wn = int(h * abs_sin + w * abs_cos)
        hn = int(h * abs_cos + w * abs_sin)
        M[0, 2] += wn / 2 - rotation_center[0]
        M[1, 2] += hn / 2 - rotation_center[1]
    else:
        wn, hn = w, h
    return cv2.warpAffine(img, M, (wn, hn), borderMode=cv2.BORDER_CONSTANT, borderValue=0), M


def largest_rotated_rect(w, h, angle):
    if w <= 0 or h <= 0:
        return 0, 0
    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a
    return wr, hr


def rotate_image_crop_max_area(image, angle):
    h, w = image.shape[:2]
    rotated, _ = rotate_image(image, angle, expand=True)
    wr, hr = largest_rotated_rect(w, h, math.radians(angle))
    h_rot, w_rot = rotated.shape[:2]
    y1 = h_rot // 2 - int(hr / 2); y2 = y1 + int(hr)
    x1 = w_rot // 2 - int(wr / 2); x2 = x1 + int(wr)
    return rotated[y1:y2, x1:x2]


def rotate_preserve_content(image_path, angle):
    import cv2
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f'Could not load image: {image_path}')
    return Image.fromarray(cv2.cvtColor(rotate_image_crop_max_area(image, angle), cv2.COLOR_BGR2RGB))
# ------------------------------------------------------------------ end of upstream code


def render_fair(src, applied_cw):
    """ORTUS AI angle-independent crop: fixed 4:3 rectangle, diagonal = min(W,H) - 4, centred; PIL bicubic affine."""
    W, H = src.size
    d = min(W, H) - 4; cw, ch = 0.8 * d, 0.6 * d; ow, oh = round(cw), round(ch)
    sx, sy = cw / ow, ch / oh
    r = math.radians(applied_cw); co, si = math.cos(r), math.sin(r)
    a, b, dd, e = co * sx, si * sy, -si * sx, co * sy
    c0 = W / 2 - a * (ow / 2) - b * (oh / 2); f0 = H / 2 - dd * (ow / 2) - e * (oh / 2)
    return src.transform((ow, oh), Image.Transform.AFFINE, (a, b, c0, dd, e, f0), resample=Image.Resampling.BICUBIC,
                         fillcolor=(124, 116, 104))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def fetch_images(coco_dir, images, download):
    coco_dir.mkdir(parents=True, exist_ok=True); bad = []
    for i, im in enumerate(images):
        p = coco_dir / im['name']
        if not p.exists() and download:
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(COCO_URL.format(im['name']), timeout=60) as r:
                        data = r.read()
                    tmp = p.with_suffix('.part'); tmp.write_bytes(data); tmp.replace(p); break
                except Exception as ex:  # noqa: BLE001
                    if attempt == 2:
                        raise SystemExit(f'download failed for {im["name"]}: {ex}')
            if (i + 1) % 100 == 0:
                print(f'  downloaded {i + 1}/{len(images)}', flush=True)
        if not p.exists():
            raise SystemExit(f'missing {p} (use --download)')
        if sha(p.read_bytes()) != im['sha256']:
            bad.append(im['name'])
    if bad:
        raise SystemExit(f'{len(bad)} source files differ from the published COCO val2014 bytes, e.g. {bad[:3]}')


def encode(im, cond):
    """-> (PIL image as the model sees it, bytes or None)."""
    if cond == 'clean':
        return im, None
    b = io.BytesIO()
    if cond == 'png':
        im.save(b, format='PNG')
    elif cond.startswith('jpeg'):
        im.save(b, 'JPEG', quality=int(cond[4:]))
    else:
        raise SystemExit(f'unknown condition {cond}')
    data = b.getvalue()
    return Image.open(io.BytesIO(data)).convert('RGB'), data


def preprocess_pil(im, size=224):
    """Equals timm create_transform(input_size=224, interpolation='bicubic', crop_pct=1.0, is_training=False):
    torchvision Resize(224) on the shorter side (long side = int(224 * long / short)), CenterCrop(224), ToTensor, Normalize."""
    w, h = im.size
    if w <= h:
        nw, nh = size, int(size * h / w)
    else:
        nw, nh = int(size * w / h), size
    im = im.resize((nw, nh), Image.BICUBIC)
    top, left = int(round((nh - size) / 2.0)), int(round((nw - size) / 2.0))
    im = im.crop((left, top, left + size, top + size))
    a = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
    return a.transpose(2, 0, 1)[None].copy()


class Model:
    def __init__(self, a):
        self.kind = 'onnx' if a.onnx else 'ckpt'
        if a.preproc == 'timm':
            import timm.data
            tf = timm.data.create_transform(input_size=(3, 224, 224), interpolation='bicubic', crop_pct=1.0,
                                            mean=tuple(MEAN), std=tuple(STD), is_training=False)
            self.pre = lambda im: tf(im).unsqueeze(0).numpy()
        else:
            self.pre = preprocess_pil
        if a.onnx:
            import onnxruntime as ort
            so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
            self.s = ort.InferenceSession(a.onnx, so, providers=['CPUExecutionProvider']); self.inp = self.s.get_inputs()[0].name
            self.ident = dict(onnx=os.path.basename(a.onnx), onnx_sha256=sha(Path(a.onnx).read_bytes()))
        else:
            import torch, timm
            torch.set_num_threads(a.threads)
            ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
            sd = {k[6:]: v for k, v in ck['state_dict'].items() if k.startswith('model.')}
            name = ck.get('hyper_parameters', {}).get('model_name', 'mambaout_base')
            self.m = timm.create_model(name, pretrained=False, num_classes=360); self.m.load_state_dict(sd); self.m.eval()
            self.torch = torch
            self.ident = dict(ckpt=os.path.basename(a.ckpt), ckpt_sha256=sha(Path(a.ckpt).read_bytes()), timm_model=name)

    def argmax(self, im):
        x = self.pre(im)
        if self.kind == 'onnx':
            y = self.s.run(None, {self.inp: x})[0][0]
        else:
            with self.torch.no_grad():
                y = self.m(self.torch.from_numpy(x))[0].numpy()
        return int(np.argmax(y))


def circ(p, t):
    e = np.abs((np.asarray(p, np.float64) - np.asarray(t, np.float64)) % 360); return np.minimum(e, 360 - e)


def axis_dist(a):
    return np.abs(((np.asarray(a, np.float64) + 45) % 90) - 45)


def summary(pred, truth):
    e = circ(pred, truth)
    return dict(n=int(len(e)), within2=int((e <= 2).sum()), within5=int((e <= 5).sum()), within10=int((e <= 10).sum()),
                acc10=round(float((e <= 10).mean()), 4), mae=round(float(e.mean()), 3), median=round(float(np.median(e)), 3),
                ge150=int((e >= 150).sum()), pred_within2_of_axis=round(float((axis_dist(pred) <= 2).mean()), 4),
                truth_within2_of_axis=round(float((axis_dist(truth) <= 2).mean()), 4))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--coco-dir', required=True, type=Path); ap.add_argument('--download', action='store_true')
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument('--onnx'); g.add_argument('--ckpt')
    ap.add_argument('--protocol', choices=['maxarea', 'fair'], default='maxarea')
    ap.add_argument('--seeds', default='0,1,2,3,4', help='maxarea protocol test seeds (upstream uses 0-4)')
    ap.add_argument('--conditions', default='clean,jpeg90', help='comma list of clean, png, jpeg<Q>')
    ap.add_argument('--preproc', choices=['pil', 'timm'], default='pil'); ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0, help='first N photos only (smoke test)')
    ap.add_argument('--defs', type=Path, default=DEFS); ap.add_argument('--out', type=Path, default=Path('w-jpeg-results.json'))
    a = ap.parse_args()
    defs = json.loads(a.defs.read_text()); images = defs['images']
    conds = a.conditions.split(',')
    fetch_images(a.coco_dir, images, a.download)
    import cv2
    model = Model(a)
    res = dict(protocol=a.protocol, conditions=conds, model=model.ident, preprocessing=a.preproc,
               environment=dict(python=platform.python_version(), numpy=np.__version__, pillow=PIL.__version__, opencv=cv2.__version__),
               started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), runs={})
    n = a.limit or len(images)
    if a.protocol == 'maxarea':
        plan = []
        for s in [int(x) for x in a.seeds.split(',')]:
            np.random.seed(s); ang = np.random.uniform(0, 360, len(images))
            plan.append((f'seed{s}', s, [(images[i]['name'], float(ang[i])) for i in range(n)]))
    else:   # fair: sorted names, one angle each from random.Random(7) (clockwise), truth in W's convention = -th
        rng = random.Random(7); names = sorted(im['name'] for im in images); th = [rng.uniform(0, 360) for _ in names]
        plan = [('fair7', None, list(zip(names, th))[:n])]
    for tag, seed, rows in plan:
        t0 = time.time()
        if a.protocol == 'maxarea':
            views = [rotate_preserve_content(a.coco_dir / nm, ang).convert('RGB') for nm, ang in rows]
            truth = np.array([ang for _, ang in rows])
            ref = defs['clean_view_rgb_sha256'][str(seed)]
            match = sum(sha(np.asarray(v, np.uint8).tobytes()) == ref[i] for i, v in enumerate(views))
            print(f'{tag}: rendered {len(views)} views, {match} match our pixel hashes', flush=True)
        else:
            views = [render_fair(Image.open(a.coco_dir / nm).convert('RGB'), t) for nm, t in rows]
            truth = np.array([(-t) % 360 for _, t in rows]); match = None
        res['runs'][tag] = dict(seed=seed, rendered_rgb_hash_matches=match, truth=truth.round(6).tolist(), conditions={})
        for c in conds:
            preds, jmatch = [], 0
            for i, v in enumerate(views):
                im, data = encode(v, c)
                if c == 'jpeg90' and a.protocol == 'maxarea' and data is not None:
                    jmatch += sha(data) == defs['jpeg90_bytes_sha256'][str(seed)][i]
                preds.append(model.argmax(im))
            s = summary(preds, truth); s['predictions'] = preds
            if c == 'jpeg90' and a.protocol == 'maxarea':
                s['jpeg_bytes_match_ours'] = jmatch
            res['runs'][tag]['conditions'][c] = s
            print(f"{tag} {c:>7}: within10 {s['within10']}/{s['n']} ({100 * s['acc10']:.1f}%)  within2 {s['within2']}  "
                  f"MAE {s['mae']:.2f}  median {s['median']:.2f}  predictions within 2 deg of an axis "
                  f"{100 * s['pred_within2_of_axis']:.1f}% (truth {100 * s['truth_within2_of_axis']:.1f}%)", flush=True)
        print(f'{tag} done in {time.time() - t0:.0f} s', flush=True)
    res['summary'] = {}
    for c in conds:
        w = [r['conditions'][c]['within10'] for r in res['runs'].values()]; nn = [r['conditions'][c]['n'] for r in res['runs'].values()]
        res['summary'][c] = dict(per_run_within10=w, mean_within10=round(float(np.mean(w)), 2), mean_acc10=round(sum(w) / sum(nn), 4))
        print(f"{c:>7}: within10 per run {w}, mean {np.mean(w):.1f} of {nn[0]} ({100 * sum(w) / sum(nn):.2f}%)")
    res['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    a.out.write_text(json.dumps(res))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
