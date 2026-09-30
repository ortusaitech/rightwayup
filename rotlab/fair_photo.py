#!/usr/bin/env python3
"""Fair natural-photo comparison: the 1,030 incumbent-benchmark COCO originals re-rendered at
seeded random angles with an ANGLE-INDEPENDENT crop (fixed 4:3 rectangle whose diagonal equals
min(W,H), so it fits inside the inscribed circle at every angle: crop shape and scale carry no
angle information). Scores the faithful incumbent (portable ORT, upstream timm preprocessing)
and a rotlab checkpoint on identical pixels.

Step 0 validates the incumbent pipeline by reproducing the archived faithful predictions on the
original benchmark views.
  python -m rotlab.fair_photo CKPT [--threads 6] [--seed 7]
"""
import argparse, glob, json, math, random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
import timm.data

from rotlab.core import BUCKETS, DATA, bucket_of, circ_err, decode, letterbox, load_pack, metrics, render
from rotlab.evaluate import confidence, predict
from rotlab.model import RotNet

INC = DATA / 'research-comparators/incumbent/windows-portable/rotation_estimator_fp32.ort'
ORIG = DATA / 'acquisitions/e3cn-exact-incumbent-coco2014-test/test_coco2014'
OUT = DATA / 'rotlab/fair-photo'


def incumbent(threads):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    s = ort.InferenceSession(str(INC), so, providers=['CPUExecutionProvider'])
    tf = timm.data.create_transform(input_size=(3, 224, 224), interpolation='bicubic', crop_pct=1.0,
                                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_training=False)
    name = s.get_inputs()[0].name

    def run(im):  # returns prediction in the pack chart (clockwise content rotation)
        x = tf(im.convert('RGB')).unsqueeze(0).numpy()
        applied_ccw = int(np.argmax(s.run(None, {name: x})[0]))
        return (-applied_ccw) % 360
    return run


def fair_views(seed=7):
    files = sorted(glob.glob(str(ORIG / '**/*.jpg'), recursive=True))
    rng = random.Random(seed); views, thetas = [], []
    for f in files:
        src = Image.open(f).convert('RGB'); W, H = src.size
        d = min(W, H) - 4; cw, ch = 0.8 * d, 0.6 * d
        th = rng.uniform(0, 360)
        views.append(render(src, th, cw, ch, round(cw), round(ch))); thetas.append(th)
    return files, views, np.array(thetas)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('ckpt'); ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--seed', type=int, default=7); ap.add_argument('--skip-validate', action='store_true')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    inc = incumbent(a.threads)
    rows = [r for r in load_pack() if r['panel'] == 'origin']
    if not a.skip_validate:
        p = np.array([inc(Image.open(r['image']['path'])) for r in rows[:100]])
        ref = np.array([r['faithful'] for r in rows[:100]])
        print('incumbent reproduction on first 100 origin rows: exact', int((p == ref).sum()), 'within1', int((circ_err(p, ref) <= 1).sum()), flush=True)
    files = sorted(glob.glob(str(ORIG / '**/*.jpg'), recursive=True))
    rng = random.Random(a.seed)
    views, thetas = [], []
    for f in files:
        src = Image.open(f).convert('RGB'); W, H = src.size
        d = min(W, H) - 4; cw, ch = 0.8 * d, 0.6 * d            # 4:3, diagonal = d
        th = rng.uniform(0, 360)
        views.append(render(src, th, cw, ch, round(cw), round(ch))); thetas.append(th)
    thetas = np.array(thetas)
    views[0].save(OUT / 'example-view.jpg')
    pi = np.array([inc(v) for v in views]); ei = circ_err(pi, thetas)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    cv = ck['config'].get('canvas', 224)
    m = RotNet(img_size=224, pretrained=False, dynamic=ck['config'].get('buckets', False) or cv != 224, arch=ck['config'].get('arch', 's')).cuda().eval()
    m.load_state_dict(ck['ema'])
    res = dict(n=len(views), crop='fixed 4:3, diagonal=min(W,H), angle-independent', seed=a.seed,
               incumbent=metrics(ei), ckpt=a.ckpt)
    buckets = ck['config'].get('buckets', False)
    canvas = BUCKETS[bucket_of(*views[0].size)] if buckets else cv   # all fair views are 4:3
    x = np.stack([letterbox(v, canvas) for v in views])
    rows_out = {}
    for tta, key in ((False, 'model'), (True, 'model_tta'), ('c4', 'model_c4')):
        if tta == 'c4' and buckets:
            continue
        prob = predict(m, x, tta180=tta); pm = decode(prob); res[key] = metrics(circ_err(pm, thetas))
        rows_out[key] = dict(pred=pm, conf=confidence(prob, pm))
    # also report the val2017 subset (images the incumbent certainly did not train on)
    v17 = set(json.loads((Path(__file__).with_name('origin_val2017_ids.json')).read_text())['in_val2017'])
    ids = np.array([int(Path(f).stem.split('_')[-1]) for f in files]); mk = np.isin(ids, list(v17))
    pm = decode(predict(m, x[mk])); res['val2017_subset'] = dict(n=int(mk.sum()), incumbent_w10=int((ei[mk] <= 10).sum()), model_w10=int((circ_err(pm, thetas[mk]) <= 10).sum()))
    np.savez(OUT / f'rows-seed{a.seed}-{Path(a.ckpt).parent.name}.npz', theta=thetas, incumbent=pi,
             **{f'{k}_{f}': v[f] for k, v in rows_out.items() for f in ('pred', 'conf')})
    print(json.dumps(res, indent=1))
    (OUT / f'result-seed{a.seed}-{Path(a.ckpt).parent.name}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
