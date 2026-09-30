#!/usr/bin/env python3
"""Campaign 2026-09-27: previous SOTA (Woehrer 2026) on view caches with the CANONICAL adapter preprocessing
(rotlab/fair_photo.py incumbent(): timm eval transform crop_pct 1.0, prediction = -argmax, clockwise chart).
Model: release-onnx/incumbent/rotation_estimator_fp32.onnx (+ .data). Validity check: --check-origin REFERENCES.jsonl
reproduces the archived 'faithful' predictions on the origin panel.
  python -m rotlab.camp_incumbent --sets origin,newval [--check-origin PACK/REFERENCES.jsonl]
Writes DATA/rotlab/camp-eval/incumbent.npz (+ .json receipt with model and view hashes); merges per set.
"""
import argparse, json
import numpy as np
import onnxruntime as ort
import timm.data

from rotlab.camp_eval import DEV, NEWVAL, OUT, load_views, sha256_file, view_file
from rotlab.core import DATA, circ_err

MODEL = DATA / 'rotlab/release-onnx/incumbent/rotation_estimator_fp32.onnx'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--sets', default='origin'); ap.add_argument('--check-origin'); ap.add_argument('--threads', type=int, default=16)
    a = ap.parse_args()
    names = []
    for s in a.sets.split(','):
        names += DEV if s == 'dev' else NEWVAL if s == 'newval' else [s]
    missing = [n for n in names if not view_file(n).exists()]
    if missing:
        raise SystemExit(f'missing view sets {missing}')
    from rotlab.camp_final import check_comparator, is_final
    if any(is_final(n) for n in names):
        check_comparator()       # opened + approved + comparator hashes equal the frozen identity
        from rotlab.camp_final import verify_views
        for n in names:
            if is_final(n):
                verify_views(n, view_file(n))
    so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
    s = ort.InferenceSession(str(MODEL), so, providers=['CPUExecutionProvider']); inp = s.get_inputs()[0].name
    tf = timm.data.create_transform(input_size=(3, 224, 224), interpolation='bicubic', crop_pct=1.0,
                                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_training=False)
    rows, rec = {}, dict(sets={})
    for n in names:
        ims, th = load_views(n)
        pred = np.array([(-int(np.argmax(s.run(None, {inp: tf(im).unsqueeze(0).numpy()})[0][0]))) % 360 for im in ims], np.float64)
        rows[f'{n}_pred'] = pred; rows[f'{n}_theta'] = th; rec['sets'][n] = sha256_file(view_file(n))
        print(n, int((circ_err(pred, th) <= 10).sum()), '/', len(th), flush=True)
        if n == 'origin' and a.check_origin:
            ref = [json.loads(l) for l in open(a.check_origin)]
            fa = np.array([r['arms']['faithful']['prediction_degrees'] for r in ref if r['panel'] == 'origin'], np.float64)
            agree = int((circ_err(pred, fa) <= 0.5).sum()); rec['origin_reproduction'] = dict(agree=agree, n=len(fa))
            print('origin faithful reproduction', agree, '/', len(fa), flush=True)
    from rotlab.camp_store import commit
    ident = dict(model=str(MODEL), model_sha256=sha256_file(str(MODEL)), data_sha256=sha256_file(str(MODEL) + '.data'),
                 adapter='timm create_transform(224, bicubic, crop_pct=1.0); pred = -argmax (clockwise)')
    r = commit(OUT, 'incumbent', rows, ident, rec['sets'])
    if 'origin_reproduction' in rec:
        print('origin_reproduction', rec['origin_reproduction'], flush=True)


if __name__ == '__main__':
    main()
