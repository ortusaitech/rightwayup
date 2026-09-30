#!/usr/bin/env python3
"""Calibration sets (used ONLY to fix routing / abstention thresholds; never for training or model selection).

  meva_cal: MEVA G419 (clean v1 calibration frames, every 10th) + G299 and G330 (exact-KRTD cameras, never
            used; 15 KF1 clips x 3 frames each, level verified visually from the contact sheets)
  poly_cal: the 596 Poly Haven parents of the clean v1 calibration split
  oi_cal:   eligible Open Images validation photos ranked 4000-7999 in the fixed hash order (disjoint from the
            photo test, which is ranks 0-3999)
Each image gives a clean and a CCTV-degraded view at seeded angles, fixed 4:3 crop (as in the tests).
  python -m rotlab.calib fetch          # G299/G330 frames + sheets
  python -m rotlab.calib build CANVAS   # views -> DATA/rotlab/suite/views-cal-c<canvas>.npz
"""
import concurrent.futures as cf, hashlib, json, random, subprocess, sys, tempfile, urllib.parse, urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from rotlab.core import DATA, degrade, letterbox, render
from rotlab.openimages import candidates
from rotlab.pack_extra import BUCKET, cam_of, s3_keys

ROOT = DATA / 'rotlab/calib'
V1 = DATA / 'archives/ortus-rotation-hybrid-v1-2026-08-27/frozen-dataset'
SPLITS = DATA / 'rotlab/clean-v1-splits.json'
KRTD_CAMS = ['G299', 'G330']


def fetch():
    keys = s3_keys('drops-123-r13/')
    todo = []
    for cam in KRTD_CAMS:
        ks = sorted((k for k in keys if cam_of(k) == cam and k.endswith('.avi')), key=lambda k: hashlib.md5(f'cal:{k}'.encode()).hexdigest())
        todo += ks[:15]

    def one(key):
        cam = cam_of(key); d = ROOT / 'frames' / cam; d.mkdir(parents=True, exist_ok=True); stem = Path(key).name[:-4]
        with tempfile.TemporaryDirectory(dir='/tmp') as td:
            clip = Path(td) / 'c.avi'; urllib.request.urlretrieve(BUCKET + urllib.parse.quote(key), clip)
            dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(clip)],
                                       capture_output=True, text=True).stdout.strip() or 0)
            for f in (0.2, 0.5, 0.8):
                png = Path(td) / 'f.png'
                subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{dur * f:.2f}', '-i', str(clip), '-frames:v', '1', str(png)], capture_output=True)
                if png.exists():
                    Image.open(png).convert('RGB').save(d / f'{stem}-{int(f * 100):02d}.jpg', quality=93)
    with cf.ThreadPoolExecutor(4) as ex:
        list(ex.map(one, todo))
    for cam in KRTD_CAMS:
        ps = sorted((ROOT / 'frames' / cam).glob('*.jpg')); sel = ps[::max(1, len(ps) // 12)][:12]
        S = Image.new('RGB', (4 * 320, 3 * 180))
        for k, p in enumerate(sel):
            im = Image.open(p).convert('RGB'); im.thumbnail((320, 180)); S.paste(im, ((k % 4) * 320, (k // 4) * 180))
        S.save(ROOT / f'sheet-{cam}.jpg', quality=85); print(cam, len(ps), 'frames')


def sources():
    cal = set(json.loads(SPLITS.read_text())['calibration'])
    v1 = [json.loads(l) for l in open(V1 / 'metadata/parents-materialized.jsonl')]
    out = {'meva_cal': [], 'poly_cal': [], 'oi_cal': []}
    for p in v1:
        if p['parent_id'] not in cal:
            continue
        if p['source_family'] == 'meva':
            if int(p['sample_id'].rsplit('-f', 1)[1]) % 10 == 2:
                out['meva_cal'].append((V1 / p['archive_parent_path'], float(p['base_orientation_degrees'])))
        else:
            out['poly_cal'].append((V1 / p['archive_parent_path'], float(p['base_orientation_degrees'])))
    for cam in KRTD_CAMS:
        out['meva_cal'] += [(p, 0.0) for p in sorted((ROOT / 'frames' / cam).glob('*.jpg'))]
    recd = DATA / 'rotlab/openimages/validation'
    for r in candidates('validation', 8000)[4000:]:
        f = recd / 'records' / f"{r['ImageID']}.json"
        if f.exists() and json.loads(f.read_text())['status'] == 'eligible':
            out['oi_cal'].append((recd / 'img' / f"{r['ImageID']}.jpg", 0.0))
    return out


def build(canvas):
    arrays = {}
    for name, items in sources().items():
        rng = random.Random(f'cal:{name}'); xs, th = [], []
        for path, base in sorted(items, key=lambda t: str(t[0])):
            src = Image.open(path).convert('RGB'); W, H = src.size; d = min(W, H) - 4; cw, ch = 0.8 * d, 0.6 * d
            for variant in ('clean', 'degraded'):
                t = rng.uniform(0, 360); v = render(src, t, cw, ch, round(cw), round(ch))
                if variant == 'degraded':
                    v = degrade(v, rng)
                xs.append(letterbox(v, canvas)); th.append((t + base) % 360)
        arrays[f'{name}_x'] = np.stack(xs); arrays[f'{name}_t'] = np.array(th); print(name, len(xs), 'views', flush=True)
    np.savez(DATA / f'rotlab/suite/views-cal-c{canvas}.npz', **arrays)


if __name__ == '__main__':
    {'fetch': lambda: fetch(), 'build': lambda: build(int(sys.argv[2]))}[sys.argv[1]]()
