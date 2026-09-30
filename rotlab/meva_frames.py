#!/usr/bin/env python3
"""Collect sparse frames from MEVA cameras that no rotlab model has trained on, for a real-CCTV test set.

Cameras with a v1 fixture acceptance (base roll 0) reuse the archived v1 frames; the rest are
extracted from the raw r13 clips with ffmpeg (their level mounting is verified separately by
rotlab.meva_roll + visual review). Output: DATA/rotlab/meva-test-v1/frames/<cam>/<clip>-<t>.jpg
and frames.jsonl.
  python -m rotlab.meva_frames
"""
import collections, glob, json, re, subprocess
from pathlib import Path

from PIL import Image

from rotlab.core import DATA

V1 = DATA / 'archives/ortus-rotation-hybrid-v1-2026-08-27/frozen-dataset'
RAW = DATA / 'raw/meva-kf1'
OUT = DATA / 'rotlab/meva-test-v1'
V1_CAMS = ['G326', 'G329', 'G331', 'G336', 'G341', 'G421', 'G423', 'G508', 'G639']
NEW_CAMS = ['G340', 'G420', 'G474', 'G475', 'G476', 'G479']
# never used here: trained on (G300 G301 G424 G436 G505 G506 G509), eval pack (G328 G638),
# protected calibration/frozen test (G419 G339)
SECONDS = (30, 90, 150, 210, 270)
MAX_SIDE = 1280


def save(im, path):
    im = im.convert('RGB')
    if max(im.size) > MAX_SIDE:
        s = MAX_SIDE / max(im.size); im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True); im.save(path, quality=95)
    return im.size


def main():
    rows = []
    v1 = [json.loads(l) for l in open(V1 / 'metadata/parents-materialized.jsonl')]
    for p in v1:
        cam = p['parent_group'].split(':')[-1]
        if p['source_family'] != 'meva' or cam not in V1_CAMS:
            continue
        clip, f = p['sample_id'].rsplit('-f', 1)
        if int(f) % 10 != 2:                                    # 5 of 50 frames per clip
            continue
        out = OUT / 'frames' / cam / f'{clip}-f{f}.jpg'
        w, h = save(Image.open(V1 / p['archive_parent_path']), out)
        rows.append(dict(cam=cam, clip=clip, frame=f'f{f}', path=str(out), w=w, h=h, source='v1-archive',
                         label_authority=p['label_authority'], base_roll_cw=p['base_orientation_degrees']))
    for cam in NEW_CAMS:
        for clip in sorted(glob.glob(str(RAW / f'**/*.{cam}.r13.avi'), recursive=True)):
            stem = Path(clip).name.replace('.r13.avi', '')
            for t in SECONDS:
                out = OUT / 'frames' / cam / f'{stem}-t{t:03d}.jpg'; out.parent.mkdir(parents=True, exist_ok=True)
                r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(t), '-i', clip, '-frames:v', '1', '-q:v', '2', str(out.with_suffix('.png'))],
                                   capture_output=True) if not out.exists() else None
                png = out.with_suffix('.png')
                if png.exists():
                    w, h = save(Image.open(png), out); png.unlink()
                elif not out.exists():
                    continue
                else:
                    w, h = Image.open(out).size
                rows.append(dict(cam=cam, clip=stem, frame=f't{t:03d}', path=str(out), w=w, h=h, source='raw-r13',
                                 label_authority='pending_rotlab_vertical_vp_plus_visual_review', base_roll_cw=0.0))
    (OUT / 'frames.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    c = collections.Counter(r['cam'] for r in rows)
    print(len(rows), 'frames', dict(sorted(c.items())))


if __name__ == '__main__':
    main()
