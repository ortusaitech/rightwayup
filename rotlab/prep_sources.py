#!/usr/bin/env python3
"""Pack training sources into compact blobs of resized JPEGs for fast random access.

Each family becomes <out>/<name>.bin (concatenated JPEG bytes) + <name>.jsonl (one row per
image: offset, length, w, h, base_roll_cw, family, group, id, dhash). Pixels are only
resized (long side <= --max-side) and EXIF-normalised; the upright/base-roll label is
carried as metadata, never baked in.

Families:
  hybrid  v3 TRAIN supported parents (DIODE/MEVA/Poly/direct-v2), base roll from metadata
  coco    CC-BY-2.0 COCO train2017 expansion cache (4,096; attribution-checked 10 Sep)
  pass    PASS v3 tar shards (CC-BY, no people); weak 'as-published is upright' label
"""
from __future__ import annotations
import argparse, io, json, os, tarfile
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageOps

DATA = Path(os.environ.get('ROTLAB_DATA', '/workspace/rotation-data'))   # data root; override with ROTLAB_DATA
V3 = DATA / 'archives/ortus-rotation-hybrid-screen-v3-2026-08-27/metadata/parents-materialized.jsonl'
COCO = DATA / 'training/coco-ccby-source-cache-expansion-20260910'
PASS_DIR = DATA / 'raw/pass-v3'
MAX_SIDE = 448


def dhash(im: Image.Image) -> str:
    g = im.convert('L').resize((9, 8), Image.BILINEAR)
    px = list(g.getdata())
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | (px[r * 9 + c] > px[r * 9 + c + 1])
    return f'{bits:016x}'


def encode(raw: bytes, max_side: int):
    im = Image.open(io.BytesIO(raw))
    im = ImageOps.exif_transpose(im).convert('RGB')
    w, h = im.size
    s = min(1.0, max_side / max(w, h))
    if s < 1:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, format='JPEG', quality=92, subsampling=0)
    return out.getvalue(), im.size, dhash(im), (w, h)


def work(item):
    meta, src = item
    try:
        raw = src if isinstance(src, (bytes, bytearray)) else Path(src).read_bytes()
        body, (w, h), dh, orig = encode(raw, MAX_SIDE)
        return meta | dict(w=w, h=h, orig_w=orig[0], orig_h=orig[1], dhash=dh), body
    except Exception as e:  # corrupt source: record and skip
        return meta | dict(error=f'{type(e).__name__}: {e}'[:200]), None


def hybrid_items():
    for line in V3.open():
        p = json.loads(line)
        if p['split'] != 'train' or p['orientation_state'] != 'supported':
            continue
        yield dict(id=p['parent_id'], family=p['source_family'], group=p['parent_group'],
                   base_roll_cw=float(p['base_orientation_degrees']) % 360,
                   weight=float(p.get('angle_loss_weight', 1.0))), p['source_path']


def coco_items():
    rows = json.load((COCO / 'SOURCES.json').open())['rows']
    for r in rows:
        yield dict(id=r['parent_id'], family='coco', group='coco_train2017_ccby', base_roll_cw=0.0,
                   weight=1.0, coco_id=r['coco_id']), str(COCO / r['original_encoded']['path'])


def pass_items(shards, every, offset=0):
    for s in shards:
        with tarfile.open(PASS_DIR / f'PASS.{s}.tar') as t:
            for i, m in enumerate(t):
                if not m.isfile() or not m.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                if (i - offset) % every:
                    continue
                yield dict(id='pass:' + m.name.split('/')[-1].rsplit('.', 1)[0], family='pass',
                           group=f'pass-shard-{s}', base_roll_cw=0.0, weight=1.0), t.extractfile(m).read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('family', choices=['hybrid', 'coco', 'pass'])
    ap.add_argument('--out', type=Path, default=DATA / 'rotlab/sources')
    ap.add_argument('--name')
    ap.add_argument('--shards', default='0,1')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--offset', type=int, default=0, help='with --every: which residue to keep (pass2 = the other half)')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    name = a.name or a.family
    items = {'hybrid': hybrid_items, 'coco': coco_items,
             'pass': lambda: pass_items([int(x) for x in a.shards.split(',')], a.every, a.offset)}[a.family]()
    binp, idxp = a.out / f'{name}.bin.partial', a.out / f'{name}.jsonl.partial'
    n = bad = off = 0
    with binp.open('wb') as fb, idxp.open('w') as fi, Pool(a.workers) as pool:
        for meta, body in pool.imap(work, items, chunksize=16):
            if body is None:
                bad += 1
                continue
            fb.write(body)
            fi.write(json.dumps(meta | dict(offset=off, length=len(body))) + '\n')
            off += len(body); n += 1
            if n % 5000 == 0:
                print(json.dumps(dict(family=name, done=n, bad=bad, gb=round(off / 1e9, 2))), flush=True)
    binp.rename(a.out / f'{name}.bin'); idxp.rename(a.out / f'{name}.jsonl')
    print(json.dumps(dict(family=name, done=n, bad=bad, bytes=off)))


if __name__ == '__main__':
    main()
