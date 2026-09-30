#!/usr/bin/env python3
"""Package the Blender renders of Poly Haven assets used in training (the `hybrid` pack) for publication.

Output (DIR/ortus-polyhaven-orientation-renders-v1/):
  renders/...            the exact training images (paths as in manifests/training/hybrid.jsonl.gz `render_file`)
  renders/....json       render parameters (Blender version, camera matrix, Poly Haven assets + hashes), internal
                         path prefixes stripped
  index.jsonl            one row per render: file, sha256, base_roll_cw, loss weight, supervision, Poly Haven assets
  README.md, LICENSE     CC BY 4.0 (renders by ORTUS AI; Poly Haven source assets are CC0)
Then: tar -cf ortus-polyhaven-orientation-renders-v1.tar -C DIR ortus-polyhaven-orientation-renders-v1

  python -m rotlab.package_renders --out DIR
"""
import argparse, gzip, hashlib, json, re, shutil
from pathlib import Path

from rotlab.core import DATA

NAME = 'ortus-polyhaven-orientation-renders-v1'
PREFIX = re.compile(r'/workspace/rotation-data/(raw/poly-haven-pilot/)?')
DIRECT = DATA / 'experiments/direct-perspective-pilot-v2-2026-08-27/materialized-pose-gated-v2'
README = """# ORTUS AI Poly Haven orientation renders (v1)

These are {n:,} images rendered in Blender by ORTUS AI from [Poly Haven](https://polyhaven.com) assets (CC0), with
known camera roll. They were used, together with public photo and CCTV datasets, to train the ORTUS AI image-orientation
model. With this release, anyone can rebuild the model's full training set from `manifests/training/*.jsonl.gz`
(`python -m rotlab.build_from_manifest manifests/training/hybrid.jsonl.gz --renders-dir <this folder>/renders`).

- `renders/`: the exact training images, byte-for-byte (SHA-256 in `index.jsonl`, matching the training manifest).
- `renders/**/*.json`: render parameters (Blender version, camera matrix, gravity projection, Poly Haven asset names
  and file hashes).
- `index.jsonl`: file, sha256, `base_roll_cw` (clockwise roll of the image content), training loss weight, supervision
  type, and Poly Haven assets.

Licence: renders © 2026 ORTUS AI, released under CC BY 4.0. Source assets: Poly Haven, CC0 (credit appreciated, not
required). Poly Haven asset names are listed per image.
"""


def assets(meta):
    names = set()
    for a in meta.get('asset_files', []):
        m = re.search(r'poly-haven-pilot/([^/]+)/', a['path'])
        if m:
            names.add(m.group(1))
    return sorted(names)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return PREFIX.sub('', o) if isinstance(o, str) else o


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); a = ap.parse_args()
    root = a.out / NAME
    if root.exists():
        shutil.rmtree(root)
    rows = [json.loads(l) for l in gzip.open(DATA / 'rotlab/manifests/hybrid.jsonl.gz', 'rt')]
    rows = [r for r in rows if 'render_file' in r['locator']]
    direct = {}
    for line in open(DIRECT / 'manifest.jsonl'):
        d = json.loads(line); direct[Path(d['archive_path']).name] = d
    index = []
    for r in rows:
        rel = r['locator']['render_file']; src = DATA / rel; dst = root / 'renders' / rel
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
        sha = hashlib.sha256(dst.read_bytes()).hexdigest()
        assert r['locator']['render_sha256'] in (None, sha), rel
        side = src.with_suffix('.json')
        meta = json.loads(side.read_text()) if side.exists() else direct.get(src.name, {})
        names = assets(meta); meta = clean(meta)   # asset names come from the original paths, before stripping
        dst.with_suffix('.json').write_text(json.dumps(meta, indent=1))
        index.append(dict(file='renders/' + rel, sha256=sha, base_roll_cw=r['base_roll_cw'], weight=r['weight'],
                          supervision=r['supervision'], used_in_training=r['used_in_training'],
                          poly_haven_assets=names, blender_version=meta.get('blender_version')))
    with open(root / 'index.jsonl', 'w') as f:
        for x in index:
            f.write(json.dumps(x) + '\n')
    (root / 'README.md').write_text(README.format(n=len(index)))
    shutil.copy2('/usr/share/common-licenses/CC-BY-4.0' if Path('/usr/share/common-licenses/CC-BY-4.0').exists() else DATA / 'rotlab/CC-BY-4.0.txt',
                 root / 'LICENSE')
    left = [str(p) for p in root.rglob('*.json') if '/workspace' in p.read_text()]
    assert not left, left[:3]
    print(json.dumps(dict(renders=len(index), with_assets=sum(bool(x['poly_haven_assets']) for x in index),
                          bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))))


if __name__ == '__main__':
    main()
