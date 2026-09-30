#!/usr/bin/env python3
"""Rebuild a training pack from its public manifest (for anyone without our packed copies).

Each original is fetched from its public source, re-encoded exactly as our packs were (EXIF-normalised, max side 448,
JPEG q92; `rotlab.prep_sources.encode`) and written to $ROTLAB_DATA/rotlab/sources/<pack>.{bin,jsonl}, the format
`rotlab.train` reads. Every rebuilt image's dHash is compared with the manifest; originals that can no longer be fetched
(e.g. deleted Flickr photos) are skipped and listed in <pack>.missing.jsonl.

  python -m rotlab.build_from_manifest manifests/training/coco.jsonl.gz            # COCO images (cocodataset.org)
  python -m rotlab.build_from_manifest manifests/training/oi.jsonl.gz              # Open Images (CVDF S3 mirror)
  python -m rotlab.build_from_manifest manifests/training/pass.jsonl.gz --pass-dir DIR     # PASS.{0..3}.tar, Zenodo 6615455
  python -m rotlab.build_from_manifest manifests/training/diode2.jsonl.gz --diode-dir DIR  # extracted DIODE train.tar
  python -m rotlab.build_from_manifest manifests/training/meva2.jsonl.gz           # MEVA clips (public S3), needs ffmpeg
  python -m rotlab.build_from_manifest manifests/training/hybrid.jsonl.gz --diode-dir DIR --renders-dir DIR
"""
import argparse, collections, concurrent.futures as cf, gzip, json, subprocess, tarfile, tempfile, threading, time
import urllib.error, urllib.parse, urllib.request
from pathlib import Path

from rotlab.core import DATA
from rotlab.prep_sources import encode

CACHE = DATA / 'rotlab/meva-clip-cache'


def http(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'rotlab-rebuild/1'}), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 410):
                return None
        except Exception:
            pass
        time.sleep(2 ** i)
    return None


_LOCKS, _LOCK = {}, threading.Lock()


def meva_clip(key):
    """Download a MEVA clip once into the cache (one lock per clip: several frames often share a clip)."""
    with _LOCK:
        lock = _LOCKS.setdefault(key, threading.Lock())
    with lock:
        p = CACHE / key
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix('.part')
            urllib.request.urlretrieve('https://mevadata-public-01.s3.amazonaws.com/' + urllib.parse.quote(key), tmp)
            tmp.rename(p)
    return p


def meva_frame(loc):
    clip = meva_clip(loc['meva_object_key'])
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / 'f.png'
        if 'frame_index' in loc:
            cmd = ['ffmpeg', '-v', 'error', '-y', '-i', str(clip), '-vf', f"select=eq(n\\,{loc['frame_index']})", '-vsync', '0',
                   '-frames:v', '1', str(png)]
        else:
            if 'frame_at_seconds' in loc:
                t = loc['frame_at_seconds']
            else:
                dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0',
                                            str(clip)], capture_output=True, text=True).stdout.strip() or 0)
                t = f"{dur * loc['frame_at_fraction_of_duration']:.2f}"
            cmd = ['ffmpeg', '-v', 'error', '-y', '-ss', str(t), '-i', str(clip), '-frames:v', '1', str(png)]
        subprocess.run(cmd, capture_output=True)
        return png.read_bytes() if png.exists() else None


def original(row, a):
    loc = row['locator']
    if 'coco_id' in loc:
        return http(loc['url'])
    if 'openimages_id' in loc:
        return http(loc['url'])
    if 'diode_file' in loc:
        p = Path(a.diode_dir) / loc['diode_file']
        return p.read_bytes() if p.exists() else None
    if 'meva_object_key' in loc:
        return meva_frame(loc)
    if 'render_file' in loc:
        p = Path(a.renders_dir) / loc['render_file']
        return p.read_bytes() if p.exists() else None
    raise ValueError(f"no fetcher for {row['id']}")


def pass_originals(rows, pass_dir):
    """Stream the PASS tars once, yielding (row, bytes) for every manifest row in tar order."""
    want = {r['locator']['pass_hash']: r for r in rows}
    for tar in sorted({r['locator']['pass_tar'] for r in rows}):
        with tarfile.open(Path(pass_dir) / tar) as t:
            for m in t:
                h = m.name.rsplit('/', 1)[-1].rsplit('.', 1)[0]
                if m.isfile() and h in want:
                    yield want.pop(h), t.extractfile(m).read()
                    if not want:
                        return
    for r in want.values():
        yield r, None


def bounded(ex, fn, items, window):
    """Order-preserving map with at most `window` tasks in flight (Executor.map would submit everything at once)."""
    q = collections.deque()
    for it in items:
        q.append(ex.submit(fn, it))
        if len(q) >= window:
            yield q.popleft().result()
    while q:
        yield q.popleft().result()


def fetch(args):
    row, a = args
    return row, original(row, a)


def rebuild(pair):
    row, raw = pair
    if raw is None:
        return row, None, None
    body, (w, h), dh, orig = encode(raw, 448)
    out = dict(id=row['id'], family=row['family'], group=row['group'], base_roll_cw=row['base_roll_cw'], weight=row['weight'],
               w=w, h=h, orig_w=orig[0], orig_h=orig[1], dhash=dh)
    return out, body, bin(int(dh, 16) ^ int(row['dhash'], 16)).count('1')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('manifest', type=Path)
    ap.add_argument('--name', help='pack name (default: manifest file stem)')
    ap.add_argument('--out', type=Path, default=DATA / 'rotlab/sources')
    ap.add_argument('--pass-dir'); ap.add_argument('--diode-dir'); ap.add_argument('--renders-dir')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--limit', type=int, help='first N rows only (smoke test)')
    ap.add_argument('--all-rows', action='store_true', help='also rebuild rows dropped by the benchmark exclusion')
    a = ap.parse_args()
    name = a.name or a.manifest.name.split('.')[0]
    rows = [json.loads(l) for l in gzip.open(a.manifest, 'rt')]
    rows = [r for r in rows if a.all_rows or r['used_in_training']][:a.limit]
    a.out.mkdir(parents=True, exist_ok=True)

    if rows and 'pass_hash' in rows[0]['locator']:
        if not a.pass_dir:
            raise SystemExit('--pass-dir is required for PASS manifests')
        pairs = pass_originals(rows, a.pass_dir)
    else:
        pairs = bounded(cf.ThreadPoolExecutor(a.workers), fetch, ((r, a) for r in rows), 4 * a.workers)

    n = off = missing = 0; dist = {}
    with cf.ProcessPoolExecutor(a.workers) as pool, (a.out / f'{name}.bin.partial').open('wb') as fb, \
            (a.out / f'{name}.jsonl.partial').open('w') as fi, (a.out / f'{name}.missing.jsonl').open('w') as fm:
        for out, body, d in bounded(pool, rebuild, pairs, 4 * a.workers):
            if body is None:
                missing += 1; fm.write(json.dumps(dict(id=out['id'], locator=out['locator'])) + '\n'); continue
            fb.write(body); fi.write(json.dumps(dict(out, offset=off, length=len(body))) + '\n'); off += len(body); n += 1
            dist[min(d, 7)] = dist.get(min(d, 7), 0) + 1
    (a.out / f'{name}.bin.partial').rename(a.out / f'{name}.bin'); (a.out / f'{name}.jsonl.partial').rename(a.out / f'{name}.jsonl')
    same = sum(v for k, v in dist.items() if k <= 2)
    print(json.dumps(dict(pack=name, rebuilt=n, missing=missing, bytes=off, dhash_within_2_bits=same,
                          dhash_distance_histogram={('7+' if k == 7 else k): v for k, v in sorted(dist.items())})))


if __name__ == '__main__':
    main()
