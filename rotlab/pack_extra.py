#!/usr/bin/env python3
"""Extra CCTV / DIODE training data (training only; every test/protected group is excluded by allowlist).

diode2: DIODE train scenes 1, 2, 4, 15, 18 (publisher-rectified rasters, base roll 0), using the earlier pipeline's v1
        zero-roll candidates (fixture + deterministic vertical screen). Scene 12 is protected
        frozen test; scenes 3/5/9/11 and validation 19-24 are benchmark panels; the rest are already in hybrid.
meva2:  more clips of the 7 MEVA training cameras (KF1 drops-123-r13, drop-4-hadcv22, drop-5-mevid; CC BY 4.0),
        3 frames per clip, plus new camera G507 which is packed only with --accept G507 after visual review.
        Near-identical frames of one camera (dHash <= 3) are dropped.

  python -m rotlab.pack_extra diode2
  python -m rotlab.pack_extra meva2-fetch [--r13-per-cam 40 --drop45-per-cam 20]
  python -m rotlab.pack_extra meva2-pack [--accept G507]
"""
import argparse, concurrent.futures as cf, hashlib, json, re, subprocess, tempfile, urllib.parse, urllib.request
from pathlib import Path

from PIL import Image

from rotlab.core import DATA
from rotlab.prep_sources import encode

SRC = DATA / 'rotlab/sources'
DIODE_CAND = DATA / 'experiments/real-corpus-preparation-2026-08-26/diode-zero-roll-candidates-v1.jsonl'
DIODE_SCENES = {'indoors:scene_00001', 'indoors:scene_00002', 'indoors:scene_00004', 'outdoor:scene_00015', 'outdoor:scene_00018'}
MEVA_TRAIN = ['G300', 'G301', 'G424', 'G436', 'G505', 'G506', 'G509']
MEVA_NEW = ['G507']
MEVA_THERMAL_TRAIN = ['G475', 'G476']   # moved from test to training (2026-09-23); level verified visually
MEVA_NEVER = {'G326', 'G329', 'G331', 'G336', 'G340', 'G341', 'G420', 'G421', 'G423', 'G474', 'G479',
              'G508', 'G639', 'G328', 'G638', 'G419', 'G339', 'G299', 'G330'}   # test / eval / protected / calibration
BUCKET = 'https://mevadata-public-01.s3.amazonaws.com/'
STAGE = DATA / 'rotlab/meva2-stage'


def write_pack(name, rows):
    with (SRC / f'{name}.bin.partial').open('wb') as fb, (SRC / f'{name}.jsonl.partial').open('w') as fi:
        off = 0
        for r, body in rows:
            fb.write(body); fi.write(json.dumps(dict(r, offset=off, length=len(body))) + '\n'); off += len(body)
    (SRC / f'{name}.bin.partial').rename(SRC / f'{name}.bin'); (SRC / f'{name}.jsonl.partial').rename(SRC / f'{name}.jsonl')
    return off


def cmd_diode2(a):
    cand = [json.loads(l) for l in open(DIODE_CAND)]
    cand = [c for c in cand if f"{c['environment']}:{c['scene_id']}" in DIODE_SCENES and c['orientation_state'] == 'supported'
            and c['base_orientation_degrees'] == 0.0]

    def one(c):
        body, (w, h), dh, orig = encode(Path(c['source_path']).read_bytes(), 448)
        return dict(id=f"diode2-{c['sample_id']}", family='diode2', group=f"diode:{c['environment']}:{c['scene_id']}",
                    base_roll_cw=0.0, weight=1.0, w=w, h=h, orig_w=orig[0], orig_h=orig[1], dhash=dh,
                    source_sha256=c['source_sha256'], label_authority=c['label_authority']), body
    with cf.ThreadPoolExecutor(6) as ex:
        rows = sorted(ex.map(one, cand), key=lambda rb: rb[0]['id'])
    n = write_pack('diode2', rows)
    by = {}
    for r, _ in rows: by[r['group']] = by.get(r['group'], 0) + 1
    print(json.dumps(dict(images=len(rows), bytes=n, scenes=by)))


def s3_keys(prefix):
    keys, tok = [], None
    while True:
        u = BUCKET + '?list-type=2&prefix=' + urllib.parse.quote(prefix) + ('&continuation-token=' + urllib.parse.quote(tok) if tok else '')
        x = urllib.request.urlopen(u, timeout=60).read().decode()
        keys += re.findall(r'<Key>([^<]*)</Key>', x)
        t = re.search(r'<NextContinuationToken>([^<]*)<', x)
        if not t: return keys
        tok = t.group(1)


def cam_of(k):
    m = re.search(r'\.(G\d{3})\.', k); return m.group(1) if m else None


def pick(keys, cams, per_cam, have):
    out = []
    for cam in cams:
        ks = sorted((k for k in keys if cam_of(k) == cam and k.endswith('.avi') and Path(k).name not in have),
                    key=lambda k: hashlib.md5(f'meva2:{k}'.encode()).hexdigest())
        out += ks[:per_cam]
    return out


def frames_of(key):
    cam = cam_of(key); stem = Path(key).name.rsplit('.avi', 1)[0]
    d = STAGE / cam; d.mkdir(parents=True, exist_ok=True)
    if (d / f'{stem}.done').exists():
        return 0
    with tempfile.TemporaryDirectory(dir='/tmp') as td:
        clip = Path(td) / 'c.avi'
        urllib.request.urlretrieve(BUCKET + urllib.parse.quote(key), clip)
        dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(clip)],
                                   capture_output=True, text=True).stdout.strip() or 0)
        n = 0
        for f in (0.2, 0.5, 0.8):
            png = Path(td) / f'{f}.png'
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{dur * f:.2f}', '-i', str(clip), '-frames:v', '1', str(png)], capture_output=True)
            if png.exists():
                Image.open(png).convert('RGB').save(d / f'{stem}-{int(f * 100):02d}.jpg', quality=93); n += 1
    (d / f'{stem}.done').write_text(key)
    return n


def cmd_meva2_fetch(a):
    have = {p.name.replace('.r13', '') for p in (DATA / 'raw/meva-kf1').rglob('*.avi')} | {p.name for p in (DATA / 'raw/meva-kf1').rglob('*.avi')}
    keys = []
    for pre, per in (('drops-123-r13/', a.r13_per_cam), ('drop-4-hadcv22/', a.drop45_per_cam), ('drop-5-mevid/', a.drop45_per_cam)):
        ks = s3_keys(pre); keys += pick(ks, a.cams or (MEVA_TRAIN + MEVA_NEW), per, have)
    assert not any(cam_of(k) in MEVA_NEVER for k in keys)
    print('clips', len(keys), flush=True)
    done = 0
    with cf.ThreadPoolExecutor(a.threads) as ex:
        for i, n in enumerate(ex.map(frames_of, keys)):
            done += n
            if (i + 1) % 50 == 0: print(json.dumps(dict(clips=i + 1, frames=done)), flush=True)
    print(json.dumps(dict(clips=len(keys), frames=done)))


def cmd_meva2_pack(a):
    from rotlab.prep_sources import dhash
    cams = MEVA_TRAIN + MEVA_THERMAL_TRAIN + [c for c in MEVA_NEW if c in (a.accept or [])]
    rows = []
    for cam in cams:
        kept = []
        for p in sorted((STAGE / cam).glob('*.jpg')):
            body, (w, h), dh, orig = encode(p.read_bytes(), 448); hv = int(dh, 16)
            if any(bin(hv ^ k).count('1') <= 3 for k in kept):
                continue
            kept.append(hv)
            rows.append((dict(id=f'meva2-{cam}-{p.stem}', family='meva2', group=f'meva-camera:{cam}', base_roll_cw=0.0, weight=1.0,
                              w=w, h=h, orig_w=orig[0], orig_h=orig[1], dhash=dh,
                              label_authority='meva_camera_zero_roll_' + ('v1_fixture' if cam in MEVA_TRAIN else 'rotlab_visual_plus_vertical_vp')), body))
    n = write_pack('meva2', rows)
    by = {}
    for r, _ in rows: by[r['group']] = by.get(r['group'], 0) + 1
    print(json.dumps(dict(images=len(rows), bytes=n, cameras=by)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd')
    ap.add_argument('--r13-per-cam', type=int, default=40); ap.add_argument('--drop45-per-cam', type=int, default=20)
    ap.add_argument('--threads', type=int, default=4); ap.add_argument('--accept', nargs='*'); ap.add_argument('--cams', nargs='*', help='fetch only these cameras')
    a = ap.parse_args()
    {'diode2': cmd_diode2, 'meva2-fetch': cmd_meva2_fetch, 'meva2-pack': cmd_meva2_pack}[a.cmd](a)
