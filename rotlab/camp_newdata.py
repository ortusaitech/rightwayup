#!/usr/bin/env python3
"""Campaign 2026-09-27: NEW evaluation data, built before any tuning. EVALUATION ONLY (never trained on).

  coco: COCO val2017 (5,000) minus the 115 origin-benchmark ids. Unseen by the previous SOTA (trained on train2017 only)
        and by us (our COCO packs are train2017). Licence recorded per image (evaluation use only; pixels not redistributed).
  oi:   Open Images validation, fixed hash order ranks 8000.. (ranks 0-3999 = old photo test, 4000-7999 = old calibration),
        Rotation == 0.0, listed CC BY 2.0, no EXIF orientation flag, re-encoded at max side 448 (same as the old test).
Screening for both: EXIF orientation flag -> excluded; dHash <= 6 bits of ANY training-pack row (all packs, incl. rows
already excluded from training), any origin original, any old OI validation (test/calibration) image, or an earlier
image of the same new set -> excluded. Split by md5('rwu27-split:' + id): first half -> 'val' (selection, calibration,
router training), second half -> 'holdout' (SEALED: views are only materialised with --i-am-final, once, at the end).
Per-image seeds (independent of set composition): theta ~ U[0,360) from Random('rwu27:<set>:<id>').
Views: fixed 4:3 crop (diagonal = min(W,H)-4; angle-independent), origin-protocol max-area crop (COCO only; reported
separately, leaky by construction), and fixed + forced CCTV degradation (core.degrade, Random('rwu27:deg:<id>')).

  python -m rotlab.camp_newdata build               # COCO then OI, one deduper; manifests DATA/rotlab/newdata/*.jsonl
  python -m rotlab.camp_newdata views val           # DATA/rotlab/newdata/views/new{coco,oi}_val_*.pkl
  python -m rotlab.camp_newdata views holdout --i-am-final
"""
import concurrent.futures as cf, csv, hashlib, io, json, random, sys, urllib.request, zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from rotlab.core import DATA, degrade, largest_rotated_rect, render
from rotlab.prep_sources import dhash, encode

ND = DATA / 'rotlab/newdata'
RAW = DATA / 'raw/newdata'
HERE = Path(__file__).parent
COCO_IMG = 'http://images.cocodataset.org/zips/val2017.zip'
COCO_ANN = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
OI_CSV = 'https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv'
OI_MIRROR = 'https://open-images-dataset.s3.amazonaws.com/validation/{}.jpg'
OI_LIC = 'https://creativecommons.org/licenses/by/2.0/'


def fetch(url, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        tmp = dst.with_suffix(dst.suffix + '.part')
        urllib.request.urlretrieve(url, tmp); tmp.replace(dst)
    return dst


def split_of(i):
    return 'val' if int(hashlib.md5(f'rwu27-split:{i}'.encode()).hexdigest()[:8], 16) < 0x80000000 else 'holdout'


# Prior-data references required for a clean eligibility claim (fail closed if any is missing or altered).
REQUIRED_PACKS = {   # name: (jsonl sha256 from release/manifests/SOURCES-SHA256.txt, rows)
    'coco': ('0d8815b896142039975a191602907871f6b23f87eaea036715303603e3c6e1ce', 4096),
    'coco2': ('742c80aba57a835f8ac6ad8dbd0159f09793925fb0956ff18b1d2104abdd67ac', 9683),
    'diode2': ('c137cdba74daa54a0ab64367630d6b5f145729ca2094f45e6fa70eb8e4d5ce73', 2837),
    'hybrid': ('018d07b2aca54739c931845c6dc3677076b08e3d0462ef8f00b7fe3b7ff43c41', 23249),
    'meva2': ('6dc481a9726418f2985cdc4f963f168098a1640a1a04917cd244f7adde3ca6b0', 550),
    'oi': ('1fc4cdf4156eaf277c671a9c644e24e443845cb76c782d198e2695a0875f5319', 33275),
    'pass': ('6ebe7d497588a78e11fe1364bc5ff39fd1e1dfee5291740d9ee710ec8a01447e', 300000),
    'pass2': ('bdfa2444e8ca0d12a47437f626d1830b4b4c1d82e2c42599cf365934c007b870', 300000)}
OLD_OI_RECORDS, OLD_OI_ELIGIBLE = 8000, 6691   # OI validation ranks 0-7999 (old photo test + old calibration)


class Deduper:
    """ONE instance for all new corpora (COCO first, then OI), so a near-duplicate can never sit in two new splits.
    References: every row of the 8 training packs (incl. rows excluded from training), the 1,030 origin originals (= fair
    photo originals), all 6,691 old OI validation test/calibration images, RotBench upright originals. Not covered
    (not natural photos; recorded as coverage limits): MEVA, DIODE and Poly Haven evaluation frames."""

    def __init__(self):
        import pickle
        refs, lab = [], []
        for name, (sha, n) in REQUIRED_PACKS.items():
            f = DATA / f'rotlab/sources/{name}.jsonl'
            if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != sha:
                raise SystemExit(f'FAIL CLOSED: training manifest {name}.jsonl missing or hash mismatch')
            rows = [json.loads(l) for l in f.open()]
            if len(rows) != n:
                raise SystemExit(f'FAIL CLOSED: {name}.jsonl has {len(rows)} rows, expected {n}')
            refs += [int(r['dhash'], 16) for r in rows]; lab += [f'pack:{name}:{r["id"]}' for r in rows]
        og = json.loads((ND / 'origin-dhash.json').read_text()); assert len(og) == 1030
        ids = json.loads((HERE / 'origin_val2017_ids.json').read_text())['origin_ids']
        refs += [int(h, 16) for h in og]; lab += [f'origin:{i}' for i in ids]
        recs = [json.loads(p.read_text()) for p in (DATA / 'rotlab/openimages/validation/records').glob('*.json')]
        el = [r for r in recs if r.get('status') == 'eligible' and 'dhash' in r]
        if len(recs) != OLD_OI_RECORDS or len(el) != OLD_OI_ELIGIBLE:
            raise SystemExit(f'FAIL CLOSED: old OI validation inventory {len(recs)}/{len(el)}, expected {OLD_OI_RECORDS}/{OLD_OI_ELIGIBLE}')
        refs += [int(r['dhash'], 16) for r in el]; lab += [f'old_oi_val:{r["id"]}' for r in el]
        rb = DATA / 'rotlab/cviews/rotbench_full.pkl'
        if not rb.exists():
            raise SystemExit('FAIL CLOSED: rotbench_full.pkl (RotBench originals) missing')
        d = pickle.loads(rb.read_bytes()); th = np.asarray(d['theta'])
        up = [i for i in range(len(th)) if abs(((th[i] + 180) % 360) - 180) < 1e-6]
        refs += [int(dhash(Image.open(io.BytesIO(d['png'][i])).convert('RGB')), 16) for i in up]; lab += [f'rotbench:{i}' for i in up]
        self.ref = np.array(refs, dtype=np.uint64); self.lab = lab; self.own, self.own_lab = [], []
        self.coverage = dict(pack_rows=sum(n for _, n in REQUIRED_PACKS.values()), origin=1030, old_oi_val=len(el), rotbench_upright=len(up),
                             not_covered='MEVA/DIODE/Poly Haven evaluation frames (not natural photos)')
        print(json.dumps(dict(dedup_refs=len(refs), **self.coverage)), flush=True)

    def check(self, h, my_id):
        """-> (status or None, receipt dict or None)."""
        x = np.uint64(int(h, 16))
        d = np.bitwise_count(self.ref ^ x)
        j = int(d.argmin())
        if d[j] <= 6:
            return 'near_duplicate_of_prior_data', dict(match=self.lab[j], hamming=int(d[j]))
        if self.own:
            d2 = np.bitwise_count(np.array(self.own, dtype=np.uint64) ^ x); k = int(d2.argmin())
            if d2[k] <= 6:
                return 'near_duplicate_within_new_sets', dict(match=self.own_lab[k], hamming=int(d2[k]))
        self.own.append(int(h, 16)); self.own_lab.append(my_id)
        return None, None


def origin_dhash():
    """dHash of the 1,030 origin originals (COCO 2014 val ids; 115 in val2017, the rest in train2017)."""
    f = ND / 'origin-dhash.json'
    if f.exists():
        return
    ids = json.loads((HERE / 'origin_val2017_ids.json').read_text())
    v17 = set(ids['in_val2017'])

    def one(i):
        sub = 'val2017' if i in v17 else 'train2017'
        with urllib.request.urlopen(f'http://images.cocodataset.org/{sub}/{i:012d}.jpg', timeout=60) as r:
            return dhash(ImageOps.exif_transpose(Image.open(io.BytesIO(r.read()))).convert('RGB'))
    with cf.ThreadPoolExecutor(16) as ex:
        hs = list(ex.map(one, ids['origin_ids']))
    assert len(hs) == 1030
    f.write_text(json.dumps(hs))


def screen(raw):
    im = Image.open(io.BytesIO(raw))
    if im.getexif().get(0x0112, 1) != 1:
        return None, 'excluded_exif_orientation'
    return ImageOps.exif_transpose(im).convert('RGB'), None


def cmd_coco(dd):
    z = zipfile.ZipFile(fetch(COCO_IMG, RAW / 'val2017.zip'))
    ann = json.loads(zipfile.ZipFile(fetch(COCO_ANN, RAW / 'annotations_trainval2017.zip')).read('annotations/instances_val2017.json'))
    lic = {l['id']: l for l in ann['licenses']}
    origin = set(json.loads((HERE / 'origin_val2017_ids.json').read_text())['origin_ids'])
    out = []
    for img in sorted(ann['images'], key=lambda r: r['id']):
        raw = z.read(f'val2017/{img["file_name"]}')
        rec = dict(id=f'coco_val2017:{img["id"]:012d}', coco_id=img['id'], file=img['file_name'], sha256=hashlib.sha256(raw).hexdigest(),
                   licence_id=img['license'], licence=lic[img['license']]['name'], flickr_url=img.get('flickr_url'), w=img['width'], h=img['height'])
        if img['id'] in origin:
            rec['status'] = 'excluded_origin_benchmark_id'
        else:
            im, why = screen(raw)
            if why:
                rec['status'] = why
            else:
                rec['dhash'] = dhash(im); why, rc = dd.check(rec['dhash'], rec['id'])
                rec['status'] = why or 'eligible'
                if rc:
                    rec['dup_receipt'] = rc
        rec['split'] = split_of(rec['id']) if rec['status'] == 'eligible' else None
        out.append(rec)
    write_manifest('coco_val2017', out)


def cmd_oi(dd, n=6000):
    rows = [r for r in csv.DictReader(open(fetch(OI_CSV, RAW / 'validation-images-with-rotation.csv')))
            if r['Rotation'] == '0.0' and r['License'] == OI_LIC]
    rows.sort(key=lambda r: hashlib.md5(f'oi:{r["ImageID"]}'.encode()).hexdigest())   # same fixed order as rotlab.openimages
    rows = rows[8000:8000 + n]
    (ND / 'oi-img').mkdir(parents=True, exist_ok=True)

    def get(r):
        f = ND / 'oi-img' / f'{r["ImageID"]}.orig'
        if not f.exists():
            try:
                with urllib.request.urlopen(OI_MIRROR.format(r['ImageID']), timeout=60) as u:
                    f.write_bytes(u.read())
            except Exception as e:
                return r, None, f'unresolved_{type(e).__name__}'
        return r, f.read_bytes(), None
    out = []
    with cf.ThreadPoolExecutor(32) as ex:
        for r, raw, err in ex.map(get, rows):
            rec = dict(id=f'oi_validation:{r["ImageID"]}', oi_id=r['ImageID'], landing=r['OriginalLandingURL'], licence='CC BY 2.0 (Open Images listing)',
                       author=r.get('Author'), rank_from=8000)
            if err:
                rec['status'] = err
            else:
                rec['sha256'] = hashlib.sha256(raw).hexdigest()
                im, why = screen(raw)
                if why:
                    rec['status'] = why
                else:
                    body, (w, h), dh, orig = encode(raw, 448)
                    (ND / 'oi-img' / f'{r["ImageID"]}.jpg').write_bytes(body)
                    rec.update(dhash=dh, w=w, h=h, orig_w=orig[0], orig_h=orig[1]); why, rc = dd.check(dh, rec['id'])
                    rec['status'] = why or 'eligible'
                    if rc:
                        rec['dup_receipt'] = rc
            rec['split'] = split_of(rec['id']) if rec['status'] == 'eligible' else None
            out.append(rec)
    write_manifest('oi_validation_r8000', out)


def write_manifest(name, out):
    f = ND / f'{name}.jsonl'
    f.write_text(''.join(json.dumps(r) + '\n' for r in out))
    import collections
    s = dict(collections.Counter(r['status'] for r in out)); s['val'] = sum(r['split'] == 'val' for r in out); s['holdout'] = sum(r['split'] == 'holdout' for r in out)
    s['manifest_sha256'] = hashlib.sha256(f.read_bytes()).hexdigest()
    (ND / f'{name}.summary.json').write_text(json.dumps(s, indent=1)); print(name, json.dumps(s), flush=True)


def image_of(rec):
    if rec['id'].startswith('coco'):
        z = image_of.z = getattr(image_of, 'z', None) or zipfile.ZipFile(RAW / 'val2017.zip')
        raw = z.read(f'val2017/{rec["file"]}')
        assert hashlib.sha256(raw).hexdigest() == rec['sha256']
        return ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB')
    raw = (ND / 'oi-img' / f'{rec["oi_id"]}.jpg').read_bytes()
    if 'sha256_encoded' in rec and hashlib.sha256(raw).hexdigest() != rec['sha256_encoded']:
        raise SystemExit(f'{rec["id"]}: encoded JPEG bytes differ from the manifest')
    return Image.open(io.BytesIO(raw)).convert('RGB')


FROZEN = DATA / 'rotlab/camp-frozen/FROZEN.json'
OPENED = ND / 'HOLDOUT-OPENED.json'


def open_holdout():
    """Final views require the camp_final gate (FROZEN.json + immutable HOLDOUT-OPENED.json bound to it)."""
    from rotlab.camp_final import require_opened
    require_opened()


def cmd_views(split, final=False):
    assert split == 'val' or (split == 'holdout' and final), 'holdout views only with --i-am-final'
    if split == 'holdout':
        open_holdout()
    (ND / 'views').mkdir(parents=True, exist_ok=True)
    for name, short, policies in (('coco_val2017', 'newcoco', ('fixed', 'maxarea', 'fixed_deg')), ('oi_validation_r8000', 'newoi', ('fixed', 'fixed_deg'))):
        if not (ND / f'{name}.jsonl').exists():
            if split == 'holdout':
                raise SystemExit(f'refusing: frozen corpus manifest {name}.jsonl missing')
            continue
        recs = [json.loads(l) for l in open(ND / f'{name}.jsonl')]
        recs = [r for r in recs if r['split'] == split]
        acc = {p: dict(png=[], theta=[], ids=[]) for p in policies}
        for r in recs:
            src = image_of(r); W, H = src.size
            t = random.Random(f'rwu27:{name}:{r["id"]}').uniform(0, 360)
            d = min(W, H) - 4; fixed = render(src, t, 0.8 * d, 0.6 * d, round(0.8 * d), round(0.6 * d))
            vs = {'fixed': fixed, 'fixed_deg': degrade(fixed, random.Random(f'rwu27:deg:{r["id"]}'), force=True)}
            if 'maxarea' in policies:
                cw, ch = largest_rotated_rect(W, H, t); cw, ch = cw - 2, ch - 2
                vs['maxarea'] = render(src, t, cw, ch, max(8, round(cw)), max(8, round(ch)))
            for p in policies:
                b = io.BytesIO(); vs[p].save(b, format='PNG')
                acc[p]['png'].append(b.getvalue()); acc[p]['theta'].append(t); acc[p]['ids'].append(r['id'])
        import pickle
        for p, d in acc.items():
            f = ND / 'views' / f'{short}_{split}_{p}.pkl'
            f.write_bytes(pickle.dumps(dict(png=d['png'], theta=np.array(d['theta']), ids=d['ids'])))
            print('views', f.name, len(d['png']), hashlib.sha256(f.read_bytes()).hexdigest()[:16], flush=True)
            if split == 'holdout':
                from rotlab.camp_final import register_views
                register_views(f.stem, f, d['ids'])


if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'build':   # both corpora, one deduper, COCO first then OI
        ND.mkdir(parents=True, exist_ok=True); origin_dhash(); dd = Deduper()
        cmd_coco(dd); cmd_oi(dd)
        (ND / 'dedup-coverage.json').write_text(json.dumps(dd.coverage, indent=1))
    elif c == 'views':
        cmd_views(sys.argv[2], '--i-am-final' in sys.argv)
    elif c == 'record-encoded':   # pre-freeze source bookkeeping: sha256 of the encoded OI JPEG actually rendered
        f = ND / 'oi_validation_r8000.jsonl'; recs = [json.loads(l) for l in open(f)]
        for r in recs:
            if r['status'] == 'eligible':
                r['sha256_encoded'] = hashlib.sha256((ND / 'oi-img' / f'{r["oi_id"]}.jpg').read_bytes()).hexdigest()
        write_manifest('oi_validation_r8000', recs)
