#!/usr/bin/env python3
"""Per-image training manifests: one row per packed training image, pointing back to its public original.

Each row: id, family, group, base_roll_cw, weight (per-image loss weight), dhash, used_in_training (False = dropped by the benchmark near-duplicate
exclusion), dataset, dataset_licence, locator (how to fetch the original from the public source), attribution
(author / title / licence per image where the licence needs it), label_authority.

  python -m rotlab.export_manifest [--out DIR] [--pass-meta PASS_METADATA.csv]

Writes DIR/<pack>.jsonl.gz plus DIR/SUMMARY.json (counts per family, dataset and licence).
"""
import argparse, collections, csv, gzip, json, re, urllib.parse
from pathlib import Path

from rotlab.core import DATA

SRC = DATA / 'rotlab/sources'
V3 = DATA / 'archives/ortus-rotation-hybrid-screen-v3-2026-08-27/metadata/parents-materialized.jsonl'
COCO = DATA / 'training/coco-ccby-source-cache-expansion-20260910/SOURCES.json'
DIODE_CAND = DATA / 'experiments/real-corpus-preparation-2026-08-26/diode-zero-roll-candidates-v1.jsonl'
MEVA_DERIVED = DATA / 'derived/meva-kf1/quarantine-previews-v1'
MEVA_BUCKET = 'https://mevadata-public-01.s3.amazonaws.com/'
OI_URL = 'https://open-images-dataset.s3.amazonaws.com/test/{}.jpg'

LIC = {
    'pass': ('PASS v3 (Asano et al., 2021)', 'CC BY 4.0', 'https://creativecommons.org/licenses/by/4.0/'),
    'coco': ('COCO 2017 train (Flickr)', 'CC BY 2.0 (per image)', 'https://creativecommons.org/licenses/by/2.0/'),
    'oi': ('Open Images V7 test subset (Flickr)', 'CC BY 2.0 (per image)', 'https://creativecommons.org/licenses/by/2.0/'),
    'diode': ('DIODE 2019 train (Vasiljevic et al.)', 'MIT', 'https://github.com/diode-dataset/diode-devkit/blob/master/LICENSE'),
    'meva': ('MEVA KF1 / drops 4-5 (Kitware, IARPA DIVA)', 'CC BY 4.0', 'https://creativecommons.org/licenses/by/4.0/'),
    'poly': ('ORTUS AI Blender renders of Poly Haven assets', 'CC0 assets; renders by ORTUS AI', 'https://polyhaven.com/license'),
}


def rows(pack):
    with open(SRC / f'{pack}.jsonl') as f:
        for line in f:
            yield json.loads(line)


def base(r, excluded, key):
    ds, lic, lic_url = LIC[key]
    return dict(id=r['id'], family=r['family'], group=r['group'], base_roll_cw=r['base_roll_cw'], weight=r['weight'], dhash=r['dhash'],
                used_in_training=r['id'] not in excluded, dataset=ds, dataset_licence=lic, dataset_licence_url=lic_url)


def flickr_attr(a, checked=None):
    return dict(author=a.get('author_name') or a.get('creator'), author_url=a.get('author_url') or a.get('creator_uri'),
                title=a.get('title', ''), source_url=a.get('web_page') or a.get('source_uri'),
                licence=a.get('license'), licence_url=a.get('license_url') or a.get('license_uri'),
                checked_utc=a.get('observed_utc') or a.get('queried_utc') or checked)


def check_date(path):
    """When the current-licence check (Flickr oEmbed) ran for this image: the top-level `queried_utc` of its record."""
    try:
        return json.loads(path.read_text()).get('queried_utc')
    except FileNotFoundError:
        return None


_MEVA_KEYS = {}


def meva_keys():
    """basename -> object key in the public MEVA bucket (the prefixes pack_extra.meva2-fetch drew from)."""
    if not _MEVA_KEYS:
        from rotlab.pack_extra import s3_keys
        for pre in ('drops-123-r13/', 'drop-4-hadcv22/', 'drop-5-mevid/'):
            for k in s3_keys(pre):
                if k.endswith('.avi'):
                    _MEVA_KEYS.setdefault(Path(k).name, k)
    return _MEVA_KEYS


def meva_clip_locator(clip_id, frame_no):
    s = json.loads((MEVA_DERIVED / clip_id / 'summary.json').read_text())
    return dict(meva_object_key=s['object_key'], url=MEVA_BUCKET + s['object_key'],
                frame_index=s['selected_frame_indices'][frame_no], frame_rate=s['probe']['avg_frame_rate'])


def export(pack, excluded, pass_meta):
    coco_attr = {}
    if pack == 'coco':
        coco_attr = {r['coco_id']: r['attribution'] for r in json.load(open(COCO))['rows']}
    hybrid = {}
    if pack == 'hybrid':
        for line in open(V3):
            p = json.loads(line)
            hybrid[p['parent_id']] = p
    diode_cand = {}
    if pack == 'diode2':
        diode_cand = {c['sample_id']: c for c in map(json.loads, open(DIODE_CAND))}

    for r in rows(pack):
        fam = r['family']
        if pack in ('pass', 'pass2'):
            o = base(r, excluded, 'pass'); h = r['id'].split(':', 1)[1]
            o['locator'] = dict(pass_hash=h, pass_tar=f"PASS.{r['group'].rsplit('-', 1)[1]}.tar",
                                url='https://www.robots.ox.ac.uk/~vgg/data/pass/')
            m = pass_meta.get(h)   # PASS pass_metadata.csv (Zenodo 6615455); geo-coordinates are deliberately not copied
            o['attribution'] = m and dict(author=urllib.parse.unquote_plus(m['unickname']), licence='CC BY 2.0 (Flickr "Attribution License")',
                                          licence_url='https://creativecommons.org/licenses/by/2.0/',
                                          source_url=f'https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images/{h[:3]}/{h[3:6]}/{h}.jpg')
        elif pack in ('coco', 'coco2'):
            o = base(r, excluded, 'coco')
            o['locator'] = dict(coco_id=r['coco_id'], coco_file=f"train2017/{r['coco_id']:012d}.jpg",
                                url=f"http://images.cocodataset.org/train2017/{r['coco_id']:012d}.jpg")
            o['attribution'] = (flickr_attr(r['attribution'], check_date(DATA / f"rotlab/coco-attribution/{r['coco_id']}.json"))
                                if pack == 'coco2' else flickr_attr(coco_attr[r['coco_id']]))
        elif pack == 'oi':
            o = base(r, excluded, 'oi'); iid = r['id'].split(':', 1)[1]
            o['locator'] = dict(openimages_id=iid, subset='test', url=OI_URL.format(iid), sha256_original=r.get('sha256_original'))
            o['attribution'] = flickr_attr(r['attribution'], check_date(DATA / f'rotlab/openimages/test/records/{iid}.json'))
        elif pack == 'diode2':
            o = base(r, excluded, 'diode'); c = diode_cand[r['id'].split('-', 1)[1]]
            o['locator'] = dict(diode_file=c['source_path'].split('/raw/diode-2019/', 1)[1].split('/', 1)[1],
                                sha256_original=r.get('source_sha256'), url='https://diode-dataset.org/')
            o['attribution'] = None
        elif pack == 'meva2':
            o = base(r, excluded, 'meva'); cam = r['group'].split(':', 1)[1]
            stem, t = re.match(r'meva2-G\d+-(.+)-(t?\d+)$', r['id']).groups()
            ks = meva_keys(); key = ks.get(stem + '.avi') or ks[stem + '.r13.avi']
            o['locator'] = dict(meva_object_key=key, url=MEVA_BUCKET + key)
            if t.startswith('t'):   # thermal frames from rotlab.meva_frames: ffmpeg -ss <seconds>
                o['locator']['frame_at_seconds'] = int(t[1:])
            else:                   # rotlab.pack_extra meva2-fetch: ffmpeg -ss <fraction x duration>
                o['locator']['frame_at_fraction_of_duration'] = int(t) / 100
            o['attribution'] = None
        elif pack == 'hybrid':
            p = hybrid[r['id']]; sp = p['source_path']
            if fam == 'diode':
                o = base(r, excluded, 'diode')
                o['locator'] = dict(diode_file=sp.split('/raw/diode-2019/', 1)[1].split('/', 1)[1], url='https://diode-dataset.org/')
            elif fam == 'meva':
                o = base(r, excluded, 'meva'); clip, fr = re.search(r'(meva-clip-[0-9a-f]+)/frame-(\d+)\.jpg$', sp).groups()
                o['locator'] = meva_clip_locator(clip, int(fr))
            else:
                o = base(r, excluded, 'poly')
                o['locator'] = dict(render_file=sp.split('/rotation-data/', 1)[1], render_sha256=p.get('source_sha256'),
                                    render_set=p['source_id'])
            o['attribution'] = None
            o['supervision'] = p['supervision_mode']
        else:
            raise SystemExit(f'unknown pack {pack}')
        if r.get('label_authority'):
            o['label_authority'] = r['label_authority']
        yield o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=DATA / 'rotlab/manifests')
    ap.add_argument('--pass-meta', type=Path, help='PASS metadata CSV (hash, creator, Flickr URL) to join, if available')
    ap.add_argument('--packs', default='hybrid,coco,coco2,pass,pass2,oi,diode2,meva2')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    excluded = set(json.load(open(SRC / 'exclude.json'))['ids'])
    pass_meta = {}
    if a.pass_meta:
        with open(a.pass_meta, newline='') as f:
            for m in csv.DictReader(f):
                pass_meta[m['hash']] = dict(unickname=m['unickname'], licensename=m['licensename'])
    summary = {}
    for pack in a.packs.split(','):
        c = collections.Counter(); missing_attr = missing_date = 0
        with gzip.open(a.out / f'{pack}.jsonl.gz', 'wt') as f:
            for o in export(pack, excluded, pass_meta):
                f.write(json.dumps(o, ensure_ascii=False) + '\n')
                c[(o['family'], o['dataset_licence'], o['used_in_training'])] += 1
                missing_attr += o['family'] in ('pass', 'coco', 'oi') and not o['attribution']
                missing_date += o['family'] in ('coco', 'oi') and not (o['attribution'] or {}).get('checked_utc')
        summary[pack] = dict(rows=sum(c.values()), missing_flickr_attribution=missing_attr, missing_licence_check_date=missing_date,
                             by=[dict(family=k[0], licence=k[1], used_in_training=k[2], n=n) for k, n in sorted(c.items())])
        print(pack, json.dumps(summary[pack]), flush=True)
    (a.out / 'SUMMARY.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
