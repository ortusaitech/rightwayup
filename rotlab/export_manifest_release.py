#!/usr/bin/env python3
"""Per-image training manifests for the RightWayUp v1.0 release models ("v2": Max, Fast, Pico, Nano), 23 grid-free families.

Same release row form as rotlab/export_manifest.py (v1, 24 Sep packs): id, family, group, base_roll_cw, weight, dhash,
used_in_training, dataset, dataset_licence, dataset_licence_url, locator{...}, attribution{...}|null, [label_authority],
plus licence_class, derivation, replaces, replacement_reason, stored_sha256. No image bytes.

Inputs (copies of the records on the campaign volume; nothing is fetched from the network):
  SRC/gf_<family>.jsonl[.gz]   the grid-free pack rows (rotation-data/rotlab/sources/, incl. dataexp-oi7/sources/gf_oi7*)
  SRC/exclude.json             rotation-data/rotlab/sources/exclude.json: ids the trainer drops (train_ddp.blobs_for)
  SRC/pass_meta_min.csv.gz     hash, unickname, licensename, datetaken of PASS pass_metadata.csv (Zenodo 6615455)
  V1/*.jsonl.gz                release/manifests/training (v1 release rows), joined by id where a grid-free row kept a
                               v1 row and the v1 source record is not on the volume any more (see JOINS below)

  python3 rotlab/export_manifest_v2.py --src SRC --v1 release/manifests/training --out release/manifests/training-v2

Writes OUT/<family>.jsonl.gz (deterministic gzip), OUT/SUMMARY.json (via OUT/verify.py summarise), OUT/SHA256SUMS,
OUT/export.log.
"""
import argparse, csv, gzip, hashlib, importlib.util, io, json, re, sys, urllib.parse
from pathlib import Path

CC_BY_2 = 'https://creativecommons.org/licenses/by/2.0/'
PASS_DS = ('PASS v3 (Asano et al., 2021)', 'CC BY 4.0', 'https://creativecommons.org/licenses/by/4.0/')
DIODE_DS = ('DIODE 2019 train (Vasiljevic et al.)', 'MIT', 'https://github.com/diode-dataset/diode-devkit/blob/master/LICENSE')
MEVA_DS = ('MEVA KF1 / drops 4-5 (Kitware, IARPA DIVA)', 'CC BY 4.0', 'https://creativecommons.org/licenses/by/4.0/')
POLY_DS = ('ORTUS AI Blender renders of Poly Haven assets', 'CC0 assets; renders by ORTUS AI', 'https://polyhaven.com/license')
COCO_DS = {'train2017': 'COCO 2017 train (Flickr)', 'unlabeled2017': 'COCO 2017 unlabeled (Flickr)'}
OI_DS = {'test': 'Open Images V7 test subset (Flickr)', 'train': 'Open Images V7 train subset (Flickr)'}
CC_DS = 'CommonCatalog CC-BY (common-canvas/commoncatalog-cc-by; YFCC100M Flickr photos)'
OI_URL = 'https://open-images-dataset.s3.amazonaws.com/{}/{}.jpg'
COCO_URL = 'http://images.cocodataset.org/{}/{:012d}.jpg'
MEVA_BUCKET = 'https://mevadata-public-01.s3.amazonaws.com/'
# grid-free family -> v1 release manifest joined by id (rows whose v1 source record is not on the volume any more)
JOINS = {'gf_coco': 'coco', 'gf_coco2': 'coco2', 'gf_oi': 'oi', 'gf_diode': 'hybrid', 'gf_diode2': 'diode2',
         'gf_meva': 'hybrid', 'gf_meva2': 'meva2', 'gf_poly_haven': 'hybrid', 'gf_poly_direct': 'hybrid'}


def load_verify(out):
    spec = importlib.util.spec_from_file_location('verify', out / 'verify.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        p = p.with_name(p.name + '.gz')
    op = gzip.open if p.suffix == '.gz' else open
    with op(p, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def flickr_attr(a, checked=None):
    """Pack-row attribution (Flickr oEmbed record) -> release attribution (same mapping as v1 export_manifest)."""
    return dict(author=a.get('author_name') or a.get('creator'), author_url=a.get('author_url') or a.get('creator_uri'),
                title=a.get('title', ''), source_url=a.get('web_page') or a.get('source_uri'),
                licence=a.get('license'), licence_url=a.get('license_url') or a.get('license_uri'),
                checked_utc=checked or a.get('observed_utc') or a.get('queried_utc'))


def per_image(o, attr):
    """dataset_licence for Flickr-hosted datasets whose licence is per image: the recorded per-image licence."""
    o['dataset_licence'] = f"{attr['licence']} (per image)" if attr and attr.get('licence') else None
    o['dataset_licence_url'] = attr.get('licence_url') if attr else None


def fetched(r):
    """The file the stored grid-free PNG was derived from, as recorded in the pack row."""
    s = r.get('source') if isinstance(r.get('source'), dict) else {}
    url = r.get('src_url') or s.get('source_url') or r.get('source_url')
    sha = r.get('src_sha256') or s.get('source_sha256') or r.get('source_sha256')
    return {k: v for k, v in (('fetched_url', url), ('fetched_sha256', sha)) if v}


class Builder:
    def __init__(self, src, v1dir, verify):
        self.src, self.v1dir, self.verify = Path(src), Path(v1dir), verify
        self.excluded = set(json.load(open(self.src / 'exclude.json'))['ids'])
        self.v1 = {}
        self.pass_meta = None
        self.notes = []

    def v1rows(self, pack):
        if pack not in self.v1:
            self.v1[pack] = {o['id']: o for o in read_jsonl(self.v1dir / f'{pack}.jsonl.gz')}
        return self.v1[pack]

    def pmeta(self):
        if self.pass_meta is None:
            self.pass_meta = {}
            p = self.src / 'pass_meta_min.csv.gz'
            with gzip.open(p, 'rt', newline='', encoding='utf-8') as f:
                for m in csv.DictReader(f):
                    self.pass_meta[m['hash']] = (m['unickname'], m['licensename'])
        return self.pass_meta

    def head(self, r, ds):
        name, lic, url = ds if isinstance(ds, tuple) else (ds, None, None)
        return dict(id=r['id'], family=r['family'], group=r['group'], base_roll_cw=r['base_roll_cw'], weight=r['weight'],
                    dhash=r['dhash'], used_in_training=r['id'] not in self.excluded, dataset=name, dataset_licence=lic,
                    dataset_licence_url=url)

    def tail(self, o, r):
        if r.get('label_authority'):
            o['label_authority'] = r['label_authority']
        lic = self.verify.image_licence(o)
        o['licence_class'] = self.verify.LICENCE_CLASS.get(lic)
        if r.get('licence_class') and r['licence_class'] != o['licence_class']:
            raise SystemExit(f"{r['id']}: pack licence_class {r['licence_class']} != {o['licence_class']} for {lic!r}")
        o['derivation'] = r.get('derivation') or ('replacement' if r.get('tier') == 'gridfix_replacement' else None)
        o['replaces'] = r.get('replacement_for')
        o['replacement_reason'] = r.get('replacement_reason')
        o['stored_sha256'] = r.get('png_sha256') or r.get('encoded_sha256')
        return o

    # ------------------------------------------------------------------ families
    def pass_row(self, r):
        o = self.head(r, PASS_DS); h = r['id'].split(':', 1)[1]
        loc = dict(pass_hash=h, url='https://www.robots.ox.ac.uk/~vgg/data/pass/', **fetched(r))
        if r.get('yfcc_photoid'):
            loc['yfcc_photoid'] = r['yfcc_photoid']
        if not r.get('replacement_for'):
            loc['pass_tar'] = f"PASS.{r['group'].rsplit('-', 1)[1]}.tar"   # as v1: shard of the 24 Sep pack row
        o['locator'] = loc
        m = self.pmeta().get(h)      # geo-coordinates of pass_metadata.csv are deliberately not copied
        if r.get('replacement_for'):
            nick, lic = r.get('pass_author_nickname'), r.get('licence')
            if m and (m[0] != nick or m[1] != 'Attribution License'):
                self.notes.append(f"{r['id']}: pack row author/licence differs from pass_metadata.csv")
        else:
            nick = m and m[0]
            lic = ('CC BY 2.0 (Flickr "Attribution License")' if m and m[1] == 'Attribution License' else None)
        o['attribution'] = nick and lic and dict(author=urllib.parse.unquote_plus(nick), licence=lic, licence_url=CC_BY_2,
                                                 source_url=loc.get('fetched_url'))
        if m is None:
            self.notes.append(f"{r['id']}: no pass_metadata.csv row")
        return o

    def coco_row(self, r):
        split = r.get('split') or 'train2017'
        if r['family'] == 'gf_fresh_coco':
            split = 'unlabeled2017'
        cid = int(r.get('coco_id') or r.get('source_id'))
        o = self.head(r, COCO_DS[split])
        url = r.get('coco_url') or r.get('source_url') or COCO_URL.format(split, cid)
        loc = dict(coco_id=cid, coco_file=f'{split}/{cid:012d}.jpg', url=url)
        if r.get('source_sha256') and (r.get('source_url') or r['family'] == 'gf_fresh_coco'):
            loc['sha256_original'] = r['source_sha256']
        if r.get('flickr_id'):
            loc['flickr_id'] = str(r['flickr_id'])
        loc.update(fetched(r))
        o['locator'] = loc
        if r['family'] == 'gf_coco':           # pack row carries no attribution: v1 release row (SOURCES.json record)
            a1 = self.v1rows('coco')[r['id']]['attribution']
            o['attribution'] = dict(a1)
        else:
            chk = r.get('licence_checked_utc')
            if not chk and r['family'] == 'gf_coco2':   # kept row: v1 release row (coco-attribution/<id>.json queried_utc)
                chk = self.v1rows('coco2')[r['id']]['attribution'].get('checked_utc')
            o['attribution'] = flickr_attr(r['attribution'], chk)
        per_image(o, o['attribution'])
        return o

    def oi_row(self, r):
        iid = r['id'].split(':', 1)[1]
        subset = 'test' if r['id'].startswith('openimages_test:') else 'train'
        o = self.head(r, OI_DS[subset])
        loc = dict(openimages_id=iid, subset=subset, url=r.get('source_url') or OI_URL.format(subset, iid))
        sha = r.get('sha256_original') or r.get('source_sha256')
        if sha:
            loc['sha256_original'] = sha
        if r.get('flickr_id'):
            loc['flickr_id'] = str(r['flickr_id'])
        if isinstance(r.get('listing'), dict) and r['listing'].get('original_url'):
            loc['flickr_original_url'] = r['listing']['original_url']
        loc.update(fetched(r))
        if r.get('oi_rotation') is not None:
            loc['oi_rotation'] = r['oi_rotation']    # Open Images recorded display orientation (not a human check)
        o['locator'] = loc
        chk = r.get('licence_checked_utc')
        if not chk and r['family'] == 'gf_oi':    # kept row: v1 release row (openimages/test/records/<id>.json queried_utc)
            v = self.v1rows('oi')[r['id']]['attribution']
            chk = v.get('checked_utc')
            if (v.get('author'), v.get('licence')) != (r['attribution'].get('author_name'), r['attribution'].get('license')):
                self.notes.append(f"{r['id']}: pack attribution differs from v1 release row")
        o['attribution'] = flickr_attr(r['attribution'], chk)
        per_image(o, o['attribution'])
        return o

    def cc_row(self, r):
        o = self.head(r, CC_DS); s = r['source']
        o['locator'] = dict(hf_dataset=s['dataset'].split(':', 1)[1], dataset_revision=s['dataset_revision'],
                            parquet_path=s['parquet_path'], row_group=s['row_group'], row=s['row'],
                            dataset_row_sha256=s['dataset_row_sha256'], yfcc_key=s.get('yfcc_key'),
                            flickr_id=str(r['flickr_id']), url=s['source_url'], sha256_original=s['source_sha256'],
                            yfcc_licence=s.get('yfcc_licence'))
        o['attribution'] = flickr_attr(r['attribution'], r.get('licence_checked_utc'))
        per_image(o, o['attribution'])
        return o

    def diode_row(self, r):
        o = self.head(r, DIODE_DS); s = r.get('source') or {}
        if s.get('official_archive_path'):
            o['locator'] = dict(diode_file=s['official_archive_path'], official_archive=s['official_archive'],
                                sha256_original=s.get('source_sha256'), url='https://diode-dataset.org/')
        else:   # kept archive row: DIODE file path from the v1 release row with the same id
            v = self.v1rows(JOINS[r['family']])[r['id']]['locator']
            o['locator'] = dict(diode_file=v['diode_file'], url='https://diode-dataset.org/')
            if v.get('sha256_original'):
                o['locator']['sha256_original'] = v['sha256_original']
        if isinstance(r.get('archive'), dict):
            o['locator']['archive_file_sha256'] = r['archive']['file_sha256']
        o['attribution'] = None
        return o

    def meva_row(self, r):
        o = self.head(r, MEVA_DS); s = r.get('source') or {}
        if s.get('object_key'):
            loc = dict(meva_object_key=s['object_key'], url=(s.get('url_base') or MEVA_BUCKET) + s['object_key'])
            for k in ('frame_index', 'fps', 'frame_at_fraction_of_duration', 'ffmpeg_ss'):
                if s.get(k) is not None:
                    loc[k] = s[k]
        else:   # kept archive row / gf_meva2: clip + frame from the v1 release row with the same id
            loc = dict(self.v1rows(JOINS[r['family']])[r['id']]['locator'])
        o['locator'] = loc
        o['attribution'] = None
        return o

    def poly_row(self, r):
        o = self.head(r, POLY_DS)
        v = self.v1rows('hybrid')[r['id']]
        loc = dict(v['locator'])
        if isinstance(r.get('archive'), dict):
            loc['archive_member'] = r['archive']['member']; loc['archive_file_sha256'] = r['archive']['file_sha256']
        o['locator'] = loc
        o['attribution'] = None
        return o

    def rows(self, fam):
        kind = ('pass' if fam.startswith('gf_pass') else 'coco' if 'coco' in fam else 'oi' if '_oi' in fam else
                'cc' if fam.startswith('gf_cc') else 'diode' if 'diode' in fam else 'meva' if 'meva' in fam else 'poly')
        fn = getattr(self, kind + '_row')
        for r in read_jsonl(self.src / f'{fam}.jsonl'):
            if r['family'] != fam:
                raise SystemExit(f'{fam}: row {r["id"]} has family {r["family"]}')
            yield self.tail(fn(r), r)


def write_gz(path, rows):
    """Deterministic gzip (mtime 0, no file name) so SHA256SUMS is reproducible."""
    with open(path, 'wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=9) as g:
        w = io.TextIOWrapper(g, encoding='utf-8', newline='\n')
        for o in rows:
            w.write(json.dumps(o, ensure_ascii=False) + '\n')
        w.flush(); w.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--v1', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--resummarise', action='store_true', help='keep the manifests, rewrite SUMMARY.json and SHA256SUMS only')
    a = ap.parse_args()
    verify = load_verify(a.out)
    b = Builder(a.src, a.v1, verify)
    log = []
    for fam in verify.FAMILIES:
        if a.resummarise:
            n = u = 0
            for o in read_jsonl(a.out / f'{fam}.jsonl.gz'):
                n += 1; u += o['used_in_training']
            log.append(f'{fam} rows {n} used {u} (kept)'); continue
        rows = list(b.rows(fam))
        write_gz(a.out / f'{fam}.jsonl.gz', rows)
        u = sum(o['used_in_training'] for o in rows)
        log.append(f'{fam} rows {len(rows)} used {u}'); print(log[-1], flush=True)
    summary, problems = verify.summarise((f, read_jsonl(a.out / f'{f}.jsonl.gz')) for f in verify.FAMILIES)
    (a.out / 'SUMMARY.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n')
    names = [f'{f}.jsonl.gz' for f in verify.FAMILIES] + ['SUMMARY.json']
    (a.out / 'SHA256SUMS').write_text(''.join(f'{hashlib.sha256((a.out / n).read_bytes()).hexdigest()}  {n}\n' for n in names))
    log += [f'totals {json.dumps(summary["totals"])}', f'duplicates {json.dumps(summary["duplicates"])}',
            f'problems {len(problems)}'] + problems[:50]
    log += [f'notes {len(b.notes)}'] + b.notes[:200]
    (a.out / 'export.log').write_text('\n'.join(log) + '\n')
    print('\n'.join(log[-(len(problems[:50]) + len(b.notes[:200]) + 4):]))


if __name__ == '__main__':
    main()
