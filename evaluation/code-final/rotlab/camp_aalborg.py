#!/usr/bin/env python3
"""Campaign 2026-09-27: Aalborg Long-term Thermal Drift (LTD) dataset — FINAL-ONLY thermal evaluation panel.

Source: https://vap.aau.dk/ltd/ (CC BY 4.0), files via the public Kaggle dataset API (anonymous GET, no credentials):
ivannikolov/longterm-thermal-drift-dataset, datasetVersionNumber=3. ONE fixed harbour thermal camera: capture dates are
temporal groups, NOT independent cameras (independent camera count = 1). Evaluation only; never trained on.
Selection (source-only, no model outputs): annotated-subset JPEGs (Data_Annotated_Subset_Object_Detectors/*), one per
capture date by md5('rwu27-aalborg:' + name) order, then second images from dates in hash order until N = 150.
Labels: synthetic rotations relative to the stored raster (base roll unvalidated). Views (fixed 4:3, angle-independent)
are materialised only after the model freeze via the holdout gate.
  python -m rotlab.camp_aalborg select       # manifest + downloads (pod) + metadata CSVs + QC sheet
  python -m rotlab.camp_aalborg views --i-am-final
"""
import collections, hashlib, io, json, pickle, random, re, sys, urllib.parse, urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from rotlab.core import DATA, render

RAW = DATA / 'raw/aalborg'
ND = DATA / 'rotlab/newdata'
API = 'https://www.kaggle.com/api/v1/datasets/download/ivannikolov/longterm-thermal-drift-dataset/{}?datasetVersionNumber=3'
N = 150


def get(name):
    with urllib.request.urlopen(urllib.request.Request(API.format(urllib.parse.quote(name, safe='')), headers={'User-Agent': 'ortus-rotlab-camp27'}), timeout=60) as r:
        return r.read()


def date_of(name):
    b = name.rsplit('/', 1)[-1]
    m = re.match(r'(\d{8})', b)
    return m.group(1) if m else None


def cmd_select():
    files = json.loads((RAW / 'filelist.json').read_text())
    jp = sorted(f['name'] for f in files if f['name'].startswith('Data_Annotated_Subset_Object_Detectors/') and f['name'].endswith('.jpg'))
    by_date = collections.defaultdict(list)
    for n in jp:
        by_date[date_of(n)].append(n)
    h = lambda n: hashlib.md5(f'rwu27-aalborg:{n}'.encode()).hexdigest()
    for d in by_date:
        by_date[d].sort(key=h)
    sel = [v[0] for d, v in sorted(by_date.items(), key=lambda kv: h(kv[0] or ''))]
    rest = [v[1] for d, v in sorted(by_date.items(), key=lambda kv: h('2:' + (kv[0] or ''))) if len(v) > 1]
    sel = (sel + rest)[:N]
    (RAW / 'img').mkdir(parents=True, exist_ok=True); recs = []
    for n in sel:
        f = RAW / 'img' / hashlib.md5(n.encode()).hexdigest()
        if not f.exists():
            f.write_bytes(get(n))
        b = f.read_bytes(); im = Image.open(io.BytesIO(b))
        recs.append(dict(id=f'aalborg_ltd:{n}', name=n, date=date_of(n), sha256=hashlib.sha256(b).hexdigest(), w=im.size[0], h=im.size[1], mode=im.mode))
    for m in sorted(f['name'] for f in files if f['name'].startswith('Data_Annotated_Subset_Object_Detectors/') and f['name'].endswith('.csv')):
        (RAW / 'meta').mkdir(exist_ok=True); (RAW / 'meta' / m.replace('/', '__')).write_bytes(get(m))
    man = dict(source=dict(url='https://vap.aau.dk/ltd/', kaggle='ivannikolov/longterm-thermal-drift-dataset', version=3, licence='CC-BY-4.0'),
               selection='annotated-subset JPEGs; 1 per date by md5 order, then 2nd images from dates in hash order, N=150; no model outputs used',
               listing=dict(files_total=len(files), annotated_jpgs=len(jp), dates=len(by_date)), independent_cameras=1,
               label_definition='theta = synthetic clockwise rotation of the stored raster (base roll unvalidated)', images=recs)
    (ND / 'aalborg.json').write_text(json.dumps(man, indent=1))
    print('selected', len(recs), 'dates', len({r['date'] for r in recs}), 'sizes', collections.Counter((r['w'], r['h'], r['mode']) for r in recs).most_common(3),
          'manifest', hashlib.sha256((ND / 'aalborg.json').read_bytes()).hexdigest(), flush=True)
    S = Image.new('RGB', (6 * 192, 4 * 144))
    for k, r in enumerate(recs[:24]):
        im = Image.open(RAW / 'img' / hashlib.md5(r['name'].encode()).hexdigest()).convert('RGB'); im.thumbnail((192, 144)); S.paste(im, ((k % 6) * 192, (k // 6) * 144))
    S.save('/workspace/ops/aalborg-sheet.jpg', quality=85)


def cmd_views():
    from rotlab.camp_newdata import open_holdout
    if '--i-am-final' not in sys.argv:
        raise SystemExit('final-only panel: views only with --i-am-final after the freeze')
    open_holdout()
    man = json.loads((ND / 'aalborg.json').read_text()); dd = dict(png=[], theta=[], ids=[], date=[])
    for r in man['images']:
        raw = (RAW / 'img' / hashlib.md5(r['name'].encode()).hexdigest()).read_bytes()
        if hashlib.sha256(raw).hexdigest() != r['sha256']:
            raise SystemExit(f'{r["id"]}: source bytes differ from the manifest')
        src = Image.open(io.BytesIO(raw)).convert('RGB'); W, H = src.size
        t = random.Random(f'rwu27:aalborg:{r["id"]}').uniform(0, 360); d = min(W, H) - 4
        v = render(src, t, 0.8 * d, 0.6 * d, round(0.8 * d), round(0.6 * d)); b = io.BytesIO(); v.save(b, format='PNG')
        dd['png'].append(b.getvalue()); dd['theta'].append(t); dd['ids'].append(r['id']); dd['date'].append(r['date'])
    dd['theta'] = np.array(dd['theta']); f = ND / 'views' / 'aalborg_final_thermal.pkl'; f.write_bytes(pickle.dumps(dd))
    print('views', f.name, len(dd['png']), hashlib.sha256(f.read_bytes()).hexdigest()[:16])
    from rotlab.camp_final import register_views
    register_views(f.stem, f, dd['ids'])


if __name__ == '__main__':
    {'select': cmd_select, 'views': cmd_views}[sys.argv[1]]()
