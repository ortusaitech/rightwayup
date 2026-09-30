#!/usr/bin/env python3
"""Expand the COCO CC-BY training pool (same rule as the 10 Sep attribution check).

Candidates: COCO train2017 images with historical licence id 4 (CC BY 2.0), minus the
origin-benchmark and E3cz exposure IDs and minus the 6,000 IDs already attempted earlier.
For each, query Flickr's public oEmbed (no key/account; <=4 req/s; no retries/redirects).
Eligible only if the CURRENT grant is exactly 'CC BY 2.0' with licence URL
https://creativecommons.org/licenses/by/2.0/, with author and title state recorded.
Eligible originals are read from the existing train2017.zip on the volume and packed
(max side 448) into sources/coco2.{bin,jsonl} with their attribution record.

  python -m rotlab.coco_expand query   # writes rotlab/coco-attribution/<id>.json
  python -m rotlab.coco_expand pack    # builds sources/coco2.*
"""
import concurrent.futures as cf, io, json, re, sys, threading, time, urllib.parse, urllib.request, zipfile
from pathlib import Path

from rotlab.core import DATA
from rotlab.prep_sources import encode

ANN = DATA / 'research-restricted/incumbent-coco2017-domain-v1/annotations_trainval2017-root-preflight-20260910.zip'
ZIP = DATA / 'research-restricted/incumbent-coco2017-domain-v1/train2017.zip'
OUT = DATA / 'rotlab/coco-attribution'
INPUTS = Path(__file__).with_name('data') / 'coco_expand_inputs.json'
LIC_URL = 'https://creativecommons.org/licenses/by/2.0/'


def photo_id(url):
    return re.search(r'/(\d+)_[0-9a-f]+(?:_[a-z])?\.jpg$', url).group(1)


def candidates():
    inp = json.loads(INPUTS.read_text())
    skip = set(inp['already_attempted']) | set(inp['excluded'])
    with zipfile.ZipFile(ANN) as z:
        train = json.loads(z.read('annotations/instances_train2017.json'))['images']
    return [r for r in train if r['license'] == 4 and r['id'] not in skip]


class Rate:
    def __init__(self, rps): self.dt = 1 / rps; self.t = 0.0; self.l = threading.Lock()
    def wait(self):
        with self.l:
            now = time.monotonic(); t = max(now, self.t); self.t = t + self.dt
        time.sleep(max(0.0, t - now))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k): return None


def query(row, rate, opener):
    pid = photo_id(row['flickr_url'])
    url = 'https://www.flickr.com/services/oembed/?format=json&url=' + urllib.parse.quote(f'https://www.flickr.com/photos/any/{pid}/', safe='')
    rate.wait()
    rec = dict(id=row['id'], file_name=row['file_name'], flickr_photo_id=pid, queried_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    try:
        with opener.open(urllib.request.Request(url, headers={'User-Agent': 'ortus-rotlab-attribution/1'}), timeout=15) as r:
            d = json.loads(r.read(65536))
        ok = (d.get('license') == 'CC BY 2.0' and str(d.get('license_id')) == '4' and d.get('license_url') == LIC_URL
              and isinstance(d.get('author_name'), str) and d['author_name'].strip() and isinstance(d.get('title'), str)
              and f'/{pid}/' in d.get('web_page', ''))
        rec.update(status='eligible' if ok else 'excluded',
                   attribution={k: d.get(k) for k in ('author_name', 'author_url', 'title', 'web_page', 'license', 'license_id', 'license_url')})
    except Exception as e:
        rec.update(status='unresolved', reason=f'{type(e).__name__}: {e}'[:160])
    (OUT / f"{row['id']}.json").write_text(json.dumps(rec))
    return rec['status']


def cmd_query():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in candidates() if not (OUT / f"{r['id']}.json").exists()]
    print('to query', len(rows), flush=True)
    rate = Rate(4.0); opener = urllib.request.build_opener(NoRedirect)
    n = {}
    with cf.ThreadPoolExecutor(4) as ex:
        for i, s in enumerate(ex.map(lambda r: query(r, rate, opener), rows)):
            n[s] = n.get(s, 0) + 1
            if (i + 1) % 500 == 0: print(json.dumps(dict(done=i + 1, **n)), flush=True)
    print(json.dumps(dict(done=len(rows), **n)))


def cmd_pack():
    recs = [json.loads(p.read_text()) for p in OUT.glob('*.json')]
    elig = sorted((r for r in recs if r['status'] == 'eligible'), key=lambda r: r['id'])
    src = DATA / 'rotlab/sources'
    with zipfile.ZipFile(ZIP) as z, (src / 'coco2.bin.partial').open('wb') as fb, (src / 'coco2.jsonl.partial').open('w') as fi:
        off = 0
        for r in elig:
            body, (w, h), dh, orig = encode(z.read('train2017/' + r['file_name']), 448)
            fb.write(body)
            fi.write(json.dumps(dict(id=f"coco2017_train:{r['id']:012d}", family='coco', group='coco_train2017_ccby_v2', base_roll_cw=0.0,
                                     weight=1.0, coco_id=r['id'], attribution=r['attribution'], w=w, h=h, dhash=dh, offset=off, length=len(body))) + '\n')
            off += len(body)
    (src / 'coco2.bin.partial').rename(src / 'coco2.bin'); (src / 'coco2.jsonl.partial').rename(src / 'coco2.jsonl')
    print(json.dumps(dict(queried=len(recs), eligible=len(elig), packed_bytes=off)))


if __name__ == '__main__':
    {'query': cmd_query, 'pack': cmd_pack}[sys.argv[1]]()
