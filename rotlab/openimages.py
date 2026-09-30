#!/usr/bin/env python3
"""Open Images (Flickr, CC BY 2.0) as extra photo training data and an independent photo test set.

Eligibility (same bar as rotlab.coco_expand): Open Images lists every image as CC BY 2.0; we also
require (1) Open Images' human orientation field Rotation == 0.0, (2) no EXIF orientation flag on
the served file, (3) Flickr's public oEmbed CURRENTLY reports exactly 'CC BY 2.0' / licence id 4 /
https://creativecommons.org/licenses/by/2.0/ with an author name (one client, <= 8 req/s with automatic
backoff on 429/5xx/timeouts, which are retried rather than recorded; no redirects), and
(4) no near-duplicate (dHash <= 6 bits) of any incumbent-benchmark original.
Subsets: Open Images 'test' -> training pool; Open Images 'validation' -> evaluation only.
Images are fetched from the CVDF mirror, re-encoded at max side 448 and only then kept.

  python -m rotlab.openimages fetch --subset validation --n 4000
  python -m rotlab.openimages fetch --subset test --n 40000
  python -m rotlab.openimages pack            # sources/oi.{bin,jsonl} from the test subset
"""
import argparse, concurrent.futures as cf, csv, glob, hashlib, io, json, re, threading, time, urllib.error, urllib.parse, urllib.request

from PIL import Image

from rotlab.core import DATA
from rotlab.prep_sources import dhash, encode

RAW = DATA / 'raw/openimages'
OUT = DATA / 'rotlab/openimages'
ORIG = DATA / 'acquisitions/e3cn-exact-incumbent-coco2014-test/test_coco2014'
LIC_URL = 'https://creativecommons.org/licenses/by/2.0/'
MIRROR = 'https://open-images-dataset.s3.amazonaws.com/{subset}/{id}.jpg'


class Rate:
    """Single-client pacing with backoff: any 429/5xx/timeout halves the rate for 120 s."""
    def __init__(self, rps): self.max_dt = 1 / rps; self.dt = self.max_dt; self.t = 0.0; self.until = 0.0; self.l = threading.Lock()
    def wait(self):
        with self.l:
            now = time.monotonic()
            if now > self.until: self.dt = self.max_dt
            t = max(now, self.t); self.t = t + self.dt
        time.sleep(max(0.0, t - now))
    def backoff(self):
        with self.l:
            self.dt = min(2.0, self.dt * 2); self.until = time.monotonic() + 120
            print(json.dumps(dict(backoff_rps=round(1 / self.dt, 2))), flush=True)


class Transient(Exception):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k): return None


def candidates(subset, n):
    rows = [r for r in csv.DictReader(open(RAW / f'{subset}-images-with-rotation.csv'))
            if r['Rotation'] == '0.0' and r['License'] == LIC_URL]
    rows.sort(key=lambda r: hashlib.md5(f'oi:{r["ImageID"]}'.encode()).hexdigest())   # fixed pseudo-random order
    return rows[:n]


def origin_hashes():
    f = OUT / 'origin-dhash.json'
    if not f.exists():
        f.write_text(json.dumps([dhash(Image.open(p).convert('RGB')) for p in sorted(glob.glob(str(ORIG / '**/*.jpg'), recursive=True))]))
    return [int(h, 16) for h in json.loads(f.read_text())]


def one(row, subset, rate, opener, oh):
    iid = row['ImageID']; d = OUT / subset; rec_f = d / 'records' / f'{iid}.json'
    rec = dict(id=iid, subset=subset, landing=row['OriginalLandingURL'], queried_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    try:
        pid = re.search(r'/(\d+)/?$', row['OriginalLandingURL'].rstrip('/') + '/').group(1)
        url = 'https://www.flickr.com/services/oembed/?format=json&url=' + urllib.parse.quote(f'https://www.flickr.com/photos/any/{pid}/', safe='')
        rate.wait()
        try:
            with opener.open(urllib.request.Request(url, headers={'User-Agent': 'ortus-rotlab-attribution/1'}), timeout=15) as r:
                o = json.loads(r.read(65536))
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                rate.backoff(); raise Transient(e.code)
            raise
        except (TimeoutError, urllib.error.URLError) as e:
            rate.backoff(); raise Transient(str(e))
        rec['attribution'] = {k: o.get(k) for k in ('author_name', 'author_url', 'title', 'web_page', 'license', 'license_id', 'license_url')}
        ok = (o.get('license') == 'CC BY 2.0' and str(o.get('license_id')) == '4' and o.get('license_url') == LIC_URL
              and isinstance(o.get('author_name'), str) and o['author_name'].strip() and f'/{pid}/' in o.get('web_page', ''))
        if not ok:
            rec['status'] = 'excluded_licence'
        else:
            with urllib.request.urlopen(MIRROR.format(subset=subset, id=iid), timeout=30) as r:
                raw = r.read()
            im = Image.open(io.BytesIO(raw))
            if im.getexif().get(0x0112, 1) != 1:
                rec['status'] = 'excluded_exif_orientation'
            else:
                body, (w, h), dh, orig = encode(raw, 448)
                if min(bin(int(dh, 16) ^ x).count('1') for x in oh) <= 6:
                    rec['status'] = 'excluded_near_duplicate_of_origin'
                else:
                    (d / 'img' / f'{iid}.jpg').write_bytes(body)
                    rec.update(status='eligible', sha256_original=hashlib.sha256(raw).hexdigest(), w=w, h=h, orig_w=orig[0], orig_h=orig[1], dhash=dh)
    except Transient:
        return 'transient_retry_later'        # no record written: retried on the next pass
    except Exception as e:
        rec.update(status='unresolved', reason=f'{type(e).__name__}: {e}'[:160])
    tmp = rec_f.with_suffix('.tmp'); tmp.write_text(json.dumps(rec)); tmp.replace(rec_f)   # atomic
    return rec['status']


def cmd_fetch(a):
    d = OUT / a.subset; (d / 'records').mkdir(parents=True, exist_ok=True); (d / 'img').mkdir(exist_ok=True)
    oh = origin_hashes()
    rows = [r for r in candidates(a.subset, a.n) if not (d / 'records' / f'{r["ImageID"]}.json').exists()]
    print('to fetch', len(rows), flush=True)
    rate = Rate(a.rps); opener = urllib.request.build_opener(NoRedirect); n = {}
    with cf.ThreadPoolExecutor(16) as ex:
        for i, s in enumerate(ex.map(lambda r: one(r, a.subset, rate, opener, oh), rows)):
            n[s] = n.get(s, 0) + 1
            if (i + 1) % 1000 == 0: print(json.dumps(dict(done=i + 1, **n)), flush=True)
    print(json.dumps(dict(done=len(rows), **n)))


def cmd_pack(a):
    d = OUT / 'test'; src = DATA / 'rotlab/sources'
    recs = sorted((json.loads(p.read_text()) for p in (d / 'records').glob('*.json')), key=lambda r: r['id'])
    elig = [r for r in recs if r['status'] == 'eligible']
    with (src / 'oi.bin.partial').open('wb') as fb, (src / 'oi.jsonl.partial').open('w') as fi:
        off = 0
        for r in elig:
            body = (d / 'img' / f"{r['id']}.jpg").read_bytes(); fb.write(body)
            fi.write(json.dumps(dict(id=f"openimages_test:{r['id']}", family='oi', group='openimages_test_ccby', base_roll_cw=0.0, weight=1.0,
                                     attribution=r['attribution'], sha256_original=r['sha256_original'], w=r['w'], h=r['h'],
                                     orig_w=r['orig_w'], orig_h=r['orig_h'], dhash=r['dhash'], offset=off, length=len(body))) + '\n')
            off += len(body)
    (src / 'oi.bin.partial').rename(src / 'oi.bin'); (src / 'oi.jsonl.partial').rename(src / 'oi.jsonl')
    st = {}
    for r in recs: st[r['status']] = st.get(r['status'], 0) + 1
    print(json.dumps(dict(records=len(recs), packed=len(elig), bytes=off, **st)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--subset', default='test'); ap.add_argument('--n', type=int, default=40000); ap.add_argument('--rps', type=float, default=4.0)
    a = ap.parse_args(); {'fetch': cmd_fetch, 'pack': cmd_pack}[a.cmd](a)
