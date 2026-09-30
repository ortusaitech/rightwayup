#!/usr/bin/env python3
"""Re-check the RightWayUp 1.0 training manifests.

    python3 verify.py [DIR]          # DIR defaults to the folder this file is in

Reads every <family>.jsonl.gz in DIR (one JSON row per training image, no image bytes), recomputes the summary
(rows, rows used in training, per licence, per licence class, per dataset, per family, per model tier, gaps) and
checks that
  1. it equals DIR/SUMMARY.json exactly,
  2. the files match DIR/SHA256SUMS,
  3. the totals equal the published counts (DATA-CARD.md): 1,251,672 rows / 1,245,604 used for the 23 families
     (Max, Fast, Pico) and 705,075 used for the 16 families of Nano,
  4. every row's licence_class follows from its recorded licence string (table LICENCE_CLASS below),
  5. ids are unique across the whole set.
Standard library only. Exit code 0 = all checks pass.
"""
import collections, gzip, hashlib, json, re, sys
from pathlib import Path

FAMILIES = ['gf_pass', 'gf_pass2', 'gf_coco', 'gf_coco2', 'gf_oi', 'gf_diode', 'gf_diode2', 'gf_meva', 'gf_meva2',
            'gf_poly_haven', 'gf_poly_direct', 'gf_fresh_coco', 'gf_fresh_oi', 'gf_oi_train2', 'gf_coco_pd', 'gf_coco_by',
            'gf_cc1', 'gf_cc2', 'gf_cc3', 'gf_cc4', 'gf_oi7a', 'gf_oi7b', 'gf_oi7c']
NANO_FAMILIES = FAMILIES[:16]          # the 16-family grid-free mix (Nano = GF-SOUP-SV2); the last 7 are the MIX v3 expansion
TIERS = {'Max, Fast, Pico (and Balanced/Pro, built from Max + Fast)': FAMILIES, 'Nano': NANO_FAMILIES}
EXPECTED = {'rows': 1251672, 'used_in_training': 1245604, 'nano_used_in_training': 705075}

# Recorded licence string (attribution.licence, else dataset_licence) -> licence class.
LICENCE_CLASS = {
    'CC BY 2.0': 'cc-by', 'CC BY 4.0': 'cc-by',
    'CC BY 2.0 (Flickr "Attribution License")': 'cc-by',
    'CC BY 2.0 (YFCC100M licensename: Attribution License; PASS release)': 'cc-by',
    'No known copyright restrictions': 'pd-like', 'Public Domain Dedication (CC0)': 'pd-like',
    'Public Domain Mark': 'pd-like', 'United States Government Work': 'pd-like',
    'MIT': 'mit',
    'CC0 assets; renders by ORTUS AI': 'cc0-assets',
}
FLICKR_DATASETS = ('PASS', 'COCO', 'Open Images', 'CommonCatalog')   # per-image Flickr licence + author required


def image_licence(r):
    a = r.get('attribution') or {}
    return a.get('licence') or r.get('dataset_licence')


def gaps(r):
    """Names of missing source/licence/attribution facts for one row (empty list = complete)."""
    g = []
    if not r.get('dataset') or not r.get('dataset_licence') or not r.get('dataset_licence_url'):
        g.append('no_dataset_licence')
    lic = image_licence(r)
    if lic not in LICENCE_CLASS:
        g.append('licence_not_in_table')
    loc = r.get('locator') or {}
    if not (loc.get('url') or loc.get('fetched_url')):
        g.append('no_locator_url')
    if r['dataset'].startswith(FLICKR_DATASETS):
        a = r.get('attribution') or {}
        if not a.get('author'):
            g.append('no_author')
        if not a.get('licence') or not a.get('licence_url'):
            g.append('no_image_licence')
        if not a.get('source_url'):
            g.append('no_source_url')
        if not r['dataset'].startswith('PASS') and not a.get('checked_utc'):
            g.append('no_licence_check_date')
    return g


def flickr_photo_id(r):
    """Flickr photo id of a row, from its recorded fields (None for non-Flickr rows)."""
    loc = r.get('locator') or {}; a = r.get('attribution') or {}
    pid = loc.get('flickr_id') or loc.get('yfcc_photoid')
    if pid:
        return str(pid)
    for u in (loc.get('fetched_url'), loc.get('url'), loc.get('flickr_original_url'), a.get('source_url')):
        m = u and (re.search(r'staticflickr\.com/\d+/(\d+)_', u) or re.search(r'flickr\.com/photos/[^/]+/(\d+)', u))
        if m:
            return m.group(1)
    return None


class _Groups:
    """Union-find over rows: rows sharing a Flickr photo id or identical stored bytes (stored_sha256) are one image."""
    def __init__(self):
        self.parent, self.first, self.used = [], {}, []

    def add(self, keys, used):
        i = len(self.parent); self.parent.append(i); self.used.append(used)
        for k in keys:
            if k in self.first:
                self._union(i, self.first[k])
            else:
                self.first[k] = i

    def _find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]; i = self.parent[i]
        return i

    def _union(self, a, b):
        a, b = self._find(a), self._find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

    def counts(self):
        roots = [self._find(i) for i in range(len(self.parent))]
        size = collections.Counter(roots)
        used_roots = {r for r, u in zip(roots, self.used) if u}
        return {'distinct_images': len(size), 'distinct_images_used_in_training': len(used_roots),
                'rows_in_groups_of_2_or_more': sum(n for n in size.values() if n > 1),
                'groups_of_2_or_more': sum(1 for n in size.values() if n > 1)}


def _add(d, k, used):
    e = d.setdefault(k, {'rows': 0, 'used_in_training': 0})
    e['rows'] += 1
    e['used_in_training'] += bool(used)


def summarise(rows_by_family):
    """rows_by_family: iterable of (family, iterator of rows). Returns (summary dict, list of problems)."""
    problems = []
    fams, by_lic, by_cls, by_ds = {}, {}, {}, {}
    gap_ids = collections.defaultdict(list)
    seen = set(); dup = 0
    groups = _Groups(); by_pid, by_sha = collections.Counter(), collections.Counter()
    for fam, rows in rows_by_family:
        f = fams.setdefault(fam, {'rows': 0, 'used_in_training': 0, 'dropped_near_duplicate': 0, 'licence': {},
                                  'licence_class': {}, 'dataset': {}, 'replacements_or_substitutes': 0, 'gaps': {}})
        for r in rows:
            if r['family'] != fam:
                problems.append(f'{fam}: row {r["id"]} has family {r["family"]}')
            if r['id'] in seen:
                dup += 1
            seen.add(r['id'])
            u = r['used_in_training'] is True
            pid = flickr_photo_id(r); keys = ['s:' + r['stored_sha256']] + (['f:' + pid] if pid else [])
            groups.add(keys, u); by_sha[r['stored_sha256']] += 1
            if pid:
                by_pid[pid] += 1
            f['rows'] += 1; f['used_in_training'] += u; f['dropped_near_duplicate'] += not u
            lic = image_licence(r); cls = LICENCE_CLASS.get(lic)
            if r.get('licence_class') != cls:
                problems.append(f'{r["id"]}: licence_class {r.get("licence_class")} != table class {cls} for {lic!r}')
            for d, k in ((f['licence'], lic), (f['licence_class'], cls), (f['dataset'], r['dataset']),
                         (by_lic, lic), (by_cls, cls), (by_ds, r['dataset'])):
                _add(d, k, u)
            f['replacements_or_substitutes'] += bool(r.get('replaces'))
            for g in gaps(r):
                f['gaps'][g] = f['gaps'].get(g, 0) + 1
                gap_ids[g].append(r['id'])
    if dup:
        problems.append(f'{dup} duplicate ids across the set')

    def tot(names):
        return {'families': len(names), 'rows': sum(fams[n]['rows'] for n in names if n in fams),
                'used_in_training': sum(fams[n]['used_in_training'] for n in names if n in fams)}

    srt = lambda d: {k: d[k] for k in sorted(d, key=str)}
    for f in fams.values():
        for k in ('licence', 'licence_class', 'dataset', 'gaps'):
            f[k] = srt(f[k])
    summary = {
        'manifest_set': 'RightWayUp v1.0 (v2 models): per-image training manifests, 23 grid-free families',
        'row_form': 'release row form of release/manifests/training (v1) plus licence_class, derivation, replaces, '
                    'replacement_reason, stored_sha256; see README.md',
        'totals': {'rows': sum(f['rows'] for f in fams.values()),
                   'used_in_training': sum(f['used_in_training'] for f in fams.values()),
                   'dropped_near_duplicate': sum(f['dropped_near_duplicate'] for f in fams.values()),
                   'unique_ids': len(seen)},
        'tiers': {t: {**tot(n), 'family_names': n} for t, n in TIERS.items()},
        'by_licence_class': srt(by_cls),
        'by_licence': srt(by_lic),
        'by_dataset': srt(by_ds),
        'duplicates': {
            'rule': 'rows with the same Flickr photo id (locator.flickr_id / yfcc_photoid / Flickr URL) or identical stored '
                    'training bytes (stored_sha256) are counted as one image',
            'rows_sharing_a_flickr_photo_id': sum(n for n in by_pid.values() if n > 1),
            'flickr_photo_ids_in_more_than_one_row': sum(1 for n in by_pid.values() if n > 1),
            'rows_sharing_stored_sha256': sum(n for n in by_sha.values() if n > 1),
            **groups.counts()},
        'gaps': {g: {'rows': len(v), 'ids_first_50': sorted(v)[:50]} for g, v in sorted(gap_ids.items())},
        'families': {k: fams[k] for k in FAMILIES if k in fams},
    }
    return summary, problems


def read_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def main():
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    ok = True
    missing = [f for f in FAMILIES if not (d / f'{f}.jsonl.gz').exists()]
    if missing:
        print('MISSING manifests:', missing); return 1
    summary, problems = summarise((f, read_rows(d / f'{f}.jsonl.gz')) for f in FAMILIES)
    for p in problems[:20]:
        print('PROBLEM', p)
    ok &= not problems
    stored = json.loads((d / 'SUMMARY.json').read_text())
    same = stored == summary
    print('SUMMARY.json reproduced exactly:', same); ok &= same
    t = summary['totals']; nano = summary['tiers']['Nano']
    print(f"rows {t['rows']:,}  used {t['used_in_training']:,}  dropped {t['dropped_near_duplicate']:,}  "
          f"unique ids {t['unique_ids']:,}  Nano: {nano['families']} families, {nano['used_in_training']:,} used")
    exp = (t['rows'], t['used_in_training'], nano['used_in_training']) == (
        EXPECTED['rows'], EXPECTED['used_in_training'], EXPECTED['nano_used_in_training'])
    print('matches the published totals:', exp); ok &= exp
    sums = d / 'SHA256SUMS'
    if sums.exists():
        bad = []
        for line in sums.read_text().splitlines():
            h, name = line.split(None, 1)
            if hashlib.sha256((d / name).read_bytes()).hexdigest() != h:
                bad.append(name)
        print('SHA256SUMS:', 'all match' if not bad else f'MISMATCH {bad}'); ok &= not bad
    dd = summary['duplicates']
    print(f"distinct images {dd['distinct_images']:,} (used {dd['distinct_images_used_in_training']:,}); "
          f"{dd['rows_in_groups_of_2_or_more']:,} rows in {dd['groups_of_2_or_more']:,} duplicate groups")
    for g, v in summary['gaps'].items():
        print(f'gap {g}: {v["rows"]:,} rows')
    print('ALL CHECKS PASS' if ok else 'CHECKS FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
