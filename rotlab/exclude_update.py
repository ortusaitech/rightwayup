#!/usr/bin/env python3
"""Add near-duplicates (dHash <= 6 bits) of any incumbent-benchmark original in the given packed families
to sources/exclude.json (read by train.blobs_for).  python -m rotlab.exclude_update pass2 meva2 diode2"""
import json, sys
from rotlab.core import DATA
from rotlab.openimages import origin_hashes

f = DATA / 'rotlab/sources/exclude.json'
ex = json.loads(f.read_text()) if f.exists() else {'ids': []}
oh = origin_hashes(); ids = set(ex['ids']); new = 0
for fam in sys.argv[1:]:
    for l in open(DATA / f'rotlab/sources/{fam}.jsonl'):
        r = json.loads(l); h = int(r['dhash'], 16)
        if r['id'] not in ids and min(bin(h ^ x).count('1') for x in oh) <= 6:
            ids.add(r['id']); new += 1
ex['ids'] = sorted(ids); f.write_text(json.dumps(ex))
print(json.dumps(dict(families=sys.argv[1:], added=new, total=len(ids))))
