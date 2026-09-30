#!/usr/bin/env python3
"""MIX v3 (PREREG 18:45): current GFMIX x 0.60 + new families (gf_oi7*, gf_cc*) 0.40 in proportion to their row counts.
Prints the --mix string. Only families present in SOURCES with a non-empty .bin and .jsonl are used.
  python3 mixv3.py SOURCES_DIR [NEW_SHARE=0.40]"""
import sys
from pathlib import Path
BASE = 'gf_pass=0.10,gf_pass2=0.07,gf_coco=0.05,gf_coco2=0.08,gf_oi=0.15,gf_diode=0.09,gf_diode2=0.04,gf_meva=0.09,gf_meva2=0.04,gf_poly_haven=0.05,gf_poly_direct=0.02,gf_fresh_coco=0.07,gf_fresh_oi=0.03,gf_oi_train2=0.10,gf_coco_pd=0.01,gf_coco_by=0.01'
S = Path(sys.argv[1]); share = float(sys.argv[2]) if len(sys.argv) > 2 else 0.40
new = {}
for j in sorted(list(S.glob('gf_oi7*.jsonl')) + list(S.glob('gf_cc*.jsonl'))):
    b = j.with_suffix('.bin')
    if b.exists() and b.stat().st_size > 0 and j.stat().st_size > 0:
        new[j.stem] = sum(1 for _ in open(j))
tot = sum(new.values())
parts = [f'{k}={float(v) * (1 - share if tot else 1):.4f}' for k, v in (kv.split('=') for kv in BASE.split(','))]
parts += [f'{k}={share * n / tot:.4f}' for k, n in new.items()] if tot else []
print(','.join(parts))
print(f'# new families {new} total {tot}', file=sys.stderr)
