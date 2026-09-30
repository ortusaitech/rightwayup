#!/usr/bin/env python3
"""Publishable definitions of the evaluation sets (owner decision D8: published at release). Pointers, not pixels,
except for the ORTUS AI renders in the clean v1 frozen test (released separately under CC BY 4.0).

  python -m rotlab.export_eval_defs --out DIR [--renders-tar FILE]

  meva-test-v1-frames.jsonl        every MEVA test-v1 frame: camera, split (dev / frozen / moved to training), public
                                   MEVA video key + frame index or time, base roll, SHA-256 of the extracted frame
  meva-frozen-views.jsonl          the 560 protected views in scoring order: frame, clean/degraded, applied roll (cw)
  clean-v1-frozen-parents.jsonl    the 1,427 clean v1 frozen-test parents: DIODE file / MEVA key + frame / render file
  clean-v1-frozen-views.jsonl      the 2,854 protected views in scoring order
Views are regenerated exactly by `python -m rotlab.protected_eval dump --i-am-releasing` (fixed seeds) from these frames.
"""
import argparse, hashlib, io, json, pickle, re, tarfile
from pathlib import Path

from rotlab.core import DATA
from rotlab.export_manifest import MEVA_BUCKET, MEVA_DERIVED, meva_keys
from rotlab.meva_test import ROOT as MEVA_ROOT, split_of
from rotlab.protected_eval import SPLITS, V1, VC


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def meva_locator(clip, frame):
    if clip.startswith('meva-clip-'):
        s = json.loads((MEVA_DERIVED / clip / 'summary.json').read_text())
        return dict(meva_object_key=s['object_key'], url=MEVA_BUCKET + s['object_key'],
                    frame_index=s['selected_frame_indices'][int(frame.lstrip('f'))])
    ks = meva_keys(); key = ks.get(clip + '.avi') or ks[clip + '.r13.avi']
    return dict(meva_object_key=key, url=MEVA_BUCKET + key, frame_at_seconds=int(frame.lstrip('t')))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--renders-tar', type=Path)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(MEVA_ROOT / 'frames.jsonl')]
    with open(a.out / 'meva-test-v1-frames.jsonl', 'w') as f:
        for r in rows:
            p = MEVA_ROOT / 'frames' / r['cam'] / Path(r['path']).name
            f.write(json.dumps(dict(frame_id=f"{r['cam']}/{Path(r['path']).stem}", camera=r['cam'], split=split_of(r['cam']),
                                    locator=meva_locator(r['clip'], r['frame']), base_roll_cw=r['base_roll_cw'],
                                    label_authority=r['label_authority'], width=r['w'], height=r['h'],
                                    frame_sha256=sha(p))) + '\n')
    frozen = [r for r in rows if split_of(r['cam']) == 'frozen']
    d = pickle.loads((VC / 'meva_frozen.pkl').read_bytes())
    assert len(d['theta']) == 2 * len(frozen)
    with open(a.out / 'meva-frozen-views.jsonl', 'w') as f:
        for i, th in enumerate(d['theta']):
            r = frozen[i // 2]
            f.write(json.dumps(dict(view=i, frame_id=f"{r['cam']}/{Path(r['path']).stem}", variant=d['variant'][i],
                                    applied_roll_cw_incl_base=round(float(th), 6))) + '\n')

    keep = set(json.loads(SPLITS.read_text())['frozen_test'])
    parents = sorted((p for p in map(json.loads, open(V1 / 'metadata/parents-materialized.jsonl')) if p['parent_id'] in keep),
                     key=lambda p: p['archive_parent_path'])
    tar = tarfile.open(a.renders_tar, 'w') if a.renders_tar else None
    with open(a.out / 'clean-v1-frozen-parents.jsonl', 'w') as f:
        for p in parents:
            sp = p['source_path']; fam = p['source_family']
            if fam == 'diode':
                loc = dict(diode_file=sp.split('/raw/diode-2019/', 1)[1].split('/', 1)[1], url='https://diode-dataset.org/')
            elif fam == 'meva':
                clip, fr = re.search(r'(meva-clip-[0-9a-f]+)/frame-(\d+)\.jpg$', sp).groups()
                loc = meva_locator(clip, f'f{int(fr):03d}')
            else:
                name = f"evaluation/clean-v1-frozen/{Path(p['archive_parent_path']).name}"
                loc = dict(render_file=name, note='ORTUS AI render of Poly Haven assets (CC0); in the renders release, CC BY 4.0')
                if tar:
                    tar.add(str(V1 / p['archive_parent_path']), arcname=f'ortus-polyhaven-orientation-renders-v1/renders/{name}')
            f.write(json.dumps(dict(parent_id=p['parent_id'], family=fam, group=p.get('source_group'), locator=loc,
                                    base_roll_cw=float(p['base_orientation_degrees']) % 360,
                                    parent_sha256=p.get('archive_file_sha256'))) + '\n')
    if tar:
        tar.close()
    d = pickle.loads((VC / 'v1_frozen.pkl').read_bytes())
    assert len(d['theta']) == 2 * len(parents)
    with open(a.out / 'clean-v1-frozen-views.jsonl', 'w') as f:
        for i, th in enumerate(d['theta']):
            f.write(json.dumps(dict(view=i, parent_id=parents[i // 2]['parent_id'], group=d['group'][i], variant=d['variant'][i],
                                    applied_roll_cw_incl_base=round(float(th), 6))) + '\n')
    print(json.dumps(dict(meva_frames=len(rows), meva_frozen_views=len(frozen) * 2, clean_v1_parents=len(parents),
                          clean_v1_views=len(parents) * 2)))


if __name__ == '__main__':
    main()
