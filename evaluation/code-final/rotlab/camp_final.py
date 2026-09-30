#!/usr/bin/env python3
"""Campaign 2026-09-27 FINAL-HOLDOUT GATE. Nothing final is materialised or scored outside this gate.

  freeze SPEC.json   -> camp-frozen/FROZEN.json (create-exclusive; once). SPEC gives candidate checkpoints and the full
                        inference configuration; this tool adds SHA-256 of every checkpoint, comparator model files,
                        new-data source manifests and the code files that define views and inference.
  open               -> newdata/HOLDOUT-OPENED.json (create-exclusive; once) after re-verifying every hash in FROZEN.json.
                        Binds sha256(FROZEN.json). Final view materialisers require this receipt (require_opened()).
  check-score CKPT CANVAS POOL -> exit 0 only if: opened receipt binds the current FROZEN.json, CKPT sha is a frozen
                        stage, (CANVAS, POOL) equal that stage's frozen config, and camp-frozen/SCORING-APPROVED.json
                        exists and binds the same FROZEN hash (written only after the coordinator's review).
SPEC.json example:
  {"candidate": "max-v2", "stages": {"small": {"ckpt": ".../S2/final.pt", "canvas": 224, "pool": null},
                                      "large": {"ckpt": ".../X/final.pt", "canvas": 224, "pool": null}},
   "route_threshold": 0.836, "decoder": "<camp_eval.DECODER>", "abstain": {...} }
FREEZE v2 (grid-fix, 29 Sep, lead code tree): a stage may have pool "fill" = the shipped fill-drop inference
(rotlab.focus FocusNet plan 'fill', checkpoint sinks/tol), scored on final sets only through `focus eval --mode fill`
behind check_score. SPEC "tiers" defines the released tiers over the stages; aggregate_tiers scores them.
"""
import hashlib, json, os, sys, time
from pathlib import Path

from rotlab.core import DATA

FZ_DIR = DATA / 'rotlab/camp-frozen'
FROZEN = FZ_DIR / 'FROZEN.json'
APPROVED = FZ_DIR / 'SCORING-APPROVED.json'
OPENED = DATA / 'rotlab/newdata/HOLDOUT-OPENED.json'
MANIFESTS = ['coco_val2017.jsonl', 'oi_validation_r8000.jsonl', 'rlivit.json', 'aalborg.json']
CODE = ['core.py', 'model.py', 'evaluate.py', 'camp_eval.py', 'camp_store.py', 'camp_newdata.py', 'camp_rlivit.py', 'camp_aalborg.py',
        'camp_incumbent.py', 'camp_final.py', 'camp_analyze.py', 'prep_sources.py',
        'focus.py', 'model_x.py']   # lead tree: fill-drop scoring (focus) and the SPECS RotNet camp_eval imports
INC = DATA / 'rotlab/release-onnx/incumbent/rotation_estimator_fp32.onnx'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def _excl_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)   # fails if it already exists
    with os.fdopen(fd, 'w') as f:
        json.dump(obj, f, indent=1); f.flush(); os.fsync(f.fileno())


def cmd_freeze(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    from rotlab.camp_eval import DECODER
    if spec.get('decoder') != DECODER:
        raise SystemExit('decoder in spec must equal camp_eval.DECODER (the decoder actually used for scoring)')
    for role, st in spec['stages'].items():
        st['sha256'] = sha(st['ckpt'])
        assert isinstance(st['canvas'], int) and 'pool' in st, role
    code = Path(__file__).parent
    fz = dict(spec, frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              candidates=[dict(role=r, ckpt=s['ckpt'], sha256=s['sha256'], canvas=s['canvas'], pool=s['pool']) for r, s in spec['stages'].items()],
              router=dict(kind='max-mass threshold on small-stage confidence', route_threshold=spec['route_threshold']),
              comparators=dict(woehrer=dict(model=str(INC), model_sha256=sha(INC), data_sha256=sha(str(INC) + '.data'),
                                            adapter='timm create_transform(224, bicubic, crop_pct=1.0); pred = -argmax (clockwise)')),
              source_manifests={m: sha(DATA / 'rotlab/newdata' / m) for m in MANIFESTS},   # raises if a required manifest is absent
              code_sha256={f: sha(code / f) for f in CODE})
    _excl_write(FROZEN, fz)
    print('FROZEN', sha(FROZEN))


def verify_frozen():
    fz = json.loads(FROZEN.read_text())
    for c in fz['candidates']:
        if sha(c['ckpt']) != c['sha256']:
            raise SystemExit(f'frozen checkpoint changed: {c["ckpt"]}')
    w = fz['comparators']['woehrer']
    if sha(w['model']) != w['model_sha256'] or sha(w['model'] + '.data') != w['data_sha256']:
        raise SystemExit('comparator model changed')
    for m, h in fz['source_manifests'].items():
        if sha(DATA / 'rotlab/newdata' / m) != h:
            raise SystemExit(f'source manifest changed: {m}')
    code = Path(__file__).parent
    bad = [f for f, h in fz['code_sha256'].items() if sha(code / f) != h]
    if bad:
        raise SystemExit(f'code changed since freeze: {bad}')
    return fz, sha(FROZEN)


def cmd_open():
    fz, fh = verify_frozen()
    _excl_write(OPENED, dict(opened_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), frozen_sha256=fh,
                             candidate=fz.get('candidate'), note='final views may now be materialised; scoring needs SCORING-APPROVED.json'))
    print('OPENED bound to FROZEN', fh)


def require_opened():
    """Called by every final-view materialiser."""
    if not (FROZEN.exists() and OPENED.exists()):
        raise SystemExit('refusing: final holdout not opened through camp_final (freeze, then open)')
    _, fh = verify_frozen()
    if json.loads(OPENED.read_text()).get('frozen_sha256') != fh:
        raise SystemExit('refusing: HOLDOUT-OPENED.json does not bind the current FROZEN.json')


def check_score(ckpt_sha, canvas, pool):
    require_opened()
    fz, fh = verify_frozen()
    if not APPROVED.exists() or json.loads(APPROVED.read_text()).get('frozen_sha256') != fh:
        raise SystemExit('refusing: final scoring not approved for this FROZEN.json (coordinator review pending)')
    hits = [c for c in fz['candidates'] if (c['sha256'], c['canvas'], c['pool']) == (ckpt_sha, canvas, pool)]
    if not hits:
        raise SystemExit(f'refusing: (checkpoint, canvas {canvas}, pool {pool}) is not a frozen stage tuple')
    return hits[0]


def check_comparator():
    require_opened()
    _, fh = verify_frozen()
    if not APPROVED.exists() or json.loads(APPROVED.read_text()).get('frozen_sha256') != fh:
        raise SystemExit('refusing: final scoring not approved for this FROZEN.json (coordinator review pending)')


FINAL_SETS = frozenset({'newcoco_holdout_fixed', 'newcoco_holdout_maxarea', 'newcoco_holdout_fixed_deg', 'newoi_holdout_fixed',
                        'newoi_holdout_fixed_deg', 'rlivit_final_rgb', 'rlivit_final_thermal', 'aalborg_final_thermal'})


def is_final(name):
    """Explicit registry; any other final-looking name fails closed."""
    if name in FINAL_SETS:
        return True
    if any(k in name for k in ('holdout', 'final', 'aalborg')):
        raise SystemExit(f'refusing: {name} looks like a final set but is not in the explicit registry')
    return False


def register_views(name, path, ids):
    """Create-exclusive per-panel receipt bound to FROZEN: file sha256, count, ordered-id hash."""
    if name not in FINAL_SETS:
        raise SystemExit(f'{name} is not a registered final panel')
    require_opened(); _, fh = verify_frozen()
    rec = dict(panel=name, frozen_sha256=fh, sha256=sha(path), n=len(ids), ids_sha256=hashlib.sha256('\n'.join(map(str, ids)).encode()).hexdigest())
    _excl_write(Path(str(path) + '.receipt.json'), rec)
    return rec


def verify_views(name, path):
    r = Path(str(path) + '.receipt.json')
    if not r.exists():
        raise SystemExit(f'refusing: no create-exclusive view receipt for {name}')
    rec = json.loads(r.read_text()); _, fh = verify_frozen()
    if rec['frozen_sha256'] != fh or rec['sha256'] != sha(path):
        raise SystemExit(f'refusing: {name} view bytes/freeze binding do not match its receipt')
    return rec


def aggregate(out_dir, small_tag, large_tag, sets):
    """Frozen Max cascade on final sets: stage receipts must equal the frozen tuples; threshold/decoder from FROZEN only."""
    from rotlab.camp_store import load
    from rotlab.camp_eval import DECODER
    import numpy as np
    check_comparator()                       # opened + approved + FROZEN integrity (shared gate)
    fz, fh = verify_frozen()
    if fz['decoder'] != DECODER:
        raise SystemExit('decoder differs from frozen')
    st = {c['role']: c for c in fz['candidates']}
    zs, rs = load(out_dir, small_tag); zl, rl = load(out_dir, large_tag)
    for role, r in (('small', rs), ('large', rl)):
        c = st[role]
        if (r['ckpt_sha256'], r['canvas'], r['pool'], r['decoder']) != (c['sha256'], c['canvas'], c['pool'], DECODER):
            raise SystemExit(f'{role} predictions were not produced by the frozen stage tuple')
    thr = fz['router']['route_threshold']; res = {}
    for s in sets:
        if s not in FINAL_SETS:
            raise SystemExit(f'{s}: not a registered final set')
        from rotlab.camp_eval import view_file
        sealed = verify_views(s, view_file(s))['sha256']          # sealed view receipt, re-hashed now
        if rs['sets'].get(s) != sealed or rl['sets'].get(s) != sealed or not np.array_equal(zs[f'{s}_theta'], zl[f'{s}_theta']):
            raise SystemExit(f'{s}: stage predictions not bound to the sealed view bytes, or misaligned')
        r = zs[f'{s}_conf'] < thr; p = np.where(r, zl[f'{s}_pred'], zs[f'{s}_pred'])
        e = np.abs((p - zs[f'{s}_theta']) % 360); e = np.minimum(e, 360 - e)
        res[s] = dict(n=int(len(e)), w10=int((e <= 10).sum()), route=float(r.mean()), mae=float(e.mean()), t90=int((e > 90).sum()), t150=int((e >= 150).sum()))
    return dict(frozen_sha256=fh, threshold=thr, results=res)


def aggregate_tiers(out_dir, tags, sets):
    """FREEZE v2 released tiers on final sets. tags: {stage role: store tag}. Every stage store must carry the frozen
    (ckpt sha, canvas, pool, decoder) and be bound to the sealed view bytes; tier definitions and thresholds come from
    FROZEN only. Routed images take the large stage's prediction and confidence (release final_stats.Tier)."""
    from rotlab.camp_store import load
    from rotlab.camp_eval import DECODER, view_file
    import numpy as np
    check_comparator()
    fz, fh = verify_frozen()
    if fz['decoder'] != DECODER:
        raise SystemExit('decoder differs from frozen')
    st = {c['role']: c for c in fz['candidates']}
    z = {}
    for role, tag in tags.items():
        zz, r = load(out_dir, tag); c = st[role]
        if (r['ckpt_sha256'], r['canvas'], r['pool'], r['decoder']) != (c['sha256'], c['canvas'], c['pool'], DECODER):
            raise SystemExit(f'{role} predictions were not produced by the frozen stage tuple')
        z[role] = (zz, r)
    res = {}
    for s in sets:
        if s not in FINAL_SETS:
            raise SystemExit(f'{s}: not a registered final set')
        sealed = verify_views(s, view_file(s))['sha256']
        th = None
        for role, (zz, r) in z.items():
            if r['sets'].get(s) != sealed:
                raise SystemExit(f'{s}: {role} predictions not bound to the sealed view bytes')
            if th is None:
                th = zz[f'{s}_theta']
            elif not np.array_equal(zz[f'{s}_theta'], th):
                raise SystemExit(f'{s}: stage predictions misaligned')
        out = {}
        for name, t in fz['tiers'].items():
            a = z[t['stage']][0]; p, c = a[f'{s}_pred'].copy(), a[f'{s}_conf'].copy(); rt = np.zeros(len(p), bool)
            if t.get('large'):
                b = z[t['large']][0]; rt = c < t['route']; p[rt], c[rt] = b[f'{s}_pred'][rt], b[f'{s}_conf'][rt]
            e = np.abs((p - th) % 360); e = np.minimum(e, 360 - e); ok = e <= 10
            ab = {}
            for k in ('standard', 'strict'):
                thr = max(t['standard'], t['strict']) if k == 'strict' else t['standard']   # effective strict = max(standard, strict)
                ans = c >= thr
                ab[k] = dict(threshold=thr, answered=float(ans.mean()), w10_answered=int(ok[ans].sum()), n_answered=int(ans.sum()),
                             wrong_answered=float((~ok[ans]).mean()) if ans.any() else 0.0)
            out[name] = dict(n=int(len(e)), w10=int(ok.sum()), route=float(rt.mean()), mae=float(e.mean()),
                             t90=int((e > 90).sum()), t150=int((e >= 150).sum()), abstain=ab)
        res[s] = out
    return dict(frozen_sha256=fh, results=res)


def aggregate_incumbent(out_dir, sets):
    """Comparator final metrics: receipt must carry the frozen comparator identity and sealed view hashes."""
    from rotlab.camp_store import load
    from rotlab.camp_eval import view_file
    import numpy as np
    check_comparator(); fz, fh = verify_frozen(); w = fz['comparators']['woehrer']
    z, r = load(out_dir, 'incumbent')
    if (r['model_sha256'], r['data_sha256'], r['adapter']) != (w['model_sha256'], w['data_sha256'], w['adapter']):
        raise SystemExit('incumbent predictions not produced by the frozen comparator identity')
    res = {}
    for s in sets:
        if s not in FINAL_SETS:
            raise SystemExit(f'{s}: not a registered final set')
        if r['sets'].get(s) != verify_views(s, view_file(s))['sha256']:
            raise SystemExit(f'{s}: incumbent predictions not bound to the sealed view bytes')
        e = np.abs((z[f'{s}_pred'] - z[f'{s}_theta']) % 360); e = np.minimum(e, 360 - e)
        res[s] = dict(n=int(len(e)), w10=int((e <= 10).sum()), mae=float(e.mean()), t90=int((e > 90).sum()), t150=int((e >= 150).sum()))
    return dict(frozen_sha256=fh, results=res)


if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'freeze':
        cmd_freeze(sys.argv[2])
    elif c == 'open':
        cmd_open()
    elif c == 'check-score':
        check_score(sys.argv[2], int(sys.argv[3]), None if sys.argv[4] == 'None' else sys.argv[4]); print('ok')
