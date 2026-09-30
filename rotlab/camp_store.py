"""Campaign 2026-09-27: shared result store for per-tag prediction files (<tag>.npz + <tag>.json receipt).

Writers compute OUTSIDE any lock, then call commit(): a short exclusive flock that re-reads the latest published pair,
verifies it (receipt['npz_sha256'] must equal the NPZ bytes; identity fields must match; per-set view hashes must not
change), merges the new sets, and publishes NPZ then receipt via temp files + atomic rename. The receipt binds the
NPZ hash, so a partial two-file publication (crash between the renames) or any lost update fails closed on the next
read. Readers use load(), which performs the same verification.
"""
import fcntl, hashlib, json, os, tempfile
from pathlib import Path

import numpy as np


class StoreError(SystemExit):
    pass


def _sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def _read_bytes(p):
    with open(p, 'rb') as f:
        return f.read()


def load(out, tag, verify=True, retries=3):
    """-> (rows dict, receipt dict); ({}, None) if absent. The NPZ is read ONCE into memory; its SHA-256 is computed on
    exactly those bytes and the arrays are materialised from the same buffer, so the returned data is the data whose
    hash matched the receipt (no reopen by path). A concurrent publish between the receipt read and the NPZ read shows
    up as a hash mismatch: retried, then fail closed."""
    import io
    f, rf = Path(out) / f'{tag}.npz', Path(out) / f'{tag}.json'
    for attempt in range(retries):
        if not f.exists() and not rf.exists():
            return {}, None
        if not (f.exists() and rf.exists()):
            raise StoreError(f'{tag}: only one of npz/receipt exists')
        rec = json.loads(_read_bytes(rf))
        buf = _read_bytes(f)
        if verify and rec.get('npz_sha256') != hashlib.sha256(buf).hexdigest():
            if attempt + 1 < retries:
                continue
            raise StoreError(f'{tag}: receipt npz_sha256 does not match the NPZ bytes (partial publication or lost update)')
        with np.load(io.BytesIO(buf)) as z:
            rows = {k: np.array(z[k]) for k in z.files}
        missing = [s for s in rec.get('sets', {}) if f'{s}_pred' not in rows]
        if missing:
            raise StoreError(f'{tag}: receipt lists sets missing from NPZ: {missing}')
        return rows, rec


def commit(out, tag, new_rows, identity, new_sets):
    """identity: fields that must equal the existing receipt's; new_sets: {set: view_sha256}."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    with open(out / f'{tag}.lock', 'a+') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        rows, rec = load(out, tag)
        if rec is None:
            rec = dict(identity, sets={})
        else:
            diff = [k for k, v in identity.items() if rec.get(k) != v]
            if diff:
                raise StoreError(f'{tag}: identity fields differ from receipt: {diff}')
            bad = [s for s, h in new_sets.items() if s in rec['sets'] and rec['sets'][s] != h]
            if bad:
                raise StoreError(f'{tag}: view hashes differ for {bad}')
        rows.update(new_rows); rec['sets'].update(new_sets)
        fd, tmp = tempfile.mkstemp(dir=out, prefix=f'.{tag}.', suffix='.npz'); os.close(fd)
        np.savez(tmp, **rows)
        rec['npz_sha256'] = _sha(tmp)
        fdj, tmpj = tempfile.mkstemp(dir=out, prefix=f'.{tag}.', suffix='.json')
        with os.fdopen(fdj, 'w') as fj:
            json.dump(rec, fj, indent=1)
        os.replace(tmp, out / f'{tag}.npz'); os.replace(tmpj, out / f'{tag}.json')
        return rec
