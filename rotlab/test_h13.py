#!/usr/bin/env python3
"""CPU equivalence tests for rotlab.camp_h13 (no GPU, no real data): synthetic source images, real DataLoader workers with
prefetch, fresh-process resumes at worker-UNALIGNED steps, repeated interruption, truncated/altered-state refusal, exact
pretrained-key accounting.  PYTHONPATH=code python -m rotlab.test_h13"""
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class FakeBlob:
    def __init__(self, n, seed):
        self.n, self.seed = n, seed
        self.rows = [dict(id=f'fake{seed}-{i}', offset=i, length=1, base_roll_cw=0.0) for i in range(n)]

    def __len__(self):
        return self.n

    def image(self, i):
        r = np.random.default_rng(self.seed * 100003 + i)
        w, h = int(r.integers(160, 320)), int(r.integers(120, 260))
        return Image.fromarray(r.integers(0, 255, (h, w, 3), dtype=np.uint8)), dict(base_roll_cw=0.0)


def fake_blobs():
    from rotlab.camp_h13 import EXMIX
    return {f: FakeBlob(50, k) for k, f in enumerate(EXMIX)}


def stream(start, n, workers, batch=4, canvas=64):
    from rotlab.camp_h13 import EXMIX, SeekableViews, batch_digest, loader
    it = iter(loader(SeekableViews(EXMIX, canvas, 0, batch, start, blobs=fake_blobs()), workers, False))
    out = []
    for _ in range(n):
        x, th, bid = next(it); out.append((int(bid), batch_digest(int(bid), x, th).hex()))
    del it   # prefetched-but-unconsumed batches are discarded with the workers
    return out


def test_stream_seek():
    ref = stream(0, 17, 3)
    assert [b for b, _ in ref] == list(range(17))
    for cut in (4, 7, 16):   # none is a multiple of 3 workers (except the chained case below)
        got = stream(0, cut, 3) + stream(cut, 17 - cut, 3)
        assert got == ref, f'resume at {cut} differs'
    got = stream(0, 4, 3) + stream(4, 3, 3) + stream(7, 10, 3)   # repeated interruption
    assert got == ref
    assert stream(5, 12, 2) == ref[5:]                           # independent of worker count
    assert stream(5, 12, 0) == ref[5:]
    return 'stream seek: resume at 4/7/16 with 3 workers + prefetch, repeated interruption, worker-count independence'


def run(tmp, *extra, kill_at=None, receipt=1, milestones='', kill_before_receipt=None, kill_before_publish=None):
    env = dict(os.environ, H13_TEST_BLOBS='1', PYTHONPATH=str(Path(__file__).resolve().parents[1]), OMP_NUM_THREADS='1')
    for k, v in (('H13_TEST_KILL_AT', kill_at), ('H13_TEST_KILL_BEFORE_RECEIPT', kill_before_receipt), ('H13_TEST_KILL_BEFORE_PUBLISH', kill_before_publish)):
        if v:
            env[k] = str(v)
    base = ['--arch', 'tiny', '--run', 'r', '--out-root', str(tmp), '--steps', '14', '--batch', '4', '--warmup', '3', '--workers', '3',
            '--recovery-every', '5', '--canvas', '64', '--milestones', milestones, '--receipt-every', str(receipt)]
    r = subprocess.run([sys.executable, '-m', 'rotlab.camp_h13', 'train'] + base + list(extra), capture_output=True, text=True, env=env, timeout=600)
    return r


def state(tmp):
    d = Path(tmp) / 'r' / 'recovery'
    st = torch.load(sorted(d.glob('step-*.pt'))[-1], weights_only=False)
    return st


def deep_equal(a, b, path='state'):
    """Every serialised field: tensors bitwise, numpy arrays exactly, containers recursively (optimizer param_groups and
    non-tensor state, NumPy/Python/torch RNG, schedule, group order, exports, config/provenance, chain, cursor)."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        assert torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), path
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        assert np.array_equal(np.asarray(a), np.asarray(b)), path
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), (path, set(a) ^ set(b))
        for k in a:
            deep_equal(a[k], b[k], f'{path}.{k}')
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            deep_equal(x, y, f'{path}[{i}]')
    else:
        assert a == b, (path, a, b)


def same(a, b):
    deep_equal(a, b)


def test_train_resume():
    from rotlab.camp_h13 import dose_summary, sha_file
    with tempfile.TemporaryDirectory() as A, tempfile.TemporaryDirectory() as A3, tempfile.TemporaryDirectory() as B, tempfile.TemporaryDirectory() as C, tempfile.TemporaryDirectory() as D:
        r = run(A); assert r.returncode == 0, r.stderr[-2000:]
        ref = state(A)
        r = run(A3, receipt=3); assert r.returncode == 0, r.stderr[-2000:]
        ref3 = state(A3)   # same protocol, sparse receipts (operational field in the config)
        # B: ABRUPT death (SIGKILL, live prefetching workers) at step 7 after the step-5 recovery (unaligned with 3 workers)
        r = run(B, kill_at=7); assert r.returncode == -9, (r.returncode, r.stderr[-800:])
        ck = Path(B) / 'r/recovery/step-000005.pt'; sha = sha_file(ck)
        r = run(B, '--resume', str(ck), '--resume-sha256', sha); assert r.returncode != 0 and 'resume-authority' in r.stderr + r.stdout
        r = run(B, '--resume', str(ck), '--resume-sha256', sha, '--resume-authority', 'test'); assert r.returncode == 0, r.stderr[-2000:]
        same(ref, state(B))
        ds = dose_summary(Path(B) / 'r', 4, 1)
        assert ds['retained_presentations'] == 14 * 4 and ds['physical_updates_range'] == [7 + 9, 8 + 9]   # inclusive: death may follow update 8 before its receipt and ds['completed'], ds
        # C: repeated interruption with sparse receipts (every 3): kill 7 -> resume 5, kill 11 -> resume 10 -> 14
        assert run(C, kill_at=7, receipt=3).returncode == -9
        c5 = Path(C) / 'r/recovery/step-000005.pt'
        assert run(C, '--resume', str(c5), '--resume-sha256', sha_file(c5), '--resume-authority', 't', kill_at=11, receipt=3).returncode == -9
        c10 = Path(C) / 'r/recovery/step-000010.pt'
        assert run(C, '--resume', str(c10), '--resume-sha256', sha_file(c10), '--resume-authority', 't', receipt=3).returncode == 0
        same(ref3, state(C))
        dc = dose_summary(Path(C) / 'r', 4, 3)   # seg1 last receipt 6 -> [6, 9]; seg2 5->last 9 -> [4, 7]; seg3 exact 4
        assert dc['physical_updates_range'] == [6 + 4 + 4, 9 + 7 + 4] and dc['completed'], dc
        # refusal: truncated state, changed config
        bad = Path(C) / 'r/recovery/trunc.pt'; bad.write_bytes(c10.read_bytes()[:1000]); bad.with_suffix('.pt.sha256').write_text(sha_file(c10) + '\n')
        r = run(C, '--resume', str(bad), '--resume-sha256', sha_file(c10), '--resume-authority', 't'); assert r.returncode != 0 and 'refusing' in r.stdout + r.stderr
        r = run(C, '--resume', str(c10), '--resume-sha256', sha_file(c10), '--resume-authority', 't', '--warmup', '4')
        assert r.returncode != 0 and 'config' in r.stdout + r.stderr
        # D: export publication: milestone 6 published after its recovery; kill 7; resume 5 recomputes step 6 -> must verify
        assert run(D, kill_at=7, milestones='6').returncode == -9
        e6 = Path(D) / 'r-s000006/final.pt'; e_sha = sha_file(e6)
        import shutil
        d5 = Path(D) / 'keep/step-000005.pt'; d5.parent.mkdir()   # keep-last-2 rotation will delete the original later
        shutil.copy(Path(D) / 'r/recovery/step-000005.pt', d5); shutil.copy(Path(D) / 'r/recovery/step-000005.pt.sha256', d5.with_suffix('.pt.sha256'))
        r = run(D, '--resume', str(d5), '--resume-sha256', sha_file(d5), '--resume-authority', 't', milestones='6'); assert r.returncode == 0 and 'verified-existing' in r.stdout, r.stdout[-500:]
        assert sha_file(e6) == e_sha
        # tamper the published milestone, resume again from 5 -> refuse (never overwritten)
        with open(e6, 'ab') as f:
            f.write(b'x')
        t_sha = sha_file(e6)
        r = run(D, '--resume', str(d5), '--resume-sha256', sha_file(d5), '--resume-authority', 't', milestones='6')
        assert r.returncode != 0 and 'never overwritten' in r.stdout + r.stderr and sha_file(e6) == t_sha
    with tempfile.TemporaryDirectory() as E, tempfile.TemporaryDirectory() as F:
        # E: death between the update completing receipt boundary 6 and its receipt: actual 6 must lie inside the bound
        assert run(E, kill_before_receipt=6, receipt=3).returncode == -9
        de = dose_summary(Path(E) / 'r', 4, 3)['segments'][0]
        assert de['last_durable_step'] == 3 and de['bounded_unknown'][0] <= 6 <= de['bounded_unknown'][1], de
        # F: death after the FINAL recovery save but before final publication; resume -> no optimizer step, final reconstructed
        assert run(F, kill_before_publish=14).returncode == -9
        assert not (Path(F) / 'r-final/final.pt').exists()
        f14 = Path(F) / 'r/recovery/step-000014.pt'
        r = run(F, '--resume', str(f14), '--resume-sha256', sha_file(f14), '--resume-authority', 't'); assert r.returncode == 0, r.stderr[-800:]
        assert 'reconciled on resume' in r.stdout and (Path(F) / 'r-final/final.pt').exists()
        df = dose_summary(Path(F) / 'r', 4, 1)
        assert df['completed'] and df['segments'][-1] == dict(seg=df['segments'][-1]['seg'], exact=0), df
        same(ref, state(F))
        # a recorded EARLIER milestone that is missing fails closed (never fabricated from later weights)
        (Path(F) / 'r-final/final.pt').unlink()
        g = Path(F) / 'g'; g.mkdir()
        import shutil
        shutil.copy(f14, g / 'step-000014.pt'); shutil.copy(f14.with_suffix('.pt.sha256'), g / 'step-000014.pt.sha256')
        st = torch.load(g / 'step-000014.pt', weights_only=False); st['exports']['r-s000006'] = '0' * 64
        torch.save(st, g / 'step-000014.pt'); (g / 'step-000014.pt.sha256').write_text(sha_file(g / 'step-000014.pt') + '\n')
        r = run(F, '--resume', str(g / 'step-000014.pt'), '--resume-sha256', sha_file(g / 'step-000014.pt'), '--resume-authority', 't')
        assert r.returncode != 0 and 'never fabricated' in r.stdout + r.stderr, r.stdout[-400:] + r.stderr[-400:]
    return ('train resume: death between update and receipt stays inside the inclusive bound; death after final recovery before '
            'publication -> resume reconstructs the final export with zero optimizer steps; missing earlier milestone refused; abrupt SIGKILL with live prefetch at unaligned step 7 (checkpoint 5, 3 workers) and repeated abrupt '
            'interruption give deep-equal state (all serialised fields); physical work exact with dense receipts, bounded [lo, hi] with '
            'sparse receipts; export verified-existing on recompute, tampered export refused (never overwritten); truncated/config-changed '
            'state and missing authority refused')


def test_mo_s_refuses_test_inputs():
    env = dict(os.environ, H13_TEST_BLOBS='1', PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    r = subprocess.run([sys.executable, '-m', 'rotlab.camp_h13', 'train', '--arch', 'mo_s', '--run', 'x', '--out-root', '/tmp/h13-never'],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode != 0 and 'refusing' in r.stdout + r.stderr and not Path('/tmp/h13-never/x').exists()
    from rotlab import camp_h13
    try:
        camp_h13.train(camp_h13.parser().parse_args(['train', '--arch', 'mo_s', '--run', 'x', '--out-root', '/tmp/h13-never']), fake_blobs())
        raise AssertionError('fake blobs accepted for mo_s')
    except SystemExit as e:
        assert 'refusing' in str(e)
    return 'mo_s refuses fake inputs (env and in-process) and test overrides'


def test_keys():
    import timm
    from safetensors.torch import save_file
    from rotlab import camp_h13
    m = camp_h13.MoS(0.0)
    src = timm.create_model('mambaout_small', pretrained=False, num_classes=1000).state_dict()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'w.safetensors'; save_file({k: v.contiguous() for k, v in src.items()}, str(p))
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        info = camp_h13.load_pretrained_small(m, p, sha); assert info['missing'] == ['head.fc.bias', 'head.fc.weight']
        assert torch.equal(m.backbone.stages[2].blocks[0].fc1.weight, src['stages.2.blocks.0.fc1.weight'])
        src2 = dict(src, extra=torch.zeros(1)); save_file({k: v.contiguous() for k, v in src2.items()}, str(p))
        try:
            camp_h13.load_pretrained_small(camp_h13.MoS(0.0), p, hashlib.sha256(p.read_bytes()).hexdigest()); raise AssertionError('unexpected key accepted')
        except SystemExit as e:
            assert 'unexpected' in str(e)
        try:
            camp_h13.load_pretrained_small(camp_h13.MoS(0.0), p, '0' * 64); raise AssertionError('sha accepted')
        except SystemExit as e:
            assert 'sha256' in str(e)
    return 'keys: missing exactly head.fc.{weight,bias}, unexpected refused, sha mismatch refused'


if __name__ == '__main__':
    for t in (test_stream_seek, test_keys, test_mo_s_refuses_test_inputs, test_train_resume):
        print('PASS', t())
