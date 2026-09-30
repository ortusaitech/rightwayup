#!/usr/bin/env python3
"""CPU tests for rotlab.earlyexit (no GPU, no real data): a tiny random DINOv2-structured ViT (timm vit_large_patch14_dinov2
with embed 64, depth 8, 4 heads) registered as arch 't'.
  pytest rotlab/test_earlyexit.py          (PYTHONPATH=code; pytest optional)
  PYTHONPATH=/workspace/code python -m rotlab.test_earlyexit   (pod layout)"""
import io, math, os, sys, tempfile
from pathlib import Path

import numpy as np
import torch

import rotlab
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in [os.path.abspath(p) for p in rotlab.__path__]:   # local layout: lead files live beside the campaign package
    rotlab.__path__.append(_here)

import timm
from timm.models import register_model
from rotlab import model as rmodel
from rotlab import earlyexit as ee
from rotlab.core import circ_err, decode
from rotlab.evaluate import confidence

TINY = dict(embed_dim=64, depth=8, num_heads=4)
if 'vit_eetest_patch14' not in timm.list_models('vit_eetest*'):
    @register_model
    def vit_eetest_patch14(pretrained=False, **kw):
        from timm.models.vision_transformer import vit_large_patch14_dinov2
        return vit_large_patch14_dinov2(pretrained=False, **dict(kw, **TINY))
rmodel.ARCH['t'] = ('vit_eetest_patch14', None)
EXITS = (2, 4, 6)


def _jitter(m, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return m


def tiny_rotnet(img=56, seed=0):
    torch.manual_seed(seed)
    return _jitter(rmodel.RotNet(img_size=img, pretrained=False, dynamic=True, arch='t', drop_path=0.0), seed).eval()


def tiny_ee(img=56, seed=0, exits=EXITS, **kw):
    torch.manual_seed(seed)
    return _jitter(ee.EarlyExitRotNet(exits=exits, img_size=img, pretrained=False, dynamic=True, arch='t', drop_path=0.0, **kw), seed).eval()


def test_exit_output_shapes():
    m = tiny_ee()
    assert m.depth == 8 and m.blocks_at == [2, 4, 6, 8]
    for cv in (56, 70, 42):
        x = torch.randn(3, 3, cv, cv)
        with torch.no_grad():
            outs = m(x, all_exits=True)
        assert len(outs) == 4 and all(o.shape == (3, 360) for o in outs)
        assert m(x).shape == (3, 360)
    assert {k.split('.')[1] for k in m.state_dict() if k.startswith('exits.')} == {'b02', 'b04', 'b06'}
    try:
        ee.EarlyExitRotNet(exits=(8,), img_size=56, pretrained=False, arch='t'); raise AssertionError('exit at depth accepted')
    except ValueError:
        pass


def test_final_exit_equals_rotnet_checkpoint():
    ref = tiny_rotnet(seed=1)
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'rot.pt'
        torch.save(dict(model=ref.state_dict(), ema=ref.state_dict(), config=dict(img_size=56, canvas=56, hidden=1024, arch='t')), f)
        ck = torch.load(f, map_location='cpu', weights_only=False)
    m = ee.EarlyExitRotNet(exits=EXITS, img_size=56, pretrained=False, dynamic=True, arch='t', drop_path=0.0)
    loaded, fresh, ignored = ee.load_init(m, ck['ema'])
    assert loaded == [] and fresh == list(EXITS) and ignored == []
    m.eval(); x = torch.randn(4, 3, 56, 56)
    with torch.no_grad():
        r, a, b = ref(x), m(x), m(x, all_exits=True)[-1]
    assert torch.equal(r, a), (r - a).abs().max()
    assert torch.allclose(r, b, atol=1e-6, rtol=0), (r - b).abs().max()
    # an early-exit checkpoint with a different exit set: shared exits load, others stay fresh, extra ones are ignored
    m2 = ee.EarlyExitRotNet(exits=(4, 7), img_size=56, pretrained=False, dynamic=True, arch='t')
    loaded, fresh, ignored = ee.load_init(m2, tiny_ee().state_dict())
    assert loaded == [4] and fresh == [7] and ignored and all(k.startswith(('exits.b02', 'exits.b06')) for k in ignored)
    # backbone mismatch fails closed
    bad = dict(ck['ema']); bad.pop('head.1.weight')
    try:
        ee.load_init(m2, bad); raise AssertionError('incomplete head accepted')
    except SystemExit:
        pass


def test_exit_init_from_final_and_strip():
    m = tiny_ee(seed=2)
    m.init_exits_from_final([4])
    t = torch.randn(2, 17, 64)
    with torch.no_grad():
        assert torch.allclose(m.exits['b04'](t), m.final(t), atol=1e-6)
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / 'ee.pt', Path(d) / 'stripped.pt'
        cfg = dict(img_size=56, canvas=56, hidden=1024, arch='t', early_exit=ee.ee_config(m))
        torch.save(dict(model=m.state_dict(), ema=m.state_dict(), config=cfg), src)
        ee.strip([str(src), str(dst)])
        ck = torch.load(dst, map_location='cpu', weights_only=False)
        r = rmodel.RotNet(img_size=56, pretrained=False, dynamic=True, arch='t'); r.load_state_dict(ck['ema'])   # strict
        assert 'early_exit' not in ck['config'] and len(ck['stripped_from']['sha256']) == 64
        m3, cfg3 = ee.build_from_ckpt(torch.load(src, map_location='cpu', weights_only=False))
        assert m3.exit_at == EXITS
        x = torch.randn(1, 3, 56, 56)
        with torch.no_grad():
            assert torch.equal(r.eval()(x), m.eval()(x))


def test_loss_and_phases():
    th = torch.tensor([10.0, 200.0, 359.0])
    zf = torch.randn(3, 360, requires_grad=True)
    tgt = ee.cgd_batch(th, 6.0)
    ce = torch.sum(-tgt * torch.log_softmax(zf, 1), 1).mean()
    loss, parts = ee.ee_loss([zf, zf, zf], th, 6.0, [0.5, 2.0], 1.0, alpha=1.0)
    assert all(float(k) < 1e-6 for k in parts['kl'])
    assert torch.allclose(loss, 3.5 * ce, atol=1e-5)
    # the final head is a teacher only through a detached posterior: with final_weight 0 it gets gradient only as an exit input
    z1 = torch.randn(3, 360, requires_grad=True); zf2 = torch.randn(3, 360, requires_grad=True)
    loss, _ = ee.ee_loss([z1, zf2], th, 6.0, [1.0], final_weight=0.0, alpha=1.0)
    loss.backward()
    assert zf2.grad is None or float(zf2.grad.abs().max()) == 0.0
    assert float(z1.grad.abs().max()) > 0
    # KL term: alpha scales it exactly
    l0, p0 = ee.ee_loss([z1.detach(), zf2.detach()], th, 6.0, [1.0], alpha=0.0)
    l1, p1 = ee.ee_loss([z1.detach(), zf2.detach()], th, 6.0, [1.0], alpha=1.0)
    assert torch.allclose(l1 - l0, p1['kl'][0], atol=1e-5) and float(p1['kl'][0]) > 0
    # phases
    m = tiny_ee()
    tr = ee.set_phase(m, heads_only=True)
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    assert names and all(n.startswith('exits.') for n in names) and len(tr) == len(names)
    assert not m.backbone.training and m.exits.training
    x = torch.randn(2, 3, 56, 56)
    outs = m(x, all_exits=True)
    assert not outs[-1].requires_grad and all(o.requires_grad for o in outs[:-1])
    loss, _ = ee.ee_loss(outs, th[:2], 6.0); loss.backward()
    assert all(p.grad is None for n, p in m.named_parameters() if not n.startswith('exits.'))
    tr = ee.set_phase(m, heads_only=False)
    assert len(tr) == len(list(m.parameters())) and m.backbone.training
    # exit_grad = 0: exit losses do not reach the backbone
    m = tiny_ee(exit_grad=0.0); ee.set_phase(m, False); m.zero_grad()
    outs = m(x, all_exits=True)
    ee.ee_loss(outs, th[:2], 6.0, final_weight=0.0)[0].backward()
    assert all(p.grad is None or float(p.grad.abs().max()) == 0 for p in m.backbone.parameters())
    # exit_grad = 0.5 halves the backbone gradient of the exit loss exactly
    grads = []
    for s in (1.0, 0.5):
        m = tiny_ee(exit_grad=s); ee.set_phase(m, False); m.backbone.eval()
        outs = m(x, all_exits=True)
        ee.ee_loss(outs, th[:2], 6.0, final_weight=0.0, alpha=0.0)[0].backward()
        grads.append(m.backbone.blocks[0].mlp.fc1.weight.grad.clone())
    assert torch.allclose(grads[1], 0.5 * grads[0], atol=1e-7, rtol=1e-4)
    # gradient checkpointing gives the same gradients
    gs = []
    for ck in (False, True):
        m = tiny_ee(seed=5); ee.set_phase(m, False); m.backbone.eval(); m.grad_ckpt = ck
        ee.ee_loss(m(x, all_exits=True), th[:2], 6.0)[0].backward()
        gs.append(m.backbone.blocks[1].attn.qkv.weight.grad.clone())
    assert torch.allclose(gs[0], gs[1], atol=1e-6)


def test_policy_arithmetic():
    blocks = [6, 12, 18, 24]
    conf = np.array([[.95, .10, .10, .10, .85],    # exit 6
                     [.99, .95, .10, .10, .99],    # exit 12
                     [.99, .99, .95, .10, .99],    # exit 18
                     [.50, .50, .50, .50, .50]])   # final (never compared)
    err = np.array([[1, 50, 50, 50, 2], [1, 3, 50, 50, 2], [1, 3, 4, 50, 2], [1, 3, 4, 5, 2]], float)
    ex, s = ee.simulate(conf, err, blocks, [0.9, 0.9, 0.9], ms_block=10.0)
    assert ex.tolist() == [0, 1, 2, 3, 1]
    assert s['w10'] == 5 and s['shares'] == [0.2, 0.4, 0.2, 0.2]
    costs = np.array([60, 120, 180, 240, 120.0])
    assert s['model_mean'] == round(costs.mean(), 1) == 144.0
    assert s['model_p90'] == 240.0                               # nearest-rank p90 of 5 views = the 5th smallest
    assert s['e2e_mean'] == round(ee.E2E_SCALE * 144.0 + ee.E2E_ADD, 1)
    ex, s = ee.simulate(conf, err, blocks, [0.8, math.inf, math.inf], ms_block=10.0)
    assert ex.tolist() == [0, 3, 3, 3, 0] and s['w10'] == 5 and s['model_mean'] == (60 + 240 * 3 + 60) / 5
    ex, s = ee.simulate(conf, err, blocks, [0.0, 0.0, 0.0], ms_block=10.0)   # everything leaves at the first exit
    assert (ex == 0).all() and s['w10'] == 2 and s['gt90'] == 0 and s['model_p90'] == 60.0
    ex, s = ee.simulate(conf, err, blocks, [0.9, 0.9, 0.9], ms_block=10.0, exit_ms=1.0)
    assert s['model_mean'] == round(np.mean([61, 122, 183, 244, 122]), 1)
    # agreement rule: exit j also needs |pred_j - pred_{j-1}| <= 10; the first exit never agrees
    pred = np.array([[0, 90, 0, 0, 0], [5, 0, 180, 0, 0], [8, 0, 0, 0, 0], [0, 0, 0, 0, 0]], float)
    ex = ee.exit_index(conf, [0.9, 0.9, 0.9], pred, agree=True)
    assert ex.tolist() == [1, 2, 3, 3, 1]
    # tails
    err2 = err.copy(); err2[3, 3] = 170.0
    assert ee.simulate(conf, err2, blocks, [0.9] * 3, ms_block=10.0)[1]['t150'] == 1
    # calibration: per-block cost = total / 24, interpolation between calibrated canvases
    assert ee.ms_full('l', 224) == 598.0 and 347.0 < ee.ms_full('l', 196) < 598.0
    # sweep / pareto / best
    res = ee.sweep(conf, err, blocks, [0.9, math.inf], ee.cum_ms_of(blocks, 10.0))
    assert len(res) == 8
    best = ee.best_policy(res, max_mean=170.0, max_p90=1e9, jitter=1.0)    # cheapest all-correct policy: e2e 1.1 * 144 + 2
    assert best[0] == (0.9, 0.9, 0.9) and best[1]['w10'] == 5 and best[1]['e2e_mean'] == 160.4
    assert ee.best_policy(res, max_mean=150.0, max_p90=1e9) is None
    front = ee.pareto(res)
    assert all(a[1]['w10'] < b[1]['w10'] for a, b in zip(front, front[1:]))


def _fake_views(n=10, seed=0):
    from PIL import Image
    r = np.random.default_rng(seed); pngs = []
    for _ in range(n):
        h, w = int(r.integers(40, 90)), int(r.integers(40, 90))
        b = io.BytesIO(); Image.fromarray(r.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(b, format='PNG'); pngs.append(b.getvalue())
    return pngs, r.uniform(0, 360, n)


def test_eval_store_roundtrip_and_policy():
    m = tiny_ee(seed=3)
    pngs, th = _fake_views(12)
    with tempfile.TemporaryDirectory() as d:
        ee.eval_views(m, pngs, th, None, 'fake', 'viewsha', 'cksha', [56], 'tiny', d, bs=5)
        ee.eval_views(m, pngs[:7], th[:7], np.arange(7), 'fake2', 'viewsha2', 'cksha', [56], 'tiny', d, bs=5)
        assert sorted(p.name for p in Path(d).glob('*.json')) == [f'tiny-e0{k}-c56.json' for k in (2, 4, 6, 8)]
        blocks, conf, err, pred, lab = ee.load_exits(d, 'tiny', 56, ['fake', 'fake2'])
        assert blocks == [2, 4, 6, 8] and conf.shape == (4, 19) and (lab == 'fake2').sum() == 7
        from PIL import Image
        from rotlab.core import letterbox
        x = np.stack([letterbox(Image.open(io.BytesIO(b)).convert('RGB'), 56) for b in pngs])
        probs = ee.exit_probs(m, x, bs=4)
        for j in range(4):
            p = decode(probs[j])
            assert np.allclose(pred[j, :12], p, atol=1e-6)
            assert np.allclose(conf[j, :12], confidence(probs[j], p), atol=1e-6)
            assert np.allclose(err[j, :12], circ_err(p, th), atol=1e-6)
        sub = ee.load_exits(d, 'tiny', 56, ['fake'], use=[4])
        assert sub[0] == [4, 8] and sub[1].shape == (2, 12)
        try:
            ee.load_exits(d, 'tiny', 56, ['fake'], use=[3]); raise AssertionError('missing exit accepted')
        except SystemExit:
            pass
        r = ee.policy(['tiny', '--canvas', '56', '--sets', 'fake,fake2', '--store', d, '--ms-full', '80', '--thresholds', '0.5,off,0.2'])
        ex, s = ee.simulate(conf, err, blocks, [0.5, math.inf, 0.2], ms_block=10.0)
        assert r['sets']['pooled']['w10'] == s['w10'] and r['sets']['pooled']['model_mean'] == s['model_mean']
        r = ee.policy(['tiny', '--canvas', '56', '--sets', 'fake,fake2', '--select-sets', 'fake', '--store', d, '--ms-full', '80',
                       '--sweep', '--grid', '0.1,0.5,off', '--max-mean', '1e9', '--max-p90', '1e9'])
        assert 'pareto' in r and 'pooled' in r['sets']
        # hybrid: a stored first-stage tag (stand-in for S2) routes views into the early-exit network when its conf < thr
        from rotlab.camp_store import commit
        r0 = np.random.default_rng(9); c0 = r0.uniform(0, 1, 19); p0 = r0.uniform(0, 360, 19); th_all = np.concatenate([th, th[:7]])
        for s_, sl in (('fake', slice(0, 12)), ('fake2', slice(12, 19))):
            commit(d, 'first-c56', {f'{s_}_conf': c0[sl], f'{s_}_pred': p0[sl], f'{s_}_theta': th_all[sl]}, dict(x=1), {s_: 'v'})
        r = ee.policy(['tiny', '--canvas', '56', '--sets', 'fake,fake2', '--store', d, '--ms-full', '80', '--thresholds', '0.5,off,0.2',
                       '--first', 'first-c56', '--first-thr', '0.6', '--first-ms', '7'])
        cf = np.vstack([c0[None], conf]); ef = np.vstack([circ_err(p0, th_all)[None], err])
        cum = np.concatenate([[7.0], 7.0 + ee.cum_ms_of(blocks, 10.0)])
        ex, s = ee.simulate(cf, ef, None, [0.6, 0.5, math.inf, 0.2], cum_ms=cum)
        assert r['sets']['pooled']['w10'] == s['w10'] and r['sets']['pooled']['model_mean'] == s['model_mean']
        assert r['sets']['pooled']['shares'][0] == round(float((c0 >= 0.6).mean()), 4)
        casc = r['sets']['pooled']['cascade_first_to_final']
        assert casc['shares'][0] + casc['shares'][-1] == 1.0


def test_segmented_onnx_matches_pytorch():
    import onnxruntime  # noqa: F401
    m = tiny_ee(seed=4)
    with tempfile.TemporaryDirectory() as d:
        cfg = dict(img_size=56, canvas=56, hidden=1024, arch='t', early_exit=ee.ee_config(m))
        torch.save(dict(model=m.state_dict(), ema=m.state_dict(), config=cfg), Path(d) / 'ee.pt')
        for cv in (56, 70):    # 70: position embedding resampled and baked for a static graph
            out = Path(d) / f'onnx{cv}'
            ee.export([str(Path(d) / 'ee.pt'), str(out), '--canvas', str(cv), '--full'])
            rt = ee.EarlyExitORT(out, threads=2)
            assert rt.blocks == [2, 4, 6, 8] and len(rt.sess) == 4
            x = np.random.default_rng(cv).standard_normal((1, 3, cv, cv)).astype(np.float32)
            ref = ee.exit_probs(m, x)
            for j in range(4):
                prob, k, conf, pred = rt(x, [0.5] * 3, stop_at=j)
                assert k == rt.blocks[j]
                assert np.abs(prob - ref[j]).max() < 1e-5, (cv, j, np.abs(prob - ref[j]).max())
                assert abs(conf - float(confidence(ref[j], decode(ref[j]))[0])) < 1e-4
            full = rt.full.run(None, {'image': x})[0]
            assert np.abs(full - ref[-1]).max() < 1e-5
            # policy decisions follow the confidences: exit at the first passing exit, else the final
            assert rt(x, [0.9, 0.9, 0.9], conf_override=[0.1, 0.95, 0.99, 0])[1] == 4
            assert rt(x, [0.9, 0.9, 0.9], conf_override=[0.1, 0.1, 0.1, 0])[1] == 8
            assert rt(x, [math.inf] * 3, conf_override=[1, 1, 1, 1])[1] == 8
            p, k, c, _ = rt(x, [0.0, 0.0, 0.0])
            assert k == 2 and abs(c - float(confidence(ref[0], decode(ref[0]))[0])) < 1e-4
        # exporting a subset of the trained exits
        ee.export([str(Path(d) / 'ee.pt'), str(Path(d) / 'sub'), '--canvas', '56', '--exits', '4'])
        rt = ee.EarlyExitORT(Path(d) / 'sub', threads=2)
        x = np.random.default_rng(0).standard_normal((1, 3, 56, 56)).astype(np.float32); ref = ee.exit_probs(m, x)
        assert rt.blocks == [4, 8] and np.abs(rt(x, [0.0])[0] - ref[1]).max() < 1e-5
        # the benchmark runs end to end on the tiny graphs (synthetic confidences)
        r = ee.bench([str(Path(d) / 'onnx56'), '--thresholds', '0.9,0.9,0.9', '--n', '20', '--rounds', '2', '--threads', '2', '--arch', 't'])
        assert abs(sum(r['shares']) - 1) < 1e-6 and r['policy_raw']['mean'] > 0 and r['canonical_scale'] == 1.0


class FakeViews(torch.utils.data.IterableDataset):
    """Stands in for train_ddp.Views (same constructor): random letterboxed-size tensors + angles."""

    def __init__(self, mix, size, seed, frac=1.0, maxarea_p=0.0):
        self.size, self.seed = size, seed

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        while True:
            yield torch.randn(3, self.size, self.size, generator=g), torch.rand((), generator=g) * 360


def test_train_loop_cpu_smoke():
    """Two-phase loop end to end on CPU (real DataLoader worker): heads-only phase leaves backbone + final head exactly at the
    init, joint phase moves them; milestone/phase/final checkpoints and config are written."""
    import rotlab.train_ddp as td
    ref = tiny_rotnet(seed=6)
    old = td.RUNS, td.Views
    with tempfile.TemporaryDirectory() as d:
        init = Path(d) / 'init.pt'
        torch.save(dict(model=ref.state_dict(), ema=ref.state_dict(), config=dict(img_size=56, arch='t')), init)
        td.RUNS, td.Views = Path(d), FakeViews
        try:
            ee.train(['--run', 'smoke', '--mix', 'a=1', '--arch', 't', '--img-size', '56', '--init', str(init), '--exits', '2,4,6',
                      '--exit-weights', '0.5,1,1', '--steps', '6', '--freeze-steps', '3', '--batch', '4', '--workers', '1',
                      '--save-every', '2', '--keep-milestones', '--warmup', '2', '--lr', '1e-3', '--multires', '42,56', '--cpu-smoke'])
        finally:
            td.RUNS, td.Views = old
        out = Path(d) / 'smoke'
        assert {p.name for p in out.glob('*.pt')} == {'phase1.pt', 'last.pt', 'ckpt-000002.pt', 'ckpt-000004.pt', 'final.pt'}
        p1 = torch.load(out / 'phase1.pt', map_location='cpu', weights_only=False)
        fin = torch.load(out / 'final.pt', map_location='cpu', weights_only=False)
        r = ref.state_dict()
        for w in ('model', 'ema'):
            assert all(torch.equal(p1[w][k], v) for k, v in r.items()), 'heads-only phase moved the backbone/final head'
        assert not torch.equal(p1['model']['exits.b02.head.1.weight'], fin['model']['exits.b02.head.1.weight'])
        assert not torch.equal(fin['model']['backbone.blocks.0.attn.qkv.weight'], r['backbone.blocks.0.attn.qkv.weight'])
        assert fin['config']['early_exit'] == dict(exits=[2, 4, 6], exit_hidden=1024, depth=8) and fin['step'] == 6
        m, _ = ee.build_from_ckpt(fin)
        assert m.exit_at == (2, 4, 6)


if __name__ == '__main__':
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn(); print('PASS', name, flush=True)
            except Exception as e:   # noqa: BLE001
                fails += 1; print('FAIL', name, repr(e), flush=True)
    sys.exit(1 if fails else 0)
