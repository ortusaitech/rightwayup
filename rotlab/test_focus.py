#!/usr/bin/env python3
"""CPU tests for rotlab.focus: tiny random ViT-S (2 blocks, LayerScale re-randomised so blocks matter), no data, no GPU.
  pod:   cd /workspace && PYTHONPATH=code python -m pytest -q code/rotlab/test_focus.py     (or: python -m rotlab.test_focus)
  local: PYTHONPATH=code python -m pytest -q rotlab/test_focus.py
"""
import io, os, pickle, subprocess, sys, tempfile, unittest, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import rotlab
HERE = str(Path(__file__).resolve().parent)
if HERE not in [str(Path(p).resolve()) for p in rotlab.__path__]:   # local checkout: lead files sit beside code/rotlab
    rotlab.__path__.append(HERE)
from rotlab import focus as fx
from rotlab.core import cgd_target, letterbox
from rotlab.model import RotNet

warnings.filterwarnings('ignore')
ASPECTS = [1.0, 4 / 3, 16 / 9, 3 / 4, 2 / 3, 5 / 3, 3.0, 1 / 3, 1.01, 7 / 5]


def tiny(depth=2, seed=0, img_size=224):
    torch.manual_seed(seed)
    net = RotNet(img_size=img_size, pretrained=False, dynamic=True, arch='s', drop_path=0.0).eval()
    net.backbone.blocks = net.backbone.blocks[:depth]
    with torch.no_grad():    # DINOv2 LayerScale starts at 1e-5: without this the blocks (and these tests) are nearly inert
        for blk in net.backbone.blocks:
            blk.ls1.gamma.normal_(1, .3); blk.ls2.gamma.normal_(1, .3)
        net.backbone.pos_embed.normal_(0, .5)
    return net


def boxed(aspects, canvas=224, seed=0):
    """Noise images of the given aspects, letterboxed exactly like the eval path -> (B,3,C,C), [(w, h)]."""
    rng = np.random.default_rng(seed); xs, sizes = [], []
    for a in aspects:
        w = int(rng.integers(150, 400)); h = max(8, round(w / a))
        xs.append(letterbox(Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)), canvas)); sizes.append((w, h))
    return torch.from_numpy(np.stack(xs)), sizes


def timm_drop_reference(net, x, keep):
    """Independent E0 reference through timm's own patch_embed/_pos_embed: select CLS + kept tokens, then blocks/head."""
    bb = net.backbone
    t = bb._pos_embed(bb.patch_embed(x))
    t = t[:, torch.cat([torch.zeros(1, dtype=torch.long), keep.nonzero()[:, 0] + 1])]
    t = bb.norm(bb.blocks(bb.norm_pre(t)))
    return net.head(torch.cat([t[:, 0], t[:, 1:].mean(1)], 1))


def cell_of(i, g):
    """Bank index -> (level, row, col) on the native g x g grid layout [native | 28 px | 56 px]."""
    for lvl in range(3):
        gl = g >> lvl
        if i < gl * gl:
            return lvl, i // gl, i % gl
        i -= gl * gl
    raise IndexError


class FillTokens(unittest.TestCase):
    def test_fill_mask_matches_letterbox_geometry(self):
        for canvas in (224, 168, 140):
            x, sizes = boxed(ASPECTS, canvas)
            got = fx.fill_mask(x).numpy()
            for k, (w, h) in enumerate(sizes):
                np.testing.assert_array_equal(got[k], fx.letterbox_fill_mask(w, h, canvas), err_msg=f'{canvas} {w}x{h}')

    def test_known_token_counts(self):
        for (w, h), n in (((320, 240), 192), ((320, 180), 160), ((240, 320), 192), ((300, 300), 256)):
            x = torch.from_numpy(letterbox(Image.new('RGB', (w, h), (10, 200, 30)), 224))[None]
            self.assertEqual(int((~fx.fill_mask(x)).sum()), n)

    def test_fill_mask_survives_resize(self):       # multires training path: bilinear antialias resize of a letterbox
        x, _ = boxed([16 / 9], 224)
        xr = F.interpolate(x, size=(168, 168), mode='bilinear', antialias=True, align_corners=False)
        m = fx.fill_mask(xr)[0]
        self.assertTrue(m[:2].all() and m[-2:].all() and not m[4:8].any())

    def test_full_plan_equals_rotnet(self):
        net = tiny()
        for canvas in (224, 140):                   # 140 exercises the timm PE resample path
            x = torch.randn(2, 3, canvas, canvas)
            with torch.no_grad():
                self.assertLess((fx.FocusNet(net)(x) - net(x)).abs().max().item(), 1e-6)

    def test_drop_is_identity_without_fill(self):
        net = tiny(); x, _ = boxed([1.0, 1.0])
        self.assertFalse(fx.fill_mask(x).any())
        with torch.no_grad():
            ref = net(x)
            for plan in ('fill', 'fillmask'):
                self.assertLess((fx.FocusNet(net, plan=plan)(x) - ref).abs().max().item(), 1e-5, plan)

    def test_drop_matches_timm_reference_mixed_batch(self):
        net = tiny(); x, _ = boxed(ASPECTS)
        keep = ~fx.fill_mask(x).flatten(1)
        with torch.no_grad():
            ref = torch.cat([timm_drop_reference(net, x[k:k + 1], keep[k]) for k in range(len(x))])
            grouped = fx.FocusNet(net, plan='fill')(x)
            masked = fx.FocusNet(net, plan='fillmask')(x)
            single = torch.cat([fx.FocusNet(net, plan='fill')(x[k:k + 1]) for k in range(len(x))])
            full = net(x)
        self.assertLess((grouped - ref).abs().max().item(), 1e-5)
        self.assertLess((masked - ref).abs().max().item(), 1e-4)
        self.assertLess((grouped - single).abs().max().item(), 1e-5)
        self.assertGreater((grouped - full)[1:4].abs().max().item(), 1e-3)     # fill present -> dropping changes the output

    def test_sinks(self):
        net = tiny(); x, _ = boxed([16 / 9, 1.0])
        m = fx.FocusNet(net, plan='fill', sinks=4)
        n_content = (~fx.fill_mask(x)).flatten(1).sum(1)
        self.assertEqual(m.keep_mask(x).sum(1).tolist(), [int(n_content[0]) + 4, 256])


class FocusTokens(unittest.TestCase):
    def test_quad_counts(self):
        self.assertEqual([fx.quad_counts(b, 16) for b in (16, 34, 40, 49, 58, 64, 256)],
                         [(0, 0), (3, 3), (4, 4), (6, 5), (7, 7), (8, 8), (16, 64)])
        for bad in (41, 15, 259):
            with self.assertRaises(ValueError):
                fx.quad_counts(bad, 16)

    def test_quad_is_an_exact_cover_with_salient_splits(self):
        x, _ = boxed([16 / 9, 1.0, 3 / 4], seed=3)
        fill = fx.fill_mask(x)
        sal = torch.rand(3, 16, 16)
        for budget in (16, 19, 34, 40, 49, 58, 64, 256):
            idx = fx.select_quad(sal, fill, budget)
            k0, k1 = fx.quad_counts(budget, 16)
            self.assertEqual(idx.shape, (3, budget))
            for b in range(3):
                self.assertEqual(len(set(idx[b].tolist())), budget)
                cover = np.zeros((16, 16), int); lv = [0, 0, 0]
                for i in idx[b].tolist():
                    lvl, r, c = cell_of(i, 16); f = 2 ** lvl; lv[lvl] += 1
                    cover[r * f:(r + 1) * f, c * f:(c + 1) * f] += 1
                self.assertTrue((cover == 1).all(), f'budget {budget}: not an exact cover')
                self.assertEqual(lv, [4 * k1, 4 * k0 - k1, 16 - k0])
                s = fx._norm_sal(sal[b:b + 1], fill[b:b + 1])[0]
                split = {r * 4 + c for i in idx[b].tolist() for lvl, r, c in [cell_of(i, 16)] if lvl < 2
                         for r, c in [(r // (4 >> lvl), c // (4 >> lvl))]}
                top = set(F.avg_pool2d(s[None, None], 4).flatten().topk(k0).indices.tolist())
                self.assertEqual(split, top)

    def test_quad_full_split_equals_full_model(self):
        net = tiny(); x, _ = boxed([1.0, 4 / 3])
        with torch.no_grad():
            got = fx.FocusNet(net, plan='quad')(x, torch.rand(2, 16, 16), 256)
            self.assertLess((got - net(x)).abs().max().item(), 1e-5)

    def test_bank_tokens_and_position_embeddings(self):
        net = tiny(); m = fx.FocusNet(net)
        with torch.no_grad():
            m.scale_emb.normal_()
            x = torch.randn(1, 3, 224, 224)
            bank, area = m.bank(x)
            bb = net.backbone
            pe = bb.pos_embed[0, 1:].reshape(16, 16, -1)
            for i in (0, 37, 255, 256, 256 + 45, 320, 320 + 9, 335):
                lvl, r, c = cell_of(i, 16); f = 2 ** lvl
                xe = F.avg_pool2d(x, f) if f > 1 else x
                emb = bb.patch_embed.proj(xe[:, :, r * 14:(r + 1) * 14, c * 14:(c + 1) * 14]).flatten()
                want = emb + pe[r * f:(r + 1) * f, c * f:(c + 1) * f].mean((0, 1)) + m.scale_emb[lvl]
                self.assertLess((bank[0, i] - want).abs().max().item(), 1e-4, i)
                self.assertEqual(area[i].item(), 4.0 ** lvl)

    def test_glance_and_fill_are_never_refined(self):
        x, _ = boxed([16 / 9, 3.0], seed=5)
        fill = fx.fill_mask(x)
        sal = torch.rand(2, 16, 16) + 10 * fill.float()          # saliency deliberately peaks on the letterbox
        g = fx.select_glance(sal, fill, 40)
        for b in range(2):
            fine = [i for i in g[b].tolist() if i < 256]
            self.assertEqual(len(fine), 24)
            self.assertFalse(fill[b].flatten()[fine].any())
            s = sal[b].flatten().clone(); s[fill[b].flatten()] = -1
            self.assertEqual(set(fine), set(s.topk(24).indices.tolist()))
            self.assertEqual(sorted(i for i in g[b].tolist() if i >= 320), list(range(320, 336)))
            k1 = fx.quad_counts(49, 16)[1]
            fine_q = torch.tensor(fx.select_quad(sal, fill, 49)[b, :4 * k1].tolist()).view(k1, 4)   # children of split 28-px cells
            self.assertFalse(fill[b].flatten()[fine_q].all(1).any())

    def test_uniform_equals_rotnet_at_small_canvas(self):
        net = tiny(); x, _ = boxed([4 / 3, 1.0])
        with torch.no_grad():
            ref = net(F.interpolate(x, size=(98, 98), mode='bilinear', antialias=True, align_corners=False))
            self.assertLess((fx.FocusNet(net, plan='uniform', uniform=7)(x) - ref).abs().max().item(), 1e-6)

    def test_s2_cls_attention_matches_explicit(self):
        net = tiny(); x, _ = boxed([4 / 3, 1.0])
        with torch.no_grad():
            lg, sal = fx.S2Saliency(net)(x)
            bb = net.backbone
            t = bb._pos_embed(bb.patch_embed(x))
            for blk in bb.blocks[:-1]:
                t = blk(t)
            a = bb.blocks[-1].attn; y = bb.blocks[-1].norm1(t)
            B, N, D = y.shape; H = a.num_heads
            qkv = a.qkv(y).reshape(B, N, 3, H, D // H).permute(2, 0, 3, 1, 4)
            w = ((qkv[0] * a.scale) @ qkv[1].transpose(-2, -1)).softmax(-1)[:, :, 0, 1:].mean(1)
        self.assertLess((sal.flatten(1) - w).abs().max().item(), 1e-6)
        self.assertLess((lg - net(x)).abs().max().item(), 1e-6)
        _, gs = fx.S2Saliency(tiny(), 'grad')(x)
        self.assertEqual(gs.shape, (2, 16, 16)); self.assertTrue((gs >= 0).all() and gs.sum() > 0)

    def test_pixel_saliency_ignores_letterbox_edge(self):
        x = torch.from_numpy(letterbox(Image.new('RGB', (320, 180), (200, 30, 30)), 224))[None]   # flat content
        self.assertEqual(fx.pixel_saliency(x).abs().max().item(), 0.0)

    def test_budget_policy(self):
        pol = fx.parse_policy('1.01:34,0.70:49')
        self.assertEqual(pol, [(0.70, 49), (1.01, 34)])
        self.assertEqual(fx.policy_budget(np.array([0.1, 0.69, 0.70, 0.9, 1.0]), pol).tolist(), [49, 49, 34, 34, 34])

    def test_loss_backward_reaches_all_levels(self):
        net = tiny(); m = fx.FocusNet(net, plan='quad').train()
        x, _ = boxed([4 / 3, 1.0])
        th = torch.tensor([10.0, 200.0])
        target = torch.from_numpy(np.stack([cgd_target(float(t)) for t in th]))
        with torch.no_grad():
            t_logits = net(x)
        loss = fx.focus_loss(m(x, torch.rand(2, 16, 16), 49), target, t_logits, alpha=0.5)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue((m.scale_emb.grad.abs().sum(1) > 0).all())
        self.assertGreater(net.backbone.blocks[0].attn.qkv.weight.grad.abs().sum().item(), 0)


class Onnx(unittest.TestCase):
    def check(self, mod, inputs, names, outs=('prob',), tol=1e-5):
        with torch.no_grad():
            ref = mod(*inputs)
        ref = ref if isinstance(ref, tuple) else (ref,)
        data = fx.export_onnx(mod, inputs, names, outputs=outs)
        got = fx.ort_session(data, 2).run(None, {k: v.numpy() for k, v in zip(names, inputs)})
        for g, r in zip(got, ref):
            self.assertLess(float(np.abs(g - r.numpy()).max()), tol)
        return data

    def test_focus_graphs_match_torch(self):
        net = tiny(); x = fx.sample_input(224, 16 / 9); sal = torch.rand(1, 16, 16)
        for plan, budget in (('quad', 40), ('quad', 49), ('glance', 40)):
            self.check(fx.ProbGraph(fx.FocusNet(net, plan=plan, budget=budget), 'attn'), [x, sal], ['image', 'saliency'])
        self.check(fx.ProbGraph(fx.FocusNet(net, plan='quad', budget=34), 'pixel'), [x], ['image'])
        self.check(fx.S2Graph(fx.S2Saliency(tiny(seed=1))), [x], ['image'], outs=('prob', 'saliency'))

    def test_fill_graph_is_dynamic(self):
        net = tiny(); m = fx.FocusNet(net, plan='fill').eval()
        data = self.check(fx.ProbGraph(m), [fx.sample_input(224, 4 / 3)], ['image'])
        s = fx.ort_session(data, 2)
        for asp in (16 / 9, 1.0, 2 / 3):
            x = fx.sample_input(224, asp, seed=7)
            with torch.no_grad():
                ref = m(x).softmax(1).numpy()
            self.assertLess(float(np.abs(s.run(None, {'image': x.numpy()})[0] - ref).max()), 1e-5)

    def test_export_cli(self):
        with tempfile.TemporaryDirectory() as d:
            fx.main(['export', 's', '--out', f'{d}/fl.onnx', '--plan', 'quad', '--budget', '40', '--s2', 's'])
            self.assertTrue(Path(f'{d}/fl.onnx').exists() and Path(f'{d}/fl-s2sal.onnx').exists())


class EvalCli(unittest.TestCase):
    """camp_eval-compatible store output from `focus eval` on a fake view cache (subprocess, fresh ROTLAB_DATA)."""

    def test_eval_modes_commit_to_store(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d)
            (data / 'rotlab/cviews').mkdir(parents=True)
            rng = np.random.default_rng(0); pngs = []
            for a in (1.0, 4 / 3, 16 / 9, 3 / 4, 4 / 3, 2 / 3):
                w = 300; h = round(w / a); b = io.BytesIO()
                Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(b, format='PNG'); pngs.append(b.getvalue())
            (data / 'rotlab/cviews/fair.pkl').write_bytes(pickle.dumps(dict(png=pngs, theta=list(rng.uniform(0, 360, 6)))))
            torch.manual_seed(0)
            net = RotNet(img_size=224, pretrained=False, dynamic=True, arch='s', drop_path=0.0)
            (data / 'rotlab/runs/T').mkdir(parents=True)
            ck = data / 'rotlab/runs/T/final.pt'
            torch.save(dict(ema=net.state_dict(), config=dict(img_size=224, arch='s', hidden=1024)), ck)
            env = dict(os.environ, ROTLAB_DATA=str(data), PYTHONPATH=str(Path(rotlab.__path__[0]).parent))
            boot = f'import rotlab; rotlab.__path__.append({HERE!r}); from rotlab.focus import main; import sys; main(sys.argv[1:])'
            for extra in (['--mode', 'fill'], ['--mode', 'fillmask'], ['--mode', 'full'],
                          ['--mode', 'focus', '--plan', 'quad', '--budget', '40', '--scorer', 'attn', '--s2', str(ck)],
                          ['--mode', 'focus', '--scorer', 'pixel', '--budget-by-conf', '0.5:49,1.01:34', '--s2', str(ck)]):
                r = subprocess.run([sys.executable, '-c', boot, 'eval', str(ck), '--sets', 'fair', '--device', 'cpu', *extra],
                                   env=env, capture_output=True, text=True, timeout=600)
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            r = subprocess.run([sys.executable, '-c', boot, 'eval', str(ck), '--sets', 'newcoco_holdout_fixed', '--device', 'cpu'],
                               env=env, capture_output=True, text=True, timeout=600)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('refusing', r.stdout + r.stderr)
            from rotlab.camp_store import load
            out = data / 'rotlab/camp-eval'
            z = {t: load(out, t)[0] for t in ('T-fill-c224', 'T-fillmask-c224', 'T-tokfull-c224', 'T-focus-quad40-attn-c224')}
            want = [int((~fx.letterbox_fill_mask(300, round(300 / a), 224)).sum()) for a in (1.0, 4 / 3, 16 / 9, 3 / 4, 4 / 3, 2 / 3)]
            self.assertEqual(want, [256, 192, 160, 192, 192, 192])
            self.assertEqual(z['T-fill-c224']['fair_ntok'].tolist(), want)
            np.testing.assert_allclose(z['T-fill-c224']['fair_prob'], z['T-fillmask-c224']['fair_prob'], atol=2e-3)
            self.assertTrue((z['T-focus-quad40-attn-c224']['fair_budget'] == 40).all())
            conf_tags = [p.stem for p in out.glob('T-focus-quadconf*-pixel-c224.json')]
            self.assertEqual(len(conf_tags), 1)
            rows, rec = load(out, conf_tags[0])
            self.assertTrue(set(rows['fair_budget'].tolist()) <= {34, 49})
            self.assertEqual(rec['focus']['budget_by_conf'], '0.5:49,1.01:34')


if __name__ == '__main__':
    unittest.main()
