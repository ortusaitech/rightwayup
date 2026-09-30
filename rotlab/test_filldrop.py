#!/usr/bin/env python3
"""CPU tests for rotlab.filldrop: tiny random timm DINOv2 ViTs (S and L configs, 2 blocks, LayerScale / PE re-randomised
so the blocks matter), synthetic data, no GPU.
  pod:   cd /workspace && PYTHONPATH=code python -m pytest -q code/rotlab/test_filldrop.py    (or: python -m rotlab.test_filldrop)
  local: PYTHONPATH=code python -m pytest -q rotlab/test_filldrop.py
"""
import copy, io, json, os, pickle, socket, subprocess, sys, tempfile, unittest, warnings
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import rotlab
HERE = str(Path(__file__).resolve().parent)
if HERE not in [str(Path(p).resolve()) for p in rotlab.__path__]:   # local checkout: lead files sit beside code/rotlab
    rotlab.__path__.append(HERE)
from rotlab import filldrop as fd
from rotlab import focus as fx
from rotlab.core import letterbox
from rotlab.model_x import SPECS, RotNet
from rotlab.train_ddp import cgd_batch

warnings.filterwarnings('ignore')
ASPECTS = [1.0, 4 / 3, 16 / 9, 3 / 4, 2 / 3, 5 / 3, 3.0, 1 / 3, 1.01, 7 / 5, 16 / 9, 4 / 3]
_BASE = {}


def tiny(arch='s', depth=2, seed=0, img_size=224, drop_path=0.0):
    """RotNet (model_x) whose backbone is the arch's timm DINOv2 config truncated to `depth` blocks, random weights."""
    key = (arch, depth, seed, img_size, drop_path)
    if key not in _BASE:
        torch.manual_seed(seed)
        net = RotNet(img_size=img_size, pretrained=False, dynamic=True, arch='s', drop_path=drop_path)
        net.backbone = timm.create_model(SPECS[arch].timm, pretrained=False, img_size=img_size, num_classes=0, depth=depth,
                                         drop_path_rate=drop_path, dynamic_img_size=True)
        d = net.backbone.embed_dim
        net.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, 1024), nn.GELU(), nn.Linear(1024, 360))
        net.arch = arch
        with torch.no_grad():   # DINOv2 LayerScale starts at 1e-5: without this the blocks (and these tests) are nearly inert
            for blk in net.backbone.blocks:
                blk.ls1.gamma.normal_(1, .3); blk.ls2.gamma.normal_(1, .3)
            net.backbone.pos_embed.normal_(0, .5)
        _BASE[key] = net.eval()
    return copy.deepcopy(_BASE[key])


def boxed(aspects, canvas=224, seed=0):
    """Noise images of the given aspects, letterboxed exactly like the eval path -> (B,3,C,C)."""
    rng = np.random.default_rng(seed); xs = []
    for a in aspects:
        w = int(rng.integers(150, 400)); h = max(8, round(w / a))
        xs.append(letterbox(Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)), canvas))
    return torch.from_numpy(np.stack(xs))


def timm_reference(net, x, keep):
    """Independent per-image reference through timm's own patch_embed/_pos_embed/blocks: CLS + kept tokens only."""
    bb, out = net.backbone, []
    for k in range(len(x)):
        t = bb._pos_embed(bb.patch_embed(x[k:k + 1]))
        t = t[:, torch.cat([torch.zeros(1, dtype=torch.long), keep[k].nonzero()[:, 0] + 1])]
        t = bb.norm(bb.blocks(bb.norm_pre(t)))
        out.append(net.head(torch.cat([t[:, 0], t[:, 1:].mean(1)], 1)))
    return torch.cat(out)


def maxdiff(a, b):
    return (a - b).abs().max().item()


class Keep(unittest.TestCase):
    def test_sink_select_equals_focus_sink_mask_exhaustively(self):
        """Selection depends only on (n, #fill), so every #fill in 0..N at every n up to MAX_SINKS covers all cases."""
        g = torch.Generator().manual_seed(0); N = 320
        fill = torch.zeros(N + 1, N, dtype=torch.bool)
        for r in range(N + 1):
            fill[r, torch.randperm(N, generator=g)[:r]] = True
        for n in range(1, fd.MAX_SINKS + 1):
            self.assertTrue(torch.equal(fd.sink_select(fill, n), fx.sink_mask(fill, n)), n)

    def test_keep_mask_equals_focus(self):
        net = tiny()
        for canvas in (224, 140):
            x = boxed(ASPECTS, canvas)
            for s in (0, 1, 4, 7):
                a = fd.FillDropNet(net, sinks=s).keep_mask(x)
                b = fx.FocusNet(net, plan='fill', sinks=s).keep_mask(x)
                self.assertTrue(torch.equal(a, b), (canvas, s))
        x = torch.from_numpy(letterbox(Image.new('RGB', (320, 180), (10, 200, 30)), 224))[None]
        self.assertEqual(int(fd.FillDropNet(net).keep_mask(x).sum()), 160)
        self.assertEqual(int(fd.FillDropNet(net, sinks=4).keep_mask(x).sum()), 164)


class Forward(unittest.TestCase):
    def test_no_fill_equals_rotnet(self):
        for arch in ('s', 'l'):
            net = tiny(arch)
            for canvas in (224, 140, 112):            # 140/112 exercise the timm PE resample path
                x = boxed([1.0, 1.0, 1.0], canvas, seed=2)
                self.assertFalse(fx.fill_mask(x).any())
                with torch.no_grad():
                    ref = net(x)
                    for b in ('group', 'pad'):
                        self.assertLess(maxdiff(fd.FillDropNet(net, batching=b)(x), ref), 1e-5, (arch, canvas, b))

    def test_batched_equals_per_image_and_timm_reference(self):
        """group == pad == per-image (batch of 1) == independent timm reference == focus eval's FocusNet 'fill'."""
        for arch in ('s', 'l'):
            net = tiny(arch)
            for canvas in (224, 140):
                x = boxed(ASPECTS, canvas, seed=1)
                for s in (0, 4):
                    g, p = fd.FillDropNet(net, sinks=s), fd.FillDropNet(net, sinks=s, batching='pad')
                    with torch.no_grad():
                        yg, yp = g(x), p(x)
                        passes = g.stats[1]
                        single = torch.cat([g(x[k:k + 1]) for k in range(len(x))])
                        ref = timm_reference(net, x, g.keep_mask(x))
                        focus = fx.FocusNet(net, plan='fill', sinks=s)(x)
                        full = net(x)
                    tag = (arch, canvas, s)
                    self.assertLess(maxdiff(yg, single), 1e-5, tag)
                    self.assertLess(maxdiff(yg, ref), 1e-5, tag)
                    self.assertLess(maxdiff(yp, ref), 1e-4, tag)
                    self.assertLess(maxdiff(yg, focus), 1e-5, tag)
                    self.assertGreater(passes, 2, tag)                        # really several dense passes
                    has_fill = fx.fill_mask(x).flatten(1).any(1)
                    self.assertGreater(maxdiff(yg[has_fill], full[has_fill]), 1e-3, tag)     # dropping changes the output
                    self.assertLess(maxdiff(yg[~has_fill], full[~has_fill]), 1e-5, tag)

    def test_multires_resized_letterbox(self):
        """train_ddp --multires resizes the 224 letterbox (bilinear, antialias): drop stays exact at every size."""
        net = tiny()
        x224 = boxed(ASPECTS, 224, seed=4)
        for sz in (112, 168, 196):
            x = F.interpolate(x224, size=(sz, sz), mode='bilinear', antialias=True, align_corners=False)
            m = fd.FillDropNet(net, sinks=2)
            keep = m.keep_mask(x)
            self.assertLess(int(keep.sum()), keep.numel())                           # fill still detected after the resize
            with torch.no_grad():
                self.assertLess(maxdiff(m(x), timm_reference(net, x, keep)), 1e-5, sz)
                self.assertLess(maxdiff(fd.FillDropNet(net, sinks=2, batching='pad')(x), m(x)), 1e-4, sz)

    def test_all_fill_keeps_everything(self):
        net = tiny()
        x = torch.from_numpy(np.stack([letterbox(Image.new('RGB', (64, 64), (124, 116, 104)), 224)]))
        m = fd.FillDropNet(net)
        self.assertEqual(int(m.keep_mask(x).sum()), 256)
        with torch.no_grad():
            self.assertLess(maxdiff(m(x), net(x)), 1e-5)


class Grad(unittest.TestCase):
    def loss(self, m, x, th):
        return torch.sum(-cgd_batch(th, 6.0) * F.log_softmax(m(x).float(), 1), 1).mean()

    def grads(self, m):
        return {n: (p.grad.clone() if p.grad is not None else None) for n, p in m.net.named_parameters()}

    def test_gradients_equal_per_image_and_pad(self):
        net = tiny()
        x = boxed(ASPECTS[:8], 224, seed=5); th = torch.linspace(0, 350, len(x))
        out = {}
        for mode in ('group', 'pad', 'single'):
            m = fd.FillDropNet(copy.deepcopy(net), sinks=4, batching='pad' if mode == 'pad' else 'group').train()
            if mode == 'single':
                sum(self.loss(m, x[k:k + 1], th[k:k + 1]) / len(x) for k in range(len(x))).backward()
            else:
                self.loss(m, x, th).backward()
            self.assertIsNone(m.scale_emb.grad)
            out[mode] = self.grads(m)
        for n, g in out['group'].items():
            self.assertIsNotNone(g, n)
            scale = g.abs().max().item() + 1e-12
            self.assertLess(maxdiff(g, out['single'][n]) / scale, 1e-4, n)
            self.assertLess(maxdiff(g, out['pad'][n]) / scale, 1e-4, n)

    def test_dropped_tokens_get_no_gradient(self):
        """At canvas 224 the PE grid is used as is: PE rows of dropped fill cells receive exactly zero gradient; kept sinks
        and content cells receive gradient. Everything upstream of the head gets gradient; scale_emb stays frozen at 0."""
        for s in (0, 4):
            net = tiny()
            x = torch.from_numpy(np.stack([letterbox(Image.fromarray(np.random.default_rng(k).integers(0, 255, (180, 320, 3), dtype=np.uint8)), 224)
                                           for k in range(3)]))                      # 16:9: rows 0-2 and 13-15 are fill
            m = fd.FillDropNet(net, sinks=s).train()
            self.loss(m, x, torch.tensor([5.0, 100.0, 260.0])).backward()
            g = net.backbone.pos_embed.grad[0, 1:].abs().sum(-1).view(16, 16)
            keep = m.keep_mask(x[:1])[0].view(16, 16)
            self.assertEqual(int(keep.sum()), 160 + s)
            self.assertTrue((g[~keep] == 0).all(), s)
            self.assertTrue((g[keep] > 0).all(), s)
            self.assertGreater(net.backbone.pos_embed.grad[0, 0].abs().sum().item(), 0)
            for p in (net.backbone.cls_token, net.backbone.patch_embed.proj.weight, net.backbone.blocks[0].attn.qkv.weight,
                      net.backbone.norm.weight, net.head[-1].weight):
                self.assertGreater(p.grad.abs().sum().item(), 0)
            self.assertIsNone(m.scale_emb.grad)
            self.assertEqual(m.scale_emb.abs().sum().item(), 0.0)

    def test_arch_l_grad_ckpt_and_drop_path(self):
        net = tiny('l')
        x = boxed(ASPECTS[:6], 140, seed=6); th = torch.linspace(0, 300, len(x))
        grads = []
        for ck in (False, True):
            m = fd.FillDropNet(copy.deepcopy(net)).train(); m.grad_ckpt = ck
            self.loss(m, x, th).backward()
            grads.append(self.grads(m))
        for n, g in grads[0].items():
            self.assertLess(maxdiff(g, grads[1][n]) / (g.abs().max().item() + 1e-12), 1e-4, n)
        m = fd.FillDropNet(tiny('l', drop_path=0.2)).train()                            # stochastic depth active
        loss = self.loss(m, x, th); loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(m.net.backbone.blocks[1].mlp.fc2.weight.grad.abs().sum().item(), 0)


class Onnx(unittest.TestCase):
    def test_dynamic_graph_parity(self):
        """One batch-1 graph (traced on 4:3) serves every aspect with the right token count, with and without sinks."""
        net = tiny()
        for s in (0, 4):
            m = fd.FillDropNet(net, sinks=s).eval()
            sess = fx.ort_session(fd.export_filldrop(m), 2)
            for asp, n in ((4 / 3, 192), (16 / 9, 160), (1.0, 256), (2 / 3, 192), (21 / 9, None), (3.0, None), (1 / 3, None)):
                x = fx.sample_input(224, asp, seed=9)
                keep = m.keep_mask(x)
                if n is not None:
                    self.assertEqual(int(keep.sum()), n + (s if n < 256 else 0))
                with torch.no_grad():
                    ref = m(x).softmax(1).numpy()
                    ref2 = timm_reference(net, x, keep).softmax(1).numpy()
                got = sess.run(None, {'image': x.numpy()})[0]
                self.assertLess(float(np.abs(got - ref).max()), 1e-5, (s, asp))
                self.assertLess(float(np.abs(got - ref2).max()), 1e-5, (s, asp))

    def test_export_cli(self):
        with tempfile.TemporaryDirectory() as d:
            fd.main(['export', 's', '--out', f'{d}/fd.onnx', '--sinks', '4'])
            self.assertTrue(Path(f'{d}/fd.onnx').stat().st_size > 50e6)


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); return s.getsockname()[1]


class Cli(unittest.TestCase):
    """train (single process and 2-rank gloo DDP) -> plain checkpoints -> filldrop eval (== focus eval --mode fill)."""

    def test_eval_argv(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'R').mkdir()
            for stem in ('final', 'ckpt-004000', 'last'):
                torch.save(dict(config=dict(img_size=224, arch='s', filldrop=dict(sinks=4, tol=fx.TOL))), f'{d}/R/{stem}.pt')
            self.assertEqual(fd.eval_argv([f'{d}/R/final.pt', '--sets', 'dev']), [f'{d}/R/final.pt', '--sets', 'dev', '--mode', 'fill', '--sinks', '4'])
            self.assertEqual(fd.eval_argv([f'{d}/R/ckpt-004000.pt', '--sinks=0'])[-2:], ['--tag', 'R-s004000-fill'])
            self.assertEqual(fd.eval_argv([f'{d}/R/ckpt-004000.pt'])[-2:], ['--tag', 'R-s004000-fillsink4'])
            with self.assertRaises(SystemExit):
                fd.eval_argv([f'{d}/R/last.pt'])
            with self.assertRaises(SystemExit):
                fd.eval_argv([f'{d}/R/final.pt', '--mode', 'focus'])

    def test_train_ddp_then_eval(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / 'data'; src = data / 'rotlab/sources'; src.mkdir(parents=True)
            rng = np.random.default_rng(0); rows, blob = [], b''
            for i, (w, h) in enumerate([(640, 480), (480, 640), (800, 450), (500, 500), (1024, 768), (600, 400)] * 2):
                b = io.BytesIO()
                Image.fromarray(rng.integers(0, 255, (h // 16 + 1, w // 16 + 1, 3), dtype=np.uint8)).resize((w, h), Image.BICUBIC).save(b, 'JPEG')
                rows.append(dict(id=f'p{i}', group=f'g{i % 3}', family='pass', offset=len(blob), length=len(b.getvalue()), base_roll_cw=0.0))
                blob += b.getvalue()
            (src / 'pass.bin').write_bytes(blob); (src / 'pass.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            runs = data / 'rotlab/runs'; (runs / 'INIT').mkdir(parents=True)
            torch.manual_seed(0)
            init = runs / 'INIT/final.pt'
            torch.save(dict(ema=RotNet(img_size=224, pretrained=False, dynamic=True, arch='s').state_dict(),
                            config=dict(img_size=224, arch='s', hidden=1024)), init)
            boot = Path(d) / 'boot.py'
            boot.write_text(f'import sys, rotlab\nrotlab.__path__.append({HERE!r})\nfrom rotlab.filldrop import main\nmain(sys.argv[1:])\n')
            env = dict(os.environ, ROTLAB_DATA=str(data), PYTHONPATH=str(Path(rotlab.__path__[0]).parent), OMP_NUM_THREADS='2')
            common = ['train', '--arch', 's', '--init', str(init), '--mix', 'pass=1', '--steps', '4', '--batch', '4', '--save-every', '2',
                      '--keep-milestones', '--multires', '112,140', '--sinks', '2', '--workers', '1', '--device', 'cpu', '--warmup', '2',
                      '--eval-every', '1000000']
            r = subprocess.run([sys.executable, str(boot), *common, '--run', 'FD1'], env=env, capture_output=True, text=True, timeout=900)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            r = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '2', str(boot), *common,
                                '--run-name', 'FD2', '--batching', 'pad'], env=env, capture_output=True, text=True, timeout=900)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            for run in ('FD1', 'FD2'):
                self.assertEqual(sorted(p.name for p in (runs / run).glob('*.pt')), ['ckpt-000002.pt', 'final.pt', 'last.pt'])
                ck = torch.load(runs / run / 'final.pt', map_location='cpu', weights_only=False)
                self.assertEqual((ck['step'], ck['presentations'], ck['config']['filldrop']['sinks'], ck['config']['multires']), (4, 16, 2, [112, 140]))
                plain = RotNet(img_size=224, pretrained=False, dynamic=True, arch='s')
                plain.load_state_dict(ck['ema']); plain.load_state_dict(ck['model'])          # strict: plain RotNet keys
                self.assertGreater(maxdiff(ck['model']['head.3.weight'], torch.load(init, weights_only=False)['ema']['head.3.weight']), 0)
            cfg = json.loads((runs / 'FD2/config.json').read_text())
            self.assertEqual((cfg['world'], cfg['global_batch'], cfg['batch']), (2, 4, 2))
            # eval through the passthrough on a fake view cache: final + milestone tags, sinks restored from the checkpoint
            (data / 'rotlab/cviews').mkdir(parents=True); pngs = []
            for a in (1.0, 4 / 3, 16 / 9, 3 / 4):
                w = 300; h = round(w / a); b = io.BytesIO()
                Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(b, format='PNG'); pngs.append(b.getvalue())
            (data / 'rotlab/cviews/fair.pkl').write_bytes(pickle.dumps(dict(png=pngs, theta=list(rng.uniform(0, 360, 4)))))
            for ck in ('final.pt', 'ckpt-000002.pt'):
                r = subprocess.run([sys.executable, str(boot), 'eval', str(runs / 'FD1' / ck), '--sets', 'fair', '--device', 'cpu'],
                                   env=env, capture_output=True, text=True, timeout=600)
                self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            from rotlab.camp_store import load
            for tag in ('FD1-fillsink2-c224', 'FD1-s000002-fillsink2-c224'):
                rows_, rec = load(data / 'rotlab/camp-eval', tag)
                self.assertEqual(rows_['fair_ntok'].tolist(), [256, 194, 162, 194], tag)
                self.assertEqual(rec['focus']['sinks'], 2)
            # the stored probabilities are FillDropNet's on the EMA weights
            ck = torch.load(runs / 'FD1/final.pt', map_location='cpu', weights_only=False)
            net = RotNet(img_size=224, pretrained=False, dynamic=True, arch='s'); net.load_state_dict(ck['ema'])
            x = torch.from_numpy(np.stack([letterbox(Image.open(io.BytesIO(p)).convert('RGB'), 224) for p in pngs]))
            with torch.no_grad():
                want = fd.FillDropNet(net.eval(), sinks=2)(x).softmax(1).numpy()
            np.testing.assert_allclose(load(data / 'rotlab/camp-eval', 'FD1-fillsink2-c224')[0]['fair_prob'], want, atol=2e-3)


if __name__ == '__main__':
    unittest.main()
