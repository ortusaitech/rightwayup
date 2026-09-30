#!/usr/bin/env python3
"""CPU tests for rotlab.model_x and rotlab.fetch_weights: random weights (pretrained=False), no data, no GPU, no network.
  pod:   cd /workspace && PYTHONPATH=code python -m pytest -q code/rotlab/test_model_x.py
  local: PYTHONPATH=code ops/venv-cpu/bin/python -m pytest -q rotlab/test_model_x.py
Real checkpoints (optional): ROTLAB_EXT_WEIGHTS=<root used by fetch_weights> also strict-loads every fetched arch.
"""
import json, os, tempfile, unittest, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm

import rotlab
HERE = str(Path(__file__).resolve().parent)
if HERE not in [str(Path(p).resolve()) for p in rotlab.__path__]:   # local checkout: lead files sit beside code/rotlab
    rotlab.__path__.append(HERE)
from rotlab import model_x as mx
from rotlab import fetch_weights as fw
from rotlab.core import MEAN, STD
from rotlab.model import ARCH as ARCH0, RotNet as RotNet0, param_groups as param_groups0

warnings.filterwarnings('ignore')
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
NEW = ('pe_s', 'tips_b', 'b_reg')
NPRE = {'s': 1, 'b': 1, 'l': 1, 'b_reg': 5, 'pe_s': 1, 'tips_b': 2}


def liven(m, seed=0):
    """Random-init DINOv2 LayerScale is 1e-5 (blocks nearly inert) and pos-embeds are tiny: re-randomise both so that
    token-layout, pos-embed and resize mistakes change the output."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith(('ls1.gamma', 'ls2.gamma', 'gamma_1', 'gamma_2')):
                p.copy_(1 + .3 * torch.randn(p.shape, generator=g))
            elif n.endswith(('pos_embed', 'cls_token', 'reg_token')):
                p.copy_(.5 * torch.randn(p.shape, generator=g))
    return m


def make(arch, img_size=224, dynamic=True, seed=0):
    torch.manual_seed(seed)
    return liven(mx.RotNetX(img_size=img_size, pretrained=False, dynamic=dynamic, arch=arch, drop_path=0.0).eval(), seed)


def rand(b, cv, seed=0):
    return torch.randn(b, 3, cv, cv, generator=torch.Generator().manual_seed(seed))


class Legacy(unittest.TestCase):
    """s/b/l must stay exactly rotlab.model.RotNet (keys, outputs, param groups, pooling) so existing checkpoints,
    --init/--teacher and every eval script keep working after the import switch."""

    def check_same(self, arch, canvases):
        torch.manual_seed(0)
        m0 = liven(RotNet0(img_size=224, pretrained=False, dynamic=True, arch=arch, drop_path=0.0).eval())
        m1 = mx.RotNetX(img_size=224, pretrained=False, dynamic=True, arch=arch, drop_path=0.0).eval()
        self.assertEqual(list(m0.state_dict()), list(m1.state_dict()))
        m1.load_state_dict(m0.state_dict())
        self.assertEqual(list(m1.buffers()), list(m0.buffers()))
        self.assertIsNone(m1.tok_norm); self.assertFalse(m1.renorm); self.assertEqual(m1.npre, 1)
        for cv in canvases:
            x = rand(2, cv)
            with torch.no_grad():
                self.assertTrue(torch.equal(m0(x), m1(x)), f'{arch}@{cv}')
        return m0, m1

    def test_s_identical(self):
        m0, m1 = self.check_same('s', (224, 140, 112))
        for net in (m0, m1):
            net.pool = (6, 4)
        with torch.no_grad():
            self.assertTrue(torch.equal(m0(rand(2, 224)), m1(rand(2, 224))))

    def test_b_identical(self):
        self.check_same('b', (140,))

    def test_param_groups_identical(self):
        m = mx.RotNetX(pretrained=False, arch='s')

        def table(groups):
            return {id(p): (g['lr'], g['weight_decay']) for g in groups for p in g['params']}
        self.assertEqual(table(param_groups0(m, 1e-4, 5e-4, 0.05, 0.8)), table(mx.param_groups(m, 1e-4, 5e-4, 0.05, 0.8)))

    def test_arch_paths(self):
        for k in ('s', 'b', 'l'):
            self.assertEqual(mx.SPECS[k].weights, ARCH0[k][1])
            self.assertEqual(mx.SPECS[k].timm, ARCH0[k][0])


class Specs(unittest.TestCase):
    def test_norm_patch_repo_match_timm(self):
        for k, sp in mx.SPECS.items():
            cfg = timm.models.get_pretrained_cfg(sp.timm)
            self.assertTrue(np.allclose(cfg.mean, sp.mean, atol=1e-6) and np.allclose(cfg.std, sp.std, atol=1e-6), k)
            self.assertEqual(sp.repo, f'{cfg.hf_hub_id.rstrip("/")}/{sp.timm}' if cfg.hf_hub_id == 'timm/' else cfg.hf_hub_id, k)
            self.assertTrue(sp.weights.startswith(mx.EXT0 + '/') and sp.weights.endswith('model.safetensors'), k)
            self.assertEqual(len(sp.revision), 40); self.assertEqual(len(sp.sha256), 64)

    def test_patch_and_prefix(self):
        for k in NEW + ('s',):
            m = mx.RotNetX(pretrained=False, arch=k)
            self.assertEqual(m.backbone.patch_embed.patch_size[0], mx.SPECS[k].patch, k)
            self.assertEqual(m.npre, NPRE[k], k)
            self.assertEqual(m.global_token, 'cls', k)
            self.assertEqual(m.tok_norm is not None, k == 'pe_s', k)

    def test_prov_matches_specs(self):
        self.assertEqual(set(fw.PROV), set(mx.SPECS))
        for k in ('pe_s', 'tips_b'):
            self.assertIsNotNone(fw.PROV[k]['upstream'])


class Layout(unittest.TestCase):
    def test_head_input_excludes_prefix(self):
        for k in NEW + ('s',):
            m = make(k); x = rand(2, 224)
            with torch.no_grad():
                t = m.backbone.forward_features(m.adapt(x))
                if m.tok_norm is not None:
                    t = m.tok_norm(t)
                want = torch.cat([t[:, 0], t[:, NPRE[k]:].mean(1)], 1)
                got = m.features(x)
            self.assertEqual(tuple(got.shape), (2, 2 * m.backbone.embed_dim))
            self.assertTrue(torch.allclose(got, want, atol=1e-6), k)
            self.assertEqual(t.shape[1] - NPRE[k], (224 // mx.SPECS[k].patch) ** 2, k)

    def test_input_renormalisation(self):
        raw = torch.rand(2, 3, 224, 224, generator=torch.Generator().manual_seed(1))
        x = (raw - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)   # what letterbox() feeds
        want = {'pe_s': (raw - .5) / .5, 'tips_b': raw, 's': x, 'b_reg': x}
        for k, w in want.items():
            m = mx.RotNetX(pretrained=False, arch=k)
            self.assertTrue(torch.allclose(m.adapt(x), w, atol=2e-6), k)
            self.assertEqual(len(dict(m.named_buffers())) - len(dict(m.backbone.named_buffers())), 2 if k in ('pe_s', 'tips_b') else 0)
            self.assertFalse(any(n.startswith('in_') for n in m.state_dict()), k)   # not persisted

    def test_fallback_global_tokens(self):
        """No-CLS backbones: MAP output if the backbone has an attention pool, else max over patch tokens."""
        base = mx.SPECS['s']
        try:
            for key, kw, want in (('_t_avg', (('class_token', False), ('global_pool', 'avg')), 'max'),
                                  ('_t_map', (('class_token', False), ('global_pool', 'map')), 'map')):
                mx.SPECS[key] = mx.Spec(**{**base.__dict__, 'kwargs': kw})
                m = make(key); x = rand(2, 140)
                self.assertEqual((m.npre, m.global_token), (0, want))
                self.assertEqual(m.tok_norm is not None, want == 'max')   # timm 'avg' moves the final norm to fc_norm
                with torch.no_grad():
                    t = m.backbone.forward_features(x)
                    t = m.tok_norm(t) if m.tok_norm is not None else t
                    g = t.amax(1) if want == 'max' else m.backbone.attn_pool(t)
                    self.assertTrue(torch.allclose(m.features(x), torch.cat([g, t.mean(1)], 1), atol=1e-6))
                    self.assertEqual(tuple(m(x).shape), (2, 360))
        finally:
            mx.SPECS.pop('_t_avg', None); mx.SPECS.pop('_t_map', None)


class Dynamic(unittest.TestCase):
    def test_all_canvases(self):
        for k in NEW:
            m = make(k)
            for cv in (112, 140, 168, 196, 224, 252):
                with torch.no_grad():
                    y = m(rand(2, cv))
                    n = m.tokens(rand(1, cv)).shape[1] - m.npre
                self.assertEqual(tuple(y.shape), (2, 360)); self.assertTrue(torch.isfinite(y).all(), f'{k}@{cv}')
                self.assertEqual(n, (mx.align(cv, m.patch) // m.patch) ** 2, f'{k}@{cv}')

    def test_pe_resize_is_explicit(self):
        m = make('pe_s'); x = rand(2, 140)
        with torch.no_grad():
            self.assertTrue(torch.equal(m(x), m(F.interpolate(x, size=(144, 144), mode='bilinear', align_corners=False))))
        self.assertEqual([mx.align(c, 16) for c in (112, 140, 168, 196, 224, 252)], [112, 144, 176, 192, 224, 256])

    def test_static_copy_matches_dynamic(self):
        for k in NEW + ('s',):
            m = make(k)
            for cv in (140, 224):
                s = mx.static_copy(m, cv); x = rand(2, cv, seed=3)
                with torch.no_grad():
                    d = (m(x) - s(x)).abs().max().item()
                self.assertLess(d, 2e-4, f'{k}@{cv}: {d}')
                self.assertFalse(s.backbone.dynamic_img_size)

    def test_train_step_and_grad_ckpt(self):
        for k in ('pe_s', 'tips_b'):
            torch.manual_seed(0)
            m = mx.RotNetX(img_size=224, pretrained=False, dynamic=True, arch=k, drop_path=0.1).train()
            m.backbone.set_grad_checkpointing(True)
            opt = torch.optim.AdamW(mx.param_groups(m, 1e-4, 5e-4, 0.05, 0.8))
            for sz in (112, 144):
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    loss = m(rand(2, sz)).float().logsumexp(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            self.assertTrue(all(p.grad is not None for n, p in m.named_parameters() if n.startswith(('head.', 'tok_norm.'))))
            lr = {id(p): g['lr'] for g in opt.param_groups for p in g['params']}
            for n, p in m.named_parameters():
                if n.startswith(('head.', 'tok_norm.')):
                    self.assertEqual(lr[id(p)], 5e-4, n)
                elif n.startswith(('backbone.norm_pre', 'backbone.patch_embed', 'backbone.pos_embed')):
                    self.assertAlmostEqual(lr[id(p)], 1e-4 * 0.8 ** 12, msg=n)


class Onnx(unittest.TestCase):
    CASES = (('s', 224), ('pe_s', 224), ('pe_s', 140), ('tips_b', 140), ('b_reg', 140))

    def test_parity_opset18_dynamic_batch(self):
        import onnx
        for k, cv in self.CASES:
            m = make(k); data = mx.export_onnx(m, cv)
            g = onnx.load_from_string(data)
            self.assertEqual([o.version for o in g.opset_import if o.domain in ('', 'ai.onnx')], [18])
            ops = {n.op_type for n in g.graph.node}
            self.assertEqual('Resize' in ops, cv % mx.SPECS[k].patch != 0, f'{k}@{cv}')
            x = rand(2, cv, seed=5)
            with torch.no_grad():
                ref = m(x).softmax(1).numpy()
            got = mx.ort_session(data, threads=2).run(None, {'image': x.numpy()})[0]
            d = float(np.abs(got - ref).max())
            self.assertLess(d, 1e-5, f'{k}@{cv}: {d}')
            self.assertTrue((got.argmax(1) == ref.argmax(1)).all())

    def test_s_graph_equals_legacy_export(self):
        import io, onnx
        m = make('s'); m0 = RotNet0(img_size=224, pretrained=False, arch='s').eval(); m0.load_state_dict(m.state_dict())
        buf = io.BytesIO()
        with torch.no_grad():
            torch.onnx.export(mx.Prob(m0), torch.randn(1, 3, 224, 224), buf, input_names=['image'], output_names=['prob'],
                              opset_version=18, dynamo=False, dynamic_axes={'image': {0: 'batch'}, 'prob': {0: 'batch'}})
        ops = lambda d: [n.op_type for n in onnx.load_from_string(d).graph.node]
        self.assertEqual(ops(mx.export_onnx(m, 224)), ops(buf.getvalue()))


class Weights(unittest.TestCase):
    def test_verify_refuses_unpinned(self):
        with tempfile.TemporaryDirectory() as d:
            old = mx.EXT; mx.EXT = d
            try:
                f = Path(mx._ext(mx.SPECS['pe_s'].weights))
                with self.assertRaises(FileNotFoundError):
                    mx.verify_weights('pe_s')
                f.parent.mkdir(parents=True); f.write_bytes(b'x' * mx.SPECS['pe_s'].size)
                with self.assertRaises(RuntimeError):
                    mx.verify_weights('pe_s')
                with self.assertRaises(RuntimeError):
                    mx.RotNetX(pretrained=True, arch='pe_s')
            finally:
                mx.EXT = old

    @unittest.skipUnless(os.environ.get('ROTLAB_EXT_WEIGHTS'), 'set ROTLAB_EXT_WEIGHTS to strict-load fetched checkpoints')
    def test_real_checkpoints(self):
        found = 0
        for k in mx.SPECS:
            if Path(mx._ext(mx.SPECS[k].weights)).is_file():
                m = mx.RotNetX(img_size=224, pretrained=True, dynamic=True, arch=k).eval(); found += 1
                with torch.no_grad():
                    self.assertTrue(torch.isfinite(m(rand(1, 224))).all(), k)
        self.assertGreater(found, 0)


class FetchOffline(unittest.TestCase):
    """fetch_weights against a file:// Hub mirror with a tiny timm model registered as a fake arch."""

    def setUp(self):
        from safetensors.torch import save_file
        self.tmp = tempfile.TemporaryDirectory(); T = Path(self.tmp.name)
        self.mirror, self.root = T / 'mirror', T / 'root'
        torch.manual_seed(0)
        sd = {k: v.contiguous() for k, v in timm.create_model('test_vit', pretrained=False, num_classes=0).state_dict().items()}
        wf = T / 'w.safetensors'; save_file(sd, str(wf)); wb = wf.read_bytes()
        self.rev, self.repo = 'a' * 40, 'org/fake'
        self.card = b'---\nlicense: apache-2.0\nlibrary_name: timm\n---\n# fake\n'
        cfg = json.dumps(dict(architecture='test_vit', pretrained_cfg=dict(license='apache-2.0'))).encode()
        lic = b'Apache License\n Version 2.0, January 2004\n'
        res = self.mirror / self.repo / 'resolve' / self.rev; res.mkdir(parents=True)
        for n, b in (('model.safetensors', wb), ('README.md', self.card), ('config.json', cfg)):
            (res / n).write_bytes(b)
        (self.mirror / 'LICENSE').write_bytes(lic)
        self.wsha = fw.sha256_bytes(wb)
        self.tree = [dict(type='file', path='model.safetensors', size=len(wb), oid='0' * 40, lfs=dict(oid=self.wsha, size=len(wb))),
                     dict(type='file', path='README.md', size=len(self.card), oid=fw.git_blob_id(self.card)),
                     dict(type='file', path='config.json', size=len(cfg), oid=fw.git_blob_id(cfg))]
        self.write_api(dict(sha=self.rev, gated=False, private=False, tags=['license:apache-2.0'], cardData=dict(license='apache-2.0')))
        mx.SPECS['_fake'] = mx.Spec('test_vit.r160_in1k', f'{mx.EXT0}/_fake-{self.rev[:8]}/model.safetensors', self.repo, self.rev,
                                    self.wsha, len(wb), 'test', patch=16, mean=(.5,) * 3, std=(.5,) * 3)
        L = dict(name='LICENSE.fake', url=(self.mirror / 'LICENSE').as_uri(), sha256=fw.sha256_bytes(lic), quote='Apache License Version 2.0, January 2004')
        fw.PROV['_fake'] = dict(licence_texts=[L], upstream=None, lineage='test')
        self.old_timm_lic = fw.TIMM_CODE_LIC; fw.TIMM_CODE_LIC = dict(L, name='LICENSE.timm')
        self.args = lambda: type('A', (), dict(endpoint=self.mirror.as_uri(), root=str(self.root), verify_upstream=False, keep_upstream=False))()

    def tearDown(self):
        mx.SPECS.pop('_fake', None); fw.PROV.pop('_fake', None); fw.TIMM_CODE_LIC = self.old_timm_lic
        self.tmp.cleanup()

    def write_api(self, info, tree=None):
        api = self.mirror / 'api/models' / self.repo
        (api / 'revision').mkdir(parents=True, exist_ok=True); (api / 'tree').mkdir(parents=True, exist_ok=True)
        (api / 'revision' / self.rev).write_text(json.dumps(info)); (api / 'tree' / self.rev).write_text(json.dumps(tree or self.tree))

    def test_fetch_receipt_verify_and_refusals(self):
        r = fw.do_fetch('_fake', self.args())
        self.assertEqual((r['status'], r['weights_sha256'], r['strict_load']), ('OK', self.wsha, True))
        d = fw.dest_dir('_fake', self.root)
        rec = json.loads((d / 'PROVENANCE._fake.json').read_text())
        self.assertEqual(rec['hub']['revision_resolved'], self.rev)
        self.assertEqual([f['path'] for f in rec['files']], ['model.safetensors', 'README.md', 'config.json'])
        self.assertEqual(rec['model_card']['yaml_licence'], 'apache-2.0')
        self.assertTrue((d / 'LICENSE.fake').exists() and (d / 'hub-api._fake.json').exists())
        self.assertEqual(fw.do_verify('_fake', self.args())['status'], 'OK')
        r2 = fw.do_fetch('_fake', self.args())                           # re-run: reuse verified file, new receipt
        self.assertTrue(r2['receipt'].startswith('PROVENANCE._fake.') and r2['receipt'] != 'PROVENANCE._fake.json')
        with open(d / 'model.safetensors', 'ab') as f:                    # tamper
            f.write(b'\0')
        with self.assertRaises(fw.Refuse):
            fw.do_verify('_fake', self.args())
        with self.assertRaises(fw.Refuse):
            fw.do_fetch('_fake', self.args())                             # never overwrites a mismatching file

    def test_refuse_before_download_on_hub_pin_mismatch(self):
        bad = [dict(e, lfs=dict(oid='f' * 64, size=e['size'])) if e['path'] == 'model.safetensors' else e for e in self.tree]
        self.write_api(dict(sha=self.rev, gated=False, private=False, tags=['license:apache-2.0'], cardData=dict(license='apache-2.0')), bad)
        with self.assertRaises(fw.Refuse):
            fw.do_fetch('_fake', self.args())
        self.assertFalse((fw.dest_dir('_fake', self.root) / 'model.safetensors').exists())

    def test_refuse_on_bytes_mismatch_and_licence(self):
        res = self.mirror / self.repo / 'resolve' / self.rev / 'model.safetensors'
        b = bytearray(res.read_bytes()); b[-1] ^= 1; res.write_bytes(bytes(b))   # Hub lists the pin, serves other bytes
        with self.assertRaises(fw.Refuse):
            fw.do_fetch('_fake', self.args())
        d = fw.dest_dir('_fake', self.root)
        self.assertFalse((d / 'model.safetensors').exists() or (d / 'model.safetensors.part').exists())
        for info in (dict(sha=self.rev, gated='auto', tags=['license:apache-2.0'], cardData=dict(license='apache-2.0')),
                     dict(sha=self.rev, gated=False, tags=['license:cc-by-nc-4.0'], cardData=dict(license='cc-by-nc-4.0')),
                     dict(sha='b' * 40, gated=False, tags=['license:apache-2.0'], cardData=dict(license='apache-2.0'))):
            self.write_api(info)
            with self.assertRaises(fw.Refuse):
                fw.do_check('_fake', self.args())


if __name__ == '__main__':
    unittest.main()
