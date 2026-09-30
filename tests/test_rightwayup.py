"""Tests for the rightwayup package.

Parity tests compare with the research code (rotlab) when it is importable. End-to-end tests need the ONNX files:
set RIGHTWAYUP_MODEL_DIR to a folder holding the release files (pico-s70-*, nano-s112-*, fast-s224-*, max-l280-*; INT8 is
enough on CPU), or set RIGHTWAYUP_TEST_HUB=1 to download them from the Hugging Face Hub.
"""
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import rightwayup as rw

HERE = Path(__file__).parent
IMG = HERE / 'upright-render.jpg'   # ORTUS AI Blender render of Poly Haven assets (CC BY 4.0), upright
HAVE_MODELS = bool(os.environ.get('RIGHTWAYUP_MODEL_DIR') or os.environ.get('RIGHTWAYUP_TEST_HUB'))


def circ(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def turned(im, cw, fill=(0, 0, 0)):
    """Image content turned `cw` degrees clockwise (PIL rotates counter-clockwise for positive angles); the canvas grows
    to fit and the uncovered corners take `fill`, as image editors and PIL/OpenCV rotations produce."""
    return im.rotate(-cw, resample=Image.BICUBIC, expand=True, fillcolor=fill)


def test_decode_confidence_match_research_code():
    rotlab = pytest.importorskip('rotlab.core')
    ev = pytest.importorskip('rotlab.evaluate')
    rng = np.random.default_rng(0)
    p = rng.random((64, 360)).astype(np.float32) ** 8; p /= p.sum(1, keepdims=True)
    np.testing.assert_allclose(rw.decode(p), rotlab.decode(p))
    a = rw.decode(p)
    np.testing.assert_allclose(rw.confidence(p, a), ev.confidence(p, a), rtol=1e-6)


def test_letterbox_matches_research_code():
    rotlab = pytest.importorskip('rotlab.core')
    im = Image.open(IMG).convert('RGB')
    for size in (70, 112, 224, 280):
        np.testing.assert_array_equal(rw.letterbox(im, size), rotlab.letterbox(im, size))


def test_decode_wraps_around_zero():
    p = np.zeros((1, 360), np.float32); p[0, [358, 359, 0, 1, 2]] = [0.1, 0.2, 0.4, 0.2, 0.1]
    a = rw.decode(p)[0]
    assert circ(a, 0.0) < 0.01
    assert rw.confidence(p, np.array([a]))[0] == pytest.approx(1.0)


def test_thresholds_are_the_frozen_values():
    # FREEZE.md (FREEZE v2, per-format addendum and Pico addendum), fitted on calibration data only
    fp32 = {t: rw.TIERS[t].thresholds['fp32'] for t in rw.TIERS}
    assert fp32 == {'pico': (None, 0.426, 0.833), 'nano': (None, 0.535, 0.800), 'fast': (None, 0.629, 0.738),
                    'balanced': (0.629, 0.691, 0.743), 'pro': (0.800, 0.729, 0.729), 'max': (None, 0.727, 0.727)}
    assert rw.TIERS['pro'].thresholds['int8'] == (0.797, 0.714, 0.714) and rw.TIERS['nano'].thresholds['batch-int8'] == (None, 0.531, 0.782)
    assert rw.TIERS['balanced'].large == 'max-l280' and rw.TIERS['balanced'].large_size == 280 and rw.TIERS['pico'].size == 70
    for t in rw.TIERS.values():
        assert set(t.thresholds) == {'fp32', 'fp16', 'int8', 'batch-fp32', 'batch-fp16', 'batch-int8'}
        assert (t.large is None) == all(v[0] is None for v in t.thresholds.values())


@pytest.mark.skipif(not HAVE_MODELS, reason='set RIGHTWAYUP_MODEL_DIR (or RIGHTWAYUP_TEST_HUB=1) to run model tests')
@pytest.mark.parametrize('batch', [False, True])
@pytest.mark.parametrize('tier', ['pico', 'nano', 'fast', 'balanced', 'pro', 'max'])
def test_recovers_known_rotations(tier, batch):
    o = rw.Orienter(tier, device='cpu', precision='int8', batch=batch)
    base = Image.open(IMG).convert('RGB')
    angles = [0, 90, 180, 270, 33, 312]
    rs = o.predict_batch([turned(base, a) for a in angles])
    errs = [circ(r.angle_cw, a) for r, a in zip(rs, angles)]
    assert max(errs) <= 10, list(zip(angles, [round(r.angle_cw, 1) for r in rs]))


@pytest.mark.skipif(not HAVE_MODELS, reason='set RIGHTWAYUP_MODEL_DIR (or RIGHTWAYUP_TEST_HUB=1) to run model tests')
@pytest.mark.parametrize('fill', [(0, 0, 0), (255, 255, 255), (128, 128, 128)])
@pytest.mark.parametrize('tier', ['pico', 'nano', 'fast', 'balanced', 'pro', 'max'])
def test_rotation_corners(tier, fill):
    """Images turned by arbitrary angles with the corner fills editors and PIL/OpenCV produce (black, white, grey).
    Known limitation (30 Sep 2026): a sky-blue fill such as (40, 120, 200) can flip Nano by 180 degrees (it reads as sky);
    Fast and larger tiers are unaffected."""
    o = rw.Orienter(tier, device='cpu', precision='int8')
    base = Image.open(IMG).convert('RGB')
    angles = [17, 33, 61, 118, 152, 205, 243, 299, 312, 341]
    rs = o.predict_batch([turned(base, a, fill) for a in angles])
    errs = [circ(r.angle_cw, a) for r, a in zip(rs, angles)]
    assert max(errs) <= 10, list(zip(angles, [round(r.angle_cw, 1) for r in rs]))


@pytest.mark.skipif(not HAVE_MODELS, reason='set RIGHTWAYUP_MODEL_DIR (or RIGHTWAYUP_TEST_HUB=1) to run model tests')
def test_correct_and_cli(tmp_path):
    o = rw.Orienter('fast', device='cpu', precision='int8')
    src = tmp_path / 'sideways.jpg'
    turned(Image.open(IMG).convert('RGB'), 90).save(src, quality=95)
    up = o.correct(src, snap=90)
    assert up.size == Image.open(IMG).size          # a quarter-turn correction restores the original shape
    assert circ(o.predict(up).angle_cw, 0) <= 10
    from rightwayup.cli import main
    assert main(['fix', str(src), '--tier', 'fast', '--device', 'cpu', '--precision', 'int8', '--snap', '90',
                 '--out', str(tmp_path / 'out')]) == 0
    assert (tmp_path / 'out' / 'sideways.jpg').exists()


@pytest.mark.skipif(not HAVE_MODELS, reason='set RIGHTWAYUP_MODEL_DIR (or RIGHTWAYUP_TEST_HUB=1) to run model tests')
def test_abstains_on_featureless_image():
    o = rw.Orienter('fast', device='cpu', precision='int8')
    flat = Image.new('RGB', (320, 240), (128, 128, 128))
    assert o.predict(flat).abstain
    assert o.correct(flat).size == flat.size   # unchanged when abstaining


def test_strict_is_never_looser_than_standard():
    from rightwayup.core import Orienter, TIERS
    for tier in TIERS:
        for fmt, (_, std_t, strict_t) in TIERS[tier].thresholds.items():
            o = Orienter.__new__(Orienter); o.abstain_standard, o.abstain_strict = std_t, strict_t
            o.abstain_mode = 'standard'; std = o.threshold
            o.abstain_mode = 'strict'; strict = o.threshold
            assert strict >= std, (tier, fmt)
