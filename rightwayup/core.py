"""RightWayUp inference: tiers, preprocessing, cascade routing, calibrated abstention.

Preprocessing, decoding and confidence reproduce the evaluated pipeline exactly (rotlab.core.letterbox / decode,
rotlab.evaluate.confidence). Routing and abstention thresholds are the frozen values fixed on calibration data
(FREEZE.md); they are not tuned per deployment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
from PIL import Image, ImageOps

FILL = (124, 116, 104)                      # ImageNet mean colour, letterbox padding
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
HUB = 'ortusai/rightwayup'           # one Hugging Face repository holds every tier's files


@dataclass(frozen=True)
class TierSpec:
    small: str                  # file stem of the first (or only) model
    size: int                   # input canvas for the small model
    large: Optional[str] = None
    large_size: int = 280       # input canvas for the large model (cascade tiers)
    thresholds: dict = None     # per file format: (route or None, abstain_standard, abstain_strict)


# FREEZE v2 (29 Sep 2026): every threshold was fitted on calibration data only, separately for each shipped file format
# (FREEZE.md, per-format addendum and Pico addendum). Tuple = (route, standard, strict): route to the large model when
# the small model's confidence < route; abstain when confidence < standard (~90% of calibration images answered) or
# < strict (<=1% wrong on answered calibration images; never looser than standard).
_T = {  # format -> tier -> (route, standard, strict)
    'fp32': {'pico': (None, 0.426, 0.833), 'nano': (None, 0.535, 0.800), 'fast': (None, 0.629, 0.738),
             'balanced': (0.629, 0.691, 0.743), 'pro': (0.800, 0.729, 0.729), 'max': (None, 0.727, 0.727)},
    'fp16': {'pico': (None, 0.425, 0.833), 'nano': (None, 0.535, 0.800), 'fast': (None, 0.629, 0.738),
             'balanced': (0.629, 0.691, 0.745), 'pro': (0.800, 0.729, 0.729), 'max': (None, 0.727, 0.727)},
    'int8': {'pico': (None, 0.415, 0.833), 'nano': (None, 0.536, 0.800), 'fast': (None, 0.626, 0.738),
             'balanced': (0.626, 0.685, 0.745), 'pro': (0.797, 0.714, 0.714), 'max': (None, 0.711, 0.711)},
    'batch-fp16': {'pico': (None, 0.425, 0.833), 'nano': (None, 0.535, 0.800), 'fast': (None, 0.629, 0.738),
                   'balanced': (0.629, 0.691, 0.743), 'pro': (0.800, 0.729, 0.729), 'max': (None, 0.727, 0.727)},
    'batch-int8': {'pico': (None, 0.423, 0.853), 'nano': (None, 0.531, 0.782), 'fast': (None, 0.623, 0.738),
                   'balanced': (0.623, 0.684, 0.745), 'pro': (0.793, 0.714, 0.714), 'max': (None, 0.711, 0.711)},
}
_T['batch-fp32'] = _T['fp32']   # batch-capable FP32 files give the same answers as the one-image files


def _spec(small, size, large=None, tier=None):
    return TierSpec(small, size, large, 280, {f: t[tier] for f, t in _T.items()})


TIERS = {
    'pico': _spec('pico-s70', 70, tier='pico'),
    'nano': _spec('nano-s112', 112, tier='nano'),
    'fast': _spec('fast-s224', 224, tier='fast'),
    'balanced': _spec('fast-s224', 224, 'max-l280', tier='balanced'),
    'pro': _spec('fast-s224', 224, 'max-l280', tier='pro'),
    'max': _spec('max-l280', 280, tier='max'),
}
ImageLike = Union[str, os.PathLike, Image.Image, np.ndarray]


@dataclass
class Result:
    angle_cw: float          # clockwise rotation of the image content, degrees in [0, 360)
    confidence: float        # probability mass within +-10 degrees of angle_cw
    abstain: bool            # True: the image doesn't define a reliable "up" (angle_cw is still reported)
    tier: str
    routed: bool = False     # cascade tiers: answered by the large model

    @property
    def correction_ccw(self) -> float:
        """Counter-clockwise rotation (degrees) that turns the image upright."""
        return self.angle_cw

    def to_dict(self):
        return asdict(self)


def load_image(x: ImageLike) -> Image.Image:
    """PIL RGB image, with EXIF orientation applied (the image as a viewer displays it)."""
    if isinstance(x, Image.Image):
        im = x
    elif isinstance(x, np.ndarray):
        im = Image.fromarray(x)
    else:
        im = Image.open(x)
        im = ImageOps.exif_transpose(im)
    return im.convert('RGB')


def letterbox(im: Image.Image, size: int) -> np.ndarray:
    """Aspect-preserving bicubic fit into a size x size canvas on mean-colour padding -> CHW float32 (normalised)."""
    w, h = im.size
    s = min(size / w, size / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    canvas = Image.new('RGB', (size, size), FILL)
    canvas.paste(im.resize((nw, nh), Image.BICUBIC), ((size - nw) // 2, (size - nh) // 2))
    a = (np.asarray(canvas, np.float32) / 255 - MEAN) / STD
    return a.transpose(2, 0, 1).copy()


def decode(prob: np.ndarray) -> np.ndarray:
    """Argmax bin refined by the local circular mean over +-10 bins."""
    k = prob.argmax(1)
    idx = (k[:, None] + np.arange(-10, 11)[None]) % 360
    w = np.take_along_axis(prob, idx, 1)
    off = (w * np.arange(-10, 11)[None]).sum(1) / np.maximum(w.sum(1), 1e-12)
    return (k + off) % 360


def confidence(prob: np.ndarray, pred: np.ndarray, width: int = 10) -> np.ndarray:
    """Probability mass within +-width degrees of the decoded angle."""
    d = np.abs((np.arange(360)[None] - pred[:, None] + 180) % 360 - 180)
    return (prob * (d <= width)).sum(1)


def _providers(device: str):
    import onnxruntime as ort
    have = ort.get_available_providers()
    cuda = ('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'HEURISTIC'})
    if device == 'auto':
        return [cuda, 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in have else ['CPUExecutionProvider']
    table = {'cpu': ['CPUExecutionProvider'], 'cuda': [cuda, 'CPUExecutionProvider'],
             'tensorrt': [('TensorrtExecutionProvider', {'trt_fp16_enable': True}), cuda, 'CPUExecutionProvider'],
             'coreml': ['CoreMLExecutionProvider', 'CPUExecutionProvider']}
    if device not in table:
        raise ValueError(f'device must be one of auto, {", ".join(table)}')
    missing = [p if isinstance(p, str) else p[0] for p in table[device][:1]]
    if missing[0] not in have:
        raise RuntimeError(f'{missing[0]} is not available in this onnxruntime build (have: {", ".join(have)})')
    return table[device]


class Orienter:
    """Estimate how far images are rotated from upright.

    tier:      'pico' | 'nano' | 'fast' | 'balanced' | 'pro' | 'max'
    device:    'auto' (CUDA if available, else CPU) | 'cpu' | 'cuda' | 'tensorrt' | 'coreml'
    precision: 'auto' (INT8 on CPU, FP16 on GPU) | 'fp32' | 'fp16' | 'int8'
    batch:     False (default): the one-image files, fastest for single images (Pico, Nano and Fast drop the letterbox
               padding inside the model); True: the batch-capable files for predict_batch throughput. Same answers.
    abstain:   'standard' (~90% answered on calibration) | 'strict' (<=1% wrong on calibration, never looser than
               standard) | 'off'
    model_dir: folder holding the ONNX files; default: download from the Hugging Face Hub (ortusai/rightwayup)
    """

    def __init__(self, tier: str = 'max', device: str = 'auto', precision: str = 'auto', abstain: str = 'standard',
                 model_dir: Optional[Union[str, os.PathLike]] = None, threads: Optional[int] = None, batch: bool = False):
        import onnxruntime as ort
        if tier not in TIERS:
            raise ValueError(f'tier must be one of {", ".join(TIERS)}')
        if abstain not in ('standard', 'strict', 'off'):
            raise ValueError("abstain must be 'standard', 'strict' or 'off'")
        self.tier, self.spec, self.abstain_mode = tier, TIERS[tier], abstain
        providers = _providers(device)
        gpu = any((p if isinstance(p, str) else p[0]) in ('CUDAExecutionProvider', 'TensorrtExecutionProvider') for p in providers)
        self.precision = precision if precision != 'auto' else ('fp16' if gpu else 'int8')
        if self.precision not in ('fp32', 'fp16', 'int8'):
            raise ValueError("precision must be 'auto', 'fp32', 'fp16' or 'int8'")
        self.batch = bool(batch)
        self.format = ('batch-' if self.batch else '') + self.precision
        self.route, self.abstain_standard, self.abstain_strict = self.spec.thresholds[self.format]
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        model_dir = model_dir or os.environ.get('RIGHTWAYUP_MODEL_DIR')
        self.sessions = {}
        for stem in filter(None, (self.spec.small, self.spec.large)):
            batched = self.batch and not stem.startswith('max-')      # Max's files take any batch size
            path = self._file(f'{stem}{"-batch" if batched else ""}-{self.precision}.onnx', model_dir)
            self.sessions[stem] = ort.InferenceSession(str(path), so, providers=providers)
        self.device = self.sessions[self.spec.small].get_providers()[0]

    def _file(self, name: str, model_dir) -> Path:
        if model_dir:
            p = Path(model_dir) / name
            if not p.exists():
                raise FileNotFoundError(p)
            return p
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(HUB, name))

    @property
    def threshold(self) -> float:
        # 'strict' is never looser than 'standard' (decision 25 Sep 2026; no test data involved).
        strict = max(self.abstain_standard, self.abstain_strict)
        return {'standard': self.abstain_standard, 'strict': strict, 'off': -1.0}[self.abstain_mode]

    def _run(self, stem: str, x: np.ndarray, batch_size: int) -> np.ndarray:
        s = self.sessions[stem]
        fixed = isinstance(s.get_inputs()[0].shape[0], int)          # one-image files have a fixed batch of 1
        bs = 1 if fixed else batch_size
        return np.concatenate([s.run(None, {'image': x[i:i + bs]})[0] for i in range(0, len(x), bs)])

    def predict_batch(self, images: Sequence[ImageLike], batch_size: int = 16) -> List[Result]:
        ims = [load_image(i) for i in images]
        sp = self.spec
        prob = self._run(sp.small, np.stack([letterbox(im, sp.size) for im in ims]), batch_size)
        ang = decode(prob); conf = confidence(prob, ang); routed = np.zeros(len(ims), bool)
        if sp.large:
            routed = conf < self.route
            if routed.any():
                idx = np.flatnonzero(routed)
                pl = self._run(sp.large, np.stack([letterbox(ims[i], sp.large_size) for i in idx]), batch_size)
                al = decode(pl); ang[idx], conf[idx] = al, confidence(pl, al)
        return [Result(float(a), float(c), bool(c < self.threshold), self.tier, bool(r)) for a, c, r in zip(ang, conf, routed)]

    def predict(self, image: ImageLike) -> Result:
        return self.predict_batch([image])[0]

    def correct(self, image: ImageLike, snap: Optional[int] = None, force: bool = False,
                result: Optional[Result] = None) -> Image.Image:
        """Return the image turned upright. If the model abstains, the image is returned unchanged unless force=True.
        snap=90 rounds the correction to the nearest quarter turn (lossless, no cropping or padding)."""
        im = load_image(image)
        r = result or self.predict(im)
        if r.abstain and not force:
            return im
        a = r.correction_ccw
        if snap:
            q = int(round(a / snap)) * snap % 360
            if snap == 90:
                return {0: im, 90: im.transpose(Image.Transpose.ROTATE_90), 180: im.transpose(Image.Transpose.ROTATE_180),
                        270: im.transpose(Image.Transpose.ROTATE_270)}[q]
            a = q
        return im.rotate(a, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))
