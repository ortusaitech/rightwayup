#!/usr/bin/env python3
"""Accuracy parity of deployable GPU graphs vs the PyTorch reference, on the real test views (suite sets).
For each tier model and provider (ORT CUDA FP16, TensorRT FP16) reports within-10 accuracy per set, the change vs the
stored PyTorch predictions, and the share of images whose decoded angle differs by more than 2 degrees.
  python -m rotlab.gpu_parity   (run where DATA/rotlab/suite and release-onnx exist, with onnxruntime-gpu + tensorrt)
"""
import os
import numpy as np
import onnxruntime as ort

from rotlab.core import DATA, circ_err, decode

S = DATA / 'rotlab/suite'; O = DATA / 'rotlab/release-onnx'
SETS = ['common', 'fresh_diode', 'origin', 'fair', 'oi', 'meva']
MODELS = [('nano-s112', 112, 'S2-vits-multires-c112'), ('fast-s224', 224, 'S2-vits-multires-c224'), ('large-l224', 224, 'soup-G7-G3-a0.5-c224')]


def session(path, kind, size):
    if kind == 'trt':
        dims = f'{size}x{size}'
        prov = [('TensorrtExecutionProvider', {'trt_fp16_enable': True, 'trt_engine_cache_enable': True, 'trt_engine_cache_path': str(O / 'trt-cache'),
                 'trt_profile_min_shapes': f'image:1x3x{dims}', 'trt_profile_opt_shapes': f'image:64x3x{dims}', 'trt_profile_max_shapes': f'image:64x3x{dims}'}),
                'CUDAExecutionProvider']
    else:
        prov = [('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'HEURISTIC'})]
    s = ort.InferenceSession(str(path), providers=prov)
    want = 'TensorrtExecutionProvider' if kind == 'trt' else 'CUDAExecutionProvider'
    assert s.get_providers()[0] == want, f'{want} not active: {s.get_providers()}'
    s.disable_fallback()
    return s


def run(s, x):
    return np.concatenate([s.run(None, {'image': np.ascontiguousarray(x[i:i + 64])})[0] for i in range(0, len(x), 64)])


def main():
    print('| model | graph | ' + ' | '.join(f'{k} acc (Δ vs PyTorch) / >2° diff' for k in SETS) + ' |'); print('|---' * (len(SETS) + 2) + '|')
    for name, size, ref in MODELS:
        views = np.load(S / f'views-c{size}.npz'); refz = np.load(S / f'{ref}.npz')
        for kind, f in (('cuda-fp16', f'{name}-fp16.onnx'), ('trt-fp16', f'{name}-fp32.onnx')):
            s = session(O / f, 'trt' if kind.startswith('trt') else 'cuda', size); cells = []
            for k in SETS:
                x = views[f'{k}_x']; th = views[f'{k}_t']; p = decode(run(s, x)); e = circ_err(p, th)
                pr = refz[f'{k}_p_pred']; er = circ_err(pr, th)
                d = circ_err(p, pr) > 2
                cells.append(f'{(e <= 10).mean() * 100:.2f}% ({((e <= 10).mean() - (er <= 10).mean()) * 100:+.2f}) / {d.mean() * 100:.2f}%')
            print(f'| {name} | {kind} | ' + ' | '.join(cells) + ' |', flush=True)


if __name__ == '__main__':
    main()
