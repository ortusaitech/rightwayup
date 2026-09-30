#!/usr/bin/env python3
"""Evaluate an exported ONNX graph (FP32/INT8) with ONNX Runtime CPU on the benchmark pack and the
fair fixed-crop photo set. Canvas is read from the graph's input shape (square).
  python -m rotlab.eval_onnx model-int8.onnx [--threads 8] [--no-fair]"""
import argparse, json
import numpy as np, onnxruntime as ort
from rotlab.core import circ_err, decode, letterbox, metrics
from rotlab.evaluate import PackCache, confidence, report, table


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('onnx'); ap.add_argument('--threads', type=int, default=8); ap.add_argument('--no-fair', action='store_true')
    a = ap.parse_args()
    so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
    s = ort.InferenceSession(a.onnx, so, providers=['CPUExecutionProvider'])
    inp = s.get_inputs()[0]; size = int(inp.shape[-1])
    run = lambda x: s.run(None, {inp.name: np.ascontiguousarray(x[None])})[0][0]
    pc = PackCache(size)
    prob = np.stack([run(x) for x in pc.x])
    res, pred = report(pc, prob, a.onnx.split('/')[-1])
    print(table(res))
    out = dict(result=res, predictions=[float(v) for v in pred], confidence=[float(c) for c in confidence(prob, pred)])
    if not a.no_fair:
        from rotlab.fair_photo import fair_views
        _, views, th = fair_views(7)
        fp = np.stack([run(letterbox(v, size)) for v in views]); fpred = decode(fp)
        out['fair'] = metrics(circ_err(fpred, th)); out['fair_pred'] = fpred.tolist(); out['fair_conf'] = confidence(fp, fpred).tolist()
        print('FAIR', json.dumps(out['fair']))
    json.dump(out, open(a.onnx + '.eval.json', 'w'))


if __name__ == '__main__':
    main()
