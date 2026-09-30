#!/usr/bin/env python3
"""Merge benchmark JSONL files (rotlab.bench / coreml_export) into Markdown hardware tables.
  python -m rotlab.bench_report release/benchmarks/*.jsonl > release/benchmarks/HARDWARE.md
Rows with errors are listed separately. The newest row wins for a duplicate (device, model, runtime, threads, batch).
"""
import json, sys
from collections import OrderedDict

TIER = {'nano-s112': 'Nano (ViT-S @112)', 'fast-s224': 'Fast (ViT-S @224)', 'large-l224': 'Large (ViT-L @224)',
        'rotation_estimator': 'Previous SOTA (MambaOut-B)'}


def tier(model):
    return next((v for k, v in TIER.items() if model.startswith(k)), model)


def runtime(r):
    m = r['model']; p = r['provider']
    prec = 'INT8' if 'int8' in m else 'FP16' if ('fp16' in m or p in ('trt', 'coreml-native')) else 'FP32'
    name = {'cpu': f'CPU {prec} ×{r.get("threads", 1)}t', 'cuda': f'CUDA {prec}', 'trt': 'TensorRT FP16', 'coreml': f'Core ML EP {prec}',
            'coreml-native': f'Core ML {r.get("units", "")} FP16'}[p]
    return name


def device(r):
    h = r.get('host', {})
    return h.get('gpu', '').split(',')[0] if r['provider'] in ('cuda', 'trt') and h.get('gpu') else (h.get('cpu') or '?')


def main():
    rows = OrderedDict(); errors = []
    for f in sys.argv[1:]:
        for line in open(f):
            r = json.loads(line)
            if 'host' not in r or 'model' not in r:
                continue
            key = (device(r), tier(r['model']), runtime(r), r.get('batch'))
            if 'error' in r:
                errors.append((key, r['error'][-120:])); continue
            rows[key] = r
    by_dev = OrderedDict()
    for (dev, t, rt, b), r in rows.items():
        by_dev.setdefault(dev, []).append((t, rt, b, r))
    order = list(TIER.values())
    for dev, items in by_dev.items():
        print(f'\n### {dev}\n')
        print('| Model | Runtime | Latency, single image (batch 1), p50 / p95 | Throughput, batched | Peak VRAM | Peak process RAM |')
        print('|---|---|---|---|---|---|')
        for t, rt, b, r in sorted(items, key=lambda x: (order.index(x[0]) if x[0] in order else 9, x[1], x[2] or 0)):
            thr = f'{r["throughput_img_s"]:,.0f} img/s (batch {b})' if r.get('throughput_img_s') else '—'
            vram = f'{r["peak_vram_mb"]:,} MiB' if r.get('peak_vram_mb') else '—'
            ram = f'{r["peak_rss_mb"]:,.0f} MiB' if r.get('peak_rss_mb') else '—'
            print(f'| {t} | {rt} | {r["p50_ms"]} / {r["p95_ms"]} ms | {thr} | {vram} | {ram} |')
    if errors:
        print('\n#### Not measured (errors)\n')
        for (dev, t, rt, b), e in errors:
            print(f'- {dev} · {t} · {rt}: {e.strip().splitlines()[-1] if e.strip() else e}')


if __name__ == '__main__':
    main()
