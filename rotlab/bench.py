#!/usr/bin/env python3
"""Standalone ONNX Runtime benchmark (numpy + onnxruntime or onnxruntime-gpu; optional psutil). Ships with the release.

For every model file and execution provider it runs a fresh subprocess and reports:
  latency (batch 1): p50 / p95 ms after warm-up;  throughput (batch B): images/s;
  peak process RAM (RSS);  peak GPU memory used by the process (nvidia-smi), CUDA arena set to kSameAsRequested.
Inputs are random tensors of the model's input shape (orientation models are shape-static; content does not affect speed).

  python bench.py MODEL.onnx [MODEL.onnx ...] --providers cpu,cuda --threads 1,4 --batch 32 --out results.jsonl
"""
import argparse, json, os, platform, subprocess, sys, threading, time


def child(path, provider, threads, batch, runs):
    import numpy as np, onnxruntime as ort
    if provider in ('cuda', 'trt') and os.name == 'nt':
        # Windows: load CUDA/cuDNN (and TensorRT) DLLs installed as pip packages
        try:
            import tensorrt_libs
            os.add_dll_directory(os.path.dirname(tensorrt_libs.__file__))
            os.environ['PATH'] = os.path.dirname(tensorrt_libs.__file__) + os.pathsep + os.environ['PATH']
        except ImportError:
            pass
        if hasattr(ort, 'preload_dlls'):
            ort.preload_dlls()
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    so.add_session_config_entry('session.intra_op.allow_spinning', '0')
    if provider == 'cuda':
        # HEURISTIC cuDNN algorithm search: 'DEFAULT' picks a pathological kernel for depthwise convolutions
        # (the previous SOTA ran ~70x slower with it), which would make CNN baselines look unfairly slow.
        prov = [('CUDAExecutionProvider', {'arena_extend_strategy': 'kSameAsRequested', 'cudnn_conv_algo_search': 'HEURISTIC'}), 'CPUExecutionProvider']
    elif provider == 'trt':   # TensorRT FP16 engine, batch profile 1..max(batch,64), cached next to the model
        import onnxruntime as _o
        hw0 = _o.InferenceSession(path, providers=['CPUExecutionProvider']).get_inputs()[0].shape[2:]
        dims = 'x'.join(str(int(d)) for d in hw0); mx = batch
        cache = os.path.join(os.path.dirname(os.path.abspath(path)), 'trt-cache')
        prov = [('TensorrtExecutionProvider', {'trt_fp16_enable': True, 'trt_engine_cache_enable': True, 'trt_engine_cache_path': cache,
                 'trt_profile_min_shapes': f'image:1x3x{dims}', 'trt_profile_opt_shapes': f'image:{mx}x3x{dims}',
                 'trt_profile_max_shapes': f'image:{mx}x3x{dims}'}),
                ('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'HEURISTIC'}), 'CPUExecutionProvider']
    elif provider == 'coreml':
        prov = [('CoreMLExecutionProvider', {'ModelFormat': 'MLProgram', 'MLComputeUnits': 'ALL'}), 'CPUExecutionProvider']
    else:
        prov = ['CPUExecutionProvider']
    s = ort.InferenceSession(path, so, providers=prov)
    if provider != 'cpu':
        s.disable_fallback()   # never silently re-run on CPU if the accelerator fails (it would report CPU speed as GPU)
    assert provider != 'cuda' or s.get_providers()[0] == 'CUDAExecutionProvider', 'CUDA provider not active'
    assert provider != 'trt' or s.get_providers()[0] == 'TensorrtExecutionProvider', 'TensorRT provider not active'
    assert provider != 'coreml' or s.get_providers()[0] == 'CoreMLExecutionProvider', 'CoreML provider not active'
    inp = s.get_inputs()[0]; hw = [int(d) for d in inp.shape[1:]]
    dtype = np.float16 if 'float16' in inp.type else np.float32
    x1 = np.random.rand(1, *hw).astype(dtype); xb = np.random.rand(batch, *hw).astype(dtype)
    batched = not isinstance(inp.shape[0], int) or inp.shape[0] != 1
    for _ in range(5): s.run(None, {inp.name: x1})
    t = []
    for _ in range(runs):
        t0 = time.perf_counter(); s.run(None, {inp.name: x1}); t.append((time.perf_counter() - t0) * 1000)
    t = sorted(t); res = dict(p50_ms=round(t[len(t) // 2], 2), p95_ms=round(t[int(len(t) * .95) - 1], 2))
    if batched:
        s.run(None, {inp.name: xb}); n = max(3, runs // 10); t0 = time.perf_counter()
        for _ in range(n): s.run(None, {inp.name: xb})
        res['batch'] = batch; res['throughput_img_s'] = round(batch * n / (time.perf_counter() - t0), 1)
    else:
        res['batch'] = 1; res['throughput_img_s'] = round(1000 / res['p50_ms'], 1)
    res['peak_rss_mb'] = peak_rss_mb()
    res['input'] = [None if not isinstance(inp.shape[0], int) else inp.shape[0]] + hw; res['dtype'] = str(dtype.__name__)
    print(json.dumps(res), flush=True)


def peak_rss_mb():
    try:
        import resource                                   # ru_maxrss: KiB on Linux, bytes on macOS
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / 2 ** 20 if sys.platform == 'darwin' else r / 1024, 1)
    except ImportError:
        import psutil                                     # Windows: peak working set
        return round(psutil.Process().memory_info().peak_wset / 2 ** 20, 1)


def gpu_used_mb():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5).stdout
    return int(out.strip().splitlines()[0])


def gpu_name():
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    except Exception:
        return None


def watch_vram(pid, peak, stop):
    while not stop.is_set():
        try:
            out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, timeout=5).stdout
            for line in out.splitlines():
                p, m = [v.strip() for v in line.split(',')]
                if int(p) == pid and m.isdigit(): peak[0] = max(peak[0], int(m))
            peak[1] = max(peak[1], gpu_used_mb())
        except Exception:
            pass
        stop.wait(0.2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('models', nargs='*'); ap.add_argument('--providers', default='cpu')
    ap.add_argument('--threads', default='1,4'); ap.add_argument('--batch', type=int, default=32); ap.add_argument('--runs', type=int, default=100)
    ap.add_argument('--out', default='bench-results.jsonl'); ap.add_argument('--child', nargs=5)
    a = ap.parse_args()
    if a.child:
        path, prov, thr, b, r = a.child; child(path, prov, int(thr), int(b), int(r)); return
    import onnxruntime as ort
    host = dict(cpu=platform.processor() or platform.machine(), cpu_count=os.cpu_count(), gpu=gpu_name(), ort=ort.__version__,
                os=f'{platform.system()} {platform.release()}')
    try:
        host['cpu'] = next(l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name'))
    except Exception:
        try:
            host['cpu'] = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip() or host['cpu']
        except Exception:
            pass
    print(json.dumps(dict(host=host)), flush=True)
    with open(a.out, 'a') as fo:
        for m in a.models:
            for prov in a.providers.split(','):
                for thr in (a.threads.split(',') if prov == 'cpu' else ['1']):
                    cmd = [sys.executable, __file__, '--child', m, prov, thr, str(a.batch if prov in ('cuda', 'coreml', 'trt') else 8), str(a.runs if prov in ('cuda', 'coreml', 'trt') else max(20, a.runs // 3))]
                    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    base = gpu_used_mb() if prov in ('cuda', 'trt') else 0
                    peak, stop = [0, 0], threading.Event()
                    th = threading.Thread(target=watch_vram, args=(p.pid, peak, stop), daemon=True)
                    if prov in ('cuda', 'trt'): th.start()
                    out, err = p.communicate(); stop.set()
                    row = dict(model=os.path.basename(m), provider=prov, threads=int(thr), host=host)
                    try:
                        row.update(json.loads(out.strip().splitlines()[-1]))
                    except Exception:
                        row['error'] = (err or out)[-300:]
                    if prov in ('cuda', 'trt'):
                        row['peak_vram_mb'] = peak[0] if peak[0] else max(0, peak[1] - base)
                        row['vram_method'] = 'per-process' if peak[0] else 'device-delta'
                    print(json.dumps({k: v for k, v in row.items() if k != 'host'}), flush=True); fo.write(json.dumps(row) + '\n'); fo.flush()


if __name__ == '__main__':
    main()
