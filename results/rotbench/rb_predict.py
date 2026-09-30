"""RotBench v2 (release/results/rotbench-v2/PREREG.md): rebuild the 24 Sep views (rotlab/competitors.py _views logic,
unchanged) and write raw v2 release predictions for all six tiers. Metrics are computed afterwards, once."""
import glob, hashlib, io, os, sys, time
import numpy as np, pandas as pd
from PIL import Image
from huggingface_hub import snapshot_download
REV = '4ebf3d3b79830316955708dbd4c565c5be611b68'
d = snapshot_download('tianyin/RotBench', repo_type='dataset', revision=REV)
imgs = []
for f in sorted(glob.glob(f'{d}/**/*.parquet', recursive=True)):
    print(f.split('/')[-1], hashlib.sha256(open(f, 'rb').read()).hexdigest(), flush=True)
    sp = 'small' if 'small' in f.lower() else 'large'
    for rec in pd.read_parquet(f)['image']:
        imgs.append((sp, Image.open(io.BytesIO(rec['bytes'])).convert('RGB')))
os.environ['RIGHTWAYUP_MODEL_DIR'] = '/workspace/a9/models'
sys.path.insert(0, '/workspace/a9/pkg')
import rightwayup as rw
print('core.py sha256', hashlib.sha256(open('/workspace/a9/pkg/rightwayup/core.py', 'rb').read()).hexdigest(), flush=True)
out = {}
for name in ['rotbench', 'rotbench_full']:
    views, th, split = [], [], []
    for sp, im in imgs:
        for ccw in (0, 90, 180, 270):
            views.append(im.rotate(ccw, expand=(name == 'rotbench_full')) if ccw else im); th.append((-ccw) % 360); split.append(sp)
    out[f'{name}_theta'] = np.array(th, float); out[f'{name}_split'] = np.array(split)
    for tier in ['pico', 'nano', 'fast', 'balanced', 'pro', 'max']:
        o = rw.Orienter(tier, device='cuda', precision='fp32'); t0 = time.time(); rs = []
        for i in range(0, len(views), 64):
            rs += o.predict_batch(views[i:i + 64])
        out[f'{name}_{tier}_pred'] = np.array([r.angle_cw for r in rs]); out[f'{name}_{tier}_conf'] = np.array([r.confidence for r in rs])
        out[f'{name}_{tier}_abstain'] = np.array([r.abstain for r in rs])
        print(name, tier, len(rs), f'{time.time() - t0:.0f}s', flush=True)
os.makedirs('/workspace/rb', exist_ok=True)
np.savez('/workspace/rb/v2-rotbench-predictions.npz', **out)
print('RB-PREDICT-DONE', hashlib.sha256(open('/workspace/rb/v2-rotbench-predictions.npz', 'rb').read()).hexdigest(), flush=True)
