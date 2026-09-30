"""A9 (release/results/protected-v2/PREREG.md): v2 release predictions on the 24 Sep held-out views. Writes raw
per-view outputs only; metrics are computed afterwards by a separate script, once."""
import io, os, pickle, sys, time, hashlib
import numpy as np
from PIL import Image
os.environ['RIGHTWAYUP_MODEL_DIR'] = '/workspace/a9/models'
sys.path.insert(0, '/workspace/a9/pkg')
import rightwayup as rw
print('core.py sha256', hashlib.sha256(open('/workspace/a9/pkg/rightwayup/core.py', 'rb').read()).hexdigest())
out = {}
for s in ['meva_frozen', 'v1_frozen']:
    f = f'/workspace/rotation-data/rotlab/cviews/{s}.pkl'
    print(s, 'pickle sha256', hashlib.sha256(open(f, 'rb').read()).hexdigest(), flush=True)
    d = pickle.loads(open(f, 'rb').read())
    ims = [Image.open(io.BytesIO(b)).convert('RGB') for b in d['png']]
    out[f'{s}_theta'] = np.asarray(d['theta'], np.float64); out[f'{s}_group'] = np.asarray(d['group']); out[f'{s}_variant'] = np.asarray(d['variant'])
    for tier in ['pico', 'nano', 'fast', 'balanced', 'pro', 'max']:
        for mode in ['standard', 'strict']:
            o = rw.Orienter(tier, device='cuda', precision='fp32', abstain=mode); t0 = time.time(); rs = []
            for i in range(0, len(ims), 64):
                rs += o.predict_batch(ims[i:i + 64])
            if mode == 'standard':
                out[f'{s}_{tier}_pred'] = np.array([r.angle_cw for r in rs]); out[f'{s}_{tier}_conf'] = np.array([r.confidence for r in rs])
            else:
                assert np.allclose(out[f'{s}_{tier}_pred'], [r.angle_cw for r in rs])
            out[f'{s}_{tier}_abstain_{mode}'] = np.array([r.abstain for r in rs])
            print(s, tier, mode, len(rs), f'{time.time() - t0:.0f}s', flush=True)
np.savez('/workspace/a9/v2-predictions.npz', **out)
print('A9-PREDICT-DONE', hashlib.sha256(open('/workspace/a9/v2-predictions.npz', 'rb').read()).hexdigest())
