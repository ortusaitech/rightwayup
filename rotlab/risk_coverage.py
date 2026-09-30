#!/usr/bin/env python3
"""Risk-coverage curves for the frozen tiers on the protected sets (from results/protected/ours.npz; no re-scoring).

Views are ranked by the answering model's confidence (cascade routing as released); at each coverage we report the
error rate (> 10° from the true roll) among the answered views. Frozen operating points (standard, strict) marked.

  python rotlab/risk_coverage.py [--out release/results/protected]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]
from rotlab.protected_eval import tier_pred               # noqa: E402
ROOT = HERE.parent / 'release' / 'results' / 'protected'   # archived protected outputs (the run's volume is gone)

TIERS = ['Nano', 'Fast', 'Balanced', 'Max']
SETS = [('meva_frozen', 'Unseen CCTV cameras (MEVA frozen, 560 views)'),
        ('v1_frozen', 'Clean v1 frozen (DIODE scene 12, MEVA G339, Poly Haven; 2,854 views)')]


def circ_err(p, t):
    d = np.abs((p - t) % 360); return np.minimum(d, 360 - d)


def curve(conf, wrong):
    o = np.argsort(-conf, kind='stable'); w = wrong[o].astype(float)
    k = np.arange(1, len(w) + 1); return k / len(w), np.cumsum(w) / k


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(ROOT)); a = ap.parse_args()
    z = np.load(ROOT / 'ours.npz'); th = {k: tuple(v) for k, v in json.loads((ROOT / 'thresholds.json').read_text()).items()}
    out = Path(a.out); rows, pts, summary = [], {}, []
    for s, title in SETS:
        theta = z[f'{s}_theta']
        for nm in TIERS:
            p, c, _ = tier_pred(z, s, nm, th); wrong = circ_err(p, theta) > 10
            cov, risk = curve(c, wrong); aurc = float(risk.mean())
            for cv, rk in zip(cov, risk):
                rows.append(f'{s},{nm},{cv:.5f},{rk:.5f}')
            ops = []
            for idx, lab in ((1, 'standard'), (2, 'strict')):
                t = th[nm][idx] if idx == 1 else max(th[nm][1], th[nm][2])   # strict is never looser than standard
                keep = c >= t
                ops.append((lab, float(keep.mean()), float(wrong[keep].mean()) if keep.any() else float('nan')))
            pts[(s, nm)] = (cov, risk, ops)
            summary.append((title, nm, float(wrong.mean()), aurc, ops))
    (out / 'risk-coverage.csv').write_text('set,tier,coverage,risk\n' + '\n'.join(rows) + '\n')
    md = ['# Risk-coverage on the protected sets (frozen weights)', '',
          'Views ranked by the answering model\'s confidence (probability mass within ±10°, cascade routing as released).',
          'Risk = share of answered views more than 10° off. AURC = mean risk over all coverages (lower is better).',
          'Operating points use the thresholds frozen on calibration data (`FREEZE.md`); no test data chose them.', '']
    for title in dict.fromkeys(t for t, *_ in summary):
        md += [f'## {title}', '', '| Tier | risk at 100% coverage | AURC | standard: answered / risk | strict: answered / risk |',
               '|---|---|---|---|---|']
        for t, nm, r100, aurc, ops in summary:
            if t == title:
                (_, c1, r1), (_, c2, r2) = ops
                md.append(f'| {nm} | {100 * r100:.1f}% | {100 * aurc:.2f}% | {100 * c1:.1f}% / {100 * r1:.1f}% | {100 * c2:.1f}% / {100 * r2:.1f}% |')
        md.append('')
    (out / 'RISK-COVERAGE.md').write_text('\n'.join(md))
    plot(pts, out / 'risk-coverage.png')
    print('\n'.join(md))


def plot(pts, path):
    import skia
    W, H = 1600, 700; surf = skia.Surface(W, H); c = surf.getCanvas(); c.clear(skia.Color(255, 255, 255))
    fdir = Path('/tmp/claude-1000/fonts/extras/ttf')
    tf = {b: skia.Typeface.MakeFromFile(str(fdir / ('Inter-SemiBold.ttf' if b else 'Inter-Regular.ttf'))) for b in (False, True)}

    def font(size, bold=False):
        return skia.Font(tf[bold], size)

    def ink(r, g, b, a=255, w=None):
        p = skia.Paint(AntiAlias=True, Color=skia.Color(r, g, b, a))
        if w is not None:
            p.setStyle(skia.Paint.kStroke_Style); p.setStrokeWidth(w)
        return p
    cols = {'Nano': (148, 163, 184), 'Fast': (59, 130, 246), 'Balanced': (16, 185, 129), 'Max': (234, 88, 12)}
    for j, (s, title) in enumerate(SETS):
        x0, y0, pw, ph = 110 + j * 780, 90, 640, 500
        c.drawString(title.split(' (')[0], x0, 50, font(26, True), ink(15, 23, 42))
        c.drawString('(' + title.split(' (')[1], x0, 76, font(17), ink(100, 116, 139))
        ymax = 0.06
        for k in range(0, 7, 1):
            y = y0 + ph - ph * (k / 100) / ymax
            c.drawLine(x0, y, x0 + pw, y, ink(226, 232, 240, w=1)); c.drawString(f'{k}%', x0 - 50, y + 6, font(16), ink(100, 116, 139))
        for k in range(0, 101, 20):
            x = x0 + pw * k / 100
            c.drawLine(x, y0, x, y0 + ph, ink(241, 245, 249, w=1)); c.drawString(f'{k}%', x - 14, y0 + ph + 28, font(16), ink(100, 116, 139))
        c.drawString('coverage (share of views answered)', x0 + pw / 2 - 140, y0 + ph + 58, font(17), ink(71, 85, 105))
        if j == 0:
            c.save(); c.rotate(-90); c.drawString('error rate among answered (> 10°)', -(y0 + ph / 2 + 140), 30, font(17), ink(71, 85, 105)); c.restore()
        for nm in TIERS:
            cov, risk, ops = pts[(s, nm)]; pth = skia.Path(); first = True
            for cv, rk in zip(cov[::max(1, len(cov) // 400)], risk[::max(1, len(cov) // 400)]):
                x, y = x0 + pw * cv, y0 + ph - ph * min(rk, ymax) / ymax
                (pth.moveTo if first else pth.lineTo)(x, y); first = False
            c.drawPath(pth, ink(*cols[nm], w=3.2 if nm == 'Max' else 2.2))
            for lab, cv, rk in ops:
                x, y = x0 + pw * cv, y0 + ph - ph * min(rk, ymax) / ymax
                if lab == 'standard':
                    c.drawCircle(x, y, 6.5, ink(*cols[nm]))
                else:
                    c.drawRect(skia.Rect(x - 6, y - 6, x + 6, y + 6), ink(*cols[nm]))
        if j == 1:
            ly = y0 + 20
            for nm in TIERS:
                c.drawLine(x0 + pw - 170, ly, x0 + pw - 140, ly, ink(*cols[nm], w=3)); c.drawString(nm, x0 + pw - 130, ly + 6, font(17), ink(30, 41, 59)); ly += 28
            c.drawCircle(x0 + pw - 155, ly, 6, ink(71, 85, 105)); c.drawString('standard threshold', x0 + pw - 130, ly + 6, font(16), ink(71, 85, 105)); ly += 26
            c.drawRect(skia.Rect(x0 + pw - 161, ly - 6, x0 + pw - 149, ly + 6), ink(71, 85, 105)); c.drawString('strict threshold', x0 + pw - 130, ly + 6, font(16), ink(71, 85, 105))
    from PIL import Image
    Image.fromarray(surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType)).convert('RGB').save(path)


if __name__ == '__main__':
    main()
