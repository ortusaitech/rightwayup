"""Per-format tier thresholds (lead, 29 Sep; FREEZE v2): the release Tier.calibrate, unchanged, on the per-format
calibration stores written by cal_onnx.py (FMT-<model>-<format>-cal-c<canvas>). Nano = SV2 @112; Fast / Balanced /
Pro small stage = kdM3 @224; large stage and Max = M3 @280; both cascade stages in the same format.
  PYTHONPATH=/workspace/release-code python fmt_thresholds.py fp32 int8 fp16 m-int8 m-fp16 [--json OUT]
"""
import argparse, json

from rotlab.final_stats import Tier


def tiers(fmt):
    m = 'm' if fmt.startswith('m-') else ''          # 'm-<fmt>': batch-capable fillmask small files (same large stage)
    f = fmt[2:] if m else fmt
    sv2, kd, m3 = f'FMT-SV2{m}-{f}', f'FMT-kdM3{m}-{f}', f'FMT-M3-{f}'
    spec = [('nano', Tier('Nano', sv2), 'c112', None), ('fast', Tier('Fast', kd), 'c224', None),
            ('balanced', Tier('Balanced', kd, m3, 0.10), 'c224', 'c280'), ('pro', Tier('Pro', kd, m3, 0.20), 'c224', 'c280'),
            ('max', Tier('Max', m3), 'c280', None)]
    out = {}
    for key, t, scv, lcv in spec:
        t.calibrate(scv, lcv or 'c224')
        d = dict(standard=round(t.t_cov90, 3), strict=round(max(t.t_strict, t.t_cov90), 3), strict_raw=round(t.t_strict, 4))
        if t.large:
            d['route'] = round(t.t_route, 3)
        out[key] = d
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('formats', nargs='+'); ap.add_argument('--json')
    a = ap.parse_args(); res = {f: tiers(f) for f in a.formats}
    for f, t in res.items():
        print(f, json.dumps(t))
    if a.json:
        open(a.json, 'w').write(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
