"""Command line: `rightwayup predict frame.jpg`, `rightwayup fix photo.jpg`."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

from . import __version__
from .core import TIERS, Orienter


def _common(p):
    p.add_argument('images', nargs='+', type=Path)
    p.add_argument('--tier', default='max', choices=list(TIERS))
    p.add_argument('--abstain', default='standard', choices=['standard', 'strict', 'off'])
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'tensorrt', 'coreml'])
    p.add_argument('--precision', default='auto', choices=['auto', 'fp32', 'fp16', 'int8'])
    p.add_argument('--model-dir', help='folder with the ONNX files (default: download from the Hugging Face Hub)')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--batch', action='store_true', help='use the batch-capable files (throughput); default: one-image files')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='rightwayup', description='Find the right way up, or recognise when the image '
                                 "doesn't define one.")
    ap.add_argument('--version', action='version', version=f'rightwayup {__version__}')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('predict', help='print angle, confidence and abstain flag for each image')
    _common(p); p.add_argument('--json', action='store_true', help='one JSON object per line')
    f = sub.add_parser('fix', help='write upright copies (images the model abstains on are left unchanged)')
    _common(f)
    f.add_argument('--out', type=Path, help='output folder (default: next to each input, with an -upright suffix)')
    f.add_argument('--snap', type=int, choices=[90], help='round the correction to the nearest quarter turn (lossless)')
    f.add_argument('--min-angle', type=float, default=1.0, help='treat smaller corrections as upright (default 1.0)')
    f.add_argument('--force', action='store_true', help='also correct images the model abstains on')
    a = ap.parse_args(argv)

    o = Orienter(a.tier, a.device, a.precision, a.abstain, a.model_dir, batch=a.batch)
    files = [p for p in a.images if p.is_file()]
    for p in a.images:
        if not p.is_file():
            print(f'{p}: not found', file=sys.stderr)
    status = 0
    for i in range(0, len(files), a.batch_size):
        chunk = files[i:i + a.batch_size]
        for path, r in zip(chunk, o.predict_batch(chunk, a.batch_size)):
            if a.cmd == 'predict':
                if a.json:
                    print(json.dumps(dict(file=str(path), **r.to_dict())))
                else:
                    print(f'{path}\t{r.angle_cw:6.1f}°\tconfidence {r.confidence:.3f}\t{"ABSTAIN" if r.abstain else "ok"}')
                continue
            dst = (a.out / path.name) if a.out else path.with_name(f'{path.stem}-upright{path.suffix}')
            dst.parent.mkdir(parents=True, exist_ok=True)
            small = min(r.angle_cw, 360 - r.angle_cw) < a.min_angle
            if r.abstain and not a.force:
                print(f'{path}: abstained (confidence {r.confidence:.3f}); left unchanged', file=sys.stderr); status = 2
                continue
            im = o.correct(path, snap=a.snap, force=True, result=r) if not small else None
            if im is None:
                print(f'{path}: already upright ({r.angle_cw:.1f}°)')
                continue
            im.save(dst, quality=95) if dst.suffix.lower() in ('.jpg', '.jpeg') else im.save(dst)
            print(f'{path} -> {dst}: turned {r.correction_ccw:.1f}° counter-clockwise (confidence {r.confidence:.3f})')
    return status


if __name__ == '__main__':
    sys.exit(main())
