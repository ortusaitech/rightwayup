"""WiSE-FT weight interpolation of two same-architecture RotNet checkpoints (lead 30 Sep 2026, research/pico/PREREG.md round 2;
recombination, labelled as such): theta = (1 - a) * A + a * B for every floating tensor of 'model' and 'ema'.
   python -m rotlab.soup A.pt B.pt a OUT.pt"""
import hashlib, os, sys
import torch


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def main(a_path, b_path, a, out):
    a = float(a)
    A = torch.load(a_path, map_location='cpu', weights_only=False); B = torch.load(b_path, map_location='cpu', weights_only=False)
    res = dict(B)
    for key in ('model', 'ema'):
        if key not in B:
            continue
        sa, sb = A[key], B[key]
        assert sa.keys() == sb.keys(), key
        res[key] = {k: ((1 - a) * sa[k].double() + a * sb[k].double()).to(sb[k].dtype) if sb[k].is_floating_point() else sb[k] for k in sb}
    cfg = dict(B['config']); cfg['soup'] = dict(a=a, A=a_path, A_sha256=sha(a_path), B=b_path, B_sha256=sha(b_path),
                                                note='WiSE-FT weight interpolation theta=(1-a)A+aB (PREREG round 2)')
    res['config'] = cfg
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(res, out)
    print('soup', a, '->', out, sha(out)[:16])


if __name__ == '__main__':
    main(*sys.argv[1:5])
