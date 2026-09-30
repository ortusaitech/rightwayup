#!/usr/bin/env python3
"""Export RotNet to ONNX (FP32) and a dynamic-INT8 copy. Output: probabilities over 360 bins.
  python -m rotlab.export CKPT_OR_random OUT_PREFIX [--img-size 224] [--ema]
"""
import argparse, copy, torch
from rotlab.model import RotNet


class Prob(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x).softmax(1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('ckpt'); ap.add_argument('out'); ap.add_argument('--img-size', type=int, default=224); ap.add_argument('--ema', action='store_true'); ap.add_argument('--arch', default='s'); ap.add_argument('--canvas', type=int, default=0, help='export at this square size (pos-embed baked)')
    a = ap.parse_args()
    if a.ckpt == 'random':
        m = RotNet(img_size=a.img_size, pretrained=False, arch=a.arch)
    else:
        ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
        cv = a.canvas or ck['config'].get('canvas', ck['config']['img_size'])
        sd = ck['ema' if a.ema else 'model']
        if cv != ck['config']['img_size']:
            # bake the runtime pos-embed resize (same timm function) into a static-size model
            from timm.layers import resample_abs_pos_embed
            g = cv // 14
            sd = dict(sd); sd['backbone.pos_embed'] = resample_abs_pos_embed(sd['backbone.pos_embed'], (g, g), num_prefix_tokens=1)
        m = RotNet(img_size=cv, pretrained=False, arch=ck['config'].get('arch', 's')); a.img_size = cv
        m.load_state_dict(sd)
    m = Prob(m.eval())
    x = torch.randn(1, 3, a.img_size, a.img_size)
    torch.onnx.export(m, x, a.out + '-fp32.onnx', input_names=['image'], output_names=['prob'], opset_version=18, dynamo=False,
                      dynamic_axes={'image': {0: 'batch'}, 'prob': {0: 'batch'}})
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(a.out + '-fp32.onnx', a.out + '-int8.onnx', weight_type=QuantType.QInt8, per_channel=True, reduce_range=True, op_types_to_quantize=['MatMul', 'Gemm'])
    if torch.cuda.is_available():   # FP16 copy for GPUs: half weights/compute, float32 input and output kept
        class Half(torch.nn.Module):
            def __init__(self, inner): super().__init__(); self.inner = inner.half()
            def forward(self, x): return self.inner(x.half()).float()
        h = Half(copy.deepcopy(m)).cuda().eval()
        torch.onnx.export(h, x.cuda(), a.out + '-fp16.onnx', input_names=['image'], output_names=['prob'], opset_version=18,
                          dynamo=False, dynamic_axes={'image': {0: 'batch'}, 'prob': {0: 'batch'}})
    else:
        print('no CUDA: FP16 copy skipped')
    print('exported', a.out)


if __name__ == '__main__':
    main()
