"""Weight-only 8-bit quantization of a large FP32 ONNX export (lead, 28 Sep 18:30). The default dynamic INT8 also quantizes
activations per tensor; ViT activation outliers make that lossy and it amplified the Max's residual probe lean. MatMulNBits
with 8-bit block-wise weights keeps activations in float (accuracy_level 0) or quantizes them block-wise (accuracy_level 4).
  python -m rotlab.int8_wo BASE   (reads BASE-fp32.onnx; writes BASE-w8b{32,128}-a{0,4}.onnx)"""
import sys
import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

base = sys.argv[1]
for bs, acc in ((128, 0), (32, 0), (128, 4)):
    m = onnx.load(base + '-fp32.onnx')
    q = MatMulNBitsQuantizer(m, bits=8, block_size=bs, is_symmetric=True, accuracy_level=acc)
    q.process()
    out = f'{base}-w8b{bs}-a{acc}.onnx'
    q.model.save_model_to_file(out, use_external_data_format=False)
    print('wrote', out, flush=True)
