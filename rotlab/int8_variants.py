"""INT8 variants of a large FP32 ONNX export (lead, 28 Sep 17:15): the default per-channel dynamic INT8 amplified the residual
probe lean of the grid-fix Max (soft rho4 0.012 -> 0.110). Variants: 'fr' reduce_range=False; 'x4' / 'x8' keep the MatMuls of
the first 4 / 8 transformer blocks and the head Gemm in FP32; 'l8' the last 8 blocks + head; 'h' only the head; 'mlp' quantizes
only the MLP MatMuls (attention + head in FP32).   python -m rotlab.int8_variants BASE [TAGS]  (BASE-fp32.onnx exists)"""
import re, sys
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

base = sys.argv[1]; src = base + '-fp32.onnx'; only = sys.argv[2].split(',') if len(sys.argv) > 2 else None
g = onnx.load(src, load_external_data=False).graph
mm = [n.name for n in g.node if n.op_type in ('MatMul', 'Gemm')]
def blk(name):
    m = re.search(r'blocks[._/](\d+)[._/]', name); return int(m.group(1)) if m else None
print(len(mm), 'MatMul/Gemm nodes; sample:', mm[:3], '...', mm[-2:])
head = [n for n in mm if blk(n) is None]
NB = 1 + max(b for b in map(blk, mm) if b is not None)
for tag, kw in (('fr', dict(reduce_range=False, nodes_to_exclude=[])),
                ('x4', dict(reduce_range=True, nodes_to_exclude=[n for n in mm if (blk(n) is not None and blk(n) < 4)] + head)),
                ('x8', dict(reduce_range=True, nodes_to_exclude=[n for n in mm if (blk(n) is not None and blk(n) < 8)] + head)),
                ('l8', dict(reduce_range=True, nodes_to_exclude=[n for n in mm if (blk(n) is not None and blk(n) >= NB - 8)] + head)),
                ('h', dict(reduce_range=True, nodes_to_exclude=head)),
                ('mlp', dict(reduce_range=True, nodes_to_exclude=[n for n in mm if '/attn/' in n] + head))):
    if only and tag not in only:
        continue
    print(tag, 'excluded', len(kw['nodes_to_exclude']), flush=True)
    quantize_dynamic(src, f'{base}-int8{tag}.onnx', weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=['MatMul', 'Gemm'], **kw)
