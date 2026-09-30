# Orientation-model benchmark bundle

Copy `rotlab/bench.py` next to these scripts and the models into `models/` (file names and SHA-256 in
`release/manifests/RELEASE-ARTIFACTS-SHA256.txt`). The models are not stored in git.

Measures CPU and GPU latency, throughput and memory for the candidate tiers and the previous model.

## Requirements
- Python 3.10–3.12
- NVIDIA GPU: a recent driver (CUDA 12 capable). `onnxruntime-gpu[cuda,cudnn]` pulls in the CUDA/cuDNN runtime libraries itself. On Windows pin `nvidia-cudnn-cu12==9.8.0.87`
  (cuDNN 9.26 fails with ONNX Runtime 1.22 and silently falls back to CPU). For TensorRT: `pip install tensorrt-cu12==10.9.0.34`.

## Run (Windows)
    run-windows.bat
## Run (Linux)
    bash run-linux.sh

Results are written to `bench-results.jsonl` (one JSON line per model × device × threads). Please send that file back.
Takes about 10–15 minutes. Close other GPU-heavy applications first, and keep the laptop on mains power with the
"best performance" power mode.
