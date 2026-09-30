#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 -m venv .venv && . .venv/bin/activate && pip install -q -r requirements.txt
M=models
python bench.py $M/nano-s112-fp32.onnx $M/nano-s112-fp16.onnx $M/fast-s224-fp32.onnx $M/fast-s224-fp16.onnx $M/large-l224-fp32.onnx $M/large-l224-fp16.onnx $M/incumbent/rotation_estimator_fp32.onnx --providers cuda --batch 64 --runs 200 --out bench-results.jsonl
python bench.py $M/nano-s112-int8.onnx $M/fast-s224-int8.onnx $M/large-l224-int8.onnx $M/incumbent/rotation_estimator_fp32.onnx --providers cpu --threads 1,4 --runs 60 --out bench-results.jsonl
echo "Done: bench-results.jsonl"
