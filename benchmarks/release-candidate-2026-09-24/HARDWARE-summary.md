# Hardware summary (provisional; ONNX Runtime 1.22)

> Terminology note (added 30 Sep 2026): in this record, 'v2' denotes the RightWayUp 1.0 release weights and 'v1' the 24 Sep release candidate; the record itself is unchanged.


**Single** = latency of one image processed on its own (batch 1, p50), what a per-camera or interactive check sees.
**Batched** = sustained throughput when many images are processed together (batch size in brackets). Batching amortises
fixed per-call overhead and fills the GPU, so throughput is far higher than 1000 / single-image latency.

## GPUs — recommended runtime per model

Our models: TensorRT FP16 (accuracy parity with PyTorch verified, see GPU-PARITY.md). Woehrer 2026: ONNX Runtime CUDA FP32 (its released graph has a fixed batch of 1, so throughput is batch-1).

| GPU | Model | Latency, single image (batch 1) | Throughput, batched | Peak VRAM |
|---|---|---|---|---|
| RTX PRO 4500 | Nano (TensorRT FP16) | 0.91 ms | 26,563 img/s (batch 64) | 970 MiB |
| RTX PRO 4500 | Fast (TensorRT FP16) | 1.04 ms | 6,683 img/s (batch 64) | 1,012 MiB |
| RTX PRO 4500 | Large (TensorRT FP16) | 3.64 ms | 713 img/s (batch 64) | 3,058 MiB |
| RTX PRO 4500 | Woehrer 2026 (CUDA FP32) | 4.04 ms | 248 img/s (batch 1) | 756 MiB |
| RTX 4000 Ada | Nano (TensorRT FP16) | 1.16 ms | 14,630 img/s (batch 64) | 744 MiB |
| RTX 4000 Ada | Fast (TensorRT FP16) | 1.22 ms | 3,118 img/s (batch 64) | 906 MiB |
| RTX 4000 Ada | Large (TensorRT FP16) | 6.2 ms | 353 img/s (batch 64) | 1,620 MiB |
| RTX 4000 Ada | Woehrer 2026 (CUDA FP32) | 6.67 ms | 150 img/s (batch 1) | 648 MiB |
| L4 | Nano (TensorRT FP16) | 1.57 ms | 13,251 img/s (batch 64) | 796 MiB |
| L4 | Fast (TensorRT FP16) | 1.18 ms | 2,461 img/s (batch 64) | 942 MiB |
| L4 | Large (TensorRT FP16) | 7.34 ms | 254 img/s (batch 64) | 1,656 MiB |
| L4 | Woehrer 2026 (CUDA FP32) | 7.28 ms | 137 img/s (batch 1) | 678 MiB |
| RTX 3060 Laptop | Nano (TensorRT FP16) | 4.98 ms | 1,626 img/s (batch 32) | 699 MiB |
| RTX 3060 Laptop | Fast (TensorRT FP16) | 4.23 ms | 1,065 img/s (batch 32) | 707 MiB |
| RTX 3060 Laptop | Large (TensorRT FP16) | 9.96 ms | 157 img/s (batch 16) | 1,955 MiB |
| RTX 3060 Laptop | Woehrer 2026 (CUDA FP32) | 38.41 ms | 26 img/s (batch 1) | 585 MiB |

## CPUs — INT8 for our models, FP32 for Woehrer 2026, 4 threads

| CPU | Model | Latency, single image (batch 1) | Throughput, batched (batch 8) | Peak RAM |
|---|---|---|---|---|
| Apple M4 | Nano | 8.37 ms | 156 img/s | 119 MiB |
| Apple M4 | Fast | 30.1 ms | 25 img/s | 189 MiB |
| Apple M4 | Large | 292.13 ms | 3 img/s | 866 MiB |
| Apple M4 | Woehrer 2026 | 138.09 ms | 7 img/s | 436 MiB |
| Intel Core i7-1260P | Nano | 7.91 ms | 166 img/s | 110 MiB |
| Intel Core i7-1260P | Fast | 29.1 ms | 43 img/s | 174 MiB |
| Intel Core i7-1260P | Large | 206.88 ms | 4 img/s | 627 MiB |
| Intel Core i7-1260P | Woehrer 2026 | 160.87 ms | 6 img/s | 437 MiB |
| AMD EPYC 7663 | Nano | 20.44 ms | 122 img/s | 119 MiB |
| AMD EPYC 7663 | Fast | 43.67 ms | 33 img/s | 174 MiB |
| AMD EPYC 7663 | Large | 338.22 ms | 4 img/s | 764 MiB |
| AMD EPYC 7663 | Woehrer 2026 | 150.26 ms | 7 img/s | 425 MiB |
| AMD EPYC 7542 | Nano | 22.88 ms | 76 img/s | 122 MiB |
| AMD EPYC 7542 | Fast | 64.07 ms | 20 img/s | 176 MiB |
| AMD EPYC 7542 | Large | 598.15 ms | 2 img/s | 767 MiB |
| AMD EPYC 7542 | Woehrer 2026 | 200.25 ms | 5 img/s | 427 MiB |
| AMD EPYC 7352 | Nano | 29.82 ms | 77 img/s | 124 MiB |
| AMD EPYC 7352 | Fast | 63.12 ms | 20 img/s | 175 MiB |
| AMD EPYC 7352 | Large | 543.41 ms | 2 img/s | 764 MiB |
| AMD EPYC 7352 | Woehrer 2026 | 218.88 ms | 5 img/s | 434 MiB |

## Apple M4 — native Core ML (FP16), Neural Engine

| Model | Latency, single image (batch 1) | Throughput, batched (batch 16) |
|---|---|---|
| Nano | 1.09 ms | 1,511 img/s |
| Fast | 3.03 ms | 290 img/s |
| Large | 32.94 ms | 32 img/s |

Notes: TensorRT processes use 5–12 GB of host RAM (TensorRT libraries and engine building); plain ONNX Runtime CUDA uses ~1 GB and is ~2x slower. Windows GPU latency includes ~2–3 ms of driver overhead. The i7-12700H laptop's CPU figures are omitted (timings varied 5–6x between runs under a Windows service session on a hybrid P/E-core CPU); HARDWARE-raw.md has every configuration.
