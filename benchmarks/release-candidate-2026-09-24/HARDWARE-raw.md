
> Terminology note (added 30 Sep 2026): in this record, 'v2' denotes the RightWayUp 1.0 release weights and 'v1' the 24 Sep release candidate; the record itself is unchanged.


### Intel Core i7-1260P (laptop)

| Model | Runtime | Latency, single image (batch 1), p50 / p95 | Throughput, batched | Peak VRAM | Peak process RAM |
|---|---|---|---|---|---|
| Nano (ViT-S @112) | CPU FP32 ×1t | 32.16 / 37.92 ms | 36 img/s (batch 8) | — | 199 MiB |
| Nano (ViT-S @112) | CPU FP32 ×4t | 13.87 / 16.74 ms | 95 img/s (batch 8) | — | 192 MiB |
| Nano (ViT-S @112) | CPU INT8 ×1t | 11.23 / 11.52 ms | 79 img/s (batch 8) | — | 110 MiB |
| Nano (ViT-S @112) | CPU INT8 ×4t | 7.91 / 11.35 ms | 166 img/s (batch 8) | — | 110 MiB |
| Fast (ViT-S @224) | CPU FP32 ×1t | 141.36 / 159.29 ms | 7 img/s (batch 8) | — | 275 MiB |
| Fast (ViT-S @224) | CPU FP32 ×4t | 57.93 / 65.29 ms | 14 img/s (batch 8) | — | 279 MiB |
| Fast (ViT-S @224) | CPU INT8 ×1t | 53.53 / 66.56 ms | 16 img/s (batch 8) | — | 172 MiB |
| Fast (ViT-S @224) | CPU INT8 ×4t | 29.1 / 38.29 ms | 43 img/s (batch 8) | — | 174 MiB |
| Large (ViT-L @224) | CPU FP32 ×1t | 1656.36 / 1771.86 ms | 1 img/s (batch 8) | — | 1,999 MiB |
| Large (ViT-L @224) | CPU FP32 ×4t | 616.51 / 744.46 ms | 1 img/s (batch 8) | — | 2,066 MiB |
| Large (ViT-L @224) | CPU INT8 ×1t | 534.16 / 674.69 ms | 2 img/s (batch 8) | — | 634 MiB |
| Large (ViT-L @224) | CPU INT8 ×4t | 206.88 / 320.05 ms | 4 img/s (batch 8) | — | 627 MiB |
| Woehrer 2026 (MambaOut-B) | CPU FP32 ×1t | 368.8 / 436.01 ms | 3 img/s (batch 1) | — | 437 MiB |
| Woehrer 2026 (MambaOut-B) | CPU FP32 ×4t | 160.87 / 183.4 ms | 6 img/s (batch 1) | — | 437 MiB |

### NVIDIA L4

| Model | Runtime | Latency, single image (batch 1), p50 / p95 | Throughput, batched | Peak VRAM | Peak process RAM |
|---|---|---|---|---|---|
| Nano (ViT-S @112) | CUDA FP16 | 2.51 / 2.85 ms | 7,641 img/s (batch 64) | 288 MiB | 938 MiB |
| Nano (ViT-S @112) | CUDA FP32 | 2.33 / 2.65 ms | 3,898 img/s (batch 64) | 552 MiB | 730 MiB |
| Nano (ViT-S @112) | TensorRT FP16 | 1.57 / 1.59 ms | 13,251 img/s (batch 64) | 796 MiB | 5,467 MiB |
| Nano (ViT-S @112) | TensorRT FP16 | 1.29 / 1.3 ms | 11,954 img/s (batch 128) | 898 MiB | 5,471 MiB |
| Fast (ViT-S @224) | CUDA FP16 | 2.35 / 2.4 ms | 1,170 img/s (batch 64) | 692 MiB | 940 MiB |
| Fast (ViT-S @224) | CUDA FP32 | 2.26 / 2.6 ms | 568 img/s (batch 64) | 1,170 MiB | 778 MiB |
| Fast (ViT-S @224) | TensorRT FP16 | 1.18 / 1.2 ms | 2,461 img/s (batch 64) | 942 MiB | 5,474 MiB |
| Fast (ViT-S @224) | TensorRT FP16 | 1.36 / 1.37 ms | 1,989 img/s (batch 128) | 1,550 MiB | 5,473 MiB |
| Large (ViT-L @224) | CUDA FP16 | 7.85 / 7.93 ms | 146 img/s (batch 64) | 1,892 MiB | 992 MiB |
| Large (ViT-L @224) | CUDA FP32 | 13.05 / 13.21 ms | 78 img/s (batch 64) | 3,440 MiB | 1,536 MiB |
| Large (ViT-L @224) | TensorRT FP16 | 7.34 / 7.36 ms | 254 img/s (batch 64) | 1,656 MiB | 10,544 MiB |
| Large (ViT-L @224) | TensorRT FP16 | 7.4 / 7.64 ms | 253 img/s (batch 128) | 3,992 MiB | 10,546 MiB |
| Woehrer 2026 (MambaOut-B) | CUDA FP32 | 7.28 / 7.58 ms | 137 img/s (batch 1) | 678 MiB | 771 MiB |

### AMD EPYC 7542 32-Core Processor

| Model | Runtime | Latency, single image (batch 1), p50 / p95 | Throughput, batched | Peak VRAM | Peak process RAM |
|---|---|---|---|---|---|
| Nano (ViT-S @112) | CPU INT8 ×1t | 34.76 / 35.31 ms | 29 img/s (batch 8) | — | 124 MiB |
| Nano (ViT-S @112) | CPU INT8 ×4t | 22.88 / 25.98 ms | 76 img/s (batch 8) | — | 122 MiB |
| Fast (ViT-S @224) | CPU INT8 ×1t | 143.58 / 144.6 ms | 6 img/s (batch 8) | — | 177 MiB |
| Fast (ViT-S @224) | CPU INT8 ×4t | 64.07 / 67.34 ms | 20 img/s (batch 8) | — | 176 MiB |
| Large (ViT-L @224) | CPU INT8 ×1t | 1771.64 / 1795.26 ms | 0 img/s (batch 8) | — | 765 MiB |
| Large (ViT-L @224) | CPU INT8 ×4t | 598.15 / 618.06 ms | 2 img/s (batch 8) | — | 767 MiB |
| Woehrer 2026 (MambaOut-B) | CPU FP32 ×1t | 409.91 / 445.82 ms | 2 img/s (batch 1) | — | 427 MiB |
| Woehrer 2026 (MambaOut-B) | CPU FP32 ×4t | 200.25 / 209.48 ms | 5 img/s (batch 1) | — | 427 MiB |

### Apple M4

| Model | Runtime | Latency, single image (batch 1), p50 / p95 | Throughput, batched | Peak VRAM | Peak process RAM |
|---|---|---|---|---|---|
| Nano (ViT-S @112) | CPU FP32 ×1t | 28.31 / 28.37 ms | 37 img/s (batch 8) | — | 275 MiB |
| Nano (ViT-S @112) | CPU FP32 ×4t | 9.8 / 9.86 ms | 125 img/s (batch 8) | — | 233 MiB |
| Nano (ViT-S @112) | CPU INT8 ×1t | 20.26 / 21.31 ms | 50 img/s (batch 8) | — | 140 MiB |
| Nano (ViT-S @112) | CPU INT8 ×4t | 8.37 / 8.63 ms | 156 img/s (batch 8) | — | 119 MiB |
| Nano (ViT-S @112) | Core ML ALL FP16 | 2.51 / 2.96 ms | 399 img/s (batch 1) | — | — |
| Nano (ViT-S @112) | Core ML ALL FP16 | 10.59 / 10.66 ms | 1,510 img/s (batch 16) | — | — |
| Nano (ViT-S @112) | Core ML CPU FP16 | 3.58 / 3.79 ms | 280 img/s (batch 1) | — | — |
| Nano (ViT-S @112) | Core ML CPU FP16 | 23.89 / 26.05 ms | 670 img/s (batch 16) | — | — |
| Nano (ViT-S @112) | Core ML CPU+ANE FP16 | 1.09 / 1.09 ms | 921 img/s (batch 1) | — | — |
| Nano (ViT-S @112) | Core ML CPU+ANE FP16 | 10.59 / 10.62 ms | 1,511 img/s (batch 16) | — | — |
| Nano (ViT-S @112) | Core ML CPU+GPU FP16 | 2.46 / 3.54 ms | 406 img/s (batch 1) | — | — |
| Nano (ViT-S @112) | Core ML CPU+GPU FP16 | 17.93 / 17.96 ms | 893 img/s (batch 16) | — | — |
| Nano (ViT-S @112) | Core ML EP FP16 | 43.59 / 44.86 ms | 32 img/s (batch 16) | — | 645 MiB |
| Fast (ViT-S @224) | CPU FP32 ×1t | 117.8 / 117.99 ms | 8 img/s (batch 8) | — | 380 MiB |
| Fast (ViT-S @224) | CPU FP32 ×4t | 36.14 / 36.72 ms | 29 img/s (batch 8) | — | 317 MiB |
| Fast (ViT-S @224) | CPU INT8 ×1t | 97.68 / 103.42 ms | 11 img/s (batch 8) | — | 203 MiB |
| Fast (ViT-S @224) | CPU INT8 ×4t | 30.1 / 33.16 ms | 25 img/s (batch 8) | — | 189 MiB |
| Fast (ViT-S @224) | Core ML ALL FP16 | 3.02 / 3.04 ms | 331 img/s (batch 1) | — | — |
| Fast (ViT-S @224) | Core ML ALL FP16 | 55.23 / 55.34 ms | 290 img/s (batch 16) | — | — |
| Fast (ViT-S @224) | Core ML CPU FP16 | 8.37 / 9.05 ms | 120 img/s (batch 1) | — | — |
| Fast (ViT-S @224) | Core ML CPU FP16 | 146.64 / 150.11 ms | 109 img/s (batch 16) | — | — |
| Fast (ViT-S @224) | Core ML CPU+ANE FP16 | 3.03 / 3.05 ms | 331 img/s (batch 1) | — | — |
| Fast (ViT-S @224) | Core ML CPU+ANE FP16 | 55.08 / 55.23 ms | 290 img/s (batch 16) | — | — |
| Fast (ViT-S @224) | Core ML CPU+GPU FP16 | 6.67 / 6.74 ms | 150 img/s (batch 1) | — | — |
| Fast (ViT-S @224) | Core ML CPU+GPU FP16 | 71.62 / 71.71 ms | 223 img/s (batch 16) | — | — |
| Fast (ViT-S @224) | Core ML EP FP16 | 136.34 / 155.2 ms | 7 img/s (batch 16) | — | 1,417 MiB |
| Large (ViT-L @224) | CPU FP32 ×1t | 1467.86 / 1534.4 ms | 1 img/s (batch 8) | — | 2,516 MiB |
| Large (ViT-L @224) | CPU FP32 ×4t | 412.44 / 416.54 ms | 2 img/s (batch 8) | — | 2,527 MiB |
| Large (ViT-L @224) | CPU INT8 ×1t | 1056.51 / 1103.46 ms | 1 img/s (batch 8) | — | 917 MiB |
| Large (ViT-L @224) | CPU INT8 ×4t | 292.13 / 303.38 ms | 3 img/s (batch 8) | — | 866 MiB |
| Large (ViT-L @224) | Core ML ALL FP16 | 32.96 / 33.17 ms | 30 img/s (batch 1) | — | — |
| Large (ViT-L @224) | Core ML ALL FP16 | 505.76 / 505.87 ms | 32 img/s (batch 16) | — | — |
| Large (ViT-L @224) | Core ML CPU FP16 | 70.23 / 92.2 ms | 14 img/s (batch 1) | — | — |
| Large (ViT-L @224) | Core ML CPU FP16 | 994.66 / 1053.29 ms | 16 img/s (batch 16) | — | — |
| Large (ViT-L @224) | Core ML CPU+ANE FP16 | 32.94 / 33.05 ms | 30 img/s (batch 1) | — | — |
| Large (ViT-L @224) | Core ML CPU+ANE FP16 | 505.58 / 510.78 ms | 32 img/s (batch 16) | — | — |
| Large (ViT-L @224) | Core ML CPU+GPU FP16 | 62.96 / 63.08 ms | 16 img/s (batch 1) | — | — |
| Large (ViT-L @224) | Core ML CPU+GPU FP16 | 798.73 / 826.77 ms | 20 img/s (batch 16) | — | — |
| Large (ViT-L @224) | Core ML EP FP16 | 1639.87 / 1690.95 ms | 1 img/s (batch 16) | — | 8,040 MiB |
| Woehrer 2026 (MambaOut-B) | CPU FP32 ×1t | 346.65 / 348.04 ms | 3 img/s (batch 1) | — | 436 MiB |
| Woehrer 2026 (MambaOut-B) | CPU FP32 ×4t | 138.09 / 143.63 ms | 7 img/s (batch 1) | — | 436 MiB |
| Woehrer 2026 (MambaOut-B) | Core ML EP FP32 | 372.02 / 376.04 ms | 3 img/s (batch 1) | — | 1,084 MiB |

### AMD EPYC 7663 56-Core Processor

| Model | Runtime | Latency, single image (batch 1), p50 / p95 | Throughput, batched | Peak VRAM | Peak process RAM |
|---|---|---|---|---|---|
| Nano (ViT-S @112) | CPU INT8 ×1t | 19.04 / 19.43 ms | 52 img/s (batch 8) | — | 124 MiB |
| Nano (ViT-S @112) | CPU INT8 ×4t | 20.44 / 20.97 ms | 122 img/s (batch 8) | — | 119 MiB |
| Fast (ViT-S @224) | CPU INT8 ×1t | 84.69 / 86.01 ms | 11 img/s (batch 8) | — | 176 MiB |
| Fast (ViT-S @224) | CPU INT8 ×4t | 43.67 / 46.91 ms | 33 img/s (batch 8) | — | 174 MiB |
| Large (ViT-L @224) | CPU INT8 ×1t | 925.08 / 931.1 ms | 1 img/s (batch 8) | — | 768 MiB |
| Large (ViT-L @224) | CPU INT8 ×4t | 338.22 / 346.09 ms | 4 img/s (batch 8) | — | 764 MiB |
| Woehrer 2026 (MambaOut-B) | CPU FP32 ×1t | 360.55 / 362.08 ms | 3 img/s (batch 1) | — | 431 MiB |
| Woehrer 2026 (MambaOut-B) | CPU FP32 ×4t | 150.26 / 151.6 ms | 7 img/s (batch 1) | — | 425 MiB |

### NVIDIA RTX PRO 4500 Blackwell

| Model | Runtime | Latency, single image (batch 1), p50 / p95 | Throughput, batched | Peak VRAM | Peak process RAM |
|---|---|---|---|---|---|
| Nano (ViT-S @112) | CUDA FP16 | 2.09 / 2.16 ms | 12,985 img/s (batch 64) | 338 MiB | 1,088 MiB |
| Nano (ViT-S @112) | CUDA FP32 | 1.9 / 1.97 ms | 9,499 img/s (batch 64) | 626 MiB | 1,788 MiB |
| Nano (ViT-S @112) | TensorRT FP16 | 0.91 / 0.92 ms | 26,563 img/s (batch 64) | 970 MiB | 5,770 MiB |
| Fast (ViT-S @224) | CUDA FP16 | 2.1 / 2.16 ms | 3,047 img/s (batch 64) | 764 MiB | 1,052 MiB |
| Fast (ViT-S @224) | CUDA FP32 | 1.9 / 1.95 ms | 1,697 img/s (batch 64) | 1,238 MiB | 949 MiB |
| Fast (ViT-S @224) | TensorRT FP16 | 1.04 / 1.05 ms | 6,683 img/s (batch 64) | 1,012 MiB | 5,761 MiB |
| Large (ViT-L @224) | CUDA FP16 | 4.42 / 4.48 ms | 413 img/s (batch 64) | 1,964 MiB | 1,100 MiB |
| Large (ViT-L @224) | CUDA FP32 | 5.78 / 5.84 ms | 218 img/s (batch 64) | 3,532 MiB | 1,538 MiB |
| Large (ViT-L @224) | TensorRT FP16 | 3.64 / 3.65 ms | 713 img/s (batch 64) | 3,058 MiB | 11,612 MiB |
| Woehrer 2026 (MambaOut-B) | CUDA FP32 | 4.04 / 4.14 ms | 248 img/s (batch 1) | 756 MiB | 954 MiB |

### NVIDIA GeForce RTX 3060 Laptop GPU

| Model | Runtime | Latency, single image (batch 1), p50 / p95 | Throughput, batched | Peak VRAM | Peak process RAM |
|---|---|---|---|---|---|
| Nano (ViT-S @112) | CUDA FP16 | 4.58 / 4.93 ms | 1,437 img/s (batch 32) | 235 MiB | 1,008 MiB |
| Nano (ViT-S @112) | CUDA FP32 | 4.46 / 5.34 ms | 1,474 img/s (batch 32) | 357 MiB | 823 MiB |
| Nano (ViT-S @112) | TensorRT FP16 | 4.98 / 12.49 ms | 1,626 img/s (batch 32) | 699 MiB | 5,877 MiB |
| Fast (ViT-S @224) | CUDA FP16 | 4.89 / 5.71 ms | 654 img/s (batch 32) | 393 MiB | 1,037 MiB |
| Fast (ViT-S @224) | CUDA FP32 | 4.06 / 5.34 ms | 360 img/s (batch 32) | 675 MiB | 841 MiB |
| Fast (ViT-S @224) | TensorRT FP16 | 4.23 / 10.16 ms | 1,065 img/s (batch 32) | 707 MiB | 5,910 MiB |
| Large (ViT-L @224) | CUDA FP16 | 21.28 / 48.92 ms | 82 img/s (batch 16) | 1,003 MiB | 1,057 MiB |
| Large (ViT-L @224) | CUDA FP32 | 39.22 / 47.93 ms | 43 img/s (batch 16) | 1,837 MiB | 1,596 MiB |
| Large (ViT-L @224) | TensorRT FP16 | 9.96 / 11.62 ms | 157 img/s (batch 16) | 1,955 MiB | 8,963 MiB |
| Woehrer 2026 (MambaOut-B) | CUDA FP32 | 38.41 / 54.07 ms | 26 img/s (batch 1) | 585 MiB | 873 MiB |

### Intel Core i7-12700H (laptop)

| Model | Runtime | Latency, single image (batch 1), p50 / p95 | Throughput, batched | Peak VRAM | Peak process RAM |
|---|---|---|---|---|---|
| Nano (ViT-S @112) | CPU INT8 ×1t | 15.86 / 17.14 ms | 64 img/s (batch 8) | — | 130 MiB |
| Nano (ViT-S @112) | CPU INT8 ×4t | 8.7 / 9.25 ms | 185 img/s (batch 8) | — | 130 MiB |
| Fast (ViT-S @224) | CPU INT8 ×1t | 71.68 / 73.13 ms | 2 img/s (batch 8) | — | 183 MiB |
| Fast (ViT-S @224) | CPU INT8 ×4t | 28.33 / 31.82 ms | 45 img/s (batch 8) | — | 185 MiB |
| Large (ViT-L @224) | CPU INT8 ×1t | 5757.98 / 5802.41 ms | 0 img/s (batch 8) | — | 770 MiB |
| Large (ViT-L @224) | CPU INT8 ×4t | 1655.97 / 1786.79 ms | 1 img/s (batch 8) | — | 770 MiB |
| Woehrer 2026 (MambaOut-B) | CPU FP32 ×1t | 2014.94 / 2184.82 ms | 0 img/s (batch 1) | — | 425 MiB |
| Woehrer 2026 (MambaOut-B) | CPU FP32 ×4t | 133.77 / 746.11 ms | 8 img/s (batch 1) | — | 428 MiB |

### NVIDIA RTX 4000 Ada Generation

| Model | Runtime | Latency, single image (batch 1), p50 / p95 | Throughput, batched | Peak VRAM | Peak process RAM |
|---|---|---|---|---|---|
| Nano (ViT-S @112) | CUDA FP16 | 2.73 / 2.86 ms | 8,613 img/s (batch 64) | 344 MiB | 932 MiB |
| Nano (ViT-S @112) | CUDA FP32 | 2.44 / 2.49 ms | 4,652 img/s (batch 64) | 522 MiB | 735 MiB |
| Nano (ViT-S @112) | TensorRT FP16 | 1.16 / 1.26 ms | 14,630 img/s (batch 64) | 744 MiB | 5,457 MiB |
| Fast (ViT-S @224) | CUDA FP16 | 2.67 / 2.75 ms | 1,494 img/s (batch 64) | 662 MiB | 962 MiB |
| Fast (ViT-S @224) | CUDA FP32 | 2.42 / 2.54 ms | 722 img/s (batch 64) | 1,140 MiB | 781 MiB |
| Fast (ViT-S @224) | TensorRT FP16 | 1.22 / 1.24 ms | 3,118 img/s (batch 64) | 906 MiB | 5,465 MiB |
| Large (ViT-L @224) | CUDA FP16 | 5.99 / 6.15 ms | 199 img/s (batch 64) | 1,862 MiB | 993 MiB |
| Large (ViT-L @224) | CUDA FP32 | 9.83 / 9.9 ms | 103 img/s (batch 64) | 3,410 MiB | 1,540 MiB |
| Large (ViT-L @224) | TensorRT FP16 | 6.2 / 6.57 ms | 353 img/s (batch 64) | 1,620 MiB | 10,550 MiB |
| Woehrer 2026 (MambaOut-B) | CUDA FP32 | 6.67 / 6.87 ms | 150 img/s (batch 1) | 648 MiB | 770 MiB |

### AMD EPYC 7352 24-Core Processor

| Model | Runtime | Latency, single image (batch 1), p50 / p95 | Throughput, batched | Peak VRAM | Peak process RAM |
|---|---|---|---|---|---|
| Nano (ViT-S @112) | CPU INT8 ×1t | 36.58 / 36.84 ms | 28 img/s (batch 8) | — | 123 MiB |
| Nano (ViT-S @112) | CPU INT8 ×4t | 29.82 / 29.99 ms | 77 img/s (batch 8) | — | 124 MiB |
| Fast (ViT-S @224) | CPU INT8 ×1t | 151.57 / 151.73 ms | 6 img/s (batch 8) | — | 176 MiB |
| Fast (ViT-S @224) | CPU INT8 ×4t | 63.12 / 63.5 ms | 20 img/s (batch 8) | — | 175 MiB |
| Large (ViT-L @224) | CPU INT8 ×1t | 1827.23 / 1831.53 ms | 0 img/s (batch 8) | — | 763 MiB |
| Large (ViT-L @224) | CPU INT8 ×4t | 543.41 / 548.43 ms | 2 img/s (batch 8) | — | 764 MiB |
| Woehrer 2026 (MambaOut-B) | CPU FP32 ×1t | 406.66 / 407.2 ms | 2 img/s (batch 1) | — | 435 MiB |
| Woehrer 2026 (MambaOut-B) | CPU FP32 ×4t | 218.88 / 219.28 ms | 5 img/s (batch 1) | — | 434 MiB |

#### Not measured (errors)

- Apple M4 · Nano (ViT-S @112) · Core ML EP FP32: 00gn/T/onnxruntime-29F2705B-25B7-4160-AFF4-99CB123BB740-13715-00003826A8D959A2.mlmodelc/model.mil' with error code: -7.
- Apple M4 · Fast (ViT-S @224) · Core ML EP FP32: 00gn/T/onnxruntime-1E919FEB-DB7D-4794-826A-31B8308E9DC3-13716-00003826A93A2019.mlmodelc/model.mil' with error code: -7.
- Apple M4 · Large (ViT-L @224) · Core ML EP FP32: 00gn/T/onnxruntime-67EE87BF-D3CB-4FFA-B6A8-7DAAD50F0722-13717-00003826A9C86025.mlmodelc/model.mil' with error code: -7.
