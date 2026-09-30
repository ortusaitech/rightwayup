# Results: single source of truth (RightWayUp 1.0, frozen 29 Sep 2026)

Every release document, model card, post and the paper takes figures from this file. Models: the released grid-fix
tiers (`FREEZE.md`; FROZEN.json `5caf3220…`) plus Pico (added after the freeze, see §1). Section 0 is the sealed final
holdout, opened once after the freeze and scored once; sections 2–4 are development panels (used for selection).
Woehrer 2026 is scored on identical images. Sources: `results/sealed-holdout/FINAL-RESULTS.json` and
`results/sealed-holdout/FINAL-PAIRED.json` (sealed), the development prediction stores (development; internal), `benchmarks/HARDWARE-summary.md`
(speed), `FREEZE.md` (thresholds, per-format addendum).

## 0. Sealed final holdout (never trained on, never used for selection, scored once on 29 Sep 2026)

Within 10°, every image answered. In brackets: paired difference vs Woehrer 2026 in images, with its 95% bootstrap
interval (10,000 resamples); **bold** = significant.

| Test | Woehrer 2026 | Max | Pro | Balanced | Fast | Nano |
|---|---|---|---|---|---|---|
| New COCO photos (2,201) | 93.6% | **95.6% (+45 [+19, +71])** | **95.5% (+41 [+14, +68])** | 94.0% (+8 [-20, +36]) | **90.4% (-70 [-100, -41])** | **87.3% (-138 [-171, -106])** |
| New COCO photos, max-area crop | 94.1% | 95.3% (+25 [-1, +50]) | 95.0% (+20 [-6, +46]) | 93.2% (-21 [-49, +7]) | **90.2% (-87 [-116, -58])** | **88.0% (-136 [-167, -105])** |
| New COCO photos, CCTV-degraded | 54.7% | **92.5% (+831 [+783, +879])** | **92.4% (+829 [+781, +877])** | **91.6% (+812 [+764, +860])** | **89.3% (+760 [+712, +808])** | **86.5% (+699 [+649, +750])** |
| New Open Images photos (2,505) | 83.8% | **90.7% (+174 [+135, +214])** | **90.5% (+168 [+128, +209])** | **89.1% (+134 [+93, +175])** | 84.5% (+17 [-25, +59]) | **75.5% (-207 [-254, -160])** |
| New Open Images, CCTV-degraded | 44.5% | **84.4% (+1000 [+947, +1055])** | **84.1% (+991 [+937, +1045])** | **83.2% (+969 [+915, +1024])** | **78.9% (+861 [+806, +917])** | **73.5% (+727 [+669, +785])** |
| R-LiViT traffic camera, RGB (74) | 98.6% | 97.3% (-1 [-5, +2]) | 93.2% (-4 [-9, +0]) | **87.8% (-8 [-14, -2])** | **87.8% (-8 [-14, -2])** | **77.0% (-16 [-24, -8])** |
| R-LiViT traffic camera, thermal (74) | 37.8% | **95.9% (+43 [+34, +52])** | **95.9% (+43 [+34, +52])** | **95.9% (+43 [+34, +52])** | **90.5% (+39 [+30, +48])** | **74.3% (+27 [+17, +37])** |
| Aalborg thermal camera (150) | 57.3% | **83.3% (+39 [+24, +54])** | **78.0% (+31 [+16, +46])** | **68.7% (+17 [+2, +32])** | 61.3% (+6 [-8, +20]) | 48.7% (-13 [-29, +3]) |

- Upside-down errors (≥150°), every image answered, Woehrer 2026 vs Max: New COCO photos 40 vs 4; New COCO photos, max-area crop 36 vs 5; New COCO photos, CCTV-degraded 78 vs 9; New Open Images photos 145 vs 29; New Open Images, CCTV-degraded 173 vs 24; R-LiViT traffic camera, RGB 1 vs 0; R-LiViT traffic camera, thermal 33 vs 0; Aalborg thermal camera 37 vs 25.
- Routed share on these sets (Balanced / Pro): New COCO photos 11% / 32%; New COCO photos, max-area crop 10% / 30%; New COCO photos, CCTV-degraded 13% / 34%; New Open Images photos 27% / 50%; New Open Images, CCTV-degraded 33% / 55%; R-LiViT traffic camera, RGB 15% / 58%; R-LiViT traffic camera, thermal 31% / 80%; Aalborg thermal camera 26% / 55%.
- Pico was chosen after the holdout was opened and is not part of this evaluation (development numbers only, §2).

## 0b. Held-out CCTV and scene tests of 24 Sep, re-scored once with the release (A9; second use of these sets)

Source: `results/heldout-rescore/RESULTS.md` (pre-registered in `results/heldout-rescore/PREREG.md`; scored once on 30 Sep 2026).

| Held-out set | Woehrer 2026 | Deep-OAD | Max | Max − Woehrer (95% CI) | Release candidate's Max (24 Sep) |
|---|---|---|---|---|---|
| MEVA, 4 unseen cameras G329 / G420 / G421 / G474 (380 views, incl. thermal G474) | 75.0% (47) | 82.4% (26) | **100.0% (0)** | +95 views [+77, +113] | 98.7%\* |
| of which thermal camera G474 (110 views) | 53.6% (47) | 67.3% (26) | **100.0% (0)** | | 95.5% (5) |
| Clean scene test: DIODE scene 12, MEVA G339, Poly Haven (2,854 views) | 70.6% (98) | 82.8% (113) | **98.8% (5)** | +806 views [+754, +858] | 98.5% (3) |

Within 10°, every view answered; (≥150° errors). \*The release candidate's Max on the same 4 cameras, from the 24 Sep per-camera rows
(100% / 100% / 100% / 95.5% on 90 / 90 / 90 / 110 views = 375 / 380). Max − Deep-OAD: +67 [+52, +82] (MEVA 4 cameras), +459 [+413, +506]
(clean test). With the standard threshold Max answers 99.7% of the 380 MEVA views, all correct, and 97.7% of the clean
test with 99.7% correct.

Disclosures: (1) second use: both sets were opened once on 24 Sep to score the release candidate, and the released models were developed after those
results were known; the released models never trained on them and never scored them before today. (2) MEVA cameras G331 and G639 (other
clips) were in a development set of the release used for model choice, so the 6-camera figure (560 views: Max 100.0%) is reported
but labelled; the held-out claim uses the 4 other cameras. (3) Pico is weak on camera G329 (41.1%; a camera rolled ~36°
in some clips): Pico was chosen on development data and is the least accurate tier.

Predictions: kept with the release records; scorer: `results/heldout-rescore/score_a9.py`. Within 10°, every view answered; (≥150° errors).

## 0c. RotBench (independent benchmark with a human baseline), re-scored once with the release

Source: `results/rotbench/RESULTS.md` (pre-registered; scored once on 30 Sep 2026). 4-way accuracy per counter-clockwise rotation, official protocol.

**RotBench-Small (50 photos), official protocol (PIL rotate, no expand)**

| Model | 0° | 90° | 180° | 270° | mean | within 10° | ≥150° |
|---|---|---|---|---|---|---|---|
| Humans (published) | 0.99 | 0.99 | 0.99 | 0.97 | — | — | — |
| GPT-5 (published) | 1.00 | 0.41 | 0.81 | 0.59 | — | — | — |
| Gemini 2.5 Pro (published) | 1.00 | 0.50 | 0.72 | 0.40 | — | — | — |
| o3 (published) | 1.00 | 0.45 | 0.70 | 0.48 | — | — | — |
| Woehrer 2026 (24 Sep predictions) | 0.90 | 0.92 | 0.88 | 0.88 | 0.90 | 89.5% | 6 |
| Deep-OAD (24 Sep predictions) | 0.94 | 0.82 | 0.86 | 0.78 | 0.85 | 71.0% | 7 |
| GeoCalib (24 Sep predictions) | 1.00 | 0.00 | 0.00 | 0.00 | 0.25 | 22.5% | 46 |
| Release candidate's Max (24 Sep, reference) | 1.00 | 1.00 | 0.98 | 1.00 | — | — | — |
| Pico | 0.94 | 0.92 | 0.90 | 0.92 | 0.92 | 87.5% | 5 |
| Nano | 0.96 | 0.98 | 0.96 | 0.94 | 0.96 | 95.5% | 0 |
| Fast | 0.98 | 0.98 | 0.96 | 0.98 | 0.97 | 97.5% | 0 |
| Balanced | 1.00 | 0.98 | 0.98 | 0.98 | 0.98 | 98.5% | 0 |
| Pro | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 100.0% | 0 |
| **Max** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 100.0% | 0 |

**RotBench-Large (300 photos), official protocol (PIL rotate, no expand)**

| Model | 0° | 90° | 180° | 270° | mean | within 10° | ≥150° |
|---|---|---|---|---|---|---|---|
| Woehrer 2026 (24 Sep predictions) | 0.96 | 0.97 | 0.97 | 0.97 | 0.97 | 96.8% | 6 |
| Deep-OAD (24 Sep predictions) | 0.98 | 0.93 | 0.90 | 0.92 | 0.93 | 81.2% | 12 |
| GeoCalib (24 Sep predictions) | 1.00 | 0.00 | 0.00 | 0.00 | 0.25 | 23.1% | 293 |
| Release candidate's Max (24 Sep, reference) | 1.00 | 0.99 | 0.99 | 1.00 | — | — | — |
| Pico | 0.97 | 0.95 | 0.96 | 0.95 | 0.96 | 92.0% | 15 |
| Nano | 0.98 | 0.97 | 0.98 | 0.98 | 0.98 | 96.8% | 2 |
| Fast | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 98.6% | 0 |
| Balanced | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 99.1% | 0 |
| Pro | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 98.8% | 0 |
| **Max** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 98.9% | 0 |

## 1. The tiers

| Tier | Model | Input | Routing (calibration data only) | Abstain if confidence < (standard / strict; PyTorch = FP32 reference) |
|---|---|---|---|---|
| **Pico** | ViT-S/14 (12 blocks), Nano fine-tuned at 56–70 px (+ rotation-corner fine-tune, weight-averaged) | 70 px, fill-drop | — | 0.421 / 0.828 |
| **Nano** | ViT-S/14 | 112 px, fill-drop | — | 0.533 / 0.800 |
| **Fast** | ViT-S/14 (distilled from Max) | 224 px, fill-drop | — | 0.628 / 0.735 |
| **Balanced** | Fast → Max cascade | 224 → 280 px | least-confident ~10% to Max (threshold 0.628) | 0.692 / 0.748 |
| **Pro** | Fast → Max cascade | 224 → 280 px | least-confident ~20% to Max (threshold 0.800) | 0.729 / 0.729 |
| **Max** | ViT-L/14 (8-checkpoint weight average) | 280 px | — | 0.726 / 0.726 |

- All models were retrained on grid-free data (Max, Fast and Pico: 1,245,604 images; Nano: 705,075; `DATA-CARD.md`) so they no longer read the JPEG-grid
  and interpolation cue; every shipped file passes the grid check (C1) in its own runtime (§7).
- Files per model: ONNX FP32 / FP16 / INT8; Nano, Fast and Pico also as batch-capable ONNX (same answers, any batch size);
  Core ML (batch 1 and 16); Pico also as full-token ONNX for web browsers. Each format has its own calibrated thresholds
  (`FREEZE.md` addendum; Pico: `results/sealed-holdout/thresholds-pico-s80.json`).
- **Pico** was chosen by the owner after the freeze (29 Sep). It fails two pre-registered Pico gates by design — the 15 MB
  size limit (INT8 24.1 MB) and the "truncated network" definition — which the owner waived; the truncated 6-block
  alternative lost ~24 pp on indoor scenes (research record: `TECHNICAL-REPORT.md` §8). On 30 Sep the owner replaced
  it with PICO-N70S80, which fixes 180° misreads of images turned with flat-colour corners: 0.2 × the 29 Sep Pico +
  0.8 × its corner fine-tune (weight average). Development accuracy is unchanged within noise; speed is unchanged
  (identical graph). Pre-registered checks and deviations: `TECHNICAL-REPORT.md` §8; record: `FREEZE.md`.

## 2. Accuracy on development panels (within 10°, all images answered)

| Test | Woehrer 2026 | Max | Pro | Balanced | Fast | Nano | Pico |
|---|---|---|---|---|---|---|---|
| Woehrer 2026's own benchmark, seed 0 (1,030) | 98.6% | 99.2% | 99.0% | 98.4% | 96.9% | 95.4% | 92.3% |
| Same, 5 seeds (5,150) | 98.0% | 98.8% | 98.7% | 97.7% | 96.5% | 96.0% | 92.2% |
| Same, 5 seeds, JPEG q90 | 30.2% | 98.4% | 98.2% | 97.5% | 96.4% | 95.9% | 92.2% |
| Realistic photos, cue-free (1,856) | 77.6% | 83.6% | 83.5% | 81.8% | 76.3% | 74.2% | 70.0% |
| Same, camera JPEG | 23.2% | 81.1% | 80.9% | 79.2% | 74.8% | 73.9% | 69.9% |
| MEVA unseen real CCTV (646) | 77.5% | 99.8% | 99.8% | 99.7% | 98.8% | 98.0% | 96.4% |
| Common mixed CCTV-like (1,200) | 25.7% | 96.7% | 96.2% | 95.6% | 93.9% | 93.2% | 91.8% |
| Fresh DIODE scenes (300) | 28.0% | 98.3% | 97.3% | 95.0% | 91.3% | 89.7% | 86.0% |
| Fair photos (1,030) | 97.2% | 99.0% | 98.8% | 98.5% | 96.9% | 94.8% | 91.9% |
| Open Images photos, old panel (3,345) | 81.6% | 91.2% | 90.8% | 89.3% | 83.9% | 74.7% | 68.8% |

- The old Open Images panel rewarded the JPEG-grid cue (digitally rotated images keep the original pixel grid); the
  cue-free realistic-photo rows replace it as the honest photo measure. The sealed Open Images holdout (§0) is the
  headline photo number.

## 3. Robustness to CCTV degradation (within 10°)

| Test | Woehrer 2026 | Max | Pro | Balanced | Fast | Nano |
|---|---|---|---|---|---|---|
| New COCO photos, CCTV-degraded (sealed) | 54.7% | 92.5% | 92.4% | 91.6% | 89.3% | 86.5% |
| New Open Images, CCTV-degraded (sealed) | 44.5% | 84.4% | 84.1% | 83.2% | 78.9% | 73.5% |
| R-LiViT traffic camera, thermal (74) (sealed) | 37.8% | 95.9% | 95.9% | 95.9% | 90.5% | 74.3% |
| Aalborg thermal camera (150) (sealed) | 57.3% | 83.3% | 78.0% | 68.7% | 61.3% | 48.7% |
| Woehrer 2026's benchmark, JPEG q90 (development) | 30.2% | 98.4% | 98.2% | 97.5% | 96.4% | 95.9% |
| Realistic photos, camera JPEG (development) | 23.2% | 81.1% | 80.9% | 79.2% | 74.8% | 73.9% |

## 4. Abstention on the sealed holdout (thresholds fixed on calibration data, then applied unchanged)

Standard operating point (answer ~90% of calibration images). Cells: share answered / wrong among answered.

| Test | Max | Pro | Balanced | Fast | Nano |
|---|---|---|---|---|---|
| New COCO photos (2,201) | 91% / 2.1% | 91% / 2.2% | 89% / 3.4% | 89% / 4.6% | 89% / 7.1% |
| New COCO photos, max-area crop | 91% / 2.4% | 91% / 2.7% | 90% / 4.5% | 90% / 5.5% | 90% / 6.3% |
| New COCO photos, CCTV-degraded | 86% / 3.0% | 86% / 3.1% | 87% / 4.0% | 87% / 4.5% | 88% / 6.8% |
| New Open Images photos (2,505) | 74% / 1.2% | 74% / 1.7% | 75% / 3.2% | 73% / 4.4% | 71% / 10.3% |
| New Open Images, CCTV-degraded | 64% / 2.4% | 64% / 2.8% | 66% / 4.1% | 67% / 5.5% | 70% / 11.1% |
| R-LiViT traffic camera, RGB (74) | 93% / 2.9% | 93% / 7.2% | 84% / 11.3% | 85% / 14.3% | 96% / 21.1% |
| R-LiViT traffic camera, thermal (74) | 84% / 0.0% | 82% / 0.0% | 86% / 3.1% | 69% / 5.9% | 77% / 15.8% |
| Aalborg thermal camera (150) | 80% / 12.5% | 84% / 19.8% | 83% / 27.2% | 74% / 31.5% | 77% / 50.0% |

Strict operating point (≤1% wrong on calibration; effective strict = max(standard, strict)):

| Test | Max | Pro | Balanced | Fast | Nano |
|---|---|---|---|---|---|
| New COCO photos (2,201) | 91% / 2.1% | 91% / 2.2% | 81% / 2.5% | 79% / 2.4% | 66% / 2.2% |
| New COCO photos, max-area crop | 91% / 2.4% | 91% / 2.7% | 82% / 2.9% | 81% / 3.2% | 66% / 2.0% |
| New COCO photos, CCTV-degraded | 86% / 3.0% | 86% / 3.1% | 79% / 2.8% | 78% / 2.7% | 64% / 2.5% |
| New Open Images photos (2,505) | 74% / 1.2% | 74% / 1.7% | 63% / 2.0% | 61% / 2.1% | 42% / 2.1% |
| New Open Images, CCTV-degraded | 64% / 2.4% | 64% / 2.8% | 56% / 2.6% | 55% / 2.7% | 40% / 2.3% |
| R-LiViT traffic camera, RGB (74) | 93% / 2.9% | 93% / 7.2% | 69% / 13.7% | 57% / 16.7% | 24% / 16.7% |
| R-LiViT traffic camera, thermal (74) | 84% / 0.0% | 82% / 0.0% | 72% / 1.9% | 51% / 2.6% | 12% / 11.1% |
| Aalborg thermal camera (150) | 80% / 12.5% | 84% / 19.8% | 73% / 25.7% | 57% / 26.7% | 25% / 62.2% |

- Thermal cameras are the weak spot for abstention. On the unseen Aalborg thermal camera every tier stays confident while
  wrong too often (standard point: Max 12.5%, Nano 50% of answered images wrong), and the strict point does not fix it
  (Nano 62% wrong among the 25% it answers). On thermal, use Max and do not rely on the confidence alone. On the
  R-LiViT thermal camera, Max and Pro made no wrong answers at either point.


## 5. Speed and memory

### GPUs — TensorRT FP16 (batch-capable files)

| GPU | Pico single · batched | Nano single · batched | Fast single · batched | Balanced (derived) | Pro (derived) | Max single · batched | Woehrer 2026 | One-image files: Pico / Nano / Fast single |
|---|---|---|---|---|---|---|---|---|
| RTX PRO 4500 Blackwell † (batch 64) | 0.75 ms · 55,950 img/s | 0.95 ms · 25,737 img/s | 1.25 ms · 6,483 img/s | 1.77 ms · 2,631 img/s | 2.29 ms · 1,650 img/s | 5.19 ms · 443 img/s | 4.04 ms | 0.55 ms / 0.55 ms / 0.68 ms |
| RTX 5090 (batch 64) | 0.68 ms · 62,071 img/s | 0.86 ms · 34,706 img/s | 0.98 ms · 10,013 img/s | 1.48 ms · 4,827 img/s | 1.98 ms · 3,180 img/s | 4.99 ms · 932 img/s | 3.73 ms | 0.71 ms / 0.67 ms / 0.63 ms |
| RTX PRO 4000 Blackwell (batch 64) | 0.85 ms · 41,751 img/s | 1.25 ms · 19,540 img/s | 1.56 ms · 4,414 img/s | 2.17 ms · 1,765 img/s | 2.78 ms · 1,103 img/s | 6.12 ms · 294 img/s | 5.20 ms | 0.70 ms / 0.74 ms / 0.76 ms |
| RTX 4000 Ada † (batch 64) | 1.05 ms · 35,240 img/s | 1.14 ms · 14,341 img/s | 1.35 ms · 3,081 img/s | 2.09 ms · 1,272 img/s | 2.83 ms · 802 img/s | 7.39 ms · 217 img/s | 6.67 ms | 0.72 ms / 0.75 ms / 1.04 ms |
| NVIDIA L4 † (batch 64) | 0.88 ms · 31,845 img/s | 1.62 ms · 12,889 img/s | 1.26 ms · 2,381 img/s | 2.18 ms · 947 img/s | 3.09 ms · 591 img/s | 9.15 ms · 157 img/s | 7.28 ms | 0.46 ms / 0.60 ms / 0.94 ms |
| RTX 3060 Laptop (Windows) † (batch 32/16) | 4.17 ms · 6,462 img/s | 5.19 ms · 1,664 img/s | 6.08 ms · 1,015 img/s | 7.43 ms · 550 img/s | 8.77 ms · 377 img/s | 13.4 ms · 120 img/s | 38.4 ms | 7.62 ms / 7.27 ms / 8.02 ms |

### CPUs — INT8, 4 threads (Woehrer 2026 FP32)

| CPU | Pico | Nano | Fast | Balanced (derived) | Pro (derived) | Max | Woehrer 2026 | Batched (batch 8): Pico / Nano / Fast / Max |
|---|---|---|---|---|---|---|---|---|
| Intel Core i7-1260P (laptop) † | 3.21 ms | 7.73 ms | 29.7 ms | 62.2 ms | 94.8 ms | 326 ms | 161 ms | 274 img/s / 152 img/s / 36 img/s / 2.5 img/s |
| Apple M4 † | 4.14 ms | 8.42 ms | 30.2 ms | 77.1 ms | 124 ms | 469 ms | 138 ms | 347 img/s / 155 img/s / 25 img/s / 2.0 img/s |
| AMD EPYC 7663 (server) † | 12.9 ms | 18.6 ms | 42.4 ms | 95.8 ms | 149 ms | 534 ms | 150 ms | 255 img/s / 141 img/s / 32 img/s / 2.3 img/s |
| Intel Xeon Gold 6530 (server) | 7.43 ms | 11.5 ms | 24.3 ms | 44.2 ms | 64.1 ms | 199 ms | 123 ms | 249 img/s / 213 img/s / 61 img/s / 6.6 img/s |
| AMD EPYC 9254 (server) | 6.30 ms | 15.5 ms | 32.5 ms | 67.7 ms | 103 ms | 352 ms | 140 ms | 238 img/s / 152 img/s / 44 img/s / 4.0 img/s |
| AMD EPYC 7352 (server) | 8.83 ms | 18.4 ms | 60.2 ms | 146 ms | 233 ms | 863 ms | 165 ms | 101 img/s / 78 img/s / 20 img/s / 1.3 img/s |
| AMD EPYC 7282 (server) | 14.0 ms | 29.6 ms | 71.9 ms | 164 ms | 255 ms | 918 ms | 190 ms | 85 img/s / 68 img/s / 18 img/s / 1.2 img/s |

### Apple M4 (Mac mini) — native Core ML FP16

| Compute units | Pico single · batch 16 | Nano single · batch 16 | Fast single · batch 16 | Balanced (derived) | Pro (derived) | Max single · batch 16 |
|---|---|---|---|---|---|---|
| CPU+ANE | 0.86 ms · 3,506 img/s | 1.04 ms · 1,382 img/s | 3.44 ms · 261 img/s | 8.78 ms · 107 img/s | 14.1 ms · 67 img/s | 53.5 ms · 18 img/s |
| ALL | 1.83 ms · 3,722 img/s | 2.54 ms · 1,308 img/s | 3.34 ms · 257 img/s | 8.69 ms · 107 img/s | 14.0 ms · 67 img/s | 53.4 ms · 18 img/s |


## 6. Runtime accuracy parity and the grid check

- Every shipped file passes the grid check C1 (argmax rho4 ≤ 0.20 on three content-free probe sets) in its runtime:
  ONNX Runtime CPU (FP32, INT8, batch INT8), ONNX Runtime CUDA (FP16, batch FP16), TensorRT FP16, OpenVINO CPU, Core ML (all
  compute units) — Nano, Fast, Max and Pico.
- Calibration accuracy by format (Open Images calibration set, 6,692 views, within 10°): FP16 = FP32 within ±8 views for
  every model; INT8 is 0.2–0.7 pp lower and slightly less confident, hence per-format thresholds; Core ML = FP32 within
  5 views. TensorRT FP16 and OpenVINO give the same answers as ONNX Runtime FP32 (median difference ≤ 0.001°).
- Batch-capable files equal the batch-1 fill-drop files at FP32 (max |Δprob| 7e-8 on 1,976 real views).

## 7. What we do not claim

- That Nano or Fast beat Woehrer 2026 on clean photos: they trail it on clean COCO (and Nano on clean Open Images);
  their wins are on degraded CCTV and thermal.
- A win on the max-area crop or R-LiViT RGB sealed panels (ties).
- That Max is faster than Woehrer 2026 per single image on CPUs (it is 1.6–5× slower) or on most GPUs (1.1–1.3×).
- Sealed-holdout numbers for Pico (development numbers only).
- Anything about nadir aerial imagery, IMU replacement, or Raspberry Pi speed (not measured).
