# Benchmarks and protocol

Weights, tiers and thresholds are frozen (`FREEZE.md`, 29 Sep 2026). The sealed final holdout was opened once after the
freeze and scored once; the 24 Sep held-out CCTV and scene sets were re-scored once with the release files (second use);
the development tables were used for model choice. Figures are taken from `RESULTS.md` (single source of truth),
`results/heldout-rescore/RESULTS.md` and `benchmarks/HARDWARE-summary.md`. Method and caveats in full:
`TECHNICAL-REPORT.md`.

Metric: share of images whose predicted rotation is within **10°** of the truth (circular error), every image
answered unless stated. Also reported: near-opposite errors (≥ 150°, "upside-down" mistakes) and paired differences
with 95% bootstrap intervals.

"Woehrer 2026" is MambaOut-base CGD (arXiv:2603.25351, MIT), the published state of the art on the COCO rotation
benchmark. It is not the strongest other model on every panel: on the held-out CCTV cameras Deep-OAD, the next-best
open model we tested, scores 82.4%. "Degraded" always means our **simulated** CCTV degradation (resolution loss,
blur, IR/greyscale, exposure, noise, JPEG), applied with a fixed seed to the same views as the clean copy; no degraded
view is a recording from a degraded camera.

## Panels

| Panel | Size | Content | Crop policy | Role |
|---|---|---|---|---|
| **Sealed: new COCO photos** | 2,201 | COCO val2017 photos; fixed crop, max-area crop and degraded copies | fixed 4:3 (and max-area, reported separately) | Once-only final evaluation (29 Sep 2026) |
| **Sealed: new Open Images photos** | 2,505 | Open Images validation photos (ranks from 8,000), CC BY 2.0 listing; fixed crop and degraded copies | fixed 4:3 | Once-only final evaluation |
| **Sealed: R-LiViT traffic camera** | 74 + 74 | R-LiViT final locations 6 and 7 (intersection A), RGB and thermal | fixed 4:3 | Once-only final evaluation |
| **Sealed: Aalborg thermal camera** | 150 | Aalborg long-term thermal drift, one camera | fixed 4:3 | Once-only final evaluation |
| Held-out MEVA test set (frozen split) | 560 | 6 unseen MEVA cameras incl. thermal G474; each frame clean + degraded | fixed 4:3, diagonal = min(W, H) | Scored the 24 Sep candidate; re-used once for the release (held-out claim: 4 cameras, 380 views) |
| Held-out clean scene test | 2,854 | 1,427 parents from DIODE scene 12, MEVA G339 and Poly Haven renders; clean + degraded | same | Scored the 24 Sep candidate; re-used once for the release |
| Woehrer 2026's benchmark, five seeds | 5,150 | 1,030 COCO 2014 val photos × 5 angle seeds, max-area crop (upstream code) | max-area, angle-dependent | Development; reported as the five-seed mean |
| same, JPEG q90 | 5,150 | each view saved once as JPEG q90 | same | Development |
| Realistic photos, cue-free | 1,856 | Open Images development photos re-rendered from 2× originals, downscaled to the same view size (no rotation traces) | fixed 4:3 | Development; the honest photo measure |
| same, camera JPEG | 1,856 | the same views with camera-like JPEG | same | Development |
| MEVA test set (dev) | 646 | 7 unseen real CCTV cameras incl. thermal G479; clean + degraded | fixed 4:3 | Development |
| Common | 1,200 | DIODE (scenes 3/5/9/11), MEVA (G328, G638), Poly Haven, renders; clean + degraded | angle-independent | Development guard |
| Fresh DIODE | 300 | DIODE validation scenes 19–24 (never trained on) | angle-independent | Development |
| Fair photo | 1,030 | Woehrer 2026's benchmark photos with a neutral crop, one fresh angle draw | fixed 4:3 | Development |
| Open Images photo test (older panel) | 3,345 | Open Images validation ranks 0–3999, 1:1 renders of 448-px parents | fixed 4:3 | Development; rewards rotation traces (see below) |
| RotBench (independent) | 50 (Small) + 300 (Large) | Photos turned 0/90/180/270° counter-clockwise with PIL (arXiv 2508.13968; data Apache-2.0) | square images, never cropped | Public benchmark with a human baseline; scored once with the release after a pre-registration (30 Sep 2026) |

The "fixed 4:3, diagonal = min(W, H)" crop fits inside the image's inscribed circle at any angle, so crop shape and
scale carry no information about the angle.

**Calibration sets** (used only to fix routing and abstention thresholds; never for training, selection or testing):
MEVA G419 + G299 + G330 (270 views), Poly Haven calibration split (1,192 views), Open Images validation
ranks 4000–7999 (6,692 views), each image clean + degraded, weighted equally.

## Findings on Woehrer 2026's benchmark (summary; full detail in `TECHNICAL-REPORT.md` §5 and `benchmarks/woehrer-2026/`)

| Version of the test (1,030 photos) | Woehrer 2026 | Max | Fast | Nano |
|---|---|---|---|---|
| As published, five-seed mean | 98.0% | **98.8%** | 96.5% | 96.0% |
| As published, each view saved once as JPEG q90, five-seed mean | 30.2% | **98.4%** | 96.4% | 95.9% |
| Neutral 4:3 crop, one angle draw (fair photos) | 97.2% | **99.0%** | 96.9% | 94.8% |

- **Seeds.** Woehrer 2026's paper reports the five-seed mean (98.0%). The 98.6% we quoted earlier was seed 0 of our
  rebuild, the best of its five seeds there: our choice of figure, not the paper's.
- **JPEG.** One JPEG q90 save of each view drops Woehrer 2026 from 98.0% to 30.2%, and its answers snap to multiples of
  90°. Our reading is that the benchmark's rotated source JPEG grid carries part of the angle; this mechanism is a
  hypothesis we have not proven.
- **Crop.** The maximal-area crop changes field of view and scale with the angle. In a paired test that changes only
  the crop (2,124 COCO val2017 photos), Woehrer 2026 gained 17 views on the max-area crop (exact McNemar p = 0.06), so
  the crop effect is not statistically established. On the sealed max-area panel Max and Woehrer 2026 tie.
- Max − Woehrer 2026 on the five-seed benchmark as published: +41 images [8, 76] over 5,150 views.
- 915 of the 1,030 benchmark photos also sit in COCO train2017; the author's pipeline removes exactly those before
  training, so we treat the benchmark photos as unseen by Woehrer 2026.

These are properties of a common protocol, not flaws of one author's work. We never train on the angle-dependent crop,
store training images losslessly, and report the benchmark both ways.

## Sealed final holdout (frozen weights, opened once, scored once, 29 Sep 2026)

Within 10°, every image answered; paired difference vs Woehrer 2026 in images [95% CI]; **bold** = significant.

| Test (images) | Woehrer 2026 | Max | Pro | Balanced | Fast | Nano |
|---|---|---|---|---|---|---|
| New COCO photos (2,201) | 93.6% | **95.6% (+45 [+19, +71])** | **95.5% (+41 [+14, +68])** | 94.0% (+8 [−20, +36]) | **90.4% (−70 [−100, −41])** | **87.3% (−138 [−171, −106])** |
| same, max-area crop | 94.1% | 95.3% (+25 [−1, +50]) | 95.0% (+20 [−6, +46]) | 93.2% (−21 [−49, +7]) | **90.2% (−87 [−116, −58])** | **88.0% (−136 [−167, −105])** |
| same, degraded | 54.7% | **92.5% (+831 [+783, +879])** | **92.4% (+829 [+781, +877])** | **91.6% (+812 [+764, +860])** | **89.3% (+760 [+712, +808])** | **86.5% (+699 [+649, +750])** |
| New Open Images photos (2,505) | 83.8% | **90.7% (+174 [+135, +214])** | **90.5% (+168 [+128, +209])** | **89.1% (+134 [+93, +175])** | 84.5% (+17 [−25, +59]) | **75.5% (−207 [−254, −160])** |
| same, degraded | 44.5% | **84.4% (+1000 [+947, +1055])** | **84.1% (+991 [+937, +1045])** | **83.2% (+969 [+915, +1024])** | **78.9% (+861 [+806, +917])** | **73.5% (+727 [+669, +785])** |
| R-LiViT traffic camera, RGB (74) | 98.6% | 97.3% (−1 [−5, +2]) | 93.2% (−4 [−9, +0]) | **87.8% (−8 [−14, −2])** | **87.8% (−8 [−14, −2])** | **77.0% (−16 [−24, −8])** |
| R-LiViT traffic camera, thermal (74) | 37.8% | **95.9% (+43 [+34, +52])** | **95.9% (+43 [+34, +52])** | **95.9% (+43 [+34, +52])** | **90.5% (+39 [+30, +48])** | **74.3% (+27 [+17, +37])** |
| Aalborg thermal camera (150) | 57.3% | **83.3% (+39 [+24, +54])** | **78.0% (+31 [+16, +46])** | **68.7% (+17 [+2, +32])** | 61.3% (+6 [−8, +20]) | 48.7% (−13 [−29, +3]) |

Upside-down errors (≥150°), Woehrer 2026 vs Max: 40 vs 4 (new COCO), 145 vs 29 (new Open Images), 33 vs 0 (R-LiViT
thermal), 37 vs 25 (Aalborg). Balanced / Pro routed 10–33% / 30–80% of these images to Max. Pico was chosen after the
holdout was opened and has no sealed-holdout numbers.

## Held-out CCTV and scene sets (24 Sep sets, second use, 30 Sep 2026)

| Held-out set (views) | Woehrer 2026 | Deep-OAD | GeoCalib | Pico | Nano | Fast | Balanced | Pro | Max |
|---|---|---|---|---|---|---|---|---|---|
| 4 unseen MEVA cameras G329/G420/G421/G474 (380) | 75.0% (47) | 82.4% (26) | 23.9% (86) | 80.0% (21) | 92.6% (21) | 97.9% (8) | 98.4% (6) | 99.2% (3) | **100.0% (0)** |
| of which thermal G474 (110) | 53.6% (47) | 67.3% (26) | 20.9% (29) | 86.4% (14) | 89.1% (12) | 93.6% (7) | 94.5% (6) | 97.3% (3) | **100.0% (0)** |
| Clean scene test: DIODE scene 12, MEVA G339, Poly Haven (2,854) | 70.6% (98) | 82.8% (113) | 25.0% (749) | 96.5% (11) | 98.0% (7) | 97.8% (10) | 98.0% (6) | 98.7% (5) | **98.8% (5)** |

Upside-down errors in brackets. Pre-registered (`results/heldout-rescore/PREREG.md`) and scored once with the release
ONNX FP32 files. Disclosures: second use of both sets (opened once on 24 Sep for the earlier candidate; the released
models were developed after those results were known, never trained or selected on them); MEVA cameras G331/G639 were in a
development set of the release, so the held-out claim uses the other four cameras (all six, 560 views: Max 100.0%); comparator
predictions are the stored 24 Sep predictions on identical pixels, reproduced exactly. GeoCalib is built for about
±45° of roll; its full-circle numbers reflect scope. Per-camera and per-group rows, strict point and abstention:
`results/heldout-rescore/RESULTS.md`.

## Development test sets (full tables in `RESULTS.md` §2–§3)

| Test (images) | Woehrer 2026 | Max | Pro | Balanced | Fast | Nano | Pico |
|---|---|---|---|---|---|---|---|
| Realistic photos, cue-free (1,856) | 77.6% | 83.6% | 83.5% | 81.8% | 76.3% | 74.2% | 70.0% |
| same, camera JPEG | 23.2% | 81.1% | 80.9% | 79.2% | 74.8% | 73.9% | 69.9% |
| MEVA unseen real CCTV (646) | 77.5% | 99.8% | 99.8% | 99.7% | 98.8% | 98.0% | 96.4% |
| Common mixed CCTV-like (1,200) | 25.7% | 96.7% | 96.2% | 95.6% | 93.9% | 93.2% | 91.8% |
| Fresh DIODE scenes (300) | 28.0% | 98.3% | 97.3% | 95.0% | 91.3% | 89.7% | 86.0% |
| Fair photos (1,030) | 97.2% | 99.0% | 98.8% | 98.5% | 96.9% | 94.8% | 91.9% |
| Open Images photos, older panel (3,345) | 81.6% | 91.2% | 90.8% | 89.3% | 83.9% | 74.7% | 68.8% |

The older Open Images panel renders 448-px parents at about 1:1, which leaves traces of the digital rotation that a
rolled camera never has; the cue-free rows replace it as the photo measure, and the sealed Open Images holdout is the
headline photo number. We think Woehrer 2026's low Common and Fresh DIODE scores come mostly from simulated CCTV
degradation, indoor close-ups and non-photographic content that its photo-only training did not cover (our inference,
not a controlled test).

## RotBench (independent benchmark, official protocol; scored once, 30 Sep 2026)

RotBench (arXiv 2508.13968; data Apache-2.0) turns each photo 0/90/180/270° counter-clockwise with PIL; the score is
4-way accuracy per rotation (our prediction snapped to the nearest quarter turn). Its images are square, so the
official protocol and a full-frame variant give the same images and numbers. The release files were scored once
after a pre-registration (`results/rotbench/PREREG.md`). RotBench is public and was used for the 24 Sep candidate,
so this is a second use of the benchmark; the release never trained on it (new training data was screened against it by
perceptual hash) and no development run of the release scored it. Human and vision-language model rows are as published by
RotBench (RotBench-Small only); we did not re-run them. Comparator rows are their stored 24 Sep predictions on the same
views, reproduced exactly. Naming a model here implies no endorsement by its authors.

| RotBench-Small (50 photos) | 0° | 90° | 180° | 270° | mean | ≥150° errors |
|---|---|---|---|---|---|---|
| RotBench human baseline (published) | 0.99 | 0.99 | 0.99 | 0.97 | — | — |
| GPT-5 (published) | 1.00 | 0.41 | 0.81 | 0.59 | — | — |
| Gemini 2.5 Pro (published) | 1.00 | 0.50 | 0.72 | 0.40 | — | — |
| o3 (published) | 1.00 | 0.45 | 0.70 | 0.48 | — | — |
| Woehrer 2026 | 0.90 | 0.92 | 0.88 | 0.88 | 0.90 | 6 |
| Deep-OAD | 0.94 | 0.82 | 0.86 | 0.78 | 0.85 | 7 |
| GeoCalib | 1.00 | 0.00 | 0.00 | 0.00 | 0.25 | 46 |
| RightWayUp Pico | 0.94 | 0.92 | 0.90 | 0.92 | 0.92 | 5 |
| RightWayUp Nano | 0.96 | 0.98 | 0.96 | 0.94 | 0.96 | 0 |
| RightWayUp Fast | 0.98 | 0.98 | 0.96 | 0.98 | 0.97 | 0 |
| RightWayUp Balanced | 1.00 | 0.98 | 0.98 | 0.98 | 0.98 | 0 |
| RightWayUp Pro | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | 0 |
| RightWayUp Max | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | 0 |

| RotBench-Large (300 photos), mean 4-way accuracy | Woehrer 2026 | Deep-OAD | GeoCalib | Pico | Nano | Fast | Balanced | Pro | Max |
|---|---|---|---|---|---|---|---|---|---|
| mean (≥150° errors) | 0.97 (6) | 0.93 (12) | 0.25 (293) | 0.96 (15) | 0.98 (2) | 0.99 (0) | 1.00 (0) | **1.00 (0)** | **1.00 (0)** |

Max and Pro answer every RotBench image correctly (1.00 at every rotation on both splits) with no upside-down answers,
above the published human baseline on RotBench-Small (0.99 / 0.99 / 0.99 / 0.97), by 1–3 points on 50 photos. Per-rotation tables for both splits and variants, within-10° shares: `results/rotbench/RESULTS.md`.

## Abstention and hardware

Abstention at both operating points on the sealed holdout: `RESULTS.md` §4; on the held-out sets:
`results/heldout-rescore/RESULTS.md`. Confidence is not reliable on unseen thermal cameras for any tier. Hardware:
`benchmarks/HARDWARE-summary.md` (single-image latency (batch 1) and batched throughput (batch N) on six NVIDIA
GPUs, Apple M4 and seven CPUs). Runtime accuracy parity: `RESULTS.md` §6.

## Reproducibility kit

- Test-set definitions: source IDs, per-view seeds and angles, crop recipes (`rotlab/fair_photo.py`,
  `rotlab/meva_test.py`, `rotlab/oi_test.py`, `rotlab/degraded_sets.py`, `rotlab/calib.py`, and for the sealed holdout
  `rotlab/camp_newdata.py`, `rotlab/camp_rlivit.py`, `rotlab/camp_aalborg.py`); held-out set definitions in
  `manifests/evaluation/`.
- Evaluation: `rotlab/camp_eval.py`, `rotlab/focus.py` (fill-drop scoring), `rotlab/camp_final.py` (freeze, open,
  score gate and aggregation), `rotlab/camp_incumbent.py` (Woehrer 2026 adapter), `rotlab/final_stats.py` (thresholds).
- Hardware benchmark harness (`rotlab/bench.py`, `rotlab/coreml_export.py`, `rotlab/gpu_parity.py`) and bundle
  (`benchmarks/bundle/`).
- Grid check: `rotlab/onnx_probe.py` (content-free probes, ρ₄).
- RotBench: `rotlab/competitors.py` (views), `results/rotbench/rb_predict.py` and `score_rotbench.py` (scoring of the release).
