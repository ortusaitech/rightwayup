# Freeze record v2: RightWayUp v1.0 (grid-fix models)

> Terminology note (added 30 Sep 2026): in this record, 'v2' denotes the RightWayUp 1.0 release weights and 'v1' the 24 Sep release candidate; the record itself is unchanged.

**Frozen 29 Sep 2026, 11:19 Sydney time (01:19 UTC), before the sealed final holdout was opened.**
Owner go-ahead in chat at about 10:50 Sydney time ("We are now ready to freeze"). No weights, tier settings or
thresholds change after this point. If anything must change, the sealed final results are void.

The machine record is `camp-frozen/FROZEN.json` on the campaign volume (read-only, created once), SHA-256
`5caf3220586079e38dd7f16238d58993dd8901853c7930cb9aa0e98eaa40b428`; a copy is in
`results/sealed-holdout/FROZEN.json` with the spec it was made from (`results/sealed-holdout/SPEC.json`).
It binds the checkpoints below, the comparator files, the four final-holdout source manifests and the 14 code files
that render and score the final sets. The 24 Sep record is kept as `FREEZE-release-candidate-2026-09-24.md`.

## Weights

| Role | Checkpoint (EMA weights) | SHA-256 |
|---|---|---|
| Nano (112 px, fill-drop) | `runs/GF-SOUP-SV2/final.pt` | `70590680d5fb12ab3aed0e796bf0c7d37c21e8f17860ad0401e086c220666bba` |
| Fast (224 px, fill-drop); cascade stage 1 for Balanced and Pro | `runs/GF-SV3-kdM3/final.pt` | `f317fa4cf92c553893148fc7e7e5a7fd1352e7b3b7e737b45bf5ecc526edef3c` |
| Max (280 px, every image); cascade stage 2 for Balanced and Pro | `runs/M3-SOUP8/final.pt` | `08dea345add0e89f8c8e2ba159e403fa9f19fa6b158aab1c1e2476f5d5e7d969` |

- Nano and Fast: DINOv2 ViT-S/14. Max: DINOv2 ViT-L/14 (same pinned backbones as v1; revisions and hashes in `RECIPE.md`).
- **Fill-drop:** the small models drop the letterbox-fill patch tokens (the padding added to make an image square)
  before the transformer, exactly as the shipped fill-drop ONNX files do. Max uses all tokens.
- All three were retrained on grid-free data (Max and Fast on 1,245,604 images, Nano on 705,075; corrected 30 Sep from
  "about 1.2 million" for all three, see `DATA-CARD.md`: Nano's mix did not include the Open Images V7 / CommonCatalog
  expansion) to remove the JPEG-grid and
  interpolation cue; research record: `TECHNICAL-REPORT.md` §4.
- The owner accepted the choice before the freeze: Max = M3-SOUP8, Fast = GF-SV3-kdM3 (distilled from M3-SOUP8),
  Nano = GF-SOUP-SV2.

Released files are the exports of these three checkpoints (ONNX FP32 / FP16 / INT8, Core ML). Their hashes go in
`manifests/RELEASE-ARTIFACTS-SHA256.txt` once exported; this record then cites that manifest's SHA-256.

## Tiers and thresholds

These were fixed on the calibration sets only: MEVA G419/G299/G330, Poly Haven clean-v1 calibration split and Open
Images validation ranks 4000–7999, each weighted equally. Confidence is the probability mass within ±10° of the
prediction, with no test-time augmentation. Route thresholds are the small model's confidence at a 10% / 20% route
share on calibration data (the released definition, decision D-R2).

| Tier | Models | Input | Route to Max if confidence < | Abstain if confidence < (standard, ~90% answered) | Abstain if confidence < (strict, ≤1% wrong; effective) |
|---|---|---|---|---|---|
| Nano | small (Nano) | 112 px | — | 0.533 | 0.800 |
| Fast | small (Fast) | 224 px | — | 0.628 | 0.735 |
| Balanced | Fast → Max | 224 → 280 px | 0.628 (~10%) | 0.692 | 0.748 |
| Pro | Fast → Max | 224 → 280 px | 0.800 (~20%) | 0.729 | 0.729 |
| Max | Max | 280 px | — | 0.726 | 0.726 |

For cascade tiers, the abstain threshold applies to the confidence of whichever model answered. Strict is the
effective value (the higher of the calibrated strict and standard, per the 25 Sep rule); the raw calibrated strict
values are Balanced 0.7475, Pro 0.695, Max 0.680.

These thresholds apply to PyTorch / FP32 inference. INT8, FP16 and Core ML files are calibrated separately, on the same
calibration sets and never on final sets, and their thresholds are added to this record before release.

## Code

Scoring runs from one code tree (`/workspace/code-final`: the campaign `rotlab` plus the lead overlay with the
fill-drop scorer `focus.py` and `model_x.py`); FROZEN.json holds the SHA-256 of all 14 files. Before the freeze,
two development panels (Fresh DIODE, Fair photos) re-scored from this tree matched the development results exactly:
3 stages, 3,990 predictions, identical answers and confidences. Final-set scoring for fill-drop stages goes through
`focus eval --mode fill` behind the same gate (`camp_final.check_score`) as full-token scoring; the released tiers are
aggregated by `camp_final.aggregate_tiers` from the frozen tier table. This record, the spec and the gate code are
committed together in this repository.

## Sealed final holdout (opened once, after this record)

Never trained on, never used for selection, never rendered before the freeze:

1. **New COCO holdout** (2,201 val2017 photos): fixed crop, max-area crop and CCTV-degraded views.
2. **New Open Images holdout** (2,505 validation photos, CC BY 2.0 listing): fixed crop and CCTV-degraded views.
3. **R-LiViT final locations 6 and 7** (traffic intersection A): RGB and thermal.
4. **Aalborg long-term thermal drift** (150 images, one thermal camera).

The raw source files had lived on pod disks that no longer exist, so before the freeze they were downloaded again
from their public origins and checked byte-for-byte against the pre-freeze manifests (COCO per-image SHA-256; Open
Images SHA-256 of the original and of the stored re-encode; R-LiViT zip MD5 and size; Aalborg per-image SHA-256).
Comparator on identical pixels: Woehrer 2026, frozen as ONNX `5fd9a686…` with weights `44d58a12…`.

Order after this record: `camp_final open` → render final views (each with a create-exclusive receipt) → independent
review of FROZEN.json and the view receipts → `SCORING-APPROVED.json` → score each frozen stage once → aggregate.

## Addendum, 29 Sep 2026 (after the freeze, before the sealed holdout was opened): per-format thresholds

Each shipped file was run on the same calibration views (MEVA / Poly Haven / Open Images calibration sets, 8,154 per
canvas) in its own runtime, and the release `Tier.calibrate` fitted its thresholds unchanged (`fmt_thresholds.py`;
predictions from `cal_onnx.py`; INT8 on ONNX Runtime CPU, FP16 on ONNX Runtime CUDA, FP32 on CPU for the small models and
CUDA for Max). No final or test data was used. Cells: standard / effective strict; route for cascades.

| Tier | ONNX FP32 | ONNX FP16 | ONNX INT8 | batch ONNX FP16 | batch ONNX INT8 | Core ML (FP16) |
|---|---|---|---|---|---|---|
| Nano | 0.535 / 0.800 | 0.535 / 0.800 | 0.536 / 0.800 | 0.535 / 0.800 | 0.531 / 0.782 | 0.534 / 0.800 |
| Fast | 0.629 / 0.738 | 0.629 / 0.738 | 0.626 / 0.738 | 0.629 / 0.738 | 0.623 / 0.738 | 0.628 / 0.738 |
| Balanced | 0.691 / 0.743; route 0.629 | 0.691 / 0.745; route 0.629 | 0.685 / 0.745; route 0.626 | 0.691 / 0.743; route 0.629 | 0.684 / 0.745; route 0.623 | 0.692 / 0.743; route 0.628 |
| Pro | 0.729 / 0.729; route 0.800 | 0.729 / 0.729; route 0.800 | 0.714 / 0.714; route 0.797 | 0.729 / 0.729; route 0.800 | 0.714 / 0.714; route 0.793 | 0.729 / 0.729; route 0.800 |
| Max | 0.727 / 0.727 | 0.727 / 0.727 | 0.711 / 0.711 | 0.727 / 0.727 | 0.711 / 0.711 | 0.727 / 0.727 |

- **Batch files** (owner decision, 29 Sep): Nano and Fast also ship as batch-capable ONNX (`*-fillmask-*`): all tokens are
  computed, letterbox-fill keys get an additive −1e4 in every attention block and fill tokens are left out of the pool.
  At FP32 their answers equal the batch-1 fill-drop files (1,976 real views: identical within-10°, max |Δprob| 7e-8),
  so batch FP32 uses the ONNX FP32 column. Max's ONNX files already take any batch size.
- **Calibration accuracy by format** (Open Images calibration set, 6,692 views, within 10°): FP16 equals FP32 within
  ±8 views for all three models; INT8 is lower by 0.2 pp (Nano), 0.4–0.7 pp (Fast) and 0.66 pp (Max) and slightly less
  confident, which is why its thresholds are lower.
- Grid check C1 (argmax rho4 ≤ 0.20 on the three content-free probe sets) passes on every shipped ONNX file (FP32, FP16,
  INT8, batch FP16 / INT8), on TensorRT FP16 and OpenVINO builds of Nano and Fast, and on all three Core ML packages.
- Core ML: the batch-16 packages (compute units ALL, the release default) were run on an Apple M4 Mac on the same calibration
  views (sent as lossless 8-bit images; exact reconstruction verified). Calibration accuracy equals FP32 within 5 views
  per model on the Open Images calibration set.

## Addendum, 29 Sep 2026: Pico (sixth tier, added after the freeze)

**Owner decision** in chat, 29 Sep ~22:25 AEST: ship Pico = `PICO-N70L @70`. Pico is **not** part of FREEZE v2 or of the
sealed final holdout. It was developed and selected on development sets only (`TECHNICAL-REPORT.md` §8)
and was chosen after the holdout had been opened, so its reported accuracy is development-panel accuracy.

| Role | Checkpoint (EMA weights) | SHA-256 |
|---|---|---|
| Pico (70 px, fill-drop) | `runs/PICO-N70L/final.pt` | `6b823e474a911fa77d31294c0122187359e5a330cc9773e62c41e2c653fb8e50` |

- The full 12-block Nano (GF-SOUP-SV2) fine-tuned at 56–70 px, distilled from Max (M3-SOUP8), with the grid-fix recipe.
- The owner waived two pre-registered Pico gates: the 15 MB INT8 size limit (the file is 24.1 MB) and the "truncated network"
  definition. The truncated 6-block candidate passed the size gate but lost ~24 pp on indoor scenes (DIODE 64.9% vs 88.6%).
- Thresholds, calibration data only (same sets and `Tier.calibrate` as every tier; `results/sealed-holdout/thresholds-pico.json`):

| Pico format | Abstain if confidence < (standard) | (strict, effective) |
|---|---|---|
| PyTorch (reference) | 0.399 | 0.795 |
| ONNX FP32 | 0.400 | 0.792 |
| ONNX FP16 | 0.400 | 0.795 |
| ONNX INT8 | 0.412 | 0.775 |
| batch ONNX FP16 | 0.400 | 0.795 |
| batch ONNX INT8 | 0.401 | 0.790 |
| browser ONNX FP32 | 0.400 | 0.792 |
| browser ONNX INT8 | 0.412 | 0.775 |
| Core ML (FP16) | 0.400 | 0.795 |

- Files: ONNX fill-drop FP32 / FP16 / INT8, batch-capable FP32 / FP16 / INT8, full-token FP32 / INT8 for web browsers, and
  Core ML batch 1 / 16. The hash list is `results/sealed-holdout/ARTIFACTS-pico-SHA256.txt`.
- The grid check C1 passes on every file and runtime: ONNX Runtime CPU and CUDA, TensorRT FP16, OpenVINO, and Core ML
  (ALL and CPU+ANE); the worst value is +0.012.
- Calibration accuracy by format (Open Images calibration set, 6,692 views): FP32 4,579, FP16 4,583, batch INT8 4,556, and
  Core ML 4,585 within 10°.

## Addendum, 30 Sep 2026: Pico replaced by PICO-N70S80 (rotation-corner fix)

**Owner decision** in chat, 30 Sep ~14:00 AEST: "Yes, we should." (replace the Pico above with `PICO-N70S80`). The
29 Sep record above is kept unchanged as history. As before, Pico is not part of FREEZE v2 or the sealed holdout, and its
accuracy is development-panel accuracy.

**Why:** the package tests found that `PICO-N70L` misreads images turned by software with a growing canvas (flat black or
white corners) by ~180° with confidence 0.49–0.81; every training view had been a crop inside the rotated frame. The other
tiers were not affected on those tests.

| Role | Checkpoint (EMA weights) | SHA-256 |
|---|---|---|
| Pico (70 px, fill-drop) | `runs/PICO-N70S80/final.pt` | `6d0b9fb06cea347377c7b4e5ff0efd133bb3f6743e28f0ca5fa1b47cb54cb8a5` |
| from: frozen Pico | `runs/PICO-N70L/final.pt` | `6b823e474a911fa77d31294c0122187359e5a330cc9773e62c41e2c653fb8e50` |
| from: corner fine-tune | `runs/PICO-N70C/final.pt` | `652f3fbd980b6e2520285e2bd039195795b815f9335600f61f2ef30901f91434` |

- `PICO-N70C`: `PICO-N70L` fine-tuned 8,000 steps (LR 2e-5) with 20% of views turned whole with flat-colour corners
  (black, white, grey, random), otherwise N70L's recipe, teacher and data. `PICO-N70S80` = 0.2 × N70L + 0.8 × N70C
  (weight interpolation, no further training), which removed N70C's small development cost (diode_cam −1.65 pp).
- Every step was pre-registered with its acceptance criteria before the result it depends on
  (`TECHNICAL-REPORT.md` §8, addendum 30 Sep 01:22 AEST and rounds after it, including the deviations).
  The original absolute corner criterion (≤ 3% ≥150° errors on the corner panel) is failed by every tier, Nano included;
  the revised, paired criterion (corner-induced flips ≤ Nano's + 1 pp) is passed: 3.25% / 3.77% (Nano 3.56% / 3.83%,
  N70L 6.02% / 5.44%).
- Development accuracy equals N70L's within noise: five-seed W benchmark 92.23% (N70L 92.31%), diode_cam −0.82 pp
  (McNemar p 0.21); every other development set within ±0.5 pp except the 126-view traffic-camera sets (−2 and −1 views).
- Package tests 30/30, including corner tests at 10 angles with black, white and grey corners for every tier.
- The ONNX graphs are structurally identical to N70L's in all eight formats, so the Pico speed rows are unchanged
  (confirmed on Apple M4 Core ML: 0.86 ms batch 1 for both).
- The grid check C1 passes on every file and runtime: ONNX Runtime CPU and CUDA, TensorRT FP16, OpenVINO, and Core ML
  (ALL and CPU+ANE); the worst value is +0.045 (TensorRT FP16).
- Thresholds, calibration data only (`results/sealed-holdout/thresholds-pico-s80.json`):

| Pico format | Abstain if confidence < (standard) | (strict, effective) |
|---|---|---|
| PyTorch (reference) | 0.421 | 0.828 |
| ONNX FP32 | 0.426 | 0.833 |
| ONNX FP16 | 0.425 | 0.833 |
| ONNX INT8 | 0.415 | 0.833 |
| batch ONNX FP16 | 0.425 | 0.833 |
| batch ONNX INT8 | 0.423 | 0.853 |
| browser ONNX FP32 | 0.426 | 0.833 |
| browser ONNX INT8 | 0.415 | 0.833 |
| Core ML (FP16) | 0.423 | 0.838 |

- Files: same set and public names as above; hash list `results/sealed-holdout/ARTIFACTS-pico-s80-SHA256.txt`. The N70L
  files were moved to the release archive (kept).
- Calibration accuracy by format (Open Images calibration set, 6,692 views): FP32 4,565, FP16 4,566, batch INT8 4,532, and
  Core ML 4,570 within 10°.
- Core ML on the Neural Engine (CPU+ANE): 18 of 1,976 real development views change by more than 2° against FP32, all of
  them two-peaked with confidence ≤ 0.46; 8 are above the standard threshold, and the net effect is −3 correct answers
  (−0.15 pp). N70L had 3 such views.
- Not shipped: repainting detected rotation corners with the letterbox fill at inference. It helped every tier on the
  corner panel (+2.8 to +5.0 pp) but also triggers on 0.1–1% of development images (photos with their own frames), which
  would change reported numbers. Recorded for a later release.
