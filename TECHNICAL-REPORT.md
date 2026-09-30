# RightWayUp 1.0: technical report

ORTUS AI, 30 September 2026. This report is the detailed companion to `MODEL-CARD.md`. It documents how the released
models were trained and evaluated, including every caveat and every deviation from our pre-registered plans. It is
the closest thing to a paper we publish for this release. Every number cites its source record (§12).

Abbreviations: **W** = Woehrer 2026 (MambaOut-base CGD, arXiv:2603.25351, code and weights MIT), the published state
of the art on the COCO rotation benchmark. **Within 10°** = share of images whose predicted clockwise rotation is
within 10° of the truth (circular error), every image answered unless stated. **≥150°** = near-opposite
("upside-down") errors. **pp** = percentage points. Brackets after a difference are 95% bootstrap intervals.

## Contents

1. Summary
2. Task, model and tiers
3. Training data
4. The grid fix: removing JPEG-grid and resampling shortcuts
5. Findings on Woehrer 2026's benchmark
6. Evaluation protocol
7. Results
8. Pico: the sixth tier and its rotation-corner fix
9. File formats, runtime parity and per-format thresholds
10. Limitations
11. Reproducibility
12. Sources

## 1. Summary

- Six tiers (Pico, Nano, Fast, Balanced, Pro, Max) built from three DINOv2 checkpoints: a ViT-L/14 at 280 px (Max),
  a ViT-S/14 at 224 px (Fast) and a ViT-S/14 at 112 px (Nano), plus Pico, a ViT-S/14 at 70 px chosen after the freeze.
  Balanced and Pro are Fast → Max cascades routing about 10% / 20% of calibration images.
- Before the release we found that our earlier models read part of the angle from traces of digital rotation (the
  source photo's rotated JPEG block grid and the rotated spectrum of resampled pixels), which a physically rolled
  camera never shows. We rebuilt the training data losslessly, changed the rendering, added probes and penalties and
  retrained every tier under a pre-registered protocol (§4). Every shipped file passes the pre-registered probe check
  in its own runtime.
- Sealed final holdout (8 panels; frozen first, opened once, scored once): on 4,706 new COCO and Open Images photos Max
  scores 93.0% vs W 88.4%, with 33 vs 185 upside-down errors; with simulated CCTV degradation 88.2% vs 49.3%; on two
  unseen thermal cameras (224 images) 87.5% vs 50.9%. Max is significantly better than W on 6 of 8 panels and ties on
  the other two. Nano and Fast trail W on clean COCO photos (§7.1).
- On four held-out CCTV cameras, re-used once for the release (second use, disclosed), Max scores 100.0% (380 views) vs W 75.0%
  and Deep-OAD 82.4% (§7.2).
- On RotBench, an independent benchmark with a human baseline, Max and Pro score 1.00 at every rotation on both splits
  with no upside-down answers, above the published human baseline on RotBench-Small (50 photos, by 1–3 points); W 0.90 / 0.97 mean
  (Small / Large). Pre-registered and scored once (§6.5, §7.6).
- W's benchmark: W 98.0% (five-seed mean) vs Max 98.8%; after one JPEG q90 re-save of every view W falls to 30.2% and
  Max stays at 98.4% (§5). The rotated source grid is our hypothesised mechanism, not a proven one.
- Abstention thresholds are fixed on calibration data only, per file format. They are useful on photos and visible-
  light CCTV and not reliable on thermal footage (§7.4).

## 2. Task, model and tiers

**Task.** Given one image, estimate the clockwise in-plane rotation of its content over the full circle (360 bins of
1°), a confidence, and whether to abstain.

**Network** (`rotlab/model.py` `RotNet`, `rotlab/model_x.py`, `rotlab/focus.py`). DINOv2 backbone → concat(CLS, mean
patch token) → LayerNorm → Linear(·, 1024) → GELU → Linear(1024, 360) → softmax. Pinned backbones:
`vit_small_patch14_dinov2.lvd142m` (revision `4610ca14…`, weights SHA-256 `04d27f34…`) and
`vit_large_patch14_dinov2.lvd142m` (revision `4741e1ca…`, `0424a5d1…`), full hashes in `RECIPE.md`.

**Input and decoding.** The image is letterboxed (aspect kept, bicubic) onto a square canvas of fill colour RGB
(124, 116, 104) and ImageNet-normalised. The answer is the argmax bin refined by the probability-weighted mean offset
over ±10 bins; confidence is the probability mass within ±10° of the answer.

**Fill-drop** (small models). The letterbox fill carries no information, so Pico, Nano and Fast drop the patch tokens
that are entirely fill before the transformer and pool only the kept tokens. The one-image ONNX files implement this
with a data-dependent token selection; the batch-capable files instead add −10⁴ to the attention logits of fill keys in
every block and leave fill tokens out of the pool. At FP32 both give the same answers (1,976 real views, identical
within-10°, max |Δprob| 7e-8). Max uses all tokens.

**Tiers** (`FREEZE.md`; PyTorch/FP32 reference thresholds; per-format values in §9):

| Tier | Checkpoint | Canvas | Route to Max if conf. < | Abstain if conf. < (standard / strict) |
|---|---|---|---|---|
| Pico | PICO-N70S80 (ViT-S/14, 12 blocks) | 70 px, fill-drop | — | 0.421 / 0.828 |
| Nano | GF-SOUP-SV2 (ViT-S/14) | 112 px, fill-drop | — | 0.533 / 0.800 |
| Fast | GF-SV3-kdM3 (ViT-S/14) | 224 px, fill-drop | — | 0.628 / 0.735 |
| Balanced | Fast → Max | 224 → 280 px | 0.628 (~10% of calibration images) | 0.692 / 0.748 |
| Pro | Fast → Max | 224 → 280 px | 0.800 (~20%) | 0.729 / 0.729 |
| Max | M3-SOUP8 (ViT-L/14) | 280 px | — | 0.726 / 0.726 |

Strict is the effective value max(standard, calibrated strict); the raw calibrated strict values for Balanced, Pro
and Max are 0.7475, 0.695 and 0.680. For a cascade, the abstain threshold applies to whichever model answered.

**Lineage** (checkpoint records; `RECIPE.md` §4). Max = uniform weight average of eight checkpoints (M3 seeds 1–4,
stage A′ at canvas 140 and stage B′ at 280), initialised from the grid-fix arm-A Max GF-SOUP-A-F2b. Fast = GF-SV3-kd
(cue-KD recipe, 12,000 steps) re-distilled from Max for 4,000 steps. Nano = uniform average of two seeds (GF-SV2-A1/A2,
12,000 steps each). Pico = Nano fine-tuned at 56–70 px and distilled from Max (PICO-N70L), then interpolated with its
rotation-corner fine-tune (PICO-N70C): 0.2 × N70L + 0.8 × N70C (§8).

## 3. Training data

The training images are not redistributed, except our own Blender renders ([ortusai/rightwayup-renders](https://huggingface.co/datasets/ortusai/rightwayup-renders), CC BY
4.0); per-image manifests identify every image (`manifests/training/`, `DATA-CARD.md`).

| Group | Families | Training rows used |
|---|---|---|
| Photos | PASS (594,056); Open Images test subset (33,275) and V7 train (297,763); COCO 2017 train CC BY (12,369), unlabeled (9,480) and public-domain-like (648); CommonCatalog CC-BY (276,016) | **1,223,607** |
| Scans | DIODE train scenes | 14,669 |
| CCTV | MEVA, 10 training cameras incl. thermal G475/G476 | 3,749 |
| Renders | ORTUS AI Blender renders of Poly Haven assets | 3,579 |
| **Total** | 23 grid-free families | **1,245,604** (1,244,222 distinct images) |

- "Used" counts are the trainer's own per-family counts after the load-time filter that drops any image within dHash
  6 bits of the 1,030 origin-benchmark photos (6,068 rows of the packs). The 23 families hold 1,251,672 rows.
- **Distinct images.** 1,351 groups of photo rows (2,740 rows) hold the same picture: rows that share a Flickr photo
  id or have byte-identical stored images. 880 groups are PASS photos that are also in Open Images or COCO, 165 are
  repeats inside PASS, 280 are byte-identical re-uploads under different Flickr ids inside CommonCatalog (259 by the
  same author), 25 are COCO photos also in Open Images and 1 is inside the Open Images test subset. Counting each
  group once gives **1,244,222 distinct images used (about 1.24 million)**, 1,382 fewer than the rows; for photos,
  1,222,225 distinct in 1,223,607 rows. The duplicates were trained as separate rows (a repeated photo is sampled
  slightly more often); they are listed in `manifests/training/SUMMARY.json` (`duplicates`).
- Max, Fast and Pico were trained on all 23 families. **Nano's final training used the 16 families that existed
  before the Open Images V7 and CommonCatalog expansion: 705,075 training rows (683,078 photo rows).** Earlier checkpoints of every
  chain (the initialisations) were trained on JPEG-stored versions of the same sources (the 24 Sep packs and the
  27 Sep campaign packs).
- Licence classes (used rows): CC BY family 1,225,738 (incl. MEVA, CC BY 4.0), public-domain-like 1,618 (CC0, Public
  Domain Mark, "no known copyright restrictions", US Government Work), MIT 14,669 (DIODE), CC0 assets 3,579 (Poly
  Haven, rendered by ORTUS AI).
- Every COCO, Open Images and CommonCatalog row (629,551 photos) had its current Flickr licence re-checked at fetch
  time; PASS (594,056) carries the licence of the PASS dataset record. Open Images rows require `Rotation = 0`, which is
  the recorded display orientation (Google's 2018 automatic match to the Flickr display), not a human check.
- Screens at fetch time: dHash ≤ 8 bits against 20,083 evaluation originals, ≤ 6 bits against the 681,574 rows of the
  existing packs and within new images, id exclusions of every pack, evaluation, holdout and calibration set (the
  expansion also excluded all COCO Flickr ids and all Open Images validation ids). MEVA, DIODE, Poly Haven, R-LiViT and
  Aalborg frames are separated by camera, scene or asset allowlists, not by dHash.

Source: `DATA-CARD.md`, which reproduces the per-family counts from the pack receipts and the training log.

## 4. The grid fix: removing JPEG-grid and resampling shortcuts

### 4.1 Diagnosis

A digitally rotated photo carries traces of the rotation that are unrelated to its content:

- **Rotated JPEG grid.** Every training parent of the earlier models was stored as a 448-px JPEG (q92, 4:4:4). When a
  parent is rotated, its 8 × 8 block grid rotates with the content and encodes the angle modulo 90°.
- **Rotated spectral support.** A view resampled from a pixel-grid parent at about 1:1 keeps the parent's frequency band
  (a square aligned with the parent raster) rotated by the angle. This is present even in lossless images and survives
  supersampled rendering.

A physically rolled camera has neither: it compresses its own frame, so its grid is aligned with the frame edges.

A pre-registered audit of the earlier release models (`research/trace-audit/REPORT.md`; content-free probe images,
n = 500 per family) found both cues. The probe statistic is **ρ₄ = mean cos(4 · (prediction − θ))** over probes rotated
to known angles θ: 0 when predictions ignore the lattice, 1 when they lock onto it modulo 90°.

| Earlier model | ρ₄ on grey + noise, JPEG q75 (G75) | pink noise JPEG q75 (P75) | lossless pink noise (PPNG) | Camera-like rendering, within-10 change (2,000 Open Images calibration views) |
|---|---|---|---|---|
| Small model S2 (224 px) | 0.958 [0.938, 0.976] | 0.796 [0.766, 0.824] | 0.922 | −6.05 pp |
| Large model L (224 px) | 0.933 [0.913, 0.950] | 0.726 [0.687, 0.763] | — | −5.65 pp |

At CCTV scale (MEVA frames downscaled about 2.6×, 1,080 views) camera-like rendering changed nothing (−0.09 pp
[−0.28, 0.00]); the traces vanish when a view is downscaled enough. Deblocking by shift-averaged JPEG re-application
did not remove the grid cue (P90 probe 0.66 → 0.56), and decoy grids cannot remove a true grid that is present.

### 4.2 Pre-registered criteria

Written 28 Sep 2026 ~10:40 AEST, before any grid-fix training result (`research/grid-fix/PREREG.md`). A model may ship
only if all four hold at its shipped canvas:

1. **C1, content-free probes:** argmax ρ₄ ≤ 0.20 on each of G75, P75 and PPNG (the campaign's pre-fix Max candidate
   SOUPF-AB8 at 280 px: 0.94 / 0.57 / 0.68).
2. **C2, camera-like views:** within-10 on grid-free camera-like development views ≥ the clean copy − 1.0 pp.
3. **C3, conflicting grid:** within-10 on a copy of the clean development views carrying a decoy JPEG grid at a random
   angle ≥ clean − 2.0 pp (SOUPF-AB8 −16.8 pp; the 24 Sep release models −10.8 pp (small) and −12.2 pp (large)).
4. **C4, no loss on honest data:** within-10 on grid-free camera-like development views ≥ the earlier model's, paired
   bootstrap by parent, lower bound ≥ −1.0 pp.

Decision rule: among candidates passing 1–4, the highest within-10 on the grid-free camera-like development set.
Secondary, reported, not gating: the standard selection panels (expected to drop, since they reward the shortcut), W's
five-seed benchmark and its JPEG q90 copy, the MEVA and R-LiViT panels, and the paired comparison with W. Baseline on
the grid-free development set (3,934 views of DIODE, Poly Haven and MEVA): SOUPF-AB8 scored 99.75% clean but lost
16.8 pp with one conflicting grid.

### 4.3 Data rebuild

- Every photo family was re-fetched from its original source at ≥ 1,020 px on the long side and stored as lossless
  448-px PNG (Lanczos, never JPEG-encoded). DIODE, MEVA and Poly Haven parents were rebuilt from lossless archive
  originals. Labels were re-verified by dHash against the old pack rows (≤ 8 bits, else replaced), and 2,100 sampled
  re-fetches were compared pixel-wise with the old parents (PSNR median 35.5–38.7 dB, none flagged).
- Images that could no longer be fetched at full size were replaced by similar images from the same source, split and
  licence class, screened like new data: 202,909 replacements in total (PASS 194,297; Open Images 5,890 + 574; COCO
  1,570 + 573 + 5). COCO pools were exhausted for 2,056 rows; 4,554 DIODE rows from scenes that the earlier archive
  assigns to calibration and test splits were removed. Result: 711,143 rows in 16 families (from 717,753).
- A development set of 3,934 grid-free views (DIODE, Poly Haven, MEVA; clean, camera-like and conflicting-grid copies)
  and grid-free photo development sets were built from development splits only.

### 4.4 Recipe and deviations (in the order they were recorded)

Pre-registered arms: **A** continues from the earlier Max (SOUPF-AB8) and **B** starts from pinned DINOv2-L; both
with the "scrub" renderer (supersampled rendering `ss`, decoy grids, camera-like degradations, content-free probes with
a uniform target), and knowledge distillation from the earlier Max seeing the **clean** grid-free view. Deviations:

1. 10:30: grid-free photo development sets added as a secondary readout (the DIODE/Poly/MEVA sets saturate).
2. 10:55: the DIODE development parents had been in the old training mix, so C4 was judged on the Poly Haven and MEVA
   parts only; DIODE training scenes assigned to calibration/test in the earlier archive stay excluded.
3. 11:55 (before any main run): an exploratory 2,000-step pilot removed grid reading but not resampling-trace reading.
   For both arms the probe share rose 0.05 → 0.10 and decoys 0.5 → 0.6, half of them pure resampling decoys. Criteria
   unchanged.
4. 12:20 (written 13:05): to fit the time budget, arm A ran 8,000 + 2,700 steps instead of 12,000 + 4,000; the small
   model was distilled in parallel from the earlier Max instead of from the new Max.
5. 12:20 (written 13:10): arm B ran 2 seeds instead of 4.
6. 13:35 (owner): the small model is re-distilled from the chosen new Max, as pre-registered.
7. 13:35, exploratory: G75 ρ₄ went negative (predictions avoided the grid axes). C1 stays one-sided as registered;
   |ρ₄| is reported alongside.
8. 13:45: the first grid-fix small model passes all four criteria at 112 px (Nano) but fails C1 at 224 px (Fast):
   P75 0.587, PPNG 0.815. A **distribution-level** statistic was added as a secondary readout: soft ρ₄ =
   E_p cos 4(φ − θ) over the output distribution (argmax ρ₄ can be large while the output is nearly flat).
9. 14:40, diagnosis before any fix result: the residual cue is the **rotated spectral support**. Probes rendered with the
   supersampled training renderer are read almost as strongly as plain ones (earlier Max 0.72 vs 0.85); after a 1.5×
   output-raster Lanczos downscale the cue is mostly gone (0.13). Every small-model view at canvas ~224 carried the cue
   aligned with the true angle, so training rewarded it. Fix under test: `down=1.5` (every rendered view, and the
   teacher's clean view, downscaled 1.5× before letterboxing).
10. 14:55: a cue-free same-resolution photo set (**gf_photo2**) was built: the 1,856 Open Images development views
    re-rendered from 2× parents and downscaled to the identical view size (ids, order and angles unchanged).
11. 15:00: the final small re-distillation started with the arm-A soup as teacher (before the F2 finish), to use idle
    GPUs.
12. 16:20, **Max decision by the registered rule**: GF-SOUP-A-F2b (arm-A soup + a 2,000-step finish at 224, seed 8201)
    passes 1–4 with the highest camera-like development score (99.69%); the arm-B models pass C1 but trail by 5.3 pp
    overall and ~20 pp on DIODE.
13. 16:28: exploratory Fast runs from DINOv2-S (no warm start) because every small model from the earlier small
    lineage failed C1 at 224.
14. 18:40: two of these runs were OOM-killed and resumed from milestones with the cosine learning rate at that step
    (optimiser state not restored).
15. 18:45, **expanded data (MIX v3)**: owner allowed doubling the data. New grid-free families: Open Images V7 train
    (author cap 20) and CommonCatalog CC-BY (author cap 60, NSFW filter), each with a per-photo current licence check.
    MIX v3 = the 16-family mix × 0.60 + new families 0.40 in proportion to their rows. The final Max ("M3"): init
    GF-SOUP-A-F2b, stage A′ 8,000 steps at 140 + stage B′ 3,000 at 280, 4 seeds, uniform soups.
16. 20:55–22:15: **cue-KD** (with probability p the student sees the cue-bearing ~1:1 view while the teacher sees the
    cue-free downscaled view, trained by distillation only) and a **lean penalty** (loss + L · mean(soft ρ₄²) on probes)
    were added; M3 stage B′ gained quantisation-aware training (`--qat`, exact ONNX Runtime dynamic-INT8 fake quant)
    and leanpen 20, with B′ shortened to 1,500 steps.
17. 23:15 (owner): the CommonCatalog fetch was extended to pass 1.2 million source photos; stage-B′ runs starting
    after 00:15 include the last partial pack through the mix recomputation.
18. 23:40: a storage-quota incident killed three runs; they were resumed or restarted (listed in the log).
19. 23:55: small-model plan fixed before results (Small-B = cue-KD + lean penalty, two seeds; Small-C = DINOv2-S line).
20. 01:55 and 04:20: two more OOM kills, resumed from milestones.
21. 05:10, speculative before the owner's Max decision: the two Fast near-ties were re-distilled from M3-SOUP8 for
    4,000 steps; each ships only if it still passes 1–4 and is not worse than its parent on the camera-like set.

### 4.5 Evidence

**The cue inflates photo accuracy at 224–280 px** (gf_photo2 vs the paired 1:1 renders of the same 1,856 photos,
within-10):

| Model | 1:1 clean | cue-free clean | difference |
|---|---|---|---|
| Pre-fix Max candidate (SOUPF-AB8 @280) | 93.75% | 76.02% | +17.73 [15.95, 19.50] |
| Pre-fix small model (SOUPS-SKG-s2 @224, Fast) | 90.09% | 69.29% | +20.80 [18.91, 22.68] |
| same small model @112 (Nano) | 74.62% | 74.41% | +0.22 [−0.43, 0.86] |

Every photo number measured on 1:1 renders of 448-px parents at canvas 224/280 overstated camera-like accuracy by 12–21
pp; at 112 px it did not. CCTV, DIODE and mixed panels showed no cue dependence (their views are already downscaled
enough; 1.5× downscaled copies scored within ±5 views).

**W on the same cue-free photos:** 77.64% clean and 23.22% with camera JPEG. The pre-fix Max candidate did **not** beat W
on camera-like clean photos (76.02%, −1.62 [−3.99, 0.81]); the grid-fix arm-A soup did (83.30%, +5.66 [3.50, 7.76]).

**C1 is sensitive to seeds on near-flat outputs.** The three finished arm-A variants:

| Variant (@280) | C1 argmax ρ₄ (max over probes) | soft ρ₄ (PPNG) | C1 |
|---|---|---|---|
| GF-SOUP-A-F2 | 0.410 | 0.018 | fail |
| **GF-SOUP-A-F2b** | **0.150** | 0.012 | **pass** |
| GF-SOUP-A-F2S (F2 + F2b) | 0.254 | 0.014 | fail |

All three have near-uniform probe outputs (soft ρ₄ ≤ 0.018), so the argmax statistic is decided by tie-breaks and
F2b's pass is partly seed luck. We record it as such. W-benchmark margins vary by seed too: arm-A seeds scored
+44, +48, +41 and +37 images over W on the five-seed benchmark at 280 px.

**Arm B** (from DINOv2-L, 2 seeds) passes C1 cleanly (argmax ≤ 0.063, soft 0.000) but trails: camera-like
development 94.36% / 94.00% (arm A 99.69%), DIODE 79.67% / 78.39%, cue-free photos 82.00% / 81.25% (vs W +4.36 / +3.61).

**Small models at 224 px.** Every small model warm-started from the pre-fix small lineage failed C1 at 224
(argmax 0.41–0.82) while passing at 112. `down=1.5` halved the reading in 4,000 steps at no cost on cue-free photos;
cue-KD brought GF-SV3-kd to 0.197 (marginal pass); the lean penalty halved the distribution lean. The shipped Fast,
GF-SV3-kdM3 (re-distilled from M3-SOUP8), has C1 0.111 (FP32 ONNX) / 0.137 (INT8 ONNX), soft 0.003 / 0.007, cue-free
photos 76.29% (vs W −1.3 [−3.5, 0.9]).

**INT8 amplifies a residual lean.** The dynamic-INT8 export of GF-SOUP-A-F2b failed C1 (argmax 0.543, soft 0.110)
while FP32, FP16 and weight-only 8-bit passed; per-tensor activation quantisation turned a 1% lean into 11%. On real
content the INT8 file was no worse (cue-free photos 81.47% vs 81.09% FP32) and it answered no probe at the Max
threshold, but it is about 5 pp less confident, which is why thresholds are calibrated per format (§9). A short
INT8-aware finish removed the amplification (soft 0.110 → 0.011) and was adopted in M3 stage B′.

**M3 and the Max choice.** All four M3 seeds pass C1–C4 with near-zero lean and gain 0.4–1.0 pp over GF-SOUP-A-F2b on
cue-free photos, at −0.25 to −0.5 pp on the camera-like development set and −1 to −1.7 pp on DIODE. **M3-SOUP8**:
C1 −0.047 (PyTorch) / −0.014 (FP32 ONNX) / +0.116 (INT8 ONNX), camera-like development 99.54%, DIODE 98.35%, cue-free
photos 83.57% (vs W +5.9 [3.8, 8.1]), five-seed W benchmark +41 [8, 76]. The owner chose M3-SOUP8 as Max, GF-SV3-kdM3
as Fast and GF-SOUP-SV2 as Nano before the freeze.

**Every shipped file.** C1 passes on every file in its target runtime: ONNX Runtime CPU (FP32, INT8, batch INT8),
ONNX Runtime CUDA (FP16, batch FP16), TensorRT FP16, OpenVINO CPU and Core ML (all compute units), for Nano, Fast, Max
and Pico.

### 4.6 What the fix does not claim

The fix removes the models' use of these cues on content-free probes and on the conflicting-grid set; it does not
make any claim about physically rolled cameras beyond the panels reported here. The cue-free photo set is a
simulation of camera-like rendering (2× parents plus downscale), not a camera recording.

## 5. Findings on Woehrer 2026's benchmark

The benchmark: 1,030 COCO val2014 photos, each rotated to a random angle under five test seeds and cropped to the
largest axis-aligned rectangle inside the rotated photo; within-10 is reported as the five-seed mean. We rebuilt the
views with the upstream code (all 5,150 views match our pixel hashes) and ran W's public checkpoint.

- **Seeds.** W's paper reports the five-seed mean, 98.0% (1009.8 / 1030). Our earlier documents quoted 98.6% (1016),
  which is seed 0 of our rebuild and the best of W's five seeds there. That was our choice of figure, not the paper's.
  Per-seed scores also depend on the order in which test files are listed, so only the five-seed mean is portable.
- **JPEG q90.** Saving each benchmark view once as JPEG q90 (Pillow 11.0.0, 4:2:0) drops W from 98.0% to 30.2%
  (five-seed mean, 311.0 / 1030); its answers snap to the nearest multiple of 90° (76% of answers within 2° of an axis,
  against 4% of true angles). A lossless PNG round trip leaves W unchanged on all 1,030 seed-0 views; the effect grows
  with compression (seed 0: q95 40.5%, q90 31.4%, q75 25.9%). The released tiers on the same JPEG copies: Max 98.4%, Pro
  98.2%, Balanced 97.5%, Fast 96.4%, Nano 95.9%, Pico 92.2% (clean: 98.8 / 98.7 / 97.7 / 96.5 / 96.0 / 92.2%).
- **Mechanism (hypothesis).** The benchmark decodes a JPEG, rotates the pixels and keeps the view lossless, so the
  source's 8 × 8 block grid rotates with the content and carries the angle modulo 90°. One JPEG save adds a grid
  aligned with the frame. The snapping and the dose response are what a model relying on the rotated grid for the
  fine angle would do. **We have not proven this mechanism**; a render that removes the source grid before rotation
  would test it directly.
- **Crop shape.** The maximal-area crop changes field of view and scale with the angle. With a neutral 4:3 crop on the
  same photos (one fresh angle draw, seed 7, PIL bicubic) W scores 97.2% and Max 99.0% (development "fair photos"). In a
  paired test on 2,124 COCO val2017 photos where only the crop changes, W gained 17 views on the max-area crop (45 / 28
  discordant, exact McNemar p = 0.06). **The crop effect is therefore not statistically established.** We report the
  benchmark both ways and never train on the angle-dependent crop.
- **The release on the benchmark as published:** Max 98.8% vs W 98.0% (five-seed means), +41 images [8, 76] paired over 5,150
  views. On the sealed holdout's max-area-crop panel (new photos) Max and W tie (95.3% vs 94.1%, +25 [−1, +50]).
- The author made all of this checkable by releasing the code, weights and test list, and removed from training the
  915 benchmark photos that reappear in COCO train2017. None of these findings is a criticism of the author's work;
  the crop and the rendering are common in RotNet-style code. A script that rebuilds every view and runs W's public
  checkpoint (`benchmarks/woehrer-2026/reproduce_w_jpeg.py`) reproduces the numbers above (5,150 / 5,150 views pixel-identical, JPEG five-seed mean 311.2 with the
  checkpoint vs 311.0 with the ONNX export).

## 6. Evaluation protocol

### 6.1 Calibration and thresholds

Thresholds are fitted on calibration sets only: MEVA cameras G419, G299 and G330 (270 views), the Poly Haven
calibration split (1,192) and Open Images validation ranks 4000–7999 (6,692), each image clean and degraded, the
three sets weighted equally. **Route** = the small model's confidence at a 10% (Balanced) / 20% (Pro) route share.
**Standard** = the 10% confidence quantile of the answering model (~90% answered). **Strict** = the lowest of 401 grid
values with mean answered error ≤ 1%, never looser than standard. No test data is used.

### 6.2 Sealed final holdout (once only)

- **Freeze.** `FREEZE.md` was written and `camp_final freeze` created `FROZEN.json` (SHA-256 `5caf3220…`) on
  29 Sep 2026, 11:19 Sydney time, before any holdout view was rendered. FROZEN binds the three checkpoints, the tier
  table and thresholds, the comparator files, the four holdout source manifests and the SHA-256 of the 14 code files
  that render and score the holdout (`evaluation/code-final/rotlab/` in this repository carries those exact files).
- **Sets.** New COCO holdout: 2,201 val2017 photos (fixed crop, max-area crop and CCTV-degraded views). New Open Images
  holdout: 2,505 validation photos (CC BY 2.0 listing; fixed crop and degraded). R-LiViT final locations 6 and 7
  (traffic intersection A), 74 RGB and 74 thermal images. Aalborg long-term thermal drift, 150 images from one thermal
  camera. None had been trained on, used for selection or rendered before the freeze. The source files were
  re-downloaded from their public origins and checked byte-for-byte against the pre-freeze manifests.
- **Order.** Open → render each view set with a create-exclusive receipt → independent review of FROZEN and the
  receipts → `SCORING-APPROVED.json` (binding the per-format thresholds) → score each frozen stage once → aggregate.
  Opened 18:40, review passed 18:56, scoring finished 19:17 (Sydney time, 29 Sep).
- **Comparator.** W on identical pixels, frozen as ONNX `5fd9a686…` with weights `44d58a12…`.
- **Statistics.** Paired difference in correct images vs W with a 95% bootstrap interval (10,000 resamples).
- **Selection** used development panels only (grid-free development, new-validation, five-seed, cue-free photos,
  R-LiViT development, W panels), never the holdout.

### 6.3 Held-out CCTV and scene sets of 24 Sep (A9, second use)

The two sets that scored the superseded 24 Sep candidate (MEVA test set, frozen split, 560 views; clean scene test,
2,854 views) were re-scored once with the release files on 30 Sep, after a written pre-registration
(`results/heldout-rescore/PREREG.md`, 14:23 AEST):

- **Second use.** Both sets were opened once on 24 Sep. The released models were developed after those results were known
  (for example the weakness on thermal camera G474). They never trained on them, and no evaluation store of the
  release contained them.
- **Training sources checked.** No row of the 36 source manifests comes from the six test cameras, MEVA G339, DIODE
  scene 12 or any clean-test Poly Haven parent.
- **Development overlap.** The development set used to choose the released models contains other clips of MEVA cameras G331 and
  G639, two of the six test cameras. The held-out claim therefore uses the other four cameras (G329, G420, G421, G474;
  380 views) and the whole clean scene test; the six-camera figure is reported with a label.
- **Scorer.** The staged one-image ONNX FP32 files, hash-verified, run through the release package (ONNX Runtime CUDA,
  FP32 thresholds, Balanced/Pro routing as shipped). Comparators are the stored 24 Sep predictions on identical views;
  their 24 Sep numbers were reproduced exactly before scoring. Paired intervals resample frames (MEVA; clean and
  degraded views of a frame together) or parents (clean test).

### 6.4 Development panels

Used for selection and reported as development results: W's five-seed benchmark and its JPEG q90 copy, the cue-free
photo set and its camera-JPEG copy, the MEVA test set's development split (646 views, 7 unseen cameras), Common mixed
(1,200), Fresh DIODE (validation scenes 19–24, 300), fair photos (1,030, neutral crop) and the older Open Images panel
(3,345; it rewards the grid cue, see §4.5).

### 6.5 RotBench (public benchmark, second use)

- **Benchmark.** RotBench (arXiv 2508.13968; Hugging Face `tianyin/RotBench`, dataset commit `4ebf3d3b…`, Apache-2.0):
  RotBench-Small (50 photos) and RotBench-Large (300 photos), each turned 0/90/180/270° counter-clockwise with PIL. The
  score is 4-way accuracy per rotation (prediction snapped to the nearest quarter turn); we also report the mean,
  within 10° and ≥150° errors. The images are square, so the official protocol (no expand) and a full-frame variant
  (expand) give identical images and numbers. Views are rebuilt with the unchanged 24 Sep code; their order and
  angles were checked against the stored 24 Sep comparator files before scoring.
- **Status.** RotBench is public and had been used to score the superseded 24 Sep candidate, so this is a second use
  of the benchmark and the release candidate's results were known. The new training families were screened against RotBench by
  perceptual hash; no evaluation store of the release contained RotBench before this run.
- **Pre-registration and scoring.** Written before any prediction of the released models on RotBench (`results/rotbench/PREREG.md`,
  owner request of 30 Sep). The staged one-image ONNX FP32 release files, through the release package (ONNX Runtime
  CUDA, Balanced/Pro routing as shipped, every image answered), were scored once at ~15:25 AEST on 30 Sep 2026; all
  tiers are reported, with no change of pipeline or thresholds after scoring.
- **Comparators.** Woehrer 2026, Deep-OAD and GeoCalib: stored 24 Sep predictions on identical views (signs fixed on
  calibration photos), reproduced exactly. Human and vision-language model rows: as published by RotBench, on
  RotBench-Small only; not re-run.

## 7. Results

### 7.1 Sealed final holdout (29 Sep 2026)

Within 10°, every image answered; paired difference vs W in images [95% CI]; **bold** = significant.

| Test (images) | W | Max | Pro | Balanced | Fast | Nano |
|---|---|---|---|---|---|---|
| New COCO photos (2,201) | 93.6% | **95.6% (+45 [+19, +71])** | **95.5% (+41 [+14, +68])** | 94.0% (+8 [−20, +36]) | **90.4% (−70 [−100, −41])** | **87.3% (−138 [−171, −106])** |
| same, max-area crop | 94.1% | 95.3% (+25 [−1, +50]) | 95.0% (+20 [−6, +46]) | 93.2% (−21 [−49, +7]) | **90.2% (−87 [−116, −58])** | **88.0% (−136 [−167, −105])** |
| same, CCTV-degraded | 54.7% | **92.5% (+831 [+783, +879])** | **92.4% (+829 [+781, +877])** | **91.6% (+812 [+764, +860])** | **89.3% (+760 [+712, +808])** | **86.5% (+699 [+649, +750])** |
| New Open Images photos (2,505) | 83.8% | **90.7% (+174 [+135, +214])** | **90.5% (+168 [+128, +209])** | **89.1% (+134 [+93, +175])** | 84.5% (+17 [−25, +59]) | **75.5% (−207 [−254, −160])** |
| same, CCTV-degraded | 44.5% | **84.4% (+1000 [+947, +1055])** | **84.1% (+991 [+937, +1045])** | **83.2% (+969 [+915, +1024])** | **78.9% (+861 [+806, +917])** | **73.5% (+727 [+669, +785])** |
| R-LiViT traffic camera, RGB (74) | 98.6% | 97.3% (−1 [−5, +2]) | 93.2% (−4 [−9, +0]) | **87.8% (−8 [−14, −2])** | **87.8% (−8 [−14, −2])** | **77.0% (−16 [−24, −8])** |
| R-LiViT traffic camera, thermal (74) | 37.8% | **95.9% (+43 [+34, +52])** | **95.9% (+43 [+34, +52])** | **95.9% (+43 [+34, +52])** | **90.5% (+39 [+30, +48])** | **74.3% (+27 [+17, +37])** |
| Aalborg thermal camera (150) | 57.3% | **83.3% (+39 [+24, +54])** | **78.0% (+31 [+16, +46])** | **68.7% (+17 [+2, +32])** | 61.3% (+6 [−8, +20]) | 48.7% (−13 [−29, +3]) |

Upside-down errors (≥150°), W vs Max: new COCO 40 vs 4; max-area crop 36 vs 5; COCO degraded 78 vs 9; new Open Images
145 vs 29; Open Images degraded 173 vs 24; R-LiViT RGB 1 vs 0; R-LiViT thermal 33 vs 0; Aalborg 37 vs 25. Routed
share (Balanced / Pro): COCO 11% / 32%, max-area 10% / 30%, COCO degraded 13% / 34%, Open Images 27% / 50%, Open
Images degraded 33% / 55%, R-LiViT RGB 15% / 58%, R-LiViT thermal 31% / 80%, Aalborg 26% / 55%. Pico is not part of
this evaluation.

### 7.2 Held-out CCTV and scene sets (A9, 30 Sep 2026)

| Slice (views) | W | Deep-OAD | GeoCalib | Pico | Nano | Fast | Balanced | Pro | Max |
|---|---|---|---|---|---|---|---|---|---|
| Held out: 4 MEVA cameras G329/G420/G421/G474 (380) | 75.0% (47) | 82.4% (26) | 23.9% (86) | 80.0% (21) | 92.6% (21) | 97.9% (8) | 98.4% (6) | 99.2% (3) | **100.0% (0)** |
| of which clean / degraded (190 / 190) | 88.9% / 61.1% | 84.7% / 80.0% | 22.1% / 25.8% | 83.7% / 76.3% | 94.2% / 91.1% | 98.4% / 97.4% | 98.4% / 98.4% | 98.4% / 100.0% | 100.0% / 100.0% |
| camera G329 (90) | 84.4% | 72.2% | 30.0% | **41.1%** | 93.3% | 98.9% | 100.0% | 100.0% | 100.0% |
| thermal camera G474 (110) | 53.6% (47) | 67.3% (26) | 20.9% (29) | 86.4% (14) | 89.1% (12) | 93.6% (7) | 94.5% (6) | 97.3% (3) | 100.0% (0) |
| All 6 cameras, G331/G639 = development cameras of the release (560) | 78.0% (48) | 85.4% (26) | 24.6% (138) | 86.4% (21) | 95.0% (21) | 98.6% (8) | 98.9% (6) | 99.5% (3) | 100.0% (0) |
| Clean scene test: DIODE scene 12, MEVA G339, Poly Haven (2,854) | 70.6% (98) | 82.8% (113) | 25.0% (749) | 96.5% (11) | 98.0% (7) | 97.8% (10) | 98.0% (6) | 98.7% (5) | **98.8% (5)** |
| DIODE scene 12 (372) | 53.8% | 64.8% | 20.7% | 85.8% | 91.4% | 91.1% | 92.2% | 94.4% | 94.9% |

Max − W: +95 views [+77, +113] of 380 and +806 [+754, +858] of 2,854; Max − Deep-OAD: +67 [+52, +82] and +459
[+413, +506]. The superseded 24 Sep Max scored 98.7% (375 / 380) on the same four cameras and 98.5% on the clean test.

### 7.3 Development panels

| Test (images) | W | Max | Pro | Balanced | Fast | Nano | Pico |
|---|---|---|---|---|---|---|---|
| W's benchmark, seed 0 (1,030) | 98.6% | 99.2% | 99.0% | 98.4% | 96.9% | 95.4% | 92.3% |
| same, 5 seeds (5,150) | 98.0% | 98.8% | 98.7% | 97.7% | 96.5% | 96.0% | 92.2% |
| same, 5 seeds, JPEG q90 | 30.2% | 98.4% | 98.2% | 97.5% | 96.4% | 95.9% | 92.2% |
| Realistic photos, cue-free (1,856) | 77.6% | 83.6% | 83.5% | 81.8% | 76.3% | 74.2% | 70.0% |
| same, camera JPEG | 23.2% | 81.1% | 80.9% | 79.2% | 74.8% | 73.9% | 69.9% |
| MEVA development cameras, real CCTV (646) | 77.5% | 99.8% | 99.8% | 99.7% | 98.8% | 98.0% | 96.4% |
| Common mixed CCTV-like (1,200) | 25.7% | 96.7% | 96.2% | 95.6% | 93.9% | 93.2% | 91.8% |
| Fresh DIODE scenes (300) | 28.0% | 98.3% | 97.3% | 95.0% | 91.3% | 89.7% | 86.0% |
| Fair photos (1,030) | 97.2% | 99.0% | 98.8% | 98.5% | 96.9% | 94.8% | 91.9% |
| Open Images photos, old panel (3,345) | 81.6% | 91.2% | 90.8% | 89.3% | 83.9% | 74.7% | 68.8% |

The seed-0 row is shown only because it was quoted before; compare the five-seed mean.

### 7.4 Abstention

Sealed holdout, standard operating point (share answered / wrong among answered):

| Test | Max | Pro | Balanced | Fast | Nano |
|---|---|---|---|---|---|
| New COCO photos | 91% / 2.1% | 91% / 2.2% | 89% / 3.4% | 89% / 4.6% | 89% / 7.1% |
| same, max-area crop | 91% / 2.4% | 91% / 2.7% | 90% / 4.5% | 90% / 5.5% | 90% / 6.3% |
| same, CCTV-degraded | 86% / 3.0% | 86% / 3.1% | 87% / 4.0% | 87% / 4.5% | 88% / 6.8% |
| New Open Images photos | 74% / 1.2% | 74% / 1.7% | 75% / 3.2% | 73% / 4.4% | 71% / 10.3% |
| same, CCTV-degraded | 64% / 2.4% | 64% / 2.8% | 66% / 4.1% | 67% / 5.5% | 70% / 11.1% |
| R-LiViT RGB | 93% / 2.9% | 93% / 7.2% | 84% / 11.3% | 85% / 14.3% | 96% / 21.1% |
| R-LiViT thermal | 84% / 0.0% | 82% / 0.0% | 86% / 3.1% | 69% / 5.9% | 77% / 15.8% |
| Aalborg thermal | 80% / 12.5% | 84% / 19.8% | 83% / 27.2% | 74% / 31.5% | 77% / 50.0% |

At the strict point, Nano answers 25% of the Aalborg images with 62.2% wrong. **Confidence is not reliable on unseen
thermal cameras for any tier.** On the A9 held-out cameras (380 views) at the standard point Max answers 99.7% with
100.0% correct and Pico 97.9% with 80.6% correct; on the clean scene test Max answers 97.7% with 99.7% correct. Strict-
point tables for every tier: `RESULTS.md` §4 and `results/heldout-rescore/RESULTS.md`.

### 7.5 Speed

Single image (batch 1), median: RTX PRO 4500 TensorRT FP16 one-image files Pico 0.55 / Nano 0.55 / Fast 0.68 ms, Max
5.19 ms (W 4.04 ms); i7-1260P ONNX Runtime INT8, 4 threads: Pico 3.21 / Nano 7.73 / Fast 29.7 / Max 326 ms (W FP32
161 ms); Apple M4 Core ML (CPU+ANE): 0.86 / 1.04 / 3.44 / 53.5 ms. Balanced and Pro are derived at the calibration
route share. All devices and the method: `benchmarks/HARDWARE-summary.md`. Rows marked † there are on the
published scale of the 24 Sep release candidate (a same-session ratio to the release candidate's file of the
same architecture).

### 7.6 RotBench (30 Sep 2026)

4-way accuracy per counter-clockwise rotation, official protocol (`results/rotbench/RESULTS.md`):

| Model | Small: 0° | 90° | 180° | 270° | mean | ≥150° | Large: mean | ≥150° |
|---|---|---|---|---|---|---|---|---|
| Humans (published, Small only) | 0.99 | 0.99 | 0.99 | 0.97 | — | — | — | — |
| GPT-5 (published) | 1.00 | 0.41 | 0.81 | 0.59 | — | — | — | — |
| Gemini 2.5 Pro (published) | 1.00 | 0.50 | 0.72 | 0.40 | — | — | — | — |
| o3 (published) | 1.00 | 0.45 | 0.70 | 0.48 | — | — | — | — |
| W | 0.90 | 0.92 | 0.88 | 0.88 | 0.90 | 6 | 0.97 | 6 |
| Deep-OAD | 0.94 | 0.82 | 0.86 | 0.78 | 0.85 | 7 | 0.93 | 12 |
| GeoCalib | 1.00 | 0.00 | 0.00 | 0.00 | 0.25 | 46 | 0.25 | 293 |
| Pico | 0.94 | 0.92 | 0.90 | 0.92 | 0.92 | 5 | 0.96 | 15 |
| Nano | 0.96 | 0.98 | 0.96 | 0.94 | 0.96 | 0 | 0.98 | 2 |
| Fast | 0.98 | 0.98 | 0.96 | 0.98 | 0.97 | 0 | 0.99 | 0 |
| Balanced | 1.00 | 0.98 | 0.98 | 0.98 | 0.98 | 0 | 1.00 | 0 |
| Pro | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 | 0 |
| Max | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 1.00 | 0 |

Max and Pro are correct on every RotBench image in the 4-way sense (within 10°: 100.0% on Small; 98.9% and 98.8% on
Large). On RotBench-Small this is above the published human baseline (0.99 / 0.99 / 0.99 / 0.97), by 1–3 points on 50 photos. The
superseded 24 Sep Max scored 1.00 / 1.00 / 0.98 / 1.00 on Small.

## 8. Pico: the sixth tier and its rotation-corner fix

Pico was added after the freeze (owner decision, 29 Sep) and selected on development data only; it is **not** part
of the sealed holdout. Its record is `research/pico/PREREG.md`.

**Pre-registered plan (29 Sep 11:31 AEST).** A truncated DINOv2-S (first k of 12 blocks) at 112 or 84 px, distilled
from M3-SOUP8 with the full grid-fix recipe. Gates: G1 grid-fix C1–C4; G2 C1 on every exported file; G3 INT8 file
≤ 15 MB; G4 CPU latency ≤ 0.60 × Nano INT8; G5 error rate ≤ 1.5 × Nano's on the camera-like development set (≥ 97.53%)
and on cue-free photos (≥ 61.3%). Rule: among candidates passing G1–G5, the highest camera-like development score.

**Deviations** (each recorded before the result it affects):

1. 11:42: the first launch was stopped at ~1,200 steps for a loader memory leak; restarted from step 0 with the same
   configuration after a fix.
2. 12:50: the 5,000-step milestones showed the 6-block students losing quarter- and half-turn decisions (the later
   blocks carry "which side is up"). Exploratory arms with spread block subsets were added at 10,000 steps.
3. 14:00: a teacher-assistant arm (KD from the 12-block Fast) was added at 20,000 steps.
4. 15:00: zero-training measurements showed the full 12-block Nano at 70 px was far better than any 6-block student at
   similar speed (resolution is cheap, depth is expensive for this task). New arm **N70**: the full Nano fine-tuned at
   56–70 px. It fails G3 by construction (INT8 24.1 MB) and is not a truncated net, so it could only be reported as the
   owner's alternative.
5. 15:30: two exploratory arms stopped at 30,000 steps (within 0.4 pp of each other).
6. 16:25, **bug found**: on resume the trainer took each student's base learning rate from the saved optimiser state,
   compounding the cosine schedule at every resume (effective factors down to ×0.22). All numbers for those arms are
   for the effective schedule; the fix kept each resumed student on its current trajectory. At the same time **N70L**
   (as N70 with learning rate ×0.2) was added.

**Outcome (21:10).** No candidate passed all gates. The truncated candidates passed G1–G4 (6-block INT8 13.2 MB) but
failed G5 (best 89.91% vs the 97.53% floor); the 6-block candidate lost ~24 pp on indoor scenes (DIODE 64.9% vs
88.6%). N70L at 70 px passed G1, G2 and G4, failed G3 (24.1 MB) and missed G5 by 36 views (96.62%). **The owner chose
N70L and waived the size gate and the "truncated network" definition** (29 Sep ~22:25).

**Corner fix (30 Sep).** The package tests then showed that N70L misread images turned by software with a growing
canvas (flat black or white corners) by ~180° with confidence 0.49–0.81: every training view had been a crop inside
the rotated frame. A fine-tune (N70C: 8,000 steps, learning rate 2e-5, 20% of views turned whole with flat-colour
corners) was pre-registered at 01:22 AEST with criteria K1 (corner panel: ≥150° errors ≤ 3% and within-10 ≥ Nano − 10
pp), K2 (package tests), K3 (no development regression > 1.0 pp vs N70L) and K4 (C1 everywhere). Deviations:

1. 01:36: the COCO originals for the planned corner panel were not available; the panel was built from development
   views instead (P1).
2. 01:48, after the P1 baselines and before any N70C result: P1 turns already-rotated crops again, which is hard for
   every tier (Nano 50.6% within 10°), so a panel P2 was added (upright crop turned once with flat corners); K1 applies
   to P2.
3. N70C **failed K1 as registered** (P2 ≥150° errors 5.19% / 11.48%), a bound that **Nano misses too** (5.26% /
   8.89%), and **failed K3** by 0.65 pp on one set (DIODE camera-like −1.65 pp, −18 / 1,092).
4. Round 2 (owner: "Can you improve it?"), registered 12:19: a paired corner criterion **K1′** (corner-induced flips,
   i.e. ≥150° on P2 but ≤10° on the same item's quarter-turn control, ≤ Nano's + 1.0 pp) replaced the absolute bound as
   the evaluator; the original K1 stays reported as failed. Candidates: weight interpolations θ = (1 − a) N70L + a N70C
   (a recombination, labelled as such), and corner repainting at inference.
5. S70 (a = 0.7) passed K1′ and K3 but failed one of 30 package tests; a dose extension a ∈ {0.8, 0.9} was registered
   before its result. **S80** passed K1′ (3.25% / 3.77%; Nano 3.56% / 3.83%; N70L 6.02% / 5.44%), K2 (30 / 30 package
   tests with all eight files), K3 (DIODE camera-like −0.82 pp, McNemar p 0.21; five-seed benchmark 92.23% vs 92.31%;
   every other development set within ±0.5 pp except the two 126-view R-LiViT development sets, −2 and −1 views) and K4
   (worst C1 +0.045, TensorRT FP16). S90 failed K3.
6. Corner repainting helped every tier on P2 (+2.8 to +5.0 pp) but also triggered on 0.1–1% of development images
   (photos with their own frames), which would change every reported number. **It is not shipped in 1.0.**

The owner replaced N70L with **PICO-N70S80** (30 Sep ~14:00). Its ONNX graphs are structurally identical to N70L's in
all eight formats, so the speed rows carry over (re-measured on Apple M4 Core ML: 0.86 ms). On Core ML's Neural Engine
18 of 1,976 development views change by more than 2° against FP32, all two-peaked with confidence ≤ 0.46; the net
effect is −3 correct answers.

## 9. File formats, runtime parity and per-format thresholds

Each model ships as ONNX FP32 / FP16 / INT8 (FP16 files keep float32 inputs and outputs), Core ML FP16 packages at
batch 1 and 16, and for Pico, Nano and Fast also batch-capable ONNX files; Pico also has full-token FP32/INT8 files for
ONNX Runtime Web. INT8 is ONNX Runtime dynamic quantisation.

Each file was run on the same calibration views in its own runtime and the release calibration fitted its thresholds
unchanged (standard / strict; route):

| Tier | ONNX FP32 | ONNX FP16 | ONNX INT8 | batch FP16 | batch INT8 | Core ML FP16 |
|---|---|---|---|---|---|---|
| Pico | 0.426 / 0.833 | 0.425 / 0.833 | 0.415 / 0.833 | 0.425 / 0.833 | 0.423 / 0.853 | 0.423 / 0.838 |
| Nano | 0.535 / 0.800 | 0.535 / 0.800 | 0.536 / 0.800 | 0.535 / 0.800 | 0.531 / 0.782 | 0.534 / 0.800 |
| Fast | 0.629 / 0.738 | 0.629 / 0.738 | 0.626 / 0.738 | 0.629 / 0.738 | 0.623 / 0.738 | 0.628 / 0.738 |
| Balanced | 0.691 / 0.743; 0.629 | 0.691 / 0.745; 0.629 | 0.685 / 0.745; 0.626 | 0.691 / 0.743; 0.629 | 0.684 / 0.745; 0.623 | 0.692 / 0.743; 0.628 |
| Pro | 0.729 / 0.729; 0.800 | 0.729 / 0.729; 0.800 | 0.714 / 0.714; 0.797 | 0.729 / 0.729; 0.800 | 0.714 / 0.714; 0.793 | 0.729 / 0.729; 0.800 |
| Max | 0.727 / 0.727 | 0.727 / 0.727 | 0.711 / 0.711 | 0.727 / 0.727 | 0.711 / 0.711 | 0.727 / 0.727 |

Parity on the Open Images calibration set (6,692 views, within 10°): FP16 = FP32 within ±8 views for every model;
INT8 is 0.2–0.7 pp lower and slightly less confident; Core ML = FP32 within 5 views; TensorRT FP16 and OpenVINO give
the same answers as ONNX Runtime FP32 (median difference ≤ 0.001°). Batch-capable files equal the one-image files at
FP32. The sealed holdout was scored with the PyTorch checkpoints and the FP32 thresholds; A9 with the ONNX FP32 files.

## 10. Limitations

- **Thermal.** Unseen thermal cameras vary widely (sealed: R-LiViT thermal Max 95.9%, Aalborg Max 83.3%, Nano 48.7%)
  and abstention is not reliable there.
- **Small tiers on clean photos.** Nano and Fast trail W on clean COCO photos; Nano also on clean Open Images photos.
- **Pico** has development numbers only and is weak on strongly rolled CCTV (41.1% on held-out camera G329).
- **Rotation corners.** A sky-blue corner fill can flip Nano by 180° (it reads as sky); black, white and grey corners
  pass the package tests for every tier; corners still cost every tier 2–5 pp on the development corner panel.
- **Second use.** The A9 sets and RotBench were used once before (for the 24 Sep candidate); the sealed holdout is
  the only once-only evaluation of the release.
- **Simulated degradation.** "Degraded" views are simulated, not recordings of degraded cameras. The cue-free photos
  simulate camera-like rendering.
- **Small samples.** R-LiViT panels have 74 images; Aalborg is one camera.
- **Routing.** Balanced and Pro route 10–33% / 30–80% on the sealed sets, more than on the calibration mix.
- **C1 statistic.** Argmax ρ₄ is decided by tie-breaks on near-flat outputs; we report soft ρ₄ alongside (§4.5).
- **Data.** Flickr-heavy photos, one US CCTV dataset with ten training cameras; licence re-checks are snapshots at fetch
  time; "no known copyright restrictions" is a statement, not a licence.
- **Not measured.** Nadir aerial imagery, IMU replacement, Raspberry Pi speed, physically rolled cameras beyond the
  panels above.

## 11. Reproducibility

`RECIPE.md` gives the environment pins, data builds, the full training lineage with commands, soups and exports.
`rotlab/` contains the training, export and evaluation code (provenance and hashes in `rotlab/CODE-SOURCES.txt`);
`evaluation/code-final/rotlab/` holds the 14 files whose SHA-256 are bound in `FROZEN.json`, byte for byte.
Pre-registrations, results, prediction scripts and scorers of the A9 re-score and of RotBench: `results/heldout-rescore/`
and `results/rotbench/` (the per-view prediction files are kept with the release records). Sealed-holdout
aggregates, paired intervals and the freeze record: `results/sealed-holdout/`. Every released file's SHA-256: `SHA256SUMS` on Hugging Face and
`manifests/RELEASE-ARTIFACTS-SHA256.txt`.

## 12. Sources

Numbers in this report come from these records. Paths are relative to the repository unless marked *campaign*, i.e.
the internal records of the 27–30 Sep improvement campaign (lead workspace), which are not published:

| Section | Source |
|---|---|
| §1, §7.1, §7.3, §7.4, §9 | `RESULTS.md` (the release copy of the internal single source of truth, FINAL-NUMBERS), from `results/sealed-holdout/FINAL-RESULTS.json` and `FINAL-PAIRED.json` |
| §2 tiers, §6.1, §6.2, §8 outcome, §9 thresholds | `FREEZE.md` (the release freeze record, per-format and Pico addenda); `results/sealed-holdout/{FROZEN.json, SPEC.json, thresholds-per-format.json, thresholds-pico-s80.json}` |
| §2 lineage, §3 | `DATA-CARD.md` (derived from the checkpoint configs, the pack receipts and the trainer's per-family log) |
| §4.1 | *campaign* trace audit report (`research/trace-audit/REPORT.md`) and its release notes |
| §4.2–§4.4 | *campaign* grid-fix pre-registration and deviation log (`research/grid-fix/PREREG.md`) |
| §4.3 | *campaign* grid-fix data table and pack receipts (`research/grid-fix/FINAL-DATA-TABLE.txt`, `data-lane2/receipts/SUMMARY.json`, `pixel-verify.json`) |
| §4.5 | *campaign* results log (grid-fix sections, 28–29 Sep) |
| §5 | `benchmarks/woehrer-2026/BENCHMARK-NOTES.md` and *campaign* claim rows; released tiers from `RESULTS.md` §2 |
| §6.2 timings | *campaign* status log (29 Sep 18:40–19:25) |
| §6.3, §7.2 | `results/heldout-rescore/PREREG.md`, `results/heldout-rescore/RESULTS.md` |
| §6.5, §7.6 | `results/rotbench/PREREG.md`, `results/rotbench/RESULTS.md` |
| §7.5 | `benchmarks/HARDWARE-summary.md` |
| §8 | *campaign* Pico pre-registration and deviation log (`research/pico/PREREG.md`); `FREEZE.md` Pico addenda |
