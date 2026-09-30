# Reproduction recipe (RightWayUp 1.0)

Everything needed to rebuild the released models from original sources. Code lives in `rotlab/`. Commands run from the repo
root with `PYTHONPATH=.`; data root from `ROTLAB_DATA` (default `/workspace/rotation-data`, `rotlab/core.py`).

The released weights, not a retrain, are the reference; their hashes are in `FREEZE.md`. The released models descend from
the 24 Sep 2026 release candidate through the 27–29 Sep improvement campaign; this file gives the final stages in
full and the earlier stages by name (§4). Method and evidence: `TECHNICAL-REPORT.md`.

## 1. Environment

| Component | Version |
|---|---|
| Training image | `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` (PyTorch 2.8.0, Python 3.11, CUDA 12.8) |
| Pinned Python packages (every pod) | numpy 2.1.2, Pillow 11.0.0, timm 1.0.28, onnx 1.19.0, onnxruntime 1.24.4, scipy |
| Training GPUs | single-GPU RunPod pods (NVIDIA RTX PRO 4500 Blackwell 32 GB and similar); ViT-L at canvas 280 with gradient checkpointing peaks at about 15.4 GB |
| Runtime checks | TensorRT 10.13 and OpenVINO 2026.4 (`rt_check`), ONNX Runtime CUDA for FP16, coremltools 9.0 on an Apple M4 Mac mini for Core ML |
| 24 Sep candidate (ancestor stages) | Python 3.12.3, PyTorch 2.8.0+cu128, same packages |

Backbones (Apache-2.0, Meta DINOv2 without registers, from the Hugging Face Hub via timm):

| Arch | timm id | Revision | Weight SHA-256 |
|---|---|---|---|
| ViT-S/14 | `vit_small_patch14_dinov2.lvd142m` | `4610ca143709d58a633b6397a74412c2c3842454` | `04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081` |
| ViT-L/14 | `vit_large_patch14_dinov2.lvd142m` | `4741e1cafbf45415e77074bb0cb42dba76c8684a` | `0424a5d1b515278cba3c6640ccbeaacc41de59d3a93df0dd5e494285eea2b355` |

## 2. Model and training objective

- `rotlab/model.py` `RotNet`: DINOv2 backbone → concat(CLS, mean patch token) → LayerNorm → Linear(·,1024) → GELU →
  Linear(1024,360). Small models are trained and exported **fill-drop** (`rotlab/filldrop.py`, `rotlab/focus.py`):
  letterbox-fill patch tokens (tolerance 1e-3 on the normalised fill colour) are dropped before the transformer and
  left out of the pool.
- Loss: cross-entropy against a circular Gaussian soft target, σ = 6°; knowledge distillation (α 0.5, T 1) from a
  teacher that sees the **clean, cue-free** view of the same sample.
- Each view gets a fresh uniform random angle in [0, 360); target = base roll + applied angle (clockwise). Crops are
  angle-independent; **no max-area crop** (`--maxarea-p 0`).
- **Scrub renderer** (`rotlab/scrub.py`, `--scrub key=value,…`), used by every stage of the release:

| Option | Value | Meaning |
|---|---|---|
| `ss` | 1 | supersampled rotation (render at 2×, antialiased downsample) |
| `decoy` | 0.6 | probability of a decoy: a conflicting JPEG grid or pure resampling traces at a random angle on real content |
| `dscale`, `cam` | 1, 0.5 | decoy scale; camera-like degradation share |
| `probe` | 0.10 | share of content-free probe images (noise, flat grey) with a uniform target and no KD |
| `dual` | 1 | the teacher sees the clean view |
| `down` | 1.5 | every rendered view (and the teacher view) downscaled 1.5× with Lanczos before letterboxing: removes the rotated spectral support of a ~1:1 render |
| `cuekd` | 0.5 (small models) | with this probability the student sees the cue-bearing 1:1 view while the teacher sees the cue-free view; KD only |
| `leanpen` | 50 (small), 20 (Max stage B′) | loss + L · mean(soft ρ₄²) on probes carrying their cue angle |
| `corner` | 0.2 (Pico corner fine-tune only) | the whole frame turned with flat-colour corners (black 45%, white 20%, grey 10%, random 25%) |

- Degradations (`core.degrade`) with p = 0.65: resolution loss, blur, greyscale/IR, exposure/contrast/gamma, noise,
  JPEG q25–90. `--qat` (`rotlab/qat.py`) adds exact ONNX Runtime dynamic-INT8 fake quantisation to the 98 quantised
  Linear layers (straight-through); checkpoints stay float.
- Optimiser: AdamW, weight decay 0.05, layer-wise LR decay 0.8, head LR 5e-4 (1e-4 for Pico), cosine with linear
  warmup, bf16 autocast, EMA 0.9995 (EMA weights are released), drop-path 0.1, batch 128.
- Decode: argmax + local circular mean over ±10 bins. Confidence: posterior mass within ±10° of the prediction.

## 3. Data

### 3.1 Grid-free packs of the release (23 families, 1,251,672 rows)

Every pack is `DATA/rotlab/sources/<family>.{bin,jsonl}` with **lossless 448-px PNG** parents (Lanczos, EXIF applied,
never JPEG-encoded). Counts and licences: `DATA-CARD.md`.

| Step | Families | How |
|---|---|---|
| Lane 1: archive originals | `gf_diode`, `gf_diode2`, `gf_meva`, `gf_meva2`, `gf_poly_haven`, `gf_poly_direct`, `gf_oi_train2` | Lossless originals streamed from the V1 hybrid archive (train split only), MEVA public bucket (frames re-extracted with ffmpeg 7.0.2), Open Images originals; row metadata copied verbatim; dHash identity check against the old rows |
| Lane 2: re-fetch + replace | `gf_pass`, `gf_pass2`, `gf_coco`, `gf_coco2`, `gf_coco_by`, `gf_coco_pd`, `gf_oi`, `gf_fresh_coco`, `gf_fresh_oi` | Flickr originals re-fetched at ≥ 1,020 px (PASS via `pass_metadata.csv` → YFCC100M → Flickr); label check dHash ≤ 8 bits vs the old row; lost images replaced from the same source/split/licence class (current oEmbed licence, eval ≤ 8 / packs ≤ 6 / within-new ≤ 6 bits dHash, EXIF absent or 1) |
| Expansion: Open Images V7 train | `gf_oi7a`, `gf_oi7b`, `gf_oi7c` (264,513) | CVDF mirror (1,024 px), `Rotation = 0`, current Flickr oEmbed licence per image, author cap 20, eval ≤ 8 / packs ≤ 6 bits dHash |
| Expansion: CommonCatalog CC-BY | `gf_cc1` … `gf_cc4` (276,016) | `common-canvas/commoncatalog-cc-by@80f50fe4`, short side ≥ 1,024 px; current Flickr licence exactly CC BY 2.0; EXIF + current-thumbnail orientation checks; NSFW filter (p ≥ 0.2 rejected); author cap 60 |

**Exclusions.** `sources/exclude.json` lists every image within dHash ≤ 6 bits of the 1,030 origin-benchmark photos;
`train_ddp.blobs_for` drops them at load (6,068 rows of the 23 packs). Test and calibration cameras and scenes are
excluded by allowlist in the builders.

**Mixes** (sampling weights per step):

- **GFMIX** (16 families; Nano's final stage and the arm-A stages): `gf_pass=0.10, gf_pass2=0.07, gf_coco=0.05,
  gf_coco2=0.08, gf_oi=0.15, gf_diode=0.09, gf_diode2=0.04, gf_meva=0.09, gf_meva2=0.04, gf_poly_haven=0.05,
  gf_poly_direct=0.02, gf_fresh_coco=0.07, gf_fresh_oi=0.03, gf_oi_train2=0.10, gf_coco_pd=0.01, gf_coco_by=0.01`.
- **MIX v3** (23 families; Max, Fast, Pico): GFMIX × 0.60 plus the new families 0.40 in proportion to their rows
  (`rotlab/mixv3.py SOURCES_DIR`), which with all packs present gives `gf_cc1=0.0740, gf_cc2=0.0740, gf_cc3=0.0383,
  gf_cc4=0.0179, gf_oi7a=0.0740, gf_oi7b=0.0740, gf_oi7c=0.0477`.

### 3.2 Manifests and rebuilding

**The release's packs.** `manifests/training/` lists every row of the 23 grid-free packs (one JSON row per image:
source, licence, attribution, public locator, `stored_sha256` of the stored PNG, `derivation` and `replaces` for
replacements). The per-row files are in the Hugging Face model repository under `manifests/training/`; the folder's
`verify.py` recomputes `SUMMARY.json` and checks the hashes and totals with the Python standard library. The builder
is `rotlab/export_manifest_release.py`. Rebuilding the grid-free packs means fetching each original from its locator
(`fetched_url` where recorded: the file the stored image was made from) and storing it as a lossless 448-px PNG (§3.1);
`stored_sha256` tells you whether a rebuilt image is byte-identical. The 3,586 render rows need the ORTUS AI renders.

**The release candidate's packs.** The ancestor stages used the JPEG-stored 24 Sep packs
(`manifests/release-candidate-2026-09-24/training/*.jsonl.gz`, 673,690 rows) and the 27 Sep campaign packs.
`rotlab.build_from_manifest` fetches each original from its public locator and re-encodes it in that format:

```bash
export ROTLAB_DATA=/data/rotlab-rebuild
M=manifests/release-candidate-2026-09-24/training
python -m rotlab.build_from_manifest $M/coco.jsonl.gz
python -m rotlab.build_from_manifest $M/pass.jsonl.gz --pass-dir /data/PASS    # PASS.0-3.tar, Zenodo 6615455
python -m rotlab.build_from_manifest $M/hybrid.jsonl.gz --diode-dir /data/diode/train --renders-dir /data/ortus-renders
```

Flickr photos deleted after our check cannot be refetched; the tool reports them.

## 4. Training lineage

```
24 Sep candidate            campaign 27–28 Sep                    grid fix 28 Sep                  release (28–30 Sep)
ViT-L: soup-G7-G3 ── LL1 ── LLX224/LL1F ── FINAL-a..d ── SOUPF-AB8 ── GF-A1..A4 ── GF-SOUP-A ── F2b ── M3-s1r/s2r/s3/s4 (A′@140, B′@280) ── M3-SOUP8   (Max)
ViT-S: S2 ── SS1 ── SS2 ── SK1 ── SK2 ── SKG-s/SKG-s2 ── SOUPS-SKG-s2 ── GF-S1/S2 ── GF-SOUP-S ── GF-SX-dn15p10 ── GF-SV2-A1/A2 ── GF-SOUP-SV2 (Nano)
                                                                                                   GF-SOUP-SV2 ── GF-SV3-kd ── GF-SV3-kdM3                    (Fast)
                                                                                                   GF-SOUP-SV2 ── PICO-N70L ─┬─ 0.2 : 0.8 ── PICO-N70S80       (Pico)
                                                                                                                 PICO-N70L ── PICO-N70C ┘
```

The 24 Sep release candidate's checkpoints are recorded in `FREEZE-release-candidate-2026-09-24.md`; the 27–28 Sep campaign stages (large-model continuations at
canvas 140, distillation of the small model from the large one, soups) are documented in `TECHNICAL-REPORT.md` and the
campaign log. The stages below produced the released weights (`--img-size 224` for every run; seeds as listed).

| Run | Init | Canvas | Mix | Steps × batch | LR | Other |
|---|---|---|---|---|---|---|
| GF-A1…A4 (arm A) | SOUPF-AB8 | stage A 140, stage B 224 | GFMIX | 8,000 + 2,700 × 128 | 2e-5 / 1e-5 | scrub `ss=1,decoy=0.6,dscale=1,cam=0.5,probe=0.10,dual=1`; KD from SOUPF-AB8 (clean view); seeds 3101–3401 |
| GF-SOUP-A | 8 checkpoints of GF-A1…A4 | — | — | — | — | uniform EMA soup |
| GF-SOUP-A-F2b | GF-SOUP-A | 224 | GFMIX | 2,000 | 5e-6 | grad-ckpt; `down=1.5`; teacher SOUPF-AB8; seed 8201 |
| M3-s{1r,2r,3,4}-a140 | GF-SOUP-A-F2b | 140 | MIX v3 | 8,000 | 2e-5 | scrub + `down=1.5`; KD 0.5 from GF-SOUP-A-F2b (clean view); s1r resumed at 4,000 with LR 1e-5 |
| M3-s…-b280 | the matching a140 | 280 | MIX v3 (recomputed at start) | 1,500 | 1e-5 | `--qat`, `leanpen=20`, grad-ckpt |
| **M3-SOUP8** (Max) | the 8 M3 checkpoints | — | — | — | — | uniform EMA soup (`rotlab/soupn.py`) |
| GF-SV2-A1 / A2 | GF-SX-dn15p10 | multires 112–224 | GFMIX | 12,000 | 5e-5 | fill-drop; `down=1.5`; teacher GF-SOUP-A; seeds 9101 / 9201 |
| **GF-SOUP-SV2** (Nano) | GF-SV2-A1 + A2 | — | — | — | — | uniform EMA soup |
| GF-SV3-kd | GF-SOUP-SV2 | multires 112–224 | GFMIX | 12,000 | — | fill-drop; `down=1.5,cuekd=0.5`; teacher GF-SOUP-A-F2b |
| **GF-SV3-kdM3** (Fast) | GF-SV3-kd | multires 112–224 | MIX v3 | 4,000 | 2e-5 (head 5e-4) | fill-drop; `down=1.5,cuekd=0.5,leanpen=50`; teacher M3-SOUP8; seed 9951 |
| PICO-N70L | GF-SOUP-SV2 (all 12 blocks) | multires 56, 70, 70 | MIX v3 | 25,000 (global steps 35k–60k) | 2e-5 effective (×0.2 of 1e-4) | `rotlab.pico` shared-stream trainer; `down=1.5,cuekd=0.5,leanpen=50`; teacher M3-SOUP8 at canvas 224; seed 7101 |
| PICO-N70C | PICO-N70L | multires 56, 70, 70 | MIX v3 | 8,000 | 2e-5 (head 1e-4) | as N70L + `corner=0.2`; warmup 300; seed 7301 |
| **PICO-N70S80** (Pico) | 0.2 × N70L + 0.8 × N70C | — | — | — | — | WiSE-FT weight interpolation (`rotlab/soup.py`) |

Commands (from the job scripts; `R=rotation-data/rotlab/runs`):

```bash
# One M3 seed of the final Max (stage A' then B'); MIXB is recomputed from the packs present when stage B' starts
python -m rotlab.train_ddp --run M3-$TAG-a140 --arch l --img-size 224 --canvas 140 --init $R/GF-SOUP-A-F2b/final.pt \
  --teacher $R/GF-SOUP-A-F2b/final.pt --distill-alpha 0.5 --scrub ss=1,decoy=0.6,dscale=1,cam=0.5,probe=0.10,dual=1,down=1.5 \
  --mix "$MIX" --steps 8000 --lr 2e-5 --warmup 300 --seed $SEED --save-every 4000 --keep-milestones --workers 24
python -m rotlab.train_ddp --run M3-$TAG-b280 --arch l --img-size 224 --canvas 280 --init $R/M3-$TAG-a140/final.pt \
  --teacher $R/GF-SOUP-A-F2b/final.pt --distill-alpha 0.5 --qat \
  --scrub ss=1,decoy=0.6,dscale=1,cam=0.5,probe=0.10,dual=1,down=1.5,leanpen=20 --mix "$(python3 rotlab/mixv3.py $DATA/rotlab/sources)" \
  --steps 1500 --lr 1e-5 --warmup 200 --grad-ckpt --seed $((SEED+1)) --save-every 1500 --keep-milestones --workers 24
python rotlab/soupn.py $R/M3-SOUP8 $R/M3-s{1r,2r,3,4}-{a140,b280}/final.pt

# Fast: re-distil the cue-KD small model from the Max
python -m rotlab.filldrop train --run GF-SV3-kdM3 --arch s --img-size 224 --multires 112,140,168,196,224 \
  --init $R/GF-SV3-kd/final.pt --teacher $R/M3-SOUP8/final.pt --distill-alpha 0.5 --distill-temp 1.0 \
  --scrub ss=1,decoy=0.6,dscale=1,cam=0.5,probe=0.10,dual=1,down=1.5,cuekd=0.5,leanpen=50 \
  --mix "$MIXV3" --steps 4000 --lr 2e-5 --warmup 300 --seed 9951 --save-every 5000 --keep-milestones --workers 16

# Pico corner fine-tune and the weight interpolation
python -m rotlab.pico train --runs PICO-N70C --archs s --init $R/PICO-N70L/final.pt --teacher $R/M3-SOUP8/final.pt \
  --teacher-canvas 224 --distill-alpha 0.5 --distill-temp 1.0 --img-size 224 --multires 56,70,70 \
  --scrub ss=1,decoy=0.6,dscale=1,cam=0.5,probe=0.10,dual=1,down=1.5,cuekd=0.5,leanpen=50,corner=0.2 \
  --mix "$MIXV3" --steps 8000 --lr 2e-5 --head-lr 1e-4 --warmup 300 --seed 7301 --save-every 2000 --keep-milestones --workers 12
python -m rotlab.soup $R/PICO-N70L/final.pt $R/PICO-N70C/final.pt 0.8 $R/PICO-N70S80/final.pt   # theta = 0.2 A + 0.8 B
```

`rotlab.soup A.pt B.pt a OUT.pt` interpolates every floating tensor of `model` and `ema`: θ = (1 − a)·A + a·B, and
records both input hashes in the output config. `rotlab/soupn.py OUT_DIR CK…` is the uniform N-way EMA soup used for
M3-SOUP8, GF-SOUP-A and GF-SOUP-SV2. Every checkpoint's `config` (and for trained runs `args`) records its init,
teacher, mix and scrub string.

**Determinism.** `ROTLAB_LEGACY_TIME_SEED` is unset in every job of the release, so the data order depends only on the seed and
the number of loader workers. GPU kernels (bf16, TF32) are not bit-deterministic across hardware; several runs were
resumed from milestones after out-of-memory or storage-quota interruptions with the cosine learning rate at that step
and without the optimiser state (`TECHNICAL-REPORT.md` §4.4, §8). Expect statistically equivalent, not identical,
weights.

## 5. Export

```bash
# Max: full-token, canvas 280 (always pass --canvas; large checkpoints carry canvas 140 in their config)
python -m rotlab.export $R/M3-SOUP8/final.pt out/max-l280 --ema --canvas 280            # FP32, FP16 (on CUDA), INT8
# Small models: fill-drop one-image files, FP32 + INT8
python -m rotlab.filldrop export $R/GF-SOUP-SV2/final.pt --out out/nano-s112-fp32.onnx --canvas 112 --int8
python -m rotlab.filldrop export $R/GF-SV3-kdM3/final.pt --out out/fast-s224-fp32.onnx --canvas 224 --int8
python -m rotlab.filldrop export $R/PICO-N70S80/final.pt --out out/pico-s70-fp32.onnx --canvas 70 --int8
# FP16 fill-drop: fill mask on the float32 input, blocks and head in FP16, softmax in float32 (exported on CUDA)
python -m rotlab.fp16_fill CKPT --canvas 112 --out out/nano-s112-fp16.onnx
# Batch-capable files: fill keys masked additively (-1e4) in every block; FP32, INT8 and FP16
python -m rotlab.fillmask_export CKPT --canvas 112 --out out/nano-s112-batch
python -m rotlab.fillmask_export --parity out/nano-s112-batch-fp32.onnx out/nano-s112-fp32.onnx 112 fair,meva,fresh_diode
# Pico browser build: full-token FP32 + INT8
python -m rotlab.model_x export $R/PICO-N70S80/final.pt out/pico-s70-web --canvas 70 --int8
```

Every file has input `image` and a softmax output `prob` (360 bins; opset 18 for `rotlab.export`); FP16 files keep
float32 inputs and outputs. INT8: `onnxruntime.quantization.quantize_dynamic`, QInt8 weights, per-channel, reduce_range, MatMul/Gemm
only. Core ML: `rotlab.coreml_export` on an Apple silicon Mac (batch 1 and 16, FP16; the small models use the
additive fill mask, since a boolean mask mis-converts). Public file names: `<tier>-s<canvas>-[batch-]<precision>.onnx`
and `coreml/<tier>-s<canvas>-b{1,16}.mlpackage` (Max: `max-l280`).

Checks on every file: parity with PyTorch (`rotlab.onnx_parity`), the grid check C1 (`python -m rotlab.onnx_probe
FILE CANVAS`; GPU and TensorRT/OpenVINO variants for FP16), and per-format calibration.

## 6. Evaluation and thresholds

| Command | What |
|---|---|
| `python -m rotlab.calib build CANVAS` | calibration views (thresholds only) |
| `python rotlab/cal_onnx.py FILE.onnx CANVAS NAME cpu\|cuda` | calibration predictions of one shipped file in its runtime |
| `python rotlab/fmt_thresholds.py fp32 int8 fp16 m-int8 m-fp16 --json OUT` | per-format thresholds with the release `Tier.calibrate` |
| `python -m rotlab.camp_eval CKPT --canvases C --sets …` / `python -m rotlab.focus eval CKPT --mode fill …` | development and holdout scoring (full-token / fill-drop) |
| `python -m rotlab.camp_final freeze SPEC.json` → `open` → `check_score` → `aggregate_tiers` | the sealed holdout: freeze record, one opening, gated scoring, aggregation |
| `python -m rotlab.camp_incumbent …` | Woehrer 2026 on identical views (ONNX FP32) |
| `python -m rotlab.onnx_probe FILE CANVAS` | grid check C1 (argmax and soft ρ₄) |
| `results/heldout-rescore/score_a9.py` | the A9 re-score of the 24 Sep held-out sets |

Test and calibration set definitions: `BENCHMARKS.md` and the builders (`fair_photo.py`, `oi_test.py`, `meva_test.py`,
`calib.py`, `camp_newdata.py`, `camp_rlivit.py`, `camp_aalborg.py`), all with fixed seeds. The 14 code files bound by
the freeze record are listed with their SHA-256 in `results/sealed-holdout/FROZEN.json` (`code_sha256`) and copied
verbatim to `evaluation/code-final/rotlab/`.
