# Freeze record: release candidates

> Terminology note (added 30 Sep 2026): in this record, 'v2' denotes the RightWayUp 1.0 release weights and 'v1' the 24 Sep release candidate; the record itself is unchanged.

**Frozen 24 Sep 2026, 15:30 Sydney time (05:30 UTC), before the protected final evaluation was opened.**
No weights, tier settings or thresholds change after this point. If anything must change, the protected results below
are void and a new protected set is needed.

## Weights

| Role | Checkpoint (EMA weights) | SHA-256 |
|---|---|---|
| Small model: Nano (112 px), Fast (224 px), cascade stage 1 | `runs/S2-vits-multires/final.pt` | `de9cad7675f77c68efd6b91693484c4e9f75a41d579a87b5ec35910ee6229284` |
| Large model: cascade stage 2 (50/50 weight average of G7 and G3) | `runs/soup-G7-G3-a0.5/final.pt` | `ae53431c92acfd123ea5b5a08c1dcc4850e88772e1d1e5fbb0fd6886d42276f9` |

Released files are the exports of these two checkpoints: ONNX FP32/FP16/INT8 for `nano-s112`, `fast-s224` and
`large-l224`, plus Core ML (batch 1 and 16). Their 27 hashes are in `manifests/RELEASE-ARTIFACTS-SHA256.txt`, whose
own SHA-256 is `f65f0b203bacc6530aa800d8fe34b4beb0546ef70e8ca17406a421ff5b325d89`.

Backbones: DINOv2 ViT-S/14 `4610ca14…` and ViT-L/14 `4741e1ca…` (full revisions and hashes in `RECIPE.md`).

## Tiers and thresholds

These were fixed on the calibration sets only: MEVA G419/G299/G330, Poly Haven clean-v1 calibration split and Open
Images validation ranks 4000–7999, each weighted equally. Confidence is the probability mass within ±10° of the
prediction, with no test-time augmentation.

| Tier | Models | Input | Route to large model if confidence < | Abstain if confidence < (standard, 90% answered) | Abstain if confidence < (strict, ≤1% wrong) |
|---|---|---|---|---|---|
| Nano | small | 112 px | — | 0.485 | 0.785 |
| Fast | small | 224 px | — | 0.706 | 0.743 |
| Balanced | small → large | 224 px | 0.706 (~10%) | 0.778 | 0.750 |
| Max | small → large | 224 px | 0.836 (~20%) | 0.808 | 0.677 |

For cascade tiers, the abstain threshold applies to the confidence of whichever model answered.

**Addendum, 25 Sep 2026 (owner decision, before release):** the effective `strict` threshold is the higher of the
two columns, so `strict` never answers more than `standard`. Effective strict: Nano 0.785, Fast 0.743, Balanced
0.778, Max 0.808. The calibrated values above stay as frozen; no test data was used for this rule.

## Code

Evaluation code: `rotlab/` as committed in `ortusaitech/rotation-model-internal` at `97dfbe7` (tiers: `rotlab/final_stats.py`;
protected run: `rotlab/protected_eval.py`, committed at `4984de0` before it was run).

## Protected sets (opened once, after this record)

1. **MEVA test v1, frozen split:** whole cameras never trained on or scored (`rotlab.meva_test`, split `frozen`).
2. **Clean v1 frozen test:** 1,427 parents (DIODE scene 12, MEVA G339, Poly Haven), from `clean-v1-splits.json`
   `frozen_test`.

Each image gives a clean and a CCTV-degraded view at seeded angles, with the fixed 4:3 crop, as in the development
tests. Comparators on identical pixels: Woehrer 2026, Deep-OAD, GeoCalib (also scored within ±45°).

## Protected evaluation: done

Run once on 24 Sep 2026 (pod, 15:40–16:05 Sydney time), after this record and the evaluation code (`4984de0`) were
committed. The checkpoint hashes were verified, and the thresholds recomputed from calibration data matched the table
above to 3 decimals. Results: `results/protected-rc-2026-09-24/RESULTS.md`, with per-view predictions in
`results/protected-rc-2026-09-24/ours.npz` and `comparator-*.npz`. One scoring-stage correction (the comparator sign convention) is
documented at the top of the results; model outputs are as first produced.
