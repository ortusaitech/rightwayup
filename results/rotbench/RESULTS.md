# RotBench with the v2 release models (see PREREG.md)

> Terminology note (added 30 Sep 2026): in this record, 'v2' denotes the RightWayUp 1.0 release weights and 'v1' the 24 Sep release candidate; the record itself is unchanged.

**Scored once, 30 Sep 2026 ~15:25 AEST** (pre-registration committed at 15:20). Summary: **Max and Pro answer every
RotBench image correctly (1.00 at 0°, 90°, 180° and 270° on both RotBench-Small and RotBench-Large), with no upside-down
answers**; on RotBench-Small the published human baseline is 0.99 / 0.99 / 0.99 / 0.97 (50 photos: above the published
human baseline by 1–3 points, never "beats humans"). Woehrer 2026 0.90 / 0.92 / 0.88 / 0.88 (Small) and 0.97 mean (Large); Deep-OAD 0.85 and 0.93 mean.
The 24 Sep comparator rows are reproduced exactly from their stored predictions. RotBench images are square, so the
full-frame variant gives the same images and numbers as the official protocol.

Predictions `v2-rotbench-predictions.npz`; scorer `score_rotbench.py`. 4-way accuracy per counter-clockwise rotation (prediction
snapped to the nearest quarter turn), mean, within 10° and upside-down errors (≥150°); every image answered.

View order and angles equal the stored 24 Sep comparator files for both variants (checked).

## RotBench-Small (50 photos), official protocol (PIL rotate, no expand)

| Model | 0° | 90° | 180° | 270° | mean | within 10° | ≥150° |
|---|---|---|---|---|---|---|---|
| Humans (published) | 0.99 | 0.99 | 0.99 | 0.97 | — | — | — |
| GPT-5 (published) | 1.00 | 0.41 | 0.81 | 0.59 | — | — | — |
| Gemini 2.5 Pro (published) | 1.00 | 0.50 | 0.72 | 0.40 | — | — | — |
| o3 (published) | 1.00 | 0.45 | 0.70 | 0.48 | — | — | — |
| Woehrer 2026 (24 Sep predictions) | 0.90 | 0.92 | 0.88 | 0.88 | 0.90 | 89.5% | 6 |
| Deep-OAD (24 Sep predictions) | 0.94 | 0.82 | 0.86 | 0.78 | 0.85 | 71.0% | 7 |
| GeoCalib (24 Sep predictions) | 1.00 | 0.00 | 0.00 | 0.00 | 0.25 | 22.5% | 46 |
| v1 Max (24 Sep, reference) | 1.00 | 1.00 | 0.98 | 1.00 | — | — | — |
| Pico | 0.94 | 0.92 | 0.90 | 0.92 | 0.92 | 87.5% | 5 |
| Nano | 0.96 | 0.98 | 0.96 | 0.94 | 0.96 | 95.5% | 0 |
| Fast | 0.98 | 0.98 | 0.96 | 0.98 | 0.97 | 97.5% | 0 |
| Balanced | 1.00 | 0.98 | 0.98 | 0.98 | 0.98 | 98.5% | 0 |
| Pro | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 100.0% | 0 |
| **Max** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 100.0% | 0 |

## RotBench-Large (300 photos), official protocol (PIL rotate, no expand)

| Model | 0° | 90° | 180° | 270° | mean | within 10° | ≥150° |
|---|---|---|---|---|---|---|---|
| Woehrer 2026 (24 Sep predictions) | 0.96 | 0.97 | 0.97 | 0.97 | 0.97 | 96.8% | 6 |
| Deep-OAD (24 Sep predictions) | 0.98 | 0.93 | 0.90 | 0.92 | 0.93 | 81.2% | 12 |
| GeoCalib (24 Sep predictions) | 1.00 | 0.00 | 0.00 | 0.00 | 0.25 | 23.1% | 293 |
| v1 Max (24 Sep, reference) | 1.00 | 0.99 | 0.99 | 1.00 | — | — | — |
| Pico | 0.97 | 0.95 | 0.96 | 0.95 | 0.96 | 92.0% | 15 |
| Nano | 0.98 | 0.97 | 0.98 | 0.98 | 0.98 | 96.8% | 2 |
| Fast | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 98.6% | 0 |
| Balanced | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 99.1% | 0 |
| Pro | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 98.8% | 0 |
| **Max** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 98.9% | 0 |

## RotBench-Small (50 photos), full-frame variant (expand=True)

| Model | 0° | 90° | 180° | 270° | mean | within 10° | ≥150° |
|---|---|---|---|---|---|---|---|
| Woehrer 2026 (24 Sep predictions) | 0.90 | 0.92 | 0.88 | 0.88 | 0.90 | 89.5% | 6 |
| Deep-OAD (24 Sep predictions) | 0.94 | 0.82 | 0.86 | 0.78 | 0.85 | 71.0% | 7 |
| GeoCalib (24 Sep predictions) | 1.00 | 0.00 | 0.00 | 0.00 | 0.25 | 22.5% | 46 |
| Pico | 0.94 | 0.92 | 0.90 | 0.92 | 0.92 | 87.5% | 5 |
| Nano | 0.96 | 0.98 | 0.96 | 0.94 | 0.96 | 95.5% | 0 |
| Fast | 0.98 | 0.98 | 0.96 | 0.98 | 0.97 | 97.5% | 0 |
| Balanced | 1.00 | 0.98 | 0.98 | 0.98 | 0.98 | 98.5% | 0 |
| Pro | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 100.0% | 0 |
| **Max** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 100.0% | 0 |

## RotBench-Large (300 photos), full-frame variant (expand=True)

| Model | 0° | 90° | 180° | 270° | mean | within 10° | ≥150° |
|---|---|---|---|---|---|---|---|
| Woehrer 2026 (24 Sep predictions) | 0.96 | 0.97 | 0.97 | 0.97 | 0.97 | 96.8% | 6 |
| Deep-OAD (24 Sep predictions) | 0.98 | 0.93 | 0.90 | 0.92 | 0.93 | 81.2% | 12 |
| GeoCalib (24 Sep predictions) | 1.00 | 0.00 | 0.00 | 0.00 | 0.25 | 22.9% | 292 |
| Pico | 0.97 | 0.95 | 0.96 | 0.95 | 0.96 | 92.0% | 15 |
| Nano | 0.98 | 0.97 | 0.98 | 0.98 | 0.98 | 96.8% | 2 |
| Fast | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 98.6% | 0 |
| Balanced | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 99.1% | 0 |
| Pro | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 98.8% | 0 |
| **Max** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 98.9% | 0 |

