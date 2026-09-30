# A9: v2 release models on the 24 Sep held-out sets (second use; see PREREG.md)

> Terminology note (added 30 Sep 2026): in this record, 'v2' denotes the RightWayUp 1.0 release weights and 'v1' the 24 Sep release candidate; the record itself is unchanged.

**Scored once, 30 Sep 2026 ~14:30 AEST**, with the staged release files (one-image ONNX FP32, ONNX Runtime CUDA) and the
release package, after the pre-registration (`PREREG.md`, written 14:23). Predictions `v2-predictions.npz` (sha256
`85b8537e…96eb7`); comparators are the stored 24 Sep predictions on identical pixels (reproduced exactly below).

## Summary (held-out views only)

| Held-out set | Woehrer 2026 | Deep-OAD | v2 Max | Max − Woehrer (95% CI) | v1 Max (24 Sep) |
|---|---|---|---|---|---|
| MEVA, 4 unseen cameras G329 / G420 / G421 / G474 (380 views, incl. thermal G474) | 75.0% (47) | 82.4% (26) | **100.0% (0)** | +95 views [+77, +113] | 98.7%\* |
| of which thermal camera G474 (110 views) | 53.6% (47) | 67.3% (26) | **100.0% (0)** | | 95.5% (5) |
| Clean v1 test: DIODE scene 12, MEVA G339, Poly Haven (2,854 views) | 70.6% (98) | 82.8% (113) | **98.8% (5)** | +806 views [+754, +858] | 98.5% (3) |

Within 10°, every view answered; (≥150° errors). \*v1 Max on the same 4 cameras, from the 24 Sep per-camera rows
(100% / 100% / 100% / 95.5% on 90 / 90 / 90 / 110 views = 375 / 380). Max − Deep-OAD: +67 [+52, +82] (MEVA 4 cameras), +459 [+413, +506]
(clean test). With the standard threshold Max answers 99.7% of the 380 MEVA views, all correct, and 97.7% of the clean
test with 99.7% correct.

Disclosures: (1) second use: both sets were opened once on 24 Sep to score v1, and v2 was developed after those
results were known; v2 never trained on them and never scored them before today. (2) MEVA cameras G331 and G639 (other
clips) were in a v2 development set used for model choice, so the 6-camera figure (560 views: Max 100.0%) is reported
but labelled; the held-out claim uses the 4 other cameras. (3) Pico is weak on camera G329 (41.1%; a camera rolled ~36°
in some clips): Pico was chosen on development data and is the least accurate tier.

Predictions: `v2-predictions.npz`; scorer: `score_a9.py`. Within 10°, every view answered; (≥150° errors).

## Pipeline check: 24 Sep comparator numbers reproduced from the stored predictions

- meva_frozen: Woehrer 2026 78.0% (48), Deep-OAD 85.4% (26), GeoCalib 24.6% (138)
- v1_frozen: Woehrer 2026 70.6% (98), Deep-OAD 82.8% (113), GeoCalib 25.0% (749)

## MEVA test v1, frozen split: unseen CCTV cameras (560 views)

### Within 10°, every view answered (≥150° errors)

| Slice (n) | Woehrer 2026 | Deep-OAD | GeoCalib | Pico | Nano | Fast | Balanced | Pro | Max |
|---|---|---|---|---|---|---|---|---|---|
| Held out: 4 cameras G329 / G420 / G421 / G474 (380) | 75.0% (47) | 82.4% (26) | 23.9% (86) | 80.0% (21) | 92.6% (21) | 97.9% (8) | 98.4% (6) | 99.2% (3) | 100.0% (0) |
| All 6 cameras (G331, G639 = v2 development cameras) (560) | 78.0% (48) | 85.4% (26) | 24.6% (138) | 86.4% (21) | 95.0% (21) | 98.6% (8) | 98.9% (6) | 99.5% (3) | 100.0% (0) |
| Held out, clean (190) | 88.9% (18) | 84.7% (11) | 22.1% (39) | 83.7% (7) | 94.2% (10) | 98.4% (3) | 98.4% (3) | 98.4% (3) | 100.0% (0) |
| Held out, degraded (190) | 61.1% (29) | 80.0% (15) | 25.8% (47) | 76.3% (14) | 91.1% (11) | 97.4% (5) | 98.4% (3) | 100.0% (0) | 100.0% (0) |
| camera G329 (90) | 84.4% (0) | 72.2% (0) | 30.0% (19) | 41.1% (0) | 93.3% (1) | 98.9% (1) | 100.0% (0) | 100.0% (0) | 100.0% (0) |
| camera G331 (v2 dev camera) (90) | 81.1% (0) | 100.0% (0) | 27.8% (27) | 100.0% (0) | 100.0% (0) | 100.0% (0) | 100.0% (0) | 100.0% (0) | 100.0% (0) |
| camera G420 (90) | 77.8% (0) | 93.3% (0) | 14.4% (15) | 95.6% (3) | 94.4% (3) | 100.0% (0) | 100.0% (0) | 100.0% (0) | 100.0% (0) |
| camera G421 (90) | 88.9% (0) | 100.0% (0) | 31.1% (23) | 95.6% (4) | 94.4% (5) | 100.0% (0) | 100.0% (0) | 100.0% (0) | 100.0% (0) |
| camera G474 (110) | 53.6% (47) | 67.3% (26) | 20.9% (29) | 86.4% (14) | 89.1% (12) | 93.6% (7) | 94.5% (6) | 97.3% (3) | 100.0% (0) |
| camera G639 (v2 dev camera) (90) | 87.8% (1) | 83.3% (0) | 24.4% (25) | 100.0% (0) | 100.0% (0) | 100.0% (0) | 100.0% (0) | 100.0% (0) | 100.0% (0) |

### Max vs comparators on the held-out views: paired difference in correct views, 95% cluster bootstrap

- Max − Woehrer 2026: +95 views [+77, +113] of 380 (significant)
- Max − Deep-OAD: +67 views [+52, +82] of 380 (significant)

### Abstention on the held-out views (thresholds as shipped, fp32)

| Tier | standard: answered | correct among answered | ≥150° answered | strict: answered | correct among answered | ≥150° answered |
|---|---|---|---|---|---|---|
| Pico | 97.9% | 80.6% | 20 | 51.8% | 94.9% | 1 |
| Nano | 95.5% | 95.3% | 14 | 76.1% | 97.9% | 5 |
| Fast | 98.9% | 98.4% | 6 | 96.6% | 98.9% | 4 |
| Balanced | 98.2% | 98.7% | 5 | 97.1% | 98.9% | 4 |
| Pro | 99.7% | 99.2% | 3 | 99.7% | 99.2% | 3 |
| Max | 99.7% | 100.0% | 0 | 99.7% | 100.0% | 0 |

## Clean v1 frozen test: DIODE scene 12, MEVA G339, Poly Haven (2854 views)

### Within 10°, every view answered (≥150° errors)

| Slice (n) | Woehrer 2026 | Deep-OAD | GeoCalib | Pico | Nano | Fast | Balanced | Pro | Max |
|---|---|---|---|---|---|---|---|---|---|
| All (held out) (2854) | 70.6% (98) | 82.8% (113) | 25.0% (749) | 96.5% (11) | 98.0% (7) | 97.8% (10) | 98.0% (6) | 98.7% (5) | 98.8% (5) |
| Clean (1427) | 80.4% (33) | 83.3% (52) | 25.0% (379) | 97.0% (4) | 98.0% (5) | 98.0% (6) | 98.2% (3) | 98.8% (2) | 99.2% (2) |
| Degraded (1427) | 60.8% (65) | 82.3% (61) | 24.9% (370) | 95.9% (7) | 98.0% (2) | 97.6% (4) | 97.8% (3) | 98.6% (3) | 98.5% (3) |
| diode:scene_00012 (372) | 53.8% (12) | 64.8% (11) | 20.7% (91) | 85.8% (0) | 91.4% (0) | 91.1% (0) | 92.2% (0) | 94.4% (0) | 94.9% (0) |
| meva:G339 (900) | 82.3% (43) | 81.8% (44) | 25.6% (242) | 95.9% (11) | 98.0% (4) | 97.0% (9) | 97.1% (6) | 98.3% (5) | 98.6% (5) |
| poly_haven (1582) | 67.9% (43) | 87.5% (58) | 25.7% (416) | 99.3% (0) | 99.6% (3) | 99.9% (1) | 99.9% (0) | 99.9% (0) | 99.9% (0) |

### Max vs comparators on the held-out views: paired difference in correct views, 95% cluster bootstrap

- Max − Woehrer 2026: +806 views [+754, +858] of 2854 (significant)
- Max − Deep-OAD: +459 views [+413, +506] of 2854 (significant)

### Abstention on the held-out views (thresholds as shipped, fp32)

| Tier | standard: answered | correct among answered | ≥150° answered | strict: answered | correct among answered | ≥150° answered |
|---|---|---|---|---|---|---|
| Pico | 98.1% | 97.6% | 7 | 82.7% | 99.6% | 1 |
| Nano | 98.2% | 98.7% | 4 | 90.8% | 99.9% | 0 |
| Fast | 98.0% | 98.5% | 2 | 95.6% | 99.3% | 1 |
| Balanced | 98.0% | 98.8% | 2 | 96.0% | 99.3% | 1 |
| Pro | 97.8% | 99.5% | 1 | 97.8% | 99.5% | 1 |
| Max | 97.7% | 99.7% | 0 | 97.7% | 99.7% | 0 |

