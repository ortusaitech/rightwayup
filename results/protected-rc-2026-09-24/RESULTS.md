# Protected final evaluation (frozen weights, run once)

> Terminology note (added 30 Sep 2026): in this record, 'v2' denotes the RightWayUp 1.0 release weights and 'v1' the 24 Sep release candidate; the record itself is unchanged.


> **Strict mode, 25 Sep 2026.** `strict` uses the higher of the two frozen thresholds, so it never answers more
> than `standard`. On Balanced and Max the calibrated strict threshold (0.750, 0.677) was below the standard one
> (0.778, 0.808): the standard point already met the ≤1%-wrong target on calibration data. Their strict rows
> therefore equal the standard rows. Nano and Fast are unchanged. No threshold was tuned on test data.

Weights and thresholds as recorded in `FREEZE.md`; views built once, identical pixels for every model.

> **Scoring note (24 Sep 2026).** The first report from this run (`RESULTS-v0-sign-bug.md`) scored Deep-OAD and
> GeoCalib on their native counter-clockwise outputs without the sign convention (−1) that was fixed for both on
> calibration photos before any testing (`competitors/*-sign.json`). That put them at chance level (about 5%). The
> report was corrected to apply the stored sign, as the benchmark scorer does. No model was re-run and no protected
> view was re-opened; predictions for every model are as first produced. Our tiers and Woehrer 2026 (sign +1) are
> unchanged between the two reports.

## MEVA test v1, frozen split: unseen CCTV cameras (560 views)

### Accuracy within 10°, all views answered (upside-down errors ≥150° in brackets)

| Slice (n) | Woehrer 2026 | Deep-OAD | GeoCalib | Nano | Fast | Balanced | Max |
|---|---|---|---|---|---|---|---|
| All (560) | 78.0% (48) | 85.4% (26) | 24.6% (138) | 94.8% (16) | 97.9% (12) | 98.8% (7) | 99.1% (5) |
| Clean (280) | 92.5% (18) | 87.9% (11) | 23.6% (68) | 93.6% (8) | 97.9% (6) | 99.3% (2) | 98.9% (3) |
| Degraded (280) | 63.6% (30) | 82.9% (15) | 25.7% (70) | 96.1% (8) | 97.9% (6) | 98.2% (5) | 99.3% (2) |
| meva:G329 (90) | 84.4% (0) | 72.2% (0) | 30.0% (19) | 78.9% (6) | 93.3% (6) | 97.8% (2) | 100.0% (0) |
| meva:G331 (90) | 81.1% (0) | 100.0% (0) | 27.8% (27) | 100.0% (0) | 100.0% (0) | 100.0% (0) | 100.0% (0) |
| meva:G420 (90) | 77.8% (0) | 93.3% (0) | 14.4% (15) | 100.0% (0) | 100.0% (0) | 100.0% (0) | 100.0% (0) |
| meva:G421 (90) | 88.9% (0) | 100.0% (0) | 31.1% (23) | 97.8% (2) | 100.0% (0) | 100.0% (0) | 100.0% (0) |
| meva:G474 (110) | 53.6% (47) | 67.3% (26) | 20.9% (29) | 92.7% (8) | 94.5% (6) | 95.5% (5) | 95.5% (5) |
| meva:G639 (90) | 87.8% (1) | 83.3% (0) | 24.4% (25) | 100.0% (0) | 100.0% (0) | 100.0% (0) | 100.0% (0) |

### Within ±45° true roll only (calibration-model scope, n = 123)

| Woehrer 2026 | Deep-OAD | GeoCalib | Nano | Fast | Balanced | Max |
|---|---|---|---|---|---|---|
| 73.2% | 89.4% | 99.2% | 94.3% | 97.6% | 99.2% | 100.0% |

### With abstention: standard (90% answered on calibration)

| Tier | answered | correct within 10° among answered | upside-down among answered |
|---|---|---|---|
| Nano | 97.5% | 95.6% | 14 |
| Fast | 96.4% | 99.1% | 5 |
| Balanced | 97.7% | 99.5% | 3 |
| Max | 99.1% | 99.8% | 1 |

### With abstention: strict (≤1% wrong on calibration; never looser than standard)

| Tier | answered | correct within 10° among answered | upside-down among answered |
|---|---|---|---|
| Nano | 83.2% | 98.7% | 4 |
| Fast | 95.5% | 99.3% | 4 |
| Balanced | 97.7% | 99.5% | 3 |
| Max | 99.1% | 99.8% | 1 |

Routed to the large model: Balanced 3.6%, Max 13.4%.

## Clean v1 frozen test: DIODE scene 12, MEVA G339, Poly Haven (2854 views)

### Accuracy within 10°, all views answered (upside-down errors ≥150° in brackets)

| Slice (n) | Woehrer 2026 | Deep-OAD | GeoCalib | Nano | Fast | Balanced | Max |
|---|---|---|---|---|---|---|---|
| All (2854) | 70.6% (98) | 82.8% (113) | 25.0% (749) | 96.8% (11) | 96.8% (6) | 97.6% (3) | 98.5% (3) |
| Clean (1427) | 80.4% (33) | 83.3% (52) | 25.0% (379) | 96.8% (4) | 97.1% (3) | 98.2% (0) | 98.8% (0) |
| Degraded (1427) | 60.8% (65) | 82.3% (61) | 24.9% (370) | 96.8% (7) | 96.5% (3) | 97.0% (3) | 98.2% (3) |
| diode:scene_00012 (372) | 53.8% (12) | 64.8% (11) | 20.7% (91) | 89.2% (0) | 88.7% (0) | 89.5% (1) | 92.7% (1) |
| meva:G339 (900) | 82.3% (43) | 81.8% (44) | 25.6% (242) | 96.0% (6) | 95.4% (3) | 97.0% (2) | 98.2% (2) |
| poly_haven (1582) | 67.9% (43) | 87.5% (58) | 25.7% (416) | 99.1% (5) | 99.5% (3) | 99.9% (0) | 100.0% (0) |

### Within ±45° true roll only (calibration-model scope, n = 718)

| Woehrer 2026 | Deep-OAD | GeoCalib | Nano | Fast | Balanced | Max |
|---|---|---|---|---|---|---|
| 68.5% | 87.0% | 93.0% | 96.1% | 96.5% | 97.1% | 98.5% |

### With abstention: standard (90% answered on calibration)

| Tier | answered | correct within 10° among answered | upside-down among answered |
|---|---|---|---|
| Nano | 98.4% | 97.5% | 10 |
| Fast | 96.3% | 98.3% | 1 |
| Balanced | 95.3% | 99.0% | 0 |
| Max | 95.6% | 99.6% | 0 |

### With abstention: strict (≤1% wrong on calibration; never looser than standard)

| Tier | answered | correct within 10° among answered | upside-down among answered |
|---|---|---|---|
| Nano | 91.9% | 99.2% | 1 |
| Fast | 95.2% | 98.8% | 0 |
| Balanced | 95.3% | 99.0% | 0 |
| Max | 95.6% | 99.6% | 0 |

Routed to the large model: Balanced 3.7%, Max 10.9%.

