# Risk-coverage on the protected sets (frozen weights)

> Terminology note (added 30 Sep 2026): in this record, 'v2' denotes the RightWayUp 1.0 release weights and 'v1' the 24 Sep release candidate; the record itself is unchanged.


Views ranked by the answering model's confidence (probability mass within ±10°, cascade routing as released).
Risk = share of answered views more than 10° off. AURC = mean risk over all coverages (lower is better).
Operating points use the thresholds frozen on calibration data (`FREEZE.md`); no test data chose them.

## Unseen CCTV cameras (MEVA frozen, 560 views)

| Tier | risk at 100% coverage | AURC | standard: answered / risk | strict: answered / risk |
|---|---|---|---|---|
| Nano | 5.2% | 1.24% | 97.5% / 4.4% | 83.2% / 1.3% |
| Fast | 2.1% | 0.11% | 96.4% / 0.9% | 95.5% / 0.7% |
| Balanced | 1.2% | 0.06% | 97.7% / 0.5% | 97.7% / 0.5% |
| Max | 0.9% | 0.03% | 99.1% / 0.2% | 99.1% / 0.2% |

## Clean v1 frozen (DIODE scene 12, MEVA G339, Poly Haven; 2,854 views)

| Tier | risk at 100% coverage | AURC | standard: answered / risk | strict: answered / risk |
|---|---|---|---|---|
| Nano | 3.2% | 0.25% | 98.4% / 2.5% | 91.9% / 0.8% |
| Fast | 3.2% | 0.15% | 96.3% / 1.7% | 95.2% / 1.2% |
| Balanced | 2.4% | 0.12% | 95.3% / 1.0% | 95.3% / 1.0% |
| Max | 1.5% | 0.06% | 95.6% / 0.4% | 95.6% / 0.4% |
