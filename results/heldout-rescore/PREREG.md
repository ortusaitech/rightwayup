# A9: v2 re-score of the 24 Sep held-out sets — pre-registration

> Terminology note (added 30 Sep 2026): in this record, 'v2' denotes the RightWayUp 1.0 release weights and 'v1' the 24 Sep release candidate; the record itself is unchanged.

Written 30 Sep 2026 14:23 AEST, **before any v2 prediction on these sets**. Owner approval in chat, 30 Sep ~14:20 AEST
("Yes, let's do all that before stopping the pod"), answering the pending A9 decision (claims register v2).

## Sets (identical pixels to the 24 Sep run)
- `meva_frozen`: MEVA test v1, frozen split, 560 views (6 cameras × clean + simulated-CCTV degradation), pickle sha256
  `7280aa646049ae05b8b31ee23b49093fab77ed72015c904e249541733331f2f4` (= the 24 Sep custody hash of `meva_frozen.pkl`).
- `v1_frozen`: clean v1 frozen test, 2,854 views (DIODE scene 12, MEVA G339, Poly Haven parents; clean + degraded),
  pickle sha256 `60bec573d7c14aba7ea2482e4c64b13dcd9ea36eed89470b4c9109718e38b5b4`; angles equal the 24 Sep record.

## Status of the sets for v2 (checked 30 Sep before scoring)
- Both sets were opened once, on 24 Sep, to score v1. This is their **second use**; v2 was developed after the v1
  results were known (e.g. the weakness on thermal camera G474). v2 never trained on them and no v2 evaluation store
  contains them (all `camp-eval` and `suite` stores checked).
- Training sources (all 36 source manifests on the volume): no row from the six test cameras, MEVA G339, DIODE scene 12,
  or any clean-test Poly Haven parent id.
- Selection: the v2 development set `gf_dev_meva` (used for grid-fix model choice) consists of archive development
  frames from **G331 and G639**, two of the six MEVA test cameras (other clips, same cameras). The development panel
  `meva` uses other cameras (the MEVA test v1 development split).
- Therefore: **held-out claim** = MEVA cameras G329, G420, G421, G474 (380 views) and the whole clean v1 test (2,854
  views). The full 560-view MEVA figure and cameras G331 / G639 are reported labelled "development cameras for v2".

## Models and scorer
- Release files as staged (`manifests/RELEASE-ARTIFACTS-SHA256.txt` sha256 `088d20552a5a5d3cf95fb15a44bf5648ba725db05394ae5f882575fd94e5059f`), one-image ONNX FP32 files
  (pico-s70, nano-s112, fast-s224, max-l280), each hash-verified on the pod before use.
- The release package `rightwayup` (core.py sha256 `4744a60687627b15609ee5057bc4a9a70a556c2abe4a5ae501dcc323bb8c2c77`), ONNX Runtime CUDA, precision fp32; Balanced / Pro routing and
  abstention exactly as shipped (fp32 thresholds). All six tiers; Pico is labelled a development tier (chosen after
  the freeze), as everywhere else.
- Comparators: the stored 24 Sep predictions on the identical views (Woehrer 2026, Deep-OAD, GeoCalib; sign convention
  from the 24 Sep sign-convention records (internal until the Perspective Fields / G3T licence check), fixed on calibration photos before any testing). The 24 Sep comparator
  numbers are reproduced first as a pipeline check; no comparator is re-run.

## Metrics (reported whatever they are; scored once; no threshold or pipeline change after scoring)
- Within 10°, every view answered; ≥150° counts. Slices: clean / degraded; per camera (MEVA); per group (clean test).
- Abstention at the standard and strict thresholds: answered share, accuracy among answered, ≥150° among answered.
- Max vs Woehrer 2026 (and vs Deep-OAD) on the held-out subsets: paired difference in correct views with a 95% bootstrap
  interval resampling frames (MEVA; clean + degraded views of one frame together) or parents (clean test).
- v1 (24 Sep) numbers are shown alongside for reference.
