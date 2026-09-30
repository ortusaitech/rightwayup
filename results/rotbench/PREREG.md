# RotBench with the v2 release models — pre-registration

> Terminology note (added 30 Sep 2026): in this record, 'v2' denotes the RightWayUp 1.0 release weights and 'v1' the 24 Sep release candidate; the record itself is unchanged.

Written 30 Sep 2026 (AEST, time in the commit), **before any v2 prediction on RotBench**. Owner in chat, 30 Sep: "No, we
need it! Re-run as needed." (RotBench had been dropped from the v2 copy because it was not re-run.)

## Benchmark and views
- RotBench (arXiv 2508.13968; Hugging Face `tianyin/RotBench`, Apache-2.0), dataset commit
  `4ebf3d3b79830316955708dbd4c565c5be611b68` (last change 25 Aug 2025, i.e. the same images as the 24 Sep run).
  Parquet sha256 prefixes: large `2f9a2b9394e4a0fc`, small `e8f7e724277ec903`.
- Views rebuilt with the unchanged 24 Sep code (`rotlab/competitors.py` `_views`): each photo turned 0/90/180/270°
  counter-clockwise with PIL; `rotbench` = official protocol (no expand), `rotbench_full` = full frame (expand=True).
  RotBench-Small 50 photos, RotBench-Large 300 photos; 1,400 views per variant. The angle and split order must equal
  the stored 24 Sep comparator files (checked before scoring).

## Status for v2
- Never trained on: the new training families were screened against RotBench by perceptual hash (DATA-CARD, dedup).
- Never scored during v2 development (no v2 evaluation store contains RotBench). The v1 results were known.

## Models and scorer
- Release files as staged (one-image ONNX FP32 for pico-s70, nano-s112, fast-s224, max-l280; hashes as in
  `manifests/RELEASE-ARTIFACTS-SHA256.txt`), release package (core.py sha256 `4744a60687627b15609ee5057bc4a9a70a556c2abe4a5ae501dcc323bb8c2c77`), ONNX Runtime CUDA, all six tiers,
  Balanced / Pro routing as shipped. Every image answered.
- Metrics as in the 24 Sep report: 4-way accuracy per rotation (prediction snapped to the nearest quarter turn), mean,
  within 10°, ≥150° errors; per split and per variant. Comparators: stored 24 Sep predictions (Woehrer 2026, Deep-OAD,
  GeoCalib; signs as fixed on calibration photos), reproduced first; human and VLM rows only as published by RotBench.
- Scored once; all tiers reported whatever the result; no change of pipeline or thresholds after scoring.
