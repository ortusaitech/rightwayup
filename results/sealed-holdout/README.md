# Sealed final holdout: freeze and scoring records

> Terminology note (added 30 Sep 2026): in this record, 'v2' denotes the RightWayUp 1.0 release weights and 'v1' the 24 Sep release candidate; the record itself is unchanged.


Machine records of the release freeze (29 Sep 2026) and of the once-only sealed-holdout scoring, copied
unchanged: `FROZEN.json` (the freeze record: checkpoints, tiers, thresholds, comparators, holdout source
manifests, code hashes), `SPEC.json` (the spec it was made from), `FINAL-RESULTS.json` and `FINAL-PAIRED.json`
(per-tier results and paired differences vs Woehrer 2026, i.e. MambaOut-base CGD by Maximilian Woehrer,
[arXiv:2603.25351](https://arxiv.org/abs/2603.25351), [code](https://github.com/maxwoe/image-rotation-angle-estimation),
MIT licence), `thresholds-per-format.json` and
`thresholds-pico-s80.json` (per-format thresholds), `thresholds-pico.json` and `ARTIFACTS-pico-SHA256.txt`
(the superseded 29 Sep Pico), `ARTIFACTS-pico-s80-SHA256.txt` (the released Pico files). The records use the
internal labels in their fields.

*Model size and training data:* Woehrer 2026 has 83M parameters and was trained on COCO 2017 (about 118k training images). Max has 306M
(ViT-L/14); Nano and Fast have 23M each (ViT-S/14); Balanced and Pro run Fast, then Max on part of the images.
Max and Fast were trained on about 1.24 million images, Nano on 705,075.
