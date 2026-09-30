# Sealed final holdout: freeze and scoring records

> Terminology note (added 30 Sep 2026): in this record, 'v2' denotes the RightWayUp 1.0 release weights and 'v1' the 24 Sep release candidate; the record itself is unchanged.


Machine records of the release freeze (29 Sep 2026) and of the once-only sealed-holdout scoring, copied
unchanged: `FROZEN.json` (the freeze record: checkpoints, tiers, thresholds, comparators, holdout source
manifests, code hashes), `SPEC.json` (the spec it was made from), `FINAL-RESULTS.json` and `FINAL-PAIRED.json`
(per-tier results and paired differences vs Woehrer 2026), `thresholds-per-format.json` and
`thresholds-pico-s80.json` (per-format thresholds), `thresholds-pico.json` and `ARTIFACTS-pico-SHA256.txt`
(the superseded 29 Sep Pico), `ARTIFACTS-pico-s80-SHA256.txt` (the released Pico files). The records use the
internal labels in their fields.
