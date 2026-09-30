# Evaluation-set definitions (published at release; owner decision D8)

> Terminology note (added 30 Sep 2026): in this record, 'v2' denotes the RightWayUp 1.0 release weights and 'v1' the 24 Sep release candidate; the record itself is unchanged. Here 'v1' in set and file names marks the sets built for the release candidate; both sets were re-scored once with the release on 30 Sep 2026 (`results/heldout-rescore/`).


These are pointers, not pixels, for the two protected test sets. With them, anyone can rebuild the exact views and
re-score RightWayUp or any other model on identical inputs.

| File | Rows | Content |
|---|---|---|
| `meva-test-v1-frames.jsonl` | 711 | Every MEVA test-v1 frame: camera, split (`dev` / `frozen` / `train` = the thermal cameras G475 and G476, moved to training on 23 Sep 2026), public MEVA video key and frame index or time, base roll, SHA-256 of the extracted frame |
| `meva-frozen-views.jsonl` | 560 | The protected MEVA views in scoring order: frame, clean or degraded, applied roll (clockwise, including base roll) |
| `clean-v1-frozen-parents.jsonl` | 1,427 | Clean v1 frozen-test parents: DIODE file (scene 12), MEVA key and frame (camera G339), or ORTUS AI render file |
| `clean-v1-frozen-views.jsonl` | 2,854 | The protected clean-v1 views in scoring order |

- **Rebuilding:** `python -m rotlab.protected_eval dump --i-am-releasing` recreates the views from the frames. It uses fixed seeds, the fixed 4:3 crop, and simulated CCTV degradation on the "degraded" half. The applied angles must match the `*-views.jsonl` files.
- **Renders:** the 791 Poly Haven render parents of the clean-v1 test are in `ortus-polyhaven-orientation-renders-v1-clean-v1-frozen.tar`, released with the renders dataset under CC BY 4.0. The source assets are Poly Haven, CC0.
- **Blindness:** these sets were scored once, on frozen weights, on 24 Sep 2026 (`FREEZE-release-candidate-2026-09-24.md`). Once published they are no longer blind. Future RightWayUp versions will be judged on a new private protected set.
- **Licences:**
  - MEVA: Kitware / IARPA, CC BY 4.0.
  - DIODE: MIT.
  - These definition files: CC BY 4.0, ORTUS AI SOFTWARE LIMITED.
