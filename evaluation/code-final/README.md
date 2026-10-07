# Sealed-holdout scoring code (release freeze)

The 14 files whose SHA-256 are bound in `results/sealed-holdout/FROZEN.json` (`code_sha256`), byte for byte. They
rendered and scored the sealed final holdout on 29 Sep 2026. To reproduce that scoring tree, copy them over
`rotlab/` (the published `rotlab/` carries later, backward-compatible versions of `model.py` and `model_x.py`
with the Pico additions, text-sanitised comments and an attribution comment in `core.py`). Being hash-bound,
these files are published unchanged, including internal workspace names in two strings of `focus.py`.

`largest_rotated_rect` in `core.py` is adapted from `rotation_utils.largest_rotated_rect` in
[maxwoe/image-rotation-angle-estimation](https://github.com/maxwoe/image-rotation-angle-estimation) (commit
`e4e35610775fbb08f3f27641e8027d14b1498395`), Copyright (c) 2026 Maximilian Woehrer, MAXI solutions e.U., used
under the MIT License; the full licence text is in `NOTICE`.
