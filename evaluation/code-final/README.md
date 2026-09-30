# Sealed-holdout scoring code (release freeze)

The 14 files whose SHA-256 are bound in `results/sealed-holdout/FROZEN.json` (`code_sha256`), byte for byte. They
rendered and scored the sealed final holdout on 29 Sep 2026. To reproduce that scoring tree, copy them over
`rotlab/` (the published `rotlab/` carries later, backward-compatible versions of `model.py` and `model_x.py`
with the Pico additions, and text-sanitised comments). Being hash-bound, these files are published unchanged,
including internal workspace names in two strings of `focus.py`.
