# Benchmark notes: crop shape, JPEG grid and seeds

We compare RightWayUp with Woehrer 2026 (MambaOut-base with a 360-bin circular Gaussian loss, arXiv:2603.25351,
code and weights under MIT, github.com/maxwoe/image-rotation-angle-estimation). It is the published state of the
art on the COCO rotation benchmark: 1,030 COCO val2014 photos (Fischer et al. 2015, classes 1 and 2), each rotated
to a random angle under five test seeds, scored as the share of views within 10° of the truth.

On that benchmark as published, our released Max tier scores 98.8% and Woehrer 2026 98.0% (five-seed means). While
building RightWayUp we looked closely at how the benchmark is built and found three properties that matter when its
numbers are read as a guide to real cameras. None of them is a flaw of one paper: the crop and the rendering are
common in RotNet-style code, and the seed point is about how we had been quoting the result. The author made all of
this checkable by releasing the code, the weights and the test list, and removed from training the 915 benchmark
photos that reappear in COCO train2017.

## Summary

| Version of the test (1,030 photos) | Woehrer 2026 | RightWayUp Max (released) |
|---|---|---|
| As published, five-seed mean | 98.0% (1009.8) | **98.8%** (1018.0) |
| As published, seed 0 only (the figure we used to quote) | 98.6% (1016) | 99.2% (1022) |
| Neutral 4:3 crop, same photos, one angle draw (seed 7) | 97.2% (1001) | **99.0%** (1020) |
| As published, each view saved once as JPEG q90, five-seed mean | 30.2% (311.0) | **98.4%** (1013.4) |
| Neutral crop, saved once as JPEG q90 | 32.7% (337) | not scored |

1. **Seeds.** The paper reports the mean over five test seeds. We had been quoting seed 0 of our rebuild, which is
   Woehrer 2026's best of the five. The per-seed scores also depend on the order in which the test files are
   listed, so only the five-seed mean is portable between machines.
2. **Crop shape.** Each rotated photo is cropped to the largest axis-aligned rectangle inside it. The rectangle's
   size and field of view are a function of the angle, and Woehrer 2026 uses the same crop for training and
   testing. With a crop that does not depend on the angle, the gap between the two models widens.
3. **JPEG grid.** The benchmark rotates JPEG photos and stores the rotated views losslessly, so each photo's 8 by 8
   JPEG block grid rotates with the content. A camera compresses its own frame, so the block grid in real camera
   output is aligned with the frame edges. Saving each benchmark view once as JPEG q90 drops Woehrer 2026 from 98.0%
   to 30.2%, and its predictions snap to the nearest multiple of 90°. Our tiers change by less than half a point.

On the benchmark as published, Max leads by 0.8 points. On the neutral crop and on the JPEG copies, the gap widens.
The crop and the JPEG grid remain properties of the protocol, whichever model leads on it.

## 1. Setup

- **Views.** Our rebuild uses the upstream `rotation_utils.py` at commit `e4e35610` (cv2 bilinear rotation with an
  expanded canvas, then a centre crop to `largest_rotated_rect`) and the upstream angle rule
  `numpy.random.seed(seed); uniform(0, 360, 1030)` for seeds 0 to 4. All 5,150 views match the pixel hashes of our
  earlier independent rebuild (opencv-python-headless 4.12.0.88, numpy 2.1.2, Pillow 11.0.0). The views are stored
  as PNG, so they are lossless like the upstream in-memory pipeline.
- **Woehrer 2026.** The public checkpoint `cgd_mambaout_base_coco2017.ckpt` (Hugging Face `maxwoe/image-rotation-angle-estimation`),
  exported to ONNX FP32 and run on CPU with ONNX Runtime. Preprocessing is the upstream eval transform (timm:
  shortest side to 224, bicubic, centre crop 224, ImageNet normalisation), decoding is argmax over 360 bins. The ONNX
  export and the checkpoint run directly in PyTorch (the reproduction script in section 7) give the same prediction
  on 10,298 of the 10,300 clean and JPEG views; one seed-3 view differs by 1°, which moves seed 3 of the JPEG copy
  from 308 to 309 (five-seed JPEG mean 311.0 with ONNX, 311.2 with the checkpoint).
- **RightWayUp.** The six released tiers (`FREEZE.md`): Max is the DINOv2 ViT-L/14 at 280 px on every image; Pro and
  Balanced are Fast → Max cascades that send about 20% and 10% of images to Max (calibration share); Fast is a
  ViT-S/14 at 224 px, Nano a ViT-S/14 at 112 px, and Pico is Nano fine-tuned to 70 px. Scores are PyTorch scores of
  the release weights; the release FP32 ONNX files give the same predictions.
- **Metric.** Within 10° of the truth, circular error. Paired intervals are 95% bootstrap intervals over source
  photos (all seeds of a photo resampled together).

### All released tiers (within 10°)

| Test (1,030 photos) | Woehrer 2026 | Max | Pro | Balanced | Fast | Nano | Pico |
|---|---|---|---|---|---|---|---|
| As published, five-seed mean | 98.04% | 98.83% | 98.66% | 97.75% | 96.52% | 96.00% | 92.23% |
| Saved once as JPEG q90, five-seed mean | 30.19% | 98.39% | 98.17% | 97.50% | 96.45% | 95.88% | 92.17% |
| Neutral 4:3 crop (seed 7) | 97.18% | 99.03% | 98.83% | 98.54% | 96.89% | 94.76% | 91.94% |

On the benchmark as published, Max and Pro score above Woehrer 2026 and the smaller tiers below it. Every tier loses
less than half a point on the JPEG copies.

## 2. Seeds

| Seed | 0 | 1 | 2 | 3 | 4 | Mean |
|---|---|---|---|---|---|---|
| Woehrer 2026, our rebuild | **1016** | 1012 | 1014 | 999 | 1008 | 1009.8 (98.04%) |
| Woehrer 2026, upstream per-seed results JSON (acc@10) | .978 | .982 | .981 | .976 | **.986** | .9806 |
| RightWayUp Max | 1022 | 1018 | 1020 | 1015 | 1015 | 1018.0 (98.83%) |

- The paper reports the five-seed mean (Acc@10 0.98). Our earlier documents quoted 98.6% (1016), which is seed 0 of
  our rebuild and the best of the five. That was our choice of figure, not the paper's.
- The upstream harness lists test images with `glob.glob`, whose order depends on the filesystem, and assigns the
  seeded angles in that order. The upstream repo's best seed is seed 4; ours is seed 0. The means agree (98.06% and
  98.04%). We publish the image order we used (`w_benchmark_views.json`), so our per-seed numbers can be reproduced
  exactly, but only the five-seed mean should be compared across setups.
- Per-seed scores for Woehrer 2026 range from 999 to 1016 of 1,030, so a single seed can move the reported
  accuracy by up to about one point.
- Paired over all 5,150 views, Max minus Woehrer 2026 is +41 views [8, 76].

## 3. JPEG grid

### 3.1 Five seeds, JPEG q90

Each view was saved once with Pillow 11.0.0, `quality=90` (default 4:2:0 chroma subsampling), after rotation and
crop, and decoded again. Nothing else changed.

| Model | Clean, five-seed mean | JPEG q90, five-seed mean | Per seed, JPEG q90 |
|---|---|---|---|
| Woehrer 2026 | 1009.8 (98.0%) | **311.0 (30.2%)** | 323 / 322 / 305 / 308 / 297 |
| RightWayUp Max | 1018.0 (98.8%) | 1013.4 (98.4%) | 1016 / 1018 / 1015 / 1012 / 1006 |

The other tiers are in the table in section 1.

### 3.2 What the errors look like (five seeds, 5,150 views)

| | Woehrer 2026 clean | Woehrer 2026 JPEG q90 | Max clean | Max JPEG q90 |
|---|---|---|---|---|
| Median error | 0.56° | 21.1° | 1.17° | 1.21° |
| Mean error | 2.82° | 23.8° | 2.04° | 2.13° |
| Within 2° | 4,706 | 460 | 3,660 | 3,562 |
| Predictions within 2° of 0/90/180/270° (truth: 4.4%) | 6.4% | **75.8%** | 4.3% | 4.3% |
| Misses (> 10°) that sit within 2° of an axis | 6 of 101 | 2,923 of 3,595 | 2 of 60 | 0 of 83 |
| Upside-down errors (≥ 150°) | 25 | 31 | 6 | 5 |

Within 10° for Woehrer 2026, by the true angle's distance to the nearest multiple of 90°:

| Distance of truth from nearest axis | 0 to 2° | 2 to 5° | 5 to 10° | 10 to 15° | 15 to 25° | 25 to 35° | 35 to 45° |
|---|---|---|---|---|---|---|---|
| Woehrer 2026, clean | 98.2% | 99.1% | 98.4% | 98.2% | 97.5% | 97.5% | 98.5% |
| Woehrer 2026, JPEG q90 | 98.7% | 99.7% | 92.3% | 35.9% | 12.3% | 6.9% | 5.2% |

After one JPEG save, 68.0% of Woehrer 2026's answers sit within 2° of the multiple of 90° nearest the true angle
(6.3% before), while errors of 90° or more rise only from 51 to 85. The model still finds roughly which way is up,
but reports the nearest multiple of 90° instead of the angle, so it stays accurate only when the truth is already
within about 10° of an axis. Max's error distribution is the same before and after the save.

### 3.3 Controls on seed 0 (1,030 views)

| Seed-0 views (Woehrer 2026) | Within 10° | Within 2° | Median error | Predictions within 2° of an axis |
|---|---|---|---|---|
| As published (lossless) | 1016 (98.6%) | 947 | 0.56° | 6.4% |
| Decoded and re-saved as PNG | 1016 (98.6%) | 947 | 0.56° | 6.4% |
| JPEG q95 | 417 (40.5%) | 161 | 16.2° | 51.6% |
| JPEG q90 | 323 (31.4%) | 93 | 20.9° | 74.7% |
| JPEG q75 | 267 (25.9%) | 73 | 22.6° | 92.1% |
| Neutral crop (seed 7), lossless | 1001 (97.2%) | 866 | 0.78° | 6.1% |
| Neutral crop (seed 7), JPEG q90 | 337 (32.7%) | 96 | 19.6° | 75.0% |

- The lossless PNG round trip gives the same prediction on all 1,030 views, so the drop is not a file-handling
  effect.
- The effect grows with compression: at quality 95 about half of the answers sit on an axis, at quality 75 almost
  all of them do. Even quality 95 costs 58 points on seed 0.
- The neutral-crop views are rendered with PIL bicubic interpolation instead of OpenCV bilinear, and the crop does
  not depend on the angle. One JPEG save has the same effect there (1001 → 337), so it is not tied to the max-area
  crop or to one resampler.
- The rebuilt neutral-crop views give the same Woehrer 2026 prediction as our published fair-photo panel on all
  1,030 views.

### 3.4 Our reading

The upstream pipeline decodes a JPEG photo, rotates the pixels and keeps the result lossless, in training and in
testing. The source photo's 8 by 8 block grid therefore rotates with the content, and its orientation carries the
angle modulo 90° at any angle. One JPEG save adds a new grid aligned with the frame, which reads as "0° modulo
90°". The snapping in 3.2 is what a model that relies on the grid for the fine angle, and on the content for the
quadrant, would do, and so is the dose response in 3.3: the stronger the new grid (the lower the quality), the
more answers sit on an axis. We have not proven this mechanism; a render that removes the source grid before rotation would
test it directly, and we have not run that yet. The author's own code notes that JPEG recompression can degrade
accuracy; what we add is the measurement.

Whatever the mechanism, one JPEG save at quality 90 is a mild change compared with what a camera, a phone or a
video encoder does to every frame, and the effect is already large at quality 95.

## 4. Crop shape

The largest axis-aligned rectangle inside a rotated photo changes with the angle (near 45° it becomes squarer and,
for a 4:3 photo, smaller). The upstream eval transform centre-crops to a square, which hides the aspect ratio from
the network, but the field of view and scale still vary with the angle, and the model is trained on the same crop it
is tested on.

- **Neutral crop, same photos** (our "fair photo" panel): a fixed 4:3 rectangle whose diagonal equals the photo's
  shorter side minus 4 px, so it fits inside the inscribed circle at every angle. Angles are one fresh draw
  (`random.Random(7)`), not the benchmark's, and rendering is PIL bicubic. Woehrer 2026 scores 97.2% (1001), Max
  99.0% (1020). The local rebuild reproduces Woehrer 2026's 1001 prediction for prediction.
- **Paired crop test on other photos:** on 2,124 COCO val2017 photos, same photo and same angle, only the crop
  changing, Woehrer 2026 gains 17 views on the max-area crop (1992 → 2009; 45 / 28 discordant, exact McNemar
  p = 0.06), while Max loses 18 (2038 → 2020; 16 / 34 discordant, exact McNemar p = 0.015). The crop gives the model
  trained on it a small gain that is not statistically established, and costs Max, which never trained on it, a small
  one that is. These photos were
  stored at 448 px as JPEG q92 before rotation, so they are not a faithful replica of the benchmark rendering.

We do not claim that the crop explains a specific number of views: for Woehrer 2026 the effect is not statistically
established. It is a train/test coupling that a rolled camera never provides, so we report the benchmark both ways.

## 5. What we do about it

- We never train release candidates on the max-area crop (owner decision, 23 Sep 2026). Training crops are
  angle-independent.
- About a third of our training views are JPEG-compressed after rotation (simulated degradation applied with
  p = 0.65, JPEG with p = 0.5, quality 25 to 90), as camera frames are.
- From 27 Sep 2026 on, a benchmark improvement counts only if it survives the JPEG q90 copy of the benchmark. Every
  evaluation readout reports both.
- We report Woehrer 2026 by its five-seed mean, as its paper does.
- **Our own earlier pre-release models read the same kind of cue.** Our 24 Sep release candidate also picked up the angle from
  the rotated JPEG grid and the resampling traces that digital rotation leaves in a photo, though less than
  Woehrer 2026 did. When we rendered our photo tests the way a rolled camera produces them, the size of this became
  clear: our 1:1 photo renders had overstated accuracy by 12 to 21 points, and on camera-like photos our Max of that
  time did not beat Woehrer 2026 (76.0% vs 77.6%). So we retrained every released tier on grid-free data (training
  images stored losslessly, never JPEG-encoded before rotation), and every shipped file passes a check for these
  cues on content-free probe images in its own runtime. On the same camera-like photos the released Max scores 83.6%
  against 77.6% (a development test). We mention this because it is the same kind of shortcut: it comes from how
  rotation data is usually rendered, not from one team's choices.

## 6. Limitations

- One JPEG encoder (Pillow 11.0.0 / libjpeg-turbo, 4:2:0). Other encoders and chroma settings may move the numbers.
- The fair-crop comparison changes the crop, the angles and the interpolation at once. The paired val2017 test
  isolates the crop but uses different photos and pre-compressed sources.
- The benchmark's seed-0 panel was read many times during our development, and the five-seed benchmark and its JPEG
  copies were development panels for the released models (used for model choice), so our figures on them carry
  selection bias. The unbiased comparison with Woehrer 2026 is the sealed final holdout in our release results
  (4,706 new photos, opened once after the weights were frozen).
- The grid mechanism is our reading, not a proven cause (3.4).

## 7. Reproduce

```bash
pip install numpy pillow opencv-python-headless onnxruntime      # add torch torchvision timm for --ckpt
# bit-exact views: opencv-python-headless==4.12.0.88 numpy==2.1.2 pillow==11.0.0
python reproduce_w_jpeg.py --coco-dir val2014 --download \
    --ckpt cgd_mambaout_base_coco2017.ckpt --seeds 0,1,2,3,4 --conditions clean,jpeg90 --out w-jpeg-results.json
python reproduce_w_jpeg.py --coco-dir val2014 --ckpt cgd_mambaout_base_coco2017.ckpt \
    --seeds 0 --conditions clean,png,jpeg95,jpeg90,jpeg75
python reproduce_w_jpeg.py --coco-dir val2014 --ckpt cgd_mambaout_base_coco2017.ckpt --protocol fair --conditions clean,jpeg90
```

The script (`benchmarks/woehrer-2026/reproduce_w_jpeg.py`) downloads the 1,030
photos from images.cocodataset.org and checks their SHA-256, rebuilds every view with the upstream rotation code,
runs the public checkpoint (or an ONNX export of it), and prints within-10 per seed, the five-seed mean and the share
of predictions near an axis. It reports how many rendered views match our pixel hashes. Our run of the command above
with the public checkpoint (pinned versions, CPU, about 45 minutes) matched our pixel hashes on all 5,150 views and
our JPEG bytes on all 5,150 copies, and printed 1009.8 (98.04%) clean and 311.2 (30.21%) for JPEG q90.
`w-jpeg-results.json` holds that run's per-view predictions for comparison.

If your numbers differ from ours, or something here misdescribes Woehrer 2026, please open an issue or write to
hello@ortusai.io. We will correct it.
