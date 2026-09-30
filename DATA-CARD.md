# Data card: RightWayUp 1.0 training and evaluation data

**The training images are not redistributed.** This card, together with per-image manifests, traces every image to
its source, with per-author credit where the licence requires it, and lets anyone rebuild the corpus from the
original sources under their original terms. Removal requests: [`TAKEDOWN.md`](TAKEDOWN.md).
Counts are those of the training runs of the released models, 30 Sep 2026.

## Principles

1. **No customer data**, ever. No CHEQIT customer images or video.
2. **Sources whose licences permit commercial use**, each with a recorded licence snapshot. The licence layers for
   data, backbone weights and runtime are tracked separately.
3. **Per-image provenance.** Every image has a source ID, original URL, author (where the licence requires it),
   licence and licence URL, check date, content hash, and every transformation applied.
4. **Orientation labels come from a recorded upright state, not assumed.** Photos need a recorded display orientation
   (for Open Images, `Rotation = 0`, i.e. as the Flickr uploader displays it) and no EXIF orientation flag, or a
   publisher-rectified raster (DIODE). CCTV cameras are checked visually plus by vertical vanishing-point geometry.
   None of these is a measured gravity direction.
5. **Leakage control.** Near-duplicates of benchmark and evaluation photos are excluded by dHash and by ID; test
   cameras and scenes are excluded by allowlist.
6. **Self-supervised angles.** Training rotates each upright image by a fresh random angle, so no manual angle
   labels are needed.
7. **Grid-free storage (RightWayUp 1.0).** Every training image is stored losslessly (448-px PNG, never JPEG-encoded), so
   training views carry no rotated JPEG grid (`TECHNICAL-REPORT.md` §4).

## Training sources (RightWayUp 1.0)

"Used" = images the trainer loaded after the benchmark near-duplicate filter; "rows" = images in the pack.

| Source | Content | Licence (per image) | Selection and screening | Rows | Used |
|---|---|---|---|---|---|
| PASS (Asano et al., 2021) | Flickr/YFCC100M photos without people, two 300k subsets | CC BY 4.0 dataset grant; per-image Flickr CC BY 2.0 attribution from `pass_metadata.csv` | Shards 0–3; re-fetched at ≥ 1,020 px; lost images replaced by other PASS images (YFCC "Attribution License"); near-duplicate exclusion | 600,000 | 594,056 |
| COCO 2017 train, CC BY | Flickr photos whose licence is still CC BY | CC BY 2.0 (licence 7/8 rescues: CC BY 2.0 211, CC BY 4.0 90) | **Current** Flickr licence re-checked (oEmbed): must still be CC BY with an author; benchmark-exposed IDs excluded | 12,408 | 12,369 |
| COCO 2017 unlabeled | Flickr photos | CC BY 2.0 | Current licence re-checked; eval near-duplicates excluded | 9,480 | 9,480 |
| COCO, public-domain-like | COCO licence 7/8 rescues | No known copyright restrictions 428, CC0 113, Public Domain Mark 100, US Government Work 7 | Current Flickr status re-checked | 648 | 648 |
| Open Images test subset | Google, Flickr photos | CC BY 2.0 (images; 14 replacements CC BY 4.0); CC BY 4.0 (Google metadata) | Recorded display orientation `Rotation = 0`; no EXIF flag; current licence re-check; near-duplicate exclusion | 33,275 | 33,275 |
| Open Images V7 train | Google, Flickr photos (CVDF mirror, ≤ 1,024 px) | CC BY 2.0 / 4.0 (296,793); public-domain-like (970: CC0, PDM, NKCR, US Gov) | As above; author cap 20 (expansion); eval ≤ 8 bits / packs ≤ 6 bits dHash | 297,763 | 297,763 |
| CommonCatalog CC-BY | `common-canvas/commoncatalog-cc-by` (revision `80f50fe4`), YFCC100M photos ≥ 1,024 px | CC BY 2.0 only | Current Flickr licence must be exactly CC BY 2.0 with author; EXIF and current-thumbnail orientation checks; NSFW filter; author cap 60; not in any pack, evaluation set, the COCO Flickr universe or Open Images validation | 276,016 | 276,016 |
| DIODE (Vasiljevic et al., 2019) | Publisher-rectified RGB scans, train scenes | MIT | Lossless originals; scenes used by any test or calibration set excluded | 14,746 | 14,669 |
| MEVA (Kitware / IARPA) | Frames of 10 training cameras incl. thermal G475/G476 | CC BY 4.0 (dataset-level) | Training cameras only; lossless frame extraction | 3,750 | 3,749 |
| Poly Haven | ORTUS AI Blender renders of Poly Haven assets and HDRIs | CC0 assets; renders by ORTUS AI | Asset IDs and hashes recorded; voluntary credit | 3,586 | 3,579 |
| **Total** | 23 families | | | **1,251,672** | **1,245,604** |

- **Photos: 1,223,607 used** (1,229,590 rows); scans, CCTV frames and renders: 21,997 used.
- **Distinct images: 1,244,222 used, about 1.24 million.** 1,351 groups of photo rows (2,740 rows) hold the same
  picture (same Flickr photo id or byte-identical stored image): PASS photos also in Open Images or COCO (880 groups),
  repeats inside PASS (165), byte-identical re-uploads under different Flickr ids inside CommonCatalog (280, 259 of
  them by the same author), COCO photos also in Open Images (25) and one inside the Open Images test subset. They were
  trained as separate rows; counting each group once gives 1,382 fewer images than rows (photos: 1,222,225 distinct).
  The groups are listed in `manifests/training/SUMMARY.json` (`duplicates`).
- **Which model saw what.** Max, Fast and Pico were trained on all 23 families. Nano's final training used the 16
  families that existed before the Open Images V7 and CommonCatalog expansion (705,075 training rows used, 683,078
  photo rows). The
  checkpoints these runs started from were trained on JPEG-stored versions of the same sources (the 24 Sep packs,
  664,340 images used, plus the 27 Sep campaign packs: 13,487 COCO/Open Images photos and 30,576 public-domain-like,
  CC BY and Open Images rows); their manifests are listed below.
- **Replacements.** For the lossless rebuild, 202,909 photos that could no longer be fetched at full size were
  replaced by similar images of the same source, split and licence class (PASS 194,297; Open Images 6,464; COCO 2,148).
  2,056 COCO rows had no admissible replacement.
- **By licence class (used):** CC BY family 1,225,738 (incl. MEVA); public-domain-like 1,618; MIT 14,669; CC0 assets
  3,579.
- **Current-licence re-checks** cover every COCO, Open Images and CommonCatalog row (629,551 photos). PASS rows carry
  the licence of the PASS dataset record and were not re-checked against today's Flickr pages.
- Every count is derived from the pack receipts, the checkpoint records and the trainer's own per-family log
  (internal release record).

## Evaluation-only sources (never trained on)

| Set | Source | Licence |
|---|---|---|
| **Sealed final holdout** (RightWayUp 1.0) | New COCO holdout: 2,201 COCO val2017 photos; new Open Images holdout: 2,505 Open Images validation photos (fixed-order ranks from 8,000); R-LiViT final locations 6 and 7 (RGB + thermal, Zenodo 16356714); Aalborg long-term thermal drift (150 images, one camera) | Flickr licences per image (COCO / Open Images listing); R-LiViT CC BY 4.0; Aalborg CC BY 4.0 |
| Held-out CCTV and scenes (24 Sep sets, re-used once for the release) | MEVA test set, frozen split (6 cameras incl. thermal G474); clean scene test (DIODE scene 12, MEVA G339, Poly Haven renders) | CC BY 4.0 / MIT / CC0 assets |
| MEVA test set, development split | 7 further MEVA cameras incl. thermal G479 | CC BY 4.0 |
| Fresh DIODE | DIODE validation scenes 19–24 | MIT |
| Woehrer 2026 benchmark and fair photos | 1,030 COCO 2014 val photos, five-seed protocol and a neutral-crop version | Flickr licences per image; pixels not redistributed |
| Cue-free realistic photos | 1,856 Open Images validation development photos re-rendered from 2× originals | CC BY 2.0 (per image) |
| Open Images photo test (older panel) | Open Images validation ranks 0–3999 (3,345 eligible) | CC BY 2.0 / CC BY 4.0 |
| Calibration (thresholds only) | MEVA G419, G299, G330; Poly Haven calibration split; Open Images validation ranks 4000–7999 | CC BY 4.0 / CC0 / CC BY 2.0 |

## Backbone lineage

DINOv2 ViT-S/14 and ViT-L/14 (Meta, Apache-2.0; `timm/*_patch14_dinov2.lvd142m`, exact revisions and SHA-256 in
`RECIPE.md`). DINOv2 was pretrained on LVD-142M, whose image-rights trail Meta does not warrant. ORTUS AI accepted
this under the explicit Apache-2.0 weight grant (decisions of 7 and 23 Sep 2026). It is disclosed here rather than
hidden.

## Manifests

**Licence of the manifests.** Manifests © ORTUS AI SOFTWARE LIMITED, CC BY 4.0. They contain identifiers and
metadata from COCO (CC BY 4.0), Open Images (CC BY 4.0), PASS (CC BY 4.0) and CommonCatalog. Each photo remains under
its author's licence, as recorded at `checked_utc`. (Owner decision, 25 Sep 2026.)

| Manifest set | Rows | Status |
|---|---|---|
| `manifests/training/<family>.jsonl.gz`: the grid-free packs of the release, 23 families (the table above) | 1,251,672 (1,245,604 used; 6,068 benchmark near-duplicates excluded) | published: per-row files in the Hugging Face model repository under `manifests/training/`; `README.md`, `SUMMARY.json`, `SHA256SUMS` and the standard-library checker `verify.py` in both repositories. No gaps in licence or attribution; the 3,586 render rows have no public URL (the renders are ORTUS AI's own) |
| `manifests/release-candidate-2026-09-24/training/{pass,pass2,coco,coco2,oi,hybrid,diode2,meva2}.jsonl.gz` (24 Sep packs, JPEG-stored; sources of the earlier checkpoints) | 673,690 (664,340 used; 9,350 benchmark near-duplicates excluded) | published; 0 rows missing attribution or check date (`SUMMARY.json` there) |
| 27 Sep campaign packs `fresh_coco`, `fresh_oi`, `coco_pd`, `coco_by`, `oi_train2` | 44,063 | built in the same row form; not published (their images are in the grid-free packs above, some replaced) |

Row fields: `id, family, group, base_roll_cw, dhash, used_in_training, dataset, dataset_licence, dataset_licence_url,
locator, attribution, label_authority` (+ `supervision` for hybrid rows); the release's manifests add `licence_class`,
`derivation`, `replaces`, `replacement_reason` and `stored_sha256` (`manifests/training/README.md`). PASS geo-coordinates are deliberately not
copied. The Blender renders of Poly Haven assets (ORTUS AI, CC0 assets) are identified by render file and hash;
rebuilding them needs the published renders. If the renders are published under CC BY 4.0, that licence covers only
ORTUS AI's own contribution; the underlying Poly Haven assets remain CC0. Evaluation-set definitions (IDs, seeds,
angles) of the held-out sets are in `manifests/evaluation/`.

## Licence notes and disclosed soft spots

- **PASS: "research" wording vs CC BY 4.0.** The PASS website and its Zenodo record (6615455) make the dataset
  available "for commercial/research purposes" under CC BY 4.0. The PASS datasheet, as recorded in our internal
  rights review (not re-verified), describes the dataset as "meant for research purposes only". We rely on the
  CC BY 4.0 grant, cite the PASS paper and publish per-image attribution, and we disclose the tension here.
- **"No known copyright restrictions" is a statement, not a licence.** It is an institution's statement (Flickr
  Commons, national archives). These rows sit in their own family so they can be dropped alone.
- **US Government Work** is public domain in the US and may be protected abroad (7 rows of the
  COCO rescues, 2 in Open Images V7).
- **Licence checks are snapshots.** Flickr licences were read at fetch time; a photo whose page no longer exists
  (HTTP 404) was excluded rather than admitted on its historical listing.
- **New licence classes** (public-domain-like rows and CommonCatalog) were admitted under the owner's rule of 27 Sep
  2026 (CC BY any version, public-domain-like); the external licence reviews of 25 Sep did not cover them.
- **Do licences attach to the weights?** ORTUS AI's position is that the weights do not contain copies of training
  images: the model outputs only a rotation angle and a confidence. Per-image attribution in `manifests/` is provided
  in case any licensor or court takes a different view. This is not legal advice.
- **MEVA.** CC BY 4.0, recorded with consenting actors. The licence does not cover publicity or privacy rights, so
  we don't show MEVA frames with identifiable people in marketing material; where frames appear, faces are blurred
  and the credit line states the changes.

## Removal requests

Photographers, people depicted in an image, and licensors can ask us to remove an image, or its attribution, by
emailing support@ortusai.io with the image URL or manifest ID. We acknowledge within 5 business days, remove the row
(or only its attribution, if asked) from the next manifest release, log the removal in `REMOVALS.md`, and exclude
the image from all future training. Published weights are not withdrawn unless required by law or by a licensor's
valid claim. Full process: [`TAKEDOWN.md`](TAKEDOWN.md).

## Known limitations of the data

- Photo sources are Flickr-heavy (Western, consumer photography).
- CCTV comes from one campus dataset (MEVA, US), 10 training cameras (2 thermal).
- Few top-down or aerial views. Straight-down (nadir) imagery has no defined "up".
- Thermal imagery is limited to two MEVA thermal cameras in training; the unseen thermal test cameras show the widest
  spread of accuracy.
- Near-duplicate screening by dHash does not cover MEVA, DIODE, Poly Haven, R-LiViT or Aalborg frames (these are
  separated by camera, scene and asset lists), and it does not catch different shots of one event by one author
  (author caps of 20 and 60 limit this).
