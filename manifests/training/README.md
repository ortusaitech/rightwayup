# RightWayUp 1.0 training-data manifests

These files list every image used to train the RightWayUp 1.0 release models (Max, Fast, Pico and Nano;
Balanced and Pro are built from Max and Fast). There is one JSON row per training image. Each row records the image's
source dataset, its licence and licence URL, a locator for the public original and, where the licence requires it, the
author credit. The files contain no image bytes.

**Where the files are.** The 23 per-row files (`<family>.jsonl.gz`, about 240 MB) are in the Hugging Face model
repository [`ortusai/rightwayup`](https://huggingface.co/ortusai/rightwayup/tree/main/manifests/training), folder
`manifests/training/`, next to copies of this README, `SUMMARY.json`, `SHA256SUMS` and `verify.py`. The GitHub
repository holds only these four small files. To check the manifests (Python 3 standard library only):

```bash
huggingface-cli download ortusai/rightwayup --include "manifests/training/*" --local-dir rightwayup-hf
python3 rightwayup-hf/manifests/training/verify.py
# or, from a GitHub checkout: python3 manifests/training/verify.py rightwayup-hf/manifests/training
```

`verify.py` reads the folder it is in, or the folder given as its argument, and prints `ALL CHECKS PASS`.

The manifests of the 24 Sep release candidate (8 packs) are in the GitHub repository under
`manifests/release-candidate-2026-09-24/training/` and are unchanged.

## Counts

| | Rows | Used in training |
|---|---|---|
| 23 grid-free families (Max, Fast, Pico) | 1,251,672 | **1,245,604** |
| of which photos (PASS, COCO, Open Images, CommonCatalog) | 1,229,590 | 1,223,607 |
| DIODE scans / MEVA CCTV frames / Poly Haven renders | 14,746 / 3,750 / 3,586 | 14,669 / 3,749 / 3,579 |
| Nano: the first 16 families only | 711,143 | **705,075** |

`used_in_training` is `false` for 6,068 rows. The trainer drops these rows when it loads the data, because each is a
near-duplicate (dHash ≤ 6 bits) of one of the 1,030 origin-benchmark photos. The dropped ids are listed in the trainer's
`exclude.json`. By family, the dropped rows are PASS 5,944, COCO 39, DIODE 77, MEVA 1 and Poly Haven 7.

**Distinct images.** Some rows hold the same picture. Counting rows that share a Flickr photo id or have byte-identical
stored training images as one image gives 1,250,283 distinct images, of which **1,244,222 were used in training** (1,382
fewer than the used rows). There are 1,351 such groups, all of them photos:

- PASS rows that are also in Open Images or COCO (the same Flickr photo in two datasets): 880 groups.
- Two or more PASS rows with the same Flickr photo or identical stored bytes: 165 groups.
- COCO rows that are also in Open Images: 25 groups.
- Byte-identical uploads under different Flickr photo ids inside CommonCatalog: 280 groups (259 of them by the same
  author).
- Open Images test: 1 group.

`SUMMARY.json` → `duplicates` gives the counts.

By licence class (used rows):

| Class | Licences (as recorded per image) | Used |
|---|---|---|
| `cc-by` | CC BY 2.0, CC BY 4.0 (Flickr photos, PASS, MEVA) | 1,225,738 |
| `pd-like` | No known copyright restrictions, Public Domain Dedication (CC0), Public Domain Mark, United States Government Work | 1,618 |
| `mit` | MIT (DIODE) | 14,669 |
| `cc0-assets` | CC0 Poly Haven assets, rendered by ORTUS AI | 3,579 |

## Which model saw which family

| Model | Families | Training record |
|---|---|---|
| Max (M3-SOUP8, uniform soup of 8 runs) | all 23 (4 of the 8 soup parts used 22: without `gf_cc4`) | mix string in each run's checkpoint `args` |
| Fast (GF-SV3-kdM3) | all 23 | checkpoint `args` |
| Pico (PICO-N70S80 = 0.2 × PICO-N70L + 0.8 × PICO-N70C) | all 23 | checkpoint `args` |
| Nano (GF-SOUP-SV2 = soup of GF-SV2-A1/A2) | first 16: `gf_pass` … `gf_coco_by` (not `gf_cc*`, `gf_oi7*`) | checkpoint `args` |

The families were trained with `--data-frac 1` (all rows) and are listed in the same order in `verify.py` (`FAMILIES`).

## Files

- `<family>.jsonl.gz`: 23 files, one JSON object per line. The gzip streams are deterministic (mtime 0).
- `SUMMARY.json`: per-family, per-licence, per-licence-class, per-dataset and per-tier counts, plus duplicate groups and
  gaps. `verify.py` recomputes it.
- `SHA256SUMS`: hashes of the 23 manifests and `SUMMARY.json`.
- `verify.py`: recomputes `SUMMARY.json` from the manifests. It checks `SHA256SUMS`, id uniqueness and the licence-class
  table, and compares the totals with the published counts. It needs only the Python 3 standard library:
  `python3 verify.py`.
- The build log (with the sha256 of each input record file) is kept with the release records.

## Row fields

The rows use the release candidate's row form (`manifests/release-candidate-2026-09-24/training/`, built by
`rotlab/export_manifest.py`), plus five fields that are new in this release:

| Field | Meaning |
|---|---|
| `id` | Image id in the training pack (unique across all 23 files). |
| `family`, `group` | Training family (the file name) and the sampling group inside it (scene, camera, source shard). |
| `base_roll_cw` | The image's upright reference angle in degrees clockwise (0 for photos). |
| `weight` | The per-image weight recorded in the pack row, carried over from the release candidate's packs. The release trainer does not read it. |
| `dhash` | 64-bit difference hash of the stored image (the value the near-duplicate screens use). |
| `used_in_training` | `false` = dropped by the origin-benchmark near-duplicate filter (see above). |
| `dataset` | Source dataset. |
| `dataset_licence`, `dataset_licence_url` | Licence of the dataset. Flickr-hosted COCO, Open Images and CommonCatalog images are licensed per image, so this field is `<per-image licence> (per image)`. PASS has a CC BY 4.0 dataset grant. |
| `licence_class` | *(new)* `cc-by`, `pd-like`, `mit` or `cc0-assets`, taken from the recorded licence string by the fixed table `LICENCE_CLASS` in `verify.py`. |
| `locator` | Where to fetch the original (see below). |
| `attribution` | For Flickr photos: `author`, `author_url`, `title`, `source_url` (the photo page), `licence`, `licence_url` and `checked_utc`, which is when the current Flickr licence was read (Flickr oEmbed). PASS rows carry `author`, `licence`, `licence_url` and `source_url` (the Flickr file fetched), and no check date. `null` for DIODE, MEVA and Poly Haven, whose credit is at dataset level (see NOTICE). |
| `label_authority` | Where the upright reference comes from, when the pack row records it (DIODE, MEVA). |
| `derivation` | *(new)* As recorded: `kept`, `re-derived`, `substituted` or `downscaled-pack-copy`. `replacement` marks a photo that replaced one that could no longer be fetched at full resolution. `null` = no derivation recorded; the row is the family's own selection. |
| `replaces`, `replacement_reason` | *(new)* The id of the release candidate's row that this row replaced, and the reason (for example `gone`, `too_small`, or a held-out scene or camera). |
| `stored_sha256` | *(new)* sha256 of the PNG as stored in the training pack. A sample of 920 rows (40 per family) was re-hashed against the pack bytes, and all 920 matched. |

Locator keys by dataset:

- **PASS:** `pass_hash`, `url` (PASS page), `fetched_url` + `fetched_sha256` (the Flickr file the stored PNG was made
  from), `yfcc_photoid` (replacements) and `pass_tar` (original rows, as in the release candidate's manifests).
- **COCO:** `coco_id`, `coco_file`, `url` (images.cocodataset.org), `sha256_original` where recorded, `flickr_id`,
  `fetched_url`, `fetched_sha256`.
- **Open Images:** `openimages_id`, `subset` (`test` or `train`), `url` (CVDF mirror), `sha256_original`, `flickr_id`,
  `flickr_original_url`, `fetched_url`, `fetched_sha256`. Where present, `oi_rotation` is Open Images' **recorded display
  orientation**: Google's 2018 automatic match to the Flickr display. It is not a human check.
- **CommonCatalog:** `hf_dataset`, `dataset_revision`, `parquet_path`, `row_group`, `row`, `dataset_row_sha256`,
  `yfcc_key`, `flickr_id`, `url` (the Flickr original), `sha256_original`, `yfcc_licence`.
- **DIODE:** `diode_file` (path inside the official train archive), `official_archive`, `sha256_original` or
  `archive_file_sha256`, `url`.
- **MEVA:** `meva_object_key`, `url` (public MEVA bucket), and the frame as `frame_index` + `fps`/`frame_rate`, or as
  `frame_at_fraction_of_duration` / `frame_at_seconds` / `ffmpeg_ss`.
- **Poly Haven renders:** `render_file`, `render_sha256`, `render_set`, `archive_member`, `archive_file_sha256`.

## Known limits (stated, not hidden)

- **Poly Haven renders (3,586 rows) have no public URL.** They are ORTUS AI renders of CC0 Poly Haven assets and are not
  published. The locator identifies the render by its file name and hash. For `gf_poly_direct` (120 rows) the original
  render file is lost: training used a 139 × 83 px copy of the 24 Sep pack image (see `derivation`).
  `SUMMARY.json` → `gaps.no_locator_url` counts these rows. No other gaps were found: every row has a dataset licence and
  URL; every Flickr photo row has an author, a per-image licence and licence URL, and a source URL; and every non-PASS
  photo row has a licence check date.
- **PASS licences come from the PASS release metadata**, not from today's Flickr page: `pass_metadata.csv` gives
  `licensename` = "Attribution License" for all 600,000 rows. PASS replacements record the same licence in the pack row.
- The ids of the COCO and Open Images rows name the dataset images, and `url` points to the dataset copy. The stored
  training PNG was made from the file in `fetched_url`:
  - For PASS, `gf_coco*`, `gf_fresh_*` and `gf_oi`, that file is the Flickr copy, and a kept row had to match its
    previous pack image within dHash 8 bits.
  - For `gf_oi_train2` and `gf_oi7*`, it is the Open Images CVDF file.
- Identical or same-photo rows across families are listed under `SUMMARY.json` → `duplicates`. They were trained as
  separate rows.

## How the manifests were built

`rotlab/export_manifest_release.py` builds the manifests from these records, read on 30 Sep 2026:

- the grid-free pack rows (`rotation-data/rotlab/sources/gf_*.jsonl`). Their sha256 values, listed in the build log, equal
  the `jsonl_sha256` values in the 23 pack receipts.
- the trainer's `exclude.json`.
- PASS `pass_metadata.csv`: hash, nickname and licence only. The geo-coordinates are deliberately not copied.
- the release candidate's manifest rows (`manifests/release-candidate-2026-09-24/training/`), joined by id where a
  grid-free row kept a release-candidate row whose source record is no
  longer on the volume:
  - `gf_coco` attribution (3,470 rows; COCO attribution record checked 10 Sep)
  - the licence check date of the kept `gf_coco2` (8,064) and `gf_oi` (27,385) rows
  - DIODE file paths of the kept `gf_diode` (4,463) and `gf_diode2` (2,477) rows
  - MEVA clip and frame of the kept `gf_meva` (950) and all `gf_meva2` (550) rows
  - the Poly Haven render locators (3,586)

The licence and attribution values come from those records unchanged: none is inferred. Build commands:

```
python3 rotlab/export_manifest_release.py --src <copies of the records> \
    --v1 manifests/release-candidate-2026-09-24/training --out manifests/training
python3 manifests/training/verify.py
```
