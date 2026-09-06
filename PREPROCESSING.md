# Preprocessing of the Ghent charter photographs

Living methods note for the paper. Every decision that changes a pixel, a
filename, or the set of pages that enter the mole archive is recorded here, with
the counts and parameters that were actually used. Fill the *Recorded* rows as
each stage is signed off; do not rewrite a stage after the fact — add a dated
erratum instead.

Companion to `hands-leroy/rationale.md` (how writer labels are attached later).
This file covers **images only**: selection, quality control, annotation, and
binarization. Writer identity is out of scope until the image set is frozen.

Mole convention the dataset must satisfy: a flat folder of images plus a
`labels.csv` whose first two columns are `filename,hand_id`. Extra columns are
ignored by mole. Unlabelled images are kept as gallery distractors.

---

## 0. Source material

**Collection.** Photographic archive of 13th-century administrative charters,
digitized at Ghent (Anouck Kuypers). The photographs sit in dated folders under
`images/archive-original/`. Each charter is typically a pair:

| suffix | meaning | role in this pipeline |
|---|---|---|
| `…o.jpg` | *oorkonde* / face | the text-bearing side; candidate for the dataset |
| `…m.jpg` | *memoriaal* / dorse | archival notes on the back; excluded from the writer-identification set |

Limited per-photo metadata (`images/metadata.xlsx`): running number, filename,
archive, fonds, shelfmark, date fields, folder name. There is **no** join from
a photograph to a Gysseling number or a Leroy hand-group — that join is a later
textual-matching stage (`hands-leroy/`).

**Queries (not part of the archive proper).** A handful of extra photographs
live in `images/additional/` (Büdingen fragment recto/verso; Genois 1327;
RA-800r). These are the literary-hand queries the retrieval experiment will
search *for*. They are preprocessed with the same recipe so they occupy the
same pixel regime as the gallery, but they are not members of the charter
dataset.

### Inventory at the start of this work

| artefact | count | notes |
|---|---:|---|
| original JPGs | 2816 | 1408 recto + 1408 dorse, 13 date-range folders |
| unique photo numbers | 1408 | one `o`/`m` pair per number |
| original JPGs | 2816 | `images/archive-original/` |
| baked rectos | 1408 | `images/pages-recto/` |
| zone crops | 1408 | `images/pages-zoned/` |
| stretched gallery | 1304 | `images/pages-zoned-stretched/` (`main_document=1`) |
| query photographs | 6 | `images/additional/` → `queries-zoned/` → `queries-zoned-stretched/` |

**Erratum 2026-09-06.** Removed folders that are not on this recipe:
`images/cropped/`, `images/archive-leroy/`, `images/pages-verso/`, and the
old `archive-recto` copy. Dorses remain as `…m.jpg` under `archive-original/`
and as rows in `data/manifest.csv`.

---

## Design principles (apply to every stage)

1. **Coverage over tightness.** Dropping a page, or clipping text, destroys
   writer signal that nothing downstream can recover. Extra background is cheap:
   mole's contrast foreground filter already ignores blank parchment at embed
   time. Prefer false positives (keep / over-crop) and correct them by hand,
   rather than aggressive automatic rejection.
2. **One frozen image set.** Near-duplicate decisions, polarity, and zone
   polygons are part of the dataset, not runtime flags. A later mole user
   should be able to `mole embed` the released folder without re-running our QC.
3. **Polarity is an image property, not a model flag.** Released pages are
   dark ink on light support. Mole's training stack still inverts at embed
   (`--invert`) because the ViT was trained white-on-black; that is a model
   convention, not a property of the scans.
4. **Log every exclusion.** A page that leaves the set gets a reason in the
   manifest (`dorse`, `near-duplicate-of`, `blank`, `not-a-charter`, …). The
   paper reports both the raw photograph count and the released count.
5. **QC is visual.** Automatic scores propose; a contact sheet or Label Studio
   view decides. No stage is signed off from a histogram alone.

---

## Stage 1 — Select foreground (text-bearing) pages

**Problem.** The digitization captured both faces of each photograph. Only the
text-bearing face is a writer-identification document. Dorses, empty mounts,
colour charts, and non-charter shots must not enter the gallery.

**Existing code.** `code/01-preprocessing.ipynb` already implements the naming
rule: keep files whose name ends in `o.jpg`, convert JPG→PNG, bake EXIF
rotation with `PIL.ImageOps.exif_transpose`. That recovered 1408 pages from
2816 photographs, with a perfect `o`/`m` pairing in every folder (no leftover
`other` suffixes).

**Procedure we will actually run (paper version).**

1. Recursively list every `*.jpg` under `images/archive-original/`.
2. Partition by suffix: `o` (keep), `m` (exclude as dorse), anything else
   (flag for manual inspection). Current inventory: 0 files in the third bin.
3. Write a **page manifest** (`data/manifest.csv`) with one row per original
   photograph: `photo_id`, `original_path`, `side` (`recto`/`dorse`/`other`),
   `keep` (bool), `reason`. This is the exclusion log for the paper.
4. Materialise keepers as PNG with EXIF rotation applied, original photo
   number preserved in the filename (`{n}o.png`). Do not yet crop or
   binarize.
5. Spot-check a contact sheet of a random sample (and every `other` / unpaired
   `o` or `m`) to confirm the suffix rule matches the actual face of the
   charter. If the rule fails on a page, override `keep` by hand and record
   why.

**Output.** `images/pages-recto/` (flat PNG) + `data/manifest.csv`.

**Recorded.** *not yet run under this recipe — legacy notebook produced 1408
rectos, 0 unpaired files.*

---

## Stage 2 — Sample one image per document series

**Problem.** The same charter can occupy several photo numbers: consecutive
frames of one leaf, or a later re-shoot (a gap in `volgnummer`). Near-duplicates
inflate retrieval scores (a page retrieves its twin) and bias writer counts.
Sibling *faces* (`o` vs `m`) are already handled by Stage 1.

**Locked keep rule.** When metadata identifies a series — same `archief` +
`fonds` + `signatuur` + calendar date (`jaar`-`maand`-`dag`) — keep only the
**first** image (lowest `volgnummer`). Dropped photos are logged
(`series_of=<first>o`), not deleted from disk. Different dates under the same
shelfmark stay **separate** series (a dossier, not one leaf). Photos with no
`signatuur` are not collapsed here.

Identity fields live on the dorse row; join on `volgnummer`. Notebook:
`code/02-sample-series.ipynb`.

**Recorded 2026-08-30.** 9 series, 32 photos → keep 9, drop 23. Gallery
1408 → 1385. Largest: Rijsel B `247` on 1299-03-27, keep `1360o`, drop
`1361–1363` and `1400–1405`. The same shelfmark on 1299-03-11 is a second
series (`1364o` kept, `1365o` dropped). Spreadsheet: `data/series_sample.csv`.
QC: `outputs/stage2_series_qc.html`.

A later perceptual-hash pass can still catch photographic twins among the
1218 rectos that have no shelfmark. That pass is Stage 2b below.

---

## Stage 2b — Perceptual-hash twins

**Problem.** Most unsigned photos are unique leaves issued on the same day
(same archive, no shelfmark). A few are a second frame of one leaf. Those
twins were invisible to Stage 2.

**Procedure.**

1. 256-bit difference hash on every baked recto (`pages-recto/`, long side
   of the hash grid 16). Pairwise Hamming distance.
2. Calibration: the 72 metadata-series pairs from Stage 2 sit at Hamming
   **49–143** (median 118). They are re-shoots / different crops, not pixel
   copies. The hash cut must be tighter than that.
3. Candidates: both still `main_document`, Hamming ≤ **32**, not already a
   Stage 2 pair. Connected components of those pairs are **clusters**.
4. Visual QC: `outputs/neardup_review.html`. Click a cluster (or Space) to
   mark **same leaf** (keep lowest `volgnummer`). Unmarked = keep all.
   Export `neardup_decisions.csv`. Do not drop until that file is applied.

Script: `code/04_near_duplicates.py`. Notebook: `code/04-near-duplicates.ipynb`.

**Recorded 2026-08-30.** 7 clusters, 15 pairs, 19 photos. Review: all 7
marked same-leaf (`outputs/neardup_decisions.csv`). Applied: drop 12
extras (`reason=neardup_of=<first>o`), keep PNGs on disk. Gallery
1385 → 1373. Apply script: `code/04_apply_neardup.py`.
Lists: `data/neardup_pairs.csv`, `data/neardup_clusters.csv`.

**Recorded 2026-08-31.** Visual QC during zone correction found more
same-leaf / same-document groups the hash pass missed, plus one keep
retarget (556o, not 554o). Applied `data/review_neardup_extra.csv` via
`code/04_apply_review_errata.py`. 27 additional gallery drops, 1 restore;
gallery 1373 → 1347. PNGs stay on disk. Already-logged series
(339/340, 1259/1260, 1403–1405) were left as they were.

**Recorded 2026-09-04.** Two more same-document pairs from visual QC of
the stretched gallery. Keep `1371o` (drop `1370o`; retarget `1372o` from
`1370o` to `1371o` — the hash cluster had kept 1370o). Keep `1399o`
(drop `1398o`). Same apply script; recto PNGs stay on disk. Gallery
1347 → 1345. Dropped ids removed from `images/pages-zoned-stretched/`
only.

---

## Stage 3 — Correct inverted colouring

**Problem.** A subset of the photographs is polarity-reversed (light ink on a
dark ground): microfilm-style inversion, or a scanner setting. Sauvola
thresholding assumes dark ink on a light support (`pixel > local threshold →
white`). Running it on a negative produces a photographic negative of the
writing and poisons every later stage.

**Procedure.**

1. Score every baked recto (`pages-recto/`): Otsu split, dark-class fraction.
   Pages with dark fraction ≥ 0.50 are **candidates** (110 on this archive).
   That list mixes true microfilm negatives with dark mounts; it is a review
   queue, not an automatic invert.
2. Open `outputs/polarity_review.html`: as-stored colour | **grayscale invert**.
   Click a row (or Space) to mark invert; unmarked = keep. Marks live in the
   browser until you Export CSV (`polarity_decisions.csv`). Eight normal pages
   sit at the bottom as controls.
3. Apply the CSV in a later step (grayscale invert on the PNG, `inverted=1` in
   the manifest). Do not invert at embed time.

Notebook: `code/03-review-polarity.ipynb`. Candidate list:
`data/polarity_candidates.csv`.

**Recorded 2026-08-30.** Round 1: 51 invert / 59 leave (of 110 candidates).
Grayscale invert applied to those 51 PNGs; all 51 now score dark_frac < 0.50.
Round 2 sheet: `outputs/polarity_review_round2.html` (spot-check the 51; 59
remaining majority-dark pages, mostly dark mounts they already skipped).
No further inversions.

**Recorded 2026-08-31.** Zone review caught 19 further negatives the Otsu
queue missed (Den Haag 1290–1300 II: 1245o, 1247o, 1251o, 1259o, 1271o,
1277o, 1279o, 1280o, 1282o, 1286o, 1289o, 1291o, 1292o, 1298o, 1299o,
1300o, 1304o, 1309o, 1310o). Same grayscale invert, once.
List: `data/polarity_round3.csv`.

**Output.** Polarity-corrected PNGs in `images/pages-recto/` +
manifest column `inverted`.

**Colour cast (same day, after polarity).** A subset of *non*-inverted
rectos still had a strong cyan/blue hue (the look of an RGB invert of a
magenta microfilm scan). Detector: downsample to long side 256, chroma
> 1 and \(B - (R+G)/2 > 10\). **31 rectos** matched; converted to
grayscale RGB **without** flipping polarity (`code/03_apply_grayscale.py`).
List: `data/grayscale_blue_cast.csv`. The same detector on all 1408 versos
found **0** hits (backs are near-white, slightly warm). Do not re-run
Stage 1 after this step: baking again from `archive-original/` would
overwrite polarity and destaturation.

Manifest: `inverted=1` on 70 rectos (51 + 19); `grayscale=1` on 101
rectos (70 inverted + 31 destaturated).

---

## Stage 4 — Main text zone (Kraken BLLA, largest region)

**Problem.** Each photograph is a charter on a table or mount: rulers, colour
charts, hands, bindings, seals, and empty parchment surround the writing.
Writer identification should see the **main text block**, not the furniture.

**Detector.** Same as the legacy notebook: Kraken `blla` (`blla.mlmodel`).
Keep **only the largest region** (axis-aligned bbox area). Other regions are
logged in `detections` but not cropped — two fragments in one photo stay one
charter. No union of boxes (that is mole's YOLO default; we do not use it
here). Do **not** skip small or dark crops, and do **not** binarize (Stage 5).

Pages with no region fall back to the whole page (`fell_back: true`).

**Coordinates (publishable at every stage).** Each zone is stored twice:

| space | file | box |
|---|---|---|
| baked recto (EXIF already in the pixels) | `images/pages-recto/{n}o.png` | `bbox` `[x0,y0,x1,y1]` + `polygon` |
| original photograph (stored JPG pixels) | `original_path` | `bbox_original` (inverse of EXIF Orientation) |

Crops (`images/pages-zoned/`) are the polygon masked onto white, then cut to
`bbox`. The crop is grayscale RGB, polarity already corrected. These files
are the unstretched zone originals.

**HTR input (2026-08-31).** Robust percentile stretch, interior polygon only:
p2 → 20, p98 → 255. Sibling folder, not in place:
`images/pages-zoned-stretched/`. White AABB fill stays 255. Series extras and
near-duplicates (`main_document=0`, 63 frames) are **not** copied here — HTR
sees the 1345 gallery pages only. Full 1408 zone crops remain in
`pages-zoned/` (logged, not deleted). Review: `outputs/stretch_review.html`
(right-hand column signed off). Stats: `data/stretch_stats.csv`.
Script: `code/06_contrast_stretch.py`. Sauvola is **not** applied here; it
remains a later mole decision (Stage 5).

**Review.** `outputs/zone_review.html`: page with boxes (green = kept, orange
= discarded regions) beside the cutout. Click / Space marks **correct later**.
Unmarked = accept. Export `zone_decisions.csv`.

Script: `code/05_blla_zones.py` (resume via `data/zones_blla.jsonl`).
Notebook: `code/05-blla-zones.ipynb`. Canonical tables: `data/zones.json`,
`data/zones.csv`.

**Recorded 2026-08-30.** Full run: Kraken BLLA, device MPS, 1408 unique
pages in `data/zones_blla.jsonl` (duplicate jsonl rows from a second
overlapping job were collapsed last-write-wins). Legacy combined this
step with Sauvola and dropped 10 pages; that skip list is not reused.

**Recorded 2026-08-31.** All 1408 boxes corrected in Label Studio project
`sluis`. Dump: `python code/05_ls_export.py` → `data/ls_zones_export.json`
(or Export → **JSON-MIN** from the LS UI). Applied with
`python code/05_apply_ls_zones.py --export data/ls_zones_export.json`:
percent boxes mapped onto full-res `pages-recto/`, `bbox` / `polygon` /
`bbox_original` rewritten, `pages-zoned/` recropped. Empty LS tasks would
be whole-page fallbacks, not Skip.

**Recorded 2026-09-04.** Two pages still had a near-full-plate MainZone:
`1190o` (LS task 1670) and `1277o` (task 1757). BLLA’s largest region was
the photograph (charter + hanging seal + mount); the first LS pass kept
that prediction. Boxes redrawn in Label Studio onto the parchment body
only. Re-exported sqlite and recropped those two; restretched into
`images/pages-zoned-stretched/`. New bboxes: 1190o `[305, 261, 2126, 540]`,
1277o `[529, 129, 1872, 699]`.

Note for the paper, not a reason to skip this stage: on already-binarized
charter corpora, restricting embed windows to the zone was a **null** for
retrieval mAP (`mole/FEATURES_RESULTS.md`). We still zone-crop the *released
images* so the public dataset is a clean text image rather than a photograph
of a table. That is a dataset-quality choice, not a retrieval-lever claim.

---

## Stage 5 — Binarize

**Problem.** The photographs mix colour, lighting, stains, and (after Stage 3)
a common polarity. Writer embeddings should not see parchment tone or the
camera's white balance. Adaptive binarization removes that, once, into a
cached copy.

**Procedure.** Mole's Sauvola implementation (`mole prep --binarize sauvola`):

\[
t(x) = m(x)\,(1 + k\,(s(x)/R - 1))
\]

with a local window on the grayscale page. Pixels above \(t\) become white
(support), below become black (ink). Defaults in mole: window 25 px, \(k =
0.2\), \(R = 128\). The legacy sluis notebook used window 51, \(k = 0.3\) on
Kraken crops — **do not copy those numbers blindly**. Tune on a QC sheet of
original | binarized page | native-resolution ink crop, then freeze.

Recommended command shape (parameters filled in when Stage 5 is signed off):

```bash
mole prep images/pages-zoned --binarize sauvola \
    --binarize-out images/ghent-bin \
    --sauvola-window W --sauvola-k K \
    --max-side S
```

`--max-side` is a compute cap (longest side, never upsamples), not a
handwriting-scale control. Script-scale normalisation (`--normalize-scale
profile`) is a **separate** decision: it equalises the x-height in pixels
across pages so a 224 px embed window sees a comparable amount of script.
Use it if this dataset will sit in the same index as the other mole archives
(leroy-bin / utrecht / …); skip it if the release should preserve original
camera resolution. Record the choice, the target module, and `scale.json`.

`labels.csv` is carried over with extensions rewritten to `.png`. `zones.json`
is **not** carried (coordinates would be wrong after `--max-side`). If zones
are still needed on the binarized copies, either binarize without downscaling
or re-detect / re-map.

**QC.** Contact sheet; reject pages that are >50 % black (failed polarity or
failed zone) or nearly empty. Those go back to Stage 3 or 4, not into a second
ad-hoc threshold.

**Output.** The mole dataset folder: binarized PNGs + `labels.csv` (hand_id
still empty at this point) + this preprocessing log.

**Recorded 2026-09-05.** On `mike`, from the 1304 stretched gallery
(`mole/data/leroy`): `mole prep data/leroy --binarize sauvola --binarize-out
data/leroy-sauvola` (mole defaults: window 25, \(k=0.2\); no second stretch;
no `--max-side`; no scale-normalise). `labels.csv` carried over. SSL retrieval
A/B (grayscale vs this Sauvola copy) is in `RETRIEVAL_SSL.md` — Sauvola SSL
epoch 15 is the shipped backbone.

---

## What this file is not

- **Writer labels.** Leroy hand-groups via HTR → Gysseling match live in
  `hands-leroy/rationale.md`. They are attached *after* the image set is
  frozen, as `hand_id` in `labels.csv`, with unmatched pages left blank
  (gallery distractors).
- **The retrieval experiment.** Query = Büdingen / Sluis literary hand;
  gallery = this dataset. That is the application paper, not preprocessing.
- **Training mole.** The released folder is consumed by mole v0.1.0 as any
  other archive (`mole embed`, `mole eval`). We do not fork mole for this
  dataset.

---

## Stage log (sign-off)

| stage | date | n in | n out | parameters / notes |
|---|---|---:|---:|---|
| 1 foreground pages | | 2816 | | suffix `o.jpg`; EXIF rotate |
| 2 sample series | 2026-08-30 | 1408 | 1385 | same shelfmark+date; keep first `volgnummer`; 9 series, 23 dropped |
| 2b hash twins | 2026-08-30 | 1385 | 1373 | dHash 256-bit, ham≤32; 7/7 clusters same-leaf; drop 12 |
| 2c review dups | 2026-09-04 | 1373 | 1345 | visual same-doc groups; keep 556o not 554o; keep 1371o/1399o; 29 extra drops |
| 3 polarity | 2026-08-31 | 1408 | 1408 | 70 grayscale-inverted (51+19); 31 destaturated; 101 gray rectos |
| 4 text zones | 2026-09-04 | 1408 | 1408 | BLLA largest region; LS boxes applied; 1190o+1277o redrawn |
| 4b stretch | 2026-09-04 | 1408 | 1345 | p2→20, p98→255; gallery only; extras stay in `pages-zoned/` |
| HTR (labels) | 2026-08-31 | 1347 | 1347 | `google/gemini-3.7-flash` low reasoning; 23 empty then refilled; ~USD 5; see `hands-leroy/rationale.md` §3.2 |
| freeze | 2026-09-05 | 1345 | **1304** | `main_document=1` in `manifest.csv` / `zones.csv`; 104 rectos dropped (series + near-dup). This is the paper gallery. |
| 5 binarize | 2026-09-05 | 1304 | 1304 | mole Sauvola w=25 k=0.2 on stretched gallery; `data/leroy-sauvola`; see `RETRIEVAL_SSL.md` |
