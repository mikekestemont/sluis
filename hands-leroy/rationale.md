# Leroy charters → writer-identification dataset — rationale

*Working note for the `hands-leroy` sub-project of `sluis`. Explains what the
pipeline does, why, where the reliability risks are, and how the resulting data
is distilled into an additional **mole** archive (a flat image directory + a
`labels.csv`).*

---

## 1. Goal

Build a labelled dataset of **13th-century Dutch charters** for **writer
identification**, reusing an existing paleographic ground truth (Leroy's
*handengroepen*, i.e. scribal hand-groups) attached to the Corpus Gysseling.

The obstacle: the paleographic labels are keyed to **charter identity**
(Gysseling numbers), but our images are an unlabelled **photo archive** (Ghent
photo collection) whose only key is a running photo number. There is no direct
join between a photo and a Gysseling number. We recover that join *through the
text*: HTR-transcribe each photo, then match the (noisy) transcription against
the (clean) Gysseling corpus. The match hands us a Gysseling number, and the
Gysseling number hands us Leroy's hand-group.

```
photo (Ncoo.jpg)  --HTR-->  transcription.txt  --fuzzy match-->  Gysseling nr
                                                                      |
                                                          Leroy handengroepen
                                                                      |
                                                                 hand_group
```

Every arrow is lossy, so the central design tension is **coverage vs.
reliability**: how many photos we can label vs. how often that label is right.

---

## 2. Inputs

| Artefact | What it is | Count |
|---|---|---|
| `../images/archive-original/` | Original Ghent photo archive, JPGs foldered by date range. Each charter has a recto (`…o.jpg`, *oorkonde*/face, the text) and a dorse (`…m.jpg`, back, archival notes). | ~2800 photos |
| `../images/archive-recto/` | Recto-only, flattened, JPG→PNG (see `../code/01-preprocessing.ipynb`). | 1408 |
| `../images/cropped/` | Recto text-zone crops, Kraken layout-segmented + Sauvola-binarised. **This is what mole embeds.** | 1398 |
| `cg-admin-orig/*.fromdb` | Corpus Gysseling administrative charters, original `.fromdb` markup. | 2228 |
| `cd-admin-txt/*.txt` | Same corpus converted to plain text (`convert_gysseling.py`). | 2228 |
| `transcriptions/*.txt` | VLM/HTR transcriptions of the recto crops (`transcribe_charters.py`). | 1408 |
| `handengroepen_gysseling.xlsx` | Leroy's hand-group assignments, keyed by Gysseling nr. | 1384 rows |
| `metadata.xlsx` | Photo-archive metadata (photo nr, archive, fonds, date…). | 2816 rows |
| `metadata-matched.xlsx` | **Output**: `metadata.xlsx` + `match`, `match_score`, `gysseling_nr`, `hand_group`, `hand`. | 2816 rows |

Naming key: a metadata row's `bestandsnaam` is `<n>o.jpg` / `<n>m.jpg`; the
crop is `<n>o.png`; the transcription is `<n>o.txt`; the corpus file is a
zero-padded Gysseling nr like `0065AA.txt`. Only **recto** (`o`) images carry
text and are transcribed/matched.

---

## 3. Pipeline stages

### 3.1 Preprocessing — `../code/01-preprocessing.ipynb`
Copies recto photos to a flat folder (JPG→PNG, EXIF rotation baked in), then
Kraken `blla` layout analysis cuts the main text zone and Sauvola-binarises it
(`WINDOW_SIZE=51`, `k=0.3`). Crops smaller than 256 px or >50 % black are
skipped — this is why `cropped/` (1398) < `archive-recto/` (1408). A handful of
non-charter reference images from `additional/` are binarised into the same
folder (e.g. `Büdingen1r.png`); they carry no metadata and act as distractors.

### 3.2 HTR transcription — `transcribe_charters.py`
Batch transcription of the recto crops via **Claude Sonnet through OpenRouter**
(`anthropic/claude-sonnet-4`). The system prompt pins the output to Corpus
Gysseling editorial conventions: silently expand abbreviations/nasal bars,
modern Latin alphabet, lowercase, preserve line breaks and word separation, no
punctuation, `[...]` for illegible passages. Idempotent (skips already-done
files), threaded, with retry/back-off on 429s. Output → `transcriptions/`.

> These transcriptions are **deliberately unreliable**. They are a retrieval key,
> not a scholarly edition. Their only job is to be *close enough* to the true
> charter text that fuzzy matching lands on the right Gysseling number.

### 3.3 Corpus conversion — `convert_gysseling.py`
Turns `.fromdb` markup into plain text that resembles the HTR convention:
expands `<A>` abbreviation tags inline, strips PoS/lemma `<C>` tags and
metadata/structural tags, uses `<L page:line>` for line breaks, splits `+`
compounds, strips punctuation. Result: `cd-admin-txt/`, one clean text per
charter — the search targets.

### 3.4 Matching — `match_charters.ipynb`
Two-stage matcher (transcription → corpus):
1. **Retrieval** — character n-gram TF-IDF (`char_wb`, 3–5 grams, sublinear tf),
   cosine similarity, top-`K=20` candidates per transcription. Character n-grams
   are robust to HTR spelling noise.
2. **Re-ranking** — `rapidfuzz` normalised Levenshtein similarity on those ~20
   pairs. Accept the best if `match_score ≥ MIN_THRESHOLD = 0.40`.

Both texts are normalised first (lowercase, strip non-alphanumerics, collapse
whitespace). For each transcription it records the best match, its score, the
runner-up and the **margin** (best − runner-up) — a confidence proxy.

Then it joins **Leroy's hand-groups**: the corpus filename → canonical Gysseling
nr (`0065AA.txt` → `65aa`), and the nr → `hand_group` (from `handengroepen`,
`Type == 'groep'` only). Everything is written back onto the photo metadata as
`metadata-matched.xlsx`.

> **Canonical matcher:** `match_charters.py` is the script form of this stage and
> is the authoritative version — it fixes the issues found in the original
> `match_charters.ipynb` (see §5): a case/format-robust Gysseling join, the
> `margin` persisted to the output, `tussengroep` dropped, and the misleading
> `hand` column removed. It needs no API key (matching runs off the local
> `transcriptions/` and `cd-admin-txt/` text) and regenerates the spreadsheet in
> ~1 min: `python match_charters.py`.

### 3.5 mole embeddings — `../code/02-extract_embeddings.py`
For reference: a self-supervised ViT-small (DINO-style teacher checkpoint)
extracts foreground patch features from each cropped image, which are
**VLAD-encoded** against a MiniBatchKMeans codebook (`k=100`) into one
L2-normalised document vector. This is the mole representation the archive
below is meant to feed and be evaluated against.

---

## 4. Label semantics — the one label to use

**`hand_group`** — Leroy's scribal hand-group, taken only from rows where
`Type == 'groep'`. **This is the writer-identity signal**, and the only label
used downstream: 461 images across 98 groups (76 with ≥2 members).

Two things the source data does *not* give you:

- **No individual-hand signal.** The original notebook had a column named `hand`
  ("individual hand") that was in fact just the raw `Groep` value of the first
  spreadsheet row per charter, silently merging `groep` **and** `tussengroep`
  rows (hence values like `tussengroep 5`). It was misleading and is **removed**
  in `match_charters.py`. The finest *reliable* unit is the hand-group.
- **`tussengroep` (intermediate/uncertain assignments) are dropped entirely.**
  They are fuzzy by construction and never become labels — matching the intent
  of the mole dataset.

---

## 5. Reliability risks (verification notes)

Findings from reading the code and auditing `metadata-matched.xlsx`; all code
fixes below are in `match_charters.py`.

1. **Misleading `hand` column — FIXED.** The notebook's `hand` column was not an
   individual scribe (see §4). Removed; only `hand_group` is emitted.

2. **`margin` was discarded — FIXED.** The retrieval margin (best − runner-up)
   is now written to the output as a second reliability signal. In practice it
   correlates ~0.77 with `match_score`, so it adds little beyond the score gate,
   but it is there if a downstream split wants it.

3. **Letter-suffix Gysseling join — hardened, but not the culprit.** The join now
   canonicalises both sides (lowercase, drop leading zeros/apostrophes:
   `0065AA.txt → 65aa`). Investigating why only **3 of 44** matched letter-suffix
   charters get a group showed this is **mostly genuine, not a join bug**: 35 of
   those 44 simply do not appear in *either* the `groep` or `tussengroep` sheet
   (Leroy never classified them), and 6 are `tussengroep` (now correctly
   excluded). The hardening is defensive; it recovers no extra labels on this
   data.

4. **The `0.40` threshold is *not* as risky as it looks — VERIFIED.** Spot-checks
   of the weakest `[0.40, 0.45)` band show the matches are **genuinely correct
   charters**; the low score is driven by poor HTR quality, not by mis-matching.
   Example `155o → 0218` (score 0.44): HTR *"Roger … standaert … hannoer caflet …
   pieter van hurfke"* vs. Gysseling *"Roger van Ghistelle … riquard standard
   ende hannoet casekin … pieter van huutkerke"* — the same charter beyond doubt.
   The n-gram→Levenshtein matcher locks onto rare proper names (people, places)
   even through ~40 %-garbage HTR, which is exactly why retrieval stays robust at
   low scores. This is the evidence behind the threshold choice in §6.

5. **Recto-only, by design.** Dorse (`m`) images are never transcribed/matched;
   they stay unlabelled. Expected, not a bug.

---

## 6. Distilling the mole archive

**Target shape (mole convention):** one flat image directory containing a
`labels.csv`. Images are the binarised crops mole already embeds. Some images
are unlabelled — they are *fake negatives* (distractors with unknown writer),
which is the norm across the other mole archives and is fine for both retrieval
evaluation and supervised training.

**Output:** `../images/archive-leroy/` containing
- every crop from `../images/cropped/` (the full retrieval pool, 1398 images), and
- `labels.csv` with one row **per image**.

`labels.csv` follows the mole convention (cf. `mole/data/brackley-set/labels.csv`):
the first two columns are `filename,hand_id`; the rest are provenance and are
ignored by mole.

| column | meaning |
|---|---|
| `filename` | image file within this directory (e.g. `6o.png`) |
| `hand_id` | `hand_group` **iff** the match clears the gate below, else empty |
| `match_score` | Levenshtein score of the underlying charter match (empty if unmatched) |
| `margin` | best − runner-up score (confidence proxy; empty if unmatched) |
| `gysseling_nr` | matched Gysseling nr (provenance/debugging; empty if unmatched) |

### The reliability gate = the coverage/reliability knob
A row gets a `hand_id` only if it has a `hand_group` **and**
`match_score ≥ --min-score`. Below the threshold the image still ships in the
directory but with an **empty `hand_id`** — it degrades gracefully into a fake
negative rather than injecting a probably-wrong class. `tussengroep` never
becomes a label.

Trade-off (labelled `hand_group` images as the threshold moves):

| `--min-score` | labelled imgs | groups | groups ≥2 | imgs in groups ≥2 |
|---|---|---|---|---|
| **0.40 (default)** | **461** | **98** | **76** | **439** |
| 0.45 | 403 | 96 | 68 | 375 |
| 0.50 | 323 | 93 | 61 | 291 |
| 0.55 | 219 | 82 | 47 | 184 |
| 0.70 | 54 | 32 | 12 | 34 |

**Recommended default: `0.40`.** Two facts drive this. First, group *count* is
remarkably stable across the range (98 → 93 groups from 0.40 → 0.50); tightening
the gate mostly thins images-per-group, not the number of classes. Second, and
decisively, the §5.4 spot-checks show the weak-score band contains *real*
matches with bad HTR, not wrong charters — so raising the gate would discard
correct labels, which is the opposite of what supervision needs here. Given the
goal is maximal labelled groups and the pipeline tolerates a noisy tail (as all
the mole archives do), keep everything the matcher accepted (`≥ 0.40`): **461
labelled images, 98 groups (76 with ≥2 members), 937 distractor negatives.**
Raise `--min-score` only to carve out a deliberately high-precision *evaluation*
split (e.g. 0.55–0.60); leave it at 0.40 for training.

### Build script
`build_mole_archive.py` (in this directory) is the reproducible recipe:

```bash
python build_mole_archive.py \
    --matched metadata-matched.xlsx \
    --crops ../images/cropped \
    --out   ../images/archive-leroy \
    --min-score 0.40
```

It copies every crop into `--out`, writes `labels.csv`, and prints a coverage
summary (labelled/unlabelled counts, group histogram). Re-running with a
different `--min-score` only rewrites `labels.csv` (add `--no-copy` to skip the
image copy) — the images are threshold-independent, so the archive can be
re-labelled without re-copying.

### Consuming it in mole
- **Evaluation** — treat `hand_id` as ground-truth writer (hand-group). Restrict
  positive pairs to groups with ≥2 labelled members; unlabelled images are
  gallery distractors. Optionally rebuild with a higher `--min-score` for a
  high-precision split.
- **Supervised training** — labelled rows are the supervision; unlabelled rows
  can still contribute via self-supervised / metric-learning objectives, exactly
  as fake negatives do in the other archives.

---

## 7. Reproduce from scratch

```bash
# 1. recto extraction + crop/binarise
jupyter run ../code/01-preprocessing.ipynb
# 2. HTR
export OPENROUTER_API_KEY=sk-or-...
python transcribe_charters.py ../images/cropped transcriptions
# 3. corpus to text
python convert_gysseling.py cg-admin-orig cd-admin-txt
# 4. match + attach hand groups  ->  metadata-matched.xlsx  (no API key needed)
python match_charters.py
# 5. distill the mole archive
python build_mole_archive.py --min-score 0.40
```
