# Leroy SSL retrieval (paper numbers)

Signed off **2026-09-05** on GPU host `mike` (`~/mole`). Same 1304-page freeze,
same `labels.csv`, same eval. Do **not** mix with the retired Sauvola-1398 /
auto-match numbers (raven macro 0.782, SSL 0.811, 435 queries).

**Application retrieval** (Büdingen as query) is below the locked table, not
inside it. Queries live in `data/leroy-queries/` (six stretched crops) and are
**not** in the 1304-page gallery. Encode them with `--codebook-from` that
gallery’s codebook.

Companion: `PREPROCESSING.md` (pixels), `hands-leroy/rationale.md` (labels).

---

## Protocol (fair, locked)

- Gallery: 1304 zoned+stretched rectos (`main_document=1`). Mole folder
  `data/leroy`. Sauvola copy `data/leroy-sauvola` from
  `mole prep data/leroy --binarize sauvola` (server mole: window 25, \(k=0.2\),
  \(R=128\); no extra percentile stretch — pages already stretched).
- Labels: match-review corpus id → Leroy `groep` only (never `tussengroep`).
  **714** pages with `hand_id`, **110** groepen, **1105** with `gysseling_nr`.
  Empty `hand_id` is **not** a class (neither query nor gallery).
- Backbone: `vit_small` patch 16, warm-start `checkpoints/raven_checkpoint.pth`,
  AttMask SSL, `configs/leroy.yaml` (20 epochs, warmup 5, batch 48, invert true,
  window 256 train / 224 embed, overlap 0.5 train / 0 embed, `use_zones=false`).
- Embed: VLAD \(K=100\), transductive codebook on the set being embedded,
  contrast foreground \(>0.05\), invert inherited from the run (raven-raw:
  `--invert`).
- Eval: `mole eval --cross-doc-only --topk 1,5`. Cosine. **697** queries
  (labelled pages with ≥1 other-charter same-hand partner). Self and same
  `gysseling_nr` siblings out. **macro-mAP** = mean of per-groep AP (93 groepen
  in the macro); **mAP** = mean AP over queries.

SSL does **not** use `hand_id`. Labels are eval-only.

---

## Results (cross-document)

| model | pixels | mAP | macro-mAP | Top-1 | Top-5 | checkpoint |
|---|---|---:|---:|---:|---:|---|
| raven-raw | grayscale | 0.5855 | 0.5632 | 0.7905 | 0.8809 | `raven_checkpoint.pth` (`vit_small@5927ab1d+step0`) |
| raven-raw, `--no-invert` | grayscale | 0.3136 | 0.2701 | 0.5667 | 0.7088 | same (polarity control) |
| raven-raw | Sauvola | 0.7578 | 0.7355 | 0.8924 | 0.9397 | same |
| SSL epoch 10 | grayscale | 0.7989 | 0.7532 | 0.8938 | 0.9412 | `runs/sluis_leroy_ssl_zoned/checkpoint_epoch0010.pth` (`@469686bd+step22011`) |
| SSL epoch 15 | grayscale | 0.8094 | 0.7678 | 0.8953 | 0.9383 | `…/checkpoint_epoch0015.pth` (`+step32016`) |
| SSL epoch 20 | grayscale | 0.8126 | 0.7715 | 0.8996 | 0.9383 | `…/checkpoint.pth` (`+step40020`) |
| SSL epoch 10 | Sauvola | 0.8284 | 0.7991 | 0.9039 | 0.9469 | `runs/sluis_leroy_ssl_sauvola/checkpoint_epoch0010.pth` (`@c3429da4+step22011`) |
| **SSL epoch 15** | **Sauvola** | **0.8369** | **0.8074** | **0.9110** | **0.9512** | **`…/checkpoint_epoch0015.pth` (`+step32016`)** |
| SSL epoch 20 | Sauvola | 0.8366 | 0.8058 | 0.9082 | 0.9498 | `…/checkpoint.pth` (`+step40020`) |

JSON sidecars on `mike`: `outputs/sluis/leroy.raven.eval.json`,
`leroy.raven.noinvert.eval.json`, `leroy.sauvola.raven.eval.json`,
`leroy.ssl.e{10,15,20}.eval.json`, `leroy.sauvola.ssl.e{10,15,20}.eval.json`.

---

## What to say in the paper

1. **Polarity.** Raven wants white-on-black. On these dark-on-light crops,
   `--invert` is required (macro 0.563 vs 0.270 without it).
2. **Tone vs adaptation.** Unadapted raven is weak on grayscale (0.563) and
   already strong on Sauvola of the *same* pages (0.736). Most of the
   grayscale-SSL jump (0.563 → 0.772) is the backbone learning raven’s native
   bitonal tone, not proof that grayscale is a better representation.
3. **After SSL, Sauvola wins.** Best run: Sauvola SSL epoch 15, macro **0.807**,
   Top-1 **0.911**. That beats grayscale SSL epoch 20 (0.772 / 0.900) by
   **+0.035** macro. Production pixels for this gallery: **Sauvola**.
4. **Schedule.** Warm-start, 20 epochs is enough; pick by retrieval not train
   loss. Grayscale still edged up to epoch 20; Sauvola peaked at **15** (20
   slightly worse). Same tail as the old 1398 Sauvola SSL.
5. **Old 0.782 / 0.811** are a different gallery (1398 Kraken+Sauvola crops) and
   easier auto-match labels (461 / ~435 queries). Not a baseline for this freeze.
6. **Top-1** = fraction of the 697 queries whose nearest labelled neighbour
   (other charter, same `hand_id`) is the same Leroy groep. **macro-mAP** gives
   each groep one vote (many groepen have only two charters).

**Shipped index for downstream Büdingen retrieval:** Sauvola SSL epoch 15,
`runs/sluis_leroy_ssl_sauvola/checkpoint_epoch0015.pth`, gallery embeddings
`outputs/sluis/leroy.sauvola.ssl.e15.npy`. Encode queries with
`--codebook-from` that run’s `.codebook.npy`, not a new k-means on six pages.

---

## Application (Büdingen needle)

Same shipped index. Haystack = 1304 gallery pages. Query pixels: zone + stretch
+ Sauvola, same as the gallery. **Do not** put Büdingen in the k-means.

| query | NN | cosine | notes |
|---|---|---:|---|
| Büdingen1r | **332o** | 0.448 | rank 1 of 1304 |
| Büdingen1v | **332o** | 0.449 | rank 1 of 1304 |
| r+v L2 mean | **332o** | **0.478** | rank 1; next charter 1358o at 0.363 |
| Büdingen r↔v | — | 0.761 | same leaf, two sides |

**332o** is `RA Gent / Gaillard / 800`, unlabelled (HTR match 0.30, below the
gate). **RA-800r** is a better extra photo of the same physical charter; it is
a query, not a gallery page. Ranking sheet:
`outputs/budingen_avg_nn_e15.html`.

A transductive 1310-page VLAD (gallery + extras, 332o kept) is an extra check,
not the paper index: with the other Büdingen side held out, 332o is still rank
1 (`outputs/leroy.all.budingen_nn.html`). UMAP of that space is
`outputs/leroy.all.sauvola.ssl.e15.viz.html`. UMAP proximity is not the
retrieval metric.

**Do not cite:** exemplar-SVM C-sweeps; the expanded run that dropped 332o;
LOAO / single-query codebook adaptation; additional2.
