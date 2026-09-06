#!/usr/bin/env python3
"""Exemplar linear SVM vs cosine on the frozen Sauvola SSL e15 gallery.

Corpus: same protocol as `mole eval --cross-doc-only` (labelled pages only,
siblings out). Each labelled query is the sole positive; all other gallery
pages (labelled + unlabelled) are negatives.

Queries: Büdingen r+v as the two positives against the 1304-page haystack.
332o (archive photo) and RA-800r (better photo of the same charter) are
scored, not used as training positives.
"""
from __future__ import annotations

import json
import unicodedata
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC

from mole.eval.retrieval import (
    _doc_resolvers,
    _hand_if_confident,
    _label_tables,
    _load_embeddings,
    _rank_metrics,
    _similarity,
)

ROOT = Path.home() / "mole"
G_NPY = ROOT / "outputs/sluis/leroy.sauvola.ssl.e15.npy"
Q_NPY = ROOT / "outputs/sluis/leroy.queries.sauvola.ssl.e15.npy"
DATA = ROOT / "data/leroy-sauvola"
OUT = ROOT / "outputs/sluis/exemplar_svm_e15.json"

PCA_DIM = 128
CS = (0.001, 0.01, 0.03, 0.1, 1.0)
NEEDLE = "332o"
SAME_DOC_HQ = "RA-800r"
BUDINGEN = ["Büdingen1r", "Büdingen1v"]
TOPK = (1, 5)


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def labelled_eval_pack(npy: Path, datasets_root: Path):
    """Reproduce mole eval's labelled subset + cross-doc allow mask."""
    X, images, model_id = _load_embeddings(npy)
    tables = _label_tables(datasets_root)
    resolvers = _doc_resolvers(datasets_root)
    solo = next(iter(tables)) if len(tables) == 1 else None
    hands, docs, keep = [], [], []
    for i, path in enumerate(images):
        p = Path(path)
        ds = p.parent.name
        if ds in tables:
            hand = _hand_if_confident(tables[ds], p.name, None)
        elif solo:
            hand, ds = _hand_if_confident(tables[solo], p.name, None), solo
        else:
            hand = None
        if hand is None:
            continue
        hands.append(hand)
        resolve = resolvers.get(ds)
        docs.append(f"{ds}/{resolve(p.name)}" if resolve else f"{ds}/{p.name}")
        keep.append(i)
    keep = np.asarray(keep, dtype=int)
    labels = np.asarray(hands, dtype=object)
    doc_arr = np.asarray(docs, dtype=object)
    n = len(keep)
    allow = (~np.eye(n, dtype=bool)) & (doc_arr[:, None] != doc_arr[None, :])
    stems = [nfc(Path(images[i]).stem) for i in range(len(images))]
    return {
        "X": np.asarray(X, dtype=np.float64),
        "images": images,
        "stems": stems,
        "keep": keep,
        "labels": labels,
        "allow": allow,
        "model_id": model_id,
    }


def scores_to_dict(s) -> dict:
    return {
        "n_queries": s.n_queries,
        "mAP": s.mean_ap,
        "macro_mAP": s.macro_map,
        "top1": s.top1,
        "top5": s.topk.get(5),
    }


def fit_pca(X: np.ndarray, dim: int) -> tuple[PCA, np.ndarray]:
    Xn = l2(X)
    k = min(dim, Xn.shape[0] - 1, Xn.shape[1])
    pca = PCA(n_components=k, random_state=0)
    Z = l2(pca.fit_transform(Xn))
    return pca, Z


def svm_scores_for_query(Z: np.ndarray, pos: np.ndarray, C: float) -> np.ndarray:
    """Decision function of an exemplar SVM: `pos` indexes the positives."""
    y = np.full(len(Z), -1, dtype=np.int32)
    y[np.asarray(pos, dtype=int)] = 1
    clf = LinearSVC(
        C=float(C),
        class_weight="balanced",
        dual=False,
        max_iter=4000,
        tol=1e-4,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        clf.fit(Z, y)
    return clf.decision_function(Z).astype(np.float64)


def corpus_svm_sim(Z: np.ndarray, keep: np.ndarray, C: float) -> np.ndarray:
    """Pairwise SVM scores among labelled pages (query row = that page's SVM)."""
    def row(gi: int) -> np.ndarray:
        return svm_scores_for_query(Z, [gi], C)[keep]

    rows = Parallel(n_jobs=-1, prefer="threads")(
        delayed(row)(int(gi)) for gi in keep
    )
    sim = np.vstack(rows)
    np.fill_diagonal(sim, -np.inf)
    return sim


def rank_haystack(scores: np.ndarray, stems: list[str], needle: str) -> dict:
    order = np.argsort(-scores)
    needle_i = stems.index(nfc(needle))
    rank = int(np.where(order == needle_i)[0][0] + 1)
    top = [
        {"rank": r, "stem": stems[i], "score": float(scores[i])}
        for r, i in enumerate(order[:20], 1)
    ]
    return {
        "nn_stem": stems[int(order[0])],
        "nn_score": float(scores[order[0]]),
        "needle": needle,
        "needle_rank": rank,
        "needle_score": float(scores[needle_i]),
        "top": top,
    }


def main() -> None:
    pack = labelled_eval_pack(G_NPY, DATA)
    X, keep, labels, allow = pack["X"], pack["keep"], pack["labels"], pack["allow"]
    stems = pack["stems"]
    print(f"gallery {len(X)}  labelled {len(keep)}  model {pack['model_id']}")

    cosine = _rank_metrics(_similarity(X[keep], "cosine"), labels, allow, TOPK)
    print(f"cosine raw VLAD   mAP {cosine.mean_ap:.4f}  macro {cosine.macro_map:.4f}  "
          f"Top-1 {cosine.top1:.4f}  n={cosine.n_queries}")

    pca, Z = fit_pca(X, PCA_DIM)
    print(f"PCA {Z.shape[1]}-d  var {pca.explained_variance_ratio_.sum():.3f}")
    cosine_pca = _rank_metrics(_similarity(Z[keep], "cosine"), labels, allow, TOPK)
    print(f"cosine PCA-{Z.shape[1]}   mAP {cosine_pca.mean_ap:.4f}  "
          f"macro {cosine_pca.macro_map:.4f}  Top-1 {cosine_pca.top1:.4f}")

    corpus = {
        "cosine_vlad": scores_to_dict(cosine),
        "cosine_pca": scores_to_dict(cosine_pca),
        "svm": {},
    }
    best_c, best_macro = None, -1.0
    for C in CS:
        print(f"\nSVM C={C} …")
        sim = corpus_svm_sim(Z, keep, C)
        s = _rank_metrics(sim, labels, allow, TOPK)
        corpus["svm"][str(C)] = scores_to_dict(s)
        print(f"  mAP {s.mean_ap:.4f}  macro {s.macro_map:.4f}  Top-1 {s.top1:.4f}")
        if s.macro_map > best_macro:
            best_macro, best_c = s.macro_map, C

    # --- Büdingen queries (haystack = full 1304, 332o in it) ---
    q_X, q_images, _ = _load_embeddings(Q_NPY)
    q_stems = [nfc(Path(p).stem) for p in q_images]
    q_Xn = l2(np.asarray(q_X, dtype=np.float64))
    Zq = l2(pca.transform(q_Xn))
    bud_idx = [q_stems.index(nfc(s)) for s in BUDINGEN]
    ra_i = q_stems.index(nfc(SAME_DOC_HQ))

    # Stack RA-800r onto the PCA gallery so we can rank both photos of the charter.
    Z_ext = np.vstack([Z, Zq[ra_i][None]])
    stems_ext = stems + [nfc(SAME_DOC_HQ)]

    queries = {}
    # cosine in raw VLAD (gallery codebook), averaged Büdingen vs gallery
    Ghat = l2(X)
    bud_vlad = l2(q_Xn[bud_idx]).mean(0)
    bud_vlad /= np.linalg.norm(bud_vlad)
    cos_g = Ghat @ bud_vlad
    queries["cosine_vlad_avg"] = rank_haystack(cos_g, stems, NEEDLE)
    # RA-800r cosine vs avg Büdingen, rank if inserted
    ra_cos = float(l2(q_Xn[ra_i][None])[0] @ bud_vlad)
    scores_with_ra = np.concatenate([cos_g, [ra_cos]])
    queries["cosine_vlad_avg_with_RA800r"] = rank_haystack(
        scores_with_ra, stems_ext, NEEDLE)
    queries["cosine_vlad_avg_with_RA800r"]["RA800r_rank"] = rank_haystack(
        scores_with_ra, stems_ext, SAME_DOC_HQ)["needle_rank"]
    queries["cosine_vlad_avg_with_RA800r"]["RA800r_score"] = ra_cos

    print("\nBüdingen avg cosine: 332o rank",
          queries["cosine_vlad_avg"]["needle_rank"],
          "cos", f"{queries['cosine_vlad_avg']['needle_score']:.3f}",
          "NN", queries["cosine_vlad_avg"]["nn_stem"])
    print("  +RA-800r in haystack: 332o rank",
          queries["cosine_vlad_avg_with_RA800r"]["needle_rank"],
          "RA-800r rank",
          queries["cosine_vlad_avg_with_RA800r"]["RA800r_rank"])

    queries["svm"] = {}
    for C in CS:
        # Train in the 1304-d gallery PCA space; positives are the two query rows
        # appended for the fit, then we score gallery + RA-800r.
        Z_fit = np.vstack([Z, Zq[bud_idx]])
        pos = list(range(len(Z), len(Z_fit)))
        raw = svm_scores_for_query(Z_fit, pos, C)
        gal_scores = raw[: len(Z)]
        rec = rank_haystack(gal_scores, stems, NEEDLE)
        # score RA-800r with the same SVM (transform already in Zq)
        y = np.full(len(Z_fit), -1, dtype=np.int32)
        y[pos] = 1
        clf = LinearSVC(C=float(C), class_weight="balanced", dual=False,
                        max_iter=4000, tol=1e-4, random_state=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            clf.fit(Z_fit, y)
        ra_score = float(clf.decision_function(Zq[ra_i][None])[0])
        ext = np.concatenate([gal_scores, [ra_score]])
        rec_ext = rank_haystack(ext, stems_ext, NEEDLE)
        rec["RA800r_score"] = ra_score
        rec["with_RA800r_332o_rank"] = rec_ext["needle_rank"]
        rec["with_RA800r_RA800r_rank"] = rank_haystack(
            ext, stems_ext, SAME_DOC_HQ)["needle_rank"]
        rec["with_RA800r_nn"] = rec_ext["nn_stem"]
        queries["svm"][str(C)] = rec
        print(f"SVM C={C}: 332o rank {rec['needle_rank']}  "
              f"NN {rec['nn_stem']}  "
              f"with RA-800r: 332o #{rec['with_RA800r_332o_rank']}  "
              f"RA-800r #{rec['with_RA800r_RA800r_rank']}")

    report = {
        "model_id": pack["model_id"],
        "pca_dim": int(Z.shape[1]),
        "pca_var": float(pca.explained_variance_ratio_.sum()),
        "best_C_by_macro": best_c,
        "corpus": corpus,
        "queries": queries,
        "note": (
            "332o and RA-800r are two photographs of the same charter; "
            "332o is the archive B&W, RA-800r the higher-quality photo. "
            "Corpus mAP uses labelled leave-one-out --cross-doc-only. "
            "Büdingen SVM positives are r+v only; gallery pages are negatives."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}")
    print(
        f"cosine macro {cosine.macro_map:.4f} vs best SVM C={best_c} "
        f"macro {best_macro:.4f}  Δ {best_macro - cosine.macro_map:+.4f}"
    )


if __name__ == "__main__":
    main()
