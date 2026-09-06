#!/usr/bin/env python3
"""
Charter-to-Corpus matching + Leroy hand-group attachment.

Two-stage matcher (keep this boring):
  1. char n-gram TF-IDF cosine -> top-K
  2. rapidfuzz normalised Levenshtein re-rank; accept if >= 0.40

Cheap post-filters (do not change retrieval):
  - search only Gysseling texts that have a unique Leroy groep
  - drop empty / TEST corpus files
  - skip Leroy nrs with two different groep values
  - if several photos match the same Gysseling text and their HTRs are
    near-identical, keep one labelled

Usage:
    python match_charters.py --transcriptions transcriptions-zoned
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
TOP_K = 20
MIN_THRESHOLD = 0.40
DUP_HTR = 0.90


def load_texts(directory: Path, *, skip_junk: bool = False) -> dict[str, str]:
    texts = {}
    for f in sorted(directory.glob("*.txt")):
        if skip_junk and "TEST" in f.stem.upper():
            continue
        try:
            raw = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw = f.read_text(encoding="latin-1")
        if skip_junk and not raw.strip():
            continue
        texts[f.name] = raw
    return texts


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canon_gys(nr) -> str | None:
    if nr is None or (isinstance(nr, float) and pd.isna(nr)):
        return None
    stem = Path(str(nr)).stem
    stem = stem.lower().replace("'", "").replace(" ", "")
    m = re.match(r"^0*(\d+.*)$", stem)
    return (m.group(1) if m else stem) or "0"


def leroy_groups(hands_file: Path) -> tuple[dict[str, str], set[str]]:
    """Unambiguous Gysseling-nr → hand_group. Conflicting nrs are the second set."""
    hands = pd.read_excel(hands_file, sheet_name="Per oorkonde")
    hands.columns = [str(c).strip() for c in hands.columns]
    groups = hands[hands["Type"] == "groep"].copy()
    groups["key"] = groups["Oorkonde (Gysseling-nr.)"].map(canon_gys)
    groups["hand_group"] = groups["Groep"].astype(str).str.strip()
    by_key: dict[str, set[str]] = defaultdict(set)
    for key, hg in zip(groups["key"], groups["hand_group"]):
        if key:
            by_key[key].add(hg)
    ambiguous = {k for k, v in by_key.items() if len(v) > 1}
    lookup = {k: next(iter(v)) for k, v in by_key.items() if len(v) == 1}
    if ambiguous:
        print(f"  skipped {len(ambiguous)} Gysseling nrs with conflicting groep: "
              f"{', '.join(sorted(ambiguous))}")
    return lookup, ambiguous


def filter_leroy_corpus(corpus: dict[str, str], lookup: dict[str, str]) -> dict[str, str]:
    kept = {name: text for name, text in corpus.items()
            if canon_gys(name) in lookup}
    print(f"  Leroy-labelled corpus: {len(kept)}/{len(corpus)}")
    return kept


def rank_neighbors(transcriptions: dict[str, str], corpus: dict[str, str],
                   k: int = TOP_K) -> list[dict]:
    trans_names = list(transcriptions)
    corpus_names = list(corpus)
    trans_normed = [normalize(transcriptions[n]) for n in trans_names]
    corpus_normed = [normalize(corpus[n]) for n in corpus_names]
    n_trans = len(trans_normed)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          min_df=1, max_df=0.95, sublinear_tf=True)
    vec.fit(corpus_normed)
    sim = cosine_similarity(vec.transform(trans_normed), vec.transform(corpus_normed))
    rows = []
    for i, tname in enumerate(tqdm(trans_names, desc="Re-ranking")):
        t_text = trans_normed[i]
        scored = []
        if t_text:
            for ci in np.argsort(sim[i])[::-1][:k]:
                lev = Levenshtein.normalized_similarity(t_text, corpus_normed[ci])
                scored.append((float(lev), float(sim[i, ci]), corpus_names[ci]))
            scored.sort(key=lambda x: (-x[0], -x[1]))
        neighbors = [
            {"corpus": c, "score": round(lev, 4), "lev": round(lev, 4),
             "tfidf": round(tf, 4)}
            for lev, tf, c in scored
        ]
        best = neighbors[0] if neighbors else None
        second = neighbors[1] if len(neighbors) > 1 else None
        best_score = best["score"] if best else 0.0
        rows.append({
            "transcription": tname,
            "htr": transcriptions[tname],
            "htr_norm": t_text,
            "empty": not bool(t_text),
            "match": best["corpus"] if best and best_score >= MIN_THRESHOLD else None,
            "match_score": best_score,
            "lev": best_score,
            "tfidf_score": best["tfidf"] if best else 0.0,
            "runner_up": second["corpus"] if second else None,
            "runner_up_score": second["score"] if second else 0.0,
            "margin": round(best_score - (second["score"] if second else 0.0), 4),
            "neighbors": neighbors,
        })
    return rows


def unlabel_near_dups(rows: list[dict], dup_htr: float = DUP_HTR) -> int:
    by_corpus: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("match"):
            by_corpus[r["match"]].append(i)
    n_dup = 0
    for idxs in by_corpus.values():
        for i in idxs:
            rows[i]["n_photos_for_gys"] = len(idxs)
            rows[i]["collision"] = len(idxs) > 1
        if len(idxs) < 2:
            continue
        parent = {i: i for i in idxs}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in combinations(idxs, 2):
            if fuzz.partial_ratio(rows[a]["htr_norm"], rows[b]["htr_norm"]) / 100.0 >= dup_htr:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
        clusters: dict[int, list[int]] = defaultdict(list)
        for i in idxs:
            clusters[find(i)].append(i)
        for members in clusters.values():
            if len(members) < 2:
                continue
            keeper = max(members, key=lambda i: (rows[i]["match_score"], rows[i]["transcription"]))
            keeper_stem = Path(rows[keeper]["transcription"]).stem
            for i in members:
                if i == keeper:
                    continue
                rows[i]["dup_of"] = keeper_stem
                rows[i]["hand_group"] = None
                n_dup += 1
    for r in rows:
        r.setdefault("n_photos_for_gys", 1 if r.get("match") else 0)
        r.setdefault("collision", False)
        r.setdefault("dup_of", None)
    return n_dup


def attach_hands(rows: list[dict], lookup: dict[str, str],
                 ambiguous: set[str]) -> None:
    for r in rows:
        for n in r["neighbors"]:
            n["gysseling_nr"] = canon_gys(n["corpus"])
            n["hand_group"] = lookup.get(n["gysseling_nr"])
        r["gysseling_nr"] = canon_gys(r["match"]) if r.get("match") else None
        if r.get("dup_of"):
            r["hand_group"] = None
        else:
            key = r.get("gysseling_nr")
            r["hand_group"] = None if key in ambiguous else lookup.get(key) if key else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcriptions", type=Path,
                    default=HERE / "transcriptions-zoned")
    ap.add_argument("--corpus", type=Path, default=HERE / "cd-admin-txt")
    ap.add_argument("--metadata", type=Path, default=HERE / "metadata.xlsx")
    ap.add_argument("--hands", type=Path, default=HERE / "handengroepen_gysseling.xlsx")
    ap.add_argument("--output", type=Path, default=HERE / "metadata-matched.xlsx")
    ap.add_argument("--topk-out", type=Path, default=HERE / "match_topk.json")
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--all-corpus", action="store_true",
                    help="Search all Gysseling texts, not only those with a Leroy groep")
    args = ap.parse_args()

    transcriptions = load_texts(args.transcriptions)
    corpus = load_texts(args.corpus, skip_junk=True)
    lookup, ambiguous = leroy_groups(args.hands)
    if not args.all_corpus:
        corpus = filter_leroy_corpus(corpus, lookup)
    print(f"Transcriptions: {len(transcriptions)} | corpus: {len(corpus)}")
    rows = rank_neighbors(transcriptions, corpus, k=args.k)
    n_dup = unlabel_near_dups(rows)
    attach_hands(rows, lookup, ambiguous)

    n_hit = sum(1 for r in rows if r["match"])
    n_hand = sum(1 for r in rows if r.get("hand_group"))
    print(f"Matches >= {MIN_THRESHOLD}: {n_hit}/{len(rows)}")
    print(f"Matched with hand_group: {n_hand}  near-dups unlabelled: {n_dup}")

    slim = [{k: r[k] for k in r if k != "htr_norm"} for r in rows]
    payload = {
        "meta": {
            "transcriptions": str(args.transcriptions),
            "corpus": str(args.corpus),
            "k": args.k,
            "threshold": MIN_THRESHOLD,
            "leroy_only": not args.all_corpus,
            "n_photos": len(rows),
            "n_corpus": len(corpus),
            "n_matched": n_hit,
            "n_hand_group": n_hand,
            "n_unlabelled_dup": n_dup,
        },
        "corpus_texts": corpus,
        "photos": slim,
    }
    args.topk_out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.topk_out} ({args.topk_out.stat().st_size / 1e6:.1f} MB)")

    info = {Path(r["transcription"]).stem: r for r in rows}
    cols = ["match", "match_score", "margin", "gysseling_nr", "hand_group", "dup_of"]

    def get(bestand):
        if pd.isna(bestand):
            return (None,) * len(cols)
        rec = info.get(Path(str(bestand)).stem)
        return tuple(rec.get(k) for k in cols) if rec is not None else (None,) * len(cols)

    meta = pd.read_excel(args.metadata)
    meta[cols] = [get(b) for b in meta["bestandsnaam"]]
    meta.to_excel(args.output, index=False)
    print(f"Saved {args.output} | rows with hand_group: {meta['hand_group'].notna().sum()}")


if __name__ == "__main__":
    main()
