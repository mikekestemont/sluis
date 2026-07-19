#!/usr/bin/env python3
"""
Charter-to-Corpus matching + Leroy hand-group attachment (script form).

Mirrors match_charters.ipynb but with three fixes over the original notebook:
  1. Case/format-robust Gysseling-nr join, so letter-suffix charters
     (e.g. 0065AA -> 65aa, 0044A' -> 44a) actually match the handengroepen sheet.
  2. Persists the retrieval `margin` (best - runner-up) into the output, as a
     second reliability signal beyond match_score.
  3. Labels on `hand_group` only (Type == 'groep'). The old, misleading `hand`
     column (which merged 'groep' + 'tussengroep') is dropped.

Two-stage matcher:
  1. char n-gram TF-IDF cosine -> top-K candidates
  2. rapidfuzz normalised Levenshtein re-rank

Usage:
    python match_charters.py            # uses the defaults below
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm

# ----- config -----
TRANSCRIPTIONS_DIR = Path("transcriptions")
CORPUS_DIR = Path("cd-admin-txt")
METADATA_FILE = Path("metadata.xlsx")
HANDS_FILE = Path("handengroepen_gysseling.xlsx")
OUTPUT_FILE = Path("metadata-matched.xlsx")
TOP_K = 20
MIN_THRESHOLD = 0.40   # keep permissive; downstream label gate lives in build_mole_archive.py


def load_texts(directory: Path) -> dict[str, str]:
    texts = {}
    for f in sorted(directory.glob("*.txt")):
        try:
            texts[f.name] = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            texts[f.name] = f.read_text(encoding="latin-1")
    return texts


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canon_gys(nr) -> str | None:
    """Canonical Gysseling key for joining corpus filenames to the hand sheet.

    Robust to the case/format drift between the two sources:
      '0065AA.txt' -> '65aa', "0044A'.txt" -> '44a', '1056a' -> '1056a',
      '1737A-AA-AB' -> '1737a-aa-ab'. Lowercase, drop leading zeros, drop
      apostrophes/spaces.
    """
    if nr is None or (isinstance(nr, float) and pd.isna(nr)):
        return None
    stem = Path(str(nr)).stem
    stem = stem.lower().replace("'", "").replace(" ", "")
    m = re.match(r"^0*(\d+.*)$", stem)
    return (m.group(1) if m else stem) or "0"


def main():
    transcriptions = load_texts(TRANSCRIPTIONS_DIR)
    corpus = load_texts(CORPUS_DIR)
    print(f"Transcriptions: {len(transcriptions)} | corpus: {len(corpus)}")

    trans_names = list(transcriptions)
    corpus_names = list(corpus)
    trans_normed = [normalize(transcriptions[n]) for n in trans_names]
    corpus_normed = [normalize(corpus[n]) for n in corpus_names]

    # stage 1: TF-IDF retrieval
    n_trans = len(trans_normed)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          min_df=1, max_df=0.95, sublinear_tf=True)
    tfidf = vec.fit_transform(trans_normed + corpus_normed)
    sim = cosine_similarity(tfidf[:n_trans], tfidf[n_trans:])
    candidates = [np.argsort(sim[i])[::-1][:TOP_K] for i in range(n_trans)]

    # stage 2: Levenshtein re-rank
    results = []
    for i, tname in enumerate(tqdm(trans_names, desc="Re-ranking")):
        t_text = trans_normed[i]
        best_score = second_score = -1.0
        best_corpus = second_corpus = None
        tfidf_of_best = 0.0
        for ci in candidates[i]:
            lev = Levenshtein.normalized_similarity(t_text, corpus_normed[ci])
            if lev > best_score:
                second_score, second_corpus = best_score, best_corpus
                best_score, best_corpus = lev, corpus_names[ci]
                tfidf_of_best = sim[i, ci]
            elif lev > second_score:
                second_score, second_corpus = lev, corpus_names[ci]
        results.append({
            "transcription": tname,
            "match": best_corpus if best_score >= MIN_THRESHOLD else None,
            "match_score": round(best_score, 4),
            "tfidf_score": round(float(tfidf_of_best), 4),
            "runner_up": second_corpus,
            "runner_up_score": round(second_score, 4),
            "margin": round(best_score - second_score, 4),
        })
    results_df = pd.DataFrame(results)
    print(f"Matches >= {MIN_THRESHOLD}: {results_df['match'].notna().sum()}/{len(results_df)}")

    # hand groups (Type == 'groep' only; tussengroep dropped)
    hands = pd.read_excel(HANDS_FILE, sheet_name="Per oorkonde")
    hands.columns = [str(c).strip() for c in hands.columns]
    groups = hands[hands["Type"] == "groep"].copy()
    groups["key"] = groups["Oorkonde (Gysseling-nr.)"].map(canon_gys)
    groups["hand_group"] = groups["Groep"].astype(str).str.strip()
    group_lookup = dict(zip(groups["key"], groups["hand_group"]))
    print(f"Hand-group lookup entries: {len(group_lookup)}")

    results_df["gysseling_nr"] = results_df["match"].map(
        lambda c: None if c is None else canon_gys(c))
    results_df["hand_group"] = results_df["gysseling_nr"].map(group_lookup)
    print(f"Matched with hand_group: {results_df['hand_group'].notna().sum()}")

    # merge onto photo metadata
    info = {Path(r["transcription"]).stem: r for _, r in results_df.iterrows()}
    cols = ["match", "match_score", "margin", "gysseling_nr", "hand_group"]

    def get(bestand):
        if pd.isna(bestand):
            return (None,) * len(cols)
        r = info.get(Path(str(bestand)).stem)
        return tuple(r[c] for c in cols) if r is not None else (None,) * len(cols)

    meta = pd.read_excel(METADATA_FILE)
    meta[cols] = [get(b) for b in meta["bestandsnaam"]]
    meta.to_excel(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE} | rows with hand_group: {meta['hand_group'].notna().sum()}")


if __name__ == "__main__":
    main()
