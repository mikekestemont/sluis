#!/usr/bin/env python3
"""
Distil the matched Leroy charters into a mole archive:
one flat image directory + a labels.csv inside it.

labels.csv follows the mole convention (cf. mole/data/brackley-set/labels.csv):
first two columns are `filename,hand_id`. Extra provenance columns
(match_score, margin, gysseling_nr) are appended and don't affect mole.

The images are the binarised text-zone crops that mole embeds. Every crop is
copied into the archive (the full retrieval pool); labels.csv carries a row per
image. An image gets a `hand_id` only when the underlying charter match is
reliable enough (match_score >= --min-score); otherwise `hand_id` is empty and
the image acts as a "fake negative" distractor, as in the other mole archives.

See rationale.md (§4, §6) for the label semantics and the coverage/reliability
trade-off. Only Leroy hand-groups (Type == 'groep') become labels; the fuzzy
`tussengroep` (intermediate) assignments are never used.

Usage:
    python build_mole_archive.py \
        --matched metadata-matched.xlsx \
        --crops ../images/cropped \
        --out   ../images/archive-leroy \
        --min-score 0.40
"""

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matched", type=Path, default=Path("metadata-matched.xlsx"),
                   help="metadata-matched.xlsx (output of match_charters.ipynb)")
    p.add_argument("--crops", type=Path, default=Path("../images/cropped"),
                   help="directory of binarised crops mole embeds")
    p.add_argument("--out", type=Path, default=Path("../images/archive-leroy"),
                   help="archive directory to create (images + labels.csv)")
    p.add_argument("--min-score", type=float, default=0.40,
                   help="min match_score to trust a hand_group label "
                        "(below -> image kept, hand_id left empty)")
    p.add_argument("--no-copy", action="store_true",
                   help="only (re)write labels.csv; assume images already copied")
    return p.parse_args()


def _fmt(v, nd=4):
    return "" if pd.isna(v) else f"{float(v):.{nd}f}"


def build_label_lookup(matched_path: Path, min_score: float) -> dict:
    """stem (e.g. '6o') -> dict(hand_id, match_score, margin, gysseling_nr) for
    matched charters. A row gets a non-empty hand_id only when it has a
    hand_group AND match_score >= min_score."""
    meta = pd.read_excel(matched_path)
    has_margin = "margin" in meta.columns
    lookup = {}
    for _, row in meta.iterrows():
        bestand = row.get("bestandsnaam")
        if pd.isna(bestand):
            continue
        stem = Path(str(bestand)).stem  # '6o.jpg' -> '6o'
        score = row.get("match_score")
        group = row.get("hand_group")
        score_val = None if pd.isna(score) else float(score)
        reliable = (not pd.isna(group)) and score_val is not None and score_val >= min_score
        lookup[stem] = {
            "hand_id": ("" if not reliable else str(group).strip()),
            "match_score": _fmt(score),
            "margin": (_fmt(row.get("margin")) if has_margin else ""),
            "gysseling_nr": ("" if pd.isna(row.get("gysseling_nr"))
                             else str(row.get("gysseling_nr")).strip()),
        }
    return lookup


def main():
    args = parse_args()
    if not args.crops.is_dir():
        raise SystemExit(f"crops dir not found: {args.crops}")

    lookup = build_label_lookup(args.matched, args.min_score)

    crops = sorted(f for f in args.crops.iterdir()
                   if f.suffix.lower() in IMAGE_EXTS)
    if not crops:
        raise SystemExit(f"no images in {args.crops}")

    args.out.mkdir(parents=True, exist_ok=True)

    empty = {"hand_id": "", "match_score": "", "margin": "", "gysseling_nr": ""}
    rows = []
    for src in crops:
        info = lookup.get(src.stem, empty)
        rows.append({
            "filename": src.name,
            "hand_id": info["hand_id"],
            "match_score": info["match_score"],
            "margin": info["margin"],
            "gysseling_nr": info["gysseling_nr"],
        })
        if not args.no_copy:
            shutil.copy2(src, args.out / src.name)

    labels_path = args.out / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["filename", "hand_id", "match_score", "margin",
                            "gysseling_nr"])
        writer.writeheader()
        writer.writerows(rows)

    # --- summary ---
    labelled = [r for r in rows if r["hand_id"]]
    groups = Counter(r["hand_id"] for r in labelled)
    multi = {g: n for g, n in groups.items() if n >= 2}
    print(f"Archive:        {args.out}")
    print(f"Images:         {len(rows)} "
          f"({'copied' if not args.no_copy else 'not copied (--no-copy)'})")
    print(f"min-score gate: {args.min_score}")
    print(f"Labelled:       {len(labelled)} images  "
          f"({len(rows) - len(labelled)} unlabelled negatives)")
    print(f"Hand groups:    {len(groups)}  "
          f"({len(multi)} with >=2 members, "
          f"{sum(multi.values())} images in those)")
    print(f"labels.csv:     {labels_path}")
    top = groups.most_common(10)
    if top:
        print("Top groups:     " + ", ".join(f"{g}:{n}" for g, n in top))


if __name__ == "__main__":
    main()
