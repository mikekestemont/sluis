#!/usr/bin/env python3
"""All Sauvola pages in one folder: 1304 gallery (incl. 332o) + six extras.

Transductive VLAD codebook is fit on this union. Büdingen is in the k-means;
it is only held out later, at ranking time.
"""
from __future__ import annotations

import csv
import unicodedata
from pathlib import Path

ROOT = Path.home() / "mole"
SRC_G = ROOT / "data/leroy-sauvola"
SRC_Q = ROOT / "data/leroy-queries-sauvola"
DST = ROOT / "data/leroy-all-sauvola"


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for p in DST.iterdir():
        if p.is_symlink() or p.suffix.lower() in {".png", ".csv", ".json"}:
            p.unlink()

    n_g = 0
    for p in sorted(SRC_G.glob("*.png")):
        (DST / p.name).symlink_to(p.resolve())
        n_g += 1

    extras = []
    for p in sorted(SRC_Q.glob("*.png")):
        dest = DST / p.name
        if dest.exists():
            raise SystemExit(f"name clash: {p.name}")
        dest.symlink_to(p.resolve())
        extras.append(nfc(p.stem))

    src_labels = SRC_G / "labels.csv"
    with src_labels.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fields = list(r.fieldnames or [])
        rows = list(r)
    if not fields or "filename" not in fields:
        raise SystemExit("labels.csv missing filename column")
    for p in sorted(SRC_Q.glob("*.png")):
        row = {k: "" for k in fields}
        row["filename"] = p.name
        if "note" in fields:
            row["note"] = "additional-query"
        rows.append(row)
    with (DST / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    pngs = list(DST.glob("*.png"))
    print(f"wrote {DST}")
    print(f"  gallery (incl. 332o): {n_g}")
    print(f"  extras: {extras}")
    print(f"  total png: {len(pngs)}  labels: {len(rows)}")
    if n_g != 1304:
        raise SystemExit(f"expected 1304 gallery pages, got {n_g}")
    if "332o" not in {nfc(p.stem) for p in pngs}:
        raise SystemExit("332o missing")
    if set(extras) != {
        "Büdingen1r", "Büdingen1v",
        "Genois-1327a", "Genois-1327b", "Genois-1327c", "RA-800r",
    }:
        raise SystemExit(f"unexpected extras: {extras}")
    if len(pngs) != n_g + len(extras):
        raise SystemExit("png count mismatch")


if __name__ == "__main__":
    main()
