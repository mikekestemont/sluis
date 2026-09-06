#!/usr/bin/env python3
"""Destaturate blue-cast pages (no polarity flip).

Reads data/grayscale_blue_cast.csv. Converts each PNG to grayscale RGB in
place. Idempotent: already-gray files stay gray. Does not touch
archive-original/.
"""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
RECTO = ROOT / "images" / "pages-recto"
VERSO = ROOT / "images" / "pages-verso"
LIST = ROOT / "data" / "grayscale_blue_cast.csv"
MANIFEST = ROOT / "data" / "manifest.csv"
POLARITY = ROOT / "data" / "polarity_decisions.csv"


def to_gray_rgb(path: Path) -> None:
    Image.open(path).convert("L").convert("RGB").save(path, "PNG", compress_level=1)


def main() -> None:
    names = [r["filename"] for r in csv.DictReader(LIST.open(encoding="utf-8"))]
    for name in names:
        if name.endswith("o.png"):
            path = RECTO / name
        elif name.endswith("m.png"):
            path = VERSO / name
        else:
            raise SystemExit(f"unexpected filename {name}")
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        to_gray_rgb(path)
    print(f"grayscale-converted {len(names)} files from {LIST.name}")

    invert_set = set()
    if POLARITY.is_file():
        invert_set = {
            r["filename"]
            for r in csv.DictReader(POLARITY.open(encoding="utf-8"))
            if r.get("invert") == "1"
        }

    man = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    fields = list(man[0].keys())
    if "grayscale" not in fields:
        fields.append("grayscale")
    gray_set = set(names) | invert_set
    for r in man:
        png = Path(r.get("released_path") or "").name
        r["grayscale"] = "1" if png in gray_set else "0"
        if r.get("inverted") == "1":
            r["grayscale"] = "1"
    with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(man)
    n = sum(1 for r in man if r.get("grayscale") == "1")
    print(f"manifest grayscale=1: {n}")


if __name__ == "__main__":
    main()
