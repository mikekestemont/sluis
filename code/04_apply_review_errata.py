#!/usr/bin/env python3
"""Apply post-LS review notes: extra same-doc drops + polarity inverts.

Does not delete PNGs. Gallery extras get main_document=0. Invert is grayscale
ImageOps.invert on pages-recto, never twice.

  python code/04_apply_review_errata.py
"""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
RECTO = ROOT / "images" / "pages-recto"
MANIFEST = ROOT / "data" / "manifest.csv"
DUP_CSV = ROOT / "data" / "review_neardup_extra.csv"
INV_CSV = ROOT / "data" / "polarity_round3.csv"
POLARITY = ROOT / "data" / "polarity_decisions.csv"


def grayscale_invert(path: Path) -> None:
    im = Image.open(path)
    ImageOps.invert(im.convert("L")).convert("RGB").save(path, "PNG", compress_level=1)


def apply_inverts(man: list[dict]) -> list[str]:
    to_invert = [r["filename"].strip() for r in csv.DictReader(INV_CSV.open(encoding="utf-8"))
                 if r.get("invert") == "1"]
    already = {Path(r.get("released_path") or "").name for r in man
               if r.get("side") == "recto" and r.get("inverted") == "1"}
    applied = []
    for name in to_invert:
        path = RECTO / name
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        if name in already:
            continue
        grayscale_invert(path)
        applied.append(name)
        already.add(name)
    inv_set = set(to_invert)
    for r in man:
        if r.get("side") != "recto":
            continue
        png = Path(r.get("released_path") or "").name
        if png in inv_set:
            r["inverted"] = "1"
            r["grayscale"] = "1"
    if POLARITY.is_file():
        pol = list(csv.DictReader(POLARITY.open(encoding="utf-8")))
        fields = list(pol[0].keys()) if pol else ["filename", "invert", "kind", "dark_frac"]
        by = {r["filename"]: r for r in pol}
        for name in to_invert:
            rec = by.get(name, {"filename": name, "kind": "review", "dark_frac": ""})
            rec["invert"] = "1"
            rec["filename"] = name
            rec.setdefault("kind", "review")
            rec.setdefault("dark_frac", "")
            by[name] = rec
        with POLARITY.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(by.values())
    return applied


def apply_dups(man: list[dict]) -> tuple[list[str], list[str]]:
    drop_to_keep: dict[str, str] = {}
    keep_restore: set[str] = set()
    for r in csv.DictReader(DUP_CSV.open(encoding="utf-8")):
        keep = r["keep"].strip()
        keep_restore.add(keep)
        for name in (r.get("drop") or "").split("|"):
            name = name.strip()
            if name:
                drop_to_keep[name] = keep
    dropped, restored = [], []
    for r in man:
        if r.get("side") != "recto":
            continue
        png = Path(r.get("released_path") or "").name
        if png in drop_to_keep:
            keep = drop_to_keep[png]
            r["main_document"] = "0"
            r["reason"] = f"neardup_of={keep.replace('.png', '')}"
            dropped.append(png)
        elif png in keep_restore and r.get("main_document") == "0":
            r["main_document"] = "1"
            if str(r.get("reason") or "").startswith("neardup_of="):
                r["reason"] = "recto"
            restored.append(png)
    return dropped, restored


def main() -> None:
    man = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    fields = list(man[0].keys())
    inverted = apply_inverts(man)
    dropped, restored = apply_dups(man)
    with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(man)
    n_main = sum(1 for r in man if r.get("side") == "recto" and r.get("main_document") == "1")
    print(f"inverted {len(inverted)}: {', '.join(inverted) or '(none new)'}")
    print(f"restored gallery {len(restored)}: {', '.join(restored) or '(none)'}")
    print(f"dropped {len(dropped)}")
    for png in sorted(dropped, key=lambda n: int(n.replace("o.png", ""))):
        print(f"  {png}")
    print(f"gallery main rectos now {n_main}")


if __name__ == "__main__":
    main()
