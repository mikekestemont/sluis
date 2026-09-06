#!/usr/bin/env python3
"""Freeze the current zoned+stretched gallery as mole/data/leroy.

Copies the 1304 main-document crops, writes labels.csv from match-review
decisions + Leroy groepen, and a zones.json whose bbox is the full crop
(already zoned on disk).
"""
from __future__ import annotations

import csv
import json
import struct
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from match_charters import canon_gys, leroy_groups
from review_matches import load_decisions

SRC = ROOT / "images" / "pages-zoned-stretched"
ZONES_CSV = ROOT / "data" / "zones.csv"
MANIFEST = ROOT / "data" / "manifest.csv"
DECISIONS = ROOT / "outputs" / "match_review_decisions.csv"
HANDS = HERE / "handengroepen_gysseling.xlsx"
OUT = Path("/Users/mikekestemont/GitRepos/mole/data/leroy")

# Same physical charter as 153o (Gysseling I 217); review row was glitched.
SIBLING_OF = {"154o": "153o"}


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as fh:
        sig = fh.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"not a PNG: {path}")
        length, typ = struct.unpack(">I4s", fh.read(8))
        if typ != b"IHDR" or length < 8:
            raise ValueError(f"no IHDR: {path}")
        w, h = struct.unpack(">II", fh.read(8))
        return int(w), int(h)


def clone_copy(src: Path, dst: Path) -> None:
    r = subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True)
    if r.returncode != 0:
        subprocess.run(["cp", str(src), str(dst)], check=True)


def main_document_stems() -> set[str]:
    stems: set[str] = set()
    with ZONES_CSV.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("main_document") or "").strip() == "1":
                stems.add(Path(row["filename"]).stem)
    return stems


def build_labels(stems: list[str]) -> list[dict]:
    lookup, ambiguous = leroy_groups(HANDS)
    marks = load_decisions(DECISIONS)
    by_stem: dict[str, dict] = {}

    def row_for(stem: str) -> dict:
        m = marks.get(stem) or {}
        corpus = (m.get("corpus") or "").strip()
        note = (m.get("note") or "").strip()
        hand_id, gys = "", ""
        if corpus.lower().startswith("hand:"):
            hand_id = corpus.split(":", 1)[1].strip()
        elif corpus:
            key = canon_gys(corpus)
            gys = key or ""
            if key and key not in ambiguous:
                hand_id = lookup.get(key, "")
        return {
            "filename": f"{stem}.png",
            "hand_id": hand_id,
            "match_score": "",
            "margin": "",
            "gysseling_nr": gys,
            "note": note,
        }

    for stem in stems:
        by_stem[stem] = row_for(stem)

    for child, parent in SIBLING_OF.items():
        if child not in by_stem or parent not in by_stem:
            continue
        src = by_stem[parent]
        if src["gysseling_nr"] and not by_stem[child]["gysseling_nr"]:
            by_stem[child]["gysseling_nr"] = src["gysseling_nr"]
            by_stem[child]["hand_id"] = src["hand_id"]
            by_stem[child]["note"] = (
                f"sibling-of-{parent};{by_stem[child]['note']}".strip(";")
            )

    return [by_stem[s] for s in stems]


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"missing {SRC}")
    pngs = sorted(p for p in SRC.iterdir() if p.suffix.lower() == ".png")
    keep = main_document_stems()
    gallery = [p for p in pngs if p.stem in keep]
    extra = [p for p in pngs if p.stem not in keep]
    missing = keep - {p.stem for p in pngs}
    if extra:
        raise SystemExit(f"stretched extras not in main_document=1: {[p.name for p in extra]}")
    if missing:
        raise SystemExit(f"main_document=1 missing from stretched: {sorted(missing)[:20]}")
    if len(gallery) != 1304:
        raise SystemExit(f"expected 1304 gallery pages, got {len(gallery)}")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_labels([p.stem for p in gallery])
    zones_images = {}
    for src, rec in zip(gallery, rows):
        dst = OUT / src.name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            clone_copy(src, dst)
        w, h = png_size(dst)
        zones_images[src.name] = {
            "bbox": [0, 0, w, h],
            "size": [w, h],
            "fell_back": False,
            "detections": [],
        }

    labels_path = OUT / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["filename", "hand_id", "match_score", "margin",
                        "gysseling_nr", "note"],
        )
        writer.writeheader()
        writer.writerows(rows)

    zones = {
        "meta": {
            "detector": "sluis-ls-zones",
            "note": "Images are already text-zone crops (pages-zoned-stretched). "
                    "bbox is identity in crop space.",
            "source": str(SRC),
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_images": len(gallery),
        },
        "images": zones_images,
    }
    (OUT / "zones.json").write_text(json.dumps(zones, indent=2), encoding="utf-8")
    (OUT / "README.txt").write_text(
        "Sluis Leroy freeze: 1304 grayscale stretched text-zone crops.\n"
        "Not Sauvola. Do not add this folder to data/pooled-bin.\n"
        "Train/embed with invert: true (dark-on-light on disk).\n",
        encoding="utf-8",
    )

    labelled = [r for r in rows if r["hand_id"]]
    with_gys = [r for r in rows if r["gysseling_nr"]]
    groups = Counter(r["hand_id"] for r in labelled)
    print(f"out            {OUT}")
    print(f"images         {len(gallery)}")
    print(f"labelled       {len(labelled)}")
    print(f"with gys nr    {len(with_gys)}")
    print(f"hand groups    {len(groups)}")
    print(f"154o           gys={next(r['gysseling_nr'] for r in rows if r['filename']=='154o.png')} "
          f"hand={next(r['hand_id'] for r in rows if r['filename']=='154o.png')}")
    print(f"153o           gys={next(r['gysseling_nr'] for r in rows if r['filename']=='153o.png')} "
          f"hand={next(r['hand_id'] for r in rows if r['filename']=='153o.png')}")


if __name__ == "__main__":
    main()
