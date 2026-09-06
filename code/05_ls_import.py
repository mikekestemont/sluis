#!/usr/bin/env python3
"""Build a Label Studio import JSON from BLLA zones.

Each task is a baked recto with one pre-drawn MainZone rectangle (the largest
BLLA region). Import as annotations so pages start 'done'; edit and Update
only the ones that need it. Do not Skip empties.

Requires Label Studio started with local-file serving from this repo:

  export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
  export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/Users/mikekestemont/GitRepos/sluis
  conda activate sluis   # or bayes, wherever label-studio lives
  label-studio start

Then: create project → paste data/ls_config.xml → Import data/ls_zones_import.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSONL = ROOT / "data" / "zones_blla.jsonl"
OUT = ROOT / "data" / "ls_zones_import.json"
RECTO = ROOT / "images" / "pages-recto"
DOC_ROOT = ROOT


def unique_rows(path: Path) -> list[dict]:
    last: dict[str, dict] = {}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        last[rec["filename"]] = rec
    return [last[k] for k in sorted(last, key=lambda n: int(n.replace("o.png", "")))]


def rect_pct(bbox, w, h) -> dict:
    x0, y0, x1, y1 = bbox
    return {
        "x": 100.0 * x0 / w,
        "y": 100.0 * y0 / h,
        "width": 100.0 * (x1 - x0) / w,
        "height": 100.0 * (y1 - y0) / h,
        "rotation": 0,
        "rectanglelabels": ["MainZone"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery-only", action="store_true",
                    help="Only main_document=1 pages")
    args = ap.parse_args()
    if not JSONL.is_file():
        raise SystemExit(f"missing {JSONL}")
    recs = unique_rows(JSONL)
    if args.gallery_only:
        recs = [r for r in recs if str(r.get("main_document")) == "1"]
    tasks = []
    n_empty = 0
    for rec in recs:
        name = rec["filename"]
        abs_png = (RECTO / name).resolve()
        rel = abs_png.relative_to(DOC_ROOT.resolve()).as_posix()
        w, h = rec["size"]
        result = []
        if rec.get("bbox") and not rec.get("fell_back"):
            result.append({
                "original_width": w,
                "original_height": h,
                "image_rotation": 0,
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "origin": "prediction",
                "value": rect_pct(rec["bbox"], w, h),
            })
        else:
            n_empty += 1
        tasks.append({
            "data": {
                "image": f"/data/local-files/?d={rel}",
                "filename": str(abs_png),
                "main_document": rec.get("main_document"),
                "n_regions": rec.get("n_regions", 0),
                "fell_back": rec.get("fell_back", False),
            },
            "annotations": [{"result": result}],
        })
    OUT.write_text(json.dumps(tasks, indent=None), encoding="utf-8")
    print(f"tasks {len(tasks)}  with box {len(tasks)-n_empty}  empty/fallback {n_empty}")
    print(f"wrote {OUT}")
    print("label config:", ROOT / "data" / "ls_config.xml")


if __name__ == "__main__":
    main()
