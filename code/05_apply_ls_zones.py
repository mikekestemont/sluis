#!/usr/bin/env python3
"""Apply a Label Studio JSON-MIN export back onto zones + recrop.

LS rectangles are percentages. Pixel boxes are taken from the full-resolution
pages-recto PNG (not the LS display size). Edited boxes replace bbox and
become a 4-corner polygon; bbox_original is remapped. Then recrop.

  python code/05_apply_ls_zones.py --export ~/Downloads/project-N.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
JSONL = ROOT / "data" / "zones_blla.jsonl"
MANIFEST = ROOT / "data" / "manifest.csv"
RECTO = ROOT / "images" / "pages-recto"
ZONED = ROOT / "images" / "pages-zoned"

from importlib.machinery import SourceFileLoader
_blla = SourceFileLoader("blla_zones", str(ROOT / "code" / "05_blla_zones.py")).load_module()


def ls_rect_to_bbox(val: dict, w: int, h: int) -> list[int]:
    x0 = val["x"] / 100.0 * w
    y0 = val["y"] / 100.0 * h
    x1 = x0 + val["width"] / 100.0 * w
    y1 = y0 + val["height"] / 100.0 * h
    return [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]


def task_filename(task: dict) -> str | None:
    data = task.get("data") or task
    fn = data.get("filename") or ""
    if fn:
        return Path(fn).name
    img = data.get("image") or ""
    if "pages-recto/" in img:
        return img.rsplit("pages-recto/", 1)[-1].split("?")[0]
    return Path(img).name or None


def regions_from_task(task: dict) -> list[dict]:
    anns = task.get("annotations") or []
    if not anns and "label" in task:
        return [r for r in (task.get("label") or []) if "width" in r]
    result = []
    for ann in anns:
        if ann.get("was_cancelled"):
            continue
        for item in ann.get("result") or []:
            val = item.get("value") or {}
            if "width" in val and "height" in val:
                result.append(val)
    return result


def unique_jsonl() -> dict[str, dict]:
    last = {}
    if JSONL.is_file():
        for line in JSONL.open(encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                last[rec["filename"]] = rec
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True)
    args = ap.parse_args()
    export = json.loads(args.export.read_text(encoding="utf-8"))
    if not isinstance(export, list):
        raise SystemExit("expected a JSON list (JSON-MIN or full export)")
    man = {Path(r["released_path"]).name: r
           for r in csv.DictReader(MANIFEST.open(encoding="utf-8"))
           if r.get("side") == "recto"}
    by = unique_jsonl()
    changed = 0
    for task in export:
        name = task_filename(task)
        if not name or name not in by:
            continue
        rec = by[name]
        w, h = rec["size"]
        regs = regions_from_task(task)
        if not regs:
            rec["fell_back"] = True
            rec["bbox"] = [0, 0, w, h]
            rec["polygon"] = [[0, 0], [w, 0], [w, h], [0, h]]
        else:
            # if several boxes, keep the largest (same rule as BLLA)
            boxes = [ls_rect_to_bbox(v, w, h) for v in regs]
            bbox = max(boxes, key=lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1]))
            rec["fell_back"] = False
            rec["bbox"] = bbox
            x0, y0, x1, y1 = bbox
            rec["polygon"] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        meta = man.get(name, {})
        if "main_document" in meta:
            rec["main_document"] = int(meta["main_document"])
        orient = meta.get("exif_orientation") or rec.get("exif_orientation") or 1
        corners = [(rec["bbox"][0], rec["bbox"][1]), (rec["bbox"][2], rec["bbox"][1]),
                   (rec["bbox"][2], rec["bbox"][3]), (rec["bbox"][0], rec["bbox"][3])]
        rec["bbox_original"] = _blla.poly_bbox(
            [_blla.baked_to_raw(x, y, w, h, orient) for x, y in corners])
        im = Image.open(RECTO / name).convert("RGB")
        ZONED.mkdir(parents=True, exist_ok=True)
        _blla.crop_largest(im, rec["polygon"], ZONED / name)
        by[name] = rec
        changed += 1
    JSONL.write_text("".join(json.dumps(by[k]) + "\n"
                             for k in sorted(by, key=lambda n: int(n.replace("o.png", "")))),
                     encoding="utf-8")
    _blla.compile_manifests({})
    print(f"updated {changed} pages from {args.export}")
    print(f"rewrote {JSONL} and { _blla.ZONES_JSON }")


if __name__ == "__main__":
    main()
