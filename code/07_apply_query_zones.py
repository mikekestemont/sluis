#!/usr/bin/env python3
"""Apply sluis-queries Label Studio boxes: crop + percentile stretch.

Reads the local LS sqlite (project title sluis-queries). Crops come from
images/additional/ (not the LS upload copies). Does not touch the gallery.

  python code/07_apply_query_zones.py
"""
from __future__ import annotations

import json
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "images" / "additional"
ZONED = ROOT / "images" / "queries-zoned"
STRETCHED = ROOT / "images" / "queries-zoned-stretched"
EXPORT = ROOT / "data" / "ls_queries_export.json"
ZONES = ROOT / "data" / "query_zones.json"
LS_DB = Path.home() / "Library/Application Support/label-studio/label_studio.sqlite3"
TITLE = "sluis-queries"
P_LO, P_HI, OUT_LO, OUT_HI = 2.0, 98.0, 20, 255


def fold(name: str) -> str:
    n = unicodedata.normalize("NFKD", name)
    return "".join(c for c in n if not unicodedata.combining(c)).lower()


def src_index() -> dict[str, Path]:
    out = {}
    for p in SRC.iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            out[fold(p.stem)] = p
    return out


def dump_ls() -> list[dict]:
    con = sqlite3.connect(f"file:{LS_DB}?mode=ro", uri=True)
    row = con.execute("SELECT id FROM project WHERE title = ?", (TITLE,)).fetchone()
    if not row:
        raise SystemExit(f"no project {TITLE!r}")
    pid = row[0]
    tasks = []
    for tid, data, result, cancelled in con.execute(
        """
        SELECT t.id, t.data, c.result, c.was_cancelled
        FROM task t
        JOIN task_completion c ON c.task_id = t.id
        WHERE t.project_id = ?
          AND c.id = (
            SELECT c2.id FROM task_completion c2
            WHERE c2.task_id = t.id ORDER BY c2.id DESC LIMIT 1)
        ORDER BY t.id
        """,
        (pid,),
    ):
        payload = json.loads(data) if isinstance(data, str) else data
        res = json.loads(result) if isinstance(result, str) else (result or [])
        tasks.append({
            "id": tid,
            "data": payload,
            "annotations": [{"was_cancelled": bool(cancelled), "result": res}],
        })
    return tasks


def bbox_from_task(task: dict) -> tuple[list[int], tuple[int, int], str]:
    img = (task.get("data") or {}).get("image") or ""
    stem = Path(img.split("?")[0]).stem
    if "-" in stem and len(stem.split("-", 1)[0]) == 8:
        stem = stem.split("-", 1)[1]
    anns = task.get("annotations") or []
    for ann in anns:
        if ann.get("was_cancelled"):
            continue
        for item in ann.get("result") or []:
            val = item.get("value") or {}
            if "width" not in val:
                continue
            w = int(item.get("original_width") or 0)
            h = int(item.get("original_height") or 0)
            x0 = val["x"] / 100.0 * w
            y0 = val["y"] / 100.0 * h
            x1 = x0 + val["width"] / 100.0 * w
            y1 = y0 + val["height"] / 100.0 * h
            bbox = [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]
            return bbox, (w, h), stem
    raise SystemExit(f"no MainZone on task {task.get('id')}")


def stretch_gray(gray: np.ndarray) -> tuple[np.ndarray, dict]:
    lo, hi = np.percentile(gray, [P_LO, P_HI])
    span = float(hi - lo)
    stats = {"p2": float(lo), "p98": float(hi), "span": span, "applied": int(span >= 8)}
    if span < 8:
        return gray.copy(), stats
    scale = (OUT_HI - OUT_LO) / span
    out = np.clip(np.round(OUT_LO + (gray.astype(np.float32) - lo) * scale), 0, 255)
    return out.astype(np.uint8), stats


def main() -> None:
    by_fold = src_index()
    tasks = dump_ls()
    EXPORT.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    ZONED.mkdir(parents=True, exist_ok=True)
    STRETCHED.mkdir(parents=True, exist_ok=True)
    images = {}
    for task in tasks:
        bbox, (ow, oh), stem = bbox_from_task(task)
        src = by_fold.get(fold(stem))
        if src is None:
            raise SystemExit(f"no source for {stem!r}; have {sorted(by_fold)}")
        im = ImageOps.exif_transpose(Image.open(src)).convert("L")
        w, h = im.size
        if (w, h) != (ow, oh):
            print(f"WARN size {src.name} file={w}x{h} ls={ow}x{oh}")
        x0, y0, x1, y1 = bbox
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        crop = np.asarray(im.crop((x0, y0, x1, y1)), np.uint8)
        name = src.stem + ".png"
        Image.fromarray(crop).convert("RGB").save(ZONED / name, "PNG", compress_level=1)
        stretched, stats = stretch_gray(crop)
        Image.fromarray(stretched).convert("RGB").save(
            STRETCHED / name, "PNG", compress_level=1)
        images[name] = {
            "source": str(src.relative_to(ROOT)),
            "page_size": [w, h],
            "bbox": [x0, y0, x1, y1],
            "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
            "stretch": stats,
        }
        print(f"{name:24} bbox={x0,y0,x1,y1} crop={crop.shape[1]}x{crop.shape[0]} "
              f"p2={stats['p2']:.0f} p98={stats['p98']:.0f}")
    ZONES.write_text(json.dumps({"meta": {"n": len(images), "source": "sluis-queries"},
                                 "images": images}, indent=2), encoding="utf-8")
    print(f"export {EXPORT}")
    print(f"crops  {ZONED}  ({len(images)})")
    print(f"stretch {STRETCHED}")


if __name__ == "__main__":
    main()
