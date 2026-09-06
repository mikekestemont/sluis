#!/usr/bin/env python3
"""Zone + stretch the literary-hand query photos (not the Leroy gallery).

Reads images/additional/, writes:
  images/queries-recto/              EXIF-baked grayscale RGB
  images/queries-zoned/              BLLA largest-region crop
  images/queries-zoned-stretched/    p2→20, p98→255 on the polygon interior
  outputs/query_zone_review.html     overlay + crop QC

Does not touch pages-zoned/, data/zones.csv, or mole/data/leroy.
Sauvola stays a later mole prep step on the stretched folder.
"""
from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "images" / "additional"
RECTO = ROOT / "images" / "queries-recto"
ZONED = ROOT / "images" / "queries-zoned"
STRETCHED = ROOT / "images" / "queries-zoned-stretched"
ZONES = ROOT / "data" / "query_zones.json"
QC = ROOT / "outputs" / "query_zone_review.html"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


blla_mod = _load("05_blla_zones.py")
stretch_mod = _load("06_contrast_stretch.py")


def jpeg_b64(im: Image.Image, long=320, quality=78) -> str:
    t = im.copy()
    t.thumbnail((long, long))
    buf = io.BytesIO()
    t.convert("RGB").save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def overlay(im: Image.Image, recs: list, kept, long=360) -> Image.Image:
    w, h = im.size
    scale = long / max(w, h)
    disp = im.convert("RGB").resize((max(1, round(w * scale)), max(1, round(h * scale))))
    draw = ImageDraw.Draw(disp)
    for r in recs:
        x0, y0, x1, y1 = r["bbox"]
        box = [x0 * scale, y0 * scale, x1 * scale, y1 * scale]
        col = (80, 220, 120) if r["bbox"] == kept else (220, 140, 60)
        draw.rectangle(box, outline=col, width=2)
    return disp


def bake_recto() -> list[Path]:
    RECTO.mkdir(parents=True, exist_ok=True)
    out = []
    for src in sorted(SRC.iterdir()):
        if src.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue
        im = ImageOps.exif_transpose(Image.open(src)).convert("L").convert("RGB")
        dest = RECTO / (src.stem + ".png")
        im.save(dest, "PNG", compress_level=1)
        out.append(dest)
        print(f"recto  {dest.name}  {im.size}")
    return out


def zone_and_stretch(files: list[Path], device: str) -> None:
    from kraken import blla
    from kraken.lib import vgsl
    import kraken as _k

    model_path = Path(_k.__file__).parent / "blla.mlmodel"
    seg_model = vgsl.TorchVGSLModel.load_model(str(model_path))
    print(f"BLLA {model_path}  device={device}")

    ZONED.mkdir(parents=True, exist_ok=True)
    STRETCHED.mkdir(parents=True, exist_ok=True)
    images = {}
    cards = []
    for p in files:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        recs = []
        err = ""
        try:
            seg = blla.segment(im, model=seg_model, device=device, raise_on_error=False)
            recs = blla_mod.region_records(seg)
        except Exception as e:
            err = str(e)
        recs.sort(key=lambda r: -r["area"])
        fell_back = not recs
        if fell_back:
            poly = [(0, 0), (w, 0), (w, h), (0, h)]
            bbox = [0, 0, w, h]
        else:
            poly = recs[0]["polygon"]
            bbox = recs[0]["bbox"]
        crop_path = ZONED / p.name
        blla_mod.crop_largest(im, poly, crop_path)
        crop = Image.open(crop_path).convert("L")
        gray = np.asarray(crop, np.uint8)
        rec = {"bbox": bbox, "polygon": poly}
        mask = stretch_mod.interior_mask(gray, rec)
        stretched, stats = stretch_mod.stretch_array(
            gray, mask, out_lo=20, out_hi=255)
        Image.fromarray(stretched).convert("RGB").save(
            STRETCHED / p.name, "PNG", compress_level=1)
        images[p.name] = {
            "bbox": bbox, "size": [w, h], "fell_back": fell_back,
            "n_regions": len(recs), "polygon": poly, "error": err,
            "stretch": stats,
        }
        cards.append({
            "name": p.name,
            "fell_back": fell_back,
            "n": len(recs),
            "bbox": bbox,
            "overlay": jpeg_b64(overlay(im, recs, bbox)),
            "crop": jpeg_b64(Image.fromarray(stretched).convert("RGB")),
            "mean": float(gray.mean()),
        })
        print(f"zone   {p.name}  regions={len(recs)} fallback={fell_back} "
              f"bbox={bbox} stretch_span={stats.get('span')}")

    ZONES.write_text(json.dumps({
        "meta": {"detector": "blla", "rule": "largest-region-by-bbox-area",
                 "source": "images/additional", "n": len(images)},
        "images": images,
    }, indent=2), encoding="utf-8")
    QC.write_text(
        "<!doctype html><meta charset=utf-8><title>query zones</title>"
        "<style>body{font:14px sans-serif;background:#111;color:#eee}"
        ".row{display:flex;gap:16px;margin:20px;align-items:flex-start}"
        "img{max-width:360px;background:#222} .mut{color:#aaa}</style>"
        + "".join(
            f"<div class=row><div><b>{c['name']}</b><div class=mut>"
            f"regions {c['n']} fallback {c['fell_back']} bbox {c['bbox']}</div>"
            f"<div class=mut>open {STRETCHED.name}/{c['name']}</div></div>"
            f"<div>page<br><img src='data:image/jpeg;base64,{c['overlay']}'></div>"
            f"<div>stretched crop<br><img src='data:image/jpeg;base64,{c['crop']}'></div></div>"
            for c in cards
        ),
        encoding="utf-8",
    )
    print(f"QC → {QC}")


def main() -> None:
    os.chdir(ROOT)
    files = bake_recto()
    if not files:
        raise SystemExit(f"no images in {SRC}")
    zone_and_stretch(files, blla_mod.pick_device("auto"))


if __name__ == "__main__":
    main()
