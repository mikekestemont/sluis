#!/usr/bin/env python3
"""Kraken BLLA text zones on pages-recto/.

Same approach as the legacy 01-preprocessing notebook: keep the **largest**
region (bbox area), not the union. Do not binarize (that is Stage 5).

Coordinates are stored in two spaces:
  - pages-recto PNG (EXIF already baked) — this is the crop source
  - archive-original JPG (stored pixels, before Orientation) so the published
    originals can show the same box

Resume-safe: data/zones_blla.jsonl is appended after each page.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
RECTO = ROOT / "images" / "pages-recto"
ZONED = ROOT / "images" / "pages-zoned"
MANIFEST = ROOT / "data" / "manifest.csv"
JSONL = ROOT / "data" / "zones_blla.jsonl"
ZONES_JSON = ROOT / "data" / "zones.json"
ZONES_CSV = ROOT / "data" / "zones.csv"
QC_HTML = ROOT / "outputs" / "zone_review.html"

logging.getLogger("kraken").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Polygonizer.*")


def baked_to_raw(x: float, y: float, bw: int, bh: int, orient) -> tuple[int, int]:
    """Map a point in EXIF-transposed pixels to stored-JPG pixels."""
    try:
        o = int(orient)
    except (TypeError, ValueError):
        o = 1
    x, y = float(x), float(y)
    if o == 1:
        rx, ry = x, y
    elif o == 2:
        rx, ry = bw - 1 - x, y
    elif o == 3:
        rx, ry = bw - 1 - x, bh - 1 - y
    elif o == 4:
        rx, ry = x, bh - 1 - y
    elif o == 5:  # transpose
        rx, ry = y, x
    elif o == 6:  # 90° CW; raw size = (bh, bw)
        rx, ry = y, bw - 1 - x
    elif o == 7:  # transverse
        rx, ry = bh - 1 - y, bw - 1 - x
    elif o == 8:  # 90° CCW; raw size = (bh, bw)
        rx, ry = bh - 1 - y, x
    else:
        rx, ry = x, y
    return int(round(rx)), int(round(ry))


def poly_bbox(poly) -> list[int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def area_bbox(b) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def region_records(seg) -> list[dict]:
    out = []
    for rtype, rlist in (seg.regions or {}).items():
        for r in rlist:
            poly = [(int(x), int(y)) for x, y in r.boundary]
            if len(poly) < 3:
                continue
            tags = r.tags or {}
            t = tags.get("type") or rtype or "region"
            if isinstance(t, (list, tuple)):
                t = t[0] if t else rtype
            label = str(t)
            bbox = poly_bbox(poly)
            out.append({
                "label": label,
                "bbox": bbox,
                "area": area_bbox(bbox),
                "polygon": poly,
            })
    return out


def crop_largest(im: Image.Image, poly, out_path: Path) -> None:
    """Mask polygon onto white, crop to AABB, save grayscale RGB PNG."""
    w, h = im.size
    poly = [(max(0, min(int(x), w - 1)), max(0, min(int(y), h - 1))) for x, y in poly]
    gray = im.convert("L")
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).polygon(poly, fill=255)
    a = np.asarray(gray)
    m = np.asarray(mask)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    cut = np.where(m[y0:y1, x0:x1] == 255, a[y0:y1, x0:x1], 255).astype(np.uint8)
    Image.fromarray(cut).convert("RGB").save(out_path, "PNG", compress_level=1)


def load_done() -> set[str]:
    done = set()
    if JSONL.is_file():
        for line in JSONL.open(encoding="utf-8"):
            line = line.strip()
            if line:
                done.add(json.loads(line)["filename"])
    return done


def compile_manifests(man_by_png: dict) -> None:
    entries = {}
    if JSONL.is_file():
        for line in JSONL.open(encoding="utf-8"):
            rec = json.loads(line)
            entries[rec["filename"]] = rec
    images = {}
    csv_rows = []
    for name, rec in sorted(entries.items(), key=lambda kv: int(kv[0].replace("o.png", ""))):
        images[name] = {
            "bbox": rec["bbox"],
            "size": rec["size"],
            "fell_back": rec["fell_back"],
            "detections": rec["detections"],
            "polygon": rec.get("polygon"),
            "n_regions": rec.get("n_regions", 0),
            "bbox_original": rec.get("bbox_original"),
            "original_path": rec.get("original_path"),
            "exif_orientation": rec.get("exif_orientation"),
            "main_document": rec.get("main_document"),
        }
        b = rec["bbox"] or [None, None, None, None]
        ob = rec.get("bbox_original") or [None, None, None, None]
        csv_rows.append({
            "filename": name,
            "main_document": rec.get("main_document"),
            "fell_back": int(rec["fell_back"]),
            "n_regions": rec.get("n_regions", 0),
            "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3],
            "width": rec["size"][0], "height": rec["size"][1],
            "ox0": ob[0], "oy0": ob[1], "ox1": ob[2], "oy1": ob[3],
            "original_path": rec.get("original_path", ""),
        })
    meta = {
        "detector": "blla",
        "model": "kraken/blla.mlmodel",
        "rule": "largest-region-by-bbox-area",
        "padding": 0,
        "padding_frac": 0.0,
        "zone_families": ["region"],
        "source": "images/pages-recto",
        "crop_dir": "images/pages-zoned",
        "n": len(images),
    }
    ZONES_JSON.parent.mkdir(parents=True, exist_ok=True)
    ZONES_JSON.write_text(json.dumps({"meta": meta, "images": images}, indent=2), encoding="utf-8")
    with ZONES_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()) if csv_rows else ["filename"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"zones.json {ZONES_JSON}  ({len(images)} pages)")
    print(f"zones.csv  {ZONES_CSV}")


def jpeg_b64(im: Image.Image, long=260, quality=70) -> str:
    t = im.copy()
    t.thumbnail((long, long))
    buf = io.BytesIO()
    t.convert("RGB").save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def draw_overlay(path: Path, rec: dict, long=320) -> Image.Image:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = long / max(w, h)
    disp = im.resize((max(1, round(w * scale)), max(1, round(h * scale))))
    draw = ImageDraw.Draw(disp)
    for d in rec.get("detections") or []:
        # [label, score, x0,y0,x1,y1]
        x0, y0, x1, y1 = [c * scale for c in d[2:6]]
        draw.rectangle((x0, y0, x1, y1), outline="#ff9f1c", width=2)
    if rec.get("bbox"):
        x0, y0, x1, y1 = [c * scale for c in rec["bbox"]]
        draw.rectangle((x0, y0, x1, y1), outline="#39d353", width=4)
    return disp


def build_qc(limit: int | None = None) -> None:
    if not JSONL.is_file():
        raise SystemExit("no zones_blla.jsonl yet")
    recs = [json.loads(l) for l in JSONL.open(encoding="utf-8") if l.strip()]
    recs.sort(key=lambda r: (
        -int(r.get("n_regions") or 0),
        int(r.get("fell_back") or 0),
        int(r["filename"].replace("o.png", "")),
    ))
    if limit:
        recs = recs[:limit]
    payload = []
    for rec in recs:
        src = RECTO / rec["filename"]
        crop = ZONED / rec["filename"]
        overlay = draw_overlay(src, rec)
        crop_im = Image.open(crop).convert("RGB") if crop.is_file() else overlay
        payload.append({
            "name": rec["filename"],
            "n": rec.get("n_regions", 0),
            "fell_back": rec.get("fell_back", False),
            "main": rec.get("main_document"),
            "bbox": rec.get("bbox"),
            "bbox_original": rec.get("bbox_original"),
            "overlay": jpeg_b64(overlay),
            "crop": jpeg_b64(crop_im),
        })
    html = r"""<!DOCTYPE html>
<meta charset="utf-8">
<title>Zone review (BLLA, largest region)</title>
<style>
  :root { --fg:#eee; --mut:#9a9a9a; --acc:#e66; --bg:#111; }
  body { margin: 0; font: 14px/1.4 system-ui, sans-serif; background: var(--bg); color: var(--fg); }
  header { position: sticky; top: 0; z-index: 2; background: #1a1a1a; border-bottom: 1px solid #333;
           padding: 12px 20px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
  h1 { font-size: 16px; margin: 0; }
  .sub { color: var(--mut); font-size: 13px; }
  button { background: #333; color: var(--fg); border: 1px solid #555; border-radius: 6px;
           padding: 6px 12px; cursor: pointer; font: inherit; }
  #sheet { padding: 16px 20px 64px; }
  .card { display: grid; grid-template-columns: 170px 1fr 1fr; gap: 12px; align-items: start;
          padding: 12px 0; border-bottom: 1px solid #2a2a2a; cursor: pointer; }
  .card.on { background: #281414; outline: 1px solid var(--acc); }
  .card img { width: 100%; max-width: 320px; display: block; border-radius: 4px; background: #222; }
  .meta { font: 12px ui-monospace, monospace; color: var(--mut); }
  .meta b { color: var(--fg); font-size: 14px; }
  .tag { display: inline-block; margin-top: 8px; padding: 2px 8px; border-radius: 999px; font-size: 11px; }
  .tag.ok { background: #333; }
  .tag.bad { background: var(--acc); color: #111; font-weight: 600; }
  .tag.fb { background: #3a3a00; color: #ee8; }
  .lbl { font-size: 11px; color: var(--mut); margin-bottom: 4px; }
</style>
<header>
  <div>
    <h1>Zone review — Kraken BLLA, largest region</h1>
    <div class="sub">Green box = kept zone. Orange = other BLLA regions (discarded).
      Click / Space = <b>correct later</b>. Unmarked = accept. j / k move.
      Sorted: most regions first, then fallbacks.</div>
  </div>
  <div class="sub" id="stats"></div>
  <button type="button" id="export">Export CSV</button>
  <button type="button" id="clear">Clear marks</button>
</header>
<div id="sheet"></div>
<script>
const ITEMS = __PAYLOAD__;
const KEY = "sluis-zones-blla-v1";
let marks = {};
try { marks = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { marks = {}; }
let focus = 0;
function save() { localStorage.setItem(KEY, JSON.stringify(marks)); stats(); }
function isOn(n) { return !!marks[n]; }
function stats() {
  const m = ITEMS.filter(it => isOn(it.name)).length;
  document.getElementById("stats").textContent = m + " marked correct / " + ITEMS.length + " shown";
}
function render() {
  const sheet = document.getElementById("sheet");
  sheet.innerHTML = ITEMS.map((it, i) => {
    const on = isOn(it.name);
    const fb = it.fell_back ? "<span class='tag fb'>full-page fallback</span> " : "";
    const tag = on ? "<span class='tag bad'>CORRECT later</span>" : "<span class='tag ok'>accept largest</span>";
    const bb = (it.bbox || []).join(",");
    return `<div class="card ${on?"on":""}" data-i="${i}" id="c${i}">
      <div class="meta"><b>${it.name}</b><br>regions ${it.n} · gallery ${it.main}${fb}<br>bbox [${bb}]<br>${tag}</div>
      <div><div class="lbl">page (green = kept)</div><img src="data:image/jpeg;base64,${it.overlay}"></div>
      <div><div class="lbl">cutout</div><img src="data:image/jpeg;base64,${it.crop}"></div>
    </div>`;
  }).join("");
  sheet.querySelectorAll(".card").forEach(el => el.addEventListener("click", () => toggle(+el.dataset.i)));
  highlight(); stats();
}
function toggle(i) {
  const n = ITEMS[i].name;
  if (marks[n]) delete marks[n]; else marks[n] = 1;
  focus = i; save();
  const el = document.getElementById("c"+i);
  el.classList.toggle("on", isOn(n));
  const tag = el.querySelector(".tag.ok, .tag.bad");
  if (tag && !tag.classList.contains("fb")) {
    tag.className = "tag " + (isOn(n) ? "bad" : "ok");
    tag.textContent = isOn(n) ? "CORRECT later" : "accept largest";
  }
}
function highlight() {
  document.querySelectorAll(".card").forEach((el, i) => {
    el.style.boxShadow = i === focus ? "inset 3px 0 0 #6ae" : "";
  });
}
function goto(i) {
  focus = Math.max(0, Math.min(ITEMS.length - 1, i));
  highlight();
  document.getElementById("c"+focus).scrollIntoView({block: "nearest"});
}
document.getElementById("export").onclick = () => {
  const lines = ["filename,correct,n_regions,fell_back,main_document,bbox"];
  ITEMS.forEach(it => lines.push([it.name, isOn(it.name)?1:0, it.n, it.fell_back?1:0, it.main,
    (it.bbox||[]).join(" ")].join(",")));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n")], {type: "text/csv"}));
  a.download = "zone_decisions.csv";
  a.click();
};
document.getElementById("clear").onclick = () => {
  if (!confirm("Clear all correction marks?")) return;
  marks = {}; save(); render();
};
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "j" || e.key === "ArrowDown") { e.preventDefault(); goto(focus + 1); }
  if (e.key === "k" || e.key === "ArrowUp") { e.preventDefault(); goto(focus - 1); }
  if (e.key === " " || e.key === "Enter") { e.preventDefault(); toggle(focus); }
});
render();
</script>
"""
    QC_HTML.parent.mkdir(parents=True, exist_ok=True)
    QC_HTML.write_text(html.replace("__PAYLOAD__", json.dumps(payload)), encoding="utf-8")
    print(f"review → {QC_HTML}  ({QC_HTML.stat().st_size/1e6:.1f} MB, {len(payload)} pages)")


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def run(limit: int | None, device: str, qc_every: int) -> None:
    from kraken import blla
    from kraken.lib import vgsl
    from tqdm import tqdm

    man = [r for r in csv.DictReader(MANIFEST.open(encoding="utf-8")) if r["side"] == "recto"]
    by_png = {Path(r["released_path"]).name: r for r in man}
    files = sorted(RECTO.glob("*.png"), key=lambda p: int(p.stem[:-1]))
    if limit:
        files = files[:limit]

    done = load_done()
    todo = [p for p in files if p.name not in done]
    print(f"recto PNGs {len(files)}  already {len(done)}  todo {len(todo)}  device {device}")
    if not todo:
        compile_manifests(by_png)
        build_qc()
        return

    import kraken as _k
    model_path = Path(_k.__file__).parent / "blla.mlmodel"
    seg_model = vgsl.TorchVGSLModel.load_model(str(model_path))
    print(f"loaded {model_path}")

    ZONED.mkdir(parents=True, exist_ok=True)
    JSONL.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    with JSONL.open("a", encoding="utf-8") as fh:
        for p in tqdm(todo, desc="BLLA", unit="page"):
            meta = by_png.get(p.name, {})
            im = Image.open(p).convert("RGB")
            w, h = im.size
            recs = []
            err = ""
            try:
                seg = blla.segment(im, model=seg_model, device=device, raise_on_error=False)
                recs = region_records(seg)
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
            crop_largest(im, poly, ZONED / p.name)
            orient = meta.get("exif_orientation") or 1
            corners = [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]
            raw_pts = [baked_to_raw(x, y, w, h, orient) for x, y in corners]
            bbox_orig = poly_bbox(raw_pts)
            row = {
                "filename": p.name,
                "size": [w, h],
                "bbox": bbox,
                "polygon": poly,
                "fell_back": fell_back,
                "n_regions": len(recs),
                "detections": [[r["label"], 1.0, *r["bbox"]] for r in recs],
                "bbox_original": bbox_orig,
                "original_path": meta.get("original_path", ""),
                "exif_orientation": orient,
                "main_document": meta.get("main_document"),
                "error": err,
            }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            n_done += 1
            if qc_every and n_done % qc_every == 0:
                compile_manifests(by_png)
                try:
                    build_qc(limit=120)
                except Exception:
                    pass

    compile_manifests(by_png)
    build_qc()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto", help="auto|cpu|mps|cuda:0")
    ap.add_argument("--qc-only", action="store_true")
    ap.add_argument("--qc-every", type=int, default=40)
    ap.add_argument("--qc-limit", type=int, default=None)
    args = ap.parse_args()
    os.chdir(ROOT)
    if args.qc_only:
        compile_manifests({})
        build_qc(limit=args.qc_limit)
        return
    run(limit=args.limit, device=pick_device(args.device), qc_every=args.qc_every)


if __name__ == "__main__":
    main()
