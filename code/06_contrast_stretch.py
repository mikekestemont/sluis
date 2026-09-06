#!/usr/bin/env python3
"""Robust percentile stretch on zone crops.

Interior pixels = the BLLA/LS polygon (the white AABB fill is ignored).
Signed-off mapping (HTR input): p2 → 20, p98 → 255, gallery pages only
(main_document=1). Unstretched originals stay in pages-zoned/ (all 1408).

  python code/06_contrast_stretch.py          # stats + review HTML
  python code/06_contrast_stretch.py --apply  # write images/pages-zoned-stretched/
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
ZONED = ROOT / "images" / "pages-zoned"
ZONES = ROOT / "data" / "zones.json"
MANIFEST = ROOT / "data" / "manifest.csv"
STATS = ROOT / "data" / "stretch_stats.csv"
OUT_HTML = ROOT / "outputs" / "stretch_review.html"
OUT_APPLY = ROOT / "images" / "pages-zoned-stretched"

P_LO, P_HI = 2.0, 98.0
OUT_LO, OUT_HI = 20, 240          # review-sheet middle column
APPLY_OUT_HI = 255               # signed-off HTR mapping (keep parchment white)
MIN_SPAN = 8
THUMB = 420
SEED = 0


def interior_mask(gray: np.ndarray, rec: dict) -> np.ndarray:
    """True on the text-block polygon, in crop coordinates."""
    h, w = gray.shape
    bbox = rec.get("bbox") or [0, 0, w, h]
    x0, y0, x1, y1 = (int(v) for v in bbox)
    poly = rec.get("polygon") or [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    pts = [(int(x) - x0, int(y) - y0) for x, y in poly]
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).polygon(pts, fill=255)
    return np.asarray(m) == 255


def stretch_array(gray: np.ndarray, interior: np.ndarray,
                 out_lo: int = OUT_LO, out_hi: int = OUT_HI) -> tuple[np.ndarray, dict]:
    pix = gray[interior]
    n = int(pix.size)
    if n < 100:
        stats = {"p2": None, "p98": None, "span": None, "n_interior": n, "applied": 0}
        return gray.copy(), stats
    lo, hi = np.percentile(pix, [P_LO, P_HI])
    span = float(hi - lo)
    stats = {
        "p2": float(lo), "p98": float(hi), "span": span,
        "n_interior": n, "mean": float(pix.mean()),
        "applied": int(span >= MIN_SPAN),
    }
    out = gray.copy()
    if span < MIN_SPAN:
        return out, stats
    scale = (out_hi - out_lo) / span
    mapped = np.clip(np.round(out_lo + (gray.astype(np.float32) - lo) * scale), 0, 255)
    out[interior] = mapped[interior].astype(np.uint8)
    return out, stats


def jpeg_b64(arr: np.ndarray, long=THUMB, quality=78) -> str:
    im = Image.fromarray(arr).convert("RGB")
    im.thumbnail((long, long))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def load_zones() -> dict:
    return json.loads(ZONES.read_text(encoding="utf-8"))["images"]


def load_main() -> set[str]:
    keep = set()
    if MANIFEST.is_file():
        for r in csv.DictReader(MANIFEST.open(encoding="utf-8")):
            if r.get("side") == "recto" and r.get("main_document") == "1":
                keep.add(Path(r["released_path"]).name)
    return keep


def inverted_names() -> set[str]:
    names = set()
    if MANIFEST.is_file():
        for r in csv.DictReader(MANIFEST.open(encoding="utf-8")):
            if r.get("side") == "recto" and r.get("inverted") == "1":
                names.add(Path(r["released_path"]).name)
    return names


def score_all(zones: dict, mains: set[str]) -> list[dict]:
    rows = []
    inv = inverted_names()
    names = sorted(mains, key=lambda n: int(n.replace("o.png", "")))
    for i, name in enumerate(names, 1):
        path = ZONED / name
        rec = zones.get(name)
        if not path.is_file() or rec is None:
            continue
        gray = np.asarray(Image.open(path).convert("L"))
        interior = interior_mask(gray, rec)
        _, st = stretch_array(gray, interior)
        st["filename"] = name
        st["inverted"] = 1 if name in inv else 0
        rows.append(st)
        if i % 200 == 0:
            print(f"  scored {i}/{len(names)}")
    return rows


def pick_sample(rows: list[dict], n_each=10) -> list[dict]:
    usable = [r for r in rows if r.get("span") is not None]
    usable.sort(key=lambda r: r["span"])
    rng = random.Random(SEED)
    inv = [r for r in usable if r.get("inverted")]
    faded = sorted(usable, key=lambda r: -(r["p2"] or 0))
    mid = usable[len(usable) // 4: 3 * len(usable) // 4]
    chosen: dict[str, dict] = {}

    def take(src, k, kind):
        for r in src[:k]:
            rec = dict(r)
            rec["kind"] = kind if rec["filename"] not in chosen else chosen[rec["filename"]]["kind"]
            if rec["filename"] not in chosen:
                rec["kind"] = kind
                chosen[rec["filename"]] = rec

    take(usable, n_each, "narrow")          # compressed histogram — stretch should help
    take(list(reversed(usable)), n_each, "wide")  # already contrasty — risk of clipping
    take(faded, n_each, "faded")            # high p2, washed ink
    take(inv, min(n_each, len(inv)), "inverted")
    take(rng.sample(mid, min(n_each, len(mid))), n_each, "typical")
    extra = ["1o.png", "346o.png", "556o.png", "1245o.png", "1259o.png", "1360o.png"]
    by = {r["filename"]: r for r in usable}
    for name in extra:
        if name in by and name not in chosen:
            rec = dict(by[name])
            rec["kind"] = "anchor"
            chosen[name] = rec
    order = ["narrow", "faded", "inverted", "typical", "wide", "anchor"]
    out = list(chosen.values())
    out.sort(key=lambda r: (order.index(r["kind"]) if r["kind"] in order else 99,
                            r["span"]))
    return out


def pack_item(row: dict, zones: dict) -> dict | None:
    name = row["filename"]
    path = ZONED / name
    rec = zones.get(name)
    if not path.is_file() or rec is None:
        return None
    gray = np.asarray(Image.open(path).convert("L"))
    interior = interior_mask(gray, rec)
    stretched, st = stretch_array(gray, interior)
    keep_white, _ = stretch_array(gray, interior, out_lo=OUT_LO, out_hi=255)
    gain = (OUT_HI - OUT_LO) / st["span"] if st.get("span") else 1.0
    return {
        "name": name,
        "kind": row["kind"],
        "p2": round(st["p2"], 1) if st["p2"] is not None else None,
        "p98": round(st["p98"], 1) if st["p98"] is not None else None,
        "span": round(st["span"], 1) if st["span"] is not None else None,
        "gain": round(gain, 2),
        "inverted": row.get("inverted", 0),
        "orig": jpeg_b64(gray),
        "str": jpeg_b64(stretched),
        "white": jpeg_b64(keep_white),
    }


def write_html(items: list[dict]) -> None:
    payload = json.dumps(items, separators=(",", ":"))
    html = HTML.replace("__PAYLOAD__", payload).replace("__N__", str(len(items)))
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"review → {OUT_HTML}  ({OUT_HTML.stat().st_size / 1e6:.1f} MB)")


def apply_all(zones: dict, names: list[str] | None = None) -> None:
    """HTR folder: gallery pages only (main_document=1), p2→20 / p98→255."""
    OUT_APPLY.mkdir(parents=True, exist_ok=True)
    if names is None:
        names = sorted(load_main(), key=lambda x: int(x.replace("o.png", "")))
    else:
        names = sorted(names, key=lambda x: int(x.replace("o.png", "")))
    n = 0
    for i, name in enumerate(names, 1):
        rec = zones.get(name)
        path = ZONED / name
        if rec is None or not path.is_file():
            continue
        gray = np.asarray(Image.open(path).convert("L"))
        out, _ = stretch_array(
            gray, interior_mask(gray, rec), out_lo=OUT_LO, out_hi=APPLY_OUT_HI)
        Image.fromarray(out).convert("RGB").save(
            OUT_APPLY / name, "PNG", compress_level=1)
        n += 1
        if i % 200 == 0:
            print(f"  wrote {i}/{len(names)}")
    print(f"wrote {n} stretched gallery crops → {OUT_APPLY}  (p2→{OUT_LO}, p98→{APPLY_OUT_HI})")


def summarize(rows: list[dict]) -> None:
    spans = [r["span"] for r in rows if r.get("span") is not None]
    p2s = [r["p2"] for r in rows if r.get("p2") is not None]
    p98s = [r["p98"] for r in rows if r.get("p98") is not None]
    print(f"gallery crops scored: {len(spans)}")
    for label, xs in [("p2 (ink-ish)", p2s), ("p98 (parchment)", p98s), ("span p98-p2", spans)]:
        a = np.array(xs)
        print(f"  {label:16}  min {a.min():6.1f}  p25 {np.percentile(a,25):6.1f}  "
              f"med {np.median(a):6.1f}  p75 {np.percentile(a,75):6.1f}  max {a.max():6.1f}")
    would = sum(1 for s in spans if s < 180)
    print(f"  span < 180 (stretch does real work): {would}/{len(spans)}")
    print(f"  span < 100 (washed / low contrast):  {sum(1 for s in spans if s < 100)}/{len(spans)}")


HTML = r"""<!doctype html>
<meta charset=utf-8>
<title>Percentile stretch review</title>
<style>
  :root { --bg:#111; --fg:#eee; --mut:#9ab; --ok:#3ddc84; --bad:#f07178; --meh:#e0c36e; }
  body { font-family: system-ui, sans-serif; margin: 0; background: var(--bg); color: var(--fg); }
  header { position: sticky; top: 0; z-index: 2; background: #1a1a1a; border-bottom: 1px solid #333;
           padding: 12px 20px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
  h1 { font-size: 16px; margin: 0; font-weight: 600; }
  h2 { font-size: 14px; margin: 28px 20px 8px; color: var(--mut); font-weight: 600; }
  .sub { color: var(--mut); font-size: 13px; }
  button { background: #333; color: var(--fg); border: 1px solid #555; border-radius: 6px;
           padding: 6px 12px; cursor: pointer; font: inherit; }
  #sheet { padding: 8px 20px 64px; }
  .card { display: grid; grid-template-columns: 200px 1fr 1fr 1fr; gap: 12px; align-items: start;
          padding: 12px 0; border-bottom: 1px solid #2a2a2a; cursor: pointer; }
  .card.help { outline: 1px solid var(--ok); background: #14281c; }
  .card.hurt { outline: 1px solid var(--bad); background: #2a1418; }
  .card.meh { outline: 1px solid var(--meh); background: #2a2614; }
  .card img { width: 100%; max-width: 420px; display: block; border-radius: 4px; background: #222; }
  .meta { font: 12px ui-monospace, monospace; color: var(--mut); }
  .meta b { color: var(--fg); font-size: 14px; }
  .tag { display: inline-block; margin-top: 8px; padding: 2px 8px; border-radius: 999px; font-size: 11px; background: #333; }
  .lbl { font-size: 11px; color: var(--mut); margin-bottom: 4px; }
</style>
<header>
  <div>
    <h1>Percentile stretch — p2→20, p98→240</h1>
    <div class="sub">Interior polygon only. Click cycles help / hurts / meh.
      j/k move, Space toggle. __N__ samples. Does not write the archive.</div>
  </div>
  <div class="sub" id="stats"></div>
  <button type="button" id="export">Export CSV</button>
</header>
<div id="sheet"></div>
<script>
const ITEMS = __PAYLOAD__;
const KEY = "sluis-stretch-v1";
const CYCLE = ["", "help", "hurt", "meh"];
let marks = {};
try { marks = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { marks = {}; }
let focus = 0;
function save() { localStorage.setItem(KEY, JSON.stringify(marks)); stats(); }
function mark(name) { return marks[name] || ""; }
function stats() {
  const c = {help:0, hurt:0, meh:0};
  ITEMS.forEach(it => { if (c[mark(it.name)] !== undefined) c[mark(it.name)]++; });
  document.getElementById("stats").textContent =
    c.help + " help · " + c.hurt + " hurt · " + c.meh + " meh";
}
function card(it, i) {
  const m = mark(it.name);
  return `<div class="card ${m}" data-i="${i}" id="c${i}">
    <div class="meta"><b>${it.name}</b><br>
      ${it.kind}${it.inverted ? " · inverted" : ""}<br>
      p2 ${it.p2} → 20<br>p98 ${it.p98} → 240<br>
      span ${it.span} ×${it.gain}<br>
      <span class="tag">${m || "unmarked"}</span></div>
    <div><div class="lbl">current crop</div><img src="data:image/jpeg;base64,${it.orig}"></div>
    <div><div class="lbl">p2→20, p98→240</div><img src="data:image/jpeg;base64,${it.str}"></div>
    <div><div class="lbl">p2→20, p98→255 (keep white)</div><img src="data:image/jpeg;base64,${it.white}"></div>
  </div>`;
}
function render() {
  const groups = {};
  ITEMS.forEach(it => { (groups[it.kind] = groups[it.kind] || []).push(it); });
  const titles = {
    narrow: "Narrow span — stretch should do the most work",
    faded: "High p2 — washed / pale ink",
    inverted: "Already polarity-inverted",
    typical: "Mid-range (typical page)",
    wide: "Wide span — already contrasty; watch for clipping",
    anchor: "Named anchors",
  };
  let html = "";
  for (const k of Object.keys(titles)) {
    const g = groups[k];
    if (!g) continue;
    html += `<h2>${titles[k]}</h2>`;
    g.forEach(it => { html += card(it, ITEMS.indexOf(it)); });
  }
  document.getElementById("sheet").innerHTML = html;
  document.querySelectorAll(".card").forEach(el => {
    el.addEventListener("click", () => toggle(+el.dataset.i));
  });
  stats();
}
function toggle(i) {
  const name = ITEMS[i].name;
  const cur = CYCLE.indexOf(mark(name));
  marks[name] = CYCLE[(cur + 1) % CYCLE.length];
  if (!marks[name]) delete marks[name];
  save(); render();
  document.getElementById("c"+i)?.scrollIntoView({block:"nearest"});
  focus = i;
}
document.getElementById("export").onclick = () => {
  const lines = ["filename,kind,p2,p98,span,gain,mark"];
  ITEMS.forEach(it => lines.push(
    [it.name, it.kind, it.p2, it.p98, it.span, it.gain, mark(it.name)].join(",")));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\\n")]));
  a.download = "stretch_decisions.csv";
  a.click();
};
document.addEventListener("keydown", e => {
  if (e.key === "j") { focus = Math.min(ITEMS.length-1, focus+1);
    document.getElementById("c"+focus)?.scrollIntoView({block:"center"}); }
  if (e.key === "k") { focus = Math.max(0, focus-1);
    document.getElementById("c"+focus)?.scrollIntoView({block:"center"}); }
  if (e.key === " ") { e.preventDefault(); toggle(focus); }
});
render();
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write stretched copies to images/pages-zoned-stretched/")
    ap.add_argument("--html-only", action="store_true",
                    help="rebuild the review HTML from data/stretch_stats.csv")
    args = ap.parse_args()
    zones = load_zones()
    if args.apply:
        apply_all(zones)
        return
    mains = load_main() or set(zones)
    if args.html_only and STATS.is_file():
        rows = list(csv.DictReader(STATS.open(encoding="utf-8")))
        for r in rows:
            for k in ("p2", "p98", "span", "mean"):
                r[k] = float(r[k]) if r.get(k) else None
            r["inverted"] = int(r.get("inverted") or 0)
    else:
        print(f"scoring {len(mains)} gallery crops…")
        rows = score_all(zones, mains)
        STATS.parent.mkdir(parents=True, exist_ok=True)
        fields = ["filename", "inverted", "p2", "p98", "span", "mean", "n_interior", "applied"]
        with STATS.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"stats → {STATS}")
        summarize(rows)
    sample = pick_sample(rows)
    items = [pack_item(r, zones) for r in sample]
    items = [it for it in items if it]
    write_html(items)


if __name__ == "__main__":
    main()
