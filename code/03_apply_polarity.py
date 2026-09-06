#!/usr/bin/env python3
"""Apply polarity_decisions.csv (grayscale invert) and build round-2 review HTML."""
from __future__ import annotations

import base64, csv, io, json, random, shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
RECTO = ROOT / "images" / "pages-recto"
DECISIONS = ROOT / "outputs" / "polarity_decisions.csv"
MANIFEST = ROOT / "data" / "manifest.csv"
OUT_HTML = ROOT / "outputs" / "polarity_review_round2.html"
DARK_FRAC_MIN = 0.50
THUMB = 320
SEED = 0
N_CONTROLS = 6


def otsu_dark_frac(gray_u8):
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    levels = np.arange(256, dtype=np.float64)
    w0 = np.cumsum(hist)
    w1 = total - w0
    m0 = np.cumsum(hist * levels)
    sum_all = float((hist * levels).sum())
    ok = (w0 > 0) & (w1 > 0)
    between = np.zeros(256)
    between[ok] = ((sum_all * w0[ok] - m0[ok] * total) ** 2
                   / (w0[ok] * w1[ok] * total * total))
    t = int(np.argmax(between))
    return t, float((gray_u8 < t).mean())


def score(path, long=384):
    im = Image.open(path).convert("L")
    w, h = im.size
    s = long / max(w, h)
    small = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    a = np.asarray(small, np.uint8)
    t, dark = otsu_dark_frac(a)
    return {"mean": float(a.mean()), "dark_frac": dark, "otsu": t}


def jpeg_b64(im, long=THUMB, quality=78):
    t = im.copy()
    t.thumbnail((long, long))
    buf = io.BytesIO()
    t.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def pack(path, kind, extra):
    rgb = Image.open(path).convert("RGB")
    ginv = ImageOps.invert(rgb.convert("L"))
    return {
        "name": path.name,
        "kind": kind,
        "dark_frac": round(extra["dark_frac"], 3),
        "mean": round(extra["mean"], 1),
        "orig": jpeg_b64(rgb),
        "inv": jpeg_b64(ginv.convert("RGB")),
    }


def grayscale_invert(path: Path):
    im = Image.open(path)
    out = ImageOps.invert(im.convert("L")).convert("RGB")
    out.save(path, "PNG", compress_level=1)


def main():
    rows = list(csv.DictReader(DECISIONS.open(encoding="utf-8")))
    to_invert = [r["filename"] for r in rows if r.get("invert") == "1"]
    shutil.copy2(DECISIONS, ROOT / "data" / "polarity_decisions.csv")

    already = set()
    if MANIFEST.is_file():
        man = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
        if man and "inverted" in man[0]:
            already = {r["filename"] for r in man
                       if r.get("side") == "recto" and r.get("inverted") == "1"}

    applied, skipped = [], []
    for name in to_invert:
        path = RECTO / name
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        if name in already:
            skipped.append(name)
            continue
        grayscale_invert(path)
        applied.append(name)
    print(f"inverted {len(applied)}  skipped-already {len(skipped)}  listed {len(to_invert)}")

    # manifest column
    man_rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    fields = list(man_rows[0].keys())
    if "inverted" not in fields:
        fields.append("inverted")
    invert_set = set(to_invert)
    for r in man_rows:
        if r.get("side") != "recto":
            r["inverted"] = r.get("inverted") or "0"
            continue
        png = Path(r.get("released_path") or "").name or f"{r['volgnummer']}o.png"
        if png in invert_set or r.get("filename", "").replace(".jpg", ".png") in invert_set:
            r["inverted"] = "1"
        else:
            r["inverted"] = r.get("inverted") or "0"
    with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(man_rows)

    files = sorted(RECTO.glob("*.png"))
    scored = []
    for p in files:
        s = score(p)
        s["path"] = p
        s["name"] = p.name
        scored.append(s)

    applied_set = set(to_invert)
    still_dark = [s for s in scored
                  if s["dark_frac"] >= DARK_FRAC_MIN and s["name"] not in applied_set]
    still_dark.sort(key=lambda r: -r["dark_frac"])
    applied_scored = [s for s in scored if s["name"] in applied_set]
    applied_scored.sort(key=lambda r: -r["dark_frac"])
    applied_still_dark = [s for s in applied_scored if s["dark_frac"] >= DARK_FRAC_MIN]

    normals = [s for s in scored if s["dark_frac"] < 0.30 and s["name"] not in applied_set]
    random.seed(SEED)
    controls = random.sample(normals, min(N_CONTROLS, len(normals)))

    print(f"applied still dark_frac>={DARK_FRAC_MIN}: {len(applied_still_dark)}")
    print(f"remaining candidates (not inverted): {len(still_dark)}")

    items = (
        [pack(s["path"], "applied", s) for s in applied_scored] +
        [pack(s["path"], "candidate", s) for s in still_dark] +
        [pack(s["path"], "control", s) for s in controls]
    )

    payload = json.dumps(items, separators=(",", ":"))
    html = HTML.replace("__PAYLOAD__", payload)
    n_app, n_left = len(applied_scored), len(still_dark)
    html = html.replace("__NAPP__", str(n_app)).replace("__NLEFT__", str(n_left))
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"round-2 review → {OUT_HTML}  ({OUT_HTML.stat().st_size/1e6:.1f} MB)")


HTML = r"""<!doctype html>
<meta charset=utf-8>
<title>Polarity review — round 2</title>
<style>
  :root { --bg:#111; --fg:#eee; --mut:#9ab; --acc:#3ddc84; --warn:#f0883e; }
  body { font-family: system-ui, sans-serif; margin: 0; background: var(--bg); color: var(--fg); }
  header { position: sticky; top: 0; z-index: 2; background: #1a1a1a; border-bottom: 1px solid #333;
           padding: 12px 20px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
  h1 { font-size: 16px; margin: 0; font-weight: 600; }
  h2 { font-size: 15px; margin: 28px 20px 8px; color: var(--mut); font-weight: 600; }
  .sub { color: var(--mut); font-size: 13px; }
  button { background: #333; color: var(--fg); border: 1px solid #555; border-radius: 6px;
           padding: 6px 12px; cursor: pointer; font: inherit; }
  button:hover { background: #444; }
  #sheet { padding: 8px 20px 64px; }
  .card { display: grid; grid-template-columns: 180px 1fr 1fr; gap: 12px; align-items: start;
          padding: 12px 0; border-bottom: 1px solid #2a2a2a; cursor: pointer; }
  .card.on { background: #14281c; outline: 1px solid var(--acc); }
  .card.on.applied { background: #2a1810; outline: 1px solid var(--warn); }
  .card img { width: 100%; max-width: 320px; display: block; border-radius: 4px; background: #222; }
  .meta { font: 12px ui-monospace, monospace; color: var(--mut); }
  .meta b { color: var(--fg); font-size: 14px; }
  .tag { display: inline-block; margin-top: 8px; padding: 2px 8px; border-radius: 999px; font-size: 11px; }
  .tag.keep { background: #333; }
  .tag.inv { background: var(--acc); color: #111; font-weight: 600; }
  .tag.rev { background: var(--warn); color: #111; font-weight: 600; }
  .tag.ctrl { background: #3a3a00; color: #ee8; }
  .lbl { font-size: 11px; color: var(--mut); margin-bottom: 4px; }
</style>
<header>
  <div>
    <h1>Polarity review — round 2</h1>
    <div class="sub">__NAPP__ already inverted (spot-check; click = revert) &nbsp;·&nbsp;
      __NLEFT__ still look dark (click = invert) &nbsp;·&nbsp; j/k move, Space toggle</div>
  </div>
  <div class="sub" id="stats"></div>
  <button type="button" id="export">Export CSV</button>
  <button type="button" id="clear">Clear marks</button>
</header>
<div id="sheet"></div>
<script>
const ITEMS = __PAYLOAD__;
const KEY = "sluis-polarity-v2";
let marks = {};
try { marks = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { marks = {}; }
let focus = 0;

function save() { localStorage.setItem(KEY, JSON.stringify(marks)); stats(); }
function isOn(name) { return !!marks[name]; }
function stats() {
  const nInv = ITEMS.filter(it => it.kind === "candidate" && isOn(it.name)).length;
  const nRev = ITEMS.filter(it => it.kind === "applied" && isOn(it.name)).length;
  const nCand = ITEMS.filter(it => it.kind === "candidate").length;
  document.getElementById("stats").textContent =
    nInv + " new invert / " + nCand + " remaining  ·  " + nRev + " revert";
}
function tagFor(it) {
  if (it.kind === "control") return "<span class='tag ctrl'>control</span>";
  if (it.kind === "applied")
    return isOn(it.name) ? "<span class='tag rev'>REVERT</span>"
                         : "<span class='tag keep'>keep inverted</span>";
  return isOn(it.name) ? "<span class='tag inv'>INVERT</span>"
                       : "<span class='tag keep'>do not invert</span>";
}
function rightLabel(it) {
  return it.kind === "applied" ? "if reverted (do not want, unless mistake)" : "if inverted";
}
function render() {
  const applied = ITEMS.filter(it => it.kind === "applied");
  const rest = ITEMS.filter(it => it.kind !== "applied");
  const card = (it, i) => `<div class="card ${isOn(it.name)?"on":""} ${it.kind}" data-i="${i}" id="c${i}">
      <div class="meta"><b>${it.name}</b><br>dark ${it.dark_frac} · mean ${it.mean}<br>${tagFor(it)}</div>
      <div><div class="lbl">current</div><img src="data:image/jpeg;base64,${it.orig}"></div>
      <div><div class="lbl">${rightLabel(it)}</div><img src="data:image/jpeg;base64,${it.inv}"></div>
    </div>`;
  let html = "<h2>Already inverted — click only if this was a mistake (revert)</h2>";
  applied.forEach((it) => { html += card(it, ITEMS.indexOf(it)); });
  html += "<h2>Still a majority-dark page — click to invert</h2>";
  rest.forEach((it) => { html += card(it, ITEMS.indexOf(it)); });
  document.getElementById("sheet").innerHTML = html;
  document.querySelectorAll(".card").forEach(el => {
    el.addEventListener("click", () => toggle(+el.dataset.i));
  });
  highlight(); stats();
}
function toggle(i) {
  const name = ITEMS[i].name;
  if (ITEMS[i].kind === "control") return;
  if (marks[name]) delete marks[name]; else marks[name] = 1;
  focus = i; save(); render();
}
function highlight() {
  document.querySelectorAll(".card").forEach((el) => {
    const i = +el.dataset.i;
    el.style.boxShadow = i === focus ? "inset 3px 0 0 #6ae" : "";
  });
}
function goto(i) {
  focus = Math.max(0, Math.min(ITEMS.length - 1, i));
  highlight();
  const el = document.getElementById("c"+focus);
  if (el) el.scrollIntoView({block: "nearest"});
}
document.getElementById("export").onclick = () => {
  const lines = ["filename,action,kind,dark_frac"];
  ITEMS.forEach(it => {
    let action = "leave";
    if (isOn(it.name) && it.kind === "applied") action = "revert";
    if (isOn(it.name) && it.kind === "candidate") action = "invert";
    lines.push([it.name, action, it.kind, it.dark_frac].join(","));
  });
  const blob = new Blob([lines.join("\n")], {type: "text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "polarity_decisions_round2.csv";
  a.click();
};
document.getElementById("clear").onclick = () => {
  if (!confirm("Clear round-2 marks?")) return;
  marks = {}; save(); render();
};
document.addEventListener("keydown", e => {
  if (e.key === "j" || e.key === "ArrowDown") { e.preventDefault(); goto(focus + 1); }
  if (e.key === "k" || e.key === "ArrowUp") { e.preventDefault(); goto(focus - 1); }
  if (e.key === " " || e.key === "Enter") { e.preventDefault(); toggle(focus); }
});
render();
</script>
"""


if __name__ == "__main__":
    main()
