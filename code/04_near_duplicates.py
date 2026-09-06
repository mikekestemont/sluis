#!/usr/bin/env python3
"""Perceptual-hash near-duplicates among gallery rectos.

256-bit difference hash on pages-recto/. Known metadata series are *not*
pixel-near (Hamming 49–143); this pass finds unsigned photographic twins.

Writes:
  data/neardup_pairs.csv
  data/neardup_clusters.csv
  outputs/neardup_review.html

Nothing is dropped until you mark clusters SAME in the HTML and we apply.
"""
from __future__ import annotations

import base64
import csv
import io
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
RECTO = ROOT / "images" / "pages-recto"
MANIFEST = ROOT / "data" / "manifest.csv"
OUT_PAIRS = ROOT / "data" / "neardup_pairs.csv"
OUT_CLUSTERS = ROOT / "data" / "neardup_clusters.csv"
OUT_HTML = ROOT / "outputs" / "neardup_review.html"

HASH_W, HASH_H = 16, 16  # 256-bit horizontal dHash
HAM_MAX = 32  # tight: photographic twins, not same-day different charters
THUMB = 280
WORKERS = 8


def dhash_bytes(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L")
    small = im.resize((HASH_W + 1, HASH_H), Image.BILINEAR)
    bits = (np.asarray(small, np.int16)[:, 1:] > np.asarray(small, np.int16)[:, :-1]).ravel()
    return np.packbits(bits)


def jpeg_b64(path: Path, long=THUMB, quality=72) -> str:
    im = Image.open(path).convert("RGB")
    im.thumbnail((long, long))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def union_find(n: int):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    return find, union


def main() -> None:
    rows = [r for r in csv.DictReader(MANIFEST.open(encoding="utf-8")) if r["side"] == "recto"]
    by = {Path(r["released_path"]).name: r for r in rows}
    main_docs = {n for n, r in by.items() if r.get("main_document") == "1"}

    files = sorted(RECTO.glob("*.png"))
    hashes: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(dhash_bytes, p): p.name for p in files}
        for fut in as_completed(futs):
            hashes[futs[fut]] = fut.result()

    names = sorted(hashes)
    H = np.stack([hashes[n] for n in names])
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    ham = lut[np.bitwise_xor(H[:, None, :], H[None, :, :])].sum(axis=2).astype(np.uint16)
    np.fill_diagonal(ham, 999)
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)

    series_pairs = set()
    groups = defaultdict(list)
    for r in rows:
        sf = (r.get("series_first") or "").strip()
        if sf:
            groups[sf].append(Path(r["released_path"]).name)
    for members in groups.values():
        members = sorted(members)
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                series_pairs.add(frozenset((a, b)))

    series_d = sorted(
        int(ham[idx[a], idx[b]])
        for a, b in (tuple(p) for p in series_pairs)
        if a in idx and b in idx
    )

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = int(ham[i, j])
            if d > HAM_MAX:
                continue
            a, b = names[i], names[j]
            if frozenset((a, b)) in series_pairs:
                continue
            if a not in main_docs or b not in main_docs:
                continue
            ra, rb = by[a], by[b]
            pairs.append({
                "hamming": d,
                "a": a,
                "b": b,
                "delta_volgnummer": abs(int(ra["volgnummer"]) - int(rb["volgnummer"])),
                "archief": ra.get("archief", ""),
                "fonds": ra.get("fonds", ""),
                "signatuur_a": ra.get("signatuur", ""),
                "signatuur_b": rb.get("signatuur", ""),
                "date_a": f"{ra.get('jaar','')}-{ra.get('maand','')}-{ra.get('dag','')}",
                "date_b": f"{rb.get('jaar','')}-{rb.get('maand','')}-{rb.get('dag','')}",
            })
    pairs.sort(key=lambda p: (p["hamming"], p["a"], p["b"]))

    involved = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
    loc = {name: i for i, name in enumerate(involved)}
    find, union = union_find(len(involved))
    for p in pairs:
        union(loc[p["a"]], loc[p["b"]])
    buckets: dict[int, list[str]] = defaultdict(list)
    for name in involved:
        buckets[find(loc[name])].append(name)
    clusters = []
    for members in buckets.values():
        members = sorted(members, key=lambda x: int(by[x]["volgnummer"]))
        ds = [
            int(ham[idx[members[i]], idx[members[j]]])
            for i in range(len(members))
            for j in range(i + 1, len(members))
        ]
        keep = members[0]
        drop = members[1:]
        r0 = by[keep]
        clusters.append({
            "cluster_id": f"c{keep.replace('o.png','')}",
            "n": len(members),
            "min_hamming": min(ds),
            "max_hamming": max(ds),
            "members": "|".join(members),
            "keep_if_same": keep,
            "drop_if_same": "|".join(drop),
            "archief": r0.get("archief", ""),
            "fonds": r0.get("fonds", ""),
            "signatuur": r0.get("signatuur", ""),
            "date": f"{r0.get('jaar','')}-{r0.get('maand','')}-{r0.get('dag','')}",
            "_members": members,
        })
    clusters.sort(key=lambda c: (c["min_hamming"], c["_members"][0]))

    OUT_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PAIRS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pairs[0].keys()) if pairs else ["hamming"])
        w.writeheader()
        w.writerows(pairs)
    with OUT_CLUSTERS.open("w", newline="", encoding="utf-8") as fh:
        fields = [k for k in clusters[0].keys() if not k.startswith("_")] if clusters else []
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for c in clusters:
            w.writerow({k: c[k] for k in fields})

    payload = []
    for c in clusters:
        items = []
        for name in c["_members"]:
            r = by[name]
            items.append({
                "name": name,
                "volgnummer": int(r["volgnummer"]),
                "thumb": jpeg_b64(RECTO / name),
                "role": "keep" if name == c["keep_if_same"] else "drop",
            })
        payload.append({
            "id": c["cluster_id"],
            "min_ham": c["min_hamming"],
            "max_ham": c["max_hamming"],
            "archief": c["archief"],
            "fonds": c["fonds"],
            "date": c["date"],
            "keep": c["keep_if_same"],
            "drop": c["drop_if_same"].split("|") if c["drop_if_same"] else [],
            "items": items,
        })

    html = r"""<!DOCTYPE html>
<meta charset="utf-8">
<title>Near-duplicate review</title>
<style>
  :root { --fg:#eee; --mut:#9a9a9a; --acc:#6ae; --bg:#111; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.4 system-ui, sans-serif; background: var(--bg); color: var(--fg); }
  header { position: sticky; top: 0; z-index: 2; background: #1a1a1a; border-bottom: 1px solid #333;
           padding: 12px 20px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
  h1 { font-size: 16px; margin: 0; font-weight: 600; }
  .sub { color: var(--mut); font-size: 13px; }
  button { background: #333; color: var(--fg); border: 1px solid #555; border-radius: 6px;
           padding: 6px 12px; cursor: pointer; font: inherit; }
  button:hover { background: #444; }
  #sheet { padding: 16px 20px 64px; }
  .card { padding: 14px 0; border-bottom: 1px solid #2a2a2a; cursor: pointer; }
  .card.on { background: #14281c; outline: 1px solid var(--acc); }
  .meta { font: 12px ui-monospace, monospace; color: var(--mut); margin-bottom: 8px; }
  .meta b { color: var(--fg); font-size: 14px; }
  .thumbs { display: flex; gap: 10px; flex-wrap: wrap; }
  .thumbs figure { margin: 0; width: min(280px, 30vw); }
  .thumbs img { width: 100%; display: block; border-radius: 4px; background: #222; }
  .thumbs figcaption { font: 11px ui-monospace, monospace; color: var(--mut); margin-top: 4px; }
  .thumbs figcaption .k { color: #8c8; }
  .thumbs figcaption .d { color: #c88; }
  .tag { display: inline-block; margin-left: 8px; padding: 2px 8px; border-radius: 999px; font-size: 11px; }
  .tag.keep { background: #333; }
  .tag.same { background: var(--acc); color: #111; font-weight: 600; }
</style>
<header>
  <div>
    <h1>Near-duplicate review</h1>
    <div class="sub">Click a cluster or press <b>Space</b> to mark <b>same leaf</b>
      (keep lowest volgnummer, drop the rest). Unmarked = keep all.
      <span style="color:#666">j / k</span> move.</div>
  </div>
  <div class="sub" id="stats"></div>
  <button type="button" id="export">Export CSV</button>
  <button type="button" id="clear">Clear marks</button>
</header>
<div id="sheet"></div>
<script>
const ITEMS = __PAYLOAD__;
const KEY = "sluis-neardup-v1";
let marks = {};
try { marks = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { marks = {}; }
let focus = 0;
function save() { localStorage.setItem(KEY, JSON.stringify(marks)); stats(); }
function isOn(id) { return !!marks[id]; }
function stats() {
  const m = ITEMS.filter(it => isOn(it.id)).length;
  document.getElementById("stats").textContent = m + " marked same / " + ITEMS.length + " clusters";
}
function render() {
  const sheet = document.getElementById("sheet");
  sheet.innerHTML = ITEMS.map((it, i) => {
    const on = isOn(it.id);
    const tag = on
      ? "<span class='tag same'>SAME leaf → keep " + it.keep + "</span>"
      : "<span class='tag keep'>keep all</span>";
    const figs = it.items.map(p => {
      const cap = on
        ? "<span class='" + (p.name === it.keep ? "k" : "d") + "'>" + (p.name === it.keep ? "KEEP " : "drop ") + p.name + "</span>"
        : p.name;
      return "<figure><img src='data:image/jpeg;base64," + p.thumb + "'><figcaption>" + cap + "</figcaption></figure>";
    }).join("");
    return `<div class="card ${on?"on":""}" data-i="${i}" id="c${i}">
      <div class="meta"><b>${it.id}</b> · ham ${it.min_ham}–${it.max_ham} · ${it.archief} ${it.fonds} · ${it.date}${tag}<br>
      if same: keep ${it.keep}, drop ${it.drop.join(", ")}</div>
      <div class="thumbs">${figs}</div>
    </div>`;
  }).join("");
  sheet.querySelectorAll(".card").forEach(el => el.addEventListener("click", () => toggle(+el.dataset.i)));
  highlight(); stats();
}
function toggle(i) {
  const id = ITEMS[i].id;
  if (marks[id]) delete marks[id]; else marks[id] = 1;
  focus = i; save(); render();
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
  const lines = ["cluster_id,same,keep,drop,min_hamming,max_hamming,members,archief,date"];
  ITEMS.forEach(it => {
    const same = isOn(it.id) ? 1 : 0;
    const keep = same ? it.keep : it.items.map(p => p.name).join("|");
    const drop = same ? it.drop.join("|") : "";
    const members = it.items.map(p => p.name).join("|");
    lines.push([it.id, same, keep, drop, it.min_ham, it.max_ham, members,
                JSON.stringify(it.archief), it.date].join(","));
  });
  const blob = new Blob([lines.join("\n")], {type: "text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "neardup_decisions.csv";
  a.click();
};
document.getElementById("clear").onclick = () => {
  if (!confirm("Clear all same-leaf marks?")) return;
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
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html.replace("__PAYLOAD__", json.dumps(payload)), encoding="utf-8")

    print(f"rectos hashed {n}")
    print(f"known series pairs {len(series_pairs)}  ham min/median/max "
          f"{series_d[0]}/{series_d[len(series_d)//2]}/{series_d[-1]}")
    print(f"new pairs ham<={HAM_MAX}: {len(pairs)}")
    print(f"clusters: {len(clusters)}")
    extra = sum(c["n"] - 1 for c in clusters)
    print(f"if all marked same: drop {extra}  gallery {len(main_docs)} → {len(main_docs) - extra}")
    print(f"pairs    → {OUT_PAIRS}")
    print(f"clusters → {OUT_CLUSTERS}")
    print(f"review   → {OUT_HTML}  ({OUT_HTML.stat().st_size / 1e6:.1f} MB)")
    for c in clusters:
        print(f"  {c['cluster_id']:8} n={c['n']} ham={c['min_hamming']:2d}–{c['max_hamming']:2d}  "
              f"{c['members']}  {c['archief'][:28]} {c['date']}")


if __name__ == "__main__":
    main()
