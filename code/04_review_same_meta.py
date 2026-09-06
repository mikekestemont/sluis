#!/usr/bin/env python3
"""Review consecutive gallery pages that share archive + fonds + date.

Stage 2 only collapsed series with a shelfmark. Unsigned same-day runs
(and a few signed pairs Stage 2 left) still sit in the stretched train
set. This sheet lists those runs so you can mark same-document (keep
lowest volgnummer) vs keep-all (distinct leaves / folios).

Thumbs prefer pages-zoned-stretched (the train crop), else pages-recto.

  python code/04_review_same_meta.py
  open http://127.0.0.1:8765/outputs/same_meta_review.html
"""
from __future__ import annotations

import base64
import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
STRETCHED = ROOT / "images" / "pages-zoned-stretched"
RECTO = ROOT / "images" / "pages-recto"
OUT_CLUSTERS = ROOT / "data" / "same_meta_clusters.csv"
OUT_HTML = ROOT / "outputs" / "same_meta_review.html"
THUMB = 360
WORKERS = 8


def date_of(r: dict) -> str:
    return f"{r.get('jaar') or ''}-{r.get('maand') or ''}-{r.get('dag') or ''}"


def meta_key(r: dict) -> tuple[str, str, str]:
    return ((r.get("archief") or "").strip(),
            (r.get("fonds") or "").strip(),
            date_of(r))


def jpeg_b64(path: Path, long=THUMB, quality=72) -> str:
    im = Image.open(path).convert("RGB")
    im.thumbnail((long, long))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def thumb_path(name: str) -> Path:
    stretched = STRETCHED / name
    if stretched.is_file():
        return stretched
    return RECTO / name


def consecutive_runs(gallery: list[dict]) -> list[list[dict]]:
    gallery = sorted(gallery, key=lambda r: int(r["volgnummer"]))
    runs: list[list[dict]] = []
    cur: list[dict] = []
    for r in gallery:
        if not cur:
            cur = [r]
            continue
        prev = cur[-1]
        consecutive = int(r["volgnummer"]) - int(prev["volgnummer"]) == 1
        if consecutive and meta_key(r) == meta_key(prev):
            cur.append(r)
        else:
            if len(cur) >= 2:
                runs.append(cur)
            cur = [r]
    if len(cur) >= 2:
        runs.append(cur)
    return runs


def pin_key(run: list[dict]) -> tuple[int, int]:
    extra = " ".join((r.get("extra_info") or "") for r in run).lower()
    reinaert = 0 if "reinaert" in extra or "reynaert" in extra else 1
    return (reinaert, int(run[0]["volgnummer"]))


def main() -> None:
    rows = [r for r in csv.DictReader(MANIFEST.open(encoding="utf-8"))
            if r.get("side") == "recto" and r.get("main_document") == "1"]
    by_png = {Path(r["released_path"]).name: r for r in rows}
    runs = consecutive_runs(rows)
    runs.sort(key=pin_key)

    names = []
    for run in runs:
        for r in run:
            names.append(Path(r["released_path"]).name)
    thumbs: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(jpeg_b64, thumb_path(n)): n for n in names}
        for fut in as_completed(futs):
            thumbs[futs[fut]] = fut.result()

    clusters = []
    payload = []
    for run in runs:
        members = [Path(r["released_path"]).name for r in run]
        keep = members[0]
        drop = members[1:]
        r0 = run[0]
        unsigned = not any((r.get("signatuur") or "").strip() for r in run)
        extra = (r0.get("extra_info") or "").strip()
        cid = f"m{r0['volgnummer']}"
        rec = {
            "cluster_id": cid,
            "n": len(members),
            "unsigned": int(unsigned),
            "members": "|".join(members),
            "keep_if_same": keep,
            "drop_if_same": "|".join(drop),
            "archief": r0.get("archief") or "",
            "fonds": r0.get("fonds") or "",
            "signatuur": " | ".join(
                dict.fromkeys((r.get("signatuur") or "").strip() or "—" for r in run)),
            "date": date_of(r0),
            "extra_info": extra,
        }
        clusters.append(rec)
        items = []
        for r in run:
            name = Path(r["released_path"]).name
            items.append({
                "name": name,
                "volgnummer": int(r["volgnummer"]),
                "signatuur": (r.get("signatuur") or "").strip(),
                "extra": (r.get("extra_info") or "").strip(),
                "thumb": thumbs.get(name, ""),
                "stretched": f"/images/pages-zoned-stretched/{name}",
                "recto": f"/images/pages-recto/{name}",
            })
        payload.append({
            "id": cid,
            "n": len(members),
            "unsigned": unsigned,
            "reinaert": pin_key(run)[0] == 0,
            "archief": rec["archief"],
            "fonds": rec["fonds"],
            "signatuur": rec["signatuur"],
            "date": rec["date"],
            "extra": extra,
            "keep": keep,
            "drop": drop,
            "items": items,
        })

    OUT_CLUSTERS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CLUSTERS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(clusters[0].keys()))
        w.writeheader()
        w.writerows(clusters)

    html = HTML.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    html = html.replace("__N__", str(len(payload)))
    html = html.replace("__PHOTOS__", str(sum(c["n"] for c in clusters)))
    html = html.replace("__GALLERY__", str(len(by_png)))
    extra = sum(c["n"] - 1 for c in clusters)
    html = html.replace("__MAXDROP__", str(extra))
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")

    print(f"gallery {len(by_png)}")
    print(f"clusters {len(clusters)}  photos in runs {sum(c['n'] for c in clusters)}")
    print(f"unsigned clusters {sum(c['unsigned'] for c in clusters)}")
    print(f"if all marked same: drop {extra}  gallery → {len(by_png) - extra}")
    print(f"clusters → {OUT_CLUSTERS}")
    print(f"review   → {OUT_HTML}  ({OUT_HTML.stat().st_size / 1e6:.1f} MB)")
    print("open     → http://127.0.0.1:8765/outputs/same_meta_review.html")
    for rec in clusters[:8]:
        print(f"  {rec['cluster_id']:8} n={rec['n']}  {rec['archief'][:36]:36}  "
              f"{rec['date']}  {rec['members']}")


HTML = r"""<!DOCTYPE html>
<meta charset="utf-8">
<title>Same archive+date review</title>
<style>
  :root { --fg:#eee; --mut:#9a9a9a; --acc:#6ae; --bg:#111; --pin:#c4a35a; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.4 system-ui, sans-serif; background: var(--bg); color: var(--fg); }
  header { position: sticky; top: 0; z-index: 2; background: #1a1a1a; border-bottom: 1px solid #333;
           padding: 12px 20px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
  h1 { font-size: 16px; margin: 0; font-weight: 600; }
  .sub { color: var(--mut); font-size: 13px; }
  button, select { background: #333; color: var(--fg); border: 1px solid #555; border-radius: 6px;
           padding: 6px 12px; cursor: pointer; font: inherit; }
  button:hover { background: #444; }
  button.keepbtn { background: var(--acc); color: #111; font-weight: 600; }
  #sheet { padding: 16px 20px 64px; }
  .card { padding: 14px 0; border-bottom: 1px solid #2a2a2a; }
  .card.on { background: #14281c; outline: 1px solid var(--acc); }
  .card.pin { outline: 1px solid var(--pin); }
  .meta { font: 12px ui-monospace, monospace; color: var(--mut); margin-bottom: 8px; cursor: pointer; }
  .meta b { color: var(--fg); font-size: 14px; }
  .thumbs { display: flex; gap: 10px; flex-wrap: wrap; }
  .thumbs figure { margin: 0; width: min(360px, 32vw); }
  .thumbs figure.drop { outline: 1px solid #c88; opacity: 0.72; }
  .thumbs img { width: 100%; display: block; border-radius: 4px; background: #222; cursor: zoom-in; }
  .thumbs figcaption { font: 11px ui-monospace, monospace; color: var(--mut); margin-top: 4px; }
  .thumbs figcaption .k { color: #8c8; }
  .thumbs figcaption .d { color: #c88; }
  .acts { display: flex; gap: 6px; margin-top: 6px; flex-wrap: wrap; align-items: center; }
  .acts button, .acts a, .acts select { font: 11px system-ui, sans-serif; padding: 3px 8px; border-radius: 4px;
    background: #2a2a2a; color: var(--fg); border: 1px solid #555; text-decoration: none; cursor: pointer; }
  .acts select { max-width: 148px; }
  .acts button.keepbtn { background: var(--acc); color: #111; }
  .tag { display: inline-block; margin-left: 8px; padding: 2px 8px; border-radius: 999px; font-size: 11px; }
  .tag.keep { background: #333; }
  .tag.same { background: var(--acc); color: #111; font-weight: 600; }
  .tag.mix { background: #c4a35a; color: #111; font-weight: 600; }
  .tag.pin { background: var(--pin); color: #111; font-weight: 600; }
  #lb { display: none; position: fixed; inset: 0; z-index: 10; background: rgba(0,0,0,.92);
        flex-direction: column; align-items: center; justify-content: center; padding: 16px; }
  #lb.on { display: flex; }
  #lb img { max-width: 96vw; max-height: calc(100vh - 88px); object-fit: contain; background: #111; }
  #lb .bar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 10px;
             color: var(--mut); font: 13px ui-monospace, monospace; }
  #lb .bar b { color: var(--fg); }
  #lb .bar a { color: var(--fg); }
</style>
<header>
  <div>
    <h1>Same archive + date — consecutive gallery runs</h1>
    <div class="sub">__N__ clusters, __PHOTOS__ photos (of __GALLERY__ gallery).
      Default = <b>keep all</b> as separate docs.
      Per photo: <b>Own doc</b> or <b>Dup of …</b> (merge a subset; leave the rest).
      Space / title = merge <em>all</em> into the first photo (then Own-doc the odd one out).
      Click the photo to enlarge. ↗ new tab. j / k move.</div>
  </div>
  <div class="sub" id="stats"></div>
  <select id="filter">
    <option value="all">all clusters</option>
    <option value="unsigned">unsigned only</option>
    <option value="signed">has shelfmark</option>
    <option value="reinaert">Reinaert only</option>
  </select>
  <button type="button" id="export">Export CSV</button>
  <button type="button" id="clear">Clear marks</button>
</header>
<div id="sheet"></div>
<div id="lb">
  <img id="lbimg" alt="">
  <div class="bar">
    <b id="lbtitle"></b>
    <button type="button" class="keepbtn" id="lbkeep">Own doc</button>
    <label>Dup of <select id="lbdup"></select></label>
    <a id="lbstretch" target="_blank" rel="noopener">↗ train crop</a>
    <a id="lbrecto" target="_blank" rel="noopener">↗ full recto</a>
    <span>← → · Esc</span>
    <button type="button" id="lbclose">Close</button>
  </div>
</div>
<script>
const ITEMS = __PAYLOAD__;
const KEY = "sluis-same-meta-v1";
let marks = {};
try { marks = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { marks = {}; }
let focus = 0;
let filter = "all";
let lb = null;
function save() { localStorage.setItem(KEY, JSON.stringify(marks)); stats(); }
function itemById(id) { return ITEMS.find(x => x.id === id); }
function dupsOf(id) {
  const raw = marks[id];
  if (!raw) return {};
  if (raw.dups) return Object.assign({}, raw.dups);
  if (raw.keep) {
    const it = itemById(id);
    const d = {};
    (it ? it.items : []).forEach(p => { if (p.name !== raw.keep) d[p.name] = raw.keep; });
    return d;
  }
  return {};
}
function nDups(id) { return Object.keys(dupsOf(id)).length; }
function setDups(id, dups) {
  const keys = Object.keys(dups);
  if (!keys.length) delete marks[id];
  else marks[id] = { dups };
}
function visible() {
  return ITEMS.filter(it => {
    if (filter === "unsigned") return it.unsigned;
    if (filter === "signed") return !it.unsigned;
    if (filter === "reinaert") return it.reinaert;
    return true;
  });
}
function stats() {
  const vis = visible();
  const m = vis.filter(it => nDups(it.id)).length;
  const drop = vis.reduce((n, it) => n + nDups(it.id), 0);
  document.getElementById("stats").textContent =
    m + " clusters with merges / " + vis.length + " shown · would drop " + drop;
}
function tagFor(it) {
  const d = dupsOf(it.id);
  const n = Object.keys(d).length;
  if (!n) return "<span class='tag keep'>keep all</span>";
  if (n === it.n - 1) {
    const keep = it.items.map(p => p.name).find(n => !d[n]);
    return "<span class='tag same'>all same → keep " + keep + "</span>";
  }
  return "<span class='tag mix'>merge " + n + " · keep " + (it.n - n) + "</span>";
}
function dupSelect(it, name, current) {
  let opts = "<option value=''>own doc</option>";
  it.items.forEach(p => {
    if (p.name === name) return;
    opts += "<option value='" + p.name + "'" + (current === p.name ? " selected" : "") + ">dup of " + p.name + "</option>";
  });
  return "<select data-act='dup'>" + opts + "</select>";
}
function render() {
  const vis = visible();
  const sheet = document.getElementById("sheet");
  sheet.innerHTML = vis.map((it, i) => {
    const d = dupsOf(it.id);
    const pin = it.reinaert ? "<span class='tag pin'>Reinaert</span>" : "";
    const figs = it.items.map((p, pi) => {
      const target = d[p.name] || "";
      const cap = target
        ? "<span class='d'>drop " + p.name + " → " + target + "</span>"
        : "<span class='k'>KEEP " + p.name + "</span>";
      const sig = p.signatuur ? " · " + p.signatuur : "";
      const extra = p.extra ? "<br>" + p.extra : "";
      return "<figure class='" + (target ? "drop" : "") + "' data-name='" + p.name + "' data-pi='" + pi + "'>"
        + "<img src='data:image/jpeg;base64," + p.thumb + "' alt='" + p.name + "'>"
        + "<figcaption>" + cap + sig + extra + "</figcaption>"
        + "<div class='acts'>"
        + "<button type='button' class='" + (target ? "" : "keepbtn") + "' data-act='own'>Own doc</button>"
        + dupSelect(it, p.name, target)
        + "<a href='" + p.stretched + "' target='_blank' rel='noopener'>↗ crop</a>"
        + "<a href='" + p.recto + "' target='_blank' rel='noopener'>↗ recto</a>"
        + "</div></figure>";
    }).join("");
    return `<div class="card ${nDups(it.id)?"on":""} ${it.reinaert?"pin":""}" data-i="${i}" id="c${i}">
      <div class="meta"><b>${it.id}</b> · n=${it.n} · ${it.archief} ${it.fonds} · ${it.date}${pin}${tagFor(it)}<br>
      shelfmark: ${it.signatuur}${it.extra ? " · " + it.extra : ""}</div>
      <div class="thumbs">${figs}</div>
    </div>`;
  }).join("");
  sheet.querySelectorAll(".card").forEach(el => {
    const i = +el.dataset.i;
    el.querySelector(".meta").addEventListener("click", () => toggleAll(i));
    el.querySelectorAll("figure").forEach(fig => {
      const name = fig.dataset.name;
      const pi = +fig.dataset.pi;
      fig.querySelector("img").addEventListener("click", e => { e.stopPropagation(); openLb(i, pi); });
      fig.querySelector("[data-act=own]").addEventListener("click", e => { e.stopPropagation(); setOwn(i, name); });
      fig.querySelector("[data-act=dup]").addEventListener("change", e => { e.stopPropagation(); setDup(i, name, e.target.value); });
      fig.querySelector("[data-act=dup]").addEventListener("click", e => e.stopPropagation());
      fig.querySelectorAll("a").forEach(a => a.addEventListener("click", e => e.stopPropagation()));
    });
  });
  highlight(); stats();
}
function toggleAll(i) {
  const vis = visible();
  const it = vis[i];
  if (!it) return;
  if (nDups(it.id) === it.n - 1) setDups(it.id, {});
  else {
    const keep = it.items[0].name;
    const d = {};
    it.items.slice(1).forEach(p => { d[p.name] = keep; });
    setDups(it.id, d);
  }
  focus = i; save(); render();
}
function setOwn(i, name) {
  const vis = visible();
  const it = vis[i];
  const d = dupsOf(it.id);
  delete d[name];
  setDups(it.id, d);
  focus = i; save(); render();
  if (lb) showLb();
}
function setDup(i, name, target) {
  const vis = visible();
  const it = vis[i];
  const d = dupsOf(it.id);
  if (!target) delete d[name];
  else {
    d[name] = target;
    Object.keys(d).forEach(k => { if (d[k] === name) d[k] = target; });
    delete d[target];
  }
  setDups(it.id, d);
  focus = i; save(); render();
  if (lb) showLb();
}
function highlight() {
  document.querySelectorAll(".card").forEach((el, i) => {
    el.style.boxShadow = i === focus ? "inset 3px 0 0 #6ae" : "";
  });
}
function goto(i) {
  const vis = visible();
  focus = Math.max(0, Math.min(vis.length - 1, i));
  highlight();
  document.getElementById("c"+focus)?.scrollIntoView({block: "nearest"});
}
function openLb(i, pi) {
  lb = { i, pi };
  showLb();
}
function showLb() {
  if (!lb) return;
  const vis = visible();
  const it = vis[lb.i];
  if (!it) return closeLb();
  lb.pi = Math.max(0, Math.min(it.items.length - 1, lb.pi));
  const p = it.items[lb.pi];
  const d = dupsOf(it.id);
  const target = d[p.name] || "";
  document.getElementById("lbimg").src = p.stretched;
  document.getElementById("lbtitle").textContent =
    it.id + " · " + p.name + (target ? " · dup of " + target : " · own doc") + " · " + (lb.pi+1) + "/" + it.items.length;
  document.getElementById("lbstretch").href = p.stretched;
  document.getElementById("lbrecto").href = p.recto;
  const sel = document.getElementById("lbdup");
  sel.innerHTML = dupSelect(it, p.name, target).replace(/^<select[^>]*>/, "").replace(/<\/select>$/, "");
  sel.value = target;
  document.getElementById("lb").classList.add("on");
}
function closeLb() {
  lb = null;
  document.getElementById("lb").classList.remove("on");
  document.getElementById("lbimg").src = "";
}
document.getElementById("lbclose").onclick = closeLb;
document.getElementById("lb").addEventListener("click", e => { if (e.target.id === "lb") closeLb(); });
document.getElementById("lbkeep").onclick = () => {
  if (!lb) return;
  setOwn(lb.i, visible()[lb.i].items[lb.pi].name);
};
document.getElementById("lbdup").onchange = e => {
  if (!lb) return;
  setDup(lb.i, visible()[lb.i].items[lb.pi].name, e.target.value);
};
document.getElementById("export").onclick = () => {
  const lines = ["cluster_id,same,keep,drop,drop_map,members,archief,fonds,signatuur,date,extra_info,unsigned"];
  ITEMS.forEach(it => {
    const d = dupsOf(it.id);
    const dropped = Object.keys(d);
    const same = dropped.length ? 1 : 0;
    const keepers = it.items.map(p => p.name).filter(n => !d[n]);
    const drop_map = dropped.map(n => n + "=" + d[n]).join("|");
    const members = it.items.map(p => p.name).join("|");
    lines.push([it.id, same, keepers.join("|"), dropped.join("|"), drop_map, members,
                JSON.stringify(it.archief), JSON.stringify(it.fonds),
                JSON.stringify(it.signatuur), it.date,
                JSON.stringify(it.extra), it.unsigned ? 1 : 0].join(","));
  });
  const blob = new Blob([lines.join("\n")], {type: "text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "same_meta_decisions.csv";
  a.click();
};
document.getElementById("clear").onclick = () => {
  if (!confirm("Clear all same-document marks?")) return;
  marks = {}; save(); render();
};
document.getElementById("filter").onchange = e => {
  filter = e.target.value; focus = 0; closeLb(); render();
};
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (lb) {
    if (e.key === "Escape") { e.preventDefault(); closeLb(); return; }
    if (e.key === "ArrowLeft") { e.preventDefault(); lb.pi--; showLb(); return; }
    if (e.key === "ArrowRight") { e.preventDefault(); lb.pi++; showLb(); return; }
    if (e.key === "o") {
      e.preventDefault();
      setOwn(lb.i, visible()[lb.i].items[lb.pi].name);
    }
    return;
  }
  if (e.key === "j" || e.key === "ArrowDown") { e.preventDefault(); goto(focus + 1); }
  if (e.key === "k" || e.key === "ArrowUp") { e.preventDefault(); goto(focus - 1); }
  if (e.key === " " || e.key === "Enter") { e.preventDefault(); toggleAll(focus); }
});
render();
</script>
"""


if __name__ == "__main__":
    main()
