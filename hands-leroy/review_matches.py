#!/usr/bin/env python3
"""Build a two-way match review sheet (photo→Gysseling and Gysseling→photos).

Needs match_topk.json from match_charters.py. Makes large readable JPEG
thumbs (long side 1800) so the hand is actually legible.

  python hands-leroy/match_charters.py --transcriptions hands-leroy/transcriptions-zoned
  python hands-leroy/review_matches.py
  python hands-leroy/review_matches.py --serve
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image
from tqdm import tqdm

from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TOPK = HERE / "match_topk.json"
STRETCHED = ROOT / "images" / "pages-zoned-stretched"
OUT_DIR = ROOT / "outputs"
HTML = OUT_DIR / "match_review.html"
THUMBS = OUT_DIR / "match-thumbs"
THUMB_LONG = 1800
THUMB_Q = 82
METADATA = ROOT / "images" / "metadata.xlsx"


def jpeg_thumb(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB")
    w, h = im.size
    scale = THUMB_LONG / max(w, h)
    if scale < 1:
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    im.save(dest, "JPEG", quality=THUMB_Q, optimize=True)


def build_thumbs(photo_ids: list[str]) -> None:
    THUMBS.mkdir(parents=True, exist_ok=True)
    todo = []
    for pid in photo_ids:
        src = STRETCHED / f"{pid}.png"
        dest = THUMBS / f"{pid}.jpg"
        if src.is_file() and not dest.is_file():
            todo.append((src, dest))
    if not todo:
        print(f"thumbs already present in {THUMBS}")
        return
    print(f"writing {len(todo)} JPEG thumbs (long side {THUMB_LONG})…")
    for src, dest in tqdm(todo):
        jpeg_thumb(src, dest)


def photo_years(path: Path) -> dict[str, int]:
    """Year lives on the dorse (m) row; map it onto the recto photo id."""
    if not path.is_file():
        return {}
    import pandas as pd
    df = pd.read_excel(path, usecols=["volgnummer", "bestandsnaam", "jaar"])
    year_by_volg: dict[int, int] = {}
    o_by_volg: dict[int, str] = {}
    for volg, name, jaar in df.itertuples(index=False):
        try:
            volg_i = int(volg)
        except (TypeError, ValueError):
            continue
        stem = Path(str(name)).stem
        if pd.notna(jaar):
            try:
                year_by_volg[volg_i] = int(jaar)
            except (TypeError, ValueError):
                pass
        if stem.lower().endswith("o"):
            o_by_volg[volg_i] = stem
    return {stem: year_by_volg[v] for v, stem in o_by_volg.items() if v in year_by_volg}


def slim_payload(raw: dict, show_k: int) -> dict:
    corpus_texts = raw.get("corpus_texts") or {}
    years = photo_years(METADATA)
    photos = []
    htr = {}
    for r in raw["photos"]:
        pid = Path(r["transcription"]).stem
        htr[pid] = r.get("htr") or ""
        neigh = []
        for n in (r.get("neighbors") or []):
            if not n.get("hand_group"):
                continue
            if len(neigh) >= show_k:
                break
            neigh.append({
                "id": n["corpus"],
                "score": n["score"],
                "lev": n.get("lev", n["score"]),
                "partial": n.get("partial", 0),
                "tfidf": n.get("tfidf", 0),
                "length_ratio": n.get("length_ratio", 0),
                "gys": n.get("gysseling_nr") or Path(n["corpus"]).stem,
                "hand": n.get("hand_group") or "",
            })
        photos.append({
            "id": pid,
            "empty": bool(r.get("empty")),
            "score": r.get("match_score") or 0,
            "lev": r.get("lev") or 0,
            "partial": r.get("partial") or 0,
            "margin": r.get("margin") or 0,
            "length_ratio": r.get("length_ratio") or 0,
            "length_mismatch": bool(r.get("length_mismatch")),
            "low_margin": bool(r.get("low_margin")),
            "collision": bool(r.get("collision")),
            "n_photos": r.get("n_photos_for_gys") or 0,
            "dup_of": r.get("dup_of") or "",
            "accept_via": r.get("accept_via") or "",
            "match": r.get("match") or "",
            "hand": r.get("hand_group") or "",
            "year": years.get(pid),
            "neighbors": neigh,
        })
    # inverse: corpus -> photos that retrieved it
    inv: dict[str, list] = {c: [] for c in corpus_texts}
    for p in photos:
        for i, n in enumerate(p["neighbors"]):
            inv.setdefault(n["id"], []).append({
                "id": p["id"], "score": n["score"], "rank": i + 1,
            })
    for hits in inv.values():
        hits.sort(key=lambda x: (-x["score"], x["rank"]))
    corpus = []
    for cid in sorted(corpus_texts):
        hits = inv.get(cid) or []
        corpus.append({
            "id": cid,
            "gys": Path(cid).stem,
            "n": len(hits),
            "hits": hits[:show_k],
        })
    texts = dict(corpus_texts)
    return {
        "meta": raw.get("meta") or {},
        "show_k": show_k,
        "photos": photos,
        "htr": htr,
        "corpus": corpus,
        "texts": texts,
        "unmatched_preview": {},
        "thumb": "match-thumbs/{id}.jpg",
        "full": "../images/pages-zoned-stretched/{id}.png",
    }


def load_decisions(path: Path) -> dict:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8").replace("\\n", "\n")
    out = {}
    for i, line in enumerate(text.splitlines()):
        if i == 0 or not line.strip():
            continue
        parts = line.split(",")
        pid = parts[0].strip()
        if not pid or pid == "id":
            continue
        corpus = parts[2].strip() if len(parts) > 2 else ""
        note = ",".join(parts[3:]).strip() if len(parts) > 3 else ""
        out[pid] = {"corpus": corpus, "note": note}
    return out


def write_html(payload: dict, seed: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    seed_blob = json.dumps(seed, ensure_ascii=False, separators=(",", ":"))
    HTML.write_text(
        HTML_TMPL.replace("__PAYLOAD__", blob).replace("__SEED__", seed_blob),
        encoding="utf-8",
    )
    print(f"review → {HTML}  ({HTML.stat().st_size / 1e6:.1f} MB)  seed {len(seed)} marks")


def serve(port: int, index=None) -> None:
    from incipit_search import search as corpus_search

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(ROOT), **kw)

        def end_headers(self):
            if self.path.split("?", 1)[0].endswith(".html"):
                self.send_header("Cache-Control", "no-store, must-revalidate")
            super().end_headers()

        def log_message(self, fmt, *args):
            path = str(args[0]) if args else ""
            if path.startswith("GET /outputs/match-thumbs") or path.startswith("GET /search"):
                return
            super().log_message(fmt, *args)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/search", "/incipit"):
                qs = parse_qs(u.query)
                q = (qs.get("q") or [""])[0]
                k = int((qs.get("k") or ["12"])[0])
                year_s = (qs.get("year") or [""])[0]
                year = int(year_s) if year_s.isdigit() else None
                if index is None:
                    body = json.dumps({"error": "search index not loaded"}).encode()
                    self.send_response(503)
                else:
                    hits = corpus_search(
                        index, q, k=k,
                        leroy_only=(qs.get("leroy") or [""])[0] in {"1", "true"},
                        year=year,
                        hide_taken=False,
                    )
                    body = json.dumps({"q": q, "hits": hits}, ensure_ascii=False).encode("utf-8")
                    self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

    url = f"http://127.0.0.1:{port}/outputs/match_review.html"
    n = len(index["rows"]) if index else 0
    print(f"serving {ROOT}  (fuzzy index: {n} charters)\nopen {url}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


HTML_TMPL = r"""<!doctype html>
<meta charset=utf-8>
<title>Match review — photo ↔ Gysseling</title>
<meta http-equiv="Cache-Control" content="no-store">
<style>
  :root { --bg:#111; --fg:#eee; --mut:#8aa; --acc:#3ddc84; --warn:#e0c36e; --bad:#f07178; --line:#2a2a2a; }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--fg);
               font: 14px/1.4 system-ui, sans-serif; }
  header { position: sticky; top: 0; z-index: 3; background: #1a1a1a; border-bottom: 1px solid #333;
           padding: 8px 14px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  h1 { font-size: 14px; margin: 0; font-weight: 600; }
  .sub { color: var(--mut); font-size: 12px; }
  button, select, input { background: #333; color: var(--fg); border: 1px solid #555; border-radius: 6px;
           padding: 5px 10px; font: inherit; }
  button.on { background: var(--acc); color: #111; font-weight: 600; }
  input[type=search] { width: 160px; }
  #layout { display: grid; grid-template-columns: 230px 1fr; height: calc(100vh - 52px); }
  #index { overflow: auto; border-right: 1px solid var(--line); padding: 6px 0; }
  .ix { display: block; padding: 4px 10px; cursor: pointer; font: 12px ui-monospace, monospace;
        color: var(--mut); border-left: 3px solid transparent; white-space: nowrap; overflow: hidden; }
  .ix:hover { background: #1c1c1c; color: var(--fg); }
  .ix.cur { background: #1e2a22; color: var(--fg); border-left-color: var(--acc); }
  .ix .s { float: right; color: var(--mut); }
  .ix.ok .s { color: var(--acc); }
  .ix.mid .s { color: var(--warn); }
  .ix.bad .s { color: var(--bad); }
  .ix.mark { outline: 1px solid var(--acc); }
  #main { overflow: auto; padding: 10px 14px 40px; }
  .pane { display: grid; grid-template-columns: minmax(420px, 1.15fr) minmax(380px, 1fr); gap: 14px; align-items: start; }
  img.page { width: 100%; max-height: 52vh; object-fit: contain; object-position: top;
             background: #000; border-radius: 4px; display: block; }
  textarea { width: 100%; min-height: 64px; background: #222; color: var(--fg);
             border: 1px solid #555; border-radius: 6px; padding: 8px 10px;
             font: 14px/1.4 ui-monospace, Menlo, monospace; margin-top: 8px; }
  .snip-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin: 8px 0 4px; }
  input[type=number] { width: 78px; }
  .col { min-width: 0; }
  .htr, .gys { white-space: pre-wrap; font: 13px/1.45 ui-monospace, Menlo, monospace;
               background: #1a1a1a; border: 1px solid var(--line); border-radius: 6px;
               padding: 10px 12px; max-height: 36vh; overflow: auto; }
  .nb { border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; margin: 8px 0; cursor: pointer; }
  .nb:hover { border-color: #666; }
  .nb.pick { border-color: var(--acc); background: #14281c; }
  .nb.taken { opacity: 0.55; }
  .nb .full { margin-left: auto; }
  .nb .hd { display: flex; gap: 10px; align-items: baseline; font: 12px ui-monospace, monospace; color: var(--mut); }
  .nb .hd b { color: var(--fg); font-size: 13px; }
  .bar { height: 4px; background: #333; border-radius: 2px; margin: 6px 0 8px; }
  .bar > i { display: block; height: 100%; background: var(--acc); border-radius: 2px; }
  .gys-sm { white-space: pre-wrap; font: 12px/1.4 ui-monospace, monospace; max-height: 5.8em; overflow: auto; color: #ccc; }
  .thumbs { display: grid; grid-template-columns: 1fr; gap: 16px; }
  .thumbs img { width: 100%; max-height: 70vh; object-fit: contain; background: #000; border-radius: 4px; }
  .tag { display: inline-block; padding: 1px 7px; border-radius: 999px; background: #333; font-size: 11px; }
  .tag.none { background: var(--bad); color: #111; font-weight: 700; }
  .tag.auto-none { background: #5a3030; color: #fcc; }
  .status { font-size: 13px; margin: 0 0 8px; }
  button.none-on { background: var(--bad); color: #111; font-weight: 600; }
  a { color: #9cf; }
  .empty { color: var(--bad); }
  .ix.nomatch .s { color: var(--bad); font-weight: 700; }
  .ldiff { width: 100%; margin: 0 0 14px; padding: 8px 0 4px;
           border-bottom: 1px solid var(--line); }
  .ldiff h3 { font-size: 13px; color: var(--mut); margin: 0 0 8px; font-weight: 600; }
  .ldiff .gap { color: var(--mut); text-align: center; padding: 7px 8px; margin: 6px 0 10px;
                font-size: 12px; border-top: 1px dashed #333; border-bottom: 1px dashed #333; }
  .ldiff .pair { display: grid; grid-template-columns: 2.4em 1.4em minmax(0, 1fr);
                 gap: 8px; padding: 2px 4px; align-items: start; }
  .ldiff .pair.htr { background: #161616; }
  .ldiff .pair.gys { background: #141a16; margin-bottom: 7px; }
  .ldiff .pair .nr, .ldiff .pair .src { color: var(--mut); font: 11px/1.45 ui-monospace, monospace; padding-top: 2px; }
  .ldiff .pair .txt { font: 12.5px/1.45 ui-monospace, Menlo, monospace; white-space: pre-wrap;
                      overflow-wrap: anywhere; min-width: 0; }
  .ldiff del { background: #4a2024; color: #f07178; text-decoration: none; border-radius: 2px; }
  .ldiff ins { background: #1a3a22; color: var(--acc); text-decoration: none; border-radius: 2px; }
  .ldiff .empty { color: #555; }
  .ldiff .preamble { font: 12px/1.4 ui-monospace, monospace; color: #889; background: #181818;
                     border: 1px dashed #333; border-radius: 6px; padding: 8px 10px; margin: 0 0 10px;
                     white-space: pre-wrap; overflow-wrap: anywhere; }
</style>
<header>
  <h1>Match review</h1>
  <button type="button" id="mPhoto" class="on">Photo → text</button>
  <button type="button" id="mCorpus">Text → photo</button>
  <select id="filter">
    <option value="todo" selected>todo (unmarked)</option>
    <option value="all">all (best first)</option>
    <option value="triage">triage (hard cases first)</option>
    <option value="collision">collisions</option>
    <option value="length">length mismatch</option>
    <option value="lowmargin">low margin</option>
    <option value="dups">near-dup unlabelled</option>
    <option value="matched">matched</option>
    <option value="unmatched">unmatched</option>
    <option value="empty">empty HTR</option>
    <option value="marked">reviewed</option>
    <option value="nopics">corpus with no photo</option>
  </select>
  <input id="q" type="search" placeholder="search id…">
  <span class="sub" id="pos"></span>
  <span class="sub" id="stats"></span>
  <button type="button" id="export">Export CSV</button>
  <button type="button" id="import">Import CSV</button>
  <input id="importFile" type="file" accept=".csv,text/csv" hidden>
  <span class="sub" id="keys">j/k · type a passage + Search · click a hit · 1–8 neighbour · a accept #1 · x no match</span>
</header>
<div id="layout">
  <div id="index"></div>
  <div id="main"></div>
</div>
<script>
const DATA = __PAYLOAD__;
const SEED = __SEED__;
const KEY = "sluis-match-review-v2";
let mode = "photo";
let filter = "todo";
let q = "";
let cur = 0;
let marks = {};
try { marks = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { marks = {}; }
marks = Object.assign({}, SEED, marks);
try { localStorage.setItem(KEY, JSON.stringify(marks)); } catch (e) {}
const snipCache = {};

function save() { localStorage.setItem(KEY, JSON.stringify(marks)); stats(); }
function markOf(id) { return marks[id] || null; }
function scoreClass(p) {
  if (typeof p === "number") return p >= 0.55 ? "ok" : p >= 0.40 ? "mid" : "bad";
  if (p.dup_of) return "mid";
  if (p.match) return (p.score >= 0.55 && !p.low_margin) ? "ok" : "mid";
  return "bad";
}
function thumb(id) { return "match-thumbs/" + id + ".jpg"; }
function full(id) { return "../images/pages-zoned-stretched/" + id + ".png"; }
function gysText(cid) { return DATA.texts[cid] || ""; }
function claimedByOther(photoId) {
  const taken = new Set();
  Object.entries(marks).forEach(([id, m]) => {
    if (id === photoId || String(id).startsWith("c:")) return;
    if (m && m.corpus) taken.add(m.corpus);
  });
  return taken;
}
function ownersExcept(photoId) {
  const owners = {};
  Object.entries(marks).forEach(([id, m]) => {
    if (id === photoId || String(id).startsWith("c:")) return;
    if (m && m.corpus) owners[m.corpus] = id;
  });
  return owners;
}
function annotatedNeighbors(p) {
  const owners = ownersExcept(p.id);
  const xs = (p.neighbors || []).map(n => Object.assign({}, n, {
    takenBy: owners[n.id] || "",
  }));
  xs.sort((a, b) => Number(!!a.takenBy) - Number(!!b.takenBy) || b.score - a.score);
  return xs;
}
function freeNeighbors(p) {
  return annotatedNeighbors(p).filter(n => !n.takenBy);
}
function corpusHits(c) {
  return (c.hits || []).map(h => {
    const m = markOf(h.id);
    const mine = !!(m && m.corpus === c.id);
    const nonePhoto = !!(m && !m.corpus);
    const takenElse = !!(m && m.corpus && m.corpus !== c.id) || nonePhoto;
    return { id: h.id, score: h.score, rank: h.rank, mine, takenElse,
             assigned: nonePhoto ? "no-match" : ((m && m.corpus) || "") };
  }).sort((a, b) => {
    if (a.mine !== b.mine) return a.mine ? -1 : 1;
    if (a.takenElse !== b.takenElse) return a.takenElse ? 1 : -1;
    return b.score - a.score;
  });
}

function photoList() {
  let xs = DATA.photos.slice();
  if (filter === "todo") xs = xs.filter(p => !markOf(p.id));
  if (filter === "matched") xs = xs.filter(p => p.match && !p.empty);
  if (filter === "unmatched") xs = xs.filter(p => !p.match || p.empty);
  if (filter === "empty") xs = xs.filter(p => p.empty);
  if (filter === "collision") xs = xs.filter(p => p.collision);
  if (filter === "length") xs = xs.filter(p => p.length_mismatch);
  if (filter === "lowmargin") xs = xs.filter(p => p.low_margin && p.match);
  if (filter === "dups") xs = xs.filter(p => p.dup_of);
  if (filter === "marked") xs = xs.filter(p => markOf(p.id));
  if (q) { const qq = q.toLowerCase(); xs = xs.filter(p => p.id.toLowerCase().includes(qq) || (p.match||"").toLowerCase().includes(qq)); }
  xs.sort((a, b) => {
    if (filter === "triage") {
      const ca = a.collision ? 1 : 0, cb = b.collision ? 1 : 0;
      if (cb !== ca) return cb - ca;
      const la = a.length_mismatch ? 1 : 0, lb = b.length_mismatch ? 1 : 0;
      if (lb !== la) return lb - la;
      const da = a.dup_of ? 1 : 0, db = b.dup_of ? 1 : 0;
      if (db !== da) return db - da;
      return a.margin - b.margin;
    }
    const sa = closeness(a), sb = closeness(b);
    if (sb !== sa) return sb - sa;
    return (b.margin || 0) - (a.margin || 0);
  });
  return xs;
}
function closeness(p) {
  if (p.empty) return -1;
  const free = freeNeighbors(p);
  if (p.match && free.some(n => n.id === p.match)) return p.score;
  if (free[0]) return free[0].score - (p.match ? 0 : 1);
  return -1;
}
function corpusList() {
  let xs = DATA.corpus.slice();
  const taken = claimedByOther("");
  if (filter === "todo") xs = xs.filter(c => !taken.has(c.id) && !isNoMatch("c:"+c.id));
  if (filter === "matched" || filter === "unmatched") xs = xs.filter(c => c.n > 0);
  if (filter === "nopics") xs = xs.filter(c => c.n === 0);
  if (filter === "marked") xs = xs.filter(c => markOf("c:"+c.id) || taken.has(c.id));
  if (q) { const qq = q.toLowerCase(); xs = xs.filter(c => c.id.toLowerCase().includes(qq) || c.gys.toLowerCase().includes(qq)); }
  if (filter === "todo" || filter === "all") {
    xs.sort((a, b) => {
      const fa = corpusHits(a).filter(h => !h.takenElse).length;
      const fb = corpusHits(b).filter(h => !h.takenElse).length;
      if (fb !== fa) return fb - fa;
      const sa = (corpusHits(a)[0] || {}).score || 0;
      const sb = (corpusHits(b)[0] || {}).score || 0;
      return sb - sa;
    });
  }
  return xs;
}
function list() { return mode === "photo" ? photoList() : corpusList(); }

function stats() {
  const nMark = Object.keys(marks).length;
  const nTodo = DATA.photos.filter(p => !markOf(p.id)).length;
  const nTaken = claimedByOther("").size;
  document.getElementById("stats").textContent =
    nTodo + " todo · " + Object.keys(marks).length + " marked · "
    + nTaken + " Gysseling taken · " + DATA.photos.length + " photos";
}
function renderIndex() {
  const xs = list();
  if (cur >= xs.length) cur = Math.max(0, xs.length - 1);
  const el = document.getElementById("index");
  if (mode === "photo") {
    el.innerHTML = xs.map((p,i) =>
      `<div class="ix ${scoreClass(p)} ${i===cur?"cur":""} ${markOf(p.id)?"mark":""} ${isNoMatch(p.id)?"nomatch":""}" data-i="${i}">
         ${p.id}<span class="s">${p.empty ? "∅" : isNoMatch(p.id) ? "NO" : p.dup_of ? "dup" : p.match ? p.score.toFixed(2) : "—"}</span></div>`
    ).join("");
  } else {
    el.innerHTML = xs.map((c,i) => {
      const hits = corpusHits(c);
      const nFree = hits.filter(h => !h.takenElse).length;
      const taken = claimedByOther("").has(c.id);
      const none = isNoMatch("c:"+c.id);
      return `<div class="ix ${none?"bad": taken?"ok": nFree?"mid":"bad"} ${i===cur?"cur":""} ${taken||none?"mark":""} ${none?"nomatch":""}" data-i="${i}">
         ${c.gys}<span class="s">${none ? "NO" : taken ? "yes" : nFree}</span></div>`;
    }).join("");
  }
  el.querySelectorAll(".ix").forEach(n => n.onclick = () => { cur = +n.dataset.i; render(); });
  const on = el.querySelector(".cur");
  if (on) on.scrollIntoView({block:"nearest"});
  document.getElementById("pos").textContent = xs.length ? ((cur+1) + " / " + xs.length) : "0 / 0";
}

function pick(photoId, corpusId, note) {
  marks[photoId] = { corpus: corpusId || "", note: note || "" };
  save(); render();
}
function isNoMatch(id) {
  const m = markOf(id);
  return !!(m && !m.corpus);
}

const DIFF_HEAD = 6;
const DIFF_TAIL = 6;

function splitLines(s) {
  if (!s) return [];
  return s.replace(/\s+$/, "").split(/\r?\n/);
}
function tokEq(a, b) { return a.toLowerCase() === b.toLowerCase(); }
function diffWords(a, b) {
  const A = a ? a.split(/\s+/).filter(Boolean) : [];
  const B = b ? b.split(/\s+/).filter(Boolean) : [];
  const n = A.length, m = B.length;
  const dp = Array.from({length: n + 1}, () => new Uint16Array(m + 1));
  for (let i = 1; i <= n; i++)
    for (let j = 1; j <= m; j++)
      dp[i][j] = tokEq(A[i - 1], B[j - 1]) ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1]);
  const left = [], right = [];
  let i = n, j = m;
  const stack = [];
  while (i > 0 && j > 0) {
    if (tokEq(A[i - 1], B[j - 1])) { stack.push("eq"); i--; j--; }
    else if (dp[i - 1][j] >= dp[i][j - 1]) { stack.push("del"); i--; }
    else { stack.push("ins"); j--; }
  }
  while (i-- > 0) stack.push("del");
  while (j-- > 0) stack.push("ins");
  i = 0; j = 0;
  while (stack.length) {
    const op = stack.pop();
    if (op === "eq") {
      const t = esc(A[i++]);
      left.push(t); right.push(esc(B[j++]));
    } else if (op === "del") left.push("<del>" + esc(A[i++]) + "</del>");
    else right.push("<ins>" + esc(B[j++]) + "</ins>");
  }
  return {
    left: left.length ? left.join(" ") : '<span class="empty">(empty)</span>',
    right: right.length ? right.join(" ") : '<span class="empty">(empty)</span>',
  };
}
function activeCandidate(p) {
  const vis = annotatedNeighbors(p);
  const free = vis.filter(n => !n.takenBy);
  const m = markOf(p.id);
  if (m && m.corpus) {
    return vis.find(n => n.id === m.corpus)
      || p.neighbors.find(n => n.id === m.corpus)
      || { id: m.corpus, gys: String(m.corpus).replace(/\.txt$/i, ""), score: 0 };
  }
  const taken = claimedByOther(p.id);
  if (p.match && !taken.has(p.match)) {
    return vis.find(n => n.id === p.match) || free[0] || vis[0] || null;
  }
  return free[0] || vis[0] || null;
}
function pairBlock(hLines, gLines, hIdx, gIdx, n) {
  const rows = [];
  for (let k = 0; k < n; k++) {
    const hi = hIdx + k, gi = gIdx + k;
    const h = hi >= 0 && hi < hLines.length ? hLines[hi] : "";
    const g = gi >= 0 && gi < gLines.length ? gLines[gi] : "";
    if (hi >= hLines.length && gi >= gLines.length) break;
    const d = diffWords(h, g);
    rows.push(`<div class="pair htr"><span class="nr">${hi < hLines.length ? hi + 1 : ""}</span><span class="src">H</span><div class="txt">${d.left}</div></div>
<div class="pair gys"><span class="nr">${gi < gLines.length ? gi + 1 : ""}</span><span class="src">G</span><div class="txt">${d.right}</div></div>`);
  }
  return rows.join("");
}
function isJunkLine(s) {
  const t = (s || "").trim();
  if (!t) return true;
  if (/^\[\.+\]$/.test(t)) return true;
  if (/^```/.test(t)) return true;
  if (/^(here is|transcription:|the image|the text reads|note:)/i.test(t)) return true;
  return false;
}
function lineWords(s) {
  return (s || "").toLowerCase().match(/[a-z0-9]{4,}/g) || [];
}
function lineScore(a, b) {
  const A = new Set(lineWords(a)), B = new Set(lineWords(b));
  if (!A.size || !B.size) return 0;
  let n = 0;
  for (const w of A) if (B.has(w)) n++;
  return n / Math.min(A.size, B.size);
}
function skipJunk(lines, maxN) {
  let i = 0;
  const cap = Math.min(maxN, lines.length);
  while (i < cap && isJunkLine(lines[i])) i++;
  return i;
}
function alignStarts(H, G) {
  const h0 = skipJunk(H, 8);
  const g0 = skipJunk(G, 8);
  const maxH = Math.min(h0 + 6, Math.max(h0, H.length - 1));
  const maxG = Math.min(g0 + 6, Math.max(g0, G.length - 1));
  let best = null;
  for (let hs = h0; hs <= maxH; hs++) {
    for (let gs = g0; gs <= maxG; gs++) {
      const s = lineScore(H[hs] || "", G[gs] || "");
      if (s < 0.4) continue;
      const skip = (hs - h0) + (gs - g0);
      if (!best || skip < best.skip || (skip === best.skip && s > best.s))
        best = {hs, gs, s, skip};
    }
  }
  return best || {hs: h0, gs: g0, s: 0, skip: 0};
}
function preambleHtml(lines, src, n) {
  if (n <= 0) return "";
  const body = lines.slice(0, n).map((ln, i) => (i + 1) + "  " + ln).join("\n");
  return `<div class="preamble">${esc(src)} opening skipped:\n${esc(body)}</div>`;
}
function tailStart(n, from, tail) {
  const left = n - from;
  if (left <= DIFF_HEAD + tail) return from + Math.min(DIFF_HEAD, left);
  return n - tail;
}
function lineDiffHtml(htr, gys, cand, how) {
  if (!cand) return "";
  const H = splitLines(htr), G = splitLines(gys);
  const al = alignStarts(H, G);
  const headN = Math.min(DIFF_HEAD, Math.max(H.length - al.hs, G.length - al.gs));
  const hTail0 = tailStart(H.length, al.hs, DIFF_TAIL);
  const gTail0 = tailStart(G.length, al.gs, DIFF_TAIL);
  const tailN = Math.max(H.length - hTail0, G.length - gTail0);
  let html = `<div class="ldiff"><h3>Line diff · HTR vs Gysseling ${esc(cand.gys)} (${esc(how)}) · first ${DIFF_HEAD} / last ${DIFF_TAIL} after opening skip</h3>`;
  html += preambleHtml(H, "HTR", al.hs);
  html += preambleHtml(G, "Gysseling", al.gs);
  html += pairBlock(H, G, al.hs, al.gs, headN);
  if (tailN > 0) {
    const hSkip = Math.max(0, hTail0 - al.hs - DIFF_HEAD);
    const gSkip = Math.max(0, gTail0 - al.gs - DIFF_HEAD);
    html += `<div class="gap">… ${hSkip} HTR / ${gSkip} Gysseling lines omitted …</div>`;
    html += pairBlock(H, G, hTail0, gTail0, tailN);
  }
  html += "</div>";
  return html;
}

function yearLabel(h) {
  if (!h || !h.year_from) return "";
  if (h.year_to && h.year_to !== h.year_from) return h.year_from + "–" + h.year_to;
  return String(h.year_from);
}
function rememberSnip(p) {
  const box = document.getElementById("snip");
  const yearEl = document.getElementById("snipYear");
  if (!box) return;
  const prev = snipCache[p.id] || {};
  snipCache[p.id] = Object.assign({}, prev, {
    q: box.value,
    year: yearEl ? yearEl.value : (prev.year || ""),
  });
}
function searchHitCard(p, h, i) {
  const id = h.id || h.file;
  const takenBy = ownersExcept(p.id)[id] || h.taken_by || "";
  const picked = (markOf(p.id) || {}).corpus === id;
  const body = h.snippet || (h.preview || "").split("\n").slice(0, 4).join("\n");
  return `<div class="nb ${picked?"pick":""} ${takenBy?"taken":""}" data-sid="${i}">
    <div class="hd"><b>${i+1}. ${esc(h.gys)}</b>
      <span>score ${Number(h.score).toFixed(3)}</span>
      <span>part ${Number(h.partial).toFixed(3)}</span>
      ${yearLabel(h) ? "<span>"+esc(yearLabel(h))+"</span>" : ""}
      ${h.hand ? '<span class="tag">'+esc(h.hand)+"</span>" : ""}
      ${takenBy ? '<span class="tag none">has '+esc(takenBy)+"</span>" : ""}</div>
    <div class="bar"><i style="width:${Math.min(100, Number(h.score)*100)}%"></i></div>
    <div class="gys-sm">${esc(body)}</div>
  </div>`;
}
function paintSearchHits(p) {
  const el = document.getElementById("snipHits");
  if (!el) return;
  const hits = (snipCache[p.id] && snipCache[p.id].hits) || [];
  const status = document.getElementById("snipStatus");
  if (!hits.length) {
    el.innerHTML = "";
    return;
  }
  if (status) status.textContent = hits.length + " hits · click to assign";
  el.innerHTML = hits.map((h, i) => searchHitCard(p, h, i)).join("");
  el.querySelectorAll(".nb").forEach(card => {
    card.onclick = () => {
      const h = hits[+card.dataset.sid];
      if (!h) return;
      const id = h.id || h.file;
      const assign = () => pick(p.id, id, "search");
      if (DATA.texts[id] || !id) { assign(); return; }
      fetch("/hands-leroy/cd-admin-txt/" + id)
        .then(r => r.ok ? r.text() : "")
        .then(t => { if (t) DATA.texts[id] = t; assign(); })
        .catch(assign);
    };
  });
}
async function runSnippetSearch(p) {
  rememberSnip(p);
  const q = ((snipCache[p.id] || {}).q || "").trim();
  const status = document.getElementById("snipStatus");
  const hitsEl = document.getElementById("snipHits");
  if (!q) {
    if (status) status.textContent = "type a passage from the photo";
    return;
  }
  if (status) status.textContent = "searching…";
  const params = new URLSearchParams({ q, k: "12" });
  const year = ((snipCache[p.id] || {}).year || "").trim();
  if (year) params.set("year", year);
  try {
    const r = await fetch("/search?" + params.toString());
    if (!r.ok) throw new Error("search HTTP " + r.status);
    const data = await r.json();
    snipCache[p.id].hits = data.hits || [];
    paintSearchHits(p);
    if (status && !(data.hits || []).length)
      status.textContent = "no hits — try more words, or clear the year";
  } catch (err) {
    if (status) status.textContent = "search needs the review server (python hands-leroy/review_matches.py --serve)";
    if (hitsEl) hitsEl.innerHTML = "";
  }
}

function neighborCard(p, n, i) {
  const picked = (markOf(p.id) || {}).corpus === n.id;
  const txt = (gysText(n.id) || "").split(/\n/).slice(0, 4).join("\n");
  const taken = n.takenBy
    ? `<span class="tag none">has ${esc(n.takenBy)}</span>` : "";
  return `<div class="nb ${picked?"pick":""} ${n.takenBy?"taken":""}" data-i="${i}">
    <div class="hd"><b>${i+1}. ${n.gys}</b>
      <span>lev ${n.score.toFixed(3)}</span>
      <span>tfidf ${n.tfidf.toFixed(3)}</span>
      ${n.partial ? `<span>part ${n.partial.toFixed(3)}</span>` : ""}
      ${n.hand ? `<span class="tag">${n.hand}</span>` : ""}
      ${taken}</div>
    <div class="bar"><i style="width:${Math.min(100, n.score*100)}%"></i></div>
    <div class="gys-sm">${esc(txt)}</div>
  </div>`;
}

function renderPhoto(p) {
  const htr = DATA.htr[p.id] || "";
  const m = markOf(p.id);
  const none = isNoMatch(p.id);
  const vis = annotatedNeighbors(p);
  const free = vis.filter(n => !n.takenBy);
  const cand = activeCandidate(p);
  let how = "suggestion #1";
  if (m && m.corpus) how = "picked";
  else if (cand && p.match && cand.id === p.match) how = "auto";
  else if (cand && cand.takenBy) how = "already attributed";
  else if (cand) how = "next free";
  let status;
  if (none) status = '<span class="tag none">NO MATCH</span> not in Gysseling / none of these neighbours';
  else if (m && m.corpus) status = '<span class="tag">picked</span> ' + esc(m.corpus);
  else if (!vis.length) status = '<span class="tag auto-none">no neighbours</span> matcher returned nothing with a Leroy groep';
  else if (!free.length) status = '<span class="tag auto-none">all neighbours already attributed</span> grey cards still clickable';
  else if (!p.match) status = '<span class="tag auto-none">auto: no match</span> neighbours below are suggestions only';
  else if (claimedByOther(p.id).has(p.match)) status = '<span class="tag auto-none">auto taken</span> ' + esc(p.match) + ' already attributed';
  else status = '<span class="tag">auto</span> ' + esc(p.match);
  document.getElementById("main").innerHTML = `
    <div class="status">${p.id} · ${status}
      <span class="sub"> · lev ${p.score.toFixed(3)} · margin ${p.margin.toFixed(3)}
        ${p.year ? " · "+p.year : ""}
        ${p.collision ? " · "+p.n_photos+" photos → this Gysseling" : ""}
        ${p.dup_of ? " · near-dup of "+p.dup_of+" (unlabelled)" : ""}
        ${p.hand ? " · "+p.hand : ""} ${p.empty ? '<span class="empty">empty HTR</span>' : ""}</span>
    </div>
    ${lineDiffHtml(htr, cand ? gysText(cand.id) : "", cand, how)}
    <div class="pane">
      <div class="col">
        <a href="${full(p.id)}" target="_blank"><img class="page" src="${thumb(p.id)}" alt="${p.id}"></a>
        <textarea id="snip" placeholder="type any distinctive passage you can read — a name, a place, a clause">${esc((snipCache[p.id]||{}).q || "")}</textarea>
        <div class="snip-row">
          <button type="button" class="on" id="snipGo">Search corpus</button>
          <label class="sub">year <input id="snipYear" type="number" min="1200" max="1310" placeholder="any" value="${esc(String((snipCache[p.id]||{}).year != null && (snipCache[p.id]||{}).year !== "" ? (snipCache[p.id]||{}).year : (p.year || "")))}"></label>
          <span class="sub" id="snipStatus">fuzzy search over all Gysseling texts</span>
        </div>
        <div id="snipHits"></div>
      </div>
      <div class="col">
        <h3 style="margin:0 0 4px;font-size:13px;color:var(--mut)">Top Gysseling neighbours</h3>
        ${vis.map((n,i) => neighborCard(p, n, i)).join("") || "<div class='sub'>no neighbours with a Leroy groep</div>"}
        <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">
          <button type="button" id="accept">Accept #1 (a)</button>
          <button type="button" id="none">No match (x)</button>
        </div>
      </div>
    </div>`;
  document.querySelectorAll(".nb").forEach(el => {
    if (el.closest("#snipHits")) return;
    el.onclick = () => pick(p.id, vis[+el.dataset.i].id, "neighbour");
  });
  document.getElementById("accept").onclick = () => {
    const first = free[0] || vis[0];
    if (first) pick(p.id, first.id, first.takenBy ? "neighbour" : "accept");
  };
  const noneBtn = document.getElementById("none");
  if (none) noneBtn.classList.add("none-on");
  noneBtn.onclick = () => pick(p.id, "", "not-in-archive");
  const snipBox = document.getElementById("snip");
  const yearBox = document.getElementById("snipYear");
  snipBox.oninput = () => rememberSnip(p);
  yearBox.oninput = () => rememberSnip(p);
  document.getElementById("snipGo").onclick = () => runSnippetSearch(p);
  snipBox.addEventListener("keydown", e => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey || !e.shiftKey)) {
      e.preventDefault();
      runSnippetSearch(p);
    }
  });
  paintSearchHits(p);
}

function renderCorpus(c) {
  const txt = gysText(c.id);
  const hits = corpusHits(c);
  const free = hits.filter(h => !h.takenElse || h.mine);
  const none = isNoMatch("c:"+c.id);
  const owners = hits.filter(h => h.mine);
  let status;
  if (none) status = '<span class="tag none">NO PHOTO</span> not in this archive';
  else if (owners.length) status = '<span class="tag">picked</span> ' + owners.map(h => h.id).join(", ");
  else if (!free.length) status = '<span class="tag auto-none">no free photos</span> retrieving photos already attributed elsewhere — click one to reassign';
  else status = '<span class="tag">click a photo</span> ' + free.length + " free candidate(s)";
  const focus = owners[0] || free[0] || hits[0];
  const cand = focus ? { id: c.id, gys: c.gys, score: focus.score } : null;
  const htr = focus ? (DATA.htr[focus.id] || "") : "";
  const cards = hits.map((h, i) => {
    const htrSm = (DATA.htr[h.id] || "").split(/\n/).slice(0, 4).join("\n");
    return `<div class="nb ${h.mine?"pick":""} ${h.takenElse?"taken":""}" data-i="${i}">
      <div class="hd"><b>${i+1}. ${h.id}</b>
        <span>lev ${h.score.toFixed(3)}</span>
        <span>rank ${h.rank}</span>
        ${h.mine ? '<span class="tag">this text</span>' : ""}
        ${h.takenElse ? '<span class="tag none">has '+esc(h.assigned)+'</span>' : ""}
        <a class="full" href="${full(h.id)}" target="_blank">full</a></div>
      <img class="page" src="${thumb(h.id)}" alt="${h.id}">
      <div class="htr" style="max-height:12vh;margin-top:6px">${esc(htrSm)}</div>
    </div>`;
  }).join("");
  document.getElementById("main").innerHTML = `
    <div class="status">Gysseling ${esc(c.gys)} · ${status}</div>
    ${cand && focus ? lineDiffHtml(htr, txt, cand, focus.id) : ""}
    <div class="pane">
      <div class="col">
        <div class="sub">${c.n} photo(s) retrieved this text in top-${DATA.show_k}
          ${c.n === 0 ? '<span class="empty"> — not retrieved for any photo</span>' : ""}</div>
        <div class="gys">${esc(txt) || "<span class='empty'>(empty corpus file)</span>"}</div>
      </div>
      <div class="col thumbs">
        ${cards || "<div class='sub'>No photo in this archive retrieved this Gysseling text in its top neighbours.</div>"}
        <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">
          <button type="button" id="accept">Accept #1 free (a)</button>
          <button type="button" id="none">No photo (x)</button>
        </div>
      </div>
    </div>`;
  document.querySelectorAll(".nb").forEach(el => {
    el.onclick = ev => {
      if (ev.target.closest("a")) return;
      const h = hits[+el.dataset.i];
      if (h) pickFromCorpus(h.id, c.id);
    };
  });
  document.getElementById("accept").onclick = () => {
    const h = free[0] || hits[0];
    if (h) pickFromCorpus(h.id, c.id);
  };
  const noneBtn = document.getElementById("none");
  if (none) noneBtn.classList.add("none-on");
  noneBtn.onclick = () => pickCorpusNone(c.id);
}
function pickFromCorpus(photoId, corpusId) {
  delete marks["c:"+corpusId];
  pick(photoId, corpusId, "text-photo");
}
function pickCorpusNone(corpusId) {
  marks["c:"+corpusId] = { corpus: "", note: "no-photo" };
  save(); render();
}

function esc(s) {
  return (s||"").replace(/[&<>]/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[ch]));
}

function render() {
  renderIndex();
  const xs = list();
  if (!xs.length) { document.getElementById("main").innerHTML = "<div class='sub'>no items</div>"; return; }
  if (mode === "photo") renderPhoto(xs[cur]);
  else renderCorpus(xs[cur]);
  stats();
}

document.getElementById("mPhoto").onclick = () => { mode="photo"; cur=0; setBtns(); render(); };
document.getElementById("mCorpus").onclick = () => { mode="corpus"; cur=0; setBtns(); render(); };
function setBtns() {
  document.getElementById("mPhoto").className = mode==="photo" ? "on" : "";
  document.getElementById("mCorpus").className = mode==="corpus" ? "on" : "";
  document.getElementById("keys").textContent = mode==="photo"
    ? "j/k · type a passage + Search · click a hit · 1–8 neighbour · a accept #1 · x no match"
    : "j/k · click a photo · 1–8 pick photo · a accept #1 · x no photo";
}
document.getElementById("filter").onchange = e => { filter = e.target.value; cur=0; render(); };
document.getElementById("q").oninput = e => { q = e.target.value.trim(); cur=0; render(); };
function parseDecisions(text) {
  text = String(text || "").replace(/\\n/g, "\n");
  const out = {};
  text.split(/\r?\n/).forEach((line, i) => {
    if (!i || !line.trim()) return;
    const parts = line.split(",");
    const id = (parts[0] || "").trim();
    if (!id || id === "id") return;
    out[id] = { corpus: (parts[2] || "").trim(), note: (parts.slice(3).join(",") || "").trim() };
  });
  return out;
}
document.getElementById("import").onclick = () => document.getElementById("importFile").click();
document.getElementById("importFile").onchange = e => {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = () => {
    marks = Object.assign({}, marks, parseDecisions(reader.result));
    save(); cur = 0; render();
  };
  reader.readAsText(f);
  e.target.value = "";
};
document.getElementById("export").onclick = () => {
  const lines = ["id,mode,corpus,note"];
  Object.entries(marks).forEach(([id, m]) => {
    lines.push([id, id.startsWith("c:")?"corpus":"photo", m.corpus||"", m.note||""].join(","));
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n")]));
  a.download = "match_review_decisions.csv";
  a.click();
};
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  const xs = list();
  if (e.key === "j") { cur = Math.min(xs.length-1, cur+1); render(); }
  if (e.key === "k") { cur = Math.max(0, cur-1); render(); }
  if (mode === "photo" && xs[cur]) {
    const p = xs[cur];
    const vis = annotatedNeighbors(p);
    const free = vis.filter(n => !n.takenBy);
    if (e.key >= "1" && e.key <= "8") {
      const i = +e.key - 1;
      if (vis[i]) pick(p.id, vis[i].id, "neighbour");
    }
    if (e.key === "a") {
      const first = free[0] || vis[0];
      if (first) pick(p.id, first.id, first.takenBy ? "neighbour" : "accept");
    }
    if (e.key === "x") pick(p.id, "", "not-in-archive");
  }
  if (mode === "corpus" && xs[cur]) {
    const c = xs[cur];
    const hits = corpusHits(c);
    const free = hits.filter(h => !h.takenElse || h.mine);
    if (e.key >= "1" && e.key <= "8") {
      const i = +e.key - 1;
      if (hits[i]) pickFromCorpus(hits[i].id, c.id);
    }
    if (e.key === "a") {
      const h = free[0] || hits[0];
      if (h) pickFromCorpus(h.id, c.id);
    }
    if (e.key === "x") pickCorpusNone(c.id);
  }
});
render();
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=Path, default=TOPK)
    ap.add_argument("--show-k", type=int, default=20)
    ap.add_argument("--no-thumbs", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--decisions", type=Path,
                    default=OUT_DIR / "match_review_decisions.csv")
    args = ap.parse_args()
    if not args.topk.is_file():
        raise SystemExit(
            f"missing {args.topk}\n"
            "Run: python hands-leroy/match_charters.py "
            "--transcriptions hands-leroy/transcriptions-zoned"
        )
    raw = json.loads(args.topk.read_text(encoding="utf-8"))
    payload = slim_payload(raw, args.show_k)
    ids = [p["id"] for p in payload["photos"]]
    if not args.no_thumbs:
        build_thumbs(ids)
    seed = load_decisions(args.decisions)
    write_html(payload, seed)
    if args.serve:
        sys.path.insert(0, str(HERE))
        from incipit_search import build_index
        print("indexing Gysseling (full texts)…", flush=True)
        index = build_index()
        print(f"  {len(index['rows'])} charters", flush=True)
        serve(args.port, index)


if __name__ == "__main__":
    main()
