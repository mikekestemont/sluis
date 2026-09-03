#!/usr/bin/env python3
"""Fuzzy search over Corpus Gysseling (any transcribed snippet).

  python hands-leroy/incipit_search.py --serve
  python hands-leroy/incipit_search.py "malin henric van laepscure"

Indexes the full administrative texts. Type any distinctive passage;
TF-IDF + partial Levenshtein rank the charters, and the hit preview
shows the matching window, not only the opening.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from rapidfuzz import fuzz
from rapidfuzz.fuzz import partial_ratio_alignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from match_charters import canon_gys, leroy_groups, load_texts, normalize

ROOT = HERE.parent
CORPUS = HERE / "cd-admin-txt"
ORIG = HERE / "cg-admin-orig"
HANDS = HERE / "handengroepen_gysseling.xlsx"
DECISIONS = ROOT / "outputs" / "match_review_decisions.csv"
LITERARY = re.compile(r"^[3-8]\d{3}$")
DATER_OPEN = re.compile(
    r"<datering\s+jaar_tot='(\d+)'\s+jaar_van='(\d+)'[^>]*>"
)
STATUS = re.compile(r"statushandkode='([^']+)'")
PORT = 8766
SHOW = 20


def first_lines(text: str, n: int) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[:n]


def match_window(norm: str, query: str, pad: int = 80) -> str:
    if not norm or not query:
        return ""
    try:
        al = partial_ratio_alignment(query, norm)
        start = max(0, int(al.dest_start) - pad)
        end = min(len(norm), int(al.dest_end) + pad)
    except Exception:
        start, end = 0, min(len(norm), 240)
    snippet = norm[start:end].strip()
    if start:
        snippet = "… " + snippet
    if end < len(norm):
        snippet = snippet + " …"
    return snippet


def charter_year(stem: str) -> tuple[int | None, int | None]:
    path = ORIG / f"{stem}.fromdb"
    if not path.is_file():
        return None, None
    raw = path.read_text(encoding="utf-8", errors="replace")[:12_000]
    exact: list[int] = []
    spans: list[tuple[int, int]] = []
    for m in DATER_OPEN.finditer(raw):
        ytot, yvan = int(m.group(1)), int(m.group(2))
        codes = STATUS.findall(raw[m.end():m.end() + 600])
        if not codes or codes[0] not in {"an", "mn"}:
            continue
        spans.append((yvan, ytot))
        if yvan == ytot and 1200 <= yvan <= 1301:
            exact.append(yvan)
    if exact:
        return min(exact), max(exact)
    if spans:
        return min(a for a, _ in spans), max(b for _, b in spans)
    return None, None


def load_decisions(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8").replace("\\n", "\n")
    out: dict[str, dict] = {}
    for i, line in enumerate(text.splitlines()):
        if i == 0 or not line.strip():
            continue
        parts = line.split(",")
        pid = parts[0].strip()
        if not pid or pid == "id":
            continue
        corpus = parts[2].strip() if len(parts) > 2 else ""
        out[pid] = {"corpus": corpus}
    return out


def taken_map(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for pid, rec in load_decisions(path).items():
        if str(pid).startswith("c:"):
            continue
        corpus = (rec.get("corpus") or "").strip()
        if corpus:
            out[canon_gys(corpus) or ""] = pid
    out.pop("", None)
    return out


def build_index() -> dict:
    texts = load_texts(CORPUS, skip_junk=True)
    lookup, _ = leroy_groups(HANDS)
    taken = taken_map(DECISIONS)
    rows = []
    for name, raw in texts.items():
        stem = Path(name).stem
        if LITERARY.match(stem):
            continue
        lines = first_lines(raw, 8)
        if not lines:
            continue
        key = canon_gys(name)
        yvan, ytot = charter_year(stem)
        rows.append({
            "file": name,
            "gys": stem,
            "key": key,
            "text": raw,
            "text_norm": normalize(raw),
            "preview": "\n".join(lines),
            "year_from": yvan,
            "year_to": ytot,
            "hand": lookup.get(key or "", ""),
            "taken_by": taken.get(key or "", ""),
        })
    norms = [r["text_norm"] for r in rows]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          min_df=1, max_df=0.95, sublinear_tf=True)
    mat = vec.fit_transform(norms)
    return {"rows": rows, "vec": vec, "mat": mat}


def search(index: dict, query: str, *, k: int = SHOW, leroy_only: bool = False,
           year: int | None = None, hide_taken: bool = False) -> list[dict]:
    q = normalize(query)
    if not q:
        return []
    qv = index["vec"].transform([q])
    sims = cosine_similarity(qv, index["mat"]).ravel()
    hits = []
    for i, r in enumerate(index["rows"]):
        if leroy_only and not r["hand"]:
            continue
        if hide_taken and r["taken_by"]:
            continue
        if year is not None:
            y0, y1 = r["year_from"], r["year_to"]
            if y0 is None or not (y0 - 1 <= year <= (y1 or y0) + 1):
                continue
        doc = r["text_norm"]
        partial = fuzz.partial_ratio(q, doc) / 100.0
        tfidf = float(sims[i])
        score = 0.65 * partial + 0.35 * tfidf
        hits.append({
            "i": i,
            "id": r["file"],
            "file": r["file"],
            "gys": r["gys"],
            "score": round(score, 4),
            "partial": round(partial, 4),
            "tfidf": round(tfidf, 4),
            "year_from": r["year_from"],
            "year_to": r["year_to"],
            "hand": r["hand"],
            "taken_by": r["taken_by"],
        })
    hits.sort(key=lambda h: -h["score"])
    out = []
    for h in hits[:k]:
        r = index["rows"][h.pop("i")]
        h["snippet"] = match_window(r["text_norm"], q)
        h["preview"] = r["preview"]
        out.append(h)
    return out


def print_hits(hits: list[dict]) -> None:
    if not hits:
        print("no hits")
        return
    for i, h in enumerate(hits, 1):
        year = h["year_from"] or "?"
        if h["year_to"] and h["year_to"] != h["year_from"]:
            year = f"{h['year_from']}–{h['year_to']}"
        groep = f"  groep {h['hand']}" if h["hand"] else ""
        taken = f"  taken {h['taken_by']}" if h["taken_by"] else ""
        print(f"{i:2}. {h['gys']:<10} {h['score']:.3f}  {year}{groep}{taken}")
        print(f"    {(h.get('snippet') or h['preview'].splitlines()[0])[:110]}")


HTML = r"""<!doctype html>
<meta charset=utf-8>
<title>Gysseling incipit search</title>
<meta http-equiv="Cache-Control" content="no-store">
<style>
  :root { --bg:#111; --fg:#eee; --mut:#8aa; --acc:#3ddc84; --warn:#e0c36e; --bad:#f07178; --line:#2a2a2a; }
  * { box-sizing: border-box; }
  html, body { margin: 0; background: var(--bg); color: var(--fg);
               font: 14px/1.4 system-ui, sans-serif; }
  header { position: sticky; top: 0; z-index: 3; background: #1a1a1a;
           border-bottom: 1px solid #333; padding: 12px 16px; }
  h1 { font-size: 15px; margin: 0 0 8px; }
  .sub { color: var(--mut); font-size: 12px; }
  textarea { width: 100%; min-height: 72px; background: #222; color: var(--fg);
             border: 1px solid #555; border-radius: 8px; padding: 8px 10px;
             font: 15px/1.4 ui-monospace, Menlo, monospace; }
  .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-top: 8px; }
  button, input, select { background: #333; color: var(--fg); border: 1px solid #555;
           border-radius: 6px; padding: 6px 10px; font: inherit; }
  button.go { background: var(--acc); color: #111; font-weight: 600; }
  label { color: var(--mut); font-size: 12px; display: flex; gap: 6px; align-items: center; }
  input[type=number] { width: 78px; }
  main { padding: 12px 16px 48px; max-width: 980px; }
  .hit { border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px;
         margin: 10px 0; background: #191919; }
  .hit .hd { display: flex; gap: 10px; flex-wrap: wrap; align-items: baseline;
             font: 12px ui-monospace, monospace; color: var(--mut); }
  .hit .hd b { color: var(--fg); font-size: 15px; cursor: pointer; }
  .tag { border: 1px solid #444; border-radius: 999px; padding: 1px 8px; }
  .tag.none { color: var(--bad); border-color: #633; }
  .bar { height: 4px; background: #333; border-radius: 2px; margin: 6px 0 8px; }
  .bar > i { display: block; height: 100%; background: var(--acc); border-radius: 2px; }
  pre { white-space: pre-wrap; font: 13px/1.4 ui-monospace, Menlo, monospace;
        margin: 0; color: #ddd; }
  .empty { color: var(--mut); padding: 24px 0; }
</style>
<header>
  <h1>Search Gysseling
    <span class="sub">type any distinctive passage · spelling can be rough</span></h1>
  <textarea id="q" placeholder="a name, a place, a clause — whatever you can read"></textarea>
  <div class="row">
    <button class="go" id="go" type="button">Search</button>
    <label>year <input id="year" type="number" min="1200" max="1310" placeholder="any"></label>
    <label><input id="leroy" type="checkbox"> Leroy groepen only</label>
    <label><input id="hide" type="checkbox"> hide already matched</label>
    <span class="sub" id="status"></span>
  </div>
</header>
<main id="out"><div class="empty">Paste or type the first line, then Search (or ⌘/Ctrl+Enter).</div></main>
<script>
const q = document.getElementById("q");
const out = document.getElementById("out");
const status = document.getElementById("status");
q.focus();

function esc(s) {
  return String(s || "").replace(/[&<>"']/g, c => ({
    "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"
  }[c]));
}
function yearLabel(h) {
  if (!h.year_from) return "";
  if (h.year_to && h.year_to !== h.year_from) return h.year_from + "–" + h.year_to;
  return String(h.year_from);
}
async function search() {
  const query = q.value.trim();
  if (!query) { out.innerHTML = '<div class="empty">Type a first line.</div>'; return; }
  const params = new URLSearchParams({ q: query, k: "20" });
  const year = document.getElementById("year").value.trim();
  if (year) params.set("year", year);
  if (document.getElementById("leroy").checked) params.set("leroy", "1");
  if (document.getElementById("hide").checked) params.set("hide_taken", "1");
  status.textContent = "searching…";
  const r = await fetch("/search?" + params.toString());
  const data = await r.json();
  status.textContent = (data.hits || []).length + " hits";
  if (!data.hits || !data.hits.length) {
    out.innerHTML = '<div class="empty">No hits. Try more words from the first line, or drop the year filter.</div>';
    return;
  }
  out.innerHTML = data.hits.map((h, i) => `
    <article class="hit">
      <div class="hd">
        <b data-gys="${esc(h.gys)}" title="click to copy">${i+1}. ${esc(h.gys)}</b>
        <span>score ${h.score.toFixed(3)}</span>
        <span>part ${h.partial.toFixed(3)}</span>
        ${yearLabel(h) ? "<span>"+esc(yearLabel(h))+"</span>" : ""}
        ${h.hand ? '<span class="tag">groep '+esc(h.hand)+"</span>" : ""}
        ${h.taken_by ? '<span class="tag none">photo '+esc(h.taken_by)+"</span>" : ""}
      </div>
      <div class="bar"><i style="width:${Math.min(100, h.score*100)}%"></i></div>
      <pre>${esc(h.snippet || h.preview)}</pre>
    </article>`).join("");
}
document.getElementById("go").onclick = search;
q.addEventListener("keydown", e => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); search(); }
});
out.addEventListener("click", e => {
  const b = e.target.closest("b[data-gys]");
  if (!b) return;
  navigator.clipboard.writeText(b.dataset.gys);
  status.textContent = "copied " + b.dataset.gys;
});
</script>
"""


def serve(index: dict, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if self.path.startswith("/search"):
                return
            super().log_message(fmt, *args)

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if u.path != "/search":
                self._send(404, b"not found", "text/plain")
                return
            qs = parse_qs(u.query)
            q = (qs.get("q") or [""])[0]
            k = int((qs.get("k") or [str(SHOW)])[0])
            year_s = (qs.get("year") or [""])[0]
            year = int(year_s) if year_s.isdigit() else None
            hits = search(
                index, q, k=k,
                leroy_only=(qs.get("leroy") or [""])[0] in {"1", "true"},
                year=year,
                hide_taken=(qs.get("hide_taken") or [""])[0] in {"1", "true"},
            )
            body = json.dumps({"q": q, "hits": hits}, ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"incipit search on {len(index['rows'])} Gysseling openings", flush=True)
    print(f"open http://127.0.0.1:{port}/", flush=True)
    httpd.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*", help="first line to search (CLI mode)")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--k", type=int, default=SHOW)
    ap.add_argument("--leroy", action="store_true")
    ap.add_argument("--year", type=int)
    args = ap.parse_args()
    print("indexing Gysseling openings…", flush=True)
    index = build_index()
    print(f"  {len(index['rows'])} charters", flush=True)
    if args.serve or not args.query:
        serve(index, args.port)
        return
    hits = search(index, " ".join(args.query), k=args.k,
                  leroy_only=args.leroy, year=args.year)
    print_hits(hits)


if __name__ == "__main__":
    main()
