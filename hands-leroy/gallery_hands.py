#!/usr/bin/env python3
"""Gallery of attributed photos grouped by Leroy hand-group.

  python hands-leroy/gallery_hands.py
  python hands-leroy/gallery_hands.py --serve
"""
from __future__ import annotations

import argparse
import html
import sys
from collections import defaultdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from match_charters import canon_gys, leroy_groups
from review_matches import load_decisions

ROOT = HERE.parent
OUT_DIR = ROOT / "outputs"
HTML = OUT_DIR / "hand_gallery.html"
DECISIONS = OUT_DIR / "match_review_decisions.csv"
HANDS = HERE / "handengroepen_gysseling.xlsx"


def group_sort_key(g: str) -> tuple:
    s = str(g)
    if s.endswith("'"):
        try:
            return (int(s[:-1]), 1, s)
        except ValueError:
            return (10**9, 1, s)
    try:
        return (int(s), 0, s)
    except ValueError:
        return (10**9, 0, s)


def rows_from_decisions(decisions: Path, hands: Path) -> tuple[dict[str, list[dict]], list[dict]]:
    lookup, ambiguous = leroy_groups(hands)
    marks = load_decisions(decisions)
    by_group: dict[str, list[dict]] = defaultdict(list)
    unlabeled: list[dict] = []
    for pid, m in marks.items():
        if str(pid).startswith("c:"):
            continue
        corpus = (m.get("corpus") or "").strip()
        if not corpus:
            continue
        rec = {
            "id": pid,
            "gys": Path(corpus).stem,
            "corpus": corpus,
        }
        if corpus.lower().startswith("hand:"):
            rec["hand"] = corpus.split(":", 1)[1].strip()
            rec["gys"] = "—"
            rec["why"] = "hand only (no Gysseling)"
            by_group[rec["hand"]].append(rec)
            continue
        key = canon_gys(corpus)
        if key in ambiguous:
            rec["why"] = "conflicting groep"
            unlabeled.append(rec)
        elif key and key in lookup:
            rec["hand"] = lookup[key]
            by_group[lookup[key]].append(rec)
        else:
            rec["why"] = "no Leroy groep join"
            unlabeled.append(rec)
    for xs in by_group.values():
        xs.sort(key=lambda r: r["id"])
    unlabeled.sort(key=lambda r: r["id"])
    return dict(by_group), unlabeled


def card_html(r: dict) -> str:
    pid = html.escape(r["id"])
    gys = html.escape(r["gys"])
    extra = f'<span class="mut">{html.escape(r["why"])}</span>' if r.get("why") else ""
    return (
        f'<a class="card" href="../images/pages-zoned-stretched/{pid}.png" '
        f'target="_blank" data-q="{pid} {gys}">'
        f'<img src="match-thumbs/{pid}.jpg" alt="{pid}" loading="lazy">'
        f'<div class="cap"><b>{pid}</b> · Gys {gys} {extra}</div></a>'
    )


def write_html(by_group: dict[str, list[dict]], unlabeled: list[dict]) -> None:
    groups = sorted(by_group, key=group_sort_key)
    n_photos = sum(len(v) for v in by_group.values()) + len(unlabeled)
    nav = []
    sections = []
    for g in groups:
        xs = by_group[g]
        gid = html.escape(g)
        nav.append(f'<a href="#g-{gid}">groep {gid} <span>{len(xs)}</span></a>')
        cards = "".join(card_html(r) for r in xs)
        sections.append(
            f'<section id="g-{gid}">'
            f'<h2>Groep {gid} <span class="mut">{len(xs)} charter{"s" if len(xs) != 1 else ""}</span></h2>'
            f'<div class="grid">{cards}</div></section>'
        )
    extra = ""
    if unlabeled:
        cards = "".join(card_html(r) for r in unlabeled)
        extra = (
            f'<section id="unlabelled">'
            f'<h2>No unique Leroy groep <span class="mut">{len(unlabeled)}</span></h2>'
            f'<div class="grid">{cards}</div></section>'
        )
        nav.append(f'<a href="#unlabelled">unlabelled <span>{len(unlabeled)}</span></a>')
    page = f"""<!doctype html>
<meta charset=utf-8>
<title>Charters by Leroy hand</title>
<meta http-equiv="Cache-Control" content="no-store">
<style>
  :root {{ --bg:#111; --fg:#eee; --mut:#8aa; --acc:#3ddc84; --line:#2a2a2a; }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; background: var(--bg); color: var(--fg);
               font: 14px/1.4 system-ui, sans-serif; }}
  header {{ position: sticky; top: 0; z-index: 3; background: #1a1a1a;
            border-bottom: 1px solid #333; padding: 10px 14px; }}
  h1 {{ font-size: 15px; margin: 0 0 6px; }}
  .sub {{ color: var(--mut); font-size: 12px; }}
  input {{ background: #333; color: var(--fg); border: 1px solid #555;
           border-radius: 6px; padding: 4px 8px; margin-left: 10px; width: 180px; }}
  nav {{ display: flex; flex-wrap: wrap; gap: 6px; max-height: 28vh; overflow: auto;
         padding: 8px 14px 10px; border-bottom: 1px solid var(--line); background: #161616; }}
  nav a {{ color: var(--fg); text-decoration: none; border: 1px solid #444;
           border-radius: 999px; padding: 2px 9px; font-size: 12px; }}
  nav a:hover {{ border-color: var(--acc); }}
  nav a span {{ color: var(--mut); }}
  main {{ padding: 12px 14px 40px; }}
  h2 {{ font-size: 16px; margin: 22px 0 10px; }}
  .mut {{ color: var(--mut); font-weight: 400; font-size: 13px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; }}
  .card {{ display: block; color: inherit; text-decoration: none;
           border: 1px solid var(--line); border-radius: 8px; padding: 6px;
           background: #191919; }}
  .card:hover {{ border-color: #666; }}
  .card img {{ width: 100%; height: 220px; object-fit: contain; background: #000;
               border-radius: 4px; display: block; }}
  .cap {{ font: 11px/1.35 ui-monospace, monospace; color: var(--mut); margin-top: 6px; }}
  .cap b {{ color: var(--fg); }}
  .card.hide {{ display: none; }}
  section.hide {{ display: none; }}
</style>
<header>
  <h1>Charters by Leroy hand
    <span class="sub">{n_photos} photos · {len(groups)} groepen · {html.escape(str(DECISIONS.name))}</span>
    <input id="q" type="search" placeholder="filter id / Gys / groep">
  </h1>
</header>
<nav>{"".join(nav)}</nav>
<main>
{"".join(sections)}
{extra}
</main>
<script>
const q = document.getElementById("q");
q.oninput = () => {{
  const s = q.value.trim().toLowerCase();
  document.querySelectorAll("section").forEach(sec => {{
    let n = 0;
    sec.querySelectorAll(".card").forEach(c => {{
      const hit = !s || c.dataset.q.toLowerCase().includes(s) || sec.id.toLowerCase().includes(s);
      c.classList.toggle("hide", !hit);
      if (hit) n++;
    }});
    sec.classList.toggle("hide", n === 0);
  }});
}};
</script>
"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HTML.write_text(page, encoding="utf-8")
    print(f"gallery → {HTML}  ({n_photos} photos, {len(groups)} groepen"
          f"{f', {len(unlabeled)} unlabelled' if unlabeled else ''})")


def serve(port: int) -> None:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(ROOT), **kw)

        def log_message(self, fmt, *args):
            if args and str(args[0]).startswith("GET /outputs/match-thumbs"):
                return
            super().log_message(fmt, *args)

    url = f"http://127.0.0.1:{port}/outputs/hand_gallery.html"
    print(f"open {url}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=Path, default=DECISIONS)
    ap.add_argument("--hands", type=Path, default=HANDS)
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    by_group, unlabeled = rows_from_decisions(args.decisions, args.hands)
    write_html(by_group, unlabeled)
    if args.serve:
        serve(args.port)


if __name__ == "__main__":
    main()
