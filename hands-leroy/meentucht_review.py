#!/usr/bin/env python3
"""Focused reconsideration sheet: Gysseling/HTR *meentucht* formula charters.

  python hands-leroy/meentucht_review.py
"""
from __future__ import annotations

import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from match_charters import canon_gys, leroy_groups
from review_matches import load_decisions

ROOT = HERE.parent
OUT = ROOT / "outputs" / "meentucht_review.html"
TOPK = HERE / "match_topk.json"
DECISIONS = ROOT / "outputs" / "match_review_decisions.csv"
HANDS = HERE / "handengroepen_gysseling.xlsx"

RX = re.compile(r"[a-z]*meente?ucht|[a-z]*mentucht", re.I)


def distinctive(text: str, shared: set[str], n: int = 10) -> list[str]:
    words = re.findall(r"[A-Za-z]{5,}", text)
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        k = w.lower()
        if k in shared or k in seen:
            continue
        seen.add(k)
        out.append(w)
        if len(out) >= n:
            break
    return out


def main() -> None:
    raw = json.loads(TOPK.read_text(encoding="utf-8"))
    lookup, _amb = leroy_groups(HANDS)
    marks = load_decisions(DECISIONS)
    photos = {}
    for r in raw["photos"]:
        photos[Path(r["transcription"]).stem] = r

    gys_hits: dict[str, dict] = {}
    tok = defaultdict(int)
    for name, text in raw["corpus_texts"].items():
        if not RX.search(text):
            continue
        for w in set(re.findall(r"[a-z]{4,}", text.lower())):
            tok[w] += 1
        key = canon_gys(name)
        gys_hits[name] = {
            "gys": Path(name).stem,
            "hand": lookup.get(key or "", "?"),
            "text": text,
        }
    n_gys = max(1, len(gys_hits))
    shared = {w for w, c in tok.items() if c >= max(3, int(0.45 * n_gys))}

    assigned_gys = set()
    for pid, m in marks.items():
        if str(pid).startswith("c:"):
            continue
        if m.get("corpus"):
            assigned_gys.add(canon_gys(m["corpus"]))

    # photos to show: HTR hit, or auto match in gys_hits, or you assigned a gys_hit
    show: list[str] = []
    for pid, r in photos.items():
        htr = r.get("htr") or ""
        auto = r.get("match") or ""
        you = (marks.get(pid) or {}).get("corpus") or ""
        if RX.search(htr) or auto in gys_hits or you in gys_hits:
            show.append(pid)
    show.sort()

    cards = []
    for pid in show:
        r = photos[pid]
        m = marks.get(pid) or {}
        you = m.get("corpus") or ""
        if pid not in marks:
            status = '<span class="tag">unmarked</span>'
        elif not you:
            status = '<span class="tag none">NO MATCH</span>'
        else:
            status = f'<span class="tag ok">picked {html.escape(Path(you).stem)}</span>'
        htr = r.get("htr") or ""
        htr_sm = "\n".join(htr.splitlines()[:8])
        neigh_html = []
        for i, n in enumerate((r.get("neighbors") or [])[:8]):
            cid = n.get("corpus") or ""
            if cid not in gys_hits:
                continue
            g = gys_hits[cid]
            dist = ", ".join(distinctive(g["text"], shared))
            taken = " · already assigned" if canon_gys(cid) in assigned_gys else ""
            neigh_html.append(
                f'<div class="nb">'
                f'<b>#{i+1} Gys {html.escape(g["gys"])}</b> '
                f'groep {html.escape(str(g["hand"]))} · lev {n["score"]:.3f}{taken}'
                f'<div class="mut">{html.escape(dist)}</div></div>'
            )
        cards.append(
            f'<section id="{html.escape(pid)}">'
            f'<h2>{html.escape(pid)} {status} '
            f'<span class="mut">auto {html.escape(Path(r.get("match") or "—").stem)} '
            f'{(r.get("match_score") or 0):.3f}</span></h2>'
            f'<div class="pane">'
            f'<a href="../images/pages-zoned-stretched/{html.escape(pid)}.png" target="_blank">'
            f'<img class="page" src="match-thumbs/{html.escape(pid)}.jpg" alt="{html.escape(pid)}"></a>'
            f'<div><div class="htr">{html.escape(htr_sm)}</div>{"".join(neigh_html) or "<div class=mut>no *meentucht* neighbour in top-8</div>"}</div>'
            f'</div></section>'
        )

    unmatched = []
    for cid, g in sorted(gys_hits.items(), key=lambda kv: (str(kv[1]["hand"]), kv[1]["gys"])):
        if canon_gys(cid) in assigned_gys:
            continue
        dist = ", ".join(distinctive(g["text"], shared))
        unmatched.append(
            f'<tr><td>{html.escape(g["gys"])}</td><td>{html.escape(str(g["hand"]))}</td>'
            f'<td>{html.escape(dist)}</td></tr>'
        )

    page = f"""<!doctype html>
<meta charset=utf-8>
<title>Meentucht reconsideration</title>
<style>
  :root {{ --bg:#111; --fg:#eee; --mut:#8aa; --acc:#3ddc84; --bad:#f07178; --line:#2a2a2a; }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; background: var(--bg); color: var(--fg);
               font: 14px/1.4 system-ui, sans-serif; }}
  header {{ position: sticky; top: 0; background: #1a1a1a; border-bottom: 1px solid #333;
            padding: 10px 14px; z-index: 2; }}
  h1 {{ font-size: 15px; margin: 0 0 4px; }}
  .mut {{ color: var(--mut); font-size: 12px; }}
  main {{ padding: 12px 14px 40px; }}
  h2 {{ font-size: 15px; margin: 28px 0 8px; }}
  .pane {{ display: grid; grid-template-columns: minmax(280px, 1fr) minmax(320px, 1.1fr); gap: 12px; }}
  img.page {{ width: 100%; max-height: 42vh; object-fit: contain; background: #000; border-radius: 4px; }}
  .htr {{ white-space: pre-wrap; font: 12px/1.4 ui-monospace, monospace; background: #1a1a1a;
          border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; max-height: 22vh; overflow: auto; }}
  .nb {{ border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px; margin: 6px 0; font-size: 13px; }}
  .tag {{ display: inline-block; padding: 1px 7px; border-radius: 999px; background: #333; font-size: 11px; }}
  .tag.ok {{ background: var(--acc); color: #111; font-weight: 600; }}
  .tag.none {{ background: var(--bad); color: #111; font-weight: 700; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  td, th {{ border-bottom: 1px solid var(--line); padding: 4px 8px; text-align: left; vertical-align: top; }}
  a {{ color: #9cf; }}
</style>
<header>
  <h1>Meentucht reconsideration</h1>
  <div class="mut">{len(gys_hits)} Gysseling files · {len(show)} photos (HTR hit or auto/picked *meentucht* text) ·
    {len(unmatched)} Gysseling still without a photo</div>
</header>
<main>
  <p>HTR usually misses the word. The three <b>NO MATCH</b> photos below have the Bruges
     schepenen/meentucht opening in the transcription and an auto-hit on groepen 3/4.
     Unmatched Gysseling rows show distinctive tokens after the shared formula.</p>
  {''.join(cards)}
  <h2>Unmatched Gysseling *meentucht* texts</h2>
  <table>
    <tr><th>Gys</th><th>Groep</th><th>Distinctive tokens</th></tr>
    {''.join(unmatched)}
  </table>
</main>
"""
    OUT.write_text(page, encoding="utf-8")
    print(f"review → {OUT}  photos {len(show)}  gys {len(gys_hits)}  unmatched {len(unmatched)}")


if __name__ == "__main__":
    main()
