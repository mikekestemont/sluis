#!/usr/bin/env python3
"""Rank Büdingen in the transductive all-pages VLAD (gallery + extras, 332o kept).

Büdingen patches are in the codebook. Ranking holds out the query page, and a
second haystack also holds out the other Büdingen side so r/v cannot retrieve
each other.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

BUDINGEN = ["Büdingen1r", "Büdingen1v"]
WATCH = ["332o", "RA-800r", "Genois-1327a", "Genois-1327b", "Genois-1327c"]
EXTRAS = set(BUDINGEN + WATCH[1:])  # extras; 332o is gallery
TOPK = 20


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def stems_from(mapping_path: Path):
    meta = json.loads(mapping_path.read_text())
    paths = [r["image"] for r in meta["rows"]]
    return [nfc(Path(p).stem) for p in paths], paths, meta


def l2(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


def thumb_b64(root: Path, rel: str, max_side: int = 240) -> str:
    im = Image.open(root / rel).convert("L")
    im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    buf_im = io.BytesIO()
    im.save(buf_im, format="JPEG", quality=70)
    return base64.b64encode(buf_im.getvalue()).decode("ascii")


def rank_against(scores: np.ndarray, allow: np.ndarray) -> np.ndarray:
    return np.argsort(-np.where(allow, scores, -np.inf))


def rank_of(order: np.ndarray, idx: int) -> int:
    return int(np.where(order == idx)[0][0] + 1)


def top_rows(order, stems, scores, k):
    rows = []
    for r, i in enumerate(order[:k], 1):
        stem = stems[i]
        rows.append({
            "rank": r,
            "stem": stem,
            "score": float(scores[i]),
            "is_332o": stem == "332o",
            "is_extra": stem in EXTRAS,
        })
    return rows


def watched(order, stems, scores, idx):
    out = {}
    for name in WATCH + BUDINGEN:
        if name not in idx:
            continue
        i = idx[name]
        out[name] = {
            "rank": rank_of(order, i) if order[0] != -1 else None,
            "score": float(scores[i]),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.home() / "mole")
    args = ap.parse_args()
    root = args.root

    npy = root / "outputs/sluis/leroy.all.sauvola.ssl.e15.npy"
    out_json = root / "outputs/sluis/leroy.all.budingen_nn.json"
    out_html = root / "outputs/sluis/leroy.all.budingen_nn.html"

    stems, paths, meta = stems_from(npy.with_suffix(".mapping.json"))
    X = l2(np.load(npy).astype(np.float64))
    idx = {s: i for i, s in enumerate(stems)}
    for s in BUDINGEN + ["332o", "RA-800r"]:
        if s not in idx:
            raise SystemExit(f"missing from index: {s}")

    bud_i = [idx[s] for s in BUDINGEN]
    reports = []

    def one_query(name: str, qvec: np.ndarray, hold: list[int]) -> dict:
        scores = X @ qvec
        allow_self = np.ones(len(stems), dtype=bool)
        # if qvec is a gallery row, drop that row from "full"
        if name in idx:
            allow_self[idx[name]] = False
        order_self = rank_against(scores, allow_self)

        allow_hold = allow_self.copy()
        for j in hold:
            allow_hold[j] = False
        order_hold = rank_against(scores, allow_hold)

        rec = {
            "query": name,
            "n_full": int(allow_self.sum()),
            "n_hold_budingen": int(allow_hold.sum()),
            "nn_full": stems[int(order_self[0])],
            "nn_full_score": float(scores[int(order_self[0])]),
            "nn_hold": stems[int(order_hold[0])],
            "nn_hold_score": float(scores[int(order_hold[0])]),
            "watched_full": {
                s: {"rank": rank_of(order_self, idx[s]), "score": float(scores[idx[s]])}
                for s in WATCH if s in idx
            },
            "watched_hold": {
                s: {"rank": rank_of(order_hold, idx[s]), "score": float(scores[idx[s]])}
                for s in WATCH if s in idx
            },
            "top_full": top_rows(order_self, stems, scores, TOPK),
            "top_hold": top_rows(order_hold, stems, scores, TOPK),
        }
        print(
            f"{name}: full NN={rec['nn_full']} ({rec['nn_full_score']:.3f})  "
            f"hold-Büdingen NN={rec['nn_hold']} ({rec['nn_hold_score']:.3f})  "
            f"332o hold-rank {rec['watched_hold']['332o']['rank']}  "
            f"RA-800r hold-rank {rec['watched_hold']['RA-800r']['rank']}"
        )
        return rec

    for s in BUDINGEN:
        reports.append(one_query(s, X[idx[s]], bud_i))

    centroid = l2(X[bud_i].mean(0, keepdims=True))[0]
    reports.append(one_query("Büdingen_avg", centroid, bud_i))

    payload = {
        "model_id": meta.get("model_id"),
        "checkpoint": meta.get("checkpoint"),
        "embeddings": str(npy),
        "n": len(stems),
        "vlad": "transductive-all-pages (Büdingen in codebook)",
        "metric": "cosine",
        "note": (
            "Codebook fit on 1304 gallery + 6 extras (332o kept). "
            "full = hold out the query page only. "
            "hold-Büdingen = also drop Büdingen1r and Büdingen1v."
        ),
        "queries": reports,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    thumbs = {}
    needed = set()
    for rec in reports:
        if rec["query"] in idx:
            needed.add(paths[idx[rec["query"]]])
        else:
            for s in BUDINGEN:
                needed.add(paths[idx[s]])
        for row in rec["top_hold"][:12] + rec["top_full"][:8]:
            needed.add(paths[stems.index(row["stem"])])
        for s in WATCH:
            if s in idx:
                needed.add(paths[idx[s]])
    for rel in needed:
        thumbs[nfc(Path(rel).stem)] = thumb_b64(root, rel)

    parts = ["""<!DOCTYPE html>
<html lang="nl"><head><meta charset="utf-8">
<title>Transductive all-pages — Büdingen NN</title>
<style>
body{font:14px/1.45 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;background:#111;color:#eee}
h1{font-size:22px;margin:0 0 6px}
.sub{color:#aaa;margin:0 0 18px}
section{border-top:1px solid #333;padding:22px 0}
h2{font-size:18px;margin:0 0 8px}
.hit{color:#6ddc8c;font-weight:700}
.miss{color:#ffb74d;font-weight:700}
.row{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0 14px}
fig{display:block;background:#1c1c1c;border:1px solid #333;border-radius:8px;padding:8px;width:170px}
fig.query{border-color:#5b9bff}
fig.needle{border-color:#e6c15a;box-shadow:0 0 0 2px #e6c15a}
fig.extra{border-color:#7e57c2}
fig img{width:100%;height:auto;display:block;background:#fff}
fig .cap{margin-top:6px;font-size:12px}
.rank{color:#888}
table{border-collapse:collapse}
td,th{padding:3px 10px;border-bottom:1px solid #333;text-align:left;font-variant-numeric:tabular-nums}
th{color:#aaa}
tr.needle td{color:#e6c15a}
tr.extra td{color:#c5a7ff}
.note{color:#aaa;font-size:13px}
</style></head><body>
<h1>Büdingen in a transductive VLAD that includes itself</h1>
"""]
    parts.append(
        f"<p class='sub'>{payload['note']} n={len(stems)}. "
        f"Gold = 332o. Purple = other extras (Genois, RA-800r, Büdingen).</p>"
    )
    for rec in reports:
        q = rec["query"]
        r332 = rec["watched_hold"]["332o"]["rank"]
        cls = "hit" if rec["nn_hold"] == "332o" else "miss"
        parts.append(f"<section id='{q}'><h2>{q}</h2>")
        parts.append(
            f"<p>Against everyone else ({rec['n_full']}): NN = "
            f"<b>{rec['nn_full']}</b> ({rec['nn_full_score']:.3f}).</p>"
            f"<p>Haystack without either Büdingen side ({rec['n_hold_budingen']}): "
            f"NN = <b>{rec['nn_hold']}</b> ({rec['nn_hold_score']:.3f}). "
            f"332o is <span class='{cls if rec['nn_hold']=='332o' else ('hit' if r332==1 else 'miss')}'>"
            f"rank {r332}</span> "
            f"({rec['watched_hold']['332o']['score']:.3f}); "
            f"RA-800r rank {rec['watched_hold']['RA-800r']['rank']} "
            f"({rec['watched_hold']['RA-800r']['score']:.3f}).</p>"
        )
        parts.append("<div class='row'>")
        if q in idx:
            parts.append(
                f"<fig class='query'><img src='data:image/jpeg;base64,{thumbs[q]}'>"
                f"<div class='cap'>query · {q}</div></fig>"
            )
        else:
            for s in BUDINGEN:
                parts.append(
                    f"<fig class='query'><img src='data:image/jpeg;base64,{thumbs[s]}'>"
                    f"<div class='cap'>query · {s}</div></fig>"
                )
        for row in rec["top_hold"][:12]:
            cls_f = "needle" if row["is_332o"] else ("extra" if row["is_extra"] else "")
            tag = " · 332o" if row["is_332o"] else (" · extra" if row["is_extra"] else "")
            parts.append(
                f"<fig class='{cls_f}'><img src='data:image/jpeg;base64,{thumbs[row['stem']]}'>"
                f"<div class='cap'><span class='rank'>#{row['rank']}</span> "
                f"{row['stem']}{tag}<br>cos {row['score']:.3f}</div></fig>"
            )
        parts.append("</div>")
        parts.append("<table><tr><th>Rank</th><th>Cosine</th><th>Charter</th></tr>")
        for row in rec["top_hold"]:
            cls_r = "needle" if row["is_332o"] else ("extra" if row["is_extra"] else "")
            mark = " · 332o" if row["is_332o"] else (" · extra" if row["is_extra"] else "")
            parts.append(
                f"<tr class='{cls_r}'><td>{row['rank']}</td>"
                f"<td>{row['score']:.4f}</td><td>{row['stem']}{mark}</td></tr>"
            )
        parts.append("</table></section>")
    parts.append(
        "<p class='note'>Table = haystack with both Büdingen pages removed. "
        "Codebook still saw their patches.</p></body></html>"
    )
    out_html.write_text("\n".join(parts))
    print(f"\nwrote {out_json}")
    print(f"wrote {out_html} ({out_html.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
