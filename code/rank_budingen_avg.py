#!/usr/bin/env python3
"""Needle-in-haystack: average Büdingen r+v vs the Sauvola SSL e15 gallery.

Run on mike from ~/mole (or pass --root). Writes JSON + a small HTML ranking sheet.
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

TARGET = "332o"
QUERY_STEMS = ["Büdingen1r", "Büdingen1v"]
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


def thumb_b64(path: Path, max_side: int = 280) -> str:
    im = Image.open(path).convert("L")
    im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def pack(order, stems, scores, k):
    rows = []
    for r, i in enumerate(order[:k], 1):
        rows.append({
            "rank": r,
            "stem": stems[i],
            "score": float(scores[i]),
            "is_target": stems[i] == TARGET,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.home() / "mole")
    args = ap.parse_args()
    root = args.root

    g_npy = root / "outputs/sluis/leroy.sauvola.ssl.e15.npy"
    q_npy = root / "outputs/sluis/leroy.queries.sauvola.ssl.e15.npy"
    g_img = root / "data/leroy-sauvola"
    q_img = root / "data/leroy-queries-sauvola"
    out_json = root / "outputs/sluis/budingen_avg_nn_e15.json"
    out_html = root / "outputs/sluis/budingen_avg_nn_e15.html"

    g_stems, g_paths, g_meta = stems_from(g_npy.with_suffix(".mapping.json"))
    q_stems, q_paths, _q_meta = stems_from(q_npy.with_suffix(".mapping.json"))
    G = np.load(g_npy).astype(np.float64)
    Q = np.load(q_npy).astype(np.float64)

    want = [nfc(s) for s in QUERY_STEMS]
    q_idx = []
    for s in want:
        hits = [i for i, t in enumerate(q_stems) if t == s]
        if not hits:
            raise SystemExit(f"query stem not found: {s!r} among {q_stems}")
        q_idx.append(hits[0])

    qhat = l2(Q[q_idx])
    centroid = l2(qhat.mean(axis=0))
    ghat = l2(G)
    scores = ghat @ centroid

    per_side = {QUERY_STEMS[j]: (ghat @ qhat[j]) for j in range(len(q_idx))}
    max_agg = np.maximum(per_side[QUERY_STEMS[0]], per_side[QUERY_STEMS[1]])

    order = np.argsort(-scores)
    target_i = g_stems.index(TARGET)
    target_rank = int(np.where(order == target_i)[0][0] + 1)
    target_score = float(scores[target_i])
    nn_i = int(order[0])
    top = pack(order, g_stems, scores, TOPK)

    print(f"Query: average of {QUERY_STEMS}")
    print(f"Gallery: {len(g_stems)} pages  model={g_meta.get('model_id')}")
    print(f"{TARGET}: rank {target_rank} / {len(g_stems)}  cosine={target_score:.4f}")
    print(f"NN: {g_stems[nn_i]}  cosine={float(scores[nn_i]):.4f}")
    print()
    print(f"{'Rank':>4}  {'Score':>8}  Filename")
    print("-" * 40)
    for row in top:
        mark = " ◄ NEEDLE" if row["is_target"] else ""
        print(f"{row['rank']:>4}  {row['score']:>8.3f}  {row['stem']}{mark}")

    print(f"\nPer-side rank of {TARGET}:")
    side_report = {}
    for name, sc in per_side.items():
        o = np.argsort(-sc)
        r = int(np.where(o == target_i)[0][0] + 1)
        side_report[name] = {
            "target_rank": r,
            "target_score": float(sc[target_i]),
            "nn_stem": g_stems[int(o[0])],
            "nn_score": float(sc[o[0]]),
        }
        print(
            f"  {name}: rank {r}  cosine={float(sc[target_i]):.4f}  "
            f"NN={g_stems[o[0]]} ({float(sc[o[0]]):.4f})"
        )
    o_max = np.argsort(-max_agg)
    r_max = int(np.where(o_max == target_i)[0][0] + 1)
    print(
        f"  max-aggregate: rank {r_max}  score={float(max_agg[target_i]):.4f}  "
        f"NN={g_stems[o_max[0]]} ({float(max_agg[o_max[0]]):.4f})"
    )
    rv = float(qhat[0] @ qhat[1])
    print(f"\nBüdingen r↔v cosine: {rv:.4f}")

    report = {
        "model_id": g_meta.get("model_id"),
        "checkpoint": g_meta.get("checkpoint"),
        "gallery_npy": str(g_npy),
        "query_npy": str(q_npy),
        "gallery_n": len(g_stems),
        "query_stems": QUERY_STEMS,
        "aggregate": "l2_mean_centroid",
        "metric": "cosine",
        "target": TARGET,
        "target_rank": target_rank,
        "target_score": target_score,
        "nn_stem": g_stems[nn_i],
        "nn_score": float(scores[nn_i]),
        "budingen_rv_cosine": rv,
        "per_side": side_report,
        "max_aggregate": {
            "target_rank": r_max,
            "target_score": float(max_agg[target_i]),
            "nn_stem": g_stems[int(o_max[0])],
            "nn_score": float(max_agg[o_max[0]]),
        },
        "top": top,
    }
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    q_thumbs = []
    for i in q_idx:
        p = q_img / Path(q_paths[i]).name
        q_thumbs.append((q_stems[i], thumb_b64(p, 360)))

    show = list(order[:TOPK])
    if target_i not in show:
        show.append(target_i)
    cards = []
    for rnk, i in enumerate(order, 1):
        if i not in show:
            continue
        p = g_img / Path(g_paths[i]).name
        cards.append((rnk, g_stems[i], float(scores[i]), g_stems[i] == TARGET, thumb_b64(p)))

    hit = target_rank == 1
    verdict = "hit" if hit else "miss"
    parts = [
        "<!DOCTYPE html>",
        '<html lang="nl"><head><meta charset="utf-8">',
        "<title>Büdingen average → gallery NN (Sauvola SSL e15)</title>",
        "<style>",
        "body{font:14px/1.4 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;"
        "margin:24px;background:#111;color:#eee}",
        "h1{font-size:20px;font-weight:650;margin:0 0 8px}",
        ".sub{color:#aaa;margin-bottom:20px}",
        ".hit{color:#6ddc8c;font-weight:700}",
        ".miss{color:#ff8a80;font-weight:700}",
        ".row{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0 20px}",
        "fig{display:block;background:#1c1c1c;border:1px solid #333;"
        "border-radius:8px;padding:8px;width:200px}",
        "fig.needle{border-color:#e6c15a;box-shadow:0 0 0 2px #e6c15a}",
        "fig img{width:100%;height:auto;display:block;background:#fff}",
        "fig .cap{margin-top:6px;font-size:12px}",
        ".rank{color:#888}",
        "table{border-collapse:collapse;margin-top:8px}",
        "td,th{padding:3px 10px;border-bottom:1px solid #333;text-align:left}",
        "th{color:#aaa;font-weight:600}",
        ".gold{color:#e6c15a}",
        "</style></head><body>",
        "<h1>Speld in de hooiberg — averaged Büdingen</h1>",
        f"<p class='sub'>Sauvola SSL epoch 15 · VLAD-100 frozen gallery codebook · "
        f"cosine · gallery {len(g_stems)} pages (queries not in the haystack)</p>",
        f"<p>Query = L2-mean of <b>Büdingen1r</b> + <b>Büdingen1v</b>. "
        f"Needle <b>{TARGET}</b> is "
        f"<span class='{verdict}'>rank {target_rank}</span> of {len(g_stems)} "
        f"(cosine {target_score:.3f}). "
        f"Nearest neighbour: <b>{g_stems[nn_i]}</b> ({float(scores[nn_i]):.3f}).</p>",
        "<h2>Query</h2><div class='row'>",
    ]
    for stem, b64 in q_thumbs:
        parts.append(
            f"<fig><img src='data:image/jpeg;base64,{b64}'>"
            f"<div class='cap'>{stem}</div></fig>"
        )
    parts.append("</div><h2>Gallery ranking</h2><div class='row'>")
    for rnk, stem, sc, is_t, b64 in cards:
        cls = "needle" if is_t else ""
        tag = " · needle" if is_t else ""
        parts.append(
            f"<fig class='{cls}'><img src='data:image/jpeg;base64,{b64}'>"
            f"<div class='cap'><span class='rank'>#{rnk}</span> {stem}{tag}"
            f"<br>cos {sc:.3f}</div></fig>"
        )
    parts.append("</div><h2>Top 20</h2><table><tr><th>Rank</th><th>Cosine</th><th>Charter</th></tr>")
    for row in top:
        cls = " class='gold'" if row["is_target"] else ""
        parts.append(
            f"<tr{cls}><td>{row['rank']}</td><td>{row['score']:.4f}</td>"
            f"<td>{row['stem']}</td></tr>"
        )
    parts.append("</table></body></html>")
    out_html.write_text("\n".join(parts))
    print(f"\nwrote {out_json}")
    print(f"wrote {out_html} ({out_html.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
