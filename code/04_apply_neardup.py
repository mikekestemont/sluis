#!/usr/bin/env python3
"""Apply neardup_decisions.csv: log extras as gallery drops, keep PNGs on disk."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ROOT / "outputs" / "neardup_decisions.csv"
MANIFEST = ROOT / "data" / "manifest.csv"
DATA_COPY = ROOT / "data" / "neardup_decisions.csv"


def main() -> None:
    if not DECISIONS.is_file():
        raise SystemExit(f"missing {DECISIONS}")
    shutil.copy2(DECISIONS, DATA_COPY)

    rows = list(csv.DictReader(DECISIONS.open(encoding="utf-8")))
    drop_to_keep: dict[str, str] = {}
    n_same = 0
    for r in rows:
        if r.get("same") != "1":
            continue
        n_same += 1
        keep = r["keep"].strip()
        for name in (r.get("drop") or "").split("|"):
            name = name.strip()
            if name:
                drop_to_keep[name] = keep
    if not drop_to_keep:
        print("no same-leaf drops in", DECISIONS)
        return

    man = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    fields = list(man[0].keys())
    applied, skipped = [], []
    for r in man:
        if r.get("side") != "recto":
            continue
        png = Path(r.get("released_path") or "").name
        if png not in drop_to_keep:
            continue
        keep = drop_to_keep[png]
        reason = f"neardup_of={keep.replace('.png', '')}"
        if r.get("main_document") == "0" and r.get("reason") == reason:
            skipped.append(png)
            continue
        r["main_document"] = "0"
        r["reason"] = reason
        applied.append(png)

    with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(man)

    n_main = sum(1 for r in man if r.get("side") == "recto" and r.get("main_document") == "1")
    print(f"clusters marked same {n_same}")
    print(f"dropped {len(applied)}  already {len(skipped)}  listed {len(drop_to_keep)}")
    print(f"gallery main rectos now {n_main}")
    print(f"decisions → {DATA_COPY}")
    for png, keep in sorted(drop_to_keep.items(), key=lambda t: int(t[0].replace("o.png", ""))):
        print(f"  {png:12} → {keep}")


if __name__ == "__main__":
    main()
