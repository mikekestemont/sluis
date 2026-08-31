#!/usr/bin/env python3
"""Dump the sluis Label Studio project to a JSON file 05_apply_ls_zones.py can read.

Reads the local Label Studio sqlite (project title 'sluis'). Does not need a
browser export. Prefer this when LS is running on this machine.

  python code/05_ls_export.py --out data/ls_zones_export.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

LS_DB = Path.home() / "Library/Application Support/label-studio/label_studio.sqlite3"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "ls_zones_export.json"
PROJECT_TITLE = "sluis"


def dump(db: Path, title: str) -> list[dict]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    row = con.execute("SELECT id FROM project WHERE title = ?", (title,)).fetchone()
    if not row:
        raise SystemExit(f"no Label Studio project titled {title!r} in {db}")
    pid = row[0]
    # latest completion per task
    tasks = con.execute(
        """
        SELECT t.id, t.data, c.result, c.was_cancelled
        FROM task t
        JOIN task_completion c ON c.task_id = t.id
        WHERE t.project_id = ?
          AND c.id = (
              SELECT c2.id FROM task_completion c2
              WHERE c2.task_id = t.id
              ORDER BY c2.id DESC LIMIT 1
          )
        ORDER BY t.id
        """,
        (pid,),
    ).fetchall()
    out = []
    for tid, data, result, cancelled in tasks:
        payload = json.loads(data) if isinstance(data, str) else data
        res = json.loads(result) if isinstance(result, str) else (result or [])
        out.append({
            "id": tid,
            "data": payload,
            "annotations": [{
                "was_cancelled": bool(cancelled),
                "result": res,
            }],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=LS_DB)
    ap.add_argument("--title", default=PROJECT_TITLE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    tasks = dump(args.db, args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(tasks), encoding="utf-8")
    print(f"wrote {len(tasks)} tasks → {args.out}")


if __name__ == "__main__":
    main()
