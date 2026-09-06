#!/usr/bin/env python3
"""Extract Gysseling palaeographic remarks from Corpus .fromdb files.

Two encodings in Reeks I (ambtelijke bescheiden):

1. Cross-charter identity, as Dutch editorial prose:
     "Geschreven door de hand van nr. 343"
2. Intra-document letter labels in registers:
     <q> [hand] <q> [A]

  python extract_gysseling_hands.py
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from match_charters import canon_gys, leroy_groups

ORIG = HERE / "cg-admin-orig"
HANDS = HERE / "handengroepen_gysseling.xlsx"
OUT_ATTR = HERE / "gysseling_hand_attributions.csv"
OUT_SECTIONS = HERE / "gysseling_section_hands.csv"
OUT_MENTIONS = HERE / "gysseling_hand_mentions.csv"

LITERARY = re.compile(r"^[3-8]\d{3}$")
HAND_BLOCK = re.compile(
    r"<statushand\s+statushandkode='([^']+)'>(.*?)"
    r"<end-statushand\s+statushandkode='\1'>",
    re.S | re.I,
)
Q_TOKEN = re.compile(r"<q>\s*([^<]+)")
EXPAND_A = re.compile(r"<A\s*>([^<]*)</A>", re.I)
HAND_MARK = re.compile(r"^\[hand[^\]]*\]$", re.I)
LABEL_MARK = re.compile(r"^\[([A-Z][A-Za-z0-9?'h,:%.-]*)\]$")
SAME_HAND = re.compile(
    r"(?:"
    r"geschreven\s+door\s+dezelfde\s+hand\s+als\s+nr\.?\s*"
    r"|geschreven\s+door\s+de\s+hand\s+van\s+nr\.?\s*"
    r"|dezelfde\s+hand\s+schreef\s+nr\.?\s*"
    r"|van\s+dezelfde\s+hand\s+als\s+nr\.?\s*"
    r")"
    r"(\d+[a-zA-Z']*)",
    re.I,
)


def is_admin(path: Path) -> bool:
    return not LITERARY.match(path.stem)


def q_tokens(block: str) -> list[str]:
    block = EXPAND_A.sub(r"\1", block)
    return [m.group(1).strip() for m in Q_TOKEN.finditer(block) if m.group(1).strip()]


def join_toks(toks: list[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(toks)).strip()


def strip_section_marks(toks: list[str]) -> list[str]:
    """Drop Gysseling's [hand] [A] markup so leftover prose can be classified."""
    out: list[str] = []
    i = 0
    while i < len(toks):
        if HAND_MARK.match(toks[i]):
            if i + 1 < len(toks) and LABEL_MARK.match(toks[i + 1]):
                i += 2
                continue
            i += 1
            continue
        if toks[i].lower() in {"[diverse]", "[handen:]", "[handen]"}:
            i += 1
            continue
        out.append(toks[i])
        i += 1
    return out


def section_labels(toks: list[str]) -> list[str]:
    labels = []
    i = 0
    while i < len(toks):
        if HAND_MARK.match(toks[i]) and i + 1 < len(toks):
            m = LABEL_MARK.match(toks[i + 1])
            if m:
                labels.append(m.group(1).rstrip(":").rstrip("?").rstrip(","))
                i += 2
                continue
        i += 1
    return labels


def classify_prose(text: str) -> str | None:
    if not re.search(r"\bhand\b", text, re.I):
        return None
    if SAME_HAND.search(text):
        return "same_hand"
    low = text.lower()
    if "hand van de voorzij" in low or "hand van de voorzijde" in low:
        return "recto_dorse"
    if re.search(r"\b(andere|latere|jongere|gelijktijdige)\s+hand\b", low):
        return "other_hand_addition"
    if re.search(r"door\s+dezelfde\s+hand", low) or re.search(
            r"dezelfde\s+hand\s+(boven|onder|vervangen|toegevoegd)", low):
        return "intra_document"
    return "unclassified"


def clip_note(text: str, span: re.Match | None, radius: int = 180) -> str:
    if span is None:
        return text[:400]
    a = max(0, span.start() - radius)
    b = min(len(text), span.end() + radius)
    snippet = text[a:b].strip()
    if a:
        snippet = "…" + snippet
    if b < len(text):
        snippet = snippet + "…"
    return snippet


def extract_file(path: Path) -> tuple[list[dict], Counter]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    nr = canon_gys(path.stem)
    notes: list[dict] = []
    labels: Counter = Counter()
    for m in HAND_BLOCK.finditer(raw):
        kode, body = m.group(1).lower(), m.group(2)
        toks = q_tokens(body)
        for lab in section_labels(toks):
            labels[lab] += 1
        prose = join_toks(strip_section_marks(toks))
        kind = classify_prose(prose)
        if kind is None:
            continue
        hit = SAME_HAND.search(prose)
        targets = [canon_gys(n) for n in SAME_HAND.findall(prose)] if hit else []
        notes.append({
            "gysseling_nr": nr,
            "file": path.name,
            "statushand": kode,
            "kind": kind,
            "targets": [t for t in targets if t],
            "note": clip_note(prose, hit if hit else re.search(r"\bhand\b", prose, re.I)),
        })
    return notes, labels


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", type=Path, default=ORIG)
    ap.add_argument("--hands", type=Path, default=HANDS)
    ap.add_argument("--out-attributions", type=Path, default=OUT_ATTR)
    ap.add_argument("--out-sections", type=Path, default=OUT_SECTIONS)
    ap.add_argument("--out-mentions", type=Path, default=OUT_MENTIONS)
    args = ap.parse_args()

    files = [f for f in sorted(args.orig.glob("*.fromdb")) if is_admin(f)]
    lookup, ambiguous = leroy_groups(args.hands)

    def groep(nr: str | None) -> str:
        if not nr:
            return ""
        if nr in ambiguous:
            return "ambiguous"
        return lookup.get(nr, "")

    attr_rows: list[dict] = []
    mention_rows: list[dict] = []
    section_rows: list[dict] = []
    kinds: Counter = Counter()

    for f in files:
        notes, labels = extract_file(f)
        nr = canon_gys(f.stem)
        if labels:
            section_rows.append({
                "gysseling_nr": nr,
                "n_labels": sum(labels.values()),
                "n_distinct": len(labels),
                "hands": ";".join(sorted(labels, key=lambda x: (len(x), x))),
                "counts": ";".join(f"{h}:{labels[h]}" for h in
                                   sorted(labels, key=lambda x: (len(x), x))),
                "leroy_groep": groep(nr),
                "file": f.name,
            })
        for rec in notes:
            kinds[rec["kind"]] += 1
            mention_rows.append({
                "gysseling_nr": rec["gysseling_nr"],
                "kind": rec["kind"],
                "same_hand_as": ";".join(rec["targets"]),
                "statushand": rec["statushand"],
                "leroy_groep": groep(rec["gysseling_nr"]),
                "note": rec["note"],
                "file": rec["file"],
            })
            if rec["kind"] != "same_hand":
                continue
            for tgt in rec["targets"] or [""]:
                gs, gt = groep(rec["gysseling_nr"]), groep(tgt) if tgt else ""
                agree = ""
                if tgt:
                    agree = ("yes" if gs and gt and gs == gt
                             else ("n/a" if not gs or not gt else "no"))
                attr_rows.append({
                    "gysseling_nr": rec["gysseling_nr"],
                    "same_hand_as": tgt,
                    "leroy_groep": gs,
                    "target_leroy_groep": gt,
                    "leroy_agrees": agree,
                    "statushand": rec["statushand"],
                    "note": rec["note"],
                    "file": rec["file"],
                })

    write_csv(args.out_attributions, attr_rows,
              ["gysseling_nr", "same_hand_as", "leroy_groep", "target_leroy_groep",
               "leroy_agrees", "statushand", "note", "file"])
    write_csv(args.out_sections, section_rows,
              ["gysseling_nr", "n_labels", "n_distinct", "hands", "counts",
               "leroy_groep", "file"])
    write_csv(args.out_mentions, mention_rows,
              ["gysseling_nr", "kind", "same_hand_as", "statushand", "leroy_groep",
               "note", "file"])

    print(f"admin .fromdb files scanned: {len(files)}")
    print(f"charters with [hand] [A/B/…] section labels: {len(section_rows)}")
    print(f"  distinct lettered hands (sum over charters): "
          f"{sum(r['n_distinct'] for r in section_rows)}")
    print("editorial 'hand' notes (after stripping section labels):")
    for k, n in kinds.most_common():
        print(f"  {k:22} {n}")
    print(f"cross-charter same-hand links: {len(attr_rows)}")
    print(f"  → {args.out_attributions}")
    print(f"  → {args.out_sections}")
    print(f"  → {args.out_mentions}")
    if attr_rows:
        print()
        for r in attr_rows:
            print(f"  {r['gysseling_nr']:>6} = {r['same_hand_as']:<6}  "
                  f"Leroy {r['leroy_groep'] or '—'} / {r['target_leroy_groep'] or '—'}  "
                  f"[{r['leroy_agrees']}]  ({r['statushand']})")
    if section_rows:
        print("\nregisters with the most distinct Gysseling letter-hands:")
        top = sorted(section_rows, key=lambda r: (-r["n_distinct"], -r["n_labels"]))[:8]
        for r in top:
            print(f"  {r['gysseling_nr']:>6}  {r['n_distinct']} hands  "
                  f"({r['n_labels']} labels)  {r['hands']}")


if __name__ == "__main__":
    main()
