#!/usr/bin/env python3
"""
Convert Corpus Gysseling .fromdb encoded files to plain text
that aligns with VLM transcription output conventions.

Conversion rules:
1. Strip all metadata/structural tags (header, datering, lokalisering,
   statushand, atlas_kenmerk, end-* tags, VN tags).
2. Expand <A> tags inline: "va<A >n</A>" → "van".
3. Strip <C ...> PoS/lemma tags; keep only the token that follows.
4. Handle <q> tags (non-Middle Dutch text): keep the token that follows.
5. Use <L page:line> tags to produce line breaks; prefix each line
   with its original line reference for traceability (optional).
6. Replace '+' in tokens with a space (separable compound parts).
7. Convert 'ho (superscript notation) → ho (strip apostrophe).
8. Collapse multiple spaces to single space; trim lines.
"""

import re
import sys
from pathlib import Path


def convert_fromdb(text: str) -> str:
    """Convert a .fromdb encoded text to plain transcription text."""

    # Step 1: Expand abbreviation tags: <A >...</A> or <A>...</A>
    # e.g., "va<A >n</A>" → "van", "en<A >de</A>" → "ende"
    text = re.sub(r'<A\s*>([^<]*)</A>', r'\1', text)

    # Step 2: Build output line by line using <L> tags
    lines = []
    current_line_tokens = []
    current_line_ref = None

    # Split into segments around <L ...> tags
    # We process the entire text sequentially
    # First, remove all metadata/structural tags (everything that is NOT <L>, <C>, <q>)
    # We'll process token by token

    # Remove structural/metadata tags entirely (including their full line if they
    # occupy the line alone). These are tags like <header>, <docId>, <genre>,
    # <bron*>, <datering>, <lokalisering>, <statushand>, <atlas_*>, <end-*>,
    # <VN ...>, and closing tags like </header>, </bron>.
    metadata_pattern = re.compile(
        r'</?(?:header|docId|genre|bron_afk|bron_oms|bron|'
        r'datering|end-datering|'
        r'lokalisering|end-lokalisering|'
        r'statushand|end-statushand|'
        r'atlas_kenmerk|end-atlas_kenmerk|'
        r'atlas_lokalisering|end-atlas_lokalisering|'
        r'VN)'
        r'[^>]*>'
    )
    text = metadata_pattern.sub('', text)

    # Remove any remaining content between metadata-only lines
    # (lines that after stripping only contain whitespace)

    # Now parse the annotated text: <L page:line> markers and <C ...> / <q> tokens
    # Strategy: scan for <L> tags to know line boundaries, then extract tokens

    # Split text by <L ...> markers. Each segment after an <L> tag belongs to that line.
    line_split = re.split(r'<L\s+([^>]+)>', text)

    # line_split[0] = text before first <L> (usually empty or metadata remnants)
    # line_split[1] = first line ref, line_split[2] = content until next <L>, etc.

    for i in range(1, len(line_split), 2):
        line_ref = line_split[i].strip()
        if i + 1 < len(line_split):
            content = line_split[i + 1]
        else:
            content = ''

        # Extract tokens from this line's content
        tokens = extract_tokens(content)
        if tokens:
            lines.append(' '.join(tokens))

    return '\n'.join(lines)


def extract_tokens(content: str) -> list[str]:
    """Extract surface-form tokens from a line's annotated content.

    Handles:
    - <C tag_lemma> token  →  token
    - <q> token            →  token (non-Middle Dutch text)
    - bare tokens (already cleaned of tags)
    """
    tokens = []

    # Remove any leftover structural tags that might remain
    content = re.sub(r'<[^>]*>', ' ', content)

    # Now we have just the tokens with possible whitespace
    # Split on whitespace
    raw_tokens = content.split()

    for tok in raw_tokens:
        tok = tok.strip()
        if not tok:
            continue

        # Convert 'ho (superscript notation) → ho before splitting
        tok = tok.replace("'ho", "ho")

        # Split compound tokens on '+' (e.g., "ons+leden", ".mho.+.ccho.+lxiij.")
        sub_tokens = tok.split('+')

        for st in sub_tokens:
            # Strip punctuation: keep only letters and digits
            st = re.sub(r'[^a-zA-Z0-9]', '', st)
            if st:
                tokens.append(st)

    return tokens


def process_directory(input_dir: Path, output_dir: Path):
    """Process all .fromdb files in input_dir, write .txt files to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fromdb_files = sorted(input_dir.glob('*.fromdb'))
    if not fromdb_files:
        print(f"No .fromdb files found in {input_dir}")
        return

    for fpath in fromdb_files:
        try:
            text = fpath.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            text = fpath.read_text(encoding='latin-1')
        converted = convert_fromdb(text)

        out_name = fpath.stem + '.txt'
        out_path = output_dir / out_name
        out_path.write_text(converted, encoding='utf-8')
        print(f"  {fpath.name} → {out_name}")

    print(f"\nConverted {len(fromdb_files)} files to {output_dir}")


if __name__ == '__main__':
    if len(sys.argv) == 3:
        input_dir = Path(sys.argv[1])
        output_dir = Path(sys.argv[2])
    else:
        # Default paths
        input_dir = Path('fromdb_corpus')
        output_dir = Path('txt_corpus')

    process_directory(input_dir, output_dir)
