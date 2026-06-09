#!/usr/bin/env python3
"""
Batch HTR transcription of Middle Dutch charter images
using Claude Sonnet via OpenRouter API.

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    python3 transcribe_charters.py /path/to/images /path/to/output [--workers 3]
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-sonnet-4"

SYSTEM_PROMPT = """You are an expert paleographer specializing in 13th-century Middle Dutch (Middelnederlands) documentary texts. Your task is to transcribe the handwritten charter in this image following the conventions of the Corpus Gysseling (Corpus van Middelnederlandse Teksten, ed. M. Gysseling).

Transcription conventions:

1. Resolve all abbreviations silently. Expand nasal bars (e.g., a bar over a vowel before a consonant → insert "n" or "m" as appropriate), suspended word endings, abbreviated prefixes (e.g., "p" with stroke → "pro"/"pre"/"pri"), the "con-"/"com-" abbreviation, truncated "-us"/"-is"/"-et" endings, and all other standard abbreviations. Do not mark expansions with brackets or italics.
2. Letter forms. Render all characters using the modern Latin alphabet. Do not distinguish between long s (ſ) and round s of the different forms of R/r — transcribe both as "s" and "r". Keep the manuscript's own u/v distribution as written (do not normalize). Keep "w" as written. Transcribe "i" and "j" as found in the manuscript without normalization.
3. Capitalization. Use lowercase throughout, except for proper names (persons, places, saints) and the very first word of the text.
4. Word separation: preserve the word separation as present in the document.
5. No punctuation. Do not insert any modern punctuation. Do not reproduce any punctuation marks from the manuscript.
6. Preserve the line breaks in the manuscript. Do not produce the transcription as a single continuous block of text.
7. Numerals and dates. Transcribe Roman numerals as they appear. Render superscript letters inline (e.g., superscript "o" after a Roman numeral → "o" appended directly, like "Mho CCho").
8. Damaged or illegible text. If a passage is completely illegible, indicate with "[...]". If you can read it partially but are uncertain, give your best reading without brackets — do not hedge with multiple options.
9. Do not add any commentary, headers, metadata, or notes. Output only the transcribed main text. Transcribe the charter in this image now."""

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}


def encode_image(path: Path) -> tuple[str, str]:
    """Read and base64-encode an image file. Returns (base64_data, media_type)."""
    ext = path.suffix.lower()
    media_types = {
        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.tif': 'image/tiff', '.tiff': 'image/tiff',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
    }
    media_type = media_types.get(ext, 'image/jpeg')
    data = base64.b64encode(path.read_bytes()).decode('utf-8')
    return data, media_type


def transcribe_image(image_path: Path, api_key: str, max_retries: int = 3) -> str:
    """Send a single image to the API and return the transcription."""
    b64_data, media_type = encode_image(image_path)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "max_tokens": 4096,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64_data}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Transcribe this charter."
                    }
                ]
            }
        ]
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers,
                                 json=payload, timeout=120)

            if resp.status_code == 429:
                wait = 2 ** attempt * 5
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code >= 400:
                print(f"    API error {resp.status_code}: {resp.text[:500]}")

            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt * 3
                print(f"    Error: {e}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    return ""


def process_batch(image_dir: Path, output_dir: Path, api_key: str,
                  workers: int = 3):
    """Process all images in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        f for f in image_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not images:
        print(f"No image files found in {image_dir}")
        return

    # Skip already-transcribed files
    todo = []
    for img in images:
        out_path = output_dir / (img.stem + '.txt')
        if out_path.exists():
            print(f"  Skipping {img.name} (already done)")
        else:
            todo.append(img)

    print(f"\n{len(todo)} images to transcribe ({len(images) - len(todo)} already done)\n")

    if not todo:
        return

    done = 0
    failed = []

    def process_one(img_path: Path) -> tuple[Path, str | None]:
        try:
            text = transcribe_image(img_path, api_key)
            return img_path, text
        except Exception as e:
            print(f"  FAILED {img_path.name}: {e}")
            return img_path, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, img): img for img in todo}

        for future in as_completed(futures):
            img_path, text = future.result()
            if text is not None:
                out_path = output_dir / (img_path.stem + '.txt')
                out_path.write_text(text, encoding='utf-8')
                done += 1
                print(f"  [{done}/{len(todo)}] {img_path.name} ✓")
            else:
                failed.append(img_path.name)

    print(f"\nDone: {done}/{len(todo)} transcribed")
    if failed:
        print(f"Failed: {', '.join(failed)}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch transcribe Middle Dutch charter images via OpenRouter"
    )
    parser.add_argument("image_dir", type=Path, help="Folder with charter images")
    parser.add_argument("output_dir", type=Path, help="Folder for transcription output")
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel API requests (default: 3)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: set OPENROUTER_API_KEY environment variable")
        print('  export OPENROUTER_API_KEY="sk-or-..."')
        sys.exit(1)

    process_batch(args.image_dir, args.output_dir, api_key, args.workers)


if __name__ == "__main__":
    main()
