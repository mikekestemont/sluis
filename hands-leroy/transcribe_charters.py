#!/usr/bin/env python3
"""
Batch HTR of charter images via OpenRouter.

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    python transcribe_charters.py ../images/pages-zoned-stretched transcriptions-zoned
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3.7-flash"

SYSTEM_PROMPT = "You transcribe medieval charters. Output only the transcription."

USER_PROMPT = """Transcribe the handwritten text in this image.

Language: Middle Dutch or Latin, whichever is written. Keep the manuscript spelling (u/v, i/j, w as written). Expand abbreviations silently (nasal bars, p-stroke, con-/com-, -us/-is/-et). Use the modern Latin alphabet (long s → s; do not mark allographs).

Output rules:
- Output only the transcribed text. No title, no preface, no notes, no English.
- Lowercase everything.
- One manuscript line per output line.
- Illegible gap → [...]. Never refuse. Never describe damage. Never give alternatives.
- Start at the first line of the charter text. Ignore seals, dorse, rulers, colour charts."""

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def encode_image(path: Path, as_jpeg: bool = False) -> tuple[str, str]:
    if as_jpeg:
        from io import BytesIO
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(path).convert("RGB")
        buf = BytesIO()
        im.save(buf, "JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"
    ext = path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tif": "image/tiff", ".tiff": "image/tiff",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }
    return base64.b64encode(path.read_bytes()).decode("utf-8"), media_types.get(ext, "image/jpeg")


def message_text(msg: dict) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content or ""
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") in (None, "text"):
                parts.append(p.get("text") or "")
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return ""


def transcribe_image(image_path: Path, api_key: str, model: str,
                     reasoning_effort: str, max_retries: int = 4,
                     max_tokens: int = 16384) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        as_jpeg = attempt > 0
        b64_data, media_type = encode_image(image_path, as_jpeg=as_jpeg)
        payload = {
            "model": model,
            "max_tokens": max_tokens if attempt == 0 else max(max_tokens, 32768),
            "reasoning": {"effort": reasoning_effort, "exclude": True},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{b64_data}"},
                        },
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
            ],
        }
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers,
                                 json=payload, timeout=240)
            if resp.status_code == 401:
                raise SystemExit("OpenRouter 401: API key rejected.")
            if resp.status_code == 429:
                wait = 2 ** attempt * 5
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code >= 400:
                print(f"    API error {resp.status_code}: {resp.text[:500]}")
            resp.raise_for_status()
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            text = message_text(choice.get("message") or {})
            finish = choice.get("finish_reason") or choice.get("native_finish_reason")
            usage = data.get("usage") or {}
            if text.strip():
                return text
            print(f"    empty {image_path.name} finish={finish} usage={usage} "
                  f"jpeg={as_jpeg} attempt={attempt+1}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return ""
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt * 3
                print(f"    Error: {e}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    return ""


def process_batch(image_dir: Path, output_dir: Path, api_key: str,
                  model: str, workers: int, limit: int | None,
                  reasoning_effort: str, max_tokens: int = 16384) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        f for f in image_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        print(f"No image files found in {image_dir}")
        return

    todo = []
    skipped = 0
    for img in images:
        if (output_dir / (img.stem + ".txt")).exists():
            skipped += 1
        else:
            todo.append(img)
    if limit is not None:
        todo = todo[:limit]

    print(f"model {model}  reasoning {reasoning_effort}  max_tokens {max_tokens}")
    print(f"{len(todo)} to transcribe  ({skipped} already done, {len(images)} in folder)\n")
    if not todo:
        return

    done, failed = 0, []

    def process_one(img_path: Path) -> tuple[Path, str | None]:
        try:
            return img_path, transcribe_image(
                img_path, api_key, model, reasoning_effort,
                max_tokens=max_tokens)
        except Exception as e:
            print(f"  FAILED {img_path.name}: {e}")
            return img_path, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, img): img for img in todo}
        for future in as_completed(futures):
            img_path, text = future.result()
            if text is not None and text.strip():
                (output_dir / (img_path.stem + ".txt")).write_text(
                    text, encoding="utf-8")
                done += 1
                print(f"  [{done}/{len(todo)}] {img_path.name} ✓")
            else:
                if text is not None:
                    print(f"  EMPTY {img_path.name} (not written)")
                failed.append(img_path.name)

    print(f"\nDone: {done}/{len(todo)} transcribed")
    if failed:
        print(f"Failed: {', '.join(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch transcribe charter images via OpenRouter")
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                        help="Transcribe at most N remaining images")
    parser.add_argument("--reasoning-effort", default="low",
                        choices=["low", "medium", "high"],
                        help="Gemini 3.7 Flash requires reasoning; low is cheapest")
    parser.add_argument("--max-tokens", type=int, default=16384,
                        help="Completion budget (empty replies retry at 32768 as JPEG)")
    args = parser.parse_args()

    api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key.startswith("sk-or-v1-") or len(api_key) < 40:
        print("Error: OPENROUTER_API_KEY is missing or is not a real key.")
        print("It must start with sk-or-v1-  (not a placeholder like sk-or-... or")
        print("the words 'the key from your desktop').")
        print("Copy the full key from the file on your Desktop, then:")
        print('  export OPENROUTER_API_KEY="sk-or-v1-...."')
        sys.exit(1)

    process_batch(args.image_dir, args.output_dir, api_key,
                  args.model, args.workers, args.limit, args.reasoning_effort,
                  max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
