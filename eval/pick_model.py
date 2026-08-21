"""
Measures which Gemini models actually work with this key, and how fast.

Written because the obvious default was wrong twice over: gemini-2.5-flash is
404 for new keys, and the newest model (3.7-flash) returned 503 under load. The
right choice is whichever is fastest among those that respond TODAY, so this
measures rather than assumes.

    ./.venv/bin/python eval/pick_model.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setu import llm  # noqa: E402  (imports .env as a side effect)

PROBE = (
    "Extract only what is stated; leave anything unsaid as null.\n"
    "'main tees saal ka hoon, roz paanch sau kamata hoon, "
    "aadhaar hai lekin bank account nahi hai'"
)


def main() -> int:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print("GEMINI_API_KEY is not set (put it in .env)")
        return 1

    listing = httpx.get(
        f"{llm.API_BASE}", headers={"x-goog-api-key": key}, timeout=30
    )
    listing.raise_for_status()

    candidates = [
        m["name"].split("/")[-1]
        for m in listing.json().get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
        and "flash" in m["name"]
        and not any(x in m["name"] for x in ("image", "tts", "omni"))
    ]

    body = {
        "contents": [{"parts": [{"text": PROBE}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": llm.PROFILE_SCHEMA,
        },
    }

    print(f"{'MODEL':30} {'STATUS':>7} {'SECONDS':>8}  EXTRACTED")
    print("-" * 92)

    working: list[tuple[float, str]] = []
    for model in candidates:
        started = time.time()
        try:
            r = httpx.post(
                f"{llm.API_BASE}/{model}:generateContent",
                headers={"x-goog-api-key": key},
                json=body,
                timeout=60,
            )
            elapsed = time.time() - start if (start := started) else 0
            if r.status_code == 200:
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                got = json.loads(text)
                summary = ", ".join(
                    f"{k}={v}" for k, v in got.items() if v not in (None, [])
                )
                print(f"{model:30} {'OK':>7} {elapsed:8.1f}  {summary[:44]}")
                working.append((elapsed, model))
            else:
                print(f"{model:30} {r.status_code:>7} {elapsed:8.1f}  {r.text[:44]}")
        except Exception as exc:
            print(f"{model:30} {'ERR':>7} {'-':>8}  {type(exc).__name__}: {str(exc)[:36]}")

    if not working:
        print("\nNothing responded. Check the key.")
        return 1

    working.sort()
    fastest = working[0]
    print(f"\nFastest working model: {fastest[1]} ({fastest[0]:.1f}s)")
    if fastest[1] != llm.MODEL:
        print(f"Currently configured:  {llm.MODEL}")
        print(f"To switch: set SETU_LLM_MODEL={fastest[1]} in .env")
    else:
        print(f"Already configured.    {llm.MODEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
