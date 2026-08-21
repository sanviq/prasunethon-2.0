"""
Fill the LLM and TTS caches before a demo.

Run this over wifi you trust. Afterwards the scripted demo path replays with no
network at all, which matters because conference wifi is the single most likely
thing to break at the worst moment.

    ./.venv/bin/python scripts/prewarm.py

WHAT THIS DOES AND DOES NOT BUY YOU

Both caches key on exact content. A prewarmed sentence is free forever; a
sentence one word different is a fresh call.

That matters because live speech never transcribes identically twice. Whisper
will render the same spoken sentence slightly differently run to run, so a LIVE
MIC DEMO WILL MISS THE CACHE and go to the network. That is fine -- extraction
is ~1.3s -- but it means SETU_LLM_MODE=offline is NOT the mode to run a live
voice demo in. It is the mode to fall back to when the network dies, using
/ask/text with the exact sentences below.

So: prewarm for the safety net, not for the happy path.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from setu import llm, voice  # noqa: E402
from setu.ladder import best_paths  # noqa: E402
from setu.rules import Status, evaluate_all, missing_fields  # noqa: E402

# The scripted demo arc. Each inner list is ONE conversation -- turns chain, and
# separate conversations start from a clean profile. Flattening these caused a
# fresh caller to inherit the previous one's bank account and open on six
# eligible schemes she had never earned.
#
# Keep the phrasing parallel across languages. The first pass had the Hindi say
# "apna thela" (my OWN cart) and the Marathi merely "I sell vegetables"; the
# first establishes self-employment and so settles EPFO, the second does not, so
# the Marathi demo opened on zero eligible schemes for no reason but wording.
CONVERSATIONS: dict[str, list[list[str]]] = {
    "hi": [
        [
            "main tees saal ka hoon, apna thela lagata hoon 2015 se, roz paanch sau "
            "kamata hoon, aadhaar hai lekin bank account nahi hai, income tax nahi bharta",
            "haan mera jan dhan khata khul gaya hai",
        ],
        [
            "main apni sabzi ki dukaan chalata hoon, meri umar chalis saal hai, "
            "aadhaar aur bank account dono hain, income tax nahi bharta",
        ],
    ],
    "mr": [
        [
            "mi tees varshacha aahe, majha swatacha bhajiwala thela aahe 2015 pasun, "
            "roj paachshe rupaye miltat, aadhaar aahe pan bank account nahi, "
            "income tax bharat nahi",
            "hoy majhe jan dhan khate ughadle aahe",
        ],
    ],
    "en": [
        [
            "I am thirty years old, I run my own vegetable cart since 2015, I earn "
            "five hundred rupees a day, I have Aadhaar but no bank account, and I do "
            "not pay income tax",
        ],
    ],
}

# Fixed lines the interface speaks regardless of what the caller said.
STATIC: dict[str, list[str]] = {
    "hi": [
        "नमस्ते, बोलिए। आप क्या काम करते हैं?",
        "माफ़ कीजिए, मुझे सुनाई नहीं दिया। दोबारा बोलिए।",
        "एक मिनट रुकिए, मैं देख रहा हूँ।",
    ],
    "mr": [
        "नमस्कार, बोला. तुम्ही काय काम करता?",
        "माफ करा, मला ऐकू आले नाही. पुन्हा बोला.",
        "एक मिनिट थांबा, मी बघतो आहे.",
    ],
    "en": [
        "Hello, please speak. What work do you do?",
        "Sorry, I did not catch that. Please say it again.",
        "One moment, let me check.",
    ],
}


def warm_pipeline() -> tuple[int, int]:
    """Run each scripted sentence through extract -> rules -> narrate -> speak."""
    done = failed = 0

    for language, conversations in CONVERSATIONS.items():
        for conversation in conversations:
            profile = None  # each conversation is a different person
            for line in conversation:
                started = time.time()
                try:
                    # Chained within a conversation: turn two is a different cache
                    # entry than turn two in isolation, because the profile it
                    # starts from differs.
                    profile = llm.extract_profile(line, language, base=profile)
                    decisions = evaluate_all(profile)
                    paths = best_paths(profile, decisions)
                    spoken = llm.narrate(decisions, paths, language, missing_fields(profile))
                    audio = voice.speak(spoken, language)

                    eligible = [d.scheme_name for d in decisions if d.status is Status.ELIGIBLE]
                    print(
                        f"  [{language}] {time.time() - started:4.1f}s  "
                        f"{audio.stat().st_size // 1024:>4}KB  "
                        f"eligible={len(eligible)} ladder={len(paths)}  "
                        f"{line[:42]}…"
                    )
                    done += 1
                except Exception as exc:
                    print(f"  [{language}] FAILED  {type(exc).__name__}: {str(exc)[:70]}")
                    print(f"           {line[:70]}…")
                    failed += 1

    return done, failed


def warm_static() -> tuple[int, int]:
    done = failed = 0
    for language, lines in STATIC.items():
        for line in lines:
            try:
                path = voice.speak(line, language)
                print(f"  [{language}] {path.stat().st_size // 1024:>4}KB  {line[:46]}")
                done += 1
            except Exception as exc:
                print(f"  [{language}] FAILED  {type(exc).__name__}: {str(exc)[:60]}")
                failed += 1
    return done, failed


def verify_offline() -> bool:
    """
    Re-run the scripts with the network refused, proving the fallback works.

    Warming without checking is how you find out on stage that a sentence never
    made it in.
    """
    llm_mode, voice_mode = llm.LLM_MODE, voice.VOICE_MODE
    llm.LLM_MODE, voice.VOICE_MODE = "offline", "offline"

    ok = True
    try:
        for language, conversations in CONVERSATIONS.items():
            for conversation in conversations:
                profile = None
                for line in conversation:
                    try:
                        profile = llm.extract_profile(line, language, base=profile)
                        decisions = evaluate_all(profile)
                        paths = best_paths(profile, decisions)
                        spoken = llm.narrate(decisions, paths, language, missing_fields(profile))
                        voice.speak(spoken, language)
                    except Exception as exc:
                        print(f"  MISS [{language}] {type(exc).__name__}: {line[:50]}…")
                        ok = False
    finally:
        llm.LLM_MODE, voice.VOICE_MODE = llm_mode, voice_mode

    return ok


def main() -> int:
    print(f"model: {llm.MODEL}   whisper: {voice.WHISPER_SIZE}\n")

    print("Pipeline (extract -> rules -> narrate -> speak)")
    pipeline_done, pipeline_failed = warm_pipeline()

    print("\nStatic interface lines")
    static_done, static_failed = warm_static()

    llm_files = len(list(llm.CACHE_DIR.glob("*.txt"))) if llm.CACHE_DIR.exists() else 0
    audio = list(voice.CACHE_DIR.glob("*.mp3")) if voice.CACHE_DIR.exists() else []
    audio_kb = sum(f.stat().st_size for f in audio) // 1024

    print(f"\ncached: {llm_files} model responses, {len(audio)} audio files ({audio_kb}KB)")

    print("\nVerifying the offline fallback…")
    if verify_offline():
        print("  every scripted sentence replays with the network refused")
    else:
        print("  SOME SENTENCES ARE NOT CACHED -- see misses above")
        return 1

    if pipeline_failed or static_failed:
        print(f"\n{pipeline_failed + static_failed} warm-ups failed; fix before demoing")
        return 1

    print("\nFallback if the network dies mid-demo:")
    print("  SETU_LLM_MODE=offline SETU_VOICE_MODE=offline \\")
    print("    ./.venv/bin/uvicorn setu.api:app --port 8000")
    print("  …then drive it from /ask/text with the exact sentences in this file.")
    print("  A live mic will still miss the cache -- Whisper never transcribes")
    print("  identically twice -- so offline mode is the fallback, not the plan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
