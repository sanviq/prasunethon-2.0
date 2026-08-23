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
# separate conversations start from a clean profile.
#
# These are SHORT, natural turns, because that is what people actually say. The
# first version crammed five facts into one sentence so the demo would resolve
# in a single turn, which flattered the system and hid the thing that matters:
# Setu earns those facts by asking, one question at a time.
#
# Nothing here constrains what the system accepts -- this file is a cache
# warmer and nothing in setu/ imports it. Any sentence works; a warmed one is
# just free instead of 1.3s.
CONVERSATIONS: dict[str, list[list[str]]] = {
    "mr": [[
        "mi bhaji vikto",
        "tees varshacha aahe",
        "aadhaar aahe, bank account nahi",
        "nahi, mi income tax bharat nahi",
        "nahi, mi kuthlahi sarkari karj ghetla nahi",
        "hoy majhe jan dhan khate ughadle aahe",
    ]],
    "hi": [[
        "main sabzi bechta hoon",
        "meri umar tees saal hai",
        "aadhaar hai lekin bank account nahi hai",
        "nahi, main income tax nahi bharta",
        "nahi, koi sarkari loan nahi liya",
        "haan mera jan dhan khata khul gaya hai",
    ]],
    "en": [[
        "I sell vegetables from a cart",
        "I am thirty years old",
        "I have Aadhaar but no bank account",
        "no, I do not pay income tax",
        "no, I have not taken any government loan",
    ]],
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


def warm_cards() -> tuple[int, int]:
    """
    Translate the whole scheme catalogue into each language, once.

    The card text is translated on demand and cached, so the FIRST caller in a
    language pays for the entire catalogue mid-conversation. Warming it here
    moves that cost off the demo.
    """
    from setu.rules import load_schemes

    strings = []
    for scheme in load_schemes()["schemes"]:
        strings.append(scheme["benefit_summary"])
        for rule in scheme["rules"]:
            strings.append(rule["description"])
            strings.append(rule["source_quote"])
            if rule.get("remedy"):
                strings.append(rule["remedy"]["action"])
                strings.append(rule["remedy"]["where"])

    strings = list(dict.fromkeys(strings))
    done = failed = 0
    for language in CONVERSATIONS:
        if language == "en":
            continue
        try:
            out = llm.translate(strings, language)
            ok = out != strings
            print(f"  [{language}] {len(strings)} strings {'translated' if ok else 'UNCHANGED'}")
            done += 1 if ok else 0
            failed += 0 if ok else 1
        except Exception as exc:
            print(f"  [{language}] FAILED {type(exc).__name__}: {str(exc)[:60]}")
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


def warm_remote(base_url: str) -> int:
    """
    Warm a DEPLOYED instance by driving its own /ask/text.

    Everything else in this file warms the cache on the machine it runs on,
    which is the wrong machine once Setu is deployed: the cache lives inside
    the container, and a laptop cannot reach in and fill it.

    Worth knowing before relying on this: a container that scales to zero loses
    the cache when it stops, so this has to be re-run after every cold start.
    If the demo is being judged live, either keep one instance warm or commit
    data/llm_cache into the image and stop depending on timing.
    """
    import httpx

    base = base_url.rstrip("/")
    done = failed = 0

    for language, conversations in CONVERSATIONS.items():
        for n, turns in enumerate(conversations):
            session = f"warm-{language}-{n}"

            # Reset first, or the second run is not a repeat of the first: the
            # server still holds the finished profile, so turn one is answered
            # from a state the real caller will never be in and every prompt --
            # and so every cache key -- comes out different. The first version
            # of this loop skipped it and showed no speed-up at all on a second
            # pass, which is exactly the symptom its own closing message warns
            # about.
            try:
                httpx.delete(f"{base}/session/{session}", timeout=30)
            except Exception:  # noqa: BLE001 - a fresh instance has no session
                pass

            for turn in turns:
                started = time.time()
                try:
                    reply = httpx.post(
                        f"{base}/ask/text",
                        data={"text": turn, "language": language, "session_id": session},
                        timeout=120,
                    )
                    reply.raise_for_status()
                    done += 1
                    print(f"  [{language}] {time.time() - started:5.1f}s  {turn[:44]}…")
                except Exception as exc:  # noqa: BLE001 - report and keep going
                    failed += 1
                    print(f"  [{language}] FAILED  {type(exc).__name__}: {exc}")
                    print(f"           {turn[:44]}…")

    print(f"\n{done} turns warmed, {failed} failed against {base}")
    if failed:
        return 1

    print("Run it once more -- the second pass should be visibly faster, and if")
    print("it is not, the cache is not being written where you think it is.")
    return 0


def main() -> int:
    # A deployed instance is warmed over HTTP; a local one, in process.
    for flag in ("--url", "--remote"):
        if flag in sys.argv:
            return warm_remote(sys.argv[sys.argv.index(flag) + 1])

    print(f"model: {llm.MODEL}   whisper: {voice.WHISPER_SIZE}\n")

    print("Pipeline (extract -> rules -> narrate -> speak)")
    pipeline_done, pipeline_failed = warm_pipeline()

    print("\nScheme catalogue translation")
    cards_done, cards_failed = warm_cards()

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

    if pipeline_failed or static_failed or cards_failed:
        print(f"\n{pipeline_failed + static_failed + cards_failed} warm-ups failed; "
              "fix before demoing")
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
