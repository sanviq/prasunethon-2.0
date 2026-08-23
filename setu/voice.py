"""
Voice in, voice out.

    ASR:  faster-whisper, local, no key, auto-detects language
    TTS:  edge-tts, free, no key, one neural voice per language

Both cache on a content hash. The demo has to survive with no internet, and
re-synthesising a sentence we already have is a needless risk on stage. Set
SETU_VOICE_MODE=offline to refuse any network call and serve cache only -- run
the live demo that way.

Whisper size defaults to `small`, measured on the same clip rather than guessed:

    base    2.5s   transcribes Hindi speech into Urdu script -- unusable
    small   4.9s   "मैं मुमबाई में सबजी बेजती हूँ मेरी उम्र चोथटीस साल है"
    medium 14.4s   clean Devanagari, correct spelling throughout

`medium` was the default until a real phone test over a tunnel took 70 seconds
to answer one sentence. Fourteen seconds of that is this line, and a caller
watching a spinner for that long has already decided the thing is broken.

The reason `medium` was chosen originally no longer holds. It was picked because
`small` garbled spoken ages past recovery -- and it does still garble them:
"चोथटीस" is not how anyone writes 34. But extraction now reads that transcript and
returns age 34, occupation street_vendor, state Maharashtra, documents [aadhaar],
every field correct. The model downstream is doing work the ASR used to have to
do alone, so the accuracy that `medium` was buying is no longer for sale here.

ASR is still ~70% of turn latency; everything downstream is ~1.7s of extraction
and milliseconds of rules. If a turn feels slow, this is the first knob to reach
for -- SETU_WHISPER_SIZE=medium if a machine can spare the time, never `base`.

Deliberately not Bhashini: it needs institutional approval we do not have. The
adapter shape below means adding it later is one function, not a rewrite.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "voice_cache"
VOICE_MODE = os.getenv("SETU_VOICE_MODE", "auto")  # auto | offline
WHISPER_SIZE = os.getenv("SETU_WHISPER_SIZE", "small")

# edge-tts neural voices, one per supported language. Male by default: the demo
# persona is a man, and a mismatched voice is a small thing that quietly
# undermines a pitch.
VOICES = {
    "hi": "hi-IN-MadhurNeural",
    "mr": "mr-IN-ManoharNeural",
    "en": "en-IN-PrabhatNeural",
    "kn": "kn-IN-GaganNeural",
    "ta": "ta-IN-ValluvarNeural",
    "te": "te-IN-MohanNeural",
    "bn": "bn-IN-BashkarNeural",
    "gu": "gu-IN-NiranjanNeural",
}

DEFAULT_LANGUAGE = "hi"

# Below this a webm container is a tap, not a sentence. MediaRecorder emits a
# few hundred bytes of header even when nothing was said.
MIN_CLIP_BYTES = 2048


class VoiceError(RuntimeError):
    pass


class VoiceOffline(VoiceError):
    """Asked to synthesise something not in cache while offline mode is on."""


# --------------------------------------------------------------------------
# ASR
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _whisper():
    """
    Loaded once, lazily.

    Importing faster-whisper pulls in ctranslate2 and costs a few seconds, so it
    must not happen at module import -- the test suite and the rule engine have
    no business paying for it.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - depends on install
        raise VoiceError("faster-whisper is not installed") from exc

    # cpu_threads matters more than anything else here: the default left
    # medium at 16.8s on this machine and pinning threads brought it to 9.4s
    # for the same audio and the same output. Nearly half the turn latency was
    # a default nobody had looked at.
    threads = int(os.getenv("SETU_WHISPER_THREADS", "0")) or min(8, os.cpu_count() or 4)
    return WhisperModel(
        WHISPER_SIZE, device="cpu", compute_type="int8", cpu_threads=threads
    )


def preload() -> None:
    """
    Load the model before the first caller needs it.

    Lazily loading costs the FIRST request ~25 seconds, and on stage the first
    request is the demo. Called from the API's startup hook so the wait happens
    while uvicorn boots, where nobody is watching.
    """
    _whisper()


def transcribe(audio_path: str | Path, language: str | None = None) -> tuple[str, str]:
    """
    Audio in, (transcript, detected_language) out.

    Passing `language` skips detection. Worth doing on the second turn of a
    conversation: we already know what they speak, and detection on a two-word
    answer like "haan" is a coin flip that can silently switch the whole session
    into the wrong language.
    """
    size = Path(audio_path).stat().st_size if Path(audio_path).exists() else 0
    if size < MIN_CLIP_BYTES:
        # A tap rather than a sentence. Decoding this raises deep inside PyAV
        # with a message about invalid data, which tells the caller nothing.
        raise VoiceError("recording too short — hold the button and speak")

    try:
        segments, info = _whisper().transcribe(
            str(audio_path),
            language=language,
            vad_filter=True,  # drops market noise between phrases
            beam_size=1,      # greedy: faster, and the gain from beam search is
                              # inaudible on short utterances
        )
        segments = list(segments)  # force the generator HERE, inside the guard
    except VoiceError:
        raise
    except Exception as exc:
        # PyAV raises InvalidDataError, not anything of ours, so this used to
        # escape as a bare 500 and reach the browser as "Internal Server Error".
        # Every undecodable recording looked like the server falling over.
        raise VoiceError(f"could not read the recording ({type(exc).__name__})") from exc

    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    detected = language or info.language

    if detected not in VOICES:
        # Whisper can return languages we have no voice for. Answering in a
        # language we cannot speak is worse than answering in Hindi.
        detected = DEFAULT_LANGUAGE

    return transcript, detected


# --------------------------------------------------------------------------
# TTS
# --------------------------------------------------------------------------

def _cache_path(text: str, voice: str) -> Path:
    digest = hashlib.sha256(f"{voice}|{text}".encode()).hexdigest()[:20]
    return CACHE_DIR / f"{digest}.mp3"


async def _synthesise(text: str, voice: str, destination: Path) -> None:
    try:
        import edge_tts
    except ImportError as exc:  # pragma: no cover - depends on install
        raise VoiceError("edge-tts is not installed") from exc

    await edge_tts.Communicate(text, voice).save(str(destination))


def _run_async(coro):
    """
    Run a coroutine whether or not a loop is already running.

    FastAPI handlers execute inside a running loop, so asyncio.run() raises
    there. Falling back to a dedicated thread with its own loop keeps one
    calling convention for both the API and the CLI, instead of forcing every
    caller to know which world it is in.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def speak(text: str, language: str = DEFAULT_LANGUAGE) -> Path:
    """
    Text in, path to an mp3 out.

    Cached on (voice, text), so a replayed demo never touches the network and
    the same sentence is byte-identical every time.
    """
    text = text.strip()
    if not text:
        raise VoiceError("nothing to speak")

    voice = VOICES.get(language, VOICES[DEFAULT_LANGUAGE])
    destination = _cache_path(text, voice)

    if destination.exists():
        return destination

    if VOICE_MODE == "offline":
        raise VoiceOffline(f"no cached audio for this sentence in {language}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _run_async(_synthesise(text, voice, destination))

    if not destination.exists() or destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise VoiceError("edge-tts produced no audio")

    return destination


def prewarm(phrases: dict[str, list[str]]) -> list[Path]:
    """
    Synthesise a set of sentences ahead of time, into the cache.

    Run this before a demo over whatever wifi you trust. Every prompt, error
    message, and IVR menu line should already be on disk by the time you are
    standing in front of judges.
    """
    warmed = []
    for language, lines in phrases.items():
        for line in lines:
            warmed.append(speak(line, language))
    return warmed
