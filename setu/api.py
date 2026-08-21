"""
The HTTP layer.

Thin on purpose. Every endpoint is the same pipeline with a different way in:

    audio or text -> transcript -> profile -> decisions -> ladder -> narration

Nothing here decides anything. If you find business logic in this file, it is
in the wrong place.
"""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import llm, voice
from .ladder import best_paths
from .rules import Profile, Status, evaluate_all, load_schemes, missing_fields

app = FastAPI(title="Setu", version="1.1.0")

# The PWA is served from a tunnel on a different origin than the API during
# development, and getUserMedia refuses plain http, so the browser origin is
# never predictable. Open for the hackathon; tighten before anything real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory, one entry per caller. A conversation is a handful of turns and the
# process outlives it; persistence would be scope we do not need yet.
SESSIONS: dict[str, Profile] = {}


def _citation(result) -> dict[str, Any]:
    return {
        "rule": result.description,
        "doc": result.citation["doc"],
        "url": result.citation["url"],
        "quote": result.citation["quote"],
    }


def _scheme_card(decision) -> dict[str, Any]:
    """
    One scheme, as the UI needs it.

    Failing and unknown rules both carry their citation, because "why not" is
    asked at least as often as "why", and the honest answer to both is the same
    government sentence.
    """
    return {
        "id": decision.scheme_id,
        "name": decision.scheme_name,
        "status": decision.status.value,
        "benefit": decision.benefit_summary,
        "amount": decision.benefit_amount_rupees,
        "passed": [_citation(r) for r in decision.results if r.passed is True],
        "failed": [_citation(r) for r in decision.failed],
        "unknown": [_citation(r) for r in decision.unknown],
    }


def _ladder_card(path) -> dict[str, Any]:
    return {
        "scheme_id": path.scheme_id,
        "scheme": path.scheme_name,
        "headline": path.headline(),
        "unlocks": path.unlocks_rupees,
        "total_cost": path.total_cost,
        "total_days": path.total_days,
        "steps": [
            {
                "action": rung.action,
                "cost": rung.cost_rupees,
                "days": rung.days,
                "where": rung.where,
                "why": rung.citation,
            }
            for rung in path.rungs
        ],
    }


def _respond(profile: Profile, session_id: str, transcript: str) -> dict[str, Any]:
    """The pipeline, from a filled profile to everything the UI renders."""
    decisions = evaluate_all(profile)
    paths = best_paths(profile, decisions)
    unknown = missing_fields(profile)

    try:
        spoken = llm.narrate(decisions, paths, profile.language, unknown)
    except llm.LLMError as exc:
        raise HTTPException(502, f"narration failed: {exc}") from exc

    audio_url = None
    try:
        audio_url = f"/audio/{voice.speak(spoken, profile.language).name}"
    except voice.VoiceError:
        # A missing voice file must not cost the caller their answer. The text
        # is already correct; the PWA falls back to showing it.
        pass

    return {
        "session_id": session_id,
        "transcript": transcript,
        "language": profile.language,
        "profile": asdict(profile),
        "spoken": spoken,
        "audio_url": audio_url,
        "next_questions": unknown[:2],
        "eligible_count": sum(1 for d in decisions if d.status is Status.ELIGIBLE),
        "schemes": [_scheme_card(d) for d in decisions],
        "ladder": [_ladder_card(p) for p in paths],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "schemes": len(load_schemes()["schemes"]),
        "languages": sorted(voice.VOICES),
    }


@app.get("/schemes")
def schemes() -> dict[str, Any]:
    """The whole catalogue, for the 'what does Setu know about' panel."""
    return {
        "schemes": [
            {
                "id": s["id"],
                "name": s["name"],
                "full_name": s["full_name"],
                "category": s["category"],
                "benefit": s["benefit_summary"],
                "amount": s["benefit_amount_rupees"],
                "rule_count": len(s["rules"]),
                "source": s["rules"][0]["source_url"],
            }
            for s in load_schemes()["schemes"]
        ]
    }


@app.post("/ask")
async def ask(
    audio: UploadFile = File(...),
    session_id: str | None = Form(None),
    language: str | None = Form(None),
) -> dict[str, Any]:
    """Spoken question in, spoken answer out."""
    session_id = session_id or str(uuid.uuid4())
    known = SESSIONS.get(session_id)

    suffix = Path(audio.filename or "clip.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        clip = Path(tmp.name)

    try:
        # Once we know what someone speaks, stop re-detecting. Detection on a
        # two-word reply is a coin flip that can flip the whole session into
        # the wrong language mid-conversation.
        hint = language or (known.language if known else None)
        transcript, detected = voice.transcribe(clip, hint)
    except voice.VoiceError as exc:
        raise HTTPException(503, f"transcription failed: {exc}") from exc
    finally:
        clip.unlink(missing_ok=True)

    if not transcript:
        raise HTTPException(422, "no speech detected in the recording")

    try:
        profile = llm.extract_profile(transcript, detected, base=known)
    except llm.LLMError as exc:
        raise HTTPException(502, f"extraction failed: {exc}") from exc

    SESSIONS[session_id] = profile
    return _respond(profile, session_id, transcript)


@app.post("/ask/text")
def ask_text(
    text: str = Form(...),
    session_id: str | None = Form(None),
    language: str = Form("hi"),
) -> dict[str, Any]:
    """
    Same pipeline, typed instead of spoken.

    Exists so the whole system can be exercised without a microphone, an HTTPS
    tunnel, or a Whisper model loaded -- which is most of what you want when
    something breaks twenty minutes before a demo.
    """
    session_id = session_id or str(uuid.uuid4())
    known = SESSIONS.get(session_id)

    try:
        profile = llm.extract_profile(text, language, base=known)
    except llm.LLMError as exc:
        raise HTTPException(502, f"extraction failed: {exc}") from exc

    SESSIONS[session_id] = profile
    return _respond(profile, session_id, text)


@app.get("/audio/{filename}")
def audio(filename: str) -> FileResponse:
    # Filenames are content hashes we generated; anything else is someone
    # walking the filesystem.
    if not filename.endswith(".mp3") or "/" in filename or ".." in filename:
        raise HTTPException(400, "bad filename")

    path = voice.CACHE_DIR / filename
    if not path.exists():
        raise HTTPException(404, "no such audio")

    return FileResponse(path, media_type="audio/mpeg")


@app.delete("/session/{session_id}")
def reset(session_id: str) -> dict[str, bool]:
    """Between demo runs, so the next caller starts clean."""
    return {"cleared": SESSIONS.pop(session_id, None) is not None}


# Mounted last so it never shadows an API route. Serving the PWA from the same
# origin as the API keeps CORS out of the demo path entirely -- one tunnel URL,
# not two, and one fewer thing to reconfigure when the tunnel rotates.
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if WEB_DIR.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
