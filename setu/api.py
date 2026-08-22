"""
The HTTP layer.

Thin on purpose. Every endpoint is the same pipeline with a different way in:

    audio or text -> transcript -> profile -> decisions -> ladder -> narration

Nothing here decides anything. If you find business logic in this file, it is
in the wrong place.
"""

from __future__ import annotations

import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import llm, voice
from .ladder import best_paths, group_by_first_step
from .rules import Profile, Status, evaluate_all, load_schemes, missing_fields

@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Pay the Whisper load during boot, not during the demo.

    Loading lazily costs the FIRST request about 25 seconds, and on stage the
    first request IS the demo.

    Best effort on purpose: a machine without the model downloaded, or without
    faster-whisper installed at all, should still serve /ask/text and the
    console rather than refusing to start.
    """
    try:
        voice.preload()
        print(f"[setu] whisper '{voice.WHISPER_SIZE}' ready")
    except Exception as exc:  # noqa: BLE001 - never block startup on this
        print(f"[setu] whisper not preloaded ({type(exc).__name__}: {exc})")
        print("[setu] /ask will be slow on first use; /ask/text is unaffected")
    yield


app = FastAPI(title="Setu", version="1.2.0", lifespan=lifespan)

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

# Which fields we have already put to each caller. Tracked so we do not read the
# same sentence out three times when someone answers a different question than
# the one asked -- which is the normal thing for a person to do, not an edge case.
ASKED: dict[str, list[str]] = {}


def _citation(result) -> dict[str, Any]:
    return {
        "rule": result.description,
        "doc": result.citation["doc"],
        "url": result.citation["url"],
        "quote": result.citation["quote"],
    }


def _localise(payload: dict[str, Any], language: str) -> None:
    """
    Translate the reading text on the cards into the caller's language.

    Collected and sent as ONE call for the whole response rather than one per
    string -- and cached, so the catalogue costs a single call per language for
    the entire demo.

    `quote` is replaced with a translation but `quote_en` keeps the government's
    exact words. The English is what makes the "Why?" panel an audit trail: a
    translated clause is easier to read but it is no longer the thing the
    government actually wrote, and a citation you have quietly paraphrased is
    not a citation. Both ship; the page shows the readable one and keeps the
    original underneath.
    """
    if language == "en":
        return

    slots: list[tuple[dict[str, Any], str]] = []
    for card in payload["schemes"]:
        slots.append((card, "benefit"))
        for group in ("passed", "failed", "unknown"):
            for citation in card[group]:
                citation["quote_en"] = citation["quote"]
                slots.append((citation, "rule"))
                slots.append((citation, "quote"))

    for step in payload.get("next_steps", []):
        slots.append((step, "action"))
        slots.append((step, "where"))

    translated = llm.translate([holder[key] for holder, key in slots], language)
    for (holder, key), value in zip(slots, translated):
        holder[key] = value


def _unique(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    One clause, one entry.

    Age bands are written as two rules -- a minimum and a maximum -- that quote
    the SAME government sentence, so the panel showed it twice in a row and
    looked like a rendering bug. It also doubled the translation payload for
    nothing.
    """
    seen: set[tuple[str, str]] = set()
    out = []
    for citation in citations:
        key = (citation["doc"], citation["quote"])
        if key not in seen:
            seen.add(key)
            out.append(citation)
    return out


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
        "passed": _unique([_citation(r) for r in decision.results if r.passed is True]),
        "failed": _unique([_citation(r) for r in decision.failed]),
        "unknown": _unique([_citation(r) for r in decision.unknown]),
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


def _last_question(session_id: str) -> str | None:
    """What we asked this caller last turn, so a bare "no" can be attributed."""
    asked = ASKED.get(session_id)
    return asked[-1] if asked else None


def _learned(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Which facts this turn actually added."""
    return [
        k
        for k, v in after.items()
        if k != "language" and v not in (None, []) and before.get(k) != v
    ]


def _respond(
    profile: Profile,
    session_id: str,
    transcript: str,
    just_learned: list[str] | None = None,
) -> dict[str, Any]:
    """The pipeline, from a filled profile to everything the UI renders."""
    decisions = evaluate_all(profile)
    paths = best_paths(profile, decisions)
    unknown = missing_fields(profile)

    asked = ASKED.setdefault(session_id, [])
    asking = llm.choose_question(unknown, asked)

    try:
        spoken = llm.narrate(
            decisions, paths, profile.language, unknown,
            just_learned=just_learned, asked_before=asked, profile=profile,
        )
    except llm.LLMError as exc:
        raise HTTPException(502, f"narration failed: {exc}") from exc

    audio_url = None
    try:
        audio_url = f"/audio/{voice.speak(spoken, profile.language).name}"
    except voice.VoiceError:
        # A missing voice file must not cost the caller their answer. The text
        # is already correct; the PWA falls back to showing it.
        pass

    if asking:
        # Append every turn, repeats included. Deduplicating broke two things at
        # once: _last_question() returned a stale field, so a bare "no" got
        # attributed to whatever was asked three turns ago, and the "was this
        # asked recently" check could not see a repeat at all.
        asked.append(asking)

    payload = {
        "session_id": session_id,
        "transcript": transcript,
        "asking_about": asking,
        "turn": len(asked),
        "language": profile.language,
        "profile": asdict(profile),
        "spoken": spoken,
        "audio_url": audio_url,
        "next_questions": unknown[:2],
        "eligible_count": sum(1 for d in decisions if d.status is Status.ELIGIBLE),
        "schemes": [_scheme_card(d) for d in decisions],
        "ladder": [_ladder_card(p) for p in paths],
        "next_steps": [
            {
                "action": g.action, "cost": g.cost_rupees, "days": g.days,
                "where": g.where, "why": g.citation,
                "schemes": g.schemes, "unlocks": g.unlocks_rupees,
                "headline": g.headline(),
            }
            for g in group_by_first_step(paths)
        ],
    }

    _localise(payload, profile.language)
    return payload


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
    before = asdict(known) if known else {}

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
        # 422, not 503: the recording is the problem, not the service, and the
        # message is written to be shown to the person holding the phone.
        raise HTTPException(422, str(exc)) from exc
    finally:
        clip.unlink(missing_ok=True)

    if not transcript:
        raise HTTPException(422, "no speech detected in the recording")

    try:
        profile = llm.extract_profile(
            transcript, detected, base=known, answering=_last_question(session_id)
        )
    except llm.LLMError as exc:
        raise HTTPException(502, f"extraction failed: {exc}") from exc

    SESSIONS[session_id] = profile
    return _respond(profile, session_id, transcript, _learned(before, asdict(profile)))


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
    before = asdict(known) if known else {}

    try:
        profile = llm.extract_profile(
            text, language, base=known, answering=_last_question(session_id)
        )
    except llm.LLMError as exc:
        raise HTTPException(502, f"extraction failed: {exc}") from exc

    SESSIONS[session_id] = profile
    return _respond(profile, session_id, text, _learned(before, asdict(profile)))


@app.get("/sessions")
def sessions() -> dict[str, Any]:
    """
    Every live caller, as an operator sees them.

    This is the view a Bank Mitra or CSC operator works from: who is mid
    conversation, what Setu has established so far, what it still needs to ask,
    and what the person is owed right now.
    """
    rows = []
    for sid, profile in SESSIONS.items():
        decisions = evaluate_all(profile)
        paths = best_paths(profile, decisions)
        eligible = [d for d in decisions if d.status is Status.ELIGIBLE]

        rows.append({
            "session_id": sid,
            "language": profile.language,
            "profile": asdict(profile),
            "known_facts": sum(
                1 for k, v in asdict(profile).items()
                if k != "language" and v not in (None, [])
            ),
            "eligible": [{"name": d.scheme_name, "amount": d.benefit_amount_rupees}
                         for d in eligible],
            "entitled_now": sum(d.benefit_amount_rupees for d in eligible),
            "unlockable": sum(p.unlocks_rupees for p in paths),
            "next_step": paths[0].headline() if paths else None,
            "next_questions": missing_fields(profile)[:2],
        })

    rows.sort(key=lambda r: -r["entitled_now"])
    return {
        "sessions": rows,
        "total_entitled": sum(r["entitled_now"] for r in rows),
        "total_unlockable": sum(r["unlockable"] for r in rows),
    }


@app.get("/eval")
def evaluation() -> dict[str, Any]:
    """
    The eval numbers, live from the same harness the test suite runs.

    Served rather than pasted so the dashboard can never show a stale figure --
    a precision number that was true last Tuesday is worse than none, because
    it is the one thing a judge is entitled to take at face value.
    """
    import io
    import sys as _sys
    from contextlib import redirect_stdout

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
    import run as harness  # noqa: PLC0415

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = harness.main()

    return {"passing": code == 0, "report": buffer.getvalue()}


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
    ASKED.pop(session_id, None)
    return {"cleared": SESSIONS.pop(session_id, None) is not None}


# Mounted last so it never shadows an API route. Serving the PWA from the same
# origin as the API keeps CORS out of the demo path entirely -- one tunnel URL,
# not two, and one fewer thing to reconfigure when the tunnel rotates.
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if WEB_DIR.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
