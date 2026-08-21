"""
Tests for the HTTP layer and the voice adapter.

No network, no Whisper model, no Gemini key. Everything external is stubbed --
what is being tested is the wiring, and specifically that a failure in one
external service degrades instead of taking the caller's answer with it.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from setu import api, llm, voice
from setu.api import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(llm, "narrate", lambda *a, **k: "aap e-Shram ke liye patra hain.")
    monkeypatch.setattr(voice, "speak", lambda text, lang="hi": voice.CACHE_DIR / "stub.mp3")
    api.SESSIONS.clear()
    return TestClient(app)


def _extract_returns(monkeypatch, **fields):
    from setu.rules import Profile

    def fake(transcript, language="hi", *, base=None, answering=None):
        p = base or Profile()
        p.language = language
        for k, v in fields.items():
            setattr(p, k, v)
        return p

    monkeypatch.setattr(llm, "extract_profile", fake)


# --------------------------------------------------------------------------
# Basics
# --------------------------------------------------------------------------

def test_health_reports_catalogue_and_languages(client):
    body = client.get("/health").json()
    assert body["ok"] is True
    assert body["schemes"] == 8
    assert "mr" in body["languages"] and "hi" in body["languages"]


def test_schemes_endpoint_exposes_a_source_for_every_scheme(client):
    for scheme in client.get("/schemes").json()["schemes"]:
        assert scheme["source"].startswith("https://")
        assert scheme["rule_count"] > 0


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------

def test_text_ask_returns_cards_and_a_ladder(client, monkeypatch):
    _extract_returns(
        monkeypatch,
        age=30,
        occupation_category="street_vendor",
        daily_income=500,
        documents=["aadhaar"],
        documents_denied=["bank_account", "jan_dhan_account"],
        vending_since_year=2019,
        is_epfo_esic_member=False,
        is_income_tax_payer=False,
        has_loan_npa=False,
    )

    body = client.post("/ask/text", data={"text": "main sabzi bechta hoon", "language": "mr"}).json()

    assert body["language"] == "mr"
    assert len(body["schemes"]) == 8
    assert body["ladder"], "a vendor with only Aadhaar should have somewhere to climb"
    assert body["ladder"][0]["steps"][0]["why"]["url"].startswith("https://")


def test_every_card_carries_citations_for_what_failed(client, monkeypatch):
    _extract_returns(monkeypatch, age=65, documents=["aadhaar", "bank_account"])
    body = client.post("/ask/text", data={"text": "main 65 saal ka hoon"}).json()

    rejected = [c for c in body["schemes"] if c["status"] == "NOT_ELIGIBLE"]
    assert rejected, "a 65-year-old should fail at least one age band"
    for card in rejected:
        assert card["failed"], f"{card['id']} rejected without saying why"
        assert all(f["quote"] for f in card["failed"])


def test_session_carries_facts_across_turns(client, monkeypatch):
    _extract_returns(monkeypatch, age=30)
    first = client.post("/ask/text", data={"text": "main tees saal ka hoon"}).json()
    session = first["session_id"]

    _extract_returns(monkeypatch, daily_income=500)
    second = client.post(
        "/ask/text", data={"text": "roz paanch sau", "session_id": session}
    ).json()

    assert second["profile"]["age"] == 30, "the second turn forgot the first"
    assert second["profile"]["daily_income"] == 500


def test_reset_clears_the_session(client, monkeypatch):
    _extract_returns(monkeypatch, age=30)
    session = client.post("/ask/text", data={"text": "hi"}).json()["session_id"]

    assert client.delete(f"/session/{session}").json()["cleared"] is True
    assert client.delete(f"/session/{session}").json()["cleared"] is False


# --------------------------------------------------------------------------
# Degrading instead of failing
# --------------------------------------------------------------------------

def test_missing_audio_does_not_cost_the_caller_their_answer(client, monkeypatch):
    """
    TTS is the most fragile link in the chain. If it breaks, the text answer is
    still correct and the PWA can show it -- losing the whole response because
    an mp3 failed would be the wrong trade.
    """
    def boom(text, lang="hi"):
        raise voice.VoiceError("edge-tts unreachable")

    monkeypatch.setattr(voice, "speak", boom)
    _extract_returns(monkeypatch, age=30, documents=["aadhaar"])

    body = client.post("/ask/text", data={"text": "kuch bhi"}).json()
    assert body["audio_url"] is None
    assert body["spoken"], "the answer text should survive a TTS failure"


def test_extraction_failure_surfaces_as_502_not_a_wrong_verdict(client, monkeypatch):
    def boom(*a, **k):
        raise llm.LLMError("quota exhausted")

    monkeypatch.setattr(llm, "extract_profile", boom)
    assert client.post("/ask/text", data={"text": "kuch bhi"}).status_code == 502


def test_silent_recording_is_rejected_before_the_llm_is_billed(client, monkeypatch):
    monkeypatch.setattr(voice, "transcribe", lambda path, lang=None: ("", "hi"))

    def should_not_run(*a, **k):
        raise AssertionError("extraction ran on an empty transcript")

    monkeypatch.setattr(llm, "extract_profile", should_not_run)

    response = client.post("/ask", files={"audio": ("clip.webm", b"\x00\x00", "audio/webm")})
    assert response.status_code == 422


# --------------------------------------------------------------------------
# Audio serving
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["../../etc/passwd", "a/b.mp3", "notes.txt"])
def test_audio_endpoint_rejects_path_traversal(client, name):
    assert client.get(f"/audio/{name}").status_code in (400, 404)


# --------------------------------------------------------------------------
# Static mount
# --------------------------------------------------------------------------

def test_pwa_is_served_from_the_same_origin_as_the_api(client):
    """One tunnel URL, not two, and no CORS in the demo path."""
    assert client.get("/").status_code == 200
    assert client.get("/manifest.webmanifest").status_code == 200


def test_static_mount_does_not_shadow_api_routes(client):
    """
    The catch-all mount is registered last for exactly this reason. If it ever
    moves up the file, every API route starts returning the PWA shell and the
    failure looks like a frontend bug for an hour.
    """
    assert client.get("/health").json()["ok"] is True
    assert client.get("/schemes").json()["schemes"]


# --------------------------------------------------------------------------
# Voice adapter
# --------------------------------------------------------------------------

def test_every_supported_language_has_a_voice():
    for code, name in voice.VOICES.items():
        assert name.startswith(f"{code}-"), f"{code} mapped to an unrelated voice"


def test_unknown_detected_language_falls_back_to_hindi(monkeypatch, tmp_path):
    """
    Whisper returns languages we have no voice for. Answering in a language we
    cannot speak is worse than answering in Hindi.
    """
    class FakeInfo:
        language = "sw"

    monkeypatch.setattr(voice, "_whisper", lambda: type(
        "M", (), {"transcribe": lambda self, *a, **k: ([], FakeInfo())}
    )())

    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"\x00" * (voice.MIN_CLIP_BYTES + 1))

    _, detected = voice.transcribe(clip)
    assert detected == "hi"


def test_a_tap_instead_of_a_sentence_is_a_clear_error(tmp_path):
    """
    Regression: a too-short recording raised PyAV's InvalidDataError deep in the
    decoder. It is not a VoiceError, so it escaped as a bare 500 and reached the
    phone as "Internal Server Error" -- every short tap looked like the server
    falling over.
    """
    clip = tmp_path / "tap.webm"
    clip.write_bytes(b"\x1aE\xdf\xa3" + b"\x00" * 100)

    with pytest.raises(voice.VoiceError, match="too short"):
        voice.transcribe(clip)


def test_undecodable_audio_becomes_a_voice_error(tmp_path, monkeypatch):
    """Anything PyAV cannot parse must surface as ours, not as a 500."""
    clip = tmp_path / "junk.webm"
    clip.write_bytes(b"not a container at all" * 200)

    class Boom:
        def transcribe(self, *a, **k):
            raise ValueError("Invalid data found when processing input")

    monkeypatch.setattr(voice, "_whisper", lambda: Boom())
    with pytest.raises(voice.VoiceError, match="could not read the recording"):
        voice.transcribe(clip)


def test_a_bad_recording_returns_422_not_500(client, monkeypatch):
    def boom(path, lang=None):
        raise voice.VoiceError("recording too short — hold the button and speak")

    monkeypatch.setattr(voice, "transcribe", boom)
    r = client.post("/ask", files={"audio": ("clip.webm", b"\x00" * 64, "audio/webm")})

    assert r.status_code == 422
    assert "hold the button" in r.json()["detail"]


def test_speak_refuses_empty_text():
    with pytest.raises(voice.VoiceError):
        voice.speak("   ")


def test_offline_mode_refuses_uncached_audio(monkeypatch, tmp_path):
    monkeypatch.setattr(voice, "VOICE_MODE", "offline")
    monkeypatch.setattr(voice, "CACHE_DIR", tmp_path)
    with pytest.raises(voice.VoiceOffline):
        voice.speak("a sentence never synthesised before", "hi")


def test_cached_audio_is_returned_without_synthesis(monkeypatch, tmp_path):
    monkeypatch.setattr(voice, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(voice, "VOICE_MODE", "offline")

    path = voice._cache_path("namaste", voice.VOICES["hi"])
    path.write_bytes(b"fake mp3")

    assert voice.speak("namaste", "hi") == path


# --------------------------------------------------------------------------
# .env loading
# --------------------------------------------------------------------------

def test_env_file_is_read(tmp_path, monkeypatch):
    """
    Regression: .env.example told you to put the key here, but nothing read the
    file. A correctly-filled .env produced "GEMINI_API_KEY is not set", which
    reads as an auth problem rather than a missing call.
    """
    import setu

    env = tmp_path / ".env"
    env.write_text('# a comment\nGEMINI_API_KEY="abc123"\n\nSETU_LLM_MODEL=flash-lite\n')
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("SETU_LLM_MODEL", raising=False)

    setu._load_env(env)
    assert os.environ["GEMINI_API_KEY"] == "abc123"
    assert os.environ["SETU_LLM_MODEL"] == "flash-lite"


def test_real_environment_beats_the_file(tmp_path, monkeypatch):
    """So `SETU_LLM_MODE=offline uvicorn ...` works without editing .env."""
    import setu

    env = tmp_path / ".env"
    env.write_text("SETU_LLM_MODE=auto\n")
    monkeypatch.setenv("SETU_LLM_MODE", "offline")

    setu._load_env(env)
    assert os.environ["SETU_LLM_MODE"] == "offline"


# --------------------------------------------------------------------------
# Operator console endpoints
# --------------------------------------------------------------------------

def test_sessions_summarises_each_caller(client, monkeypatch):
    _extract_returns(
        monkeypatch,
        age=30,
        occupation_category="street_vendor",
        daily_income=500,
        documents=["aadhaar"],
        documents_denied=["bank_account", "jan_dhan_account"],
        vending_since_year=2019,
        is_epfo_esic_member=False,
        is_income_tax_payer=False,
        has_loan_npa=False,
    )
    client.post("/ask/text", data={"text": "main sabzi bechta hoon"})

    body = client.get("/sessions").json()
    assert len(body["sessions"]) == 1

    row = body["sessions"][0]
    assert row["eligible"], "an e-Shram-eligible vendor should show as entitled to something"
    assert row["entitled_now"] > 0
    assert row["next_step"], "a vendor with no bank account has somewhere to climb"
    assert body["total_unlockable"] > 0


def test_sessions_ranks_by_what_the_caller_is_owed(client, monkeypatch):
    _extract_returns(monkeypatch, age=30, documents=["aadhaar", "bank_account"])
    client.post("/ask/text", data={"text": "one"})
    _extract_returns(monkeypatch, occupation_category="potter")
    client.post("/ask/text", data={"text": "two"})

    amounts = [r["entitled_now"] for r in client.get("/sessions").json()["sessions"]]
    assert amounts == sorted(amounts, reverse=True)


def test_eval_endpoint_serves_live_numbers(client):
    """
    Served rather than pasted: a precision figure that was true last Tuesday is
    worse than none, because it is the one number a judge takes at face value.
    """
    body = client.get("/eval").json()
    assert body["passing"] is True
    assert "MICRO" in body["report"]
    assert "false rejections on missing facts: 0/" in body["report"]


def test_console_is_served(client):
    assert client.get("/console.html").status_code == 200
