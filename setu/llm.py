"""
The LLM adapter.

Two jobs, both narrow:

    extract_profile()  speech transcript -> structured Profile
    narrate()          rule-engine verdict -> plain spoken sentence

It never decides eligibility. That is rules.py's job, and keeping the boundary
absolute is what makes every verdict citable and replayable. If you ever find
yourself asking the model "is she eligible", stop -- the answer belongs in
schemes.json.

Provider sits behind _generate() so swapping Gemini for anything else is one
function, not a rewrite.

Currently gemini-3.5-flash-lite on the v1 endpoint. Chosen by measurement, not
by reputation -- every plausible default here was wrong:

    2.5-flash        404, no longer served to new keys
    3.7-flash        503, high demand
    3.6-flash        works, 10-21s
    3.5-flash        works, 2.7-5.7s
    3.5-flash-lite   works, 1.3s        <- and identical extraction quality

All three working models got every probe right, including the two that matter:
a daily wage stated as "roz paanch sau" landing in daily_income rather than
monthly, and a Marathi sentence resolving to the right trade. When accuracy
ties, latency decides, and a demo that promises fifteen seconds cannot spend
ten of them waiting.

Re-run eval/pick_model.py when the demo feels slow or a model starts 503ing --
availability here shifted twice in one afternoon.

Everything is cached on content hash. The demo has to survive a bad conference
wifi connection, and re-calling the model for a sentence we already have is a
needless risk on stage. Set SETU_LLM_MODE=offline to refuse network calls
entirely and serve only from cache -- run the demo that way.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import httpx

from .ladder import Path as LadderPath
from .ladder import group_by_first_step
from .rules import Decision, Profile, Status

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "llm_cache"
API_BASE = "https://generativelanguage.googleapis.com/v1/models"

MODEL = os.getenv("SETU_LLM_MODEL", "gemini-3.5-flash-lite")
LLM_MODE = os.getenv("SETU_LLM_MODE", "auto")  # auto | offline
TIMEOUT_SECONDS = float(os.getenv("SETU_LLM_TIMEOUT", "20"))

LANGUAGE_NAMES = {
    "hi": "Hindi",
    "mr": "Marathi",
    "en": "Indian English",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "gu": "Gujarati",
}

# Must match the values the rules in schemes.json actually test against.
OCCUPATIONS = [
    "street_vendor", "artisan", "tailor", "carpenter", "blacksmith", "potter",
    "cobbler", "goldsmith", "barber", "washerman", "mason", "farmer",
    "domestic_worker", "construction", "driver", "other",
]

DOCUMENTS = [
    "aadhaar", "bank_account", "jan_dhan_account", "upi_id", "pan",
    "voter_id", "ration_card", "vending_certificate", "letter_of_recommendation",
]


class LLMError(RuntimeError):
    pass


class LLMOffline(LLMError):
    """Asked for something not in cache while offline mode is on."""


# --------------------------------------------------------------------------
# Provider adapter -- the only place that knows it is Gemini
# --------------------------------------------------------------------------

def _cache_key(kind: str, payload: dict[str, Any]) -> str:
    blob = json.dumps({"kind": kind, "model": MODEL, **payload}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def _cached(key: str) -> str | None:
    path = CACHE_DIR / f"{key}.txt"
    return path.read_text(encoding="utf-8") if path.exists() else None


def _store(key: str, value: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.txt").write_text(value, encoding="utf-8")


def _generate(
    prompt: str,
    *,
    kind: str,
    schema: dict[str, Any] | None = None,
    temperature: float = 0.0,
) -> str:
    """
    One call to the model. Cached on content, so identical input never costs
    twice and the same demo replays identically.

    temperature is 0 by default: extraction must be deterministic, and a
    creative rephrasing of someone's income is a bug, not a flourish.
    """
    key = _cache_key(kind, {"prompt": prompt, "schema": schema, "t": temperature})

    hit = _cached(key)
    if hit is not None:
        return hit

    if LLM_MODE == "offline":
        raise LLMOffline(f"no cached {kind} response and offline mode is on")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is not set (put it in .env)")

    config: dict[str, Any] = {"temperature": temperature}
    if schema:
        config["responseMimeType"] = "application/json"
        config["responseSchema"] = schema

    try:
        response = httpx.post(
            f"{API_BASE}/{MODEL}:generateContent",
            headers={"x-goog-api-key": api_key},
            json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": config},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except httpx.HTTPError as exc:
        raise LLMError(f"Gemini call failed: {exc}") from exc
    except (KeyError, IndexError) as exc:
        raise LLMError(f"unexpected Gemini response shape: {exc}") from exc

    _store(key, text)
    return text


# --------------------------------------------------------------------------
# Speech -> Profile
# --------------------------------------------------------------------------

PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "age": {"type": "integer", "nullable": True},
        "gender": {"type": "string", "nullable": True},
        "occupation_category": {"type": "string", "enum": OCCUPATIONS, "nullable": True},
        "monthly_income": {"type": "number", "nullable": True},
        "daily_income": {"type": "number", "nullable": True},
        "state": {"type": "string", "nullable": True},
        "documents": {"type": "array", "items": {"type": "string", "enum": DOCUMENTS}},
        "documents_denied": {"type": "array", "items": {"type": "string", "enum": DOCUMENTS}},
        "is_epfo_esic_member": {"type": "boolean", "nullable": True},
        "is_income_tax_payer": {"type": "boolean", "nullable": True},
        "has_loan_npa": {"type": "boolean", "nullable": True},
        "vending_since_year": {"type": "integer", "nullable": True},
        "took_govt_credit_scheme_last_5y": {"type": "boolean", "nullable": True},
    },
    "required": ["documents"],
}

EXTRACT_PROMPT = """You are transcribing facts, not judging them.

Read what this person said and fill in only the fields they actually stated or
clearly implied. This is the single most important rule: if they did not say
something, leave it null. Do not guess, do not infer from stereotypes about
their occupation, and do not fill a plausible default.

A wrong guess here becomes a wrong eligibility verdict downstream, and the
person never finds out why. Null is always the safe answer.

Specific guidance:
- Income: most people quote a DAILY figure ("500 rupees a day"). Put that in
  daily_income, not monthly_income. Only use monthly_income if they clearly
  said a monthly amount.
- Documents they said they HAVE go in `documents`. Documents they said they do
  NOT have go in `documents_denied`. Both matter: "I have no bank account" is a
  fact, and recording it is what lets Setu tell her how to open one. Leaving it
  out entirely means Setu can only say "I don't know" forever.
  A document in neither list means it never came up.
- has_loan_npa is true only if they describe a loan gone bad, written off, or
  defaulted. Merely having a loan is NOT an NPA.
- vending_since_year: the year they started vending, if stated.

Two fields need care, because nobody in this population will ever say the words
"EPFO", "ESIC" or "income tax payer" — and if you wait for those exact words,
the answer stays unknown forever and they never learn what they are owed. These
are not guesses; each membership is *defined* by the thing being described:

- is_epfo_esic_member: EPFO and ESIC come with FORMAL SALARIED EMPLOYMENT. So
  "I work for myself", "it's my own cart/shop", "I have no employer", "no
  company job", "I do daily wage work" all establish this as FALSE. Set it TRUE
  only if they describe a salaried job with a company, a PF deduction, or an
  ESIC card. If they never touch on how they work, leave it null.

- is_income_tax_payer: set FALSE if they say they do not file or pay income
  tax, or that they have no PAN for tax purposes. Do NOT infer it from a low
  income figure alone — earning little is not the same as saying you don't file,
  and that one really would be a guess.

They were speaking {language}.

What they said:
\"\"\"{transcript}\"\"\"
"""


def extract_profile(transcript: str, language: str = "hi", *, base: Profile | None = None) -> Profile:
    """
    Build a Profile from what someone said.

    `base` carries forward what we already learned earlier in the conversation.
    New facts overwrite old ones; a null never erases a fact we already have,
    because "she didn't mention it this turn" is not "she retracted it".
    """
    raw = _generate(
        EXTRACT_PROMPT.format(
            language=LANGUAGE_NAMES.get(language, "Hindi"),
            transcript=transcript.strip(),
        ),
        kind="extract",
        schema=PROFILE_SCHEMA,
    )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"model returned non-JSON despite schema: {exc}") from exc

    profile = base or Profile()
    profile.language = language

    for name, value in data.items():
        if value is None or not hasattr(profile, name):
            continue
        if name in ("documents", "documents_denied"):
            merged = list(dict.fromkeys([*getattr(profile, name), *value]))
            setattr(profile, name, merged)
        else:
            setattr(profile, name, value)

    # Acquiring a document retracts the earlier denial. Without this, "I have no
    # bank account" on turn one outlives "I opened one" on turn three, and Setu
    # keeps routing her to a branch she has already visited.
    profile.documents_denied = [
        d for d in profile.documents_denied if d not in profile.documents
    ]

    return profile


# --------------------------------------------------------------------------
# Verdict -> spoken sentence
# --------------------------------------------------------------------------

NARRATE_PROMPT = """You are telling someone the result of an eligibility check
that has already been decided. You are a translator, not a decision-maker.

Hard constraints:
- Do NOT add any scheme that is not in the list below.
- Do NOT remove or hedge any scheme that is in the list.
- Do NOT restate or reinterpret the eligibility rules.
- Do NOT invent amounts, deadlines, offices, or timelines.

The decision was made by a rule engine reading government documents. Your only
job is to say it in a way a person with no formal education can act on.

Style:
- Speak {language}, plainly, the way a helpful neighbour would.
- Short sentences. This will be read aloud, and possibly over a phone line.
- Lead with the good news: what they can get right now.
- Then the single most useful next step, if there is one.
- Never say "you are ineligible" as a closing thought. If there is a path,
  the path IS the answer.
- No greetings, no sign-off, no markdown, no bullet points. Just what you'd say.
- Under 90 words.

ALREADY ELIGIBLE FOR:
{eligible}

NEXT STEP THAT UNLOCKS MORE:
{ladder}

STILL NEED TO ASK:
{missing}
"""


def narrate(
    decisions: list[Decision],
    paths: list[LadderPath],
    language: str = "hi",
    missing: list[str] | None = None,
) -> str:
    """
    Turn the engine's output into something a person can hear and act on.

    The model phrases; it does not decide. Everything it is allowed to say is
    already fixed by the time it is called.
    """
    eligible = [d for d in decisions if d.status is Status.ELIGIBLE]

    eligible_text = (
        "\n".join(f"- {d.scheme_name}: {d.benefit_summary}" for d in eligible)
        or "(nothing yet)"
    )
    # One step that unlocks five schemes is one sentence, not five. Reading the
    # same action out repeatedly is how a strong finding gets buried.
    grouped = group_by_first_step(paths)
    if grouped:
        g = grouped[0]
        ladder_text = (
            f"- {g.action} at {g.where}. "
            f"Takes about {g.days} days, costs "
            f"{'nothing' if g.cost_rupees == 0 else f'Rs {g.cost_rupees}'}. "
            f"This one step unlocks {len(g.schemes)} schemes "
            f"({', '.join(g.schemes)}), worth Rs {g.unlocks_rupees:,} in total."
        )
    else:
        ladder_text = "(no next step available)"
    missing_text = (
        "\n".join(f"- {f.replace('_', ' ')}" for f in (missing or [])[:2]) or "(nothing)"
    )

    return _generate(
        NARRATE_PROMPT.format(
            language=LANGUAGE_NAMES.get(language, "Hindi"),
            eligible=eligible_text,
            ladder=ladder_text,
            missing=missing_text,
        ),
        kind="narrate",
        temperature=0.2,
    ).strip()
