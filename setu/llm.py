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
        "started_work_age": {"type": "integer", "nullable": True},
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
- Starting work: if they name a YEAR ("since 2015"), put it in
  vending_since_year. If instead they say how OLD they were ("I started when I
  was fifteen"), put 15 in started_work_age and leave vending_since_year null.
  Do NOT convert one into the other yourself -- Setu does that arithmetic from
  their stated age, and it is the kind of sum that must be exact.

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

{answering}
They were speaking {language}.

What they said:
\"\"\"{transcript}\"\"\"
"""

ANSWERING_NOTE = """IMPORTANT -- they were just asked: {question}

So a bare "yes", "no", "haan", "nahi", "hoy", "nahin" answers THAT question and
nothing else. Record it against the right field. A one-word answer with no
context recorded against nothing is how this conversation ends up asking the
same question a third time.
"""


def extract_profile(
    transcript: str,
    language: str = "hi",
    *,
    base: Profile | None = None,
    answering: str | None = None,
) -> Profile:
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
            answering=(
                ANSWERING_NOTE.format(question=QUESTION_HINTS[answering])
                if answering and answering in QUESTION_HINTS
                else ""
            ),
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

# What each missing field sounds like as a QUESTION A HUMAN WOULD ASK.
#
# Without this the field name itself reached the model, and Setu said the words
# "क्या आप ईपीएफओ या ईएसआईसी के सदस्य हैं" -- EPFO and ESIC, spoken aloud to a
# vegetable seller. Nobody on earth answers that question. The scheme's own
# vocabulary is not the caller's vocabulary, and the gap between them is where
# this product either works or doesn't.
QUESTION_HINTS: dict[str, str] = {
    "age": "how old they are",
    "occupation_category": "what work they do to earn money",
    "monthly_income_est": "roughly what they earn -- let them answer per day if that is how they think",
    "daily_income": "roughly what they earn in a day",
    # One question per document, never "what papers do you have". That question
    # cannot be answered in a way that closes the gap: a caller who says
    # "Aadhaar and a ration card" HAS answered, yet the bank account is still
    # unknown, so the gap survives and the same sentence gets asked again. Ask
    # about the one document that is actually blocking, as a yes/no.
    "documents": (
        "which papers they already have. Name a few plainly -- Aadhaar card, "
        "a bank passbook, a ration card. Never say 'KYC' or 'documentation'."
    ),
    "documents:aadhaar": "whether they have an Aadhaar card. A simple yes or no.",
    "documents:bank_account": (
        "whether they have a bank account of any kind -- any bank passbook, "
        "including a Jan Dhan account. A simple yes or no."
    ),
    "documents:jan_dhan_account": (
        "whether they have a bank account of any kind, including a Jan Dhan "
        "account. A simple yes or no."
    ),
    "documents:vending_certificate": (
        "whether the municipality has given them a vending certificate, a "
        "vendor ID card, or a letter from the Town Vending Committee. Many "
        "vendors have none of these, so ask as though the answer is probably no."
    ),
    "documents:letter_of_recommendation": (
        "whether they have a letter from the Town Vending Committee or the "
        "municipal office."
    ),
    "documents:ration_card": "whether they have a ration card.",
    "documents:pan": "whether they have a PAN card.",
    "documents:voter_id": "whether they have a voter ID card.",
    "documents:upi_id": "whether they use UPI on a phone.",
    "is_epfo_esic_member": (
        "whether they work for themselves or have a salaried job with a company. "
        "Ask it exactly that way. NEVER say EPFO, ESIC, or PF -- those are our "
        "words, not theirs, and asking directly gets no answer at all."
    ),
    "is_income_tax_payer": (
        "whether they file income tax every year. Keep it light -- most will "
        "simply say no."
    ),
    "vending_since_year": (
        "when they started this work. Let them answer either way -- the year, "
        "or simply how old they were when they started. Most people remember "
        "their own age far better than a calendar year, so offer that option "
        "out loud."
    ),
    "vending_since_year_est": (
        "when they started this work. Let them answer either way -- the year, "
        "or simply how old they were when they started. Most people remember "
        "their own age far better than a calendar year, so offer that option "
        "out loud."
    ),
    "took_govt_credit_scheme_last_5y": (
        "whether they have taken any government loan in the last five years, "
        "such as a vendor loan. Do not list scheme acronyms."
    ),
    "has_loan_npa": (
        "whether any earlier loan of theirs went bad or was written off. "
        "Ask gently; this one carries shame."
    ),
}


def describe_fact(field: str, value: Any) -> str:
    """
    Render a learned fact as something safe to say back.

    Carries the VALUE, not just the field name. Passing "whether they file
    income tax" let the model acknowledge a caller who had just said she does
    NOT file with "you pay tax, right" -- inverting the one fact she had
    volunteered. An acknowledgement that gets the fact backwards is worse than
    none, because it teaches the caller that Setu is not listening.
    """
    if field == "age":
        return f"they are {value} years old"
    if field == "occupation_category":
        return f"they work as: {str(value).replace('_', ' ')}"
    if field == "monthly_income":
        return f"they earn about Rs {value:,.0f} a month"
    if field == "daily_income":
        return f"they earn about Rs {value:,.0f} a day"
    if field == "vending_since_year":
        return f"they have done this work since {value}"
    if field == "started_work_age":
        return f"they started this work at age {value}"
    if field == "documents":
        return "they have: " + ", ".join(str(v).replace("_", " ") for v in value)
    if field == "documents_denied":
        return "they do NOT have: " + ", ".join(str(v).replace("_", " ") for v in value)
    if field == "is_epfo_esic_member":
        return "they work for themselves" if value is False else "they have a company job"
    if field == "is_income_tax_payer":
        return "they do NOT file income tax" if value is False else "they do file income tax"
    if field == "has_loan_npa":
        return "an earlier loan went bad" if value else "no earlier loan went bad"
    if field == "took_govt_credit_scheme_last_5y":
        return (
            "they have taken a government loan recently" if value
            else "they have not taken a government loan recently"
        )
    if field == "state":
        return f"they live in {value}"
    if field == "gender":
        return f"their gender: {value}"
    return f"{field.replace('_', ' ')}: {value}"


NARRATE_PROMPT = """You are a helpful person sitting beside someone, telling
them what government help they can get. You speak their language and nothing
else.

The eligibility decision has ALREADY been made by a rule engine reading
government documents. You are putting it into words. You are not deciding.

Hard constraints:
- Do NOT add any scheme that is not listed below.
- Do NOT remove or hedge any scheme that is listed.
- Do NOT restate or reinterpret the eligibility rules.
- Do NOT invent amounts, deadlines, offices, or timelines.

How to speak:
- Speak {language}, the way a helpful neighbour would. Warm, direct, ordinary.
- Short sentences. This is read ALOUD, possibly down a phone line.
- Lead with what they can get right now. That is the reason they called.
- Then the single next step, if there is one, with what it costs and how long.
- Never end on "you don't qualify". If there is a path, the path IS the answer.
- No greetings, no sign-off, no markdown, no lists. Just what you would say.
- Under 80 words.

When the eligible list below is empty, that means WE ARE STILL FINDING OUT --
it does NOT mean nothing is available to them. Never say "no scheme applies to
you", "nothing is available", or anything a person would hear as a rejection.
We have not finished asking. Say you are still checking, then ask your one
question. Telling someone they get nothing because they have not yet answered a
question is the exact failure this whole system exists to prevent.

WHAT THEY JUST TOLD YOU:
{just_learned}

Acknowledge that in a few words before anything else -- briefly, the way a
person does. "Right, your own cart." Someone who answers a question and hears
no sign they were heard stops answering.

Say it back ACCURATELY. If the line says they do NOT have something, never
acknowledge it as though they do. Getting a fact backwards is worse than
staying silent, because it teaches them you are not listening.

If the line below reads NOTHING-NEW, say no acknowledgement at all -- go
straight to the rest. Never read the words "nothing new" aloud, and never say
"right, nothing"; that is an instruction to you, not something they said.

THE QUESTION:
{question}

End by asking that ONE question, naturally, as the last thing you say. Ask only
that one -- never stack two questions in a turn. A person answering aloud can
hold exactly one question in their head, and asking two gets you an answer to
neither. If the question is "(nothing to ask)", do not ask anything; close by
telling them what to do next instead.

{repeat_note}

ALREADY ELIGIBLE FOR:
{eligible}

NEXT STEP THAT UNLOCKS MORE:
{ladder}
"""

REPEAT_NOTE = """You have already asked this once and they answered something
else. Ask it a different way this time, more concretely, and do not let it sound
like a form being re-read. If they dodge twice, they may not understand the
word you are using."""


def choose_question(
    missing: list[str], asked_before: list[str] | None = None
) -> str | None:
    """
    Which single thing to ask about next.

    Prefers something we have not asked yet. A caller who answers a different
    question than the one posed has usually not understood it, and asking the
    identical sentence a third time is how a conversation dies -- so we move on
    and come back to it later rather than grinding.
    """
    history = list(asked_before or [])
    known = [f for f in missing if f in QUESTION_HINTS]
    if not known:
        return None

    # `missing` arrives ranked by how many schemes the field gates, so the first
    # entry is always the question worth the most. Take it -- UNLESS we asked it
    # in the last two turns, in which case they have dodged it and hearing it a
    # third time is how someone decides the thing is broken.
    #
    # Coming back to a dodged question matters more than politely never
    # repeating: age gates nearly every scheme, so a caller who answered
    # something else when asked her age must be asked again, or she reaches the
    # end of the conversation still eligible for nothing.
    recent = set(history[-2:])
    return next((f for f in known if f not in recent), known[0])


def narrate(
    decisions: list[Decision],
    paths: list[LadderPath],
    language: str = "hi",
    missing: list[str] | None = None,
    just_learned: list[str] | None = None,
    asked_before: list[str] | None = None,
    profile: Profile | None = None,
) -> str:
    """
    Turn the engine's output into something a person can hear and act on.

    The model phrases; it does not decide. Everything it is allowed to say is
    already fixed by the time it is called.

    `just_learned` is what changed on the profile this turn, so the reply can
    acknowledge it. Without that the prompt is identical across turns where
    eligibility did not move, so the cache returns the same sentence verbatim
    and Setu reads a caller the same question three times in a row.
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
    # Exactly one question, phrased the way a person asks it -- not the field
    # name, and not two questions stacked into one breath.
    field = choose_question(missing or [], asked_before)
    question = QUESTION_HINTS[field] if field else "(nothing to ask)"

    learned_text = (
        "; ".join(describe_fact(f, getattr(profile, f)) for f in just_learned)
        if just_learned and profile is not None
        else "NOTHING-NEW"
    )

    return _generate(
        NARRATE_PROMPT.format(
            language=LANGUAGE_NAMES.get(language, "Hindi"),
            eligible=eligible_text,
            ladder=ladder_text,
            question=question,
            just_learned=learned_text,
            repeat_note=REPEAT_NOTE if field and field in (asked_before or []) else "",
        ),
        kind="narrate",
        temperature=0.2,
    ).strip()
