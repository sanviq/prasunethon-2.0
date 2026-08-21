"""
Tests for the conversation layer.

The engine can be perfectly correct and the product still fail here. These
cover the ways a conversation dies: jargon nobody answers, the same question
read out twice, an acknowledgement that gets the fact backwards, and telling
someone they get nothing when the truth is that we have not finished asking.
"""

from __future__ import annotations

import json

import pytest

from setu import llm
from setu.rules import Profile, missing_fields


@pytest.fixture
def stub(monkeypatch):
    def _install(payload):
        text = payload if isinstance(payload, str) else json.dumps(payload)
        calls = []

        def fake(prompt, *, kind, schema=None, temperature=0.0):
            calls.append(prompt)
            return text

        monkeypatch.setattr(llm, "_generate", fake)
        return calls

    return _install


# --------------------------------------------------------------------------
# Every askable field must have human phrasing
# --------------------------------------------------------------------------

def test_every_askable_field_has_a_human_phrasing():
    """
    A field missing_fields() can return but QUESTION_HINTS cannot phrase is a
    field that never gets asked, so the conversation stalls forever on a fact
    nobody will volunteer.
    """
    askable = set()
    for profile in (Profile(), Profile(age=30), Profile(age=30, documents=["aadhaar"])):
        askable.update(missing_fields(profile))

    missing_hints = askable - set(llm.QUESTION_HINTS)
    assert not missing_hints, f"no way to ask about: {sorted(missing_hints)}"


def test_no_question_leaks_scheme_jargon():
    """
    Setu once said the words "EPFO" and "ESIC" aloud to a vegetable seller.
    Nobody answers that question -- it is our vocabulary, not theirs.

    Jargon may appear in a hint only inside a prohibition ("never say KYC").
    That is the fix, not the bug, so the check is per sentence rather than a
    flat substring search.
    """
    for field, hint in llm.QUESTION_HINTS.items():
        for sentence in hint.upper().replace("--", ".").split("."):
            leaked = [j for j in ("EPFO", "ESIC", "KYC", "NPA") if j in sentence]
            if not leaked:
                continue
            assert any(word in sentence for word in ("NEVER", "NOT", "DO NOT")), (
                f"{field} hint uses {leaked} without banning it: {sentence.strip()!r}"
            )


def test_the_epfo_ban_is_explicit():
    """The one field where the wrong wording guarantees no answer at all."""
    assert "NEVER say EPFO" in llm.QUESTION_HINTS["is_epfo_esic_member"]


# --------------------------------------------------------------------------
# One question at a time, and never the same one twice
# --------------------------------------------------------------------------

def test_choose_question_skips_what_was_already_asked():
    assert llm.choose_question(["documents", "age"], ["documents"]) == "age"


def test_choose_question_falls_back_rather_than_going_silent():
    """If everything has been asked, ask again -- silence is worse."""
    assert llm.choose_question(["documents"], ["documents"]) == "documents"


def test_choose_question_ignores_fields_it_cannot_phrase():
    assert llm.choose_question(["some_internal_field", "age"], []) == "age"


def test_repeat_note_appears_only_on_a_re_ask(stub):
    plain = stub("kahi tari")
    llm.narrate([], [], "mr", ["age"], asked_before=[])
    assert llm.REPEAT_NOTE not in plain[0]

    again = stub("kahi tari")
    llm.narrate([], [], "mr", ["age"], asked_before=["age"])
    assert llm.REPEAT_NOTE in again[0]


# --------------------------------------------------------------------------
# Acknowledgements must be accurate
# --------------------------------------------------------------------------

def test_a_denial_is_acknowledged_as_a_denial():
    """
    Regression: a caller said she does NOT file income tax and Setu replied
    "you pay tax, right". Getting a fact backwards teaches her Setu is not
    listening, which is worse than saying nothing.
    """
    assert "NOT file" in llm.describe_fact("is_income_tax_payer", False)
    assert "do file" in llm.describe_fact("is_income_tax_payer", True)


def test_self_employment_is_acknowledged_in_her_words():
    assert llm.describe_fact("is_epfo_esic_member", False) == "they work for themselves"


def test_missing_documents_are_described_as_missing():
    said = llm.describe_fact("documents_denied", ["bank_account"])
    assert "do NOT have" in said and "bank account" in said


def test_nothing_new_is_an_instruction_not_a_line_to_read(stub):
    """
    Regression: with no new facts, the model read the sentinel out and told a
    caller "right, nothing at all".
    """
    calls = stub("kahi tari")
    llm.narrate([], [], "mr", ["age"], just_learned=None)
    assert "NOTHING-NEW" in calls[0]
    assert "Never read the words" in calls[0]


def test_learned_facts_reach_the_prompt_with_their_values(stub):
    calls = stub("kahi tari")
    p = Profile(age=30, daily_income=500)
    llm.narrate([], [], "hi", ["documents"], just_learned=["age", "daily_income"], profile=p)
    assert "30 years old" in calls[0]
    assert "500 a day" in calls[0]


# --------------------------------------------------------------------------
# NEED_INFO is not rejection
# --------------------------------------------------------------------------

def test_prompt_forbids_narrating_need_info_as_rejection(stub):
    """
    With nothing eligible yet, Setu said "no government scheme applies to you".
    That is the engine's NEED_INFO being spoken as NOT_ELIGIBLE -- the exact
    failure the three-valued design exists to prevent, reintroduced one layer up.
    """
    calls = stub("kahi tari")
    llm.narrate([], [], "hi", ["age"])
    assert "STILL FINDING OUT" in calls[0]
    assert "no scheme applies" in calls[0]


# --------------------------------------------------------------------------
# The stuck-conversation bugs, from a real session
# --------------------------------------------------------------------------

def test_document_gaps_are_reported_per_document():
    """
    Regression: "documents" was one coarse field, so once it was the top gap
    Setu asked "what papers do you have?" forever. A caller answering "Aadhaar
    and a ration card" HAS answered -- but the bank account is still unknown,
    so the gap survived and the identical sentence came back three times.
    """
    from setu.rules import missing_fields as mf

    gaps = mf(Profile(
        age=30, occupation_category="street_vendor", daily_income=1500,
        vending_since_year=2010, documents=["aadhaar", "ration_card"],
        is_epfo_esic_member=False,
    ))
    assert "documents" not in gaps, "still asking about papers in general"
    assert "documents:bank_account" in gaps
    assert all(g in llm.QUESTION_HINTS for g in gaps), "a gap nobody can phrase"


def test_the_same_question_is_never_asked_twice_running():
    asked = ["age", "documents:bank_account"]
    assert llm.choose_question(
        ["documents:bank_account", "is_income_tax_payer"], asked
    ) != "documents:bank_account"


def test_a_dodged_question_comes_back():
    """
    Age gates nearly every scheme. A caller who answered something else when
    asked her age must be asked again, or she reaches the end of the
    conversation eligible for nothing -- which is exactly what happened.
    """
    # asked long ago, and it is the highest-impact gap left
    assert llm.choose_question(["age", "documents:pan"], ["age", "x", "y"]) == "age"


def test_falls_back_rather_than_going_silent():
    assert llm.choose_question(["age"], ["age", "age"]) == "age"


def test_extraction_is_told_what_was_just_asked(stub):
    """
    A bare "no" means nothing without the question. Without this the answer got
    attributed to whatever was asked three turns ago, changed nothing, and the
    same question came round again.
    """
    calls = stub({"documents": []})
    llm.extract_profile("no", "en", answering="documents:bank_account")

    assert "they were just asked" in calls[0].lower()
    assert "bank account" in calls[0]


def test_no_answering_context_means_no_note(stub):
    calls = stub({"documents": []})
    llm.extract_profile("main sabzi bechta hoon", "hi")
    assert "they were just asked" not in calls[0].lower()
