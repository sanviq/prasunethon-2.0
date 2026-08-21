"""
Tests for the LLM adapter.

No network. Every test stubs _generate(), because what matters here is not what
Gemini says -- it is what we do with the answer, and specifically that the model
can never widen its own remit into deciding eligibility.
"""

from __future__ import annotations

import json

import pytest

from setu import llm
from setu.ladder import best_paths
from setu.rules import Profile, Status, evaluate_all, evaluate_scheme


@pytest.fixture
def stub(monkeypatch):
    """Replace the provider call with a canned response."""

    def _install(payload):
        text = payload if isinstance(payload, str) else json.dumps(payload)
        calls = []

        def fake(prompt, *, kind, schema=None, temperature=0.0):
            calls.append({"prompt": prompt, "kind": kind, "temperature": temperature})
            return text

        monkeypatch.setattr(llm, "_generate", fake)
        return calls

    return _install


# --------------------------------------------------------------------------
# Extraction: silence must never become a fact
# --------------------------------------------------------------------------

def test_unstated_fields_stay_none(stub):
    stub({"documents": [], "occupation_category": "street_vendor"})
    p = llm.extract_profile("main sabzi bechta hoon", "hi")

    assert p.occupation_category == "street_vendor"
    assert p.age is None and p.monthly_income is None and p.has_loan_npa is None


def test_null_does_not_erase_a_known_fact(stub):
    """
    "She didn't mention it this turn" is not "she retracted it". Wiping the
    profile on every turn would make a multi-turn conversation lose ground.
    """
    known = Profile(age=34, documents=["aadhaar"])
    stub({"documents": [], "age": None, "daily_income": 500})

    p = llm.extract_profile("roz paanch sau kamata hoon", "hi", base=known)
    assert p.age == 34
    assert p.daily_income == 500


def test_documents_merge_without_duplicates(stub):
    known = Profile(documents=["aadhaar"])
    stub({"documents": ["aadhaar", "bank_account"]})

    p = llm.extract_profile("bank account bhi hai", "hi", base=known)
    assert p.documents == ["aadhaar", "bank_account"]


def test_denials_are_recorded_not_dropped(stub):
    """
    "I have no bank account" is a fact. Recording it is what lets Setu route her
    to opening one; dropping it means Setu can only say "I don't know" forever.
    """
    stub({"documents": ["aadhaar"], "documents_denied": ["bank_account"]})
    p = llm.extract_profile("aadhaar hai, bank account nahi hai", "hi")

    assert p.documents == ["aadhaar"]
    assert p.documents_denied == ["bank_account"]


def test_acquiring_a_document_retracts_the_denial(stub):
    """Turn one: no bank account. Turn three: opened one. Turn one must yield."""
    known = Profile(documents=["aadhaar"], documents_denied=["bank_account"])
    stub({"documents": ["bank_account"], "documents_denied": []})

    p = llm.extract_profile("bank account khul gaya", "hi", base=known)
    assert "bank_account" in p.documents
    assert p.documents_denied == []


def test_extraction_runs_at_temperature_zero(stub):
    """A creative rephrasing of someone's income is a bug, not a flourish."""
    calls = stub({"documents": []})
    llm.extract_profile("kuch bhi", "hi")
    assert calls[0]["temperature"] == 0.0


def test_extraction_prompt_carries_the_daily_income_guidance(stub):
    """The single most common shape of input in this population."""
    calls = stub({"documents": []})
    llm.extract_profile("roz paanch sau", "hi")
    assert "daily_income" in calls[0]["prompt"]


def test_non_json_response_is_an_error_not_a_silent_empty_profile(stub, monkeypatch):
    stub("sorry, I cannot help with that")
    with pytest.raises(llm.LLMError):
        llm.extract_profile("anything", "hi")


# --------------------------------------------------------------------------
# Narration: the model phrases, it never decides
# --------------------------------------------------------------------------

def _vendor() -> Profile:
    return Profile(
        age=30,
        occupation_category="street_vendor",
        daily_income=500,
        documents=["aadhaar"],
        documents_denied=["bank_account", "jan_dhan_account"],
        vending_since_year=2019,
        is_epfo_esic_member=False,
        is_income_tax_payer=False,
        has_loan_npa=False,
        language="mr",
    )


def test_narration_is_only_shown_schemes_the_engine_approved(stub):
    p = _vendor()
    decisions = evaluate_all(p)
    calls = stub("tumhi e-Shram sathi patra ahat.")

    llm.narrate(decisions, best_paths(p, decisions), "mr")
    prompt = calls[0]["prompt"]

    eligible = {d.scheme_name for d in decisions if d.status is Status.ELIGIBLE}
    not_eligible = {d.scheme_name for d in decisions if d.status is Status.NOT_ELIGIBLE}

    section = prompt.split("ALREADY ELIGIBLE FOR:")[1].split("NEXT STEP")[0]
    for name in eligible:
        assert name in section
    for name in not_eligible:
        assert name not in section


def test_narration_prompt_forbids_inventing_schemes(stub):
    p = _vendor()
    decisions = evaluate_all(p)
    calls = stub("kahi tari")

    llm.narrate(decisions, best_paths(p, decisions), "mr")
    prompt = calls[0]["prompt"]
    assert "Do NOT add any scheme" in prompt
    assert "Do NOT invent amounts" in prompt


def test_narration_leads_with_the_ladder_step_when_one_exists(stub):
    p = _vendor()
    decisions = evaluate_all(p)
    paths = best_paths(p, decisions)
    calls = stub("kahi tari")

    llm.narrate(decisions, paths, "mr")
    assert paths[0].rungs[0].action in calls[0]["prompt"]


def test_narration_handles_having_no_path_without_crashing(stub):
    """A 43-year-old who cannot reach APY still deserves a sentence."""
    p = Profile(age=43, documents=["aadhaar", "bank_account"], is_income_tax_payer=False)
    stub("kahi tari")
    assert llm.narrate([evaluate_scheme("atal_pension", p)], [], "hi")


def test_narration_speaks_the_language_the_caller_used(stub):
    calls = stub("kahi tari")
    llm.narrate([], [], "mr")
    assert "Marathi" in calls[0]["prompt"]


# --------------------------------------------------------------------------
# Offline mode -- the demo has to survive bad conference wifi
# --------------------------------------------------------------------------

def test_offline_mode_refuses_uncached_calls(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "LLM_MODE", "offline")
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    with pytest.raises(llm.LLMOffline):
        llm._generate("something never asked before", kind="extract")


def test_cache_hit_needs_no_key_and_no_network(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    key = llm._cache_key("extract", {"prompt": "p", "schema": None, "t": 0.0})
    llm._store(key, "cached answer")

    assert llm._generate("p", kind="extract") == "cached answer"
