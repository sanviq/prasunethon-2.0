"""
Tests for the eligibility engine and the Ladder.

The three-valued behaviour is the thing most worth protecting. Every regression
we care about is some version of "a missing fact got treated as a failure".
"""

from __future__ import annotations

import pytest

from setu.ladder import best_paths, build_path, verify_path
from setu.rules import (
    Profile,
    Status,
    evaluate_all,
    evaluate_scheme,
    load_schemes,
    missing_fields,
)


# --------------------------------------------------------------------------
# The invariant: unknown is not a failure
# --------------------------------------------------------------------------

def test_empty_profile_is_never_not_eligible():
    """An empty profile means we have not asked yet, not that she qualifies for nothing."""
    for decision in evaluate_all(Profile()):
        assert decision.status is Status.NEED_INFO, (
            f"{decision.scheme_id} rejected a caller we know nothing about"
        )


def test_empty_documents_list_does_not_fail_document_rules():
    """[] means unasked. Opening a conversation with a rejection is the bug."""
    p = Profile(age=30, occupation_category="street_vendor", documents=[])
    decision = evaluate_scheme("pm_svanidhi", p)
    assert decision.status is Status.NEED_INFO
    assert not decision.failed


def test_known_fact_can_still_fail():
    """NEED_INFO must not swallow real disqualifications."""
    p = Profile(age=52, documents=["aadhaar", "bank_account"])
    assert evaluate_scheme("pmjjby", p).status is Status.NOT_ELIGIBLE


# --------------------------------------------------------------------------
# Income derivation
# --------------------------------------------------------------------------

def test_daily_income_derives_annual_turnover():
    """Informal workers quote a daily figure. 26 working days is the convention."""
    assert Profile(daily_income=500).get("annual_turnover") == 500 * 26 * 12


def test_monthly_income_takes_precedence():
    assert Profile(monthly_income=12000, daily_income=500).get("annual_turnover") == 144000


def test_turnover_unknown_when_no_income_stated():
    assert Profile().get("annual_turnover") is None


def test_daily_earner_clears_a_monthly_income_ceiling():
    """
    Regression: PM-SYM's ceiling is written monthly, but nobody in this
    population says "13,000 a month" -- they say "500 a day". Pointing the rule
    at raw monthly_income silently dropped PM-SYM for every daily earner.
    """
    p = Profile(
        age=30,
        daily_income=500,  # -> 13,000/month, inside the 15,000 ceiling
        documents=["aadhaar", "bank_account"],
        is_epfo_esic_member=False,
        is_income_tax_payer=False,
    )
    assert evaluate_scheme("pm_sym", p).status is Status.ELIGIBLE


def test_ladder_surfaces_schemes_blocked_behind_another_blocker():
    """
    Regression: a scheme whose income rule was unknown got masked by a failing
    bank-account rule, so it never showed as NEED_INFO, was never asked about,
    and the Ladder dropped it at verification. The caller never heard about a
    Rs 36,000 pension she qualified for.
    """
    p = Profile(
        age=30,
        occupation_category="street_vendor",
        daily_income=500,
        documents=["aadhaar"],
        is_epfo_esic_member=False,
        is_income_tax_payer=False,
        has_loan_npa=False,
    )
    reached = {path.scheme_id for path in best_paths(p, evaluate_all(p))}
    assert "pm_sym" in reached


# --------------------------------------------------------------------------
# NPA vs merely holding a loan
# --------------------------------------------------------------------------

def test_holding_a_loan_is_not_a_disqualification():
    """Only an NPA disqualifies. Conflating the two rejects most real applicants."""
    p = Profile(
        age=34,
        occupation_category="street_vendor",
        documents=["aadhaar", "bank_account", "vending_certificate"],
        vending_since_year=2019,
        has_loan_npa=False,
    )
    assert evaluate_scheme("pm_svanidhi", p).status is Status.ELIGIBLE


def test_npa_blocks_credit_schemes():
    p = Profile(age=34, documents=["aadhaar", "bank_account"], has_loan_npa=True)
    decision = evaluate_scheme("mudra_shishu", p)
    assert any(r.rule_id == "mudra_no_npa" and r.passed is False for r in decision.results)


# --------------------------------------------------------------------------
# The Ladder
# --------------------------------------------------------------------------

def test_ladder_orders_aadhaar_before_bank_account():
    """You cannot open a Jan Dhan account without Aadhaar, however cheap it is."""
    p = Profile(age=30, occupation_category="street_vendor", documents=["vending_certificate"])
    path = build_path(evaluate_scheme("pm_svanidhi", p), p)

    ids = [r.rule_id for r in path.rungs]
    assert ids.index("svanidhi_has_aadhaar") < ids.index("svanidhi_has_bank_account")


def test_ladder_actually_reaches_eligible():
    """A path that does not work is worse than admitting there is none."""
    p = Profile(
        age=30,
        occupation_category="street_vendor",
        documents=["aadhaar", "bank_account"],
        vending_since_year=2019,
        has_loan_npa=False,
    )
    path = build_path(evaluate_scheme("pm_svanidhi", p), p)
    assert path.reachable
    assert verify_path(path, p)


def test_ladder_headline_matches_the_pitch_shape():
    """'LoR from TVC - Rs 0 - 7 days -> unlocks Rs 10,000'"""
    p = Profile(
        age=30,
        occupation_category="street_vendor",
        documents=["aadhaar", "bank_account"],
        vending_since_year=2019,
        has_loan_npa=False,
    )
    headline = build_path(evaluate_scheme("pm_svanidhi", p), p).headline()
    assert "Rs 0" in headline and "7 days" in headline and "unlocks Rs 10,000" in headline


def test_age_failure_is_a_dead_end_not_a_rung():
    """You cannot act your way out of being 43. Saying so is the honest answer."""
    p = Profile(age=43, documents=["aadhaar", "bank_account"], is_income_tax_payer=False)
    path = build_path(evaluate_scheme("atal_pension", p), p)
    assert not path.reachable
    assert any(r.rule_id == "apy_age_max" for r in path.dead_ends)


def test_best_paths_are_all_verified():
    p = Profile(age=28, occupation_category="street_vendor", documents=[], has_loan_npa=False)
    for path in best_paths(p, evaluate_all(p)):
        assert verify_path(path, p)


# --------------------------------------------------------------------------
# Ranking and next-question selection
# --------------------------------------------------------------------------

def test_eligible_schemes_rank_above_ineligible():
    p = Profile(
        age=28,
        occupation_category="street_vendor",
        monthly_income=9000,
        documents=["aadhaar", "bank_account", "vending_certificate"],
        vending_since_year=2019,
        is_epfo_esic_member=False,
        is_income_tax_payer=False,
        has_loan_npa=False,
    )
    statuses = [d.status for d in evaluate_all(p)]
    assert statuses == sorted(
        statuses, key=lambda s: {Status.ELIGIBLE: 0, Status.NEED_INFO: 1, Status.NOT_ELIGIBLE: 2}[s]
    )


def test_missing_fields_prioritises_the_most_unblocking_question():
    """Age gates the most schemes, so it should be the first thing we ask."""
    assert missing_fields(Profile())[0] == "age"


# --------------------------------------------------------------------------
# Data integrity -- catches typos in schemes.json before a demo does
# --------------------------------------------------------------------------

def test_every_rule_carries_a_citation():
    for scheme in load_schemes()["schemes"]:
        for rule in scheme["rules"]:
            assert rule["source_doc"] and rule["source_url"] and rule["source_quote"], (
                f"{scheme['id']}/{rule['id']} has no citation and cannot be defended"
            )


def test_depends_on_targets_exist_within_the_same_scheme():
    for scheme in load_schemes()["schemes"]:
        ids = {r["id"] for r in scheme["rules"]}
        for rule in scheme["rules"]:
            for dep in rule["depends_on"]:
                assert dep in ids, f"{scheme['id']}/{rule['id']} depends on unknown rule {dep}"


def test_remedies_grant_something_the_rule_checks():
    """A remedy that does not satisfy its own rule builds a ladder to nowhere."""
    for scheme in load_schemes()["schemes"]:
        for rule in scheme["rules"]:
            remedy = rule.get("remedy")
            if not remedy:
                continue
            grants = remedy["grants"]
            assert grants["field"] == rule["field"], (
                f"{scheme['id']}/{rule['id']} remedy grants {grants['field']} "
                f"but the rule checks {rule['field']}"
            )
