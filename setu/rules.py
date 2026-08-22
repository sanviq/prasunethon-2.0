"""
The eligibility engine.

Pure functions over typed rules loaded from data/schemes.json. There is no LLM
in this file and there never should be: every verdict has to be deterministic,
replayable, and defensible by pointing at the government text it came from.

Three-valued on purpose:

    ELIGIBLE      every gating rule passed
    NOT_ELIGIBLE  at least one gating rule failed on a known fact
    NEED_INFO     nothing failed, but a fact we need is missing

A missing fact is never a failure. Telling a vendor she does not qualify
because she never mentioned her age is the exact behaviour this product exists
to replace.
"""

from __future__ import annotations

import json
from datetime import date
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

SCHEMES_PATH = Path(__file__).resolve().parent.parent / "data" / "schemes.json"

# Conventional number of working days per month for daily-wage work.
WORKING_DAYS_PER_MONTH = 26


class Status(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    NEED_INFO = "NEED_INFO"


@dataclass
class Profile:
    """What we know about a person. Every field is optional by design."""

    age: int | None = None
    gender: str | None = None
    occupation_category: str | None = None
    monthly_income: float | None = None
    daily_income: float | None = None
    state: str | None = None
    documents: list[str] = field(default_factory=list)
    # Documents she has explicitly told us she does NOT hold. Distinct from
    # "not in documents", which only ever means we have not asked.
    documents_denied: list[str] = field(default_factory=list)
    is_epfo_esic_member: bool | None = None
    is_income_tax_payer: bool | None = None
    has_loan_npa: bool | None = None
    vending_since_year: int | None = None
    # "I started when I was fifteen" is how people answer this. Almost nobody
    # remembers the calendar year, but everyone remembers their own age.
    started_work_age: int | None = None
    took_govt_credit_scheme_last_5y: bool | None = None
    language: str = "hi"

    def get(self, name: str) -> Any:
        if name in DERIVED_FIELDS:
            return DERIVED_FIELDS[name](self)
        return getattr(self, name, None)


def _annual_turnover(p: Profile) -> float | None:
    """
    Annual turnover from whatever the person actually said.

    Most informal workers quote a daily figure, not a monthly one. Deriving only
    from monthly_income leaves turnover unknown for nearly everyone we serve, so
    fall back to the daily figure at the conventional 26-day month.
    """
    if p.monthly_income is not None:
        return p.monthly_income * 12
    if p.daily_income is not None:
        return p.daily_income * WORKING_DAYS_PER_MONTH * 12
    return None


def _monthly_income_est(p: Profile) -> float | None:
    """
    Monthly income from whatever the person actually said.

    Scheme income ceilings are written monthly, but almost nobody in this
    population states a monthly figure -- they say "500 a day". Pointing the
    ceiling rules at the raw monthly_income field left them permanently unknown
    for exactly the callers the scheme is designed for.
    """
    if p.monthly_income is not None:
        return p.monthly_income
    if p.daily_income is not None:
        return p.daily_income * WORKING_DAYS_PER_MONTH
    return None


def _vending_since_year_est(p: Profile) -> int | None:
    """
    Which year they started, from whatever they actually said.

    Asking "which year did you start" gets you a blank look; asking a
    fifty-year-old to convert "I was fifteen" into 1991 on the spot is a literacy
    test, not a question. But if she has told us she is 20 and started at 15,
    the year is arithmetic on two facts she stated -- and arithmetic is exactly
    what belongs in this file rather than in a model's guess.
    """
    if p.vending_since_year is not None:
        return p.vending_since_year
    if p.age is not None and p.started_work_age is not None:
        years_working = p.age - p.started_work_age
        if 0 <= years_working <= 80:
            return date.today().year - years_working
    return None


DERIVED_FIELDS: dict[str, Callable[[Profile], Any]] = {
    "annual_turnover": _annual_turnover,
    "monthly_income_est": _monthly_income_est,
    "vending_since_year_est": _vending_since_year_est,
}


# --------------------------------------------------------------------------
# Operators. Each returns True, False, or None (meaning "cannot tell").
# --------------------------------------------------------------------------

def _cmp(op: str, actual: Any, expected: Any) -> bool | None:
    if actual is None:
        return None
    try:
        if op == "eq":
            return actual == expected
        if op == "ne":
            return actual != expected
        if op == "gt":
            return actual > expected
        if op == "gte":
            return actual >= expected
        if op == "lt":
            return actual < expected
        if op == "lte":
            return actual <= expected
        if op == "in":
            return actual in expected
        if op == "not_in":
            return actual not in expected
        if op == "contains":
            return expected in actual
        if op == "contains_any":
            return any(v in actual for v in expected)
        if op == "not_contains":
            return expected not in actual
        if op == "is_true":
            return actual is True
        if op == "is_false":
            return actual is False
    except TypeError:
        return None
    raise ValueError(f"unknown operator: {op}")


DOCUMENT_OPS = {"contains", "contains_any", "not_contains"}

# Some documents are a KIND of another. A Jan Dhan account is a bank account;
# holding one satisfies any rule asking for the other, and -- the half that
# actually bit us -- denying the general kind denies the specific one too.
#
# "I don't have a bank account" is never meant as "...but I might have a Jan
# Dhan one". Without this, that denial left jan_dhan_account merely unasked, so
# contains_any never failed, so the Ladder had no failing rung to route from and
# silently offered nothing to a woman who needed exactly one step.
DOCUMENT_IMPLIES: dict[str, set[str]] = {
    "jan_dhan_account": {"bank_account"},
}


def _expand_held(held: list[str]) -> set[str]:
    out = set(held)
    for doc in held:
        out |= DOCUMENT_IMPLIES.get(doc, set())
    return out


def _expand_denied(denied: list[str]) -> set[str]:
    """Denying a general kind denies everything that is a kind of it."""
    out = set(denied)
    for specific, generals in DOCUMENT_IMPLIES.items():
        if generals & set(denied):
            out.add(specific)
    return out


def _evaluate_document_rule(rule: dict[str, Any], profile: Profile) -> bool | None:
    """
    Document rules are three-valued against two lists, not one.

    `documents` is what she has told us she holds. `documents_denied` is what she
    has told us she does NOT hold. Anything in neither list is simply unasked.

    The subtle version of this bug is worse than the empty-list version: a caller
    who mentions her Aadhaar has a non-empty documents list, so a naive
    `"bank_account" in documents` returns False and Setu starts hard-rejecting
    every bank-gated scheme -- on the strength of a question nobody asked her.
    Mentioning one document is not a claim about all the others.

    This is why the Ladder cannot route until we have asked. That is correct:
    you cannot tell someone how to fix a gap you have not established exists.
    """
    op = rule["op"]
    value = rule.get("value")
    held = _expand_held(profile.documents)
    denied = _expand_denied(profile.documents_denied)

    if op == "contains":
        if value in held:
            return True
        return False if value in denied else None

    if op == "contains_any":
        if any(v in held for v in value):
            return True
        return False if all(v in denied for v in value) else None

    if op == "not_contains":
        if value in held:
            return False
        return True if value in denied else None

    raise ValueError(f"not a document op: {op}")


@dataclass
class RuleResult:
    rule_id: str
    description: str
    passed: bool | None  # None -> unknown
    gating: bool
    field_name: str
    citation: dict[str, Any]
    remedy: dict[str, Any] | None
    # For document rules: which documents would satisfy this. "documents" as a
    # single field is too coarse to ask about -- a caller who answers "I have
    # Aadhaar and a ration card" has answered, yet bank_account is still
    # unknown, so the gap persists and the same question gets asked forever.
    wanted: tuple[str, ...] = ()

    @property
    def is_fixable(self) -> bool:
        return self.passed is False and self.remedy is not None


@dataclass
class Decision:
    scheme_id: str
    scheme_name: str
    status: Status
    benefit_summary: str
    benefit_amount_rupees: int
    results: list[RuleResult]

    @property
    def failed(self) -> list[RuleResult]:
        return [r for r in self.results if r.passed is False]

    @property
    def unknown(self) -> list[RuleResult]:
        return [r for r in self.results if r.passed is None]

    @property
    def blockers(self) -> list[RuleResult]:
        """Failing rules that actually stop the application."""
        return [r for r in self.failed if r.gating]


@lru_cache(maxsize=1)
def load_schemes() -> dict[str, Any]:
    with SCHEMES_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def get_scheme(scheme_id: str) -> dict[str, Any]:
    for scheme in load_schemes()["schemes"]:
        if scheme["id"] == scheme_id:
            return scheme
    raise KeyError(f"no such scheme: {scheme_id}")


def get_rule(scheme_id: str, rule_id: str) -> dict[str, Any]:
    for rule in get_scheme(scheme_id)["rules"]:
        if rule["id"] == rule_id:
            return rule
    raise KeyError(f"no such rule: {scheme_id}/{rule_id}")


def evaluate_rule(rule: dict[str, Any], profile: Profile) -> RuleResult:
    if rule["op"] in DOCUMENT_OPS:
        passed = _evaluate_document_rule(rule, profile)
    else:
        passed = _cmp(rule["op"], profile.get(rule["field"]), rule.get("value"))

    value = rule.get("value")
    wanted = (
        tuple(value) if isinstance(value, list) else (value,)
    ) if rule["op"] in DOCUMENT_OPS else ()

    return RuleResult(
        rule_id=rule["id"],
        description=rule["description"],
        passed=passed,
        gating=rule.get("gating", True),
        field_name=rule["field"],
        citation={
            "doc": rule["source_doc"],
            "url": rule["source_url"],
            "quote": rule["source_quote"],
        },
        remedy=rule.get("remedy"),
        wanted=wanted,
    )


def evaluate_scheme(scheme: dict[str, Any] | str, profile: Profile) -> Decision:
    if isinstance(scheme, str):
        scheme = get_scheme(scheme)

    results = [evaluate_rule(rule, profile) for rule in scheme["rules"]]

    gating = [r for r in results if r.gating]
    if any(r.passed is False for r in gating):
        status = Status.NOT_ELIGIBLE
    elif any(r.passed is None for r in gating):
        status = Status.NEED_INFO
    else:
        status = Status.ELIGIBLE

    return Decision(
        scheme_id=scheme["id"],
        scheme_name=scheme["name"],
        status=status,
        benefit_summary=scheme["benefit_summary"],
        benefit_amount_rupees=scheme["benefit_amount_rupees"],
        results=results,
    )


def evaluate_all(profile: Profile) -> list[Decision]:
    """
    Every scheme, ranked: eligible first, then fixable, then the rest.

    Within a band, higher benefit first -- someone deciding where to spend a day
    of their life should see the biggest win at the top.
    """
    decisions = [evaluate_scheme(s, profile) for s in load_schemes()["schemes"]]
    order = {Status.ELIGIBLE: 0, Status.NEED_INFO: 1, Status.NOT_ELIGIBLE: 2}
    return sorted(
        decisions,
        key=lambda d: (order[d.status], -d.benefit_amount_rupees),
    )


def missing_fields(profile: Profile) -> list[str]:
    """
    Which unknown facts would resolve the most schemes.

    Drives the next question the assistant asks, so the conversation converges
    instead of interrogating someone about paperwork that changes nothing.
    """
    counts: dict[str, int] = {}
    for decision in evaluate_all(profile):
        # NOT_ELIGIBLE decisions count too. A scheme blocked by one fixable rule
        # may still hide an unknown fact behind it, and if we only looked at
        # NEED_INFO we would never ask -- the Ladder would then quietly drop that
        # scheme at verification and the caller would never hear about it.
        if decision.status is Status.ELIGIBLE:
            continue
        for result in decision.unknown:
            if not result.gating:
                continue
            # Document gaps are reported per document, so the next question can
            # name the one thing still missing rather than re-asking "what
            # papers do you have" at someone who just listed her papers.
            key = f"documents:{result.wanted[0]}" if result.wanted else result.field_name
            counts[key] = counts.get(key, 0) + 1
    return [f for f, _ in sorted(counts.items(), key=lambda kv: -kv[1])]
