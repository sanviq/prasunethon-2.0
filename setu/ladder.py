"""
The Ladder -- counterfactual path search.

Every other part of this system returns a verdict. This turns NOT_ELIGIBLE into
a route: the cheapest, fastest ordered set of steps that flips a person to
ELIGIBLE, costed in rupees and days.

This is only possible because eligibility is typed rules rather than a model's
opinion. You can invert a rule; you cannot invert a vibe.

The search stays deliberately simple. Rules are independent, so the minimal set
of mutations is just "fix every failing rule that has a remedy" -- no
combinatorial search required. What matters is:

  ordering  -- respect depends_on, then cheapest and fastest first, so the first
               rung is something the person can actually do tomorrow
  honesty   -- if a failing rule has no remedy, say so rather than inventing a
               path that dead-ends at a government counter
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .rules import Decision, Profile, RuleResult, Status, evaluate_scheme, get_rule


@dataclass
class Rung:
    rule_id: str
    action: str
    cost_rupees: int
    days: int
    where: str
    citation: dict[str, Any]

    def as_line(self) -> str:
        cost = "free" if self.cost_rupees == 0 else f"Rs {self.cost_rupees}"
        return f"{self.action} ({cost}, about {self.days} days, at {self.where})"


@dataclass
class Path:
    scheme_id: str
    scheme_name: str
    reachable: bool
    rungs: list[Rung]
    dead_ends: list[RuleResult]
    unlocks_rupees: int

    @property
    def total_cost(self) -> int:
        return sum(r.cost_rupees for r in self.rungs)

    @property
    def total_days(self) -> int:
        """Steps are sequential because later ones depend on earlier ones."""
        return sum(r.days for r in self.rungs)

    def headline(self) -> str:
        """The one-liner for the pitch: 'LoR from TVC - Rs 0 - 7 days -> unlocks Rs 15,000'."""
        if not self.reachable:
            return f"{self.scheme_name}: no path available"
        cost = "Rs 0" if self.total_cost == 0 else f"Rs {self.total_cost}"
        return (
            f"{self.rungs[0].action} - {cost} - {self.total_days} days "
            f"-> unlocks Rs {self.unlocks_rupees:,}"
        )


def apply_remedy(profile: Profile, rule: dict[str, Any]) -> Profile:
    """Return a copy of the profile as it would be once this remedy is done."""
    remedy = rule.get("remedy")
    grants = remedy.get("grants") if remedy else None
    if not grants:
        return profile

    updated = copy.deepcopy(profile)
    name, value = grants["field"], grants["value"]

    if name == "documents":
        if value not in updated.documents:
            updated.documents = [*updated.documents, value]
    else:
        setattr(updated, name, value)

    return updated


def _order_rungs(scheme_id: str, fixable: list[RuleResult]) -> list[RuleResult]:
    """
    Topological order over depends_on, cheapest-and-fastest first within a tier.

    You cannot open a Jan Dhan account before you hold Aadhaar. Sorting purely by
    cost would hand someone a list they physically cannot follow in order.
    """
    pending = {r.rule_id: r for r in fixable}
    ordered: list[RuleResult] = []
    placed: set[str] = set()

    def sort_key(r: RuleResult) -> tuple[int, int]:
        remedy = r.remedy or {}
        return (remedy.get("cost_rupees", 0), remedy.get("days", 0))

    while pending:
        ready = [
            r
            for r in pending.values()
            if all(dep in placed or dep not in pending for dep in get_rule(scheme_id, r.rule_id)["depends_on"])
        ]
        if not ready:
            # Cycle in depends_on -- a data bug. Emit the rest cheapest-first
            # rather than dropping steps silently.
            ordered.extend(sorted(pending.values(), key=sort_key))
            break
        nxt = sorted(ready, key=sort_key)[0]
        ordered.append(nxt)
        placed.add(nxt.rule_id)
        del pending[nxt.rule_id]

    return ordered


def build_path(decision: Decision, profile: Profile) -> Path:
    """
    The route from where this person is to ELIGIBLE for one scheme.

    Only failing rules are considered. NEED_INFO is not a blocker to route
    around -- it is a question to ask, and asking is the narrator's job.
    """
    blockers = decision.blockers
    fixable = [r for r in blockers if r.is_fixable]
    dead_ends = [r for r in blockers if not r.is_fixable]

    ordered = _order_rungs(decision.scheme_id, fixable)

    rungs = [
        Rung(
            rule_id=r.rule_id,
            action=r.remedy["action"],
            cost_rupees=r.remedy.get("cost_rupees", 0),
            days=r.remedy.get("days", 0),
            where=r.remedy.get("where", ""),
            citation=r.citation,
        )
        for r in ordered
    ]

    return Path(
        scheme_id=decision.scheme_id,
        scheme_name=decision.scheme_name,
        reachable=bool(rungs) and not dead_ends,
        rungs=rungs,
        dead_ends=dead_ends,
        unlocks_rupees=decision.benefit_amount_rupees,
    )


def verify_path(path: Path, profile: Profile) -> bool:
    """
    Walk the ladder on a copy of the profile and confirm it really lands on
    ELIGIBLE.

    Cheap to run and worth running every time: a path that does not actually
    work is worse than admitting there is no path, because the person spends
    real days finding out.
    """
    if not path.reachable:
        return False

    walked = profile
    for rung in path.rungs:
        walked = apply_remedy(walked, get_rule(path.scheme_id, rung.rule_id))

    return evaluate_scheme(path.scheme_id, walked).status is Status.ELIGIBLE


def best_paths(profile: Profile, decisions: list[Decision]) -> list[Path]:
    """
    Verified ladders for every scheme this person could reach, best first.

    Ranked by benefit per day of effort: a free week that unlocks Rs 10,000
    should outrank a free month that unlocks Rs 2,000.
    """
    paths = []
    for decision in decisions:
        if decision.status is not Status.NOT_ELIGIBLE:
            continue
        path = build_path(decision, profile)
        if path.reachable and verify_path(path, profile):
            paths.append(path)

    return sorted(paths, key=lambda p: -(p.unlocks_rupees / max(p.total_days, 1)))
