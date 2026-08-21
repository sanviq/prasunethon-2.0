"""
Scores the rule engine against eval/personas.json.

Precision and recall are reported per scheme and overall, because the two
failure modes cost completely different things:

  a false positive  sends someone to a government office to be turned away
  a false negative  silently withholds money they were entitled to

The second is the one this product exists to fix, so recall is the number that
matters most -- but a false positive costs a day of someone's wages and a
wasted bus fare, so neither is free.

NEED_INFO is never scored as a prediction of eligibility. Not knowing is not
the same as saying no, and collapsing the two would flatter the numbers by
hiding exactly the behaviour the three-valued design exists to produce.

    ./.venv/bin/python eval/run.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from setu.rules import Profile, Status, evaluate_all, load_schemes  # noqa: E402

PERSONAS = Path(__file__).resolve().parent / "personas.json"


def score(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> int:
    personas = json.loads(PERSONAS.read_text(encoding="utf-8"))["personas"]
    scheme_ids = [s["id"] for s in load_schemes()["schemes"]]

    per_scheme = {sid: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for sid in scheme_ids}
    need_info = 0
    mismatches: list[str] = []

    for case in personas:
        profile = Profile(**case["profile"])
        decisions = evaluate_all(profile)

        predicted = {d.scheme_id for d in decisions if d.status is Status.ELIGIBLE}
        need_info += sum(1 for d in decisions if d.status is Status.NEED_INFO)
        expected = set(case["expect_eligible"])

        for sid in scheme_ids:
            bucket = per_scheme[sid]
            if sid in predicted and sid in expected:
                bucket["tp"] += 1
            elif sid in predicted:
                bucket["fp"] += 1
            elif sid in expected:
                bucket["fn"] += 1
            else:
                bucket["tn"] += 1

        if predicted != expected:
            missed = ", ".join(sorted(expected - predicted)) or "-"
            extra = ", ".join(sorted(predicted - expected)) or "-"
            mismatches.append(f"  {case['id']:28} missed: {missed:34} wrongly offered: {extra}")

    print(f"\n{len(personas)} personas x {len(scheme_ids)} schemes "
          f"= {len(personas) * len(scheme_ids)} decisions\n")
    print(f"{'SCHEME':18} {'PREC':>7} {'RECALL':>7} {'F1':>7} {'TP':>4} {'FP':>4} {'FN':>4}")
    print("-" * 60)

    totals = {"tp": 0, "fp": 0, "fn": 0}
    macro = []

    for sid in scheme_ids:
        b = per_scheme[sid]
        p, r, f = score(b["tp"], b["fp"], b["fn"])
        macro.append((p, r, f))
        for k in totals:
            totals[k] += b[k]
        print(f"{sid:18} {p:7.3f} {r:7.3f} {f:7.3f} {b['tp']:4} {b['fp']:4} {b['fn']:4}")

    mp, mr, mf = score(totals["tp"], totals["fp"], totals["fn"])
    print("-" * 60)
    print(f"{'MICRO':18} {mp:7.3f} {mr:7.3f} {mf:7.3f} "
          f"{totals['tp']:4} {totals['fp']:4} {totals['fn']:4}")
    print(f"{'MACRO':18} {sum(x[0] for x in macro)/len(macro):7.3f} "
          f"{sum(x[1] for x in macro)/len(macro):7.3f} "
          f"{sum(x[2] for x in macro)/len(macro):7.3f}")

    print(f"\nNEED_INFO returned {need_info} times "
          f"(never counted as a prediction of eligibility)")

    # ----------------------------------------------------------------------
    # False rejections on incomplete information.
    #
    # This is the number that matters most for a first turn. Everything above
    # measures a complete profile; this measures what happens when someone has
    # said one sentence, which is the state they are actually in when the
    # system first has to behave well.
    # ----------------------------------------------------------------------
    partial = json.loads(PERSONAS.read_text(encoding="utf-8")).get("partial", [])
    wrong_rejections: list[str] = []
    checked = 0

    for case in partial:
        profile = Profile(**case["profile"])
        allowed = set(case["may_reject"])
        for decision in evaluate_all(profile):
            checked += 1
            if decision.status is Status.NOT_ELIGIBLE and decision.scheme_id not in allowed:
                blocking = ", ".join(r.field_name for r in decision.blockers)
                wrong_rejections.append(
                    f"  {case['id']:26} rejected {decision.scheme_id:16} on: {blocking}"
                )

    rate = len(wrong_rejections) / checked if checked else 0.0
    print(f"\n{len(partial)} partial profiles x {len(scheme_ids)} schemes = {checked} decisions")
    print(f"false rejections on missing facts: {len(wrong_rejections)}/{checked} ({rate:.1%})")

    if wrong_rejections:
        print("\n".join(wrong_rejections))

    if mismatches:
        print(f"\n{len(mismatches)} personas disagreed with their expected outcome:")
        print("\n".join(mismatches))

    if mismatches or wrong_rejections:
        return 1

    print("\nEvery persona matched its hand-written expectation, and nothing was")
    print("rejected for a fact the caller had not been asked for yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
