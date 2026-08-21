"""
Builds eval/personas.json.

Expected outcomes here are derived BY HAND from the published scheme criteria,
never from Setu's own output. An eval whose labels come from the system under
test measures nothing at all -- it just proves the code is self-consistent.

Each persona fully specifies every gating field it needs, because a persona
with an unspecified field lands on NEED_INFO rather than ELIGIBLE, and that is
a different assertion than the one we mean to make.

Gating criteria, transcribed from schemes.json:

  pm_svanidhi     street_vendor, vending proof, aadhaar, bank, vending<=2020
  pm_sym          18-40, monthly<=15000, not EPFO/ESIC, not taxpayer, bank, aadhaar
  e_shram         16-59, not EPFO/ESIC, not taxpayer, aadhaar
  pmjjby          18-50, bank
  pmsby           18-70, bank
  pm_vishwakarma  listed trade, 18+, aadhaar, no PMEGP/SVANidhi loan in 5y
  atal_pension    18-40, not taxpayer, bank
  mudra_shishu    18+, bank
"""

from __future__ import annotations

import json
from pathlib import Path

TRADES = ["artisan", "tailor", "carpenter", "blacksmith", "potter",
          "cobbler", "goldsmith", "barber", "washerman", "mason"]

FULL_DOCS = ["aadhaar", "bank_account"]
VENDOR_DOCS = ["aadhaar", "bank_account", "vending_certificate"]


def persona(pid, note, expect, **profile):
    base = {
        "documents": [],
        "is_epfo_esic_member": False,
        "is_income_tax_payer": False,
        "has_loan_npa": False,
        "took_govt_credit_scheme_last_5y": False,
    }
    base.update(profile)
    return {"id": pid, "note": note, "profile": base, "expect_eligible": sorted(expect)}


P = []

# -- street vendors, the core persona ---------------------------------------
P += [
    persona("vendor_full", "30, vending since 2015, all papers",
            ["pm_svanidhi", "pm_sym", "e_shram", "pmjjby", "pmsby", "atal_pension", "mudra_shishu"],
            age=30, occupation_category="street_vendor", daily_income=500,
            documents=VENDOR_DOCS, vending_since_year=2015),

    persona("vendor_no_vending_proof", "no CoV or LoR -- SVANidhi blocked, rest fine",
            ["pm_sym", "e_shram", "pmjjby", "pmsby", "atal_pension", "mudra_shishu"],
            age=30, occupation_category="street_vendor", daily_income=500,
            documents=FULL_DOCS, vending_since_year=2015),

    persona("vendor_started_2023", "started after the 24 Mar 2020 cutoff",
            ["pm_sym", "e_shram", "pmjjby", "pmsby", "atal_pension", "mudra_shishu"],
            age=30, occupation_category="street_vendor", daily_income=500,
            documents=VENDOR_DOCS, vending_since_year=2023),

    persona("vendor_no_bank", "aadhaar only -- every bank-gated scheme blocked",
            ["e_shram"],
            age=30, occupation_category="street_vendor", daily_income=500,
            documents=["aadhaar"], vending_since_year=2015),

    persona("vendor_no_aadhaar", "bank but no aadhaar; APY has no aadhaar rule",
            ["pmjjby", "pmsby", "atal_pension", "mudra_shishu"],
            age=30, occupation_category="street_vendor", daily_income=500,
            documents=["bank_account"], vending_since_year=2015),

    persona("vendor_lor_not_cov", "LoR satisfies the vending proof rule",
            ["pm_svanidhi", "pm_sym", "e_shram", "pmjjby", "pmsby", "atal_pension", "mudra_shishu"],
            age=30, occupation_category="street_vendor", daily_income=500,
            documents=["aadhaar", "bank_account", "letter_of_recommendation"],
            vending_since_year=2015),

    persona("vendor_jan_dhan", "jan dhan counts as a bank account",
            ["pm_svanidhi", "pm_sym", "e_shram", "pmjjby", "pmsby", "atal_pension", "mudra_shishu"],
            age=30, occupation_category="street_vendor", daily_income=500,
            documents=["aadhaar", "jan_dhan_account", "vending_certificate"],
            vending_since_year=2015),

    persona("vendor_npa", "NPA is non-gating, so eligibility is unchanged",
            ["pm_svanidhi", "pm_sym", "e_shram", "pmjjby", "pmsby", "atal_pension", "mudra_shishu"],
            age=30, occupation_category="street_vendor", daily_income=500,
            documents=VENDOR_DOCS, vending_since_year=2015, has_loan_npa=True),
]

# -- age boundaries ----------------------------------------------------------
P += [
    persona("age_17", "below every adult scheme; e-Shram opens at 16",
            ["e_shram"],
            age=17, occupation_category="other", daily_income=200, documents=FULL_DOCS),

    persona("age_18", "lower boundary, inclusive everywhere",
            ["pm_sym", "e_shram", "pmjjby", "pmsby", "atal_pension", "mudra_shishu"],
            age=18, occupation_category="other", daily_income=300, documents=FULL_DOCS),

    persona("age_40", "PM-SYM and APY upper boundary, inclusive",
            ["pm_sym", "e_shram", "pmjjby", "pmsby", "atal_pension", "mudra_shishu"],
            age=40, occupation_category="other", daily_income=300, documents=FULL_DOCS),

    persona("age_41", "just past the pension entry window",
            ["e_shram", "pmjjby", "pmsby", "mudra_shishu"],
            age=41, occupation_category="other", daily_income=300, documents=FULL_DOCS),

    persona("age_50", "PMJJBY upper boundary, inclusive",
            ["e_shram", "pmjjby", "pmsby", "mudra_shishu"],
            age=50, occupation_category="other", daily_income=300, documents=FULL_DOCS),

    persona("age_51", "past PMJJBY",
            ["e_shram", "pmsby", "mudra_shishu"],
            age=51, occupation_category="other", daily_income=300, documents=FULL_DOCS),

    persona("age_59", "e-Shram upper boundary, inclusive",
            ["e_shram", "pmsby", "mudra_shishu"],
            age=59, occupation_category="other", daily_income=300, documents=FULL_DOCS),

    persona("age_60", "past e-Shram",
            ["pmsby", "mudra_shishu"],
            age=60, occupation_category="other", daily_income=300, documents=FULL_DOCS),

    persona("age_70", "PMSBY upper boundary, inclusive",
            ["pmsby", "mudra_shishu"],
            age=70, occupation_category="other", daily_income=300, documents=FULL_DOCS),

    persona("age_71", "past every insurance scheme",
            ["mudra_shishu"],
            age=71, occupation_category="other", daily_income=300, documents=FULL_DOCS),
]

# -- income, stated daily and monthly ---------------------------------------
P += [
    persona("income_daily_at_ceiling", "576/day -> 14,976/month, just inside PM-SYM",
            ["pm_sym", "e_shram", "pmjjby", "pmsby", "atal_pension", "mudra_shishu"],
            age=30, occupation_category="other", daily_income=576, documents=FULL_DOCS),

    persona("income_daily_over_ceiling", "600/day -> 15,600/month, just outside PM-SYM",
            ["e_shram", "pmjjby", "pmsby", "atal_pension", "mudra_shishu"],
            age=30, occupation_category="other", daily_income=600, documents=FULL_DOCS),

    persona("income_monthly_at_ceiling", "exactly 15,000 stated monthly, inclusive",
            ["pm_sym", "e_shram", "pmjjby", "pmsby", "atal_pension", "mudra_shishu"],
            age=30, occupation_category="other", monthly_income=15000, documents=FULL_DOCS),

    persona("income_monthly_over_ceiling", "15,001 monthly",
            ["e_shram", "pmjjby", "pmsby", "atal_pension", "mudra_shishu"],
            age=30, occupation_category="other", monthly_income=15001, documents=FULL_DOCS),

    persona("income_high_earner", "60,000/month, taxpayer",
            ["pmjjby", "pmsby", "mudra_shishu"],
            age=30, occupation_category="other", monthly_income=60000,
            documents=FULL_DOCS, is_income_tax_payer=True),
]

# -- exclusions --------------------------------------------------------------
P += [
    persona("epfo_member", "formal-sector worker",
            ["pmjjby", "pmsby", "atal_pension", "mudra_shishu"],
            age=30, occupation_category="other", daily_income=400,
            documents=FULL_DOCS, is_epfo_esic_member=True),

    persona("taxpayer", "excluded from PM-SYM, e-Shram and APY",
            ["pmjjby", "pmsby", "mudra_shishu"],
            age=30, occupation_category="other", daily_income=400,
            documents=FULL_DOCS, is_income_tax_payer=True),

    persona("epfo_and_taxpayer", "both exclusions at once",
            ["pmjjby", "pmsby", "mudra_shishu"],
            age=30, occupation_category="other", monthly_income=40000,
            documents=FULL_DOCS, is_epfo_esic_member=True, is_income_tax_payer=True),

    persona("no_documents_at_all", "holds a voter id and nothing that gates a scheme",
            [],
            age=30, occupation_category="other", daily_income=300,
            documents=["voter_id"]),
]

# -- artisans and the Vishwakarma cross-scheme rule -------------------------
P += [
    persona("tailor_clean", "listed trade, no prior govt credit",
            ["pm_vishwakarma", "pm_sym", "e_shram", "pmjjby", "pmsby", "atal_pension", "mudra_shishu"],
            age=30, occupation_category="tailor", daily_income=400, documents=FULL_DOCS),

    persona("tailor_took_svanidhi", "prior SVANidhi loan closes the Vishwakarma door",
            ["pm_sym", "e_shram", "pmjjby", "pmsby", "atal_pension", "mudra_shishu"],
            age=30, occupation_category="tailor", daily_income=400,
            documents=FULL_DOCS, took_govt_credit_scheme_last_5y=True),

    persona("potter_no_bank", "trade qualifies, no bank account",
            ["pm_vishwakarma", "e_shram"],
            age=35, occupation_category="potter", daily_income=350,
            documents=["aadhaar"]),

    persona("barber_65", "past the pension and e-Shram windows",
            ["pm_vishwakarma", "pmsby", "mudra_shishu"],
            age=65, occupation_category="barber", daily_income=400, documents=FULL_DOCS),

    persona("mason_18", "youngest eligible artisan",
            ["pm_vishwakarma", "pm_sym", "e_shram", "pmjjby", "pmsby", "atal_pension", "mudra_shishu"],
            age=18, occupation_category="mason", daily_income=450, documents=FULL_DOCS),

    persona("driver_not_a_trade", "auto driver is not one of the 18 trades",
            ["pm_sym", "e_shram", "pmjjby", "pmsby", "atal_pension", "mudra_shishu"],
            age=30, occupation_category="driver", daily_income=500, documents=FULL_DOCS),

    persona("farmer_not_a_trade", "farming is outside the Vishwakarma list",
            ["pm_sym", "e_shram", "pmjjby", "pmsby", "atal_pension", "mudra_shishu"],
            age=30, occupation_category="farmer", daily_income=300, documents=FULL_DOCS),

    persona("domestic_worker", "not a listed trade, otherwise fully eligible",
            ["pm_sym", "e_shram", "pmjjby", "pmsby", "atal_pension", "mudra_shishu"],
            age=32, occupation_category="domestic_worker", daily_income=250,
            gender="female", documents=FULL_DOCS),

    persona("construction_worker", "not a listed trade",
            ["pm_sym", "e_shram", "pmjjby", "pmsby", "atal_pension", "mudra_shishu"],
            age=28, occupation_category="construction", daily_income=450, documents=FULL_DOCS),

    persona("cobbler_no_aadhaar", "trade qualifies but Vishwakarma needs aadhaar",
            ["pmjjby", "pmsby", "mudra_shishu"],
            age=45, occupation_category="cobbler", daily_income=300,
            documents=["bank_account"]),
]

# -- combined edges ----------------------------------------------------------
P += [
    persona("vendor_41_full_papers", "aged out of pensions, still a SVANidhi case",
            ["pm_svanidhi", "e_shram", "pmjjby", "pmsby", "mudra_shishu"],
            age=41, occupation_category="street_vendor", daily_income=500,
            documents=VENDOR_DOCS, vending_since_year=2018),

    persona("vendor_60_full_papers", "past e-Shram, SVANidhi has no upper age",
            ["pm_svanidhi", "pmsby", "mudra_shishu"],
            age=60, occupation_category="street_vendor", daily_income=500,
            documents=VENDOR_DOCS, vending_since_year=2018),

    persona("goldsmith_taxpayer", "trade qualifies, taxpayer exclusions bite",
            ["pm_vishwakarma", "pmjjby", "pmsby", "mudra_shishu"],
            age=38, occupation_category="goldsmith", monthly_income=45000,
            documents=FULL_DOCS, is_income_tax_payer=True),

    persona("washerman_epfo", "formal employment excludes PM-SYM and e-Shram",
            ["pm_vishwakarma", "pmjjby", "pmsby", "atal_pension", "mudra_shishu"],
            age=30, occupation_category="washerman", monthly_income=14000,
            documents=FULL_DOCS, is_epfo_esic_member=True),

    persona("blacksmith_everything", "every scheme he can possibly hold",
            ["pm_vishwakarma", "pm_sym", "e_shram", "pmjjby", "pmsby", "atal_pension", "mudra_shishu"],
            age=25, occupation_category="blacksmith", daily_income=400,
            documents=["aadhaar", "bank_account", "pan", "ration_card"]),

    persona("carpenter_59_boundary", "last year of e-Shram eligibility",
            ["pm_vishwakarma", "e_shram", "pmsby", "mudra_shishu"],
            age=59, occupation_category="carpenter", daily_income=500, documents=FULL_DOCS),
]

# ---------------------------------------------------------------------------
# Partial-information personas.
#
# The set above fully specifies every gating field, which measures the rule
# engine but never exercises the three-valued behaviour -- and that behaviour is
# the whole point. These callers have said something real but not everything,
# which is what an actual first turn looks like.
#
# The assertion is deliberately one-sided: NOTHING may be rejected on a fact we
# do not have. A scheme may come back ELIGIBLE or NEED_INFO; it may not come
# back NOT_ELIGIBLE unless a stated fact actually rules it out.
# ---------------------------------------------------------------------------

def partial(pid, note, may_reject, **profile):
    """`may_reject` lists schemes a STATED fact legitimately rules out."""
    return {"id": pid, "note": note, "profile": profile, "may_reject": sorted(may_reject)}


PARTIAL = [
    partial("said_nothing", "opened the app and tapped the mic", []),

    partial("only_occupation", "'main sabzi bechta hoon' -- not a Vishwakarma trade",
            ["pm_vishwakarma"],
            occupation_category="street_vendor", documents=[]),

    partial("only_age", "'main tees saal ka hoon'", [],
            age=30, documents=[]),

    partial("occupation_and_income", "'sabzi bechta hoon, roz paanch sau'",
            ["pm_vishwakarma"],
            occupation_category="street_vendor", daily_income=500, documents=[]),

    partial("age_rules_out_pensions", "72 -- age legitimately closes several doors",
            ["pm_sym", "atal_pension", "e_shram", "pmjjby", "pmsby"],
            age=72, documents=[]),

    partial("high_income_stated", "40,000/month rules out the PM-SYM ceiling",
            ["pm_sym"],
            monthly_income=40000, documents=[]),

    partial("taxpayer_stated", "being a taxpayer is a stated disqualification",
            ["pm_sym", "e_shram", "atal_pension"],
            is_income_tax_payer=True, documents=[]),

    partial("vendor_mid_conversation", "trade, age and aadhaar known; paperwork not yet asked",
            ["pm_vishwakarma"],
            age=30, occupation_category="street_vendor", documents=["aadhaar"]),

    partial("artisan_partial", "tailor, 35, no paperwork discussed -- not a street vendor",
            ["pm_svanidhi"],
            age=35, occupation_category="tailor", documents=[]),

    partial("income_only_daily", "'roz teen sau' and nothing else", [],
            daily_income=300, documents=[]),
]

OUT = Path(__file__).resolve().parent / "personas.json"
OUT.write_text(
    json.dumps({"personas": P, "partial": PARTIAL}, indent=2) + "\n", encoding="utf-8"
)
print(f"wrote {len(P)} full + {len(PARTIAL)} partial personas to {OUT.name}")
