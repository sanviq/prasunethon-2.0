# Setu — voice-first government scheme discovery

**Setu walks an informal-sector worker from what they said, in their own
language, to the government schemes they qualify for and the exact next step to
take — over a smartphone browser or a plain feature phone.**

Built for **Prasunethon 2.0** (21–23 August 2026).

---

## The problem

Only 15% of Indian adults access formal credit against a 24% global average.
89% hold a bank account but 16% of those sit inactive. The MSME credit gap is
₹25 lakh crore, and just 14% of 63 million MSMEs have formal credit access
despite employing 90% of the workforce.

The root cause is not ineligibility. It is that discovery and application
require literacy, English, or a smartphone. Setu removes all three
requirements.

---

## What it does

**Scheme discovery by voice.** Speak a sentence in your own language. Setu
extracts a structured profile, matches it against the scheme catalogue, and
answers out loud in the language you used.

**The Ladder.** When you do not qualify, Setu does not stop at "no". It runs a
counterfactual search over the failing rules and returns the cheapest, fastest
ordered route to qualifying, costed in rupees and days:

```
Request a Letter of Recommendation from your Town Vending Committee
  — Rs 0 — 7 days — unlocks Rs 10,000
```

Ordering respects dependencies: you cannot open a Jan Dhan account before you
hold Aadhaar, so the ladder never hands you a list you cannot follow in order.
Every path is verified by replaying it against the rule engine before it is
shown — a route that does not actually work is worse than admitting there is
none, because the person spends real days finding out.

**Missed-call callback — the zero-cost funnel.** No smartphone, no data plan,
no app, no literacy required. The user gives a missed call to the Setu number.
Setu hangs up, calls back at its own cost, and runs the entire flow over IVR:
language selection on the keypad, spoken question, spoken answer with the
scheme match and the next step. This is the channel that reaches the users who
need it most, and it costs the user nothing.

**"Why?" citations.** Every verdict points at the government clause it came
from — document, page, and the quoted sentence. Users and auditors verify
decisions rather than trusting them.

---

## The design commitment

> **The LLM translates. A deterministic rule engine decides.**

The LLM's job is narrow: turn speech into structured JSON, and turn the rule
engine's output into a plain spoken sentence. It never decides eligibility.

That decision is what makes everything else possible:

- **No hallucination surface on the verdict.** The decision is Python over
  typed rules.
- **The Ladder exists at all.** You can invert a rule; you cannot invert a
  vibe.
- **Every answer is citable**, because rules carry their source text.
- **Marginal cost per match is $0.001–$0.01** — one ASR pass and one JSON
  extraction. Everything after that is deterministic Python.

### Three-valued, on purpose

```
ELIGIBLE      every gating rule passed
NOT_ELIGIBLE  a gating rule failed on a fact we actually know
NEED_INFO     nothing failed, but a fact we need is missing
```

A missing fact is never a failure. Telling a vendor she does not qualify
because she never mentioned her age is precisely the behaviour this product
exists to replace.

---

## Architecture

```
voice in ──┬─ browser mic (PWA)
           └─ Twilio missed call → callback → IVR
                    │
              faster-whisper ASR   (local, 8 Indian languages, auto-detect)
                    │
              LLM adapter          (speech → structured profile JSON)
                    │
              rule engine          (deterministic, three-valued)  ← decides
                    │
              the Ladder           (counterfactual path search, ₹/days)
                    │
              LLM adapter          (verdict → plain spoken sentence)
                    │
              edge-tts             (free, no key, 8 voices)
                    │
           ┌────────┴────────┐
      audio to PWA      audio over Twilio
```

Both the ASR and LLM layers sit behind single-function adapters, so swapping a
provider is one function, not a rewrite.

### Languages

Hindi, Marathi, English, Kannada, Tamil, Telugu, Bengali, Gujarati. Whisper
auto-detects; edge-tts supplies a neural voice per language at no cost and with
no API key.

---

## Repository layout

```
data/schemes.json    the scheme catalogue — this file decides who is eligible
setu/rules.py        the eligibility engine (no LLM, by design)
setu/ladder.py       counterfactual path search
tests/               18 tests, and the invariants they protect
```

### Adding a scheme

Add an entry to `data/schemes.json`. Every rule needs:

- `field` / `op` / `value` — the predicate
- `source_doc` / `source_url` / `source_quote` — the citation the "Why?" panel
  shows. A rule you cannot cite is a rule you cannot defend, so the quote must
  be checkable against the URL rather than paraphrased from memory.
- `remedy` — how a failing rule gets fixed, or `null` if it cannot be. The
  Ladder can only offer what is written here. Age and income carry no remedy on
  purpose: you cannot act your way out of being 43.
- `depends_on` — rule ids whose remedies must be completed first.

`tests/test_rules.py` validates all four on every scheme, so a typo fails CI
rather than a live demo.

---

## Running it

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m pytest tests/ -q
```

---

## Provenance

Setu began as an idea at IBM HackVerse 2.0 with
[Palak Patnaik](https://github.com/pxlkele), who consented to the concept being
carried forward here. **This repository is a fresh implementation written
during the Prasunethon 2.0 build period** — every commit is dated within the
event, and no code is carried over from the earlier project. The scheme
catalogue, rule engine, and Ladder in this repo were rewritten from scratch.

## Roadmap

Retrieval over the full scheme corpus for free-form questions beyond the rule
set; Document Doctor for name-mismatch detection before an application fails
silently; Voice Ledger producing a bank-readable cash-flow PDF from spoken
daily takings.
