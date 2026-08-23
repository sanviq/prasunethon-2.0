# Setu — voice-first government scheme discovery

**Setu walks an informal-sector worker from what they said, in their own
language, to the government schemes they qualify for and the exact next step to
take — over a smartphone browser or a plain feature phone.**

Built for **Prasunethon 2.0** (21–23 August 2026).

---

## ▶ Live demo

# **[setu-ubco.onrender.com](https://setu-ubco.onrender.com)**

Open it on a phone, pick a language, hold the green button and say what you do
for a living. Also there: the [operator
console](https://setu-ubco.onrender.com/console.html) and the [technical
documentation](https://setu-ubco.onrender.com/technical.html).

> ### Please read before judging
>
> **The first request may take 30–60 seconds.** This runs on a free tier that
> puts the container to sleep after 15 minutes without visitors, and the first
> request has to wake it. Every request after that is normal speed — around
> **3–9 seconds** for a complete turn. If the page seems to hang on the first
> try, it is waking up, not broken.
>
> **The demo offers three languages — Hindi, Marathi and English — not eight.**
> That is deliberate and explained in [Languages](#languages) below.
>
> Neither is a limit of the design. Both are the honest state of a system built
> in a 36-hour window on free infrastructure, and we would rather say so than
> have you discover it.

**[Technical documentation →](web/technical.html)** — architecture, the rule
schema, the Ladder's search and verification, the evaluation method, and the
operational design. Served by the app at `/technical.html`.

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

**"Why?" citations.** Every verdict points at the government clause it came
from — document, URL, and the quoted sentence. Users and auditors verify
decisions rather than trusting them.

When the conversation is in Hindi or Marathi the clause is shown translated,
with the government's exact published sentence kept underneath it and labelled
as the original. A translated clause reads better, but it is no longer what the
government wrote, and a citation you have quietly paraphrased has stopped being
a citation.

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
voice in ─── browser mic (PWA)
                    │
              faster-whisper ASR   (local, auto-detects the language)
                    │
              LLM adapter          (speech → structured profile JSON)
                    │
              rule engine          (deterministic, three-valued)  ← decides
                    │
              the Ladder           (counterfactual path search, ₹/days)
                    │
              LLM adapter          (verdict → plain spoken sentence)
                    │
              edge-tts             (free, no key, neural voice per language)
                    │
              audio back to the PWA
```

Both the ASR and LLM layers sit behind single-function adapters, so swapping a
provider is one function, not a rewrite.

### The LLM layer

Gemini 3.5 Flash Lite, on the free tier. Extraction runs at `temperature=0`
against a `response_schema`, so the JSON that reaches the rule engine is
structurally guaranteed rather than hoped for.

The model was chosen by measurement, not preference — `eval/pick_model.py`
times the candidates against the same extraction prompt. `gemini-2.5-flash`
returns 404 for new API keys, and of the models that do answer, this one
matched the accuracy of the larger ones at 1.3s against 10.3s. The free tier
also rate-limits at 15 requests/minute, so every call carries a three-attempt
retry with linear backoff; a judge running the demo twice in a row is exactly
the burst that would otherwise fail in front of them.

Every call is cached on a content hash. Conference wifi is unreliable and
re-calling a model for a sentence we already have is a needless risk on stage,
so `SETU_LLM_MODE=offline` refuses network calls entirely and serves only from
cache. Run the demo that way.

The extraction prompt's most important instruction is to leave unstated fields
null. A guessed age becomes a wrong verdict the person never gets to contest,
so silence must never become a fact.

### Languages

**Three are offered. Eight are built. The gap is deliberate.**

| | Hindi · Marathi · English | Kannada · Tamil · Telugu · Bengali · Gujarati |
|---|---|---|
| Neural TTS voice | ✅ | ✅ |
| ASR | ✅ | ✅ |
| Pipeline support | ✅ | ✅ |
| In the language picker | ✅ | ❌ |
| **Tested end to end with a speaker** | ✅ | ❌ |

Nothing in the pipeline is language-specific — `setu/voice.py` carries a neural
voice for all eight and the rest of the system never asks what language it is
handling. Turning the other five on is one line in `web/index.html`.

We have not turned them on because **nobody has held a conversation with Setu in
Tamil or Bengali.** We do not speak them, and there was no time in a 36-hour
window to sit with someone who does. A language that half-works — a
mistranscribed number, a stilted machine voice, a scheme card in the wrong
script — is worse for the person in front of it than a language that was never
offered, because it wastes their time while looking like it worked.

So the picker is shorter than the capability, on purpose. It is a claim we can
defend rather than a feature count.

**Where the other five are going.** They exist for the missed-call channel (see
[Designed, not yet built](#designed-not-yet-built)), which is the part of this
that reaches people without a smartphone at all. That channel is where language
coverage actually matters: a Tamil-speaking construction worker with a feature
phone is precisely who Setu is for, and precisely who a browser demo cannot
serve. The voices are in place so that stage is a test-and-enable job rather
than a rebuild.

Scheme cards are translated into the conversation's language too, so a Hindi
caller does not get a Hindi answer attached to an English card. The government's
original wording is kept alongside the translation rather than replaced — see
*Citations* below.

---

## Repository layout

```
data/schemes.json    the scheme catalogue — this file decides who is eligible
setu/rules.py        the eligibility engine (no LLM, by design)
setu/ladder.py       counterfactual path search
setu/llm.py          Gemini adapter (extraction + narration)
setu/voice.py        faster-whisper ASR, edge-tts TTS
setu/api.py          HTTP layer
web/index.html       the PWA, single file, no build step
web/console.html     operator console — live callers, benefits, catalogue, eval
web/technical.html   the technical documentation, served at /technical.html
eval/                53 personas and the scoring harness
scripts/prewarm.py   fills both caches before the demo
tests/               104 tests, and the invariants they protect
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

cp .env.example .env        # add your Gemini key
./.venv/bin/uvicorn setu.api:app --reload --port 8000
```

Open `http://localhost:8000` — the PWA is served from the same origin as the
API, so there is one URL and no CORS in the demo path.

**The microphone needs HTTPS.** `getUserMedia` refuses `http://` on anything
but localhost, which is why the phone demo runs behind a tunnel:

```bash
ngrok http 8000     # then open the https:// URL on the phone
```

Before demoing, warm both caches over wifi you trust and then pin them shut:

```bash
./.venv/bin/python scripts/prewarm.py
```

That walks whole conversations through the live pipeline, translates all 80
catalogue strings into each language, and then verifies the result by
re-running everything with the network refused. A cache that has not been
checked in offline mode is a cache you are guessing about.

```bash
SETU_LLM_MODE=offline SETU_VOICE_MODE=offline ./.venv/bin/uvicorn setu.api:app
```

Both adapters then refuse network calls and serve only from cache, so the demo
replays identically and conference wifi stops being a dependency.

### Endpoints

| | |
|---|---|
| `POST /ask` | audio in, spoken answer out |
| `POST /ask/text` | same pipeline, typed — no mic, no tunnel, no Whisper |
| `GET /schemes` | the catalogue with its sources |
| `GET /health` | scheme count and supported languages |
| `GET /sessions` | live conversations, for the operator console |
| `GET /eval` | the persona scores, served to the console |
| `DELETE /session/{id}` | reset between demo runs |

The operator console is at `/console.html` — live callers, benefit totals, the
catalogue with its sources, and the eval report on one screen.

`/ask/text` exists because when something breaks twenty minutes before a demo,
you want to know whether it is the microphone, the model, or the rules — and
that endpoint answers the question in one curl.

---

## Deploying it

One container, one port. `Dockerfile` bakes the ASR model into the image, so a
cold start is a process start rather than a 464MB download.

### The constraint that decides the architecture

Local Whisper wants a real CPU. Free hosting gives you roughly a tenth of one,
which turns a five-second transcription into fifty. So the deployed image sets
`SETU_ASR=gemini` and the provider does ASR:

| | local `small` | `SETU_ASR=gemini` |
|---|---|---|
| transcription | 4.9s | **2.1s** |
| full mic turn | 12.3s | **3.4s** |
| image | ~2GB | ~150MB |
| build | ~10 min | under 2 min |
| accuracy | `मुमबाई … चोथटीस साल` | `मुंबई … 34 साल` |

The remote path being both faster and more accurate was not the expected
result — the local model was chosen partly on the assumption it would win on
quality.

**What it costs, stated plainly:** speech now leaves the machine, offline mode
covers only what is already cached, and a turn spends four provider calls
instead of three against a 15/minute free tier. Local remains the default in
`setu/voice.py` for laptops and for the IVR stage; only the container flips it.
This is the swap the adapter shape was built for — one function, and nothing
outside it changes.

### Render (free, no card)

`render.yaml` is a blueprint, so this is four clicks:

1. [dashboard.render.com](https://dashboard.render.com) → **New** → **Blueprint**
2. Connect the GitHub repo — it reads `render.yaml`
3. Paste `GEMINI_API_KEY` when prompted (stored as a secret, never in the repo)
4. **Apply**

Free web services **spin down after 15 minutes idle**, and the first request
after that takes 30–60s. The image is small, so that wake is a container start
rather than a model download.

### Anywhere else

Any host that reads `$PORT` works — Fly, Railway, Cloud Run:

```bash
docker build -t setu .
docker run -p 8000:8000 -e PORT=8000 -e GEMINI_API_KEY=... setu
```

**Not Vercel or Netlify.** Serverless functions are stateless, and the
conversation is not: `SESSIONS` holds the profile between "I sell vegetables"
and "I am 34". Their execution limits are also shorter than a turn.

**Hugging Face Spaces** no longer works for this on a free account — Docker and
Gradio SDKs now require PRO. Only Static Spaces are free, which cannot run a
Python service.

**After deploying, warm the deployed instance** — not your laptop. The cache
lives inside the container and nothing on your machine can reach into it, so
the warmer drives the deployed app through its own endpoint:

```bash
./.venv/bin/python scripts/prewarm.py --url https://your-space.hf.space
```

Run it twice. The second pass should report `0.0s` on every turn; if it does
not, the cache is not being written where you think it is. Translation is
cached per string, so this is what takes a turn from ~16s to under a second on
the scripted path.

One caveat worth planning around: **a container that scales to zero loses the
cache when it stops**, so this has to be re-run after every cold start. If you
are being judged live, keep one instance warm rather than trusting the timing.

**Never commit `.env`.** It is gitignored, `.env.example` ships an empty key,
and the key belongs in the host's secret store.

---

## Evaluation

```bash
./.venv/bin/python eval/run.py
```

53 personas — 43 with complete profiles, 10 mid-conversation — scored against
expectations **written by hand from the published government criteria**, never
from Setu's own output. An eval whose labels come from the system under test
measures nothing; it only proves the code is self-consistent.

```
43 personas x 8 schemes = 344 decisions

SCHEME                PREC  RECALL      F1
pm_svanidhi          1.000   1.000   1.000
pm_sym               1.000   1.000   1.000
e_shram              1.000   1.000   1.000
pmjjby               1.000   1.000   1.000
pmsby                1.000   1.000   1.000
pm_vishwakarma       1.000   1.000   1.000
atal_pension         1.000   1.000   1.000
mudra_shishu         1.000   1.000   1.000
MICRO                1.000   1.000   1.000

false rejections on missing facts: 0/80 (0.0%)
```

**A perfect score here is the expected result, not a boast.** The decision layer
is a lookup over typed predicates — it has no error surface to speak of. What
this eval actually proves is that the catalogue is transcribed correctly from
the source documents, which is where the real risk lives. It is a test of the
data, not a claim about a model.

The number worth arguing about is the second one. Precision and recall are
measured on complete profiles; **false rejections on partial information** is
measured on people who have said one sentence, which is the state they are
actually in when the system first has to behave well.

The two failure modes cost different things. A false positive sends someone to
a government office to be turned away — a day of wages and a bus fare. A false
negative silently withholds money they were entitled to, and they never find
out. The second is what this product exists to fix, which is why NEED_INFO is a
first-class verdict rather than a rounding error.

## Designed, not yet built

These are specified but **not implemented in this repository**. They are listed
here because the architecture accommodates them, not because they ship today.

### Missed-call callback — the zero-cost funnel

The channel that reaches the users who need Setu most: no smartphone, no data
plan, no app, no literacy. The user gives a missed call to the Setu number.
Setu hangs up, calls back at its own cost, and runs the same flow over IVR —
language selection on the keypad, spoken question, spoken answer carrying the
scheme match and the next step.

It costs the user nothing, which is the entire point: every rupee of friction
at the top of the funnel removes exactly the people the scheme was written for.

The pieces it needs, none of which are present here: a Twilio webhook to catch
the inbound call and trigger the callback, a keypad language menu, and an audio
bridge between the TTS output and the call leg. The core loop it would sit on
top of — ASR, extraction, rule engine, Ladder, TTS — is built and tested.

### Others

Retrieval over the full scheme corpus for free-form questions beyond the rule
set; Document Doctor for name-mismatch detection before an application fails
silently; Voice Ledger producing a bank-readable cash-flow PDF from spoken
daily takings.
