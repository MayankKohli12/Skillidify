# Skillidify — Implementation Plan & Model Requirements

**Team:** Leonardo Da VSCoders
**Problem statement:** Intelligent Talent and Workforce Ecosystem
**Event:** Build For Bharat 2.0 (36-hour hackathon)

This document turns the pitch deck's claims into a concrete build plan: what gets
built, in what order, with which models and libraries, and how each claim on the
slides gets checked rather than assumed.

## Table of contents

1. [Team & roles](#1-team--roles)
2. [System architecture](#2-system-architecture)
3. [Data requirements & sources](#3-data-requirements--sources)
4. [Model & tooling requirements](#4-model--tooling-requirements)
5. [Build phases](#5-build-phases)
6. [Validation plan](#6-validation-plan)
7. [Known limits & mitigations](#7-known-limits--mitigations)
8. [Post-hackathon roadmap](#8-post-hackathon-roadmap)
9. [Requirements quick-reference](#9-requirements-quick-reference)

---

## 1. Team & roles

| Name | Roll number | Suggested track | Owns |
|---|---|---|---|
| | | Data & verified-expectations rule | Job-posting ingestion, deduplication, employer-count threshold, ESCO alias list |
| | | Syllabus & depth mapper | PDF parsing, embedding-based skill matching, Know/Apply/Build tagging |
| | | Trend & gap scoring | Survey harmonization, trend fit, gap-score weighting, backtest |
| | | Challenge factory | LLM generation, mutation testing, seed variation, sandbox grading |
| | | Product & dashboard | Streamlit app, department dashboard, student challenge UI, pitch assets |

Team roster (fill in the track column above): Mayank Kohli (26BAI70414),
Atharvashreshtha Shukla (26BCS10848), Gourav Choudhury (26BCS10218), Kumar Sourish
(26BAI10170), Atul Singh (26BCS10379).

Each track is close to independent after the first two hours — the only shared
contract between them is the JSON schemas in section 2, so agree on those before
anyone starts writing pipeline code.

---

## 2. System architecture

Matches the five-step loop on slide 4, plus the two-engine split on slide 8.

```
                    ┌─────────────────────────────────────────┐
                    │              SHARED SCHEMAS              │
                    │  skills.json · syllabus.json · jobs.json │
                    └─────────────────────────────────────────┘
                                      │
   ┌───────────────┐   ┌───────────────┐   ┌──────────────────┐
   │ 1. Ingest &    │──▶│ 2. Map the    │──▶│ 3. Generate       │
   │    verify      │   │    syllabus   │   │    challenges     │
   │ (job postings) │   │ (skill+depth) │   │ (gap skills only) │
   └───────────────┘   └───────────────┘   └──────────────────┘
                                                      │
                                                      ▼
                                           ┌──────────────────┐
                                           │ 4. Verify & grade │
                                           │ (mutation-test,   │
                                           │  sandbox)         │
                                           └──────────────────┘
                                                      │
                                                      ▼
                                           ┌──────────────────┐
                                           │ 5. Compare gaps   │
                                           │ (inferred vs      │
                                           │  measured)        │
                                           └──────────────────┘
                                                      │
                                 (measured results feed back into step 3)
```

**Gap engine** = steps 1, 2, 3(ranking), 5 — the analytics side.
**Challenge factory** = steps 3(generation), 4 — the proof side.

### Shared data contracts

Agree on these three JSON shapes in the first hour, before any pipeline code is
written, so every track can build against a fixture file instead of waiting on
another track's output:

```jsonc
// skills.json — one row per matched skill
{
  "skill_id": "esco:1234",
  "skill_name": "Machine Learning",
  "source": "syllabus | job_ad",
  "source_text": "definitions of supervised and unsupervised learning",
  "depth": "know | apply | build",
  "confidence": 0.82
}

// employer_verified.json — one row per skill, from the job-posting pipeline
{
  "skill_id": "esco:1234",
  "distinct_employers": 14,
  "demand_depth": "apply | build",
  "trend": "rising | steady | fading",
  "trend_2030_estimate": 90,
  "trend_confidence_band": [78, 97]
}

// challenge.json — one row per generated+verified challenge
{
  "challenge_id": "chg_0091",
  "skill_id": "esco:1234",
  "depth": "apply",
  "prompt": "Clean the orders table...",
  "starter_code": "def revenue_by_city(orders): ...",
  "reference_solution": "...",
  "tests": ["..."],
  "seed_count": 5,
  "mutation_pass_rate": 1.0,
  "verified": true
}
```

---

## 3. Data requirements & sources

Directly from slide 7's table, with acquisition notes.

| Source | Used for | How to get it | Note |
|---|---|---|---|
| **ESCO skills dictionary** | Shared skill vocabulary | Free CSV download at `esco.ec.europa.eu/en/use-esco/download` | EU-oriented; add an alias list for newer tools it doesn't cover |
| **Stack Overflow Developer Survey, 2018–2025** | Direction of each skill over time | CSVs at `survey.stackoverflow.co` | Filter to India; column names shift between years, budget time to harmonize |
| **Indian fresher job postings** | What employers ask for today | Kaggle datasets (Glassdoor/LinkedIn/AmbitionBox India sets) as a free starting sample; Apify scrapers (Naukri, Hirist) if more volume is needed | Check each dataset's licence; salary fields are often sparse |
| **University syllabi (PDF)** | What is taught, and at what depth | Your own college's syllabus plus 2 public syllabi for contrast | Collect before the hackathon starts — this is on the critical path |
| **AISHE / NIRF (government open data)** | Supply-side context by discipline and region | Government open-data portals | Sanity-check only, not core to the loop — don't block on this |

**Pre-hackathon task:** download all five sources and commit them to the repo (or a
shared drive) before hour 0, so no one burns hackathon hours on dataset hunting.

---

## 4. Model & tooling requirements

Everything below is a pretrained model, a classical statistical method, or a rule
system — nothing here requires training a model from scratch, which is correct for
a 36-hour build.

### 4.1 Skill mapping (syllabus text ↔ job-ad text ↔ ESCO skill)

- **Model:** a small sentence-embedding model, e.g. `all-MiniLM-L6-v2` via the
  `sentence-transformers` library. Runs on CPU, ~80 MB, fast enough to embed every
  ESCO skill description once and cache it.
- **Method:** embed the syllabus topic / job-ad phrase, cosine-similarity against
  the cached ESCO embeddings, take the top match (and top-3 for the accuracy
  check in section 6).

### 4.2 Depth tagging (Know / Apply / Build)

- **Primary approach — rule-based, not a model:** a verb lexicon mapping common
  verbs to a level (`define / list / describe` → Know, `implement / use /
  calculate` → Apply, `design / build / deploy` → Build). Extract the governing
  verb with a lightweight POS tagger (`spaCy`, model `en_core_web_sm`) and look it
  up in the lexicon.
- **Fallback for ambiguous phrasing:** a few-shot prompt to an LLM API, asking it
  to return just `know | apply | build` for a given sentence. Use this only where
  the rule-based lexicon has no match, to keep API calls and latency down.

### 4.3 Employer-verified skill extraction

- **Method:** match each job posting's text against the ESCO skill list (same
  embedding approach as 4.1, or faster fuzzy string matching via `rapidfuzz` for
  exact/near-exact tool names). Count **distinct employers** per matched skill, not
  raw posting count, so one repeated listing can't inflate a skill.
- **Deduplication:** near-duplicate postings (reposts, near-identical text) removed
  before counting — e.g. via `rapidfuzz` similarity on the posting body, or a
  cheap MinHash/shingling check if posting volume is large.

### 4.4 Trend forecasting

- **Model:** none needed beyond classical regression. Fit a linear (or
  log-linear, if the shape warrants it) trend per skill on the 8 yearly survey
  points, via `numpy.polyfit` or `scikit-learn LinearRegression`.
- **Classification:** slope sign + magnitude against a threshold →
  rising / steady / fading.
- **Confidence range:** residual-based or bootstrap interval around the 2030
  extrapolation — report a range, never a single number (matches slide 6's stated
  claim).
- **Backtest (required, see section 6):** fit on 2018–2022, check the predicted
  direction against the real 2023–2025 values.

### 4.5 Challenge generation

- **Model:** an LLM API call (provider-agnostic — use whichever of
  Claude / GPT-4o-class / Gemini your team has API access to). Not something to
  self-host in 36 hours.
- **Method:** a structured prompt requesting JSON output only — problem
  statement, starter code, reference solution, test cases, and a skill tag — so
  the response can be parsed directly into `challenge.json`'s shape without manual
  cleanup. Ask for JSON-mode / structured-output if the provider supports it, to
  avoid fragile string parsing.

### 4.6 Challenge verification (mutation testing)

- **Not a model — a code tool.** Steps, matching slide 9:
  1. Run the reference solution against the generated tests — must pass all.
  2. Apply a small set of hand-written mutation operators to the reference
     (flip a comparison operator, swap a `groupby` column, off-by-one a slice)
     via Python's `ast` module, or simpler regex-based source mutation if time is
     short.
  3. Run each mutant against the same tests — **reject the challenge** if any
     mutant still passes everything (the test suite is too weak to catch it).
  4. Re-run the whole check across 3–5 random data seeds, so no test can be
     hard-coded to one specific answer.
  - The `mutmut` library can do steps 2–3 off the shelf if there's time to wire
    it in; hand-rolled mutations are a reasonable fallback under time pressure.

### 4.7 Sandbox execution (student code grading)

- **Not a model — an execution environment.**
- **Hackathon-grade approach:** Python `subprocess.run(..., timeout=5)` combined
  with the `resource` module (`RLIMIT_CPU`, `RLIMIT_AS`) for CPU/memory caps, and
  no network access granted to the subprocess.
- **Better, if time allows:** Docker with `--network none --memory 256m --cpus 0.5`
  per run, closer to real isolation.
- **State this limit explicitly in the pitch** (already on slide 13): this is
  hackathon-grade testing, not production security.

### 4.8 Syllabus parsing (PDF → structured topics)

- **Tooling:** `pdfplumber` or `PyMuPDF` (`fitz`) for text extraction.
- **Structuring:** regex/heuristic parsing for well-formatted syllabi (unit
  numbers, topic lines, hour counts); an LLM fallback (ask for JSON output of
  `{unit, topic, hours}`) for messier PDFs that don't follow a clean pattern.

### 4.9 Application layer

- **Framework:** Streamlit — fastest path to a working UI in Python, matches the
  department-dashboard and student-challenge mockups on slide 11.
- **Storage:** SQLite or flat JSON files are sufficient at hackathon scale; no
  need for a database server.
- **Two views, one app:** a multi-page Streamlit app (department dashboard +
  student challenge + evidence card) reading from the same underlying
  `skills.json` / `challenge.json` files.

---

## 5. Build phases

A practical breakdown — not for the pitch deck, but for the team's own tracking.

### Before hour 0 (pre-hackathon)

- [ ] Collect and commit all 5 data sources (section 3)
- [ ] Download/cache the embedding model (`all-MiniLM-L6-v2`) and spaCy model
      locally, so hour 0 isn't spent on downloads
- [ ] Confirm LLM API access and quota for the challenge-generation track
- [ ] Agree on the 3 JSON schemas in section 2
- [ ] Hand-write 2 "gold" challenges (one easy, one with a deliberate weak-test
      trap) to test the mutation-verifier against before trusting it on
      LLM-generated ones

### Hours 0–2: setup

- Repo scaffolding, shared schema fixtures, environment setup per track

### Hours 2–12: core build (parallel, independent tracks)

- Each track builds against the shared JSON fixtures, not against each other's
  live code

### Hour 12: first end-to-end run

- Wire the tracks together on one real syllabus; expect it to be rough

### Hours 12–26: refine & validate

- Run the validation checks in section 6
- Fix whatever the first end-to-end run exposed

### Hour 26: feature freeze

### Hours 26–33: polish, dashboard, pitch

### Hours 33–36: buffer, demo recording, submission

---

## 6. Validation plan

Directly from slide 12 — each check has an owner and a concrete method, not just a
metric name.

| Check | Method | Reported as |
|---|---|---|
| Skill mapping | Hand-label ~50 syllabus topics with their correct skill | Top-1 and top-3 accuracy |
| Depth tagging | Hand-label ~50 job-ad sentences as Know/Apply/Build | Agreement rate vs the rule-based tagger |
| Trend calls | Train on 2018–2022, predict direction for 2023–2025 | Directional accuracy |
| Challenge quality | Reference passes; mutants fail; stable across seeds | Share of generated challenges that pass verification |
| Real-world signal | Pilot: 15–20 volunteer students attempt verified challenges | Inferred vs measured gap, per skill |

**Report every number as measured**, including disappointing ones. If the pilot
gets fewer volunteers than planned, label the result a prototype signal, not
evidence — don't let the slide imply more rigor than the sample supports.

---

## 7. Known limits & mitigations

From slide 13, with the engineering decision behind each:

| Limit (stated on slide 13) | What we actually do about it |
|---|---|
| Skill matching is imperfect | Report top-1/top-3 accuracy from section 6; don't hide it |
| Forecasts are directional, not exact | Always display a range (section 4.4), never a bare number |
| AI tools can solve take-home challenges | Proof is strongest in a supervised/proctored session; state this as the intended deployment condition, not an afterthought |
| Sandbox is hackathon-grade | Named explicitly in section 4.7; Docker isolation is the stated upgrade path |

---

## 8. Post-hackathon roadmap

From slide 13's roadmap column, restated as concrete next engineering steps:

1. **Monthly job-data snapshots** — a scheduled scrape (cron job) to build a
   proprietary time series, instead of relying on one hackathon-weekend snapshot.
   This is also the fix for "how does the system avoid going stale itself?" — the
   same discipline it asks of a university syllabus.
2. **Swap-plan optimizer** — an integer program (`PuLP` or `OR-Tools`) that
   proposes which syllabus topics to drop/add within a fixed hour budget.
3. **Branching beyond tech** — extend the skill taxonomy and depth-tagging lexicon
   to non-tech disciplines and regional-language syllabi.
4. **Faculty-readiness scoring** — a friction score (1–3) per candidate topic,
   flagging ones that need new labs or faculty upskilling before being added to
   the swap-plan optimizer's recommendations.

---

## 9. Requirements quick-reference

```txt
# Core
pandas
numpy
scikit-learn
scipy

# Skill mapping
sentence-transformers
spacy
# then: python -m spacy download en_core_web_sm

# Fuzzy matching / dedup
rapidfuzz

# PDF parsing
pdfplumber
pymupdf

# Mutation testing (optional — hand-rolled mutations also work)
mutmut
pytest

# App
streamlit

# LLM access — pick whichever your team has API credits for
openai            # or
anthropic         # or
google-generativeai
```

No GPU, no custom training, no production database — everything here runs on a
laptop CPU within the hackathon window.