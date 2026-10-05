# Skillidify — Exact To-Do List

**Team:** Leonardo Da VSCoders · **Event:** Build For Bharat 2.0 (36-hour hackathon)
**Use this as:** a literal checklist, checked off day by day, then hour by hour.

Tracks referenced below (assign names in the implementation plan's role table):
- **T1 — Data:** job postings, deduplication, employer-verified rule
- **T2 — Syllabus:** PDF parsing, skill mapping, depth tagging
- **T3 — Trend:** survey harmonization, trend fit, gap scoring
- **T4 — Challenge factory:** LLM generation, mutation testing, sandbox
- **T5 — Product:** Streamlit app, dashboard, pitch assets

---

## Part 1 — 4 days out (before the hackathon starts)

### Day -4: Accounts & data

- [ ] **(All)** Create a Groq account at console.groq.com → generate API key → save in a
      shared `.env.example` (not the real key) so everyone's local setup matches
- [ ] **(All)** Create a Gemini API key at Google AI Studio (aistudio.google.com) as
      the fallback LLM — same reason, free, no card
- [ ] **(T1)** Download 1–2 Indian fresher job-posting datasets (Kaggle: Glassdoor
      India / LinkedIn / AmbitionBox sets) → commit to repo under `/data/raw/jobs/`
- [ ] **(T2)** Collect 3 syllabus PDFs: your own college's + 2 public ones → commit
      under `/data/raw/syllabi/`
- [ ] **(T3)** Download the Stack Overflow Developer Survey CSVs (2018–2025) from
      survey.stackoverflow.co → commit under `/data/raw/survey/`
- [ ] **(T1/T2/T3)** Download the ESCO skills CSV from esco.ec.europa.eu/en/use-esco/download
      → commit under `/data/raw/esco/`
- [ ] **(T5)** Create the GitHub repo, add everyone, set up the folder structure below

```
/data/raw/{jobs,syllabi,survey,esco}/
/data/processed/
/src/{ingest,mapping,trend,challenge,app}/
/tests/
```

### Day -3: Environment & individual setup

- [ ] **(All)** `pip install sentence-transformers spacy rapidfuzz pdfplumber pymupdf
      scikit-learn numpy pandas scipy mutmut pytest streamlit groq google-generativeai`
- [ ] **(All)** `python -m spacy download en_core_web_sm`
- [ ] **(All)** Run a one-line test script confirming `all-MiniLM-L6-v2` downloads
      and embeds a sentence successfully (do this now, not at hour 0 — it's a
      ~80MB download you don't want fighting venue wifi for)
- [ ] **(T1)** Write the job-posting cleaning script skeleton: load → dedupe (rapidfuzz)
      → filter to fresher-level (experience regex/keyword match)
- [ ] **(T2)** Write the PDF-to-text extraction skeleton (`pdfplumber`), test on
      all 3 collected syllabi, note which ones parse cleanly vs. need the LLM
      fallback
- [ ] **(T3)** Write the CSV-harmonization skeleton for the survey years — column
      names differ by year, map them to a single consistent schema
- [ ] **(T4)** Write and test one LLM call (Groq) that returns valid JSON for a
      single hardcoded prompt — confirm the JSON-mode / structured-output path
      actually works before relying on it
- [ ] **(T5)** Scaffold the Streamlit app with 2 empty pages (Department, Student)
      reading from placeholder JSON

### Day -2: Shared schemas & fixtures

- [ ] **(All, 30-min call)** Agree on the exact field names in `skills.json`,
      `employer_verified.json`, `challenge.json` (see the implementation plan for
      the starting shapes) — lock this before anyone builds against it
- [ ] **(Each track)** Create a hand-written fixture file matching the agreed
      schema, with 3–5 fake rows, committed to `/data/processed/` — this is what
      everyone else builds against until real pipelines produce real output
- [ ] **(T4)** Hand-write 2 "gold" challenges: one clean, one with a deliberate
      weak-test trap (a test that a wrong solution could still pass) — save both
      to `/tests/gold_challenges/` to validate the mutation-tester against before
      trusting it on LLM-generated challenges

### Day -1: Dry run & rehearsal prep

- [ ] **(T2)** Run the embedding-matching pipeline on one real syllabus against a
      small hand-picked ESCO subset — confirm matches look sane before hour 0
- [ ] **(T3)** Fit the trend model on 2018–2022 survey data for 3 known skills,
      sanity-check the direction against what you'd expect
- [ ] **(T4)** Run the mutation-tester against both gold challenges from Day -2 —
      confirm it correctly accepts the clean one and rejects the trap
- [ ] **(All)** Confirm the hackathon's rules on pre-written code — know exactly
      what's allowed to bring in vs. what must be written live
- [ ] **(All)** Pack: laptop chargers, a mobile hotspot as wifi backup, and a
      offline copy of all datasets/models (don't depend on venue internet for
      anything from Day -4)

---

## Part 2 — Hackathon day (36 hours)

### Hour 0–2: Setup

- [ ] **(All)** Pull the repo, confirm everyone's environment from Day -3 still works
- [ ] **(T5)** Confirm the shared JSON schemas one final time; create the fixture
      files in the actual repo if not already committed
- [ ] **(All)** Each track opens its own branch; no one blocks on another track
      until the Hour 12 integration point

### Hour 2–12: Core build (parallel)

- [ ] **(T1)** Finish ingest → dedupe → fresher-filter → employer-count pipeline;
      output real `employer_verified.json` from the real job-posting data
- [ ] **(T2)** Finish PDF → topics → embedding-match → depth-tag pipeline; output
      real `skills.json` from your own college's syllabus
- [ ] **(T3)** Finish trend fit + backtest on the real survey data; output trend
      direction + 2030 estimate per skill into `employer_verified.json`
- [ ] **(T4)** Finish generate → verify (mutation-test) → seed-vary pipeline;
      produce 5–10 real verified challenges into `challenge.json`
- [ ] **(T5)** Build the real department dashboard view (inferred vs. measured gap
      bars) and the real challenge-attempt view, both reading live from the
      other tracks' JSON outputs

### Hour 12: First end-to-end run — checkpoint

- [ ] **(All)** Stop and run the whole pipeline on one real syllabus, start to
      finish: syllabus in → gap list out → 1 challenge generated and verified →
      shown in the dashboard
- [ ] **(All)** It will be rough — that's expected. Write down what broke before
      moving on, don't fix silently

### Hour 12–26: Refine & validate

- [ ] **(T2)** Hand-label ~50 syllabus topics with correct skill; compute top-1 /
      top-3 accuracy (per the validation plan)
- [ ] **(T2)** Hand-label ~50 job-ad sentences as Know/Apply/Build; compute
      agreement rate against the rule-based tagger
- [ ] **(T3)** Confirm the 2023–2025 backtest direction accuracy; record the number
      even if it's not great
- [ ] **(T4)** Run the full challenge batch through verification; record the
      pass-rate percentage (target stated in the deck: ≥90%)
- [ ] **(T4)** If time allows, recruit 15–20 volunteer students for the pilot;
      have them attempt 2–3 verified challenges each
- [ ] **(T1)** Fix whatever the Hour 12 checkpoint exposed in the data pipeline
- [ ] **(T5)** Wire the real validation numbers into the dashboard/report, replacing
      any placeholder numbers

### Hour 26: Feature freeze

- [ ] **(All)** No new features after this point — bugs and polish only
- [ ] **(All)** Tag this commit in git as the freeze point, so there's a known-good
      fallback if later changes break something

### Hour 26–33: Polish, dashboard, pitch

- [ ] **(T5)** Final UI pass on both dashboard views
- [ ] **(All)** Build the pitch deck's data slides from the real numbers now
      available (mapping accuracy, backtest accuracy, challenge pass rate, pilot
      results if collected)
- [ ] **(All)** 2–3 full pitch rehearsals, timed
- [ ] **(All)** Prepare answers for the hard questions: "how do you stop students
      using AI to solve challenges," "is your pilot sample meaningful," "what
      happens when this data goes stale" (see the counter-argument prep from
      earlier in this conversation)

### Hour 33–36: Buffer & submission

- [ ] **(T5)** Record a backup demo video — don't depend on live wifi/APIs during
      the actual judging
- [ ] **(All)** Pre-cache every result the live demo needs, so nothing depends on
      an API call succeeding in front of judges
- [ ] **(All)** Final submission: repo, deck, demo video, README
- [ ] **(All)** Sleep, if any hours remain

---

## Definition of done, per checkpoint

| Checkpoint | Must be true before moving on |
|---|---|
| Hour 12 | One syllabus goes in, one gap list and one verified challenge come out, end to end |
| Hour 26 | All 5 validation numbers in section 6 of the implementation plan are real, not placeholders |
| Hour 33 | Pitch rehearsed at least twice, hard questions have rehearsed answers |
| Hour 36 | Demo works with zero live API dependency |