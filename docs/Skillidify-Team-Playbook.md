# Skillidify — Team Playbook

**How five people build one working system in 36 hours, using the GitHub repo.**

This is the full walkthrough — read it once, start to finish, as a team. After that,
use `Skillidify-To-Do-Checklist.md` for the day-of quick checklist and this doc for
"wait, how exactly does my track work / connect to everyone else's."

---

## 1. Repo setup (do this today, before anything else)

### 1.1 Folder structure

Create this structure in the repo now, with empty placeholder files where noted.
Having the skeleton in place before anyone writes real code is what lets all five
tracks start on Day -3 instead of waiting on each other.

```
skillidify/
├── README.md
├── requirements.txt
├── .env.example              # GROQ_API_KEY=, GEMINI_API_KEY= (no real keys committed)
├── data/
│   ├── raw/
│   │   ├── jobs/              <- T1 drops datasets here
│   │   ├── syllabi/           <- T2 drops PDFs here
│   │   ├── survey/            <- T3 drops Stack Overflow CSVs here
│   │   └── esco/              <- shared ESCO skills CSV
│   └── processed/             <- THE SHARED CONTRACT (see 1.3)
│       ├── skills.json
│       ├── employer_verified.json
│       ├── gap_list.json
│       └── challenge.json
├── src/
│   ├── ingest/                 <- T1's code
│   ├── mapping/                 <- T2's code
│   ├── trend/                   <- T3's code
│   ├── challenge/                <- T4's code
│   └── app/                      <- T5's code
├── tests/
│   ├── fixtures/               <- hand-written fake data, built Day -2
│   └── gold_challenges/        <- T4's 2 hand-written test challenges
└── docs/
    ├── Skillidify-Implementation-Plan.md
    ├── Skillidify-To-Do-Checklist.md
    └── Skillidify-Team-Playbook.md   (this file)
```

### 1.2 Branches

- **`main`** — always kept in a working, demo-able state. Nobody force-pushes to it.
- **One branch per track:** `track1-data`, `track2-syllabus`, `track3-trend`,
  `track4-challenge`, `track5-product`
- Each person works on their own track branch, commits freely, opens a PR into
  `main` whenever their piece reaches a working checkpoint (not just once at the end)

```bash
git clone <repo-url>
git checkout -b track1-data        # each person, their own track name
```

### 1.3 The shared contract — this is what makes parallel work possible

Everyone's code reads and writes these four files in `data/processed/`. **Nobody
hand-edits another track's output file.** If T3 needs something different from
T1's output shape, that's a 2-minute conversation to change the schema, not a
workaround written into T3's code.

```jsonc
// data/processed/skills.json — T2 writes this
[{
  "skill_id": "esco:1234",
  "skill_name": "Machine Learning",
  "source": "syllabus",
  "source_text": "definitions of supervised and unsupervised learning",
  "depth": "know",
  "confidence": 0.82
}]

// data/processed/employer_verified.json — T1 writes this
[{
  "skill_id": "esco:1234",
  "distinct_employers": 14,
  "demand_depth": "build"
}]

// data/processed/gap_list.json — T3 writes this (reads the two files above)
[{
  "skill_id": "esco:1234",
  "skill_name": "Machine Learning",
  "taught_depth": "know",
  "demand_depth": "build",
  "depth_gap": 2,
  "trend": "rising",
  "trend_2030_estimate": 90,
  "trend_confidence_band": [78, 97],
  "priority_score": 0.91
}]

// data/processed/challenge.json — T4 writes this (reads gap_list.json)
[{
  "challenge_id": "chg_0091",
  "skill_id": "esco:1234",
  "depth": "build",
  "prompt": "...",
  "starter_code": "...",
  "reference_solution": "...",
  "tests": ["..."],
  "mutation_pass_rate": 1.0,
  "verified": true
}]
```

**Day -2 task (from the to-do checklist):** each track writes 3–5 fake rows into
these files by hand before any pipeline exists. This is what unblocks T3 (needs
T1+T2's shape), T4 (needs T3's shape), and T5 (needs everyone's shape) from sitting
idle during Hours 2–12.

---

## 2. How the five tracks connect

```
   T1 (Data)  ──┐
                ├──▶  T3 (Trend) ──▶  T4 (Challenge factory)
   T2 (Syllabus)┘            │                  │
                             ▼                  ▼
                        gap_list.json     challenge.json
                             │                  │
                             └────────┬─────────┘
                                      ▼
                                 T5 (Product)
                         reads everything, shows it
```

- **T1 and T2 have no dependency on each other** — both can start immediately and
  work entirely in parallel.
- **T3 depends on T1 and T2's output shape** (not their finished pipelines — the
  Day -2 fixtures are enough to start).
- **T4 depends on T3's output shape.**
- **T5 depends on everyone** — but again, only on the *shape*, so it starts Hour 2
  against fixtures, not Hour 20 against real data.

This is why the fixture files matter more than anything else in the setup: they
turn a hard dependency chain into something every track can build against from
hour 0.

---

## 3. The five tracks, in full detail

### T1 — Data

**Goal:** turn raw, noisy job postings into a trustworthy "which skills do
employers actually want" signal.

**Reads:** `data/raw/jobs/*`, `data/raw/esco/*`
**Writes:** `data/processed/employer_verified.json`

**Steps, in order:**
1. Load the raw postings into a dataframe (`src/ingest/clean_postings.py`)
2. Remove near-duplicate postings (reposts, copy-pasted listings) using
   `rapidfuzz` similarity on the posting body text — anything above a similarity
   threshold (start at 90%) is treated as one posting
3. Filter to fresher-level roles only — regex/keyword match on experience text
   ("0-1 years", "fresher", "entry level", "0-2 yrs")
4. For each remaining posting, match its text against the ESCO skill list
   (embedding similarity, same model T2 uses) to extract which skills it mentions
5. Count **distinct employers** per skill (not posting count — one employer
   reposting doesn't count twice)
6. Apply the threshold: a skill only "counts" if ≥ k distinct employers ask for it
   (pick k after looking at the real distribution — don't hardcode it blind)
7. Write `employer_verified.json`

**Rough function shape:**
```python
# src/ingest/clean_postings.py
def load_postings(path) -> pd.DataFrame: ...
def dedupe_postings(df) -> pd.DataFrame: ...
def filter_fresher(df) -> pd.DataFrame: ...

# src/ingest/verified_skills.py
def extract_skills(text, esco_embeddings) -> list[str]: ...
def count_employers_per_skill(df) -> dict[str, int]: ...
def apply_threshold(counts, k) -> dict: ...
```

### T2 — Syllabus

**Goal:** turn a PDF syllabus into a structured "what's taught, at what depth" list.

**Reads:** `data/raw/syllabi/*`, `data/raw/esco/*`
**Writes:** `data/processed/skills.json`

**Steps, in order:**
1. Extract raw text from the PDF (`pdfplumber`, fallback to `pymupdf` if a PDF
   doesn't parse cleanly)
2. Split the text into unit/topic/hour chunks — regex for well-formatted syllabi;
   for messy ones, send the page text to an LLM and ask for JSON back
3. Embed each topic's text (`sentence-transformers`, `all-MiniLM-L6-v2`)
4. Pre-compute embeddings for every ESCO skill description once, cache them
5. For each topic, cosine-similarity match against the cached ESCO embeddings,
   keep the top match and its confidence score
6. Extract the governing verb from the topic's description (`spaCy` POS tagging)
7. Look the verb up in the depth lexicon → Know / Apply / Build
8. Write `skills.json`

**Rough function shape:**
```python
# src/mapping/parse_pdf.py
def extract_topics(pdf_path) -> list[dict]: ...  # {unit, topic, hours}

# src/mapping/embed_match.py
def build_esco_index(esco_csv) -> np.ndarray: ...
def match_skill(topic_text, esco_index) -> tuple[str, float]: ...

# src/mapping/depth_tag.py
VERB_LEXICON = {"define": "know", "implement": "apply", "design": "build", ...}
def tag_depth(text) -> str: ...
```

### T3 — Trend

**Goal:** decide which skills are rising/steady/fading, and merge T1 + T2's output
into one ranked gap list.

**Reads:** `data/raw/survey/*`, `data/processed/skills.json`,
`data/processed/employer_verified.json`
**Writes:** `data/processed/gap_list.json`

**Steps, in order:**
1. Load each year's survey CSV, harmonize column names into one consistent
   schema (they genuinely differ year to year — budget real time for this)
2. Filter to India-tagged respondents
3. For each skill, build an 8-point yearly usage series
4. Fit a trend (linear regression is enough) per skill
5. Classify direction from the slope: rising / steady / fading
6. Backtest: refit on 2018–2022 only, check whether the predicted direction
   matches the real 2023–2025 values — record the accuracy, whatever it is
7. Extrapolate to 2030, with a residual-based or bootstrap confidence range
8. Join with `skills.json` (taught depth) and `employer_verified.json` (demand +
   employer count) on `skill_id`
9. Compute `depth_gap` (demand depth minus taught depth) and a weighted
   `priority_score` combining gap size, employer count, and trend
10. Sort, write `gap_list.json`

**Rough function shape:**
```python
# src/trend/harmonize_survey.py
def load_and_harmonize(years: list[int]) -> pd.DataFrame: ...

# src/trend/fit_trend.py
def fit_trend(series: list[float]) -> dict: ...  # {direction, 2030_estimate, ci}
def backtest(series: list[float]) -> float: ...  # directional accuracy

# src/trend/rank_gaps.py
def rank_gaps(skills_json, employer_json, trend_by_skill) -> list[dict]: ...
```

### T4 — Challenge factory

**Goal:** generate coding challenges for gap skills, and verify each one before
it's trusted.

**Reads:** `data/processed/gap_list.json`
**Writes:** `data/processed/challenge.json`

**Steps, in order:**
1. For each high-priority gap skill, prompt an LLM (Groq first, Gemini fallback)
   for: problem statement, starter code, reference solution, tests, skill tag —
   JSON output only
2. Parse the response into the `challenge.json` shape
3. Run the reference solution against the generated tests — must pass all; if not,
   discard and regenerate
4. Apply hand-written mutation operators to the reference (flip a comparison,
   swap a groupby column, off-by-one a slice) — each mutant must **fail** at
   least one test
5. If any mutant still passes everything, the test suite is too weak — discard
   and regenerate (or ask the LLM to strengthen the tests)
6. Re-run the whole check across 3–5 random data seeds, so no test can be
   satisfied by a hardcoded answer
7. Build the grading sandbox: run submitted student code via `subprocess` with a
   timeout and memory cap, no network access
8. Write verified challenges to `challenge.json`

**Rough function shape:**
```python
# src/challenge/generate.py
def generate_challenge(skill_name, depth) -> dict: ...  # LLM call

# src/challenge/mutate_test.py
def run_reference(challenge) -> bool: ...
def run_mutants(challenge) -> bool: ...        # True if all mutants caught
def run_across_seeds(challenge, n=5) -> bool: ...

# src/challenge/sandbox.py
def run_student_code(code, tests, timeout=5) -> dict: ...  # {"passed": n, "total": m}
```

### T5 — Product

**Goal:** make everything visible — the department view and the student view —
and assemble the pitch.

**Reads:** `data/processed/gap_list.json`, `data/processed/challenge.json`
**Writes:** the Streamlit app, the final pitch deck

**Steps, in order:**
1. Build the department dashboard (`src/app/dashboard.py`): bar comparison of
   inferred vs. measured gap per skill, reads `gap_list.json`
2. Build the student view (`src/app/student_view.py`): pick a challenge from
   `challenge.json`, submit code, call T4's sandbox, show pass/fail
3. Build the evidence card: what the student has proven, next skill to learn
4. Once T2–T4 have real numbers (Hour 12+), wire the validation numbers (mapping
   accuracy, backtest accuracy, challenge pass rate) into a results view
5. Hour 26 onward: build the pitch deck slides from the real numbers, record the
   backup demo video

**Rough function shape:**
```python
# src/app/dashboard.py — streamlit, reads gap_list.json
# src/app/student_view.py — streamlit, reads challenge.json, calls
#     src.challenge.sandbox.run_student_code()
```

---

## 4. Git workflow — the exact mechanics

- **Commit often**, on your own track branch — there's no such thing as too small
  a commit during the build phase
- **Open a PR into `main` at every working checkpoint**, not just once at the end.
  For a 5-person, 36-hour hackathon, self-merging your own PR after a quick look
  is fine — the point of the PR is visibility for the team, not approval gates
- **Never hand-edit another track's file in `data/processed/`.** If the shape is
  wrong, fix it in a 2-minute team message and have the owning track's script
  produce the corrected shape
- **Pull before you push**, always — five people on one repo for 36 hours will
  have conflicts if you don't
- **Tag the Hour 26 feature-freeze commit:**
  ```bash
  git tag freeze-h26
  git push --tags
  ```
  so there's a known-good fallback if something breaks during polish

### Integration points (merge to `main` happens here, deliberately)

| When | What merges |
|---|---|
| Hour 12 | Every track's first real (non-fixture) output, for the first end-to-end run |
| Hour 26 | Final feature-complete state — tag it |
| Hour 33 | Final polish commits, before recording the backup demo |

Outside these points, merge to `main` whenever your branch reaches a clean working
state — don't wait for the checkpoints to merge, just make sure `main` never
breaks.

---

## 5. Step by step, start to finish

This is the to-do checklist's timeline, with the git actions made explicit. Full
task detail is in `Skillidify-To-Do-Checklist.md` — this is the "what happens in
the repo, when" version.

**Today — repo setup:** create the folder structure (section 1.1), push to `main`,
everyone clones and creates their track branch.

**Day -4 to Day -1:** each track works on its own branch. Day -2 is when the
fixture files (section 1.3) get committed to `main` — this is the single most
important merge before the hackathon starts, since it's what every track builds
against.

**Hour 0–2:** everyone pulls `main`, confirms the fixtures are there, confirms
their environment works.

**Hour 2–12:** heads-down on track branches, replacing fixture output with real
pipeline output. Commit often. Open PRs as pieces finish, even mid-phase.

**Hour 12:** merge everything into `main`, run the full pipeline on one real
syllabus end to end. This will surface integration bugs — that's the point of
doing it now rather than at hour 30.

**Hour 12–26:** fix what Hour 12 exposed, run the validation checks (section 6 of
the implementation plan), keep merging to `main` as things stabilize.

**Hour 26:** tag `freeze-h26`. No new features merge after this — only fixes.

**Hour 26–33:** polish on `main` directly is fine at this point (small team, late
stage) — or still via quick branches if you prefer the safety net.

**Hour 33–36:** final commits, record the demo video, submit.

---

## 6. If something breaks

- **A track is badly behind at Hour 12:** the other tracks keep building against
  the fixture file for that track — don't let one delay cascade into everyone
  blocking
- **A merge conflict in a `data/processed/*.json` file:** this shouldn't happen if
  the "only the owning track writes to its own file" rule is followed — if it
  does happen, it means someone edited a file they don't own; fix the rule
  violation, not just the conflict
- **The Hour 26 freeze reveals a broken demo path:** roll back to the
  `freeze-h26` tag for the live demo, keep fixing on a branch in parallel, swap
  back in only if the fix is solid well before Hour 36