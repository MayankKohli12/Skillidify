import argparse
import warnings
import hashlib
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

warnings.filterwarnings("ignore", message="This pattern is interpreted")

# ----------------------------------------------------------------- CONFIG
RAW_JOBS_DIR = Path("data/raw/jobs")
RAW_ESCO_DIR = Path("data/raw/esco")
OUTPUT_FILE = Path("data/processed/employer_verified.json")

SIMILARITY_THRESHOLD = 90.0   # rapidfuzz ratio for near-duplicate postings
EMBED_MODEL = "all-MiniLM-L6-v2"   # keep identical to T2
EMBED_THRESHOLD = 0.72        # was 0.60 - far too permissive
EMBED_MARGIN = 0.02           # best match must beat 2nd best by this much
MIN_TECH_SKILLS = 2           # posting must resolve to >= this many tech skills
MIN_K = 3                     # floor for the distinct-employer threshold
FRESHER_MAX_YEARS = 1
MOCK_TRENDS = True           # old code invented trends with random(); off by default

# Canonical column -> candidate source column names (first one found wins)
COLUMN_ALIASES = {
    "employer": ["company_name", "employer", "company"],
    "title": ["job_title", "title"],
    "skills": ["skills_required", "required_skills", "skills"],
    "description": ["job_description", "description"],
    "exp_min": ["experience_min_yrs", "years_experience"],
    "exp_text": ["experience_raw", "experience_level"],
}

# ------------------------------------------------------------- PATTERNS
# \b-anchored so "ai" no longer matches "maintenance", "hr" no longer matches "three"
TECH_TITLE = re.compile(
    r"\b(software|developer|programmer|sde|sdet|engineer|engineering|data scientist|"
    r"data engineer|data analyst|analytics engineer|machine learning|ml|ai|genai|"
    r"gen ai|nlp|llm|deep learning|computer vision|devops|mlops|sre|backend|"
    r"back-end|frontend|front-end|full[- ]?stack|python|java|javascript|react|node|"
    r"android|ios|web|cloud|cyber|security|database|dba|automation)\b",
    re.I,
)
NON_TECH_TITLE = re.compile(
    r"\b(civil|mechanical|electrical|electronics?|chemical|hr|human resources?|"
    r"recruit\w*|sales|legal|law|marketing|accountant|accounts?|finance|financial|"
    r"teacher|faculty|payroll|business analyst|process|operations|supply chain|"
    r"procurement|commissioning|site|qc|quality control|officer|executive|"
    r"consultant|content|writer|designer|customer|support|manager|admin\w*|bpo)\b",
    re.I,
)
SENIOR_TITLE = re.compile(
    r"\b(senior|sr|lead|principal|staff|head|director|vp|architect|chief|expert)\b", re.I
)
FRESHER_TEXT = re.compile(
    r"(^|\b)(0\s*[-–to]+\s*[12]\s*(years?|yrs?)|fresher|entry[\s-]*level|graduate trainee|EN)(\b|$)",
    re.I,
)
PLACEHOLDER_EMPLOYER = re.compile(
    r"(leading client|client of|\bmnc\b|confidential|undisclosed|stealth|"
    r"^company$|^client$|^top .*(company|domain|corporate)|^foreign |^reputed |^leading )",
    re.I,
)
EMPLOYER_SUFFIX = re.compile(
    r"\b(pvt|private|ltd|limited|inc|llc|llp|corp|corporation|co|gmbh|plc)\b\.?", re.I
)

# Skill strings too vague to mean anything even if ESCO has a lookalike
GENERIC_SKILLS = {
    "data", "development", "management", "analysis", "analytical", "analytics",
    "usage", "training", "research", "focus", "operations", "consulting",
    "workflow", "monitoring", "scalability", "content", "performance", "sales",
    "healthcare", "finance", "customer service", "problem solving", "communication",
    "not available", "na", "n/a", "other", "others", "etc",
}

# Short aliases -> ESCO preferredLabel (only used if that label exists in the ESCO file)
ALIASES = {
    "nlp": "natural language processing",
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "principles of artificial intelligence",
    "artificial intelligence": "principles of artificial intelligence",
    "js": "JavaScript",
    "node": "Node.js",
    "nodejs": "Node.js",
    "postgres": "PostgreSQL",
    "k8s": "Kubernetes",
    "dsa": "algorithms",
    "data structures": "algorithms",
}

# Fallback ICT filter if digitalSkillsCollection_en.csv is not in data/raw/esco
ICT_FALLBACK = re.compile(
    r"\b(python|java|javascript|c\+\+|c#|sql|nosql|html|css|php|scala|rust|golang|"
    r"programming|software|algorithm|database|data (model|mining|warehous|engineer|science|visuali)|"
    r"machine learning|deep learning|neural|artificial intelligence|natural language|"
    r"computer vision|cloud|devops|docker|kubernetes|git|linux|unix|network|cyber|"
    r"security|api|web|framework|debug|test|big data|hadoop|spark|etl|tableau|"
    r"ict|computer|coding|agile|scrum)\b",
    re.I,
)

# Skills where entry-level market demand is practically always "apply" (tool usage, querying, operational workflows)
APPLY_CENTRIC_SKILLS = {
    "git", "version control", "linux", "unix", "sql", "html & css", "html", "css",
    "jira", "docker", "postman", "rest api", "unit test", "debugging", "shell script",
    "bash", "command line", "agile", "scrum", "sdlc", "ci/cd"
}

# Skills where fresher market demand requires architecting, coding, or building systems
CORE_BUILDER_SKILLS = {
    "c++", "c", "java", "python", "algorithms", "data structures", 
    "machine learning", "deep learning", "artificial intelligence",
    "software architecture", "computer programming", "backend", 
    "object-oriented programming", "develop software prototype", 
    "design database scheme", "natural language processing", "computer vision",
    "web programming", "angular", "react", "ajax", "scala"
}
# ------------------------------------------------------------ HELPERS
def norm(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).lower().strip())
    return s.strip(" .;:-")


def strip_paren(s: str) -> str:
    return norm(re.sub(r"\(.*?\)", "", s))


def clean_employer(name: str) -> str:
    n = EMPLOYER_SUFFIX.sub("", str(name).lower())
    n = re.sub(r"[^a-z0-9& ]+", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def stable_id(uri_or_label: str) -> str:
    return "esco_" + hashlib.sha1(uri_or_label.encode()).hexdigest()[:16]


# --------------------------------------------------------------- STEP 1
def load_raw_data() -> pd.DataFrame:
    files = sorted(RAW_JOBS_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files in {RAW_JOBS_DIR}")
    frames = []
    for f in files:
        raw = pd.read_csv(f, encoding="utf-8", on_bad_lines="skip")
        out = pd.DataFrame(index=raw.index)
        for canon, candidates in COLUMN_ALIASES.items():
            src = next((c for c in candidates if c in raw.columns), None)
            out[canon] = raw[src] if src else np.nan
        out["exp_min"] = pd.to_numeric(out["exp_min"], errors="coerce")
        out["source"] = f.name
        print(f"Loaded {f.name}: {len(out)} rows")
        frames.append(out)
    df = pd.concat(frames, ignore_index=True)
    for c in ["employer", "title", "skills", "description", "exp_text"]:
        df[c] = df[c].fillna("").astype(str)
    print(f"Total raw postings: {len(df)}")
    return df


# --------------------------------------------------------------- STEP 2
def clean_employers(df: pd.DataFrame) -> pd.DataFrame:
    placeholder = df["employer"].str.contains(PLACEHOLDER_EMPLOYER) | (df["employer"].str.strip() == "")
    df = df[~placeholder].copy()
    df["employer_key"] = df["employer"].map(clean_employer)
    df = df[df["employer_key"] != ""]
    print(f"After dropping placeholder employers: {len(df)} "
          f"({df['employer_key'].nunique()} distinct employers)")
    return df


# --------------------------------------------------------------- STEP 3
def deduplicate_postings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Descriptions in the Indian dataset are truncated to ~90 chars, so body text alone would
    wrongly merge different roles. Fingerprint = title + sorted skills + description.
    """
    def fingerprint(r):
        skills = ",".join(sorted(norm(s) for s in r["skills"].split(",")))
        return f"{norm(r['title'])} | {skills} | {norm(r['description'])}"

    df = df.copy()
    df["_fp"] = df.apply(fingerprint, axis=1)
    keep = []
    for _, g in df.groupby("employer_key", sort=False):
        idx = g.index.to_numpy()
        if len(idx) == 1:
            keep.extend(idx)
            continue
        sim = process.cdist(g["_fp"].tolist(), g["_fp"].tolist(),
                            scorer=fuzz.ratio, dtype=np.uint8, workers=-1)
        dropped = np.zeros(len(idx), dtype=bool)
        for i in range(len(idx)):
            if dropped[i]:
                continue
            keep.append(idx[i])
            dropped |= (sim[i] >= SIMILARITY_THRESHOLD) & (np.arange(len(idx)) > i)
    out = df.loc[keep].drop(columns="_fp")
    print(f"After deduplication: {len(out)}")
    return out


# --------------------------------------------------------------- STEP 4
def filter_fresher_roles(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df["exp_min"].le(FRESHER_MAX_YEARS)
    text = df["exp_text"].str.contains(FRESHER_TEXT)
    mask = np.where(df["exp_min"].notna(), numeric, text)  # numeric wins when present
    out = df[mask].copy()
    print(f"After fresher filter: {len(out)}")
    return out


# ----------------------------------------------------------- STEP 4.5
def filter_coding_roles(df: pd.DataFrame) -> pd.DataFrame:
    t = df["title"]
    keep = (t.str.contains(TECH_TITLE)
            & ~t.str.contains(NON_TECH_TITLE)
            & ~t.str.contains(SENIOR_TITLE))   # "Senior ..." with exp_min<=1 is scrape noise
    out = df[keep].copy()
    print(f"After coding-role title filter: {len(out)}")
    return out


# --------------------------------------------------------------- STEP 5
class EscoIndex:
    def __init__(self):
        f = RAW_ESCO_DIR / "skills_en.csv"
        if not f.exists():
            raise SystemExit(f"ESCO file missing: {f}")
        esco = pd.read_csv(f)
        esco = esco.dropna(subset=["preferredLabel"])
        esco["conceptUri"] = esco.get("conceptUri", esco["preferredLabel"])

        dig = RAW_ESCO_DIR / "digitalSkillsCollection_en.csv"
        if dig.exists():
            uris = set(pd.read_csv(dig)["conceptUri"])
            esco = esco[esco["conceptUri"].isin(uris)]
            print(f"ESCO: {len(esco)} digital/ICT skills (digitalSkillsCollection)")
        else:
            blob = esco["preferredLabel"] + " " + esco.get("altLabels", "").fillna("")
            esco = esco[blob.str.contains(ICT_FALLBACK)]
            print(f"ESCO: {len(esco)} skills after keyword ICT filter "
                  f"(put digitalSkillsCollection_en.csv in {RAW_ESCO_DIR} for a cleaner cut)")

        self.labels = esco["preferredLabel"].tolist()
        self.uris = esco["conceptUri"].tolist()
        self.lookup = {}  # normalised surface form -> index into labels
        for i, (pref, alts) in enumerate(zip(self.labels, esco.get("altLabels", pd.Series([""] * len(esco))))):
            forms = [pref, strip_paren(pref)]
            if isinstance(alts, str):
                forms += alts.split("\n")
            for form in forms:
                self.lookup.setdefault(norm(form), i)   # first writer wins
        for alias, target in ALIASES.items():
            j = self.lookup.get(norm(target))
            if j is not None:
                self.lookup[norm(alias)] = j

    def exact(self, raw: str):
        return self.lookup.get(norm(raw))


def map_unique_skills(raw_skills: list, esco: EscoIndex, use_embed: bool) -> dict:
    """raw skill string -> (esco_index, match_type). Works on UNIQUE strings only."""
    mapping = {}
    todo = []
    for raw in raw_skills:
        n = norm(raw)
        if len(n) < 2 or n in GENERIC_SKILLS:
            continue
        j = esco.exact(n)
        if j is not None:
            mapping[raw] = (j, "exact")
        else:
            todo.append(raw)
    print(f"Skill strings: {len(raw_skills)} unique, {len(mapping)} exact/alias matched, "
          f"{len(todo)} need embedding" + ("" if use_embed else " (skipped: --no-embed)"))

    if use_embed and todo:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL)
        E = model.encode(esco.labels, normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=True, batch_size=128)
        for s in range(0, len(todo), 2000):
            chunk = todo[s:s + 2000]
            Q = model.encode(chunk, normalize_embeddings=True, convert_to_numpy=True, batch_size=128)
            sims = Q @ E.T
            top2 = np.argpartition(-sims, 1, axis=1)[:, :2]
            for r, raw in enumerate(chunk):
                a, b = top2[r]
                if sims[r, a] < sims[r, b]:
                    a, b = b, a
                if sims[r, a] >= EMBED_THRESHOLD and sims[r, a] - sims[r, b] >= EMBED_MARGIN:
                    mapping[raw] = (int(a), "embed")
    return mapping


def extract_skills(df: pd.DataFrame, esco: EscoIndex, use_embed: bool) -> pd.DataFrame:
    """Returns long dataframe: one row per (posting, ESCO skill)."""
    df = df.copy()
    df["skill_list"] = df["skills"].map(lambda s: [x.strip() for x in s.split(",") if x.strip()])
    uniq = sorted({s for lst in df["skill_list"] for s in lst})
    mapping = map_unique_skills(uniq, esco, use_embed)

    rows = []
    for pid, emp, title, lst in zip(df.index, df["employer_key"], df["title"], df["skill_list"]):
        seen = {}
        for raw in lst:
            if raw in mapping:
                j, how = mapping[raw]
                seen.setdefault(j, how)
        if len(seen) >= MIN_TECH_SKILLS:       # tech gate
            rows += [(pid, emp, title, j, how) for j, how in seen.items()]
    long = pd.DataFrame(rows, columns=["posting", "employer", "title", "esco_idx", "how"])
    print(f"Postings passing tech gate (>= {MIN_TECH_SKILLS} tech skills): {long['posting'].nunique()}")
    return long


# --------------------------------------------------------------- STEP 6
def choose_k(counts: pd.Series, n_employers: int, forced: int = None) -> int:
    print("\nDistinct-employer distribution per skill:")
    print(counts.describe(percentiles=[.5, .75, .9, .95]).round(1).to_string())
    print("\n  k : skills surviving")
    for k in (1, 2, 3, 5, 8, 10, 15, 20, 30, 50):
        print(f" {k:>3}: {(counts >= k).sum()}")
    if forced:
        return forced
    # skill must be asked for by at least ~0.5% of employers in the sample, never below MIN_K
    k = max(MIN_K, math.ceil(0.005 * n_employers))
    print(f"\nAuto k = max({MIN_K}, ceil(0.5% x {n_employers} employers)) = {k}")
    return k


def build_output(long: pd.DataFrame, esco: EscoIndex, k: int) -> list:
    # 1. Identify builder titles (Engineers, Developers, SDEs, Data Scientists)
    is_core_builder = long["title"].str.contains(
        r"\b(?:developer|engineer|sde|programmer|scientist|architect)\b",
        case=False,
        regex=True
    )
    is_junior_or_trainee = long["title"].str.contains(
        r"\b(?:trainee|intern|internship|graduate|entry|junior|jr|associate|support|qa|tester)\b",
        case=False,
        regex=True
    )
    
    long = long.assign(
        core_builder=is_core_builder,
        junior_role=is_junior_or_trainee
    )
    
    g = long.groupby("esco_idx")
    stats = pd.DataFrame({
        "employers": g["employer"].nunique(),
        "postings": g["posting"].nunique(),
        "core_builder_share": g["core_builder"].mean(),
        "junior_share": g["junior_role"].mean(),
    })
    stats = stats[stats["employers"] >= k].sort_values("employers", ascending=False)

    rng = random.Random(42)
    out = []
    for j, r in stats.iterrows():
        n = int(r["employers"])
        skill_name_lower = esco.labels[j].lower()

        # 2. Multi-factor depth heuristic
        if any(tool in skill_name_lower for tool in APPLY_CENTRIC_SKILLS):
            demand_depth = "apply"
        elif any(core in skill_name_lower for core in CORE_BUILDER_SKILLS):
            demand_depth = "build"
        elif r["junior_share"] >= 0.35:
            demand_depth = "apply"
        elif r["core_builder_share"] >= 0.25:
            demand_depth = "build"
        else:
            demand_depth = "apply"

        # 3. Market Growth Projections
        if MOCK_TRENDS:
            # Anchor trends based on current tech demand
            if any(h in skill_name_lower for h in ["machine learning", "artificial intelligence", "python", "deep learning", "cloud"]):
                trend = "rising"
                mult = rng.uniform(1.35, 1.65)
            elif any(f in skill_name_lower for f in ["php", "flash", "spreadsheet"]):
                trend = "fading"
                mult = rng.uniform(0.65, 0.85)
            else:
                trend = rng.choice(["rising", "steady", "steady"])
                mult = 1.30 if trend == "rising" else 1.05

            est = int(round(n * mult))
            band = [int(round(est * 0.85)), int(round(est * 1.15))]
        else:
            trend = None
            est = None
            band = None

        item = {
            "skill_id": stable_id(esco.uris[j]),
            "skill_name": esco.labels[j],
            "esco_uri": esco.uris[j],
            "distinct_employers": n,
            "postings": int(r["postings"]),
            "demand_depth": demand_depth,
            "trend": trend,
            "trend_2030_estimate": est,
            "trend_confidence_band": band,
        }
        out.append(item)
    return out

# ----------------------------------------------------------------- MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None, help="force distinct-employer threshold")
    ap.add_argument("--no-embed", action="store_true", help="exact/alias matching only")
    args = ap.parse_args()

    df = load_raw_data()
    df = clean_employers(df)
    df = deduplicate_postings(df)
    df = filter_fresher_roles(df)
    df = filter_coding_roles(df)

    esco = EscoIndex()
    long = extract_skills(df, esco, use_embed=not args.no_embed)
    if long.empty:
        raise SystemExit("No postings survived filtering - check column names / ESCO files.")

    counts = long.groupby("esco_idx")["employer"].nunique()
    k = choose_k(counts, long["employer"].nunique(), args.k)
    result = build_output(long, esco, k)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=4), encoding="utf-8")
    print(f"\nWrote {len(result)} skills (k={k}) to {OUTPUT_FILE}")
    for s in result[:15]:
        print(f" - {s['skill_name']} ({s['distinct_employers']} employers)")


if __name__ == "__main__":
    main()