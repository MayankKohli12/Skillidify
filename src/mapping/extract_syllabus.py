import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file immediately
load_dotenv()

import numpy as np
import pandas as pd
import pdfplumber
import spacy
import torch
from sentence_transformers import SentenceTransformer, util

# --- CONFIG ---
SYLLABUS_DIR = Path("data/raw/syllabi")
ESCO_FILE = Path("data/raw/esco/skills_en.csv")
OUTPUT_FILE = Path("data/processed/skills.json")
CACHE_EMB = Path("data/processed/esco_embeddings.npy")
CACHE_META = Path("data/processed/esco_meta.json")

MIN_SCORE = 0.50
MIN_REGEX_TOPICS = 3  # Below this threshold, PDF triggers Groq fallback
LLM_MODEL = "llama-3.3-70b-versatile"

print("Loading NLP and SentenceTransformer models...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

TECH_KEYWORDS = {
    "c", "c++", "java", "python", "sql", "html", "css", "javascript", "matlab",
    "programming", "code", "coding", "algorithm", "data structure", "array", "pointer",
    "stack", "queue", "linked list", "tree", "graph", "sorting", "searching", "recursion",
    "database", "dbms", "table", "query", "normalization", "schema", "relational",
    "machine learning", "artificial intelligence", "deep learning", "neural network", "nlp",
    "computer vision", "pandas", "numpy", "matplotlib", "scikit", "model",
    "operating system", "process", "thread", "memory", "cpu", "cache", "paging",
    "software engineering", "agile", "scrum", "git", "devops", "docker", "cloud", "aws", "azure",
    "network", "tcp", "udp", "ip", "protocol", "security", "encryption",
    "cad", "autocad", "graphics", "2d", "3d", "orthographic", "isometric", "modeling",
    "matrix", "matrices", "calculus", "probability", "statistics", "algebra", "discrete"
}

BANNED_CONCEPTS = {
    "tree operation", "milling", "beverage", "yoga", "moral", "spiritual", "gender",
    "family law", "railway", "train staff", "navigation", "fertilizer", "forestry",
    "meditation", "patient", "hospital", "psycholog", "adolescent", "nature conservation",
    "game protect", "fossil", "printing program", "offset printing", "electricity meter",
    "clothing", "dance", "theatre", "relig", "water treatment"
}

BANNED_COURSE_KEYWORDS = [
    "career", "pathway", "values", "ethics", "environmental", "life skills", "soft skills",
    "human rights", "constitution", "gender", "wellness", "mentoring", "universal human"
]

DEPTH_LEXICON = {
    "know": {"define", "describe", "explain", "identify", "list", "state", "understand",
             "recall", "outline", "summarize", "discuss", "study", "recognize", "name",
             "classify", "know", "learn", "introduce"},
    "apply": {"implement", "use", "apply", "write", "solve", "demonstrate", "execute",
              "compute", "calculate", "perform", "analyze", "compare", "trace", "illustrate",
              "run", "test", "debug", "configure", "install", "simulate", "practice"},
    "build": {"design", "develop", "build", "create", "construct", "evaluate", "integrate",
              "deploy", "architect", "optimize", "synthesize", "formulate", "engineer",
              "propose", "compose", "plan", "produce"}
}
VERB2DEPTH = {v: d for d, vs in DEPTH_LEXICON.items() for v in vs}
DEPTH_RANK = {"know": 0, "apply": 1, "build": 2}


# ---------- Step 1: Text extraction & Course vetting ----------
def extract_pages(pdf_path: Path) -> list[str]:
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = [(p.extract_text() or "") for p in pdf.pages]
    except Exception as e:
        print(f"  pdfplumber failed on {pdf_path.name}: {e}")

    if sum(len(p.strip()) for p in pages) < 200:
        try:
            import fitz  # pymupdf
            with fitz.open(pdf_path) as doc:
                pages = [page.get_text() for page in doc]
            print(f"  Used PyMuPDF fallback for {pdf_path.name}")
        except Exception:
            pass
    return pages


def is_core_course(page1: str) -> tuple[bool, str]:
    if not page1.strip():
        return False, "Empty PDF"
    if "Mandatory Non-Graded" in page1 or "(MNG)" in page1:
        return False, "Mandatory Non-Graded (MNG)"
    
    m = re.search(r"Course Name\s*\|\s*([^\n|]+)", page1, re.I) or \
        re.search(r"Course Name\s*([A-Za-z\s&]+?)(?:Course Code|\n)", page1)
    name = m.group(1).strip() if m else ""
    
    for b in BANNED_COURSE_KEYWORDS:
        if b in name.lower():
            return False, f"Non-core course title: '{name}'"
    return True, f"Core course: '{name}'"


# ---------- Step 6-7: Verb & Depth analysis ----------
def governing_verb(text: str) -> str | None:
    doc = nlp(text)
    for tok in doc:
        if tok.dep_ == "ROOT" and tok.pos_ == "VERB":
            return tok.lemma_.lower()
    for tok in doc:
        if tok.pos_ == "VERB":
            return tok.lemma_.lower()
    first = doc[0].lemma_.lower() if len(doc) else ""
    return first if first in VERB2DEPTH else None


def bt_to_depth(s: str) -> str | None:
    lv = [int(x) for x in re.findall(r"BT([1-6])", str(s).upper())]
    if not lv:
        return None
    m = max(lv)
    return "know" if m <= 2 else "apply" if m == 3 else "build"


def infer_depth(raw: str, bt_str: str) -> tuple[str | None, str]:
    verb = governing_verb(raw)
    if verb in VERB2DEPTH:
        return verb, VERB2DEPTH[verb]
    return verb, bt_to_depth(bt_str) or "apply"


# ---------- Step 2: Chunking & Fallback ----------
def clean_topic(t: str) -> str:
    t = re.sub(r"^(Write a (C )?program to|Implement a (C )?program to|Develop a program( that)?|"
               r"Demonstrate( a program to)?|Study of|Introduction to|Overview of)\s+", "", t, flags=re.I)
    t = re.sub(r"using (for loop|while loop|functions|pointers|dma|recursion|loops|arrays).*$", "", t, flags=re.I)
    return t.strip(" -:,.\t")


def valid_topic(t: str) -> bool:
    tl = t.lower().strip()
    return len(tl) >= 4 and bool(re.search(r"[a-zA-Z]", tl)) and not any(b in tl for b in BANNED_CONCEPTS)


SKIP = re.compile(r"^(po\d+|peo\d+|pso\d+|mission|vision|sr no|credit|mode of|assessment|mst-|"
                  r"practical end|attendance|exam name|co vs po)", re.I)


def regex_chunks(pages: list[str]) -> list[dict]:
    out, in_sec, unit = [], False, None
    for text in pages:
        for line in text.split("\n"):
            s = line.strip()
            if "Lecture Plan Preview" in s or "Experiment Name" in s:
                in_sec = True
                continue
            if "Assessment Model" in s or "CO vs PO" in s:
                in_sec = False
                continue
            um = re.match(r"^unit\s*[-:]?\s*([ivx\d]+)", s, re.I)
            if um:
                unit = um.group(1)
            if not in_sec or len(s) < 5 or SKIP.search(s):
                continue

            bt = re.search(r"CO\d+\s*-\s*BT[1-6]", s)
            hrs = re.search(r"(\d+)\s*(?:hrs?|hours?)\b", s, re.I)
            body = re.sub(r"^\d+\s+\d+\s+", "", s)
            body = re.sub(r"CO\d+\s*-\s*BT[1-6].*$", "", body)
            body = re.sub(r"\bT-[A-Za-z0-9\s:,]+", "", body)
            body = re.sub(r"\bR-[A-Za-z0-9\s:,]+", "", body)
            for part in re.split(r"•|;|\d+\)", body):
                part = part.strip()
                if part:
                    out.append({
                        "raw": part,
                        "bt": bt.group(0) if bt else "",
                        "unit": unit,
                        "hours": int(hrs.group(1)) if hrs else None
                    })
    return out


def llm_chunks(pages: list[str]) -> list[dict]:
    """Groq API fallback for unstructured or complex table layouts."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("  [INFO] GROQ_API_KEY not set - skipping Groq fallback.")
        return []

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
    except ImportError:
        print("  [WARNING] 'groq' package not installed. Run 'pip install groq'.")
        return []

    out = []
    prompt = (
        "Extract technical computer science syllabus topics from this document text. "
        "Output ONLY a raw valid JSON array of objects with keys: 'unit' (str or null), "
        "'topic' (str), 'hours' (int or null). "
        "Do not include Markdown blocks (no ```json). Ignore administrative and grading text.\n\n"
    )

    for text in pages:
        if len(text.strip()) < 80:
            continue
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt + text}],
                model=LLM_MODEL,
                temperature=0.1
            )
            raw = chat_completion.choices[0].message.content or ""
            raw = re.sub(r"```json|```", "", raw).strip()
            
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                for it in parsed:
                    if it.get("topic"):
                        out.append({
                            "raw": it["topic"],
                            "bt": "",
                            "unit": it.get("unit"),
                            "hours": it.get("hours")
                        })
        except Exception as e:
            print(f"  Groq API call warning: {e}")
            
    return out


def extract_topics() -> list[dict]:
    records = []
    for pdf_path in sorted(SYLLABUS_DIR.glob("*.pdf")):
        pages = extract_pages(pdf_path)
        ok, reason = is_core_course(pages[0] if pages else "")
        if not ok:
            print(f"[SKIPPED] {pdf_path.name} -> {reason}")
            continue
        print(f"[PARSING] {pdf_path.name} -> {reason}")

        chunks = regex_chunks(pages)
        if len(chunks) < MIN_REGEX_TOPICS:
            print(f"  Few regex topics found ({len(chunks)}) - triggering Groq fallback...")
            fallback = llm_chunks(pages)
            if fallback:
                chunks = fallback

        for c in chunks:
            verb, depth = infer_depth(c["raw"], c["bt"])
            topic = clean_topic(c["raw"])
            if valid_topic(topic):
                records.append({
                    "raw_text": topic,
                    "verb": verb,
                    "depth": depth,
                    "unit": c["unit"],
                    "hours": c["hours"],
                    "source": pdf_path.name
                })
                
    print(f"\nExtracted {len(records)} total topic units across approved syllabi.")
    return records


# ---------- Step 4: Cached ESCO embeddings ----------
def load_esco_embeddings():
    if CACHE_EMB.exists() and CACHE_META.exists():
        meta = json.loads(CACHE_META.read_text(encoding="utf-8"))
        print(f"Loaded cached ESCO embeddings ({len(meta)} skills).")
        return meta, np.load(CACHE_EMB)

    df = pd.read_csv(ESCO_FILE).dropna(subset=["preferredLabel"]).drop_duplicates("preferredLabel")
    df = df[~df["preferredLabel"].str.lower().apply(lambda s: any(b in s for b in BANNED_CONCEPTS))]
    desc = df["description"].fillna("") if "description" in df.columns else ""
    texts = (df["preferredLabel"] + ". " + desc).tolist()
    
    # Standardize skill IDs with T1 contract (esco_<hash>)
    meta = [
        {"id": f"esco_{abs(hash(lbl))}", "name": lbl}
        for lbl in df["preferredLabel"]
    ]

    print(f"Computing embeddings for {len(texts)} ESCO skills (cached for future runs)...")
    emb = embedder.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    
    CACHE_EMB.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_EMB, emb)
    CACHE_META.write_text(json.dumps(meta), encoding="utf-8")
    return meta, emb


def has_tech_signal(*texts: str) -> bool:
    for t in texts:
        tl = t.lower()
        for kw in TECH_KEYWORDS:
            if re.search(rf"(?<![\w+]){re.escape(kw)}(?![\w+])", tl):
                return True
    return False


# ---------- Step 3 + 5: Embed & Match ----------
def map_to_esco(records: list[dict]) -> list[dict]:
    meta, esco_emb = load_esco_embeddings()
    topic_emb = embedder.encode([r["raw_text"] for r in records], batch_size=64, convert_to_numpy=True)
    scores = util.cos_sim(topic_emb, esco_emb).numpy()
    
    best_idx = scores.argmax(axis=1)
    best_score = scores.max(axis=1)

    best: dict[str, dict] = {}
    for rec, i, sc in zip(records, best_idx, best_score):
        skill = meta[int(i)]
        if sc < MIN_SCORE or not has_tech_signal(skill["name"], rec["raw_text"]):
            continue
            
        entry = {
            "skill_id": skill["id"],
            "skill_name": skill["name"],
            "original_topic": rec["raw_text"],
            "verb": rec["verb"],
            "depth": rec["depth"],
            "confidence": round(float(sc), 4),
            "unit": rec["unit"],
            "hours": rec["hours"],
            "source": rec["source"]
        }
        
        prev = best.get(skill["id"])
        if prev is None:
            best[skill["id"]] = entry
        else:
            # Preserve highest depth taught (build > apply > know)
            deepest = max(prev["depth"], entry["depth"], key=DEPTH_RANK.get)
            if entry["confidence"] > prev["confidence"]:
                prev = entry
            prev["depth"] = deepest
            best[skill["id"]] = prev

    print(f"Mapped {len(best)} unique technical skills.")
    return list(best.values())


def main():
    records = extract_topics()
    if not records:
        print("No valid syllabus records extracted.")
        return
        
    skills = map_to_esco(records)
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(skills, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(f"\nSaved verified T2 contract to: {OUTPUT_FILE}")
    for s in skills[:8]:
        print(f" - [{s['depth'].upper()}] {s['skill_name']} ({s['confidence']}) <- '{s['original_topic']}'")


if __name__ == "__main__":
    main()