
import hashlib
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pdfplumber

# ------------------------------------------------------------------ CONFIG
SYLLABUS_DIR = Path("data/raw/syllabi")
ESCO_FILE = Path("data/raw/esco/skills_en.csv")
OUTPUT_FILE = Path("data/processed/skills.json")
CACHE_EMB = Path("data/processed/esco_labels_v2.npy")   # new names: old cache had unstable ids
CACHE_META = Path("data/processed/esco_labels_v2.json")

LINK_ESCO = True            # attach esco_id / esco_name to accepted skills (needs ESCO_FILE)
ESCO_LINK_MIN = 0.80
SEMANTIC_MIN = 0.62         # embedding fallback threshold against the curated taxonomy
MIN_TOPICS = 3              # below this, fall back to line regex, then Groq
LLM_MODEL = "llama-3.3-70b-versatile"

# Which parts of the taxonomy are kept. Add "data_science" if you also want ML / NumPy / pandas.
ALLOWED_CATEGORIES = {"programming", "oop", "dsa", "algorithms", "database", "tools", "web","data_science"}

# Only technical CSE subject prefixes (MEP = mechanical practical/CAD, ECT = electronics theory removed)
ALLOWED_COURSE_PREFIXES = {"CSH", "CST", "CSP", "ECP"}

BANNED_COURSE_KEYWORDS = [
    "career", "pathway", "values", "ethics", "environmental", "life skills", "soft skills",
    "human rights", "constitution", "gender", "wellness", "mentoring", "universal human",
    "drawing", "workshop", "mechanical", "communication",
]

# ------------------------------------------------------------------ TAXONOMY
# (category, canonical skill name, [aliases]).  Aliases are case-insensitive, whole-word,
# plural-tolerant.  Prefix an alias with "re:" to supply a raw regex instead.
TAXONOMY = [
    # ---- programming languages
    ("programming", "C programming", ["c programming", "c language", "c program", "programming in c", "turbo c"]),
    ("programming", "C++", ["c++", "cpp"]),
    ("programming", "Java", ["java", "jdk", "jvm", "core java"]),
    ("programming", "Python", ["python", "python programming"]),
    ("programming", "MATLAB", ["matlab"]),
    ("web", "JavaScript", ["javascript", "node.js", "nodejs", "ecmascript", "typescript"]),
    ("web", "HTML & CSS", ["html", "html5", "css", "css3"]),
    ("web", "Web development", ["web development", "web application", "full stack", "full-stack", "responsive design", "react", "reactjs", "rest api", "restful"]),

    # ---- programming fundamentals
    ("programming", "Problem solving & algorithmic thinking", ["problem solving", "problem-solving", "flowchart", "pseudocode", "pseudo code", "algorithmic thinking", "algorithm development"]),
    ("programming", "Variables & data types", ["data type", "datatype", "variable", "constant", "identifier", "primitive type"]),
    ("programming", "Operators & expressions", ["operator", "expression", "operator precedence", "arithmetic operator", "relational operator", "logical operator"]),
    ("programming", "Bit manipulation", ["bit manipulation", "bitwise", "bit-wise"]),
    ("programming", "Conditional statements", ["decision making", "conditional statement", "if-else", "if else", "nested if", "switch case", "switch statement", "switch-case", "selection statement", "branching"]),
    ("programming", "Loops & iteration", ["loop", "looping", "iteration", "iterative", "for loop", "while loop", "do-while", "do while"]),
    ("programming", "Functions & modular programming", ["function", "modular programming", "call by value", "call by reference", "parameter passing", "user defined function", "user-defined function", "library function"]),
    ("programming", "Recursion", ["recursion", "recursive"]),
    ("programming", "Strings", ["string", "character array", "string handling", "string manipulation"]),
    ("programming", "Pointers", ["pointer", "dereferencing", "address of operator"]),
    ("programming", "Dynamic memory allocation", ["dynamic memory", "malloc", "calloc", "realloc", "dma", "memory allocation"]),
    ("programming", "Structures & unions", ["re:(?<!data )(?<!information )(?<!file )(?<!database )structures?", "struct", "union", "typedef", "enum"]),
    ("programming", "File handling", ["file handling", "file operation", "file i/o", "fopen", "file input", "text file", "binary file"]),
    ("programming", "Console input/output", ["input/output", "input and output", "scanf", "printf", "cin", "cout", "user input", "formatted i/o"]),
    ("programming", "Preprocessor & macros", ["preprocessor", "macro", "header file", "#define", "#include"]),
    ("programming", "Type conversion", ["type casting", "typecasting", "type conversion"]),
    ("programming", "Exception handling", ["exception handling", "try catch", "try-catch", "exception"]),
    ("programming", "Debugging & unit testing", ["debugging", "debugger", "gdb", "unit test", "unit testing", "test case"]),

    # ---- OOP
    ("oop", "Object-oriented programming", ["object oriented", "object-oriented", "oop", "oops", "class", "classes and objects"]),
    ("oop", "Inheritance", ["inheritance", "base class", "derived class"]),
    ("oop", "Polymorphism", ["polymorphism", "overloading", "overriding", "virtual function"]),
    ("oop", "Encapsulation & abstraction", ["encapsulation", "abstraction", "abstract class", "data hiding", "access specifier", "access modifier"]),
    ("oop", "Constructors & destructors", ["constructor", "destructor"]),
    ("oop", "Templates & generics", ["template", "generic programming", "generics"]),
    ("oop", "STL / Collections framework", ["stl", "standard template library", "collections framework", "arraylist", "iterator"]),
    ("oop", "Multithreading", ["multithreading", "multi-threading"]),

    # ---- Data structures
    ("dsa", "Data structures (general)", ["data structure", "abstract data type", "adt"]),
    ("dsa", "Arrays", ["array", "1d array", "2d array", "two dimensional array", "multidimensional array", "sparse matrix"]),
    ("dsa", "Linked lists", ["linked list", "singly linked", "doubly linked", "circular linked"]),
    ("dsa", "Stacks", ["re:(?<!full )(?<!tech )stacks?"]),
    ("dsa", "Queues", ["queue", "deque", "circular queue"]),
    ("dsa", "Heaps & priority queues", ["priority queue", "re:(?<!memory )heaps?(?! memory)", "min heap", "max heap"]),
    ("dsa", "Trees & tree traversal", ["re:(?<!decision )(?<!family )(?<!syntax )trees?", "binary tree", "tree traversal", "inorder", "preorder", "postorder", "in-order", "pre-order", "post-order"]),
    ("dsa", "Binary search trees", ["binary search tree", "bst"]),
    ("dsa", "Balanced trees", ["avl", "red-black", "red black", "b-tree", "b+ tree", "balanced tree", "segment tree"]),
    ("dsa", "Graphs", ["re:graphs?", "adjacency matrix", "adjacency list", "graph representation"]),
    ("dsa", "Hashing", ["hashing", "hash table", "hash function", "hash map", "hashmap", "collision resolution"]),
    ("dsa", "Tries", ["trie", "prefix tree"]),

    # ---- Algorithms
    ("algorithms", "Sorting algorithms", ["sorting", "sort", "bubble sort", "selection sort", "insertion sort", "merge sort", "quick sort", "quicksort", "heap sort", "radix sort", "counting sort", "shell sort", "bucket sort"]),
    ("algorithms", "Searching algorithms", ["searching", "linear search", "sequential search", "re:binary search(?! trees?)", "interpolation search"]),
    ("algorithms", "Complexity analysis", ["time complexity", "space complexity", "big o", "big-o", "asymptotic", "complexity analysis", "algorithm analysis", "analysis of algorithm", "omega notation", "theta notation"]),
    ("algorithms", "Dynamic programming", ["dynamic programming", "memoization", "tabulation", "knapsack", "longest common subsequence", "lcs"]),
    ("algorithms", "Greedy algorithms", ["greedy", "huffman", "activity selection"]),
    ("algorithms", "Divide and conquer", ["divide and conquer", "divide-and-conquer"]),
    ("algorithms", "Backtracking", ["backtracking", "n-queen", "n queen", "branch and bound"]),
    ("algorithms", "Graph traversal (BFS/DFS)", ["bfs", "dfs", "breadth first", "breadth-first", "depth first", "depth-first"]),
    ("algorithms", "Shortest path algorithms", ["dijkstra", "bellman", "floyd", "warshall", "shortest path"]),
    ("algorithms", "Minimum spanning tree", ["minimum spanning", "spanning tree", "kruskal", "prim's", "prims algorithm"]),
    ("algorithms", "String matching algorithms", ["pattern matching", "string matching", "kmp", "rabin-karp", "rabin karp"]),
    ("algorithms", "NP-completeness", ["np-complete", "np complete", "np-hard", "np hard"]),
    ("algorithms", "Competitive programming", ["competitive programming", "leetcode", "hackerrank", "codeforces"]),

    # ---- Databases (query / schema coding)
    ("database", "SQL", ["sql", "mysql", "postgresql", "pl/sql", "ddl", "dml", "dcl", "tcl", "query", "queries", "subquery", "join", "group by", "aggregate function", "stored procedure", "trigger"]),
    ("database", "Database design & normalization", ["normalization", "normal form", "er diagram", "er model", "entity relationship", "entity-relationship", "database design", "schema", "functional dependency", "functional dependencies", "bcnf", "primary key", "foreign key"]),
    ("database", "Relational model & algebra", ["relational model", "relational algebra", "relational database", "rdbms"]),
    ("database", "Database management systems", ["dbms", "database management", "database system"]),
    ("database", "Transactions & concurrency control", ["transaction", "acid propert", "concurrency control", "serializability", "database recovery", "indexing"]),
    ("database", "NoSQL", ["nosql", "mongodb", "redis", "cassandra"]),

    # ---- Dev tools
    ("tools", "Git & version control", ["git", "github", "gitlab", "version control"]),
    ("tools", "Shell scripting & command line", ["shell script", "shell scripting", "bash", "command line", "linux command", "unix command"]),

    # ---- Optional: data science / ML coding (off by default, see ALLOWED_CATEGORIES)
    ("data_science", "NumPy / pandas", ["numpy", "pandas", "dataframe"]),
    ("data_science", "Matplotlib / data visualization", ["matplotlib", "seaborn", "data visualization"]),
    ("data_science", "Machine learning", ["machine learning", "supervised learning", "unsupervised learning", "regression", "clustering", "scikit-learn", "sklearn"]),
    ("data_science", "Deep learning", ["deep learning", "neural network", "cnn", "rnn", "tensorflow", "pytorch", "keras"]),
    ("data_science", "NLP & computer vision", ["nlp", "natural language processing", "computer vision", "opencv"]),
]


def _compile(alias: str) -> re.Pattern:
    if alias.startswith("re:"):
        body = alias[3:]
    else:
        body = re.escape(alias.lower()) + r"(?:e?s)?"
    return re.compile(r"(?<![\w+#])" + body + r"(?![\w#+])", re.I)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


SKILLS = [
    {
        "id": f"cs_{_slug(name)}",
        "name": name,
        "category": cat,
        "patterns": [_compile(a) for a in aliases],
        "sem_text": f"{name}: " + ", ".join(a for a in aliases if not a.startswith("re:"))[:200],
    }
    for cat, name, aliases in TAXONOMY
    if cat in ALLOWED_CATEGORIES
]

# ------------------------------------------------------------------ DEPTH
DEPTH_LEXICON = {
    "know": {"define", "describe", "explain", "identify", "list", "state", "understand", "recall",
             "outline", "summarize", "discuss", "study", "recognize", "name", "classify", "know",
             "learn", "introduce", "introduction", "overview", "basics"},
    "apply": {"implement", "use", "apply", "write", "solve", "demonstrate", "execute", "compute",
              "calculate", "perform", "analyze", "compare", "trace", "illustrate", "run", "test",
              "debug", "configure", "install", "simulate", "practice", "create", "insert", "delete"},
    "build": {"design", "develop", "build", "construct", "evaluate", "integrate", "deploy",
              "architect", "optimize", "synthesize", "formulate", "engineer", "propose", "compose",
              "plan", "produce"},
}
VERB2DEPTH = {v: d for d, vs in DEPTH_LEXICON.items() for v in vs}
DEPTH_RANK = {"know": 0, "apply": 1, "build": 2}


def leading_verb(text: str):
    """First action word among the first 3 words, with crude suffix stripping."""
    for w in re.findall(r"[a-z]+", text.lower())[:3]:
        for cand in (w, w[:-1], w[:-2], w[:-3]):
            if cand in VERB2DEPTH:
                return cand
    return None


def bt_to_depth(s: str):
    lv = [int(x) for x in re.findall(r"BT\s*([1-6])", str(s).upper())]
    if not lv:
        return None
    m = max(lv)
    return "know" if m <= 2 else "apply" if m == 3 else "build"


def infer_depth(fragment: str, bt: str, kind: str):
    verb = leading_verb(fragment)
    if verb:
        return verb, VERB2DEPTH[verb]
    return None, bt_to_depth(bt) or ("apply" if kind == "practical" else "know")


# ------------------------------------------------------------------ STEP 1: course vetting
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


CODE_RE = re.compile(r"\b\d{2}([A-Z]{3})[-_]?\d{3}\b")


def is_core_course(page1: str, filename: str):
    if not page1.strip():
        return False, "Empty PDF"
    if "Mandatory Non-Graded" in page1 or "(MNG)" in page1:
        return False, "Mandatory Non-Graded (MNG)"

    m = CODE_RE.search(filename) or CODE_RE.search(page1)
    if m and m.group(1) not in ALLOWED_COURSE_PREFIXES:
        return False, f"Non-coding subject prefix '{m.group(1)}'"

    nm = re.search(r"Course Name\s*\|?\s*(.+?)\s*Course Code", page1, re.I | re.S)
    name = re.sub(r"\s+", " ", nm.group(1)).strip() if nm else ""
    for b in BANNED_COURSE_KEYWORDS:
        if b in name.lower():
            return False, f"Non-core course title: '{name}'"
    return True, f"Core course: '{name}'"


# ------------------------------------------------------------------ STEP 2: topic extraction
def _cell(c) -> str:
    return re.sub(r"\s+", " ", (c or "").replace("\n", " ")).strip()


HDR_TOPIC = re.compile(r"^(topics?|experiment name|experiment title|contents?)$", re.I)


def _detect_header(cells: list[str]):
    t = next((i for i, c in enumerate(cells) if HDR_TOPIC.match(c)), None)
    if t is None:
        return None
    m = {"topic": t, "ncols": len(cells),
         "kind": "practical" if "experiment" in cells[t].lower() else "theory"}
    for i, c in enumerate(cells):
        cl = c.lower()
        if cl.startswith("unit"):
            m.setdefault("unit", i)
        elif "bt" in cl and "co" in cl:
            m["bt"] = i
        elif re.search(r"\bhou?rs?\b|\bhrs\b", cl):
            m["hours"] = i
    return m


def table_chunks(pdf_path: Path) -> list[dict]:
    """Read ONLY the Topic / Experiment-Name column of the lecture-plan tables."""
    out, cmap = [], None
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for row in table:
                    if not row:
                        continue
                    cells = [_cell(c) for c in row]
                    hdr = _detect_header(cells)
                    if hdr:
                        cmap = hdr
                        continue
                    # continuation pages repeat no header: reuse the last map if the shape matches
                    if cmap is None or len(cells) != cmap["ncols"]:
                        continue
                    topic = cells[cmap["topic"]]
                    if len(topic) < 3:
                        continue
                    hrs = cells[cmap["hours"]] if "hours" in cmap else ""
                    out.append({
                        "raw": topic,
                        "bt": cells[cmap["bt"]] if "bt" in cmap else "",
                        "unit": (cells[cmap["unit"]] or None) if "unit" in cmap else None,
                        "hours": int(re.search(r"\d+", hrs).group()) if re.search(r"\d+", hrs) else None,
                        "kind": cmap["kind"],
                    })
    return out


SKIP_LINE = re.compile(r"^(po\d+|peo\d+|pso\d+|mission|vision|sr no|credit|mode of|assessment|mst-|"
                       r"practical end|attendance|exam name|co vs po|r-|t-)", re.I)


def regex_chunks(pages: list[str]) -> list[dict]:
    """Fallback for PDFs whose tables pdfplumber cannot see. Noisy, but the taxonomy filters it."""
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
            if not in_sec or len(s) < 5 or SKIP_LINE.search(s):
                continue
            bt = re.search(r"CO\d+\s*-\s*BT[1-6]", s)
            hrs = re.search(r"(\d+)\s*(?:hrs?|hours?)\b", s, re.I)
            body = re.sub(r"^\d+\s+\d+\s+", "", s)
            body = re.sub(r"CO\d+\s*-\s*BT[1-6].*$", "", body)
            body = re.sub(r"\b[TR]-[A-Za-z0-9\s:,]+", "", body)
            if body.strip():
                out.append({"raw": body.strip(), "bt": bt.group(0) if bt else "", "unit": unit,
                            "hours": int(hrs.group(1)) if hrs else None, "kind": "theory"})
    return out


def llm_chunks(pages: list[str]) -> list[dict]:
    """Groq fallback for layouts that neither tables nor regex can read."""
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

    prompt = (
        "From this syllabus text, extract ONLY programming / data structures / algorithms / "
        "database-coding topics or lab experiments. Output ONLY a raw JSON array of objects with "
        "keys 'unit' (str or null), 'topic' (str), 'hours' (int or null). No Markdown. "
        "Ignore administrative, grading, book and outcome text.\n\n"
    )
    out = []
    for text in pages:
        if len(text.strip()) < 80:
            continue
        try:
            resp = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt + text}],
                model=LLM_MODEL, temperature=0.1)
            raw = re.sub(r"```json|```", "", resp.choices[0].message.content or "").strip()
            for it in json.loads(raw):
                if it.get("topic"):
                    out.append({"raw": it["topic"], "bt": "", "unit": it.get("unit"),
                                "hours": it.get("hours"), "kind": "theory"})
        except Exception as e:
            print(f"  Groq API call warning: {e}")
    return out


def split_topics(text: str) -> list[str]:
    """Split on , ; • but never inside parentheses."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if ch in ",;•" and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    cleaned = []
    for p in parts:
        p = re.sub(r"^(unit\s*[-:]?\s*\w+|experiment\s*\d*)\s*[:\-]?\s*", "", p.strip(), flags=re.I)
        p = p.strip(" -:,.\t")
        if len(p) >= 3:
            cleaned.append(p)
    return cleaned or [text]


def extract_chunks() -> list[dict]:
    chunks_all = []
    for pdf_path in sorted(SYLLABUS_DIR.glob("*.pdf")):
        pages = extract_pages(pdf_path)
        ok, reason = is_core_course(pages[0] if pages else "", pdf_path.name)
        if not ok:
            print(f"[SKIPPED] {pdf_path.name} -> {reason}")
            continue
        print(f"[PARSING] {pdf_path.name} -> {reason}")

        try:
            chunks = table_chunks(pdf_path)
        except Exception as e:
            print(f"  table extraction failed: {e}")
            chunks = []
        if len(chunks) < MIN_TOPICS:
            print(f"  Only {len(chunks)} table rows - trying line regex...")
            chunks = regex_chunks(pages)
        if len(chunks) < MIN_TOPICS:
            print(f"  Only {len(chunks)} regex rows - triggering Groq fallback...")
            chunks = llm_chunks(pages) or chunks

        for c in chunks:
            c["source"] = pdf_path.name
        print(f"  {len(chunks)} topic rows")
        chunks_all.extend(chunks)
    return chunks_all


# ------------------------------------------------------------------ STEP 3: match to coding skills
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print("Loading SentenceTransformer...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def match_chunks(chunks: list[dict]) -> list[dict]:
    """Returns evidence rows: {skill, confidence, match_type, fragment, chunk}"""
    evidence, unmatched = [], []

    for ch in chunks:
        text = ch["raw"]
        frags = split_topics(text)
        hit_any = False
        for sk in SKILLS:
            if any(p.search(text) for p in sk["patterns"]):
                frag = next((f for f in frags if any(p.search(f) for p in sk["patterns"])), text)
                evidence.append({"skill": sk, "confidence": 0.95, "match_type": "alias",
                                 "fragment": frag, "chunk": ch})
                hit_any = True
        if not hit_any:
            unmatched.extend((f, ch) for f in frags if len(f) >= 6)

    # strict embedding fallback, only against the curated coding taxonomy
    if unmatched and SKILLS:
        emb = get_embedder()
        from sentence_transformers import util
        sk_emb = emb.encode([s["sem_text"] for s in SKILLS], convert_to_numpy=True)
        fr_emb = emb.encode([f for f, _ in unmatched], batch_size=64, convert_to_numpy=True)
        sims = util.cos_sim(fr_emb, sk_emb).numpy()
        for (frag, ch), row in zip(unmatched, sims):
            j = int(row.argmax())
            if row[j] >= SEMANTIC_MIN:
                evidence.append({"skill": SKILLS[j], "confidence": round(float(row[j]), 4),
                                 "match_type": "semantic", "fragment": frag, "chunk": ch})
    return evidence


def aggregate(evidence: list[dict]) -> list[dict]:
    by_skill: dict[str, dict] = {}
    for ev in evidence:
        sk, ch, frag = ev["skill"], ev["chunk"], ev["fragment"]
        verb, depth = infer_depth(frag, ch["bt"], ch["kind"])
        rec = by_skill.get(sk["id"])
        if rec is None:
            rec = by_skill[sk["id"]] = {
                "skill_id": sk["id"], "skill_name": sk["name"], "category": sk["category"],
                "depth": depth, "confidence": ev["confidence"], "match_type": ev["match_type"],
                "original_topic": frag, "verb": verb, "unit": ch["unit"],
                "hours": ch["hours"], "source": ch["source"],
                "sources": [], "evidence": [],
            }
        else:
            if DEPTH_RANK[depth] > DEPTH_RANK[rec["depth"]]:
                rec["depth"] = depth
            if ev["confidence"] > rec["confidence"]:
                rec.update(confidence=ev["confidence"], match_type=ev["match_type"],
                           original_topic=frag, verb=verb, unit=ch["unit"], source=ch["source"])
            if ch["hours"]:
                rec["hours"] = (rec["hours"] or 0) + ch["hours"]
        if ch["source"] not in rec["sources"]:
            rec["sources"].append(ch["source"])
        if frag not in rec["evidence"] and len(rec["evidence"]) < 5:
            rec["evidence"].append(frag)
    return sorted(by_skill.values(), key=lambda r: (r["category"], r["skill_name"]))


# ------------------------------------------------------------------ STEP 4 (optional): link to ESCO
def link_esco(skills: list[dict]) -> None:
    for s in skills:
        s["esco_id"], s["esco_name"] = None, None
    if not (LINK_ESCO and ESCO_FILE.exists() and skills):
        return
    import pandas as pd
    from sentence_transformers import util
    emb = get_embedder()

    if CACHE_EMB.exists() and CACHE_META.exists():
        meta = json.loads(CACHE_META.read_text(encoding="utf-8"))
        esco_emb = np.load(CACHE_EMB)
    else:
        df = pd.read_csv(ESCO_FILE).dropna(subset=["preferredLabel"]).drop_duplicates("preferredLabel")
        labels = df["preferredLabel"].tolist()
        uris = df["conceptUri"].tolist() if "conceptUri" in df.columns else [None] * len(labels)
        meta = [{"id": u or "esco_" + hashlib.sha1(l.encode()).hexdigest()[:16], "name": l}
                for u, l in zip(uris, labels)]
        print(f"Embedding {len(labels)} ESCO labels (cached afterwards)...")
        esco_emb = emb.encode(labels, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        CACHE_EMB.parent.mkdir(parents=True, exist_ok=True)
        np.save(CACHE_EMB, esco_emb)
        CACHE_META.write_text(json.dumps(meta), encoding="utf-8")

    q = emb.encode([s["skill_name"] for s in skills], convert_to_numpy=True)
    sims = util.cos_sim(q, esco_emb).numpy()
    for s, row in zip(skills, sims):
        j = int(row.argmax())
        if row[j] >= ESCO_LINK_MIN:
            s["esco_id"], s["esco_name"] = meta[j]["id"], meta[j]["name"]
            # Keep skill_id aligned with T1's expected namespace
            s["skill_id"] = meta[j]["id"]

# ------------------------------------------------------------------ MAIN
def main():
    chunks = extract_chunks()
    if not chunks:
        print("No syllabus rows extracted.")
        return

    skills = aggregate(match_chunks(chunks))
    if not skills:
        print("No coding / DSA skills matched.")
        return
    link_esco(skills)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(skills, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved {len(skills)} coding/DSA skills to {OUTPUT_FILE}")
    cats: dict[str, int] = {}
    for s in skills:
        cats[s["category"]] = cats.get(s["category"], 0) + 1
    print("By category:", cats)
    for s in skills[:10]:
        print(f" - [{s['depth'].upper():5}] {s['skill_name']} ({s['confidence']}, {s['match_type']}) <- '{s['original_topic']}'")


if __name__ == "__main__":
    main()
