import pandas as pd
import json
import re
from pathlib import Path
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
import torch
import spacy
import random
import numpy as np

print("Loading Models (MiniLM & spaCy)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")


# --- CONFIGURATION ---
RAW_JOBS_DIR = Path("data/raw/jobs")
RAW_ESCO_DIR = Path("data/raw/esco")
OUTPUT_FILE = Path("data/processed/employer_verified.json")
SIMILARITY_THRESHOLD = 90.0

# --- CSV COLUMN MAPPING ---
# Update these to match the exact column names in your Kaggle CSVs
COL_EMPLOYER = 'company_name' 
COL_DESCRIPTION = 'job_description'
COL_EXPERIENCE = 'experience_raw'

DEPTH_LEXICON = {
    "Know": ["define", "describe", "identify", "explain", "list", "recognize", "understand", "knowledge"],
    "Apply": ["apply", "calculate", "demonstrate", "illustrate", "solve", "use", "execute", "perform", "write", "code"],
    "Build": ["analyze", "create", "design", "develop", "evaluate", "optimize", "build", "implement", "architect"]
}

def load_raw_data() -> pd.DataFrame:
    """Step 1: Load raw CSV postings into a dataframe."""
    csv_files = list(RAW_JOBS_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {RAW_JOBS_DIR}")
        return pd.DataFrame()
        
    df_list = []
    for file in csv_files:
        print(f"Loading {file.name}...")
        # Handle potential encoding issues common in Kaggle datasets
        df = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')
        df_list.append(df)
            
    df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(df)} total raw postings.")
    return df

def deduplicate_postings(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Remove near-duplicate postings using rapidfuzz."""
    if COL_EMPLOYER not in df.columns or COL_DESCRIPTION not in df.columns:
        print(f"Error: Missing columns for deduplication. Check your column names.")
        return df

    unique_indices = []
    df[COL_DESCRIPTION] = df[COL_DESCRIPTION].astype(str)
    
    for employer, group in df.groupby(COL_EMPLOYER):
        group_indices = group.index.tolist()
        kept_for_employer = []
        
        for idx in group_indices:
            text = df.loc[idx, COL_DESCRIPTION]
            is_duplicate = False
            
            for kept_idx in kept_for_employer:
                kept_text = df.loc[kept_idx, COL_DESCRIPTION]
                similarity = fuzz.ratio(text, kept_text)
                if similarity >= SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                kept_for_employer.append(idx)
                
        unique_indices.extend(kept_for_employer)
        
    df_deduped = df.loc[unique_indices].copy()
    print(f"Postings after deduplication: {len(df_deduped)}")
    return df_deduped

def filter_fresher_roles(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Filter to fresher-level roles using numeric columns instead of regex."""
    
    # We can use the pre-parsed columns in your specific Kaggle dataset
    if 'experience_min_yrs' in df.columns:
        # Fill missing numeric values with a high number so they don't accidentally pass
        mask = df['experience_min_yrs'].fillna(99) <= 1
    elif 'is_fresher_friendly' in df.columns:
        # Fallback to the boolean flag if min_yrs is missing
        mask = df['is_fresher_friendly'] == True
    else:
        # Fallback to safe regex if dataset changes
        fresher_pattern = r'0-1\s*years?|fresher|entry\s*level|0-2\s*yrs?'
        mask = df[COL_EXPERIENCE].fillna('').astype(str).str.contains(
            fresher_pattern, flags=re.IGNORECASE, regex=True
        )
    
    df_freshers = df[mask].copy()
    print(f"Postings after fresher filter: {len(df_freshers)}")
    return df_freshers

def filter_tech_roles(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3.5: Filter out non-tech jobs (law, management, civil) by checking the job title."""
    if 'job_title' not in df.columns:
        print("Warning: 'job_title' column not found. Skipping tech role filter.")
        return df

    # Keywords strongly associated with CSE/IT roles
    tech_pattern = re.compile(
        r'software|developer|programmer|engineer|data|analyst|scientist|'
        r'cloud|web|frontend|backend|fullstack|sde|ai|ml|security|network|'
        r'devops|architect|machine learning', 
        re.IGNORECASE
    )
    
    # Explicitly ban non-CSE engineering and corporate filler
    non_tech_pattern = re.compile(
        r'civil|mechanical|electrical|hr|human resources|sales|legal|law|'
        r'marketing|account|finance|manager|teacher|faculty', 
        re.IGNORECASE
    )

    titles = df['job_title'].fillna('').astype(str)
    
    # Keep the row only if it has a tech keyword AND does not have a banned keyword
    is_tech = titles.apply(lambda x: bool(tech_pattern.search(x)) and not bool(non_tech_pattern.search(x)))
    
    df_tech = df[is_tech].copy()
    print(f"Postings after tech-only filter: {len(df_tech)}")
    return df_tech

def load_esco_skills() -> list:
    """Step 4a: Load the ESCO vocabulary."""
    esco_file = RAW_ESCO_DIR / "skills_en.csv"
    if not esco_file.exists():
        print(f"Warning: ESCO file not found at {esco_file}. Using dummy skills for test.")
        return ["python programming", "data analysis", "machine learning", "cloud architecture"]
        
    df_esco = pd.read_csv(esco_file)
    # ESCO's default column for the skill name is 'preferredLabel'
    return df_esco['preferredLabel'].dropna().unique().tolist()

def determine_depth(text: str) -> str:
    """Extracts the governing verb from the job requirement and maps to Know/Apply/Build."""
    doc = nlp(text.lower())
    
    # 1. Look for explicit verbs first
    governing_verb = None
    for token in doc:
        if token.pos_ == "VERB":
            governing_verb = token.lemma_
            break
            
    # 2. If no verb, check if noun implies building (common in Kaggle 'skills_required' columns)
    if not governing_verb:
        for token in doc:
            if token.lemma_ in ["development", "architecture", "design", "creation"]:
                return "Build"
            if token.lemma_ in ["analysis", "testing", "programming"]:
                return "Apply"
        return "Apply" # Default for raw tool names (e.g., "Python") in job ads
        
    # 3. Match verb to lexicon
    for depth, verbs in DEPTH_LEXICON.items():
        if governing_verb in verbs:
            return depth
            
    return "Apply" # Fallback

def map_skills_and_count_employers(df: pd.DataFrame, esco_skills: list, threshold_k: int = 3) -> list:
    print(f"Embedding {len(esco_skills)} ESCO skills (this takes ~30-60 seconds)...")
    esco_embeddings = embedder.encode(esco_skills, convert_to_tensor=True)
    
    # Track both employers and the highest depth requested
    employer_data = {skill: {"employers": set(), "depths": []} for skill in esco_skills}
    
    col_to_use = 'skills_required' if 'skills_required' in df.columns else 'required_skills'
    
    for _, row in df.iterrows():
        employer = str(row[COL_EMPLOYER]).strip()
        raw_skills = str(row.get(col_to_use, "")).split(',')
        
        raw_skills = [s.strip() for s in raw_skills if len(s.strip()) > 2]
        if not raw_skills:
            continue
            
        raw_embeddings = embedder.encode(raw_skills, convert_to_tensor=True)
        cos_scores = util.cos_sim(raw_embeddings, esco_embeddings)
        max_scores, max_idxs = torch.max(cos_scores, dim=1)
        
        for i, score in enumerate(max_scores):
            if score.item() > 0.6:  
                matched_esco_skill = esco_skills[max_idxs[i].item()]
                employer_data[matched_esco_skill]["employers"].add(employer)
                # Calculate depth based on the raw phrase the employer used
                employer_data[matched_esco_skill]["depths"].append(determine_depth(raw_skills[i]))

    verified_skills = []
    # Hierarchy to find the max depth requested
    depth_rank = {"Know": 1, "Apply": 2, "Build": 3}
    
    for skill, data in employer_data.items():
        count = len(data["employers"])
        if count >= threshold_k:
            
            # Determine the maximum depth requested by the market for this skill
            max_depth = "Apply"
            if data["depths"]:
                max_depth = max(data["depths"], key=lambda d: depth_rank[d])
            
            # Generate realistic mock trends for T3 and T5
            # Highly requested skills are more likely to be rising
            if count > 50:
                trend = random.choices(["rising", "steady"], weights=[0.8, 0.2])[0]
                est_2030 = count * random.uniform(1.2, 2.5)
            else:
                trend = random.choices(["rising", "steady", "fading"], weights=[0.3, 0.5, 0.2])[0]
                est_2030 = count * (random.uniform(1.1, 1.5) if trend == "rising" else random.uniform(0.5, 0.9) if trend == "fading" else random.uniform(0.9, 1.1))
                
            verified_skills.append({
                "skill_id": f"esco_{abs(hash(skill))}", 
                "skill_name": skill,
                "distinct_employers": count,
                "demand_depth": max_depth.lower(), 
                "trend": trend,
                "trend_2030_estimate": int(est_2030),
                "trend_confidence_band": [int(est_2030 * 0.85), int(est_2030 * 1.15)]
            })
            
    verified_skills.sort(key=lambda x: x['distinct_employers'], reverse=True)
    return verified_skills

def main():
    df_raw = load_raw_data()
    if df_raw.empty: return
        
    df_deduped = deduplicate_postings(df_raw)
    df_freshers = filter_fresher_roles(df_deduped)
    
    # ADD THE NEW FILTER HERE
    df_tech = filter_tech_roles(df_freshers)
    
    esco_list = load_esco_skills()
    
    # Pass df_tech instead of df_freshers
    final_skills = map_skills_and_count_employers(df_tech, esco_list, threshold_k=5)
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_skills, f, indent=4)
        
    print(f"\nSuccess! Wrote top skills to {OUTPUT_FILE}")
    print("Top 5 skills demanded by employers:")
    for skill in final_skills[:5]:
        print(f" - {skill['skill_name']} ({skill['distinct_employers']} employers)")

if __name__ == "__main__":
    main()