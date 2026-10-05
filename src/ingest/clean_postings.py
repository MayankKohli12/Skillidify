import pandas as pd
import json
import re
from pathlib import Path
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

print("Loading MiniLM embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

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

def load_esco_skills() -> list:
    """Step 4a: Load the ESCO vocabulary."""
    esco_file = RAW_ESCO_DIR / "skills_en.csv"
    if not esco_file.exists():
        print(f"Warning: ESCO file not found at {esco_file}. Using dummy skills for test.")
        return ["python programming", "data analysis", "machine learning", "cloud architecture"]
        
    df_esco = pd.read_csv(esco_file)
    # ESCO's default column for the skill name is 'preferredLabel'
    return df_esco['preferredLabel'].dropna().unique().tolist()

def map_skills_and_count_employers(df: pd.DataFrame, esco_skills: list, threshold_k: int = 3) -> list:
    """Steps 4b, 5 & 6: Match skills, count distinct employers, apply threshold."""
    print(f"Embedding {len(esco_skills)} ESCO skills (this takes ~30-60 seconds)...")
    esco_embeddings = embedder.encode(esco_skills, convert_to_tensor=True)
    
    employer_counts = {skill: set() for skill in esco_skills}
    
    print("Mapping job posting skills to ESCO vocabulary...")
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
            if score.item() > 0.6:  # Confidence threshold
                matched_esco_skill = esco_skills[max_idxs[i].item()]
                employer_counts[matched_esco_skill].add(employer)

    verified_skills = []
    for skill, employers in employer_counts.items():
        count = len(employers)
        if count >= threshold_k:
            verified_skills.append({
                "skill_id": f"esco_{abs(hash(skill))}", 
                "skill_name": skill,
                "distinct_employers": count,
                "demand_depth": "apply", 
                "trend": "steady",
                "trend_2030_estimate": 0,
                "trend_confidence_band": [0, 0]
            })
            
    verified_skills.sort(key=lambda x: x['distinct_employers'], reverse=True)
    print(f"\nExtracted {len(verified_skills)} verified skills meeting threshold k={threshold_k}")
    return verified_skills

def main():
    df_raw = load_raw_data()
    if df_raw.empty: return
        
    df_deduped = deduplicate_postings(df_raw)
    df_freshers = filter_fresher_roles(df_deduped)
    
    esco_list = load_esco_skills()
    
    # We set k=5 meaning 5 DIFFERENT employers must ask for the skill for it to count
    final_skills = map_skills_and_count_employers(df_freshers, esco_list, threshold_k=5)
    
    # Write Step 7: employer_verified.json
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_skills, f, indent=4)
        
    print(f"\nSuccess! Wrote top skills to {OUTPUT_FILE}")
    print("Top 5 skills demanded by employers:")
    for skill in final_skills[:5]:
        print(f" - {skill['skill_name']} ({skill['distinct_employers']} employers)")

if __name__ == "__main__":
    main()