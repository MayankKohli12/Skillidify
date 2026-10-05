import pandas as pd
import json
import re
from pathlib import Path
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
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



def main():
    # Execute the pipeline up to the filter step
    df_raw = load_raw_data()
    if df_raw.empty:
        return
        
    # Print the columns so you can verify what names to put in the CONFIG variables
    print(f"\nDetected columns: {df_raw.columns.tolist()}\n")
    
    df_deduped = deduplicate_postings(df_raw)
    df_freshers = filter_fresher_roles(df_deduped)
    
    print("\nInitial pipeline test complete. Please check the column names printed above.")

if __name__ == "__main__":
    main()