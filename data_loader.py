import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd

from preprocessing import clean_resume


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


def extract_years_experience(text: str) -> int:
    """Extract years of experience from resume text using regex patterns."""
    text_lower = text.lower()
    
    # Common patterns: "5 years", "5+ years", "5-7 years", "five years"
    patterns = [
        r"(\d+)\s*\+?\s*years?\s+(?:of\s+)?experience",
        r"experience\s+of\s+(\d+)\s*\+?\s*years?",
        r"(\d+)\s*-\s*\d+\s+years?",
    ]
    
    years = []
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        years.extend([int(m) for m in matches])
    
    return max(years) if years else 0


def categorize_experience_level(years: int) -> str:
    """Categorize experience into junior/mid/senior levels."""
    if years == 0:
        return "unknown"
    elif years <= 2:
        return "junior"
    elif years <= 5:
        return "mid"
    else:
        return "senior"


def load_csv_data(file_path: Path) -> pd.DataFrame:
    """Load resume data from CSV file."""
    df = pd.read_csv(file_path, encoding="utf-8")
    
    # Normalize column names
    if "Resume" in df.columns and "Category" in df.columns:
        df = df[["Resume", "Category"]].copy()
        df.columns = ["text", "category"]
    elif "resume" in df.columns and "category" in df.columns:
        df.columns = ["text", "category"]
    elif "text" in df.columns and "label" in df.columns:
        df.columns = ["text", "category"]
    
    return df


def load_json_data(file_path: Path) -> pd.DataFrame:
    """Load resume data from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Support both array of objects and single object with arrays
    if isinstance(data, list):
        return pd.DataFrame(data)
    elif isinstance(data, dict):
        return pd.DataFrame(data)
    
    raise ValueError(f"Unsupported JSON structure in {file_path}")


def load_text_directory(dir_path: Path) -> pd.DataFrame:
    """Load resumes from text files organized in category subdirectories."""
    records = []
    
    for category_dir in dir_path.iterdir():
        if not category_dir.is_dir():
            continue
        
        category = category_dir.name
        for text_file in category_dir.glob("*.txt"):
            with open(text_file, "r", encoding="utf-8") as f:
                text = f.read()
            records.append({"text": text, "category": category})
    
    return pd.DataFrame(records)


def load_all_data() -> pd.DataFrame:
    """Load and combine data from all supported formats in the data/ directory."""
    all_data = []
    
    # Load existing CSV if it exists at root
    root_csv = BASE_DIR / "data" / "resume_dataset.csv"
    if root_csv.exists():
        df = load_csv_data(root_csv)
        all_data.append(df)
    
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(exist_ok=True)
    
    # Load CSV files from data/
    for csv_file in DATA_DIR.glob("*.csv"):
        df = load_csv_data(csv_file)
        all_data.append(df)
    
    # Load JSON files from data/
    for json_file in DATA_DIR.glob("*.json"):
        df = load_json_data(json_file)
        all_data.append(df)
    
    # Load text files organized by category (data/category_name/*.txt)
    text_dir = DATA_DIR / "text_resumes"
    if text_dir.exists():
        df = load_text_directory(text_dir)
        all_data.append(df)
    
    if not all_data:
        raise ValueError("No data found. Please add CSV, JSON, or text files to the data/ directory.")
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    
    # Ensure we have required columns
    if "text" not in combined.columns or "category" not in combined.columns:
        raise ValueError("Combined data must have 'text' and 'category' columns")
    
    # Clean text and extract experience
    combined["cleaned_text"] = combined["text"].apply(clean_resume)
    combined["years_experience"] = combined["text"].apply(extract_years_experience)
    combined["experience_level"] = combined["years_experience"].apply(categorize_experience_level)
    
    return combined


def get_data_by_level(df: pd.DataFrame, level: str) -> Tuple[List[str], List[str]]:
    """Filter data by experience level and return X, y arrays."""
    if level == "all":
        subset = df
    else:
        subset = df[df["experience_level"] == level]
    
    if subset.empty:
        raise ValueError(f"No data found for experience level: {level}")
    
    X = subset["cleaned_text"].values.tolist()
    y = subset["category"].values.tolist()
    
    return X, y
