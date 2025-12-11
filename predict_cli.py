import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from preprocessing import clean_resume


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "resume_classifier.joblib"
ROLE_DB_PATH = BASE_DIR / "role_database.csv"


def load_role_database():
    """Load the role database with descriptions, certifications, and skills."""
    if not ROLE_DB_PATH.exists():
        return None
    return pd.read_csv(ROLE_DB_PATH)


def load_model():
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Model file not found at {MODEL_PATH}. Run 'python train.py' first."
        )
    return joblib.load(MODEL_PATH)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict resume category from text using the trained model."
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Resume text. If omitted, the script reads from stdin.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information including certifications and skills",
    )
    args = parser.parse_args(argv)

    if args.text:
        raw_text = args.text
    else:
        print("Reading resume text from stdin...", file=sys.stderr)
        raw_text = sys.stdin.read()

    if not raw_text.strip():
        raise SystemExit("No resume text provided.")

    # Load role database
    role_db = load_role_database()
    
    model = load_model()
    cleaned = clean_resume(raw_text)
    pred = model.predict([cleaned])[0]

    # Match against role database using keywords
    text_low = cleaned.lower()
    
    if role_db is not None:
        # Calculate match score for each role in database
        role_scores = {}
        for idx, row in role_db.iterrows():
            keywords = [k.strip() for k in row['keywords'].split(',')]
            score = sum(1 for k in keywords if k in text_low)
            if score > 0:
                role_scores[row['role_name']] = score
        
        # Find best match
        if role_scores:
            best_role = max(role_scores.items(), key=lambda x: x[1])
            # Override if we have strong evidence (3+ matches)
            if best_role[1] >= 3:
                pred = best_role[0]
    
    # Print result
    if args.verbose and role_db is not None:
        # Show detailed info
        role_info = role_db[role_db['role_name'] == pred]
        if not role_info.empty:
            info = role_info.iloc[0]
            print(f"Predicted Role: {pred}")
            print(f"Description: {info['description']}")
            print(f"Experience Level: {info['experience_level']}")
            print(f"Salary Range: {info['salary_range']}")
            print(f"Required Skills: {info['required_skills']}")
            print(f"Relevant Certifications: {info['certifications']}")
        else:
            print(pred)
    else:
        print(pred)


if __name__ == "__main__":
    main()

