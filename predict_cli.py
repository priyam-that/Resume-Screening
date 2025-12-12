import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from preprocessing import clean_resume


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "resume_classifier.joblib"
ROLE_DB_PATH = BASE_DIR / "data" / "role_database.csv"


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
    parser.add_argument(
        "--top",
        "-t",
        type=int,
        default=3,
        help="Number of top matching roles to show (default: 3)",
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
    
    top_matches = []
    
    if role_db is not None:
        # Calculate match score for each role in database
        role_scores = {}
        for idx, row in role_db.iterrows():
            keywords = [k.strip() for k in row['keywords'].split(',')]
            score = sum(1 for k in keywords if k in text_low)
            if score > 0:
                role_scores[row['role_name']] = score
        
        # Get top N matches
        if role_scores:
            sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)
            top_matches = sorted_roles[:args.top]
            
            # Use the best match as primary prediction if strong evidence
            if top_matches[0][1] >= 3:
                pred = top_matches[0][0]
    
    # Print results
    if args.verbose and role_db is not None and top_matches:
        print(f"{'='*80}")
        print(f"TOP {len(top_matches)} RECOMMENDED POSITIONS FOR YOUR RESUME")
        print(f"{'='*80}\n")
        
        for rank, (role_name, score) in enumerate(top_matches, 1):
            role_info = role_db[role_db['role_name'] == role_name]
            if not role_info.empty:
                info = role_info.iloc[0]
                print(f"#{rank} - {role_name} (Match Score: {score})")
                print(f"{'-'*80}")
                print(f"📋 Description: {info['description']}")
                print(f"💼 Experience Level: {info['experience_level']}")
                print(f"💰 Salary Range: ${info['salary_range']}")
                print(f"🛠️  Required Skills: {info['required_skills']}")
                print(f"📜 Relevant Certifications: {info['certifications']}")
                print(f"\n")
    elif top_matches:
        # Show just the role names
        print(f"Top {len(top_matches)} matching positions:")
        for rank, (role_name, score) in enumerate(top_matches, 1):
            print(f"  {rank}. {role_name} (Match Score: {score})")
    else:
        print(pred)


if __name__ == "__main__":
    main()

