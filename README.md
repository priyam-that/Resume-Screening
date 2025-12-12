# AI Resume Screening System

Lightweight NLP resume engine that categorizes resumes into 100+ professional roles and extracts 300+ skills using ML and semantic matching. No PyTorch, ~100MB total.

## Features

- **Smart Classification**: 100+ roles across Tech, Finance, Marketing, HR, Operations
- **Top 3 Recommendations**: Multiple position matches with confidence scores
- **NLP Skill Extraction**: 300+ skills in 20 categories (programming, ML, cloud, etc.)
- **Semantic Matching**: Fuzzy skill matching using TF-IDF (e.g., "reactjs" → "react")
- **Lightweight**: Works with or without SpaCy (~100-125MB)
- **Multiple Interfaces**: CLI, Web UI (Streamlit), Python API

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train.py

# CLI usage
python predict_cli.py "Python, React, AWS, Docker..."
python predict_cli.py -v "Resume text..."  # verbose mode
python predict_cli.py --top 5 "Resume..."  # top 5 matches

# Web interface
streamlit run app.py
```

## Optional: Enhanced NLP

For better skill extraction accuracy:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

Without SpaCy, the system uses regex-based pattern matching (still works well).

## Python API

```python
from skill_extractor import extract_skills
from embedding_matcher import enrich_skills

skills = extract_skills("Resume text here...")
enriched = enrich_skills(skills)
```

## Architecture

- **ML Model**: TF-IDF + Logistic Regression with balanced class weights
- **Skill Extraction**: SpaCy NLP (optional) or regex-based fallback
- **Semantic Matching**: Character n-gram TF-IDF + cosine similarity
- **Role Database**: CSV with 100+ roles, keywords, certs, salary ranges

## Project Structure

```
├── train.py              # Model training
├── predict_cli.py        # CLI tool
├── app.py                # Streamlit web UI
├── skill_extractor.py    # NLP skill extraction
├── embedding_matcher.py  # Semantic matching
├── skills.json           # 300+ skills database
├── preprocessing.py      # Text cleaning
├── data_loader.py        # Data loading utilities
└── data/
    ├── resume_dataset.csv
    └── role_database.csv
```

## Adding New Roles/Skills

**Add role**: Edit `data/role_database.csv`
```csv
role_name,keywords,certifications,experience_level,description,salary_range,required_skills
```

**Add skills**: Edit `skills.json`
```json
{
  "category_name": ["skill1", "skill2", ...]
}
```

## License

MIT
