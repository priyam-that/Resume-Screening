# Setup Guide

## Quick Start

1. **Install required dependencies:**
```bash
pip install -r requirements.txt
```

2. **Train the model:**
```bash
python train.py
```

3. **Run the Streamlit app:**
```bash
streamlit run app.py
```

## Optional: Enhanced NLP Features

For better skill extraction accuracy, you can optionally install SpaCy:

```bash
# Install SpaCy (optional, ~25MB)
pip install spacy

# Download the English language model
python -m spacy download en_core_web_sm
```

**Note:** The skill extraction works in two modes:
- **Without SpaCy**: Uses regex-based pattern matching (fast, lightweight)
- **With SpaCy**: Uses NLP for better context understanding (more accurate)

## Dependency Sizes

| Package | Size | Required |
|---------|------|----------|
| scikit-learn | ~30MB | ✅ Yes |
| pandas | ~15MB | ✅ Yes |
| streamlit | ~20MB | ✅ Yes |
| spacy | ~10MB | ❌ Optional |
| en_core_web_sm | ~15MB | ❌ Optional |

**Total:**
- Minimum: ~100MB
- With SpaCy: ~125MB

## Testing

Run the test suite to verify installation:

```bash
# Test skill extraction and matching
python test_skills.py

# Test CLI predictions
python predict_cli.py "Your resume text here..."

# Test with verbose output
python predict_cli.py -v "Your resume text here..."
```

## Features

✅ **100+ Professional Roles** - Comprehensive role database  
✅ **Multi-Position Recommendations** - Get top 3 matching positions  
✅ **Skill Extraction** - NLP-based skill identification (300+ skills)  
✅ **Semantic Matching** - TF-IDF based fuzzy skill matching  
✅ **Lightweight** - No PyTorch, no heavy ML frameworks  
✅ **Fast** - Instant predictions and skill analysis  

## Troubleshooting

### SpaCy Installation Issues

If you encounter issues installing SpaCy, you can skip it:
- The system will automatically fall back to regex-only mode
- Skill extraction will still work, just slightly less accurate

### Virtual Environment

If you get import errors, make sure you're in the virtual environment:

```bash
# Activate venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```
