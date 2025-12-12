"""
Skill extraction module that identifies technical and soft skills from resume text.
Uses SpaCy NLP and dictionary-based keyword matching (with fallback to regex-only mode).
"""
import json
import re
from pathlib import Path
from typing import List, Set

# Try to load SpaCy, but make it optional
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        print("⚠️  SpaCy model 'en_core_web_sm' not found.")
        print("    Install with: python -m spacy download en_core_web_sm")
        print("    Falling back to regex-only mode...\n")
        SPACY_AVAILABLE = False
        nlp = None
except ImportError:
    print("⚠️  SpaCy not installed. Using regex-only mode.")
    print("    For better results, install: pip install spacy")
    SPACY_AVAILABLE = False
    nlp = None

# Load skills dictionary
SKILLS_PATH = Path(__file__).parent / "skills.json"
with open(SKILLS_PATH, "r", encoding="utf-8") as f:
    SKILL_DICT = json.load(f)

# Flatten all skills and create a lookup set
ALL_SKILLS = {}
for category, skills in SKILL_DICT.items():
    for skill in skills:
        ALL_SKILLS[skill.lower()] = category

# Create multi-word skill patterns
MULTIWORD_SKILLS = sorted(
    [skill for skill in ALL_SKILLS.keys() if " " in skill],
    key=len,
    reverse=True
)


def extract_skills(text: str, min_confidence: float = 0.5) -> dict:
    """
    Extract skills from resume text.
    
    Args:
        text: Resume text
        min_confidence: Minimum confidence score (0-1)
    
    Returns:
        Dictionary with extracted skills organized by category
    """
    text_lower = text.lower()
    
    extracted_skills = {}
    found_skill_texts = set()
    
    # 1. Direct multi-word phrase matching (longest first)
    for skill in MULTIWORD_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            if skill not in found_skill_texts:
                category = ALL_SKILLS[skill]
                if category not in extracted_skills:
                    extracted_skills[category] = []
                extracted_skills[category].append(skill)
                found_skill_texts.add(skill)
    
    # 2. Single-word token matching
    if SPACY_AVAILABLE and nlp:
        doc = nlp(text_lower)
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
            token_text = token.text.lower()
            
            # Skip if already found as part of multi-word
            if token_text in found_skill_texts:
                continue
            
            # Check direct match
            if token_text in ALL_SKILLS:
                category = ALL_SKILLS[token_text]
                if category not in extracted_skills:
                    extracted_skills[category] = []
                if token_text not in extracted_skills[category]:
                    extracted_skills[category].append(token_text)
                    found_skill_texts.add(token_text)
        
        # 3. Noun chunk extraction (SpaCy only)
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            if chunk_text in ALL_SKILLS and chunk_text not in found_skill_texts:
                category = ALL_SKILLS[chunk_text]
                if category not in extracted_skills:
                    extracted_skills[category] = []
                extracted_skills[category].append(chunk_text)
                found_skill_texts.add(chunk_text)
    else:
        # Fallback: Simple regex word matching without SpaCy
        words = re.findall(r'\b[a-z]+(?:\+\+|#)?\b', text_lower)
        for word in words:
            if word in ALL_SKILLS and word not in found_skill_texts:
                category = ALL_SKILLS[word]
                if category not in extracted_skills:
                    extracted_skills[category] = []
                extracted_skills[category].append(word)
                found_skill_texts.add(word)
    
    # Deduplicate
    for category in extracted_skills:
        extracted_skills[category] = list(set(extracted_skills[category]))
    
    return extracted_skills


def get_skills_by_category(extracted_skills: dict, category: str) -> List[str]:
    """Get skills for a specific category."""
    return extracted_skills.get(category, [])


def flatten_skills(extracted_skills: dict) -> List[str]:
    """Flatten extracted skills into a single list."""
    return [
        skill 
        for skills_list in extracted_skills.values() 
        for skill in skills_list
    ]


def get_skill_categories() -> dict:
    """Return the skill dictionary."""
    return SKILL_DICT
