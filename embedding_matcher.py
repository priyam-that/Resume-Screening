"""
Lightweight semantic skill matcher using TF-IDF and cosine similarity.
Maps user-written skills to canonical skill names without heavy dependencies.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load skills dictionary
SKILLS_PATH = Path(__file__).parent / "skills.json"
with open(SKILLS_PATH, "r", encoding="utf-8") as f:
    SKILL_DICT = json.load(f)

# Flatten and create canonical skill list
CANONICAL_SKILLS = []
SKILL_TO_CATEGORY = {}
for category, skills in SKILL_DICT.items():
    for skill in skills:
        skill_lower = skill.lower()
        CANONICAL_SKILLS.append(skill_lower)
        SKILL_TO_CATEGORY[skill_lower] = category

# Initialize TF-IDF vectorizer for semantic matching
vectorizer = TfidfVectorizer(
    analyzer='char_wb',  # Character n-grams for fuzzy matching
    ngram_range=(2, 4),  # 2-4 character n-grams
    min_df=1,
    lowercase=True
)

# Fit vectorizer on canonical skills
SKILL_VECTORS = vectorizer.fit_transform(CANONICAL_SKILLS)


def match_skill_to_canonical(skill: str, threshold: float = 0.5) -> Tuple[str, float]:
    """
    Match a user-written skill to canonical skill using TF-IDF + cosine similarity.
    
    Args:
        skill: User-written skill string
        threshold: Minimum similarity score (0-1) to accept match
    
    Returns:
        Tuple of (canonical_skill, confidence_score)
    """
    skill_lower = skill.lower().strip()
    
    # Direct exact match first (fastest)
    if skill_lower in SKILL_TO_CATEGORY:
        return skill_lower, 1.0
    
    # Try substring match (e.g., "pytorch" in "deep learning pytorch")
    for canonical in CANONICAL_SKILLS:
        if canonical in skill_lower or skill_lower in canonical:
            # Use string similarity for confidence
            similarity = SequenceMatcher(None, skill_lower, canonical).ratio()
            if similarity >= threshold:
                return canonical, similarity
    
    # TF-IDF semantic matching for fuzzy matches
    try:
        skill_vector = vectorizer.transform([skill_lower])
        similarities = cosine_similarity(skill_vector, SKILL_VECTORS)[0]
        
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        best_match = CANONICAL_SKILLS[best_idx]
        
        if best_score >= threshold:
            return best_match, best_score
    except:
        pass
    
    # Return original if no good match found
    return skill_lower, 0.0


def match_skills_batch(skills: List[str], threshold: float = 0.5) -> Dict[str, Tuple[str, float]]:
    """
    Match multiple skills to canonical skills.
    
    Args:
        skills: List of user-written skills
        threshold: Minimum similarity threshold
    
    Returns:
        Dictionary mapping original skill → (canonical_skill, confidence)
    """
    matches = {}
    for skill in skills:
        canonical, confidence = match_skill_to_canonical(skill, threshold)
        matches[skill] = (canonical, confidence)
    return matches


def get_skill_category(skill: str) -> str:
    """Get the category of a canonical skill."""
    skill_lower = skill.lower().strip()
    return SKILL_TO_CATEGORY.get(skill_lower, "unknown")


def enrich_skills(extracted_skills: dict, threshold: float = 0.5) -> dict:
    """
    Enrich extracted skills with semantic matching and confidence scores.
    
    Args:
        extracted_skills: Dictionary of skills by category from skill_extractor
        threshold: Minimum similarity threshold for fuzzy matching
    
    Returns:
        Enriched dictionary with confidence scores and canonical forms
    """
    enriched = {}
    
    for category, skills in extracted_skills.items():
        enriched[category] = []
        
        for skill in skills:
            canonical, confidence = match_skill_to_canonical(skill, threshold)
            enriched[category].append({
                "original": skill,
                "canonical": canonical,
                "confidence": float(confidence),
                "category": get_skill_category(canonical)
            })
    
    return enriched


def get_top_skills(enriched_skills: dict, top_n: int = 20) -> List[dict]:
    """
    Get top N skills by confidence score.
    
    Args:
        enriched_skills: Dictionary from enrich_skills()
        top_n: Number of top skills to return
    
    Returns:
        Sorted list of top skills with metadata
    """
    flat_skills = []
    for category, skills in enriched_skills.items():
        flat_skills.extend(skills)
    
    # Sort by confidence descending, then alphabetically
    sorted_skills = sorted(
        flat_skills,
        key=lambda x: (-x["confidence"], x["canonical"])
    )
    
    return sorted_skills[:top_n]
