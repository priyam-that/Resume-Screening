import json
from pathlib import Path
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

SKILLS_PATH = Path(__file__).parent / "skills.json"
with open(SKILLS_PATH, "r", encoding="utf-8") as f:
    SKILL_DICT = json.load(f)

CANONICAL_SKILLS = []
SKILL_TO_CATEGORY = {}
for category, skills in SKILL_DICT.items():
    for skill in skills:
        skill_lower = skill.lower()
        CANONICAL_SKILLS.append(skill_lower)
        SKILL_TO_CATEGORY[skill_lower] = category

vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(2, 4),
    min_df=1,
    lowercase=True
)

SKILL_VECTORS = vectorizer.fit_transform(CANONICAL_SKILLS)


def match_skill_to_canonical(skill: str, threshold: float = 0.5) -> Tuple[str, float]:
    skill_lower = skill.lower().strip()
    
    if skill_lower in SKILL_TO_CATEGORY:
        return skill_lower, 1.0
    
    for canonical in CANONICAL_SKILLS:
        if canonical in skill_lower or skill_lower in canonical:
            similarity = SequenceMatcher(None, skill_lower, canonical).ratio()
            if similarity >= threshold:
                return canonical, similarity
    
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
    skill_lower = skill.lower().strip()
    return SKILL_TO_CATEGORY.get(skill_lower, "unknown")


def enrich_skills(extracted_skills: dict, threshold: float = 0.5) -> dict:
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
    flat_skills = []
    for category, skills in enriched_skills.items():
        flat_skills.extend(skills)
    
    sorted_skills = sorted(
        flat_skills,
        key=lambda x: (-x["confidence"], x["canonical"])
    )
    
    return sorted_skills[:top_n]
