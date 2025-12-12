from skill_extractor import extract_skills
from embedding_matcher import match_skill_to_canonical, enrich_skills

test_resume = """
Senior Software Engineer with 5 years of experience in Python, JavaScript, and React.
Expertise in machine learning using TensorFlow and PyTorch. 
Proficient with AWS cloud services, Docker containerization, and CI/CD pipelines.
Strong background in data structures, algorithms, and system design.
Experience with MongoDB, PostgreSQL databases and RESTful API development.
"""

print("=" * 60)
print("Testing Lightweight Skill Extraction & Matching")
print("=" * 60)

print("\n1. Extracting skills from resume...")
extracted = extract_skills(test_resume)

print("\nExtracted skills by category:")
for category, skills in extracted.items():
    if skills:
        print(f"\n  {category}:")
        for skill in skills:
            print(f"    • {skill}")

# Test individual matching
print("\n" + "=" * 60)
print("2. Testing fuzzy skill matching:")
print("=" * 60)

test_skills = [
    "python",
    "reactjs",  # Should match "react"
    "tensorflow",
    "aws",
    "mongodb",
    "deep learning"
]

for skill in test_skills:
    canonical, confidence = match_skill_to_canonical(skill)
    print(f"\n  '{skill}' → '{canonical}' (confidence: {confidence:.2f})")

print("\n" + "=" * 60)
print("3. Testing skill enrichment:")
print("=" * 60)

enriched = enrich_skills(extracted)
print(f"\nEnriched {sum(len(v) for v in enriched.values())} total skills")

from embedding_matcher import get_top_skills
top = get_top_skills(enriched, top_n=10)

print("\nTop 10 skills with confidence:")
for skill in top:
    print(f"  • {skill['canonical']} ({skill['category']}) - {skill['confidence']:.2f}")

print("\n" + "=" * 60)
print("✅ All tests completed successfully!")
print("=" * 60)
