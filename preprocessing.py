import re


def clean_resume(resume_text: str) -> str:
    """Clean a single resume string using simple regex rules.

    This mirrors the cleaning logic used in the notebook but is reusable
    from scripts, CLI tools, or web apps.
    """
    resume_text = re.sub(r"http\S+\s*", " ", resume_text)  # remove URLs
    resume_text = re.sub(r"RT|cc", " ", resume_text)  # remove RT and cc
    resume_text = re.sub(r"#\S+", "", resume_text)  # remove hashtags
    resume_text = re.sub(r"@\S+", "  ", resume_text)  # remove mentions
    resume_text = re.sub(r"[%s]" % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), " ", resume_text)
    resume_text = re.sub(r"[^\x00-\x7f]", " ", resume_text)
    resume_text = re.sub(r"\s+", " ", resume_text).strip()
    return resume_text
