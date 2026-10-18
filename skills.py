"""Skills Dictionary and Extraction module for HireSense AI.

Provides a comprehensive, categorized skill taxonomy, fuzzy/synonym matching,
and configurable skill extraction from resume text and job descriptions.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Any, Tuple


# Default categorized skills taxonomy with canonical names and regex match patterns
DEFAULT_SKILLS_TAXONOMY: Dict[str, Dict[str, str]] = {
    "Programming Languages": {
        "Python": r"\bpython\b",
        "Java": r"\bjava\b(?!script)",
        "JavaScript": r"\b(?:javascript|js)\b",
        "TypeScript": r"\b(?:typescript|ts)\b",
        "C++": r"\b(?:c\+\+|cplusplus)\b",
        "C#": r"\b(?:c\#|csharp)\b",
        "C": r"\b(?<!\w)c(?!\w|[+#])\b",
        "Go": r"\b(?:golang|go)\b(?!\s+(?:to|for|with))",
        "Rust": r"\brust\b",
        "Ruby": r"\bruby\b",
        "PHP": r"\bphp\b",
        "Swift": r"\bswift\b",
        "Kotlin": r"\bkotlin\b",
        "SQL": r"\bsql\b",
        "R": r"\b(?<!\w)r(?!\w|[+&])\b",
        "Scala": r"\bscala\b",
        "Dart": r"\bdart\b",
        "MATLAB": r"\bmatlab\b",
        "Bash/Shell": r"\b(?:bash|shell\s*scripting|zsh)\b",
    },
    "Web & Backend Frameworks": {
        "React": r"\b(?:react|react\.?js)\b",
        "Angular": r"\bangular(?:\.?js)?\b",
        "Vue.js": r"\b(?:vue|vue\.?js)\b",
        "Next.js": r"\bnext\.?js\b",
        "Node.js": r"\bnode(?:\.?js)?\b",
        "Express": r"\bexpress(?:\.?js)?\b",
        "Django": r"\bdjango\b",
        "FastAPI": r"\bfastapi\b",
        "Flask": r"\bflask\b",
        "Spring Boot": r"\bspring\s*boot\b",
        "ASP.NET": r"\b(?:asp\.?net|\.net(?:\s*core)?)\b",
        "Ruby on Rails": r"\b(?:rails|ruby\s+on\s+rails)\b",
        "GraphQL": r"\bgraphql\b",
        "REST API": r"\b(?:rest|restful|rest\s*apis?)\b",
        "HTML/CSS": r"\b(?:html|html5|css|css3|sass|tailwind|bootstrap)\b",
    },
    "Data Science & AI/ML": {
        "Machine Learning": r"\b(?:machine\s*learning|ml)\b",
        "Deep Learning": r"\bdeep\s*learning\b",
        "NLP": r"\b(?:nlp|natural\s+language\s+processing)\b",
        "Computer Vision": r"\b(?:computer\s*vision|cv)\b",
        "Scikit-Learn": r"\b(?:scikit[\s-]*learn|sklearn)\b",
        "TensorFlow": r"\btensorflow\b",
        "PyTorch": r"\bpytorch\b",
        "Keras": r"\bkeras\b",
        "Pandas": r"\bpandas\b",
        "NumPy": r"\bnumpy\b",
        "OpenCV": r"\bopencv\b",
        "Large Language Models (LLMs)": r"\b(?:llms?|large\s+language\s+models?)\b",
        "Generative AI": r"\b(?:generative\s*ai|genai)\b",
        "Transformers": r"\b(?:transformers?|hugging\s*face)\b",
        "Retrieval-Augmented Generation (RAG)": r"\b(?:rag|retrieval[\s-]augmented\s+generation)\b",
        "XGBoost": r"\bxgboost\b",
        "Data Analysis": r"\bdata\s+analysis\b",
        "Data Visualization": r"\b(?:data\s+visualization|power\s*bi|tableau|matplotlib|seaborn)\b",
    },
    "Databases & Storage": {
        "PostgreSQL": r"\b(?:postgresql|postgres)\b",
        "MySQL": r"\bmysql\b",
        "MongoDB": r"\bmongodb\b",
        "Redis": r"\bredis\b",
        "SQLite": r"\bsqlite\b",
        "Oracle": r"\boracle(?:\s+db)?\b",
        "Cassandra": r"\bcassandra\b",
        "Elasticsearch": r"\belasticsearch\b",
        "DynamoDB": r"\bdynamodb\b",
        "Snowflake": r"\bsnowflake\b",
        "BigQuery": r"\bbigquery\b",
    },
    "Cloud & DevOps": {
        "AWS": r"\b(?:aws|amazon\s+web\s+services)\b",
        "Azure": r"\b(?:azure|microsoft\s+azure)\b",
        "Google Cloud (GCP)": r"\b(?:gcp|google\s+cloud(?:\s+platform)?)\b",
        "Docker": r"\bdocker\b",
        "Kubernetes": r"\b(?:kubernetes|k8s)\b",
        "Terraform": r"\bterraform\b",
        "CI/CD": r"\b(?:ci[\/\\]cd|continuous\s+integration|github\s*actions|gitlab\s*ci|jenkins)\b",
        "Linux": r"\b(?:linux|ubuntu|centos|debian)\b",
        "Git": r"\b(?:git|github|gitlab)\b",
    },
    "Methodologies & Professional Skills": {
        "Agile/Scrum": r"\b(?:agile|scrum|kanban)\b",
        "System Design": r"\bsystem\s+design\b",
        "Microservices": r"\bmicroservices\b",
        "Unit Testing": r"\b(?:unit\s+testing|pytest|unittest|tdd)\b",
        "Problem Solving": r"\bproblem\s+solving\b",
        "Team Leadership": r"\b(?:leadership|team\s+lead|mentoring)\b",
        "Communication": r"\b(?:communication\s+skills?|cross-functional\s+collaboration)\b",
        "Project Management": r"\bproject\s+management\b",
    },
}


def build_flat_skill_patterns(
    taxonomy: Dict[str, Dict[str, str]] | None = None,
) -> Dict[str, re.Pattern]:
    """Compiles flat lookup dictionary of canonical_skill_name -> compiled regex pattern."""
    tax = taxonomy or DEFAULT_SKILLS_TAXONOMY
    patterns: Dict[str, re.Pattern] = {}
    for _, skills in tax.items():
        for canonical_name, pattern_str in skills.items():
            patterns[canonical_name] = re.compile(pattern_str, re.IGNORECASE)
    return patterns


_FLAT_PATTERNS = build_flat_skill_patterns()


def parse_skill_string(skill_text: str) -> List[str]:
    """Parses a comma, semicolon, or newline separated list of skills entered by HR.

    Cleans up whitespace, removes duplicates while preserving order.
    """
    if not skill_text or not skill_text.strip():
        return []

    # Split by commas, semicolons, or newlines
    raw_tokens = re.split(r"[,;\n\r]+", skill_text)
    cleaned_skills: List[str] = []
    seen: Set[str] = set()

    for token in raw_tokens:
        clean_token = token.strip()
        if clean_token and clean_token.lower() not in seen:
            cleaned_skills.append(clean_token)
            seen.add(clean_token.lower())

    return cleaned_skills


def extract_skills_from_text(
    text: str,
    taxonomy: Dict[str, Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """Extracts detected skills from text against the configured taxonomy.

    Returns:
    {
        "all_skills": List[str] (canonical skill names),
        "by_category": Dict[str, List[str]] (category -> list of skills),
        "total_count": int
    }
    """
    if not text:
        return {"all_skills": [], "by_category": {}, "total_count": 0}

    tax = taxonomy or DEFAULT_SKILLS_TAXONOMY
    all_matched: Set[str] = set()
    by_category: Dict[str, List[str]] = {}

    for category, skills in tax.items():
        cat_matches: List[str] = []
        for canonical, pattern_str in skills.items():
            pattern = re.compile(pattern_str, re.IGNORECASE)
            if pattern.search(text):
                cat_matches.append(canonical)
                all_matched.add(canonical)
        if cat_matches:
            by_category[category] = cat_matches

    return {
        "all_skills": sorted(list(all_matched)),
        "by_category": by_category,
        "total_count": len(all_matched),
    }


def match_target_skills(
    candidate_text: str,
    target_skills: List[str],
) -> Tuple[List[str], List[str]]:
    """Checks which target skills (required or preferred) are present in candidate text.

    Supports both taxonomy-known patterns and custom user-provided skill keywords.

    Returns:
        (matched_skills, missing_skills)
    """
    if not target_skills:
        return [], []

    matched: List[str] = []
    missing: List[str] = []

    for skill in target_skills:
        skill_clean = skill.strip()
        if not skill_clean:
            continue

        # Check if skill matches a known pattern
        pattern = _FLAT_PATTERNS.get(skill_clean)
        if pattern is None:
            # Check case-insensitively in flat pattern keys
            for canon, pat in _FLAT_PATTERNS.items():
                if canon.lower() == skill_clean.lower():
                    pattern = pat
                    break

        if pattern is not None:
            is_present = bool(pattern.search(candidate_text))
        else:
            # Fallback regex for custom skill keyword
            # Escape regex special characters, allow word boundaries
            escaped = re.escape(skill_clean)
            # If skill contains non-word characters (like C++ or .NET), boundary needs care
            pattern_str = rf"(?:\b|(?<=\s)){escaped}(?:\b|(?=\s|$|[,\.;]))"
            custom_pattern = re.compile(pattern_str, re.IGNORECASE)
            is_present = bool(custom_pattern.search(candidate_text))

        if is_present:
            matched.append(skill_clean)
        else:
            missing.append(skill_clean)

    return matched, missing
