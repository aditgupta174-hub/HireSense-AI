"""Scoring and Match Analysis module for HireSense AI.

Calculates reproducible TF-IDF cosine similarity, skill-match scores,
overall weighted scores, and generates human-readable explanations and recommendations.
"""

from __future__ import annotations

import math
from typing import Dict, List, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessor import clean_text_for_nlp
from src.skills import (
    extract_skills_from_text,
    match_target_skills,
    parse_skill_string,
)


def compute_tfidf_similarity(resume_text: str, jd_text: str) -> float:
    """Computes cosine similarity between cleaned resume and job description using TF-IDF.

    Returns a score between 0.0 and 100.0.
    """
    clean_resume = clean_text_for_nlp(resume_text, remove_stopwords=True)
    clean_jd = clean_text_for_nlp(jd_text, remove_stopwords=True)

    if not clean_resume.strip() or not clean_jd.strip():
        return 0.0

    try:
        # Using unigram and bigram ranges for domain phrase capturing
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform([clean_resume, clean_jd])
        sim_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        score = float(sim_matrix[0][0]) * 100.0
        # Clip to [0, 100]
        return round(max(0.0, min(100.0, score)), 1)
    except Exception:
        return 0.0


def calculate_skill_match_score(
    resume_text: str,
    required_skills: List[str],
    preferred_skills: Optional[List[str]] = None,
    jd_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculates transparent skill-match score and identifies matched / missing skills.

    Scoring logic:
    - If required skills are entered:
        If preferred skills also provided:
          Skill Score = (0.80 * Required Match Ratio + 0.20 * Preferred Match Ratio) * 100
        Else:
          Skill Score = (Required Match Ratio) * 100
    - If NO required skills are entered (Fallback mode):
        Extracts skills automatically from the Job Description text.
        If JD contains skills:
          Skill Score = (Matched JD Skills / Total JD Skills) * 100
        Else:
          Returns fallback note (score None or aligned with TF-IDF similarity).
    """
    preferred_skills = preferred_skills or []
    fallback_used = False
    fallback_note = ""

    # Check if required skills were explicitly provided
    if required_skills:
        effective_required = required_skills
    else:
        # Fallback: extract from JD text
        fallback_used = True
        jd_skills_info = extract_skills_from_text(jd_text or "")
        effective_required = jd_skills_info["all_skills"]
        if effective_required:
            fallback_note = (
                f"No required skills entered by HR; automatically benchmarked against "
                f"{len(effective_required)} skills detected in the Job Description."
            )
        else:
            fallback_note = (
                "No required skills provided and none auto-detected in the Job Description. "
                "Skill match score defaulted to text similarity."
            )

    # Match required skills
    matched_req, missing_req = match_target_skills(resume_text, effective_required)

    # Match preferred skills (if any)
    matched_pref, missing_pref = match_target_skills(resume_text, preferred_skills)

    # Calculate score
    if not effective_required and not preferred_skills:
        skill_score = None  # Signals caller to fall back to TF-IDF text similarity
    elif effective_required and not preferred_skills:
        req_ratio = len(matched_req) / len(effective_required)
        skill_score = round(req_ratio * 100.0, 1)
    elif not effective_required and preferred_skills:
        pref_ratio = len(matched_pref) / len(preferred_skills)
        skill_score = round(pref_ratio * 100.0, 1)
    else:
        # Both required and preferred present
        req_ratio = len(matched_req) / len(effective_required)
        pref_ratio = len(matched_pref) / len(preferred_skills)
        composite = (0.80 * req_ratio + 0.20 * pref_ratio) * 100.0
        skill_score = round(composite, 1)

    return {
        "skill_score": skill_score,
        "effective_required": effective_required,
        "matched_required": matched_req,
        "missing_required": missing_req,
        "matched_preferred": matched_pref,
        "missing_preferred": missing_pref,
        "fallback_used": fallback_used,
        "fallback_note": fallback_note,
    }


def generate_recommendation_and_explanation(
    overall_score: float,
    threshold: float,
    matched_required: List[str],
    missing_required: List[str],
    matched_preferred: List[str],
    missing_preferred: List[str],
    text_sim_score: float,
    skill_score: float,
) -> Dict[str, Any]:
    """Generates a structured recommendation and human-readable explanation."""
    if overall_score >= threshold:
        recommendation = "Strong Match"
        rec_badge = "success"
        verdict = f"Recommended for candidate interview (Score: {overall_score:.1f}% >= threshold {threshold:.1f}%)."
    elif overall_score >= max(40.0, threshold - 15.0):
        recommendation = "Potential Match"
        rec_badge = "warning"
        verdict = f"Requires secondary HR review (Score: {overall_score:.1f}% within border range of threshold {threshold:.1f}%)."
    else:
        recommendation = "Low Match"
        rec_badge = "danger"
        verdict = f"Significant skill or contextual mismatch (Score: {overall_score:.1f}% below threshold {threshold:.1f}%)."

    # Formulate bulleted explanations
    reasons: List[str] = []

    # Text similarity observation
    if text_sim_score >= 65.0:
        reasons.append(f"High resume-to-job contextual alignment ({text_sim_score:.1f}% text similarity).")
    elif text_sim_score >= 40.0:
        reasons.append(f"Moderate vocabulary and domain overlap ({text_sim_score:.1f}% text similarity).")
    else:
        reasons.append(f"Low contextual vocabulary overlap with the job description ({text_sim_score:.1f}% text similarity).")

    # Required skills observation
    total_req = len(matched_required) + len(missing_required)
    if total_req > 0:
        req_pct = (len(matched_required) / total_req) * 100.0
        if req_pct == 100.0:
            reasons.append(f"Demonstrates 100% of required competencies ({len(matched_required)}/{total_req}): {', '.join(matched_required)}.")
        elif req_pct >= 60.0:
            reasons.append(
                f"Meets core requirements ({len(matched_required)}/{total_req} skills): {', '.join(matched_required)}. "
                f"Missing: {', '.join(missing_required)}."
            )
        else:
            reasons.append(
                f"Lacks multiple required skills ({len(missing_required)}/{total_req} missing): {', '.join(missing_required)}."
            )

    # Preferred skills observation
    if matched_preferred:
        reasons.append(f"Bonus skills identified: {', '.join(matched_preferred)}.")
    elif missing_preferred:
        reasons.append(f"Missing preferred/optional skills: {', '.join(missing_preferred)}.")

    return {
        "recommendation": recommendation,
        "badge": rec_badge,
        "verdict": verdict,
        "explanation_bullets": reasons,
        "fairness_notice": (
            "Screening Indicator Notice: These scores are algorithmic estimates of resume-to-job "
            "alignment and do not confirm real-world competency. Manual human HR evaluation is required."
        ),
    }


def analyze_candidate_resume(
    resume_data: Dict[str, Any],
    job_description: str,
    required_skills: List[str],
    preferred_skills: Optional[List[str]] = None,
    threshold: float = 70.0,
    weight_text: float = 0.70,
    weight_skill: float = 0.30,
) -> Dict[str, Any]:
    """Performs full end-to-end analysis on an extracted resume dictionary against a job description.

    Formula:
      Overall score = (weight_text * text_similarity_score) + (weight_skill * skill_match_score)
      Default weights: 70% text similarity + 30% skill match.
    """
    preferred_skills = preferred_skills or []
    resume_text = resume_data.get("text", "")

    # 1. Text Similarity Score via TF-IDF Cosine Similarity
    text_sim_score = compute_tfidf_similarity(resume_text, job_description)

    # 2. Skill Match Score
    skill_result = calculate_skill_match_score(
        resume_text=resume_text,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        jd_text=job_description,
    )

    skill_score = skill_result["skill_score"]
    if skill_score is None:
        # Fallback when no skills could be evaluated
        effective_skill_score = text_sim_score
    else:
        effective_skill_score = skill_score

    # 3. Overall Score Calculation (Normalized weights)
    total_weight = weight_text + weight_skill
    if total_weight <= 0:
        total_weight = 1.0

    w_text_norm = weight_text / total_weight
    w_skill_norm = weight_skill / total_weight

    overall_score = round(
        (w_text_norm * text_sim_score) + (w_skill_norm * effective_skill_score), 1
    )
    overall_score = max(0.0, min(100.0, overall_score))

    # 4. Extract all general skills found in the resume for deep profile view
    extracted_resume_skills = extract_skills_from_text(resume_text)

    # 5. Recommendation and Explanation
    rec_info = generate_recommendation_and_explanation(
        overall_score=overall_score,
        threshold=threshold,
        matched_required=skill_result["matched_required"],
        missing_required=skill_result["missing_required"],
        matched_preferred=skill_result["matched_preferred"],
        missing_preferred=skill_result["missing_preferred"],
        text_sim_score=text_sim_score,
        skill_score=effective_skill_score,
    )

    return {
        "candidate_name": resume_data.get("candidate_name", "Unknown Candidate"),
        "filename": resume_data.get("filename", "resume.pdf"),
        "email": resume_data.get("email"),
        "phone": resume_data.get("phone"),
        "linkedin": resume_data.get("linkedin"),
        "github": resume_data.get("github"),
        "page_count": resume_data.get("page_count", 0),
        "overall_score": overall_score,
        "text_similarity_score": text_sim_score,
        "skill_match_score": effective_skill_score,
        "weight_text_pct": int(w_text_norm * 100),
        "weight_skill_pct": int(w_skill_norm * 100),
        "matched_required_skills": skill_result["matched_required"],
        "missing_required_skills": skill_result["missing_required"],
        "matched_preferred_skills": skill_result["matched_preferred"],
        "missing_preferred_skills": skill_result["missing_preferred"],
        "all_detected_skills": extracted_resume_skills["all_skills"],
        "skills_by_category": extracted_resume_skills["by_category"],
        "fallback_used": skill_result["fallback_used"],
        "fallback_note": skill_result["fallback_note"],
        "recommendation": rec_info["recommendation"],
        "badge": rec_info["badge"],
        "verdict": rec_info["verdict"],
        "explanation_bullets": rec_info["explanation_bullets"],
        "fairness_notice": rec_info["fairness_notice"],
        "raw_text_snippet": resume_text[:600] + ("..." if len(resume_text) > 600 else ""),
    }
