"""HireSense AI — Intelligent Automated Resume Screening & Skill Gap Analyzer.

A modern, ethical Streamlit application for HR professionals to screen multiple resumes,
calculate transparent TF-IDF and skill-match scores, identify critical skill gaps,
and make informed, unbiased preliminary hiring decisions.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Ensure project root directory is in sys.path for robust cloud imports
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Application modules (supports both nested src/ structure and root-level deployment)
try:
    from src.extractor import extract_text_from_pdf
    from src.preprocessor import clean_text_for_nlp
    from src.skills import (
        DEFAULT_SKILLS_TAXONOMY,
        extract_skills_from_text,
        parse_skill_string,
    )
    from src.scorer import (
        analyze_candidate_resume,
        compute_tfidf_similarity,
    )
    from src.demo_data import (
        DEFAULT_SAMPLE_JD,
        SAMPLE_CANDIDATES,
        create_pdf_bytes,
        create_scanned_empty_pdf_bytes,
    )
except (ImportError, ModuleNotFoundError):
    from extractor import extract_text_from_pdf
    from preprocessor import clean_text_for_nlp
    from skills import (
        DEFAULT_SKILLS_TAXONOMY,
        extract_skills_from_text,
        parse_skill_string,
    )
    from scorer import (
        analyze_candidate_resume,
        compute_tfidf_similarity,
    )
    from demo_data import (
        DEFAULT_SAMPLE_JD,
        SAMPLE_CANDIDATES,
        create_pdf_bytes,
        create_scanned_empty_pdf_bytes,
    )


# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="HireSense AI — Automated Resume Screening",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Custom Professional CSS (Navy, Slate, Clean Corporate Palette)
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Global Fonts and Tone */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Main Background & Padding */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }
    
    /* Professional Header Styles */
    .app-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
        color: #FFFFFF;
        padding: 1.75rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08);
    }
    .app-header h1 {
        color: #FFFFFF !important;
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.02em;
    }
    .app-header p {
        color: #93C5FD;
        font-size: 0.95rem;
        margin: 0;
    }
    .app-header .ethical-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        color: #F8FAFC;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        margin-top: 0.75rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    
    /* Metric KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.06);
    }
    .kpi-title {
        color: #64748B;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        color: #0F172A;
        font-size: 1.75rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .kpi-subtitle {
        color: #94A3B8;
        font-size: 0.75rem;
        margin-top: 0.35rem;
    }
    
    /* Skill Pills / Badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }
    .badge-success {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #BBF7D0;
    }
    .badge-danger {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }
    .badge-warning {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    .badge-info {
        background-color: #DBEAFE;
        color: #1E40AF;
        border: 1px solid #BFDBFE;
    }
    .badge-neutral {
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #E2E8F0;
    }

    /* Score Badges */
    .score-badge {
        font-weight: 700;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        display: inline-block;
    }
    .score-high {
        background: #DCFCE7;
        color: #15803D;
    }
    .score-mid {
        background: #FEF3C7;
        color: #B45309;
    }
    .score-low {
        background: #FEE2E2;
        color: #B91C1C;
    }

    /* Card styling for Candidate Details */
    .candidate-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    }
    .candidate-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 1rem;
        margin-bottom: 1rem;
    }
    .candidate-name {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
    }
    .candidate-contact {
        color: #64748B;
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }
    
    /* Demo Banner */
    .demo-banner {
        background: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 0.85rem 1.2rem;
        border-radius: 0 8px 8px 0;
        color: #1E3A8A;
        font-size: 0.88rem;
        margin-bottom: 1.25rem;
    }
    
    /* Ethical Disclaimer Box */
    .ethical-box {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        color: #475569;
        font-size: 0.82rem;
        line-height: 1.45;
        margin-top: 1rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "job_title" not in st.session_state:
    st.session_state["job_title"] = DEFAULT_SAMPLE_JD["job_title"]

if "job_description" not in st.session_state:
    st.session_state["job_description"] = DEFAULT_SAMPLE_JD["description_text"]

if "required_skills" not in st.session_state:
    st.session_state["required_skills"] = DEFAULT_SAMPLE_JD["required_skills"]

if "preferred_skills" not in st.session_state:
    st.session_state["preferred_skills"] = DEFAULT_SAMPLE_JD["preferred_skills"]

if "threshold" not in st.session_state:
    st.session_state["threshold"] = 70.0

if "weight_text" not in st.session_state:
    st.session_state["weight_text"] = 70

if "weight_skill" not in st.session_state:
    st.session_state["weight_skill"] = 30

if "demo_mode" not in st.session_state:
    st.session_state["demo_mode"] = False

if "uploaded_results" not in st.session_state:
    st.session_state["uploaded_results"] = []

if "parsing_errors" not in st.session_state:
    st.session_state["parsing_errors"] = []

if "selected_candidate_idx" not in st.session_state:
    st.session_state["selected_candidate_idx"] = 0


NAV_PAGES = [
    "📊 Screening Dashboard",
    "🔍 Candidate Gap Analysis",
    "📋 Job Description & Config",
    "🧠 Skills Taxonomy",
    "⚖️ Privacy, Fairness & Ethics",
]

if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = NAV_PAGES[0]

# Handle programmatic page transitions (e.g., clicking 'Deep Dive' or 'Edit Job')
if "target_nav" in st.session_state and st.session_state["target_nav"] in NAV_PAGES:
    st.session_state["nav_selection"] = st.session_state.pop("target_nav")


# ---------------------------------------------------------
# Sidebar: Navigation, Settings, and Mode Switcher
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎯 HireSense AI")
    st.caption("Intelligent Automated Screening & Skill Gap Analyzer")
    st.markdown("---")

    # Mode Selector
    st.subheader("🛠️ Operation Mode")
    demo_toggle = st.toggle(
        "Activate Demo Mode",
        value=st.session_state["demo_mode"],
        help="Simulate screening with realistic benchmark resumes. Real uploads and demo data are strictly isolated.",
    )
    if demo_toggle != st.session_state["demo_mode"]:
        st.session_state["demo_mode"] = demo_toggle
        st.rerun()

    if st.session_state["demo_mode"]:
        st.info("ℹ️ **Demo Mode Active**\nLoaded 4 benchmark candidates. Scores are illustrative.")

    st.markdown("---")

    # Navigation Menu
    current_nav_idx = (
        NAV_PAGES.index(st.session_state["nav_selection"])
        if st.session_state["nav_selection"] in NAV_PAGES
        else 0
    )
    nav_selection = st.radio(
        "Navigate",
        NAV_PAGES,
        index=current_nav_idx,
        key="nav_selection_radio",
    )
    st.session_state["nav_selection"] = nav_selection

    st.markdown("---")
    st.subheader("⚙️ Scoring Configuration")

    # Threshold Slider
    threshold_val = st.slider(
        "Minimum Match Threshold (%)",
        min_value=40.0,
        max_value=95.0,
        value=float(st.session_state["threshold"]),
        step=5.0,
        help="Candidates scoring at or above this threshold will be flagged as 'Strong Match'.",
    )
    st.session_state["threshold"] = threshold_val

    # Formula Weight Sliders
    st.caption("Scoring Formula: Overall = Text Sim + Skill Match")
    weight_text = st.slider(
        "Text Similarity Weight (%)",
        min_value=10,
        max_value=90,
        value=st.session_state["weight_text"],
        step=5,
        help="Weight assigned to TF-IDF contextual similarity (Default: 70%).",
    )
    weight_skill = 100 - weight_text
    st.session_state["weight_text"] = weight_text
    st.session_state["weight_skill"] = weight_skill
    st.caption(f"Weights: **{weight_text}% Text Similarity** + **{weight_skill}% Skill Match**")

    st.markdown("---")
    # Quick Reset
    if st.button("🔄 Reset Analysis / Clear Data", use_container_width=True):
        st.session_state["uploaded_results"] = []
        st.session_state["parsing_errors"] = []
        st.rerun()


# ---------------------------------------------------------
# Helper: Run Analysis on Resumes
# ---------------------------------------------------------
def process_uploaded_files(uploaded_files, jd_text, req_skills, pref_skills, threshold, w_text, w_skill):
    results = []
    errors = []

    req_list = parse_skill_string(req_skills)
    pref_list = parse_skill_string(pref_skills)

    progress_bar = st.progress(0, text="Analyzing resumes...")
    total_files = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files):
        progress_bar.progress((idx + 1) / total_files, text=f"Processing {uploaded_file.name}...")
        
        extracted = extract_text_from_pdf(uploaded_file, filename=uploaded_file.name)
        
        if not extracted["is_valid"]:
            errors.append({
                "filename": uploaded_file.name,
                "error": extracted.get("error", "Failed to extract readable text."),
            })
            continue

        analysis = analyze_candidate_resume(
            resume_data=extracted,
            job_description=jd_text,
            required_skills=req_list,
            preferred_skills=pref_list,
            threshold=threshold,
            weight_text=w_text / 100.0,
            weight_skill=w_skill / 100.0,
        )
        results.append(analysis)

    progress_bar.empty()
    return results, errors


def get_active_results():
    """Returns active candidates based on Demo Mode vs Real Uploads."""
    req_list = parse_skill_string(st.session_state["required_skills"])
    pref_list = parse_skill_string(st.session_state["preferred_skills"])

    if st.session_state["demo_mode"]:
        # Run live analysis on the 4 sample candidates against current JD
        demo_results = []
        for cand in SAMPLE_CANDIDATES:
            analysis = analyze_candidate_resume(
                resume_data=cand,
                job_description=st.session_state["job_description"],
                required_skills=req_list,
                preferred_skills=pref_list,
                threshold=st.session_state["threshold"],
                weight_text=st.session_state["weight_text"] / 100.0,
                weight_skill=st.session_state["weight_skill"] / 100.0,
            )
            demo_results.append(analysis)
        return demo_results, []
    else:
        return st.session_state["uploaded_results"], st.session_state["parsing_errors"]


# ---------------------------------------------------------
# App Header Component
# ---------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>🎯 HireSense AI</h1>
        <p>Intelligent Automated Resume Screening & Skill Gap Analyzer</p>
        <span class="ethical-tag">🛡️ Algorithmic Fairness Guardrails Active • Non-Biased Screening</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# TAB 1: SCREENING DASHBOARD
# ---------------------------------------------------------
if nav_selection == "📊 Screening Dashboard":
    if st.session_state["demo_mode"]:
        st.markdown(
            """
            <div class="demo-banner">
                <strong>💡 DEMO MODE ACTIVE:</strong> Showing 4 illustrative candidate profiles evaluated 
                against the benchmark <em>"Senior AI / Machine Learning Engineer"</em> job description.
                Uploads are simulated. Scores are illustrative.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Job Banner Summary
    col_jd1, col_jd2 = st.columns([3, 1])
    with col_jd1:
        st.markdown(f"**Target Role:** `{st.session_state['job_title']}`")
        req_count = len(parse_skill_string(st.session_state["required_skills"]))
        pref_count = len(parse_skill_string(st.session_state["preferred_skills"]))
        st.caption(
            f"Configured Skills: **{req_count} Required** • **{pref_count} Preferred** | "
            f"Formula: **{st.session_state['weight_text']}% TF-IDF Context + {st.session_state['weight_skill']}% Skill Gap** | "
            f"Pass Threshold: **{st.session_state['threshold']:.0f}%**"
        )
    with col_jd2:
        if st.button("✏️ Edit Job Description", use_container_width=True):
            # Switch to JD tab
            st.session_state["target_nav"] = "📋 Job Description & Config"
            st.rerun()

    # Upload Section (Hidden when in Demo Mode to prevent confusion)
    if not st.session_state["demo_mode"]:
        st.markdown("### 📤 Upload Resumes")

        # In-memory benchmark resume download for cloud evaluators
        with st.expander("📥 Need test resumes? Download benchmark candidate PDFs", expanded=False):
            st.caption(
                "Click below to download realistic benchmark candidate PDFs to your device, "
                "then drop them into the file uploader to test real-time PDF parsing and scoring."
            )
            dl_cols = st.columns(len(SAMPLE_CANDIDATES) + 1)
            for i, cand in enumerate(SAMPLE_CANDIDATES):
                with dl_cols[i]:
                    pdf_bytes = create_pdf_bytes(cand["text"])
                    st.download_button(
                        label=f"📄 {cand['candidate_name'].split()[-1]}",
                        data=pdf_bytes,
                        file_name=cand["filename"],
                        mime="application/pdf",
                        use_container_width=True,
                        key=f"sample_dl_btn_{i}",
                        help=f"Download {cand['filename']}",
                    )
            with dl_cols[-1]:
                st.download_button(
                    label="📄 Scanned (Empty)",
                    data=create_scanned_empty_pdf_bytes(),
                    file_name="Scanned_Image_Empty_Resume.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="sample_dl_scanned",
                    help="Test error handling for non-OCR / scanned image resumes",
                )

        uploaded_files = st.file_uploader(
            "Select one or multiple candidate PDF resumes",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload text-based PDF resumes. Scanned or image-only PDFs will be reported.",
        )

        if uploaded_files:
            analyze_col, clear_col = st.columns([2, 1])
            with analyze_col:
                if st.button("🚀 Analyze Uploaded Resumes", type="primary", use_container_width=True):
                    with st.spinner("Extracting PDF text and running NLP similarity pipeline..."):
                        results, errors = process_uploaded_files(
                            uploaded_files=uploaded_files,
                            jd_text=st.session_state["job_description"],
                            req_skills=st.session_state["required_skills"],
                            pref_skills=st.session_state["preferred_skills"],
                            threshold=st.session_state["threshold"],
                            w_text=st.session_state["weight_text"],
                            w_skill=st.session_state["weight_skill"],
                        )
                        st.session_state["uploaded_results"] = results
                        st.session_state["parsing_errors"] = errors
                        st.success(f"Analysis complete! Processed {len(results)} valid resumes.")
                        st.rerun()

    # Retrieve Active Data
    active_candidates, active_errors = get_active_results()

    # Display Parsing / Scanned Errors if any
    if active_errors:
        st.error(f"⚠️ {len(active_errors)} file(s) could not be parsed:")
        for err in active_errors:
            st.markdown(f"- **{err['filename']}**: {err['error']}")

    if not active_candidates and not st.session_state["demo_mode"]:
        st.info(
            "👋 **No resumes analyzed yet.**\n\n"
            "1. Upload PDF resumes using the file uploader above, or\n"
            "2. Turn on **'Activate Demo Mode'** in the sidebar to test immediately with 4 realistic candidate benchmarks."
        )
    else:
        # Sort candidates by overall score descending
        sorted_candidates = sorted(active_candidates, key=lambda c: c["overall_score"], reverse=True)
        total_resumes = len(sorted_candidates)
        avg_score = sum(c["overall_score"] for c in sorted_candidates) / total_resumes if total_resumes else 0.0
        top_score = sorted_candidates[0]["overall_score"] if sorted_candidates else 0.0
        above_threshold = sum(1 for c in sorted_candidates if c["overall_score"] >= st.session_state["threshold"])
        potential_count = sum(1 for c in sorted_candidates if c["recommendation"] == "Potential Match")

        # KPI Metrics Cards
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-card">
                    <div class="kpi-title">Total Resumes</div>
                    <div class="kpi-value">{total_resumes}</div>
                    <div class="kpi-subtitle">{"Demo Benchmarks" if st.session_state['demo_mode'] else "Processed PDFs"}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Average Match</div>
                    <div class="kpi-value">{avg_score:.1f}%</div>
                    <div class="kpi-subtitle">Weighted Mean Score</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Qualified Candidates</div>
                    <div class="kpi-value" style="color: #16A34A;">{above_threshold}</div>
                    <div class="kpi-subtitle">&ge; {st.session_state['threshold']:.0f}% Pass Threshold</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Highest Score</div>
                    <div class="kpi-value" style="color: #2563EB;">{top_score:.1f}%</div>
                    <div class="kpi-subtitle">Top Candidate Match</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Filters and Controls
        st.markdown("### 📋 Candidate Rankings & Screening Results")
        f_col1, f_col2, f_col3 = st.columns([2, 2, 2])

        with f_col1:
            rec_filter = st.selectbox(
                "Filter by Recommendation",
                ["All Candidates", "Strong Match", "Potential Match", "Low Match"],
            )

        with f_col2:
            search_query = st.text_input("Search Candidate or Skill", placeholder="e.g. Python, Aris, SQL...")

        with f_col3:
            # CSV Download Button
            csv_rows = []
            for c in sorted_candidates:
                csv_rows.append({
                    "Candidate Name": c["candidate_name"],
                    "Filename": c["filename"],
                    "Overall Match Score (%)": c["overall_score"],
                    "Text Similarity Score (%)": c["text_similarity_score"],
                    "Skill Match Score (%)": c["skill_match_score"],
                    "Matching Skills": "; ".join(c["matched_required_skills"] + c["matched_preferred_skills"]),
                    "Missing Required Skills": "; ".join(c["missing_required_skills"]),
                    "Missing Preferred Skills": "; ".join(c["missing_preferred_skills"]),
                    "Recommendation": c["recommendation"],
                    "Email": c.get("email", ""),
                    "Phone": c.get("phone", ""),
                })
            df_export = pd.DataFrame(csv_rows)
            csv_data = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download CSV Report",
                data=csv_data,
                file_name=f"hiresense_screening_{st.session_state['job_title'].replace(' ', '_').lower()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Filter candidate list
        filtered = sorted_candidates
        if rec_filter != "All Candidates":
            filtered = [c for c in filtered if c["recommendation"] == rec_filter]

        if search_query.strip():
            sq = search_query.strip().lower()
            filtered = [
                c for c in filtered
                if sq in c["candidate_name"].lower()
                or sq in c["filename"].lower()
                or any(sq in s.lower() for s in c["matched_required_skills"] + c["matched_preferred_skills"])
            ]

        # Candidate Cards / Table
        if not filtered:
            st.warning("No candidates matched your search/filter criteria.")
        else:
            for idx, c in enumerate(filtered):
                score_class = (
                    "score-high" if c["overall_score"] >= st.session_state["threshold"]
                    else "score-mid" if c["overall_score"] >= st.session_state["threshold"] - 15
                    else "score-low"
                )

                with st.container():
                    st.markdown(
                        f"""
                        <div class="candidate-card">
                            <div class="candidate-header">
                                <div>
                                    <h3 class="candidate-name">{idx + 1}. {c['candidate_name']}</h3>
                                    <div class="candidate-contact">
                                        📁 <code>{c['filename']}</code> • 
                                        ✉️ {c['email'] or 'Email not listed'} • 
                                        📞 {c['phone'] or 'Phone not listed'}
                                    </div>
                                </div>
                                <div style="text-align: right;">
                                    <span class="score-badge {score_class}">{c['overall_score']:.1f}% Overall</span>
                                    <div style="font-size: 0.78rem; font-weight: 600; color: #475569; margin-top: 0.25rem;">
                                        {c['recommendation']}
                                    </div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Score Breakdown Columns
                    sc1, sc2, sc3 = st.columns([1.5, 3.5, 1])
                    with sc1:
                        st.markdown(
                            f"""
                            **Score Breakdown:**  
                            - Context Similarity: **{c['text_similarity_score']:.1f}%** *(weight: {st.session_state['weight_text']}%)*  
                            - Skill Alignment: **{c['skill_match_score']:.1f}%** *(weight: {st.session_state['weight_skill']}%)*
                            """
                        )
                    with sc2:
                        # Skills Display
                        st.markdown("**Skill Gap Breakdown:**")
                        matched_html = "".join([f'<span class="badge badge-success">✓ {s}</span>' for s in c["matched_required_skills"]])
                        if not matched_html:
                            matched_html = '<span style="color:#94A3B8; font-size:0.8rem;">None</span>'
                        st.markdown(f"**Matched Required:** {matched_html}", unsafe_allow_html=True)

                        missing_html = "".join([f'<span class="badge badge-danger">✗ {s}</span>' for s in c["missing_required_skills"]])
                        if not missing_html:
                            missing_html = '<span class="badge badge-neutral">All Required Skills Present</span>'
                        st.markdown(f"**Missing Required:** {missing_html}", unsafe_allow_html=True)

                        if c["missing_preferred_skills"]:
                            pref_html = "".join([f'<span class="badge badge-warning">− {s}</span>' for s in c["missing_preferred_skills"]])
                            st.markdown(f"**Missing Preferred:** {pref_html}", unsafe_allow_html=True)

                    with sc3:
                        if st.button("🔍 Deep Dive", key=f"btn_dive_{idx}_{c['filename']}", use_container_width=True):
                            st.session_state["selected_candidate_name"] = c["candidate_name"]
                            st.session_state["target_nav"] = "🔍 Candidate Gap Analysis"
                            st.rerun()

                    st.markdown("---")

        # Screening Indicator Notice
        st.markdown(
            """
            <div class="ethical-box">
                ⚖️ <strong>Screening Indicator Notice:</strong> Match scores are calculated using automated 
                TF-IDF cosine similarity and pattern-based skill matching. Keyword presence does not guarantee 
                real-world competence. Scores are decision-support aids; all hiring decisions must be conducted 
                by qualified human HR professionals following interview evaluations.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# TAB 2: CANDIDATE DEEP DIVE & SKILL GAP ANALYSIS
# ---------------------------------------------------------
elif nav_selection == "🔍 Candidate Gap Analysis":
    active_candidates, _ = get_active_results()

    if not active_candidates:
        st.info("Please upload resumes or activate Demo Mode in the sidebar first to view detailed gap analysis.")
    else:
        top_c1, _ = st.columns([1, 4])
        with top_c1:
            if st.button("⬅️ Back to Rankings", use_container_width=True):
                st.session_state["target_nav"] = "📊 Screening Dashboard"
                st.rerun()

        candidate_names = [c["candidate_name"] for c in active_candidates]
        default_idx = 0
        if "selected_candidate_name" in st.session_state:
            if st.session_state["selected_candidate_name"] in candidate_names:
                default_idx = candidate_names.index(st.session_state["selected_candidate_name"])

        selected_name = st.selectbox(
            "Select Candidate for In-Depth Evaluation",
            candidate_names,
            index=default_idx,
        )

        candidate = next(c for c in active_candidates if c["candidate_name"] == selected_name)

        # Candidate Profile Card
        st.markdown(
            f"""
            <div class="candidate-card">
                <div class="candidate-header">
                    <div>
                        <h2 class="candidate-name">{candidate['candidate_name']}</h2>
                        <div class="candidate-contact">
                            📁 Source: <code>{candidate['filename']}</code> • 📄 Pages: {candidate['page_count']}
                        </div>
                        <div class="candidate-contact" style="margin-top: 0.35rem;">
                            ✉️ <strong>Email:</strong> {candidate['email'] or 'Not detected'} | 
                            📞 <strong>Phone:</strong> {candidate['phone'] or 'Not detected'} | 
                            🔗 <strong>LinkedIn:</strong> {candidate['linkedin'] or 'Not detected'}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 2.2rem; font-weight: 800; color: #1E3A8A;">
                            {candidate['overall_score']:.1f}%
                        </div>
                        <span class="badge badge-info">{candidate['recommendation']}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metrics Breakdown
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Overall Score", f"{candidate['overall_score']:.1f}%", delta=f"{candidate['overall_score'] - st.session_state['threshold']:.1f}% vs threshold")
        with m2:
            st.metric("TF-IDF Text Similarity", f"{candidate['text_similarity_score']:.1f}%", help=f"Weighted at {st.session_state['weight_text']}%")
        with m3:
            st.metric("Skill Alignment Score", f"{candidate['skill_match_score']:.1f}%", help=f"Weighted at {st.session_state['weight_skill']}%")
        with m4:
            total_req = len(candidate["matched_required_skills"]) + len(candidate["missing_required_skills"])
            st.metric("Required Skills Matched", f"{len(candidate['matched_required_skills'])} / {total_req}")

        st.markdown("### 📊 Competency & Skill Gap Analysis")

        gap_col1, gap_col2 = st.columns(2)

        with gap_col1:
            st.markdown("#### ✅ Demonstrated Competencies")
            if candidate["matched_required_skills"]:
                st.markdown("**Core Required Skills Verified:**")
                for s in candidate["matched_required_skills"]:
                    st.markdown(f"- <span class='badge badge-success'>✓ {s}</span>", unsafe_allow_html=True)
            else:
                st.markdown("*No required skills detected in resume.*")

            if candidate["matched_preferred_skills"]:
                st.markdown("**Bonus / Preferred Skills Verified:**")
                for s in candidate["matched_preferred_skills"]:
                    st.markdown(f"- <span class='badge badge-info'>★ {s}</span>", unsafe_allow_html=True)

        with gap_col2:
            st.markdown("#### ⚠️ Identified Skill Gaps")
            if candidate["missing_required_skills"]:
                st.markdown("**Missing Required Skills (Action Item for HR Interview):**")
                for s in candidate["missing_required_skills"]:
                    st.markdown(f"- <span class='badge badge-danger'>✗ Missing: {s}</span>", unsafe_allow_html=True)
            else:
                st.success("🎉 Zero required skill gaps detected! Candidate meets all required competencies.")

            if candidate["missing_preferred_skills"]:
                st.markdown("**Missing Preferred Skills:**")
                for s in candidate["missing_preferred_skills"]:
                    st.markdown(f"- <span class='badge badge-warning'>− Missing: {s}</span>", unsafe_allow_html=True)

        st.markdown("---")

        # Full Skills Profile by Category
        st.markdown("#### 🗂️ All Skills Detected on Candidate's Resume (Taxonomy Breakdown)")
        if candidate.get("skills_by_category"):
            cat_cols = st.columns(len(candidate["skills_by_category"]))
            for idx, (cat_name, cat_skills) in enumerate(candidate["skills_by_category"].items()):
                with cat_cols[idx % len(cat_cols)]:
                    st.markdown(f"**{cat_name}:**")
                    for sk in cat_skills:
                        st.markdown(f"<span class='badge badge-neutral'>{sk}</span>", unsafe_allow_html=True)
        else:
            st.info("No standard categorized skills detected in resume text.")

        st.markdown("---")

        # Recommendation & Rationale
        st.markdown("#### 📝 Algorithmic Evaluation & Screening Rationale")
        st.info(f"**Recommendation Verdict:** {candidate['verdict']}")
        for bullet in candidate["explanation_bullets"]:
            st.markdown(f"- {bullet}")

        if candidate.get("fallback_used"):
            st.warning(f"ℹ️ {candidate.get('fallback_note')}")

        # Collapsible Raw Resume Text Inspector
        with st.expander("📄 View Extracted Raw Resume Text"):
            st.text_area("Cleaned / Extracted Content", candidate.get("raw_text_snippet", ""), height=250)


# ---------------------------------------------------------
# TAB 3: JOB DESCRIPTION & CONFIGURATION
# ---------------------------------------------------------
elif nav_selection == "📋 Job Description & Config":
    st.markdown("### 📋 Configure Job Benchmark Requirements")
    st.caption("Customize the job title, requirements, and skill definitions used to evaluate resumes.")

    with st.expander("⚡ Load Pre-built Job Profiles", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🤖 Senior AI / ML Engineer", use_container_width=True):
                st.session_state["job_title"] = DEFAULT_SAMPLE_JD["job_title"]
                st.session_state["job_description"] = DEFAULT_SAMPLE_JD["description_text"]
                st.session_state["required_skills"] = DEFAULT_SAMPLE_JD["required_skills"]
                st.session_state["preferred_skills"] = DEFAULT_SAMPLE_JD["preferred_skills"]
                st.rerun()
        with c2:
            if st.button("🌐 Full Stack Python Developer", use_container_width=True):
                st.session_state["job_title"] = "Full Stack Python Web Developer"
                st.session_state["job_description"] = """We are seeking a Full Stack Python Developer to build robust web applications, REST APIs, and responsive frontends.
Key skills include Python, Django, FastAPI, React, PostgreSQL, Docker, Git, and automated testing."""
                st.session_state["required_skills"] = "Python, Django, FastAPI, React, PostgreSQL, Docker, Git"
                st.session_state["preferred_skills"] = "TypeScript, Redis, AWS, CI/CD"
                st.rerun()
        with c3:
            if st.button("☁️ DevOps & Cloud Specialist", use_container_width=True):
                st.session_state["job_title"] = "DevOps & Cloud Engineer"
                st.session_state["job_description"] = """Looking for a Cloud DevOps engineer with proven expertise in AWS, Kubernetes, Docker, Terraform, CI/CD automation, and Linux administration."""
                st.session_state["required_skills"] = "AWS, Kubernetes, Docker, Terraform, CI/CD, Linux, Git"
                st.session_state["preferred_skills"] = "Python, Bash/Shell, Prometheus, Helm"
                st.rerun()

    # Form
    job_title_input = st.text_input("Job Title", value=st.session_state["job_title"])
    jd_text_input = st.text_area(
        "Job Description Text",
        value=st.session_state["job_description"],
        height=260,
        help="Paste the full job specification here. The text will be processed for TF-IDF cosine similarity comparison.",
    )

    col_req, col_pref = st.columns(2)
    with col_req:
        req_skills_input = st.text_area(
            "Required Skills (Comma or Newline Separated)",
            value=st.session_state["required_skills"],
            height=100,
            help="Mandatory competencies. Missing these reduces the candidate's skill score significantly.",
        )
    with col_pref:
        pref_skills_input = st.text_area(
            "Preferred / Optional Skills (Comma Separated)",
            value=st.session_state["preferred_skills"],
            height=100,
            help="Bonus skills that provide competitive advantage in ranking.",
        )

    # Auto-detect skills helper
    if st.button("🔍 Auto-Detect Skills from Job Description Text"):
        detected = extract_skills_from_text(jd_text_input)
        if detected["all_skills"]:
            st.session_state["required_skills"] = ", ".join(detected["all_skills"])
            st.success(f"Detected {len(detected['all_skills'])} skills from JD text: {', '.join(detected['all_skills'])}")
            st.rerun()
        else:
            st.warning("No standard skills detected from the taxonomy in this JD text.")

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if st.button("💾 Save Job Configuration", type="primary", use_container_width=True):
            st.session_state["job_title"] = job_title_input
            st.session_state["job_description"] = jd_text_input
            st.session_state["required_skills"] = req_skills_input
            st.session_state["preferred_skills"] = pref_skills_input
            st.success("✅ Job benchmark successfully updated!")
    with btn_col2:
        if st.button("📊 Save & Return to Dashboard", use_container_width=True):
            st.session_state["job_title"] = job_title_input
            st.session_state["job_description"] = jd_text_input
            st.session_state["required_skills"] = req_skills_input
            st.session_state["preferred_skills"] = pref_skills_input
            st.session_state["target_nav"] = "📊 Screening Dashboard"
            st.rerun()


# ---------------------------------------------------------
# TAB 4: SKILLS TAXONOMY & CONFIGURATION
# ---------------------------------------------------------
elif nav_selection == "🧠 Skills Taxonomy":
    st.markdown("### 🧠 Built-In Skills Taxonomy & Dictionary")
    st.caption(
        "HireSense AI uses an extensible, case-insensitive taxonomy with regex word boundaries "
        "and technical token preservation (e.g. C++, .NET, CI/CD) to prevent false positives."
    )

    for cat_name, skills_dict in DEFAULT_SKILLS_TAXONOMY.items():
        with st.expander(f"📁 {cat_name} ({len(skills_dict)} skills)", expanded=False):
            skills_list = list(skills_dict.keys())
            cols = st.columns(4)
            for i, skill in enumerate(skills_list):
                with cols[i % 4]:
                    st.markdown(f"- `{skill}`")

    st.markdown("---")
    st.markdown("#### 📐 Mathematical Scoring Formulas")
    st.markdown(
        r"""
        HireSense AI employs a reproducible, transparent multi-stage scoring pipeline:

        1. **TF-IDF Context Similarity Score ($S_{\text{tfidf}}$)**:
           $$\text{Sim}(R, J) = \frac{\mathbf{v}_R \cdot \mathbf{v}_J}{\|\mathbf{v}_R\| \|\mathbf{v}_J\|} \times 100$$
           where $\mathbf{v}_R$ and $\mathbf{v}_J$ are sublinear unigram/bigram TF-IDF vectors representing the cleaned resume and job description.

        2. **Skill-Match Score ($S_{\text{skill}}$)**:
           $$S_{\text{skill}} = \left( 0.80 \times \frac{|\text{Matched}_{\text{req}}|}{|\text{Required}|} + 0.20 \times \frac{|\text{Matched}_{\text{pref}}|}{|\text{Preferred}|} \right) \times 100$$
           *(If only required skills are entered, the full $100\%$ weight applies to the required ratio).*

        3. **Overall Composite Score ($S_{\text{overall}}$)**:
           $$S_{\text{overall}} = (w_{\text{text}} \times S_{\text{tfidf}}) + (w_{\text{skill}} \times S_{\text{skill}})$$
           Default weights: $w_{\text{text}} = 0.70$ ($70\%$), $w_{\text{skill}} = 0.30$ ($30\%$).
        """
    )


# ---------------------------------------------------------
# TAB 5: PRIVACY, FAIRNESS & ETHICS
# ---------------------------------------------------------
elif nav_selection == "⚖️ Privacy, Fairness & Ethics":
    st.markdown("### ⚖️ AI Ethics, Privacy & Fairness Charter")
    st.caption("HireSense AI is designed with responsible, transparent, and non-discriminatory HR technology principles.")

    st.markdown(
        """
        #### 1. Non-Discrimination Policy
        - **Protected Attributes Excluded:** HireSense AI explicitly **does not** extract, parse, or evaluate candidate age, gender, race, ethnicity, religion, caste, nationality, physical ability, sexual orientation, or photographs.
        - **Name Sanitization:** Candidate names are extracted strictly for human HR indexing and identification. Algorithms never assign points, weights, or ranking variations based on names.

        #### 2. Local & Ephemeral Processing
        - **Zero Cloud API Reliance:** Core text processing runs locally using Scikit-Learn TF-IDF, PyPDF, and deterministic regex.
        - **No Third-Party Transmission:** Candidate resumes are processed strictly in-memory during the active browser session. They are **never** transmitted to paid external AI APIs or remote storage endpoints.
        - **No Permanent Retention:** Uploaded files and in-memory caches are purged upon session termination or clicking the "Reset Analysis" button.

        #### 3. Human-in-the-Loop Imperative
        - **No Automated Rejection:** HireSense AI is strictly a preliminary candidate screening and prioritization aid. It does not issue automatic rejection emails or hiring decisions.
        - **Contextual Awareness:** Keyword matching does not equal real-world competence. HR professionals must manually review borderline candidates and conduct structured competency interviews.

        #### 4. Auditability & Explainability
        - Every score is backed by an itemized, human-readable breakdown detailing:
          - Exact TF-IDF context match percentage
          - Specific demonstrated skills vs missing skills
          - Transparent fallback notes whenever inputs are missing
        """
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "HireSense AI • Built with Python, Streamlit, Scikit-Learn, and PyPDF • "
    "Designed for Student AI / HR Analytics Demonstration."
)
