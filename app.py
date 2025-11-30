import os
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

# NLTK lemmatizer with safe fallback (works even if wordnet isn't downloaded)
try:
    from nltk.stem import WordNetLemmatizer
    _lemm = WordNetLemmatizer()
    _ = _lemm.lemmatize("tests")
except Exception:
    _lemm = None

# ----------------------------
# Paths & folders
# ----------------------------
BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")
Path(DATA).mkdir(parents=True, exist_ok=True)
Path(MODELS).mkdir(parents=True, exist_ok=True)

# ----------------------------
# Broad skill bank for UI picker (soft + domain skills)
# ----------------------------
SKILL_BANK = [
    "Communication","Public Speaking","Writing","Negotiation","Leadership","Teamwork","Time Management","Project Management",
    "Problem Solving","Critical Thinking","Creativity","Data Analysis","Excel","PowerPoint","Documentation","Research",
    "Finance","Accounting","Law","Policy","Customer Service","Counseling","Sales","Marketing","Digital Marketing",
    "Statistics","Econometrics","Budgeting","Risk Management","Supply Chain","Procurement","Logistics","Quality Control",
    "Design Thinking","CAD","Drafting","Graphic Design","Video Editing","Photography","First Aid","Clinical Diagnosis",
    "Pharmacology","Anatomy","Rehabilitation","Event Management","Hospitality","Foreign Language","CRM","ERP"
]

# ----------------------------
# Normalization helpers (synonyms, typos, lemmatization)
# ----------------------------
_SYNONYMS = {
    # tech
    r"\b(ai|ml|deeplearning|neural\s*nets?|dl)\b": "machine learning",
    r"\bdata\s*viz\b": "visualization",
    r"\bdbs?\b": "database",
    r"\bfrontend\b": "web",
    r"\bback[-\s]?end\b": "backend",
    r"\bjs\b": "javascript",
    r"\bui/ux|ux/ui|ux\b": "ui ux",
    r"\bcyber(sec| security)?\b": "security",
    r"\binfosec\b": "security",
    r"\bdev[-\s]?ops\b": "devops",
    # non-tech & soft skills
    r"\bppt\b": "powerpoint",
    r"\bfin(ance|ancial)?\b": "finance",
    r"\baccts?\b": "accounting",
    r"\brisk\b": "risk management",
    r"\bhr\b": "human resources",
    r"\bsupply\s*chain\b": "supply chain",
    r"\bprocure(ment)?\b": "procurement",
    r"\blogistics?\b": "logistics",
    r"\blegal\b": "law",
    r"\bdoctor|physician|gp\b": "doctor",
    r"\bphysio\b": "rehabilitation",
    r"\bprof(essor)?\b": "professor",
    r"\breporter\b": "journalist",
    r"\bvideo\s*edit(or|ing)\b": "video editing",
    r"\bphoto(graphy|grapher)\b": "photography",
    r"\bchef|cook\b": "chef",
    r"\bhotel\b": "hospitality",
    r"\bevents?\b": "event management",
    r"\bpm\b": "product manager",
    r"\bpresentations?\b": "communication",
}

_COMMON_TYPO = {
    "phyton": "python",
    "javscript": "javascript",
    "mchine": "machine",
    "marketting": "marketing",
    "finacne": "finance",
    "accouting": "accounting",
}

def _simple_token_lemmatize(text: str) -> str:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens:
        return ""
    if _lemm is None:
        return " ".join(tokens)
    return " ".join(_lemm.lemmatize(t) for t in tokens)

def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    for bad, good in _COMMON_TYPO.items():
        t = t.replace(bad, good)
    for pat, repl in _SYNONYMS.items():
        t = re.sub(pat, repl, t)
    t = _simple_token_lemmatize(t)
    return t

def blob_from_inputs(free_text, selected_skills, interest_tags):
    parts = [normalize_text(free_text or "")]
    if selected_skills:
        parts.append(normalize_text(" ".join(selected_skills)))
    if interest_tags:
        parts.append(normalize_text(" ".join(interest_tags)))
    return " ".join(parts)

# ----------------------------
# Personality logic
# ----------------------------
def compute_mbti(scores):
    dims = []
    for a, b in [('E','I'), ('S','N'), ('T','F'), ('J','P')]:
        sa = np.sum(scores.get(a, []))
        sb = np.sum(scores.get(b, []))
        dims.append(a if sa >= sb else b)
    return "".join(dims)

def personality_boost(mbti, career_name):
    clusters = {
        "analytical": [
            "Data Scientist","Machine Learning Engineer","Software Developer","Data Engineer",
            "NLP Engineer","Computer Vision Engineer","Economist","Risk Analyst"
        ],
        "ops": [
            "Cloud Engineer","DevOps Engineer","Cybersecurity Analyst","SRE","Network Engineer",
            "Operations Manager","Supply Chain Manager","Logistics Coordinator","Procurement Specialist"
        ],
        "people": [
            "Product Manager","Digital Marketer","Business Analyst","Customer Success Manager","Technical Writer",
            "Data Product Manager","Sales Manager","Business Development Executive","HR Generalist","Recruiter/Talent Acquisition"
        ],
        "creative": [
            "UI/UX Designer","Web Developer","Mobile App Developer","Game Developer","AR/VR Developer",
            "Content Strategist","Video Editor","Photographer","Music Producer","Architect","Chef","Event Planner","Journalist"
        ],
        "service": [
            "Doctor (General Practitioner)","Nurse","Pharmacist","Physiotherapist","Teacher (School)","Professor/Lecturer",
            "Social Worker","Travel Consultant","Hotel Manager"
        ],
        "law_policy_fin": [
            "Lawyer (Litigation)","Corporate Lawyer","Legal Analyst","Paralegal","Policy Analyst",
            "Political Campaign Manager","Civil Services Officer","Diplomat/Foreign Service",
            "Investment Analyst","Accountant","Financial Planner"
        ]
    }
    mbti_pref = {
        "analytical": ["INTJ","INTP","ENTJ"],
        "ops": ["ISTJ","ESTJ","ENTJ"],
        "people": ["ENFP","ENFJ","ENTP","ESFJ"],
        "creative": ["INFP","ISFP","ENFP"],
        "service": ["ISFJ","ESFJ","ENFJ"],
        "law_policy_fin": ["INTJ","ISTJ","ENTJ","INFJ"]
    }
    bucket = None
    for k, vals in clusters.items():
        if career_name in vals:
            bucket = k
            break
    if bucket and mbti in mbti_pref.get(bucket, []):
        return 1.10
    return 1.00

# ----------------------------
# Dataset availability guard
# ----------------------------
def ensure_dataset_available():
    careers_path = Path(DATA) / "careers.csv"
    questions_path = Path(DATA) / "questions.json"

    if not careers_path.exists():
        st.warning("`data/careers.csv` not found. Upload your dataset (CSV) below.")
        up = st.file_uploader("Upload careers.csv", type=["csv"])
        if up is not None:
            df_try = pd.read_csv(up)
            expected_cols = {"career_id","career_name","core_skills","interests","personality_fit","description"}
            if not expected_cols.issubset(set(df_try.columns)):
                st.error(f"CSV must include columns: {sorted(list(expected_cols))}")
                st.stop()
            df_try.to_csv(careers_path, index=False, encoding="utf-8")
            st.success("Saved to data/careers.csv. Now run in terminal:  python scripts/build_index.py  and refresh.")
            st.stop()

    if not questions_path.exists():
        st.error("`data/questions.json` is missing. Add it and refresh.")
        st.stop()

ensure_dataset_available()

# ----------------------------
# Caches
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA, "careers.csv"))
    with open(os.path.join(DATA, "questions.json"), "r", encoding="utf-8") as f:
        q = json.load(f)
    return df, q

@st.cache_resource
def load_models():
    with open(os.path.join(MODELS, "tfidf_vectorizer.pkl"), "rb") as f:
        vec = pickle.load(f)
    X = sparse.load_npz(os.path.join(MODELS, "tfidf_matrix.npz"))
    return vec, X

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Astra", page_icon="🎓", layout="wide")
st.title("🎓 Astra — Your Career Counselor")
st.write("Answer a short quiz and get personalized career recommendations with explanations.")

df, q = load_data()

# Sidebar steps
st.sidebar.header("Steps")
st.sidebar.markdown("1. Enter interests/skills")
st.sidebar.markdown("2. Take quizzes")
st.sidebar.markdown("3. See results")

with st.expander("Step 1: Interests & Skills", expanded=True):
    free_text = st.text_area(
        "Describe your interests (free text)",
        height=100,
        placeholder="e.g., I enjoy numbers, markets, policy research and writing; I also like public speaking."
    )
    # CSV skills + broad skill bank
    csv_skills = sorted(set(sum([str(s).split(';') for s in df['core_skills']], [])))
    all_skills = sorted(set(csv_skills + SKILL_BANK))
    skills = st.multiselect("Select skills you have or want to learn (optional)", options=all_skills)

with st.expander("Step 2: Quick Interests Quiz"):
    interest_tags = []
    for item in q["interests"]:
        val = st.slider(item["text"], 1, 5, 3, key=f"int_{item['id']}")
        if val >= 4:
            interest_tags.append(item["tag"])

with st.expander("Step 3: Personality Quiz (MBTI-lite)"):
    mbti_scores = {"E":[], "I":[], "S":[], "N":[], "T":[], "F":[], "J":[], "P":[]}
    for item in q["personality"]:
        val = st.slider(item["text"], 1, 5, 3, key=f"per_{item['id']}")
        mbti_scores[item["dimension"]].append(val)
    mbti = compute_mbti(mbti_scores)
    st.info(f"Your MBTI-lite result (approx.): **{mbti}**")

st.divider()
run = st.button("🔍 Get Recommendations")

if run:
    try:
        vectorizer, X = load_models()
    except Exception:
        st.error("Models not found. Run in terminal:  python scripts/build_index.py  and then refresh.")
        st.stop()

    user_blob = blob_from_inputs(free_text, skills, interest_tags)
    if not user_blob.strip():
        st.warning("Please add some interests or select skills to proceed.")
    else:
        uvec = vectorizer.transform([user_blob])
        sims = cosine_similarity(uvec, X).ravel()

        # personality-aware boost
        boosted = []
        for i, row in df.iterrows():
            boost = personality_boost(mbti, row["career_name"])
            boosted.append(sims[i] * boost)
        boosted = np.array(boosted)

        df_scores = df.copy()
        df_scores["score"] = boosted
        topk = df_scores.sort_values("score", ascending=False).head(5)

        c1, c2 = st.columns([3, 2])
        with c1:
            fig = px.bar(topk, x="career_name", y="score", title="Top Career Matches")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            dims = [("E","I"),("S","N"),("T","F"),("J","P")]
            axes = []
            values = []
            for a, b in dims:
                sa = np.sum(mbti_scores[a])
                sb = np.sum(mbti_scores[b])
                axes.append(f"{a}/{b}")
                total = sa + sb if (sa + sb) > 0 else 1
                values.append(sa / total)
            radar_df = pd.DataFrame({"Dimension": axes, "Preference": values})
            fig2 = px.line_polar(radar_df, r="Preference", theta="Dimension", line_close=True, range_r=[0, 1])
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Why these recommendations? (Top keywords)")
        try:
            vocab = vectorizer.get_feature_names_out()
            user_vec = uvec.toarray()[0]
            top_indices = user_vec.argsort()[-20:][::-1]
            keywords = [vocab[i] for i in top_indices if user_vec[i] > 0]
            st.write(", ".join(keywords) if keywords else "Add more details about your interests/skills to see keywords here.")
        except Exception:
            st.write("Keyword explanation unavailable.")

        st.subheader("Career Details")
        for _, row in topk.iterrows():
            with st.expander(f"ℹ️ {row['career_name']} — details"):
                st.markdown(f"**Core skills:** {row['core_skills']}")
                st.markdown(f"**Common interests:** {row['interests']}")
                st.markdown(f"**Typical personality fit:** {row['personality_fit']}")
                st.markdown(f"**About:** {row['description']}")
