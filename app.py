import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from features import CLASSES, ORDINAL, encode

st.set_page_config(page_title="Student Job Readiness Predictor", page_icon="🎓", layout="centered")

BASE = Path(__file__).parent


@st.cache_resource
def load():
    model = joblib.load(BASE / "model.joblib")
    metrics = json.loads((BASE / "metrics.json").read_text())
    return model, metrics


model, metrics = load()

st.title("🎓 Student Job Readiness Predictor")
st.caption("Answer the questions below to estimate whether a student is Job Ready, "
           "Needs Improvement, or Not Job Ready.")

with st.form("student"):
    st.subheader("Academics")
    c1, c2 = st.columns(2)
    cgpa = c1.number_input("CGPA", 0.0, 10.0, 8.0, 0.1)
    year = c2.selectbox("Year", ORDINAL["year"], index=1)
    attendance = st.select_slider("Attendance (%)", ORDINAL["attendance"], value="80-90")

    st.subheader("Technical skills")
    c1, c2 = st.columns(2)
    python = c1.selectbox("Python", ORDINAL["python"], index=1)
    sql = c2.selectbox("SQL", ORDINAL["sql"], index=1)
    ml = c1.selectbox("Machine Learning", ORDINAL["machine_learning"], index=1)
    da = c2.selectbox("Data Analytics", ORDINAL["data_analyst"], index=1)
    dsa = c1.selectbox("DSA", ORDINAL["dsa"], index=1)
    entry = c2.selectbox("Entry-level skills", ORDINAL["entry_level_skills"], index=1)

    st.subheader("Experience")
    c1, c2 = st.columns(2)
    projects = c1.selectbox("Technical projects", ORDINAL["technical_projects"], index=1)
    coding = c2.selectbox("Coding problems solved", ORDINAL["coding_problems"], index=1)
    internship = c1.radio("Internship", ORDINAL["internship"], index=0, horizontal=True)
    github = c2.radio("GitHub profile", ORDINAL["github"], index=0, horizontal=True)
    hackathon = c1.radio("Hackathon", ORDINAL["hackathon"], index=0, horizontal=True)

    st.subheader("Placement preparation")
    c1, c2 = st.columns(2)
    interview = c1.selectbox("Interview practice", ORDINAL["interview_practice"], index=2)
    prep = c2.selectbox("Daily preparation hours", ORDINAL["preparation_hours"], index=1)
    resume = c1.radio("Resume ready", ORDINAL["resume"], index=1, horizontal=True)
    mock = c2.radio("Mock interview done", ORDINAL["mock_interview"], index=0, horizontal=True)
    tconf = c1.selectbox("Technical confidence", ORDINAL["technical_confidence"], index=2)
    jconf = c2.selectbox("Job confidence", ORDINAL["job_confidence"], index=1)

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "cgpa": cgpa, "year": year, "attendance": attendance, "python": python, "sql": sql,
        "machine_learning": ml, "data_analyst": da, "dsa": dsa, "technical_projects": projects,
        "internship": internship, "github": github, "hackathon": hackathon,
        "interview_practice": interview, "coding_problems": coding, "resume": resume,
        "mock_interview": mock, "technical_confidence": tconf, "job_confidence": jconf,
        "preparation_hours": prep, "entry_level_skills": entry,
    }])
    probs = pd.Series(model.predict_proba(encode(row))[0], index=model.classes_)[CLASSES]
    label = probs.idxmax()

    {"Job Ready": st.success, "Needs Improvement": st.warning, "Not Job Ready": st.error}[label](
        f"### Prediction: {label}  \nConfidence: {probs.max():.0%}")
    st.bar_chart(probs.rename("Probability"))

    tips = []
    if python in ("Beginner", "Basic") or dsa in ("Beginner", "Basic"):
        tips.append("Strengthen Python and DSA fundamentals.")
    if coding in ("0", "1-25"):
        tips.append("Solve more coding problems (aim for 50+).")
    if projects in ("0", "1"):
        tips.append("Build at least 2–3 technical projects and publish them on GitHub.")
    if internship == "No":
        tips.append("Look for an internship or a virtual work experience.")
    if mock == "No" or interview in ("Never", "Rarely"):
        tips.append("Practice interviews regularly and do mock interviews.")
    if prep in ("0", "1-3"):
        tips.append("Increase daily preparation time.")
    if tips:
        st.subheader("Suggestions")
        for t in tips:
            st.write("• " + t)

with st.expander("About the model"):
    acc = metrics["cv_accuracy"][metrics["best_model"]]
    st.write(f"**Model:** {metrics['best_model']} · trained on {metrics['n_samples']} student survey responses.")
    st.write(f"**Cross-validated accuracy:** {acc['mean']:.0%} (±{acc['std']:.0%}); random guessing on the "
             "most common class would be about 38%.")
    st.caption("Treat results as guidance, not a verdict. Data is small and self-reported, "
               "so predictions are indicative only.")
    st.write("**Most influential factors**")
    imp = pd.Series(metrics["feature_importance"]).head(8)
    st.bar_chart(imp)
