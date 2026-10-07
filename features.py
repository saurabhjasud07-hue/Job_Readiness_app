"""Shared feature definitions so training and the app encode data identically."""
import pandas as pd

TARGET = "job_readiness"
CLASSES = ["Not Job Ready", "Needs Improvement", "Job Ready"]

SKILL = ["Beginner", "Basic", "Intermediate", "Advanced"]
YESNO = ["No", "Yes"]

# column -> ordered categories (low -> high)
ORDINAL = {
    "year": ["1st Year", "2nd Year", "3rd Year", "4th Year"],
    "attendance": ["1-10", "10-20", "20-30", "30-40", "40-50", "50-60",
                   "60-70", "70-80", "80-90", "90-100"],
    "python": SKILL, "sql": SKILL, "machine_learning": SKILL,
    "data_analyst": SKILL, "dsa": SKILL,
    "technical_projects": ["0", "1", "2", "3", "More than 3"],
    "internship": YESNO, "github": YESNO, "hackathon": YESNO,
    "interview_practice": ["Never", "Rarely", "Sometimes", "Frequently", "Regularly"],
    "coding_problems": ["0", "1-25", "26-50", "51-100", "More than 100"],
    "resume": YESNO, "mock_interview": YESNO,
    "technical_confidence": ["Very Low", "Low", "Average", "High", "Very High"],
    "job_confidence": ["Not confident", "Neutral", "Confident", "Very Confident"],
    "preparation_hours": ["0", "1-3", "4-7", "8-12", "More than 12 hours"],
    "entry_level_skills": ["No", "Partially", "Yes"],
}
NUMERIC = ["cgpa"]
FEATURES = NUMERIC + list(ORDINAL)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].astype(str).str.strip()
    # one entry is 86.0 (an 86% typed instead of a CGPA) -> convert to 8.6
    df.loc[df["cgpa"] > 10, "cgpa"] = df.loc[df["cgpa"] > 10, "cgpa"] / 10
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Map every categorical answer to its ordinal rank (0..n-1)."""
    out = pd.DataFrame(index=df.index)
    out["cgpa"] = df["cgpa"].astype(float)
    for col, order in ORDINAL.items():
        out[col] = df[col].astype(str).str.strip().map({v: i for i, v in enumerate(order)})
    return out[FEATURES]
