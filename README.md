# Student Job Readiness Predictor

Streamlit app that predicts whether a student is **Job Ready**, **Needs Improvement** or **Not Job Ready**
from a survey of academics, skills, projects and placement preparation.

## Files
| File | Purpose |
|---|---|
| `data/student_job_readiness_cleaned.csv` | Survey data (143 responses) |
| `features.py` | Cleaning + ordinal encoding shared by training and the app |
| `train.py` | Compares 3 models with repeated 5-fold CV, saves the best to `model.joblib` |
| `app.py` | Streamlit UI |

## Run locally
```bash
pip install -r requirements.txt
python train.py        # optional - model.joblib is already included
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io → **Create app** → pick the repo, branch `main`, main file `app.py`.
3. Deploy.

## Notes
- Cross-validated accuracy is ~57% (3 classes; majority-class baseline ≈ 38%). The dataset is small and self-reported, so use predictions as guidance only. More responses will improve it - add rows and rerun `train.py`.
- One CGPA value of 86 was treated as 8.6.
