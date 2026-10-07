"""Train and compare models, save the best one to model.joblib."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import CLASSES, FEATURES, TARGET, clean, encode

from pathlib import Path
CSV = next(p for p in (Path("data/student_job_readiness_cleaned.csv"), Path("student_job_readiness_cleaned.csv")) if p.exists())
df = clean(pd.read_csv(CSV))
X = encode(df)
y = df[TARGET]
assert X.isna().sum().sum() == 0, "unmapped category found"

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
    "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=42),
}

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
scores = {}
for name, m in models.items():
    s = cross_val_score(m, X, y, cv=cv, scoring="accuracy")
    scores[name] = (s.mean(), s.std())
    print(f"{name:22s} accuracy = {s.mean():.3f} +/- {s.std():.3f}")

best_name = max(scores, key=lambda k: scores[k][0])
best = models[best_name]
print("\nBest:", best_name)

pred = cross_val_predict(best, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=1))
print(classification_report(y, pred, labels=CLASSES, zero_division=0))
print(confusion_matrix(y, pred, labels=CLASSES))

best.fit(X, y)

# feature importance (for the app)
if hasattr(best, "feature_importances_"):
    imp = best.feature_importances_
else:
    imp = np.abs(best[-1].coef_).mean(axis=0)
importance = dict(sorted(zip(FEATURES, map(float, imp / imp.sum())), key=lambda kv: -kv[1]))

joblib.dump(best, "model.joblib")
json.dump({
    "best_model": best_name,
    "cv_accuracy": {k: {"mean": round(v[0], 4), "std": round(v[1], 4)} for k, v in scores.items()},
    "n_samples": int(len(df)),
    "feature_importance": importance,
}, open("metrics.json", "w"), indent=2)
print("\nTop features:", list(importance.items())[:8])
