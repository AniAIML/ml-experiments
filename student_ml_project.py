"""
=====================================================================================
  STUDENT PERFORMANCE PREDICTION, RISK CLASSIFICATION & STUDENT SEGMENTATION SYSTEM
  Capstone 1 : Student Performance & Early-Intervention System   (single-file project)
=====================================================================================

Everything for the capstone lives in this ONE file, including the dataset (embedded,
compressed) and an interactive Streamlit dashboard.

    REGRESSION      -> predicts Final_Exam_Score        (Linear Regression / Random Forest / XGBoost)
    CLASSIFICATION  -> predicts At_Risk (Yes / No)      (Logistic Regression / Decision Tree / Random Forest)
    CLUSTERING      -> discovers learning personas      (K-Means, elbow + silhouette, PCA)
    DECISION ENGINE -> Risk Level (Low/Medium/High), flag (🟢 Normal / 🟡 Monitor / 🔴 Intervention Required)
                       and a persona-aware recommendation for every student.

Two ways to run it
------------------
    streamlit run student_ml_project.py        -> interactive dashboard (this is what you deploy)
    python student_ml_project.py               -> full text report + charts written to ./output

Deploy on Streamlit Community Cloud
-----------------------------------
    1. Put this file and a requirements.txt in a GitHub repo.
    2. requirements.txt:  streamlit  pandas  numpy  scikit-learn  matplotlib  seaborn  xgboost
    3. share.streamlit.io -> New app -> pick the repo -> main file = student_ml_project.py -> Deploy.
"""

import argparse
import base64
import gzip
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, confusion_matrix,
                             f1_score, fbeta_score, make_scorer, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:                          # XGBoost is optional
    HAS_XGB = False

if hasattr(sys.stdout, "reconfigure"):       # emoji on Windows consoles
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# =============================================================================
# CONFIGURATION
# =============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20

ID_COL = "StudentID"
REG_TARGET = "Final_Exam_Score"
CLF_TARGET = "At_Risk"

RAW_FEATURES = ["Study_Hours_per_week", "Attendance_Percent", "Previous_GPA",
                "Extracurricular_Score", "Sleep_Hours", "Parental_Support_Score", "Part_Time_Job"]
ENGINEERED = ["Engagement_Index", "Study_Sleep_Ratio", "Support_Adjusted_GPA"]
FEATURES = RAW_FEATURES + ENGINEERED

# Clustering uses ONLY behaviour (never the targets): "what kind of learner is this?"
CLUSTER_FEATURES = ["Study_Hours_per_week", "Attendance_Percent", "Sleep_Hours", "Extracurricular_Score"]

VALID_RANGES = {"Study_Hours_per_week": (0, 80), "Attendance_Percent": (0, 100), "Previous_GPA": (0, 4),
                "Extracurricular_Score": (0, 10), "Sleep_Hours": (0, 14),
                "Parental_Support_Score": (0, 10), "Part_Time_Job": (0, 1)}

SCORE_LOW, SCORE_MID = 50, 70           # predicted-score bands from the brief
ATTENDANCE_WARN, SLEEP_WARN = 65, 5.5   # behavioural red flags

FLAG_RED, FLAG_YELLOW, FLAG_GREEN = "🔴 Intervention Required", "🟡 Monitor", "🟢 Normal"
FLAG_ORDER = [FLAG_RED, FLAG_YELLOW, FLAG_GREEN]
FLAG_COLORS = ["#d62728", "#f2c230", "#2ca02c"]

sns.set_theme(style="whitegrid")

# =============================================================================
# EMBEDDED DATASET (student_capstone.csv, gzip + base64) so the file is self-contained
# =============================================================================
_STUDENT_CSV_B64 = "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"

# Instructor answer key: NEVER used for training, only to score the clustering afterwards
_ANSWER_KEY_B64 = "H4sIAAAAAAAC/4Vay24cRwy86yv0AWukm2S/jknkAAF8CKDcjYU8WC8grYyx1snnZ40cVVW8Ts90N8kiu6bYj2/XL9vl7c+Hw9/7dfv8aTvul/Pl9Pmvbf/+ejne1cPj87Z9+/CwfdvPP7Yv97/vx5eXbb+zw8d/3/bj03Xfz0/X5+P+4Y/Xp+v37cudHx7f9uvp9Hyb5/6X+0+v/9x/vJyOp+3lttBdyNEmR7scHXJ0MkMWNaSWw8P5+Xy6fX7/69PX8/bj9nqth9+Oz8fL022Kx/+dd1eNTV4dvB1o2gZe7Hxr2thKra0LLG6FLmQcANRoczoSfKGGNtbfe8UGeDb5vIttxgsdqRzcAvcg1o5i7Y3P0dH7Az2cYDUU3KBmBo1t0NgGjW3w2AY3OECEA9kbFNKBrG5F1xke4UZtb7qyNe6ABrK78exug26BA70tXTspDHoF/uvUC50ioOva3lGG905n0yWuA/z39f7Z0EAYyPhhYB4d/IESfTRm3EiOMlDjBjB46JjP8v6TyZE/gdnT+es63BNgfqLyNlG6T5ruE6X7ouheKMALWLq4pQsFd/GitpCZC8R08XReyMxaICUpiJMUTkoKYiUl0ENES4oGby2DrzwF2xJMrNL41lrFZyDOtbr4ADmhalZaa+KPG1njC07IMSlpqcY9YZCaIheYcIFBemqNL9vRCgPOAq01EXgvYgzZeyNoYAl3ztBDrMCt9s6HEnLuIgkghauhD7EayBM/SZz8yJNxCITgLokkDyLxS0xkBXRIQu+q4He1QYTcyB16GtTaxh3ROpxLlIGGTG8L/RMmlkM2Vzmdqx0a3kVWdPiviqpAwuFqF5mAyFwd8PwbCP4jgf/gJWEI40dyGAz10564Y8DyOJAjJj8GJkTAhKifSQ2YHP+T4x/SvDozzYKLFhO5YCWZsCoXQXg6CBpYF2IHggfWhZICEcG6UAFYmt5bEbpNqWIMcAIrXLVB1NBKSzaHtJsyxK4m3wCIv3FOaJWrVlWXBauejEcy3vjSKC2sDv4B9whniGY6L8wEMhBbNKHnQbJolkADynqQMZpxFwjaaJ74wDlAXOia6Kg0QSDNwVFpnDpaQh3NJ9wBFHWjwKfgrLTgqqYFNFqofRbI6OjiA1EVECm0QPUg4YQmOKG1pCpAdmgNor81sQ4PfUO69o0X6n0lJ0QHCpB1BIGeeADSREskP+vCF10gosNigDQ/6xD8iexnUPezhDMa54w20Bk5YFMjEf9sQONHAoVEC7QpyMLk5RCJgja5H6YoDEgSNEgWbcIuj2jzIIpoS9gMhUFbohIumAIrSYGVHIZLpMFKjoOl+l7AIV54z6tUPmRoKuQNL9obnrBGL53vArbBOGt0ISg67O06542e8EZHnV6HrV6vDT6Frb+k0et1ChsRAqygh9xuS+w2T8YTQHBh0a2LNuwQY4gkuSFvJDzRkcTonrjEE5d4oEm5Hzhf9IQvOuwSu8iLQPAI0Q8X1NEDdsRhTkRSFwJmB+wUO28VewjTG6+Ngjo6lBO9ufgixJi4G9DAb5Mj2uhITvQmbO9FjPHi0NHBwBvE3mHsu7oPAeOe6IreOQKQrOhQVnTIER01h32IcI+kAvIusQsx0VGP2KGC6IPfgZnQcigh+kzqHmeFznVEnyL6EyE+0RF9zmSce2Oh6rdE8gu66Cs5BmBX2YWc6LCv7AlJ9AVvCvFiEJwkBieJUTQ6oiT3Aou4P1QaX7Yn0w4xLfBMFP0fFZA5xk/mKL/iN6pEPzoStTGS1nRUdL0q4ZRRkVuqAIy4PxiWOMbQ8RmiQR1Qc4xEcwxBJgNKjwFpZPD7hMEvFAbXGwM2rANdJgwPPktiPrxYGD74hPzHIvA9Q6g3hrhkmDSoIwQIAnDo4L3pSHrTge8cBoRAJBWigVPkP8DALUBoLQAA"


def _decode(b64: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(gzip.decompress(base64.b64decode(b64))))


def load_embedded_data() -> pd.DataFrame:
    return _decode(_STUDENT_CSV_B64)


def load_answer_key() -> pd.DataFrame:
    return _decode(_ANSWER_KEY_B64)


# =============================================================================
# STEP 1-2  VALIDATE + CLEAN
# =============================================================================
def validate(df: pd.DataFrame) -> list[str]:
    problems = []
    missing = [c for c in [ID_COL] + RAW_FEATURES if c not in df.columns]
    if missing:
        problems.append(f"Missing required columns: {missing}")
    if len(df) == 0:
        problems.append("The file has no rows")
    return problems


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop duplicate IDs, coerce numerics, null-out impossible values, impute with median."""
    df = df.copy()
    before = len(df)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates(subset=[ID_COL])
    nulled = 0
    for col, (lo, hi) in VALID_RANGES.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad = (df[col] < lo) | (df[col] > hi)
        nulled += int(bad.sum())
        df.loc[bad, col] = np.nan
    imputed = int(df[RAW_FEATURES].isnull().sum().sum())
    for col in RAW_FEATURES:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    if CLF_TARGET in df:
        df[CLF_TARGET] = df[CLF_TARGET].astype(str).str.strip().str.title()
        df["At_Risk_Flag"] = (df[CLF_TARGET] == "Yes").astype(int)
    if REG_TARGET in df:
        df[REG_TARGET] = pd.to_numeric(df[REG_TARGET], errors="coerce")
        df = df.dropna(subset=[REG_TARGET])
    df = df.reset_index(drop=True)
    info = {"rows_before": before, "rows_after": len(df), "duplicates_removed": before - len(df),
            "out_of_range_nulled": nulled, "values_imputed": imputed}
    return df, info


# =============================================================================
# STEP 3  EDA  (returns figures; caller decides whether to save or display)
# =============================================================================
def eda_figures(df: pd.DataFrame) -> dict[str, plt.Figure]:
    figs = {}
    num = RAW_FEATURES + [REG_TARGET]
    corr = df[num].corr()

    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    for ax, col in zip(axes.ravel(), num):
        sns.histplot(df[col], kde=(col != "Part_Time_Job"), ax=ax, bins=25); ax.set_title(col)
    fig.suptitle("Feature distributions"); fig.tight_layout(); figs["01_distributions"] = fig

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title("Correlation matrix"); fig.tight_layout(); figs["02_correlation"] = fig

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, col in zip(axes, ["Study_Hours_per_week", "Attendance_Percent", "Previous_GPA", "Sleep_Hours"]):
        sns.scatterplot(data=df, x=col, y=REG_TARGET, hue=CLF_TARGET, alpha=.7, ax=ax,
                        palette={"No": "tab:blue", "Yes": "tab:red"}); ax.set_title(f"{col} vs score")
    fig.tight_layout(); figs["03_scatter_vs_score"] = fig

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df[CLF_TARGET].value_counts().plot.bar(ax=axes[0], color=["tab:blue", "tab:red"]); axes[0].set_title("At_Risk balance")
    sns.boxplot(data=df, x=CLF_TARGET, y=REG_TARGET, ax=axes[1], palette={"No": "tab:blue", "Yes": "tab:red"})
    axes[1].set_title("Final score by At_Risk"); fig.tight_layout(); figs["04_at_risk"] = fig
    return figs


# =============================================================================
# STEP 4  FEATURE ENGINEERING
# =============================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features built only from inputs (no target leakage)."""
    df = df.copy()
    df["Engagement_Index"] = (0.5 * df["Attendance_Percent"]
                              + 0.3 * (df["Study_Hours_per_week"].clip(upper=40) / 40 * 100)
                              + 0.2 * (df["Extracurricular_Score"] * 10))          # "showing up" composite
    df["Study_Sleep_Ratio"] = df["Study_Hours_per_week"] / (df["Sleep_Hours"] + 0.5)  # cramming signature
    df["Support_Adjusted_GPA"] = df["Previous_GPA"] * (1 + df["Parental_Support_Score"] / 20)
    return df


# =============================================================================
# STEP 5  REGRESSION
# =============================================================================
def run_regression(train, test) -> dict:
    X_tr, y_tr, X_te, y_te = train[FEATURES], train[REG_TARGET], test[FEATURES], test[REG_TARGET]
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=3,
                                               random_state=RANDOM_STATE, n_jobs=-1),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
                                         random_state=RANDOM_STATE, n_jobs=-1)
    cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = [], {}
    for name, model in models.items():
        pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
        cv_mae = -cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="neg_mean_absolute_error").mean()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({"Model": name, "CV MAE": cv_mae, "Test MAE": mean_absolute_error(y_te, pred),
                     "Test RMSE": np.sqrt(mean_squared_error(y_te, pred)), "Test R2": r2_score(y_te, pred)})
        fitted[name] = pipe
    table = pd.DataFrame(rows).set_index("Model").round(3)
    best = table["CV MAE"].idxmin()                 # selected on validation, never on the test set
    coef = pd.Series(fitted["Linear Regression"].named_steps["model"].coef_, index=FEATURES).sort_values()

    fig, axes = plt.subplots(1, len(fitted), figsize=(6 * len(fitted), 5))
    for ax, (name, pipe) in zip(np.atleast_1d(axes), fitted.items()):
        p = pipe.predict(X_te); ax.scatter(y_te, p, alpha=.6); ax.plot([10, 100], [10, 100], "r--", lw=1)
        ax.set_xlabel("Actual score"); ax.set_ylabel("Predicted score")
        ax.set_title(f"{name}\nMAE={mean_absolute_error(y_te, p):.2f}  R2={r2_score(y_te, p):.3f}")
    fig.tight_layout()
    fig2, ax = plt.subplots(figsize=(8, 4.5))
    coef.plot.barh(ax=ax, color=np.where(coef > 0, "tab:green", "tab:red"))
    ax.set_title("Linear Regression: standardized coefficients (points of score per 1 SD)"); fig2.tight_layout()
    return {"table": table, "best": best, "model": fitted[best], "coef": coef,
            "figs": {"05_regression_actual_vs_pred": fig, "05b_regression_coefficients": fig2}}


# =============================================================================
# STEP 6  CLASSIFICATION
# =============================================================================
def run_classification(train, test) -> dict:
    X_tr, y_tr, X_te, y_te = train[FEATURES], train["At_Risk_Flag"], test[FEATURES], test["At_Risk_Flag"]
    models = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, class_weight="balanced",
                                                random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=3,
                                                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    f2 = make_scorer(fbeta_score, beta=2)           # recall weighted 2x precision
    rows, fitted, cms = [], {}, {}
    for name, model in models.items():
        pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
        cv_recall = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="recall").mean()
        cv_f2 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring=f2).mean()
        pipe.fit(X_tr, y_tr)
        pred, proba = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]
        rows.append({"Model": name, "CV Recall": cv_recall, "CV F2": cv_f2,
                     "Accuracy": accuracy_score(y_te, pred), "Precision": precision_score(y_te, pred),
                     "Recall": recall_score(y_te, pred), "F1": f1_score(y_te, pred),
                     "ROC-AUC": roc_auc_score(y_te, proba)})
        fitted[name] = pipe
        cms[name] = confusion_matrix(y_te, pred, labels=[0, 1])
    table = pd.DataFrame(rows).set_index("Model").round(3)
    best = table["CV F2"].idxmax()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred No", "Pred Yes"], yticklabels=["True No", "True Yes"])
        ax.set_title(f"{name}\nmissed {cm[1, 0]} of {cm[1].sum()} at-risk students")
    fig.tight_layout()
    return {"table": table, "best": best, "model": fitted[best], "cms": cms,
            "baseline_accuracy": float(1 - y_te.mean()), "figs": {"06_confusion_matrices": fig}}


# =============================================================================
# STEP 7  CLUSTERING
# =============================================================================
def name_persona(z: pd.Series) -> str:
    """Interpret a cluster from its standardized profile (assigned AFTER clustering)."""
    study, att, sleep, extra = z["Study_Hours_per_week"], z["Attendance_Percent"], z["Sleep_Hours"], z["Extracurricular_Score"]
    if study > 0.3 and sleep < -0.3:  return "Sleep-Deprived Crammer"
    if study > 0.3 and att > 0.3:     return "Diligent Achiever"
    if study < -0.3 and att < -0.3:   return "Struggling / Low Engagement"
    if extra > 0.3:                   return "Extracurricular-Focused"
    return "Balanced Student"


def run_clustering(df: pd.DataFrame, k_override: int | None = None, answer_key: pd.DataFrame | None = None) -> dict:
    df = df.copy()
    scaler = StandardScaler().fit(df[CLUSTER_FEATURES])
    X = scaler.transform(df[CLUSTER_FEATURES])

    ks = list(range(2, 11)); inertia, sil = [], []
    for k in ks:
        km = KMeans(k, n_init=20, random_state=RANDOM_STATE).fit(X)
        inertia.append(km.inertia_); sil.append(silhouette_score(X, km.labels_))
    ksel = pd.DataFrame({"k": ks, "inertia": np.round(inertia, 1), "silhouette": np.round(sil, 3)}).set_index("k")

    if k_override:
        k = k_override
    else:
        # Silhouette is flat across 3-5 here; the elbow bends at 4-5. Take the largest k (<=6)
        # whose silhouette is within 0.03 of the best -> distinct but still interpretable.
        best_sil = max(sil)
        k = max(kk for kk, s in zip(ks, sil) if s >= best_sil - 0.03 and kk <= 6)

    kmeans = KMeans(k, n_init=20, random_state=RANDOM_STATE).fit(X)
    df["Cluster"] = kmeans.labels_
    profile = df.groupby("Cluster")[CLUSTER_FEATURES].mean().round(1)
    profile_z = pd.DataFrame(X, columns=CLUSTER_FEATURES).assign(Cluster=df["Cluster"]).groupby("Cluster").mean()
    names = {c: name_persona(r) for c, r in profile_z.iterrows()}
    seen = {}
    for c, n in list(names.items()):
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1: names[c] = f"{n} ({seen[n]})"
    df["Persona"] = df["Cluster"].map(names)
    profile["n"] = df["Cluster"].value_counts().sort_index()
    profile["Persona"] = pd.Series(names)

    check = None
    if REG_TARGET in df and "At_Risk_Flag" in df:
        check = df.groupby("Persona").agg(n=("Cluster", "size"), mean_final_score=(REG_TARGET, "mean"),
                                          at_risk_share=("At_Risk_Flag", "mean")).round(2) \
                  .sort_values("mean_final_score", ascending=False)
    ari = agree = None
    if answer_key is not None and ID_COL in df:
        m = df[[ID_COL, "Persona"]].merge(answer_key, on=ID_COL)
        if len(m):
            ari = adjusted_rand_score(m["True_Learning_Persona"], m["Persona"])
            agree = float((m["Persona"] == m["True_Learning_Persona"]).mean())

    pca = PCA(2, random_state=RANDOM_STATE).fit(X)
    figs = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(ks, inertia, "o-"); axes[0].set_title("Elbow method"); axes[0].set_xlabel("k"); axes[0].set_ylabel("inertia")
    axes[1].plot(ks, sil, "o-", color="tab:orange"); axes[1].set_title("Silhouette score"); axes[1].set_xlabel("k")
    for ax in axes: ax.axvline(k, color="grey", ls="--", lw=1)
    fig.tight_layout(); figs["07_elbow_silhouette"] = fig

    pcs, cent = pca.transform(X), pca.transform(kmeans.cluster_centers_)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=df["Persona"], alpha=.75, ax=ax)
    ax.scatter(cent[:, 0], cent[:, 1], marker="X", s=200, c="black", label="centroids")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)"); ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
    ax.set_title(f"K-Means personas (k={k}) in PCA space"); ax.legend(fontsize=8); fig.tight_layout(); figs["08_pca_clusters"] = fig

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(profile_z.rename(index=names), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Persona profiles (standardized: + above average, - below)"); fig.tight_layout(); figs["09_persona_heatmap"] = fig

    return {"df": df, "k": k, "ksel": ksel, "profile": profile, "profile_z": profile_z, "names": names,
            "check": check, "ari": ari, "agree": agree, "scaler": scaler, "kmeans": kmeans, "figs": figs}


# =============================================================================
# STEP 8  DECISION ENGINE
# =============================================================================
PERSONA_ACTION = {
    "Sleep-Deprived Crammer": {
        FLAG_RED: "Academic counselling + structured sleep/study timetable; shorter spaced sessions instead of longer cram sessions.",
        FLAG_YELLOW: "Coach on spaced revision and a fixed bedtime; watch for burnout before exams.",
        FLAG_GREEN: "Performing, but at a cost: nudge toward more sleep to protect consistency."},
    "Struggling / Low Engagement": {
        FLAG_RED: "Attendance contract, weekly faculty check-in, and remedial academic support.",
        FLAG_YELLOW: "Attendance monitoring and a peer study group to rebuild engagement.",
        FLAG_GREEN: "Engagement is low despite results; keep a light-touch attendance watch."},
    "Extracurricular-Focused": {
        FLAG_RED: "Time-management plan that ring-fences study hours around activities; reduce activity load temporarily.",
        FLAG_YELLOW: "Help balance the activity schedule with fixed study blocks before assessments.",
        FLAG_GREEN: "Well-rounded profile; consider leadership or enrichment roles."},
    "Diligent Achiever": {
        FLAG_RED: "Unexpected for this persona: check for external stressors and review study technique, not effort.",
        FLAG_YELLOW: "Review study strategy and exam technique; effort is already high.",
        FLAG_GREEN: "Offer advanced coursework or peer-mentoring opportunities."},
    "Balanced Student": {
        FLAG_RED: "Academic counselling plus a targeted skills review to find the specific gap.",
        FLAG_YELLOW: "Light-touch tutoring in the weakest subject; re-assess next cycle.",
        FLAG_GREEN: "Maintain current routine."},
}
DEFAULT_ACTION = {FLAG_RED: "Refer to academic counselling and assign a faculty mentor.",
                  FLAG_YELLOW: "Check in at the next assessment cycle.", FLAG_GREEN: "No action needed."}


def decide(r: pd.Series) -> pd.Series:
    """score + at-risk + persona  ->  Risk Level, Flag, Why, Recommendation."""
    score, at_risk = r["Predicted_Score"], r["At_Risk_Pred"] == "Yes"
    reasons, red_flags = [], 0
    if at_risk:              reasons.append(f"classifier: at risk (p={r['At_Risk_Prob']:.0%})")
    if score < SCORE_LOW:    reasons.append(f"predicted score {score:.0f} < {SCORE_LOW}")
    elif score < SCORE_MID:  reasons.append(f"predicted score {score:.0f} in {SCORE_LOW}-{SCORE_MID} band")
    if r["Attendance_Percent"] < ATTENDANCE_WARN:
        reasons.append(f"attendance {r['Attendance_Percent']:.0f}% < {ATTENDANCE_WARN}%"); red_flags += 1
    if r["Sleep_Hours"] < SLEEP_WARN:
        reasons.append(f"sleep {r['Sleep_Hours']:.1f}h < {SLEEP_WARN}h"); red_flags += 1

    if score < SCORE_LOW and at_risk:                       level, flag = "High Risk", FLAG_RED
    elif (score < SCORE_LOW or at_risk) and red_flags >= 1: level, flag = "High Risk", FLAG_RED
    elif score < SCORE_MID or at_risk:                      level, flag = "Medium Risk", FLAG_YELLOW
    else:                                                   level, flag = "Low Risk", FLAG_GREEN
    table = next((t for p, t in PERSONA_ACTION.items() if str(r["Persona"]).startswith(p)), DEFAULT_ACTION)
    return pd.Series({"Risk_Level": level, "Flag": flag,
                      "Why": "; ".join(reasons) or "all indicators healthy", "Recommendation": table[flag]})


def combine_and_flag(df, regressor, classifier) -> pd.DataFrame:
    df = df.copy()
    X = df[FEATURES]
    df["Predicted_Score"] = regressor.predict(X).round(1).clip(0, 100)
    df["At_Risk_Prob"] = classifier.predict_proba(X)[:, 1].round(3)
    df["At_Risk_Pred"] = np.where(df["At_Risk_Prob"] >= 0.5, "Yes", "No")
    df[["Risk_Level", "Flag", "Why", "Recommendation"]] = df.apply(decide, axis=1)
    return df


def flags_by_persona_fig(df) -> plt.Figure:
    ct = pd.crosstab(df["Persona"], df["Flag"]).reindex(columns=FLAG_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ct.plot.bar(stacked=True, ax=ax, color=FLAG_COLORS)
    ax.set_title("Flags by learning persona"); ax.set_ylabel("students"); ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); return fig


FINAL_COLS = [ID_COL, "Predicted_Score", "At_Risk_Pred", "At_Risk_Prob", "Risk_Level", "Persona", "Flag", "Why", "Recommendation"]


# =============================================================================
# FULL PIPELINE (used by both the CLI and the dashboard)
# =============================================================================
def run_pipeline(raw: pd.DataFrame, k_override: int | None = None, use_answer_key: bool = True) -> dict:
    df, clean_info = clean_data(raw)
    df = engineer_features(df)
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["At_Risk_Flag"])
    reg = run_regression(train, test)
    clf = run_classification(train, test)
    clu = run_clustering(df, k_override, load_answer_key() if use_answer_key else None)
    final = combine_and_flag(clu["df"], reg["model"], clf["model"])
    return {"clean_info": clean_info, "df": final, "train_rows": len(train), "test_rows": len(test),
            "reg": reg, "clf": clf, "clu": clu}


def score_new_students(new_raw: pd.DataFrame, res: dict) -> pd.DataFrame:
    """Score an uploaded CSV (targets optional) with the models already trained in `res`."""
    df = new_raw.copy()
    df.columns = [c.strip() for c in df.columns]
    if CLF_TARGET not in df: df[CLF_TARGET] = "No"
    if REG_TARGET not in df: df[REG_TARGET] = np.nan
    df, _ = clean_data(df.assign(**{REG_TARGET: df[REG_TARGET].fillna(0)}))
    df = engineer_features(df)
    clu = res["clu"]
    Xc = clu["scaler"].transform(df[CLUSTER_FEATURES])
    df["Cluster"] = clu["kmeans"].predict(Xc)
    df["Persona"] = df["Cluster"].map(clu["names"])
    return combine_and_flag(df, res["reg"]["model"], res["clf"]["model"])


# =============================================================================
# MODE 1: COMMAND-LINE REPORT      python student_ml_project.py
# =============================================================================
def banner(title):
    print("\n" + "=" * 90 + f"\n  {title}\n" + "=" * 90)


def run_cli():
    ap = argparse.ArgumentParser(description="Student performance / risk / segmentation ML system")
    ap.add_argument("--data", default=None, help="CSV path (default: embedded student_capstone.csv)")
    ap.add_argument("--k", type=int, default=None, help="force number of clusters")
    ap.add_argument("--out", default="output")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.data) if args.data else load_embedded_data()
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 60)

    banner("STEP 1  LOAD & UNDERSTAND THE DATA")
    print(f"Source : {args.data or 'embedded student_capstone.csv'}\nShape  : {raw.shape[0]} rows x {raw.shape[1]} columns\n")
    print(raw.head(), "\n"); print("Missing values:\n", raw.isnull().sum(), "\n")
    print("Duplicate rows:", raw.duplicated().sum()); print("\nSummary statistics:\n", raw.describe().T.round(2))
    share = (raw[CLF_TARGET].str.strip().str.title() == "Yes").mean()
    print(f"\nAt_Risk balance: {share:.1%} Yes / {1 - share:.1%} No  -> imbalanced, so recall matters more than accuracy")

    res = run_pipeline(raw, args.k)
    df, ci = res["df"], res["clean_info"]

    banner("STEP 2  DATA CLEANING")
    print(f"Rows {ci['rows_before']} -> {ci['rows_after']} | duplicate IDs removed: {ci['duplicates_removed']} | "
          f"out-of-range values nulled: {ci['out_of_range_nulled']} | values imputed: {ci['values_imputed']}")

    banner("STEP 3  EXPLORATORY DATA ANALYSIS")
    corr = df[RAW_FEATURES + [REG_TARGET]].corr()[REG_TARGET].drop(REG_TARGET)
    print("Correlation with Final_Exam_Score:\n" + corr.sort_values(ascending=False).round(2).to_string())
    print("\nMean of each feature by At_Risk:\n" + df.groupby(CLF_TARGET)[RAW_FEATURES].mean().round(2).T.to_string())

    banner("STEP 4  FEATURE ENGINEERING + TRAIN/TEST SPLIT")
    print("Engineered features:", ENGINEERED)
    print(f"Train {res['train_rows']} rows / Test {res['test_rows']} rows (stratified on At_Risk)")

    banner("STEP 5  REGRESSION  ->  predict Final_Exam_Score")
    print(res["reg"]["table"].to_string())
    print(f"\n>> Selected (lowest cross-validated MAE): {res['reg']['best']}")
    print("\nLinear Regression standardized coefficients:\n" + res["reg"]["coef"].sort_values(ascending=False).round(2).to_string())

    banner("STEP 6  CLASSIFICATION  ->  predict At_Risk (Yes / No)")
    print(f"Baseline 'always No' accuracy = {res['clf']['baseline_accuracy']:.1%} with recall = 0%\n")
    print(res["clf"]["table"].to_string())
    print("\nConfusion matrices [[TN FP] [FN TP]]  (FN = at-risk students MISSED):")
    for name, cm in res["clf"]["cms"].items():
        print(f"  {name:20s} {cm.tolist()}   missed {cm[1, 0]} of {cm[1].sum()}")
    print(f"\n>> Selected (best cross-validated recall-weighted F2): {res['clf']['best']}")

    clu = res["clu"]
    banner("STEP 7  CLUSTERING  ->  discover learner personas (unsupervised)")
    print("k selection:\n" + clu["ksel"].T.to_string())
    print(f"\n>> Chosen k = {clu['k']}  (silhouette {clu['ksel'].loc[clu['k'], 'silhouette']:.3f})")
    print("\nCluster profiles + persona names:\n" + clu["profile"].to_string())
    print("\nSanity check AFTER clustering (targets not used to cluster):\n" + clu["check"].to_string())
    if clu["ari"] is not None:
        print(f"\nOptional check vs instructor answer key: ARI = {clu['ari']:.3f}, name agreement = {clu['agree']:.1%}")

    banner("STEP 8  COMBINE  ->  Risk Level, Flag, Recommendation")
    print("Flag counts:\n" + df["Flag"].value_counts().to_string())
    print("\nFlags by persona:\n" + pd.crosstab(df["Persona"], df["Flag"]).to_string())

    banner("STEP 9  FINAL STUDENT DASHBOARD")
    final = df[FINAL_COLS]
    final.to_csv(out / "final_student_dashboard.csv", index=False, encoding="utf-8-sig")
    print(final[[ID_COL, "Predicted_Score", "At_Risk_Pred", "Risk_Level", "Persona", "Flag"]].head(15).to_string(index=False))
    ex = df[df[ID_COL] == 214]
    r = (ex if len(ex) else df[df["Flag"] == FLAG_RED]).iloc[0]
    print(f"\nExample conclusion:\n  Student #{r[ID_COL]} -> Predicted score: {r.Predicted_Score} · At_Risk: {r.At_Risk_Pred} · "
          f"Risk level: {r.Risk_Level} · Persona: {r.Persona}\n  Flag: {r.Flag}\n  Why: {r.Why}\n  Recommendation: {r.Recommendation}")

    if not args.no_plots:
        figs = {**eda_figures(df), **res["reg"]["figs"], **res["clf"]["figs"], **clu["figs"],
                "10_flags_by_persona": flags_by_persona_fig(df)}
        for name, fig in figs.items():
            fig.savefig(out / f"{name}.png", dpi=110); plt.close(fig)
        print(f"\n{len(figs)} charts saved to {out}/")
    print(f"Final table written to {out / 'final_student_dashboard.csv'}")

    banner("SUMMARY")
    print(f"Regression      : {res['reg']['best']}\nClassification  : {res['clf']['best']}\n"
          f"Clustering      : K-Means, k={clu['k']} -> {sorted(df['Persona'].unique())}\n"
          f"Decision engine : score + at-risk + persona -> Risk Level / Flag / Recommendation")


# =============================================================================
# MODE 2: STREAMLIT DASHBOARD      streamlit run student_ml_project.py
# =============================================================================
def run_dashboard():
    import streamlit as st

    st.set_page_config(page_title="Student Early-Intervention System", page_icon="🎓", layout="wide")

    @st.cache_resource(show_spinner="Training regression, classification and clustering models...")
    def train_all(k_override):
        return run_pipeline(load_embedded_data(), k_override)

    # ---------------- sidebar
    with st.sidebar:
        st.title("🎓 Controls")
        k_choice = st.selectbox("Number of clusters (k)", ["auto (elbow + silhouette)", 3, 4, 5, 6], index=0)
        k_override = None if isinstance(k_choice, str) else int(k_choice)
        st.divider()
        st.markdown("**Score your own students**")
        uploaded = st.file_uploader("Upload CSV", type=["csv"],
                                    help="Needs: " + ", ".join([ID_COL] + RAW_FEATURES))
        st.caption("Targets are optional in an upload; models trained on the capstone data score the new rows.")

    res = train_all(k_override)
    df, reg, clf, clu = res["df"], res["reg"], res["clf"], res["clu"]

    st.title("🎓 Student Performance & Early-Intervention System")
    st.caption("Regression (predicted score) + Classification (at-risk flag) + Clustering (learning persona) "
               "→ Risk Level, flag and a persona-aware recommendation for every student.")

    tabs = st.tabs(["📊 Overview", "🔍 Data & EDA", "📈 Regression", "🚦 Classification",
                    "🧩 Clustering", "📋 Student Table", "👤 Student Detail", "ℹ️ About"])

    # ---------------- Overview
    with tabs[0]:
        counts = df["Flag"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔴 Intervention Required", int(counts.get(FLAG_RED, 0)))
        c2.metric("🟡 Monitor", int(counts.get(FLAG_YELLOW, 0)))
        c3.metric("🟢 Normal", int(counts.get(FLAG_GREEN, 0)))
        c4.metric("Students", len(df))
        m1, m2, m3 = st.columns(3)
        m1.metric("Regression", reg["best"], f"MAE {reg['table'].loc[reg['best'], 'Test MAE']:.2f} · R² {reg['table'].loc[reg['best'], 'Test R2']:.2f}")
        m2.metric("Classification", clf["best"], f"recall {clf['table'].loc[clf['best'], 'Recall']:.0%} · AUC {clf['table'].loc[clf['best'], 'ROC-AUC']:.2f}")
        m3.metric("Clustering", f"K-Means, k = {clu['k']}", f"silhouette {clu['ksel'].loc[clu['k'], 'silhouette']:.2f}")
        left, right = st.columns(2)
        with left:
            st.markdown("**Flags by learning persona**")
            ct = pd.crosstab(df["Persona"], df["Flag"]).reindex(columns=FLAG_ORDER, fill_value=0)
            st.bar_chart(ct, color=FLAG_COLORS)
        with right:
            st.markdown("**Predicted score distribution**")
            st.bar_chart(df["Predicted_Score"].round(-1).value_counts().sort_index())
        st.markdown("**Pipeline**")
        st.code("student data → clean → feature engineering → [regression | classification | clustering] "
                "→ decision engine → Risk Level + Flag + Recommendation", language="text")

    # ---------------- Data & EDA
    with tabs[1]:
        ci = res["clean_info"]
        st.markdown(f"**Cleaning:** {ci['rows_before']} rows in, {ci['rows_after']} out · duplicates removed: "
                    f"{ci['duplicates_removed']} · out-of-range values nulled: {ci['out_of_range_nulled']} · imputed: {ci['values_imputed']}")
        st.dataframe(df[[ID_COL] + RAW_FEATURES + [REG_TARGET, CLF_TARGET]].head(20), hide_index=True, use_container_width=True)
        st.markdown("**Summary statistics**")
        st.dataframe(df[RAW_FEATURES + [REG_TARGET]].describe().T.round(2), use_container_width=True)
        share = df["At_Risk_Flag"].mean()
        st.info(f"At_Risk is imbalanced: {share:.1%} Yes vs {1 - share:.1%} No. A model that always says 'No' would be "
                f"{1 - share:.0%} accurate while catching nobody, so recall is the metric that matters.")
        for fig in eda_figures(df).values():
            st.pyplot(fig); plt.close(fig)

    # ---------------- Regression
    with tabs[2]:
        st.markdown("Target: `Final_Exam_Score`. Models compared on **5-fold cross-validated MAE** on the training set; "
                    "the test set is used once for the final numbers.")
        st.dataframe(reg["table"], use_container_width=True)
        st.success(f"Selected: **{reg['best']}** (lowest cross-validated MAE). The simplest model wins because the "
                   "relationships in this data are close to linear.")
        for fig in reg["figs"].values():
            st.pyplot(fig); plt.close(fig)

    # ---------------- Classification
    with tabs[3]:
        st.markdown("Target: `At_Risk` (Yes = 1). All models are class-weighted; selection uses cross-validated **F2** "
                    "(recall weighted twice as much as precision) because missing an at-risk student costs more than a false alarm.")
        st.dataframe(clf["table"], use_container_width=True)
        st.success(f"Selected: **{clf['best']}** (best cross-validated recall-weighted F2). "
                   f"Baseline 'always No' accuracy would be {clf['baseline_accuracy']:.1%} with 0% recall.")
        for fig in clf["figs"].values():
            st.pyplot(fig); plt.close(fig)

    # ---------------- Clustering
    with tabs[4]:
        st.markdown("Features clustered: " + ", ".join(f"`{c}`" for c in CLUSTER_FEATURES) +
                    ". **No target column is used.** Persona names are assigned *after* clustering by reading each cluster's profile.")
        st.dataframe(clu["ksel"].T, use_container_width=True)
        st.success(f"Chosen k = **{clu['k']}** (silhouette is flat across 3-5, the elbow bends at 4-5, and 5 clusters give distinct, nameable profiles).")
        st.markdown("**Cluster profiles and persona names**")
        st.dataframe(clu["profile"], use_container_width=True)
        st.markdown("**Sanity check after clustering** (targets were not used to build the clusters)")
        st.dataframe(clu["check"], use_container_width=True)
        if clu["ari"] is not None:
            st.info(f"Optional check against the instructor answer key (never used in training): "
                    f"Adjusted Rand Index = {clu['ari']:.3f}, exact name agreement = {clu['agree']:.1%}")
        for fig in clu["figs"].values():
            st.pyplot(fig); plt.close(fig)

    # ---------------- Student table
    with tabs[5]:
        st.markdown("**Decision rules:** score < 50 and at-risk → 🔴 · (score < 50 or at-risk) + attendance < 65% or sleep < 5.5h → 🔴 · "
                    "score < 70 or at-risk → 🟡 · otherwise 🟢. Recommendation text depends on the persona.")
        source_df = df
        if uploaded is not None:
            try:
                new_raw = pd.read_csv(uploaded)
                problems = validate(new_raw)
                if problems:
                    st.error("; ".join(problems))
                else:
                    source_df = score_new_students(new_raw, res)
                    st.success(f"Scored {len(source_df)} uploaded students with the trained models.")
            except Exception as e:
                st.error(f"Could not read the upload: {e}")
        f1, f2, f3 = st.columns(3)
        flag_sel = f1.multiselect("Flag", FLAG_ORDER, default=FLAG_ORDER)
        personas = sorted(source_df["Persona"].unique())
        persona_sel = f2.multiselect("Persona", personas, default=personas)
        risk_sel = f3.multiselect("At risk", ["Yes", "No"], default=["Yes", "No"])
        view = source_df[source_df["Flag"].isin(flag_sel) & source_df["Persona"].isin(persona_sel) & source_df["At_Risk_Pred"].isin(risk_sel)]
        view = view.sort_values(["At_Risk_Prob", "Predicted_Score"], ascending=[False, True])[FINAL_COLS]
        st.dataframe(view, hide_index=True, use_container_width=True, height=450,
                     column_config={"Predicted_Score": st.column_config.ProgressColumn("Predicted score", min_value=0, max_value=100, format="%.1f"),
                                    "At_Risk_Prob": st.column_config.NumberColumn("P(at risk)", format="%.2f"),
                                    "Why": st.column_config.TextColumn("Why flagged", width="large"),
                                    "Recommendation": st.column_config.TextColumn("Recommended action", width="large")})
        st.download_button("Download this table as CSV", view.to_csv(index=False).encode("utf-8-sig"),
                           "student_recommendations.csv", "text/csv")
        st.session_state["view_ids"] = view[ID_COL].tolist()
        st.session_state["source_df"] = source_df

    # ---------------- Student detail
    with tabs[6]:
        src = st.session_state.get("source_df", df)
        ids = st.session_state.get("view_ids", src[ID_COL].tolist()) or src[ID_COL].tolist()
        default = ids.index(214) if 214 in ids else 0
        sid = st.selectbox("StudentID", ids, index=default)
        r = src[src[ID_COL] == sid].iloc[0]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Predicted score", f"{r.Predicted_Score:.1f}")
        d2.metric("At risk", r.At_Risk_Pred, f"p = {r.At_Risk_Prob:.0%}")
        d3.metric("Persona", r.Persona)
        d4.metric("Flag", r.Flag)
        st.markdown(f"**Risk level:** {r.Risk_Level}")
        st.markdown(f"**Why:** {r.Why}")
        st.markdown(f"**Recommendation:** {r.Recommendation}")
        st.markdown("**Input data for this student**")
        st.dataframe(src[src[ID_COL] == sid][[ID_COL] + RAW_FEATURES], hide_index=True, use_container_width=True)
        st.markdown("**How this student compares with their persona and the whole class**")
        comp = pd.DataFrame({"this student": r[CLUSTER_FEATURES + ["Previous_GPA"]].astype(float),
                             f"persona mean ({r.Persona})": src[src["Persona"] == r.Persona][CLUSTER_FEATURES + ["Previous_GPA"]].mean(),
                             "class mean": src[CLUSTER_FEATURES + ["Previous_GPA"]].mean()}).round(2)
        st.dataframe(comp, use_container_width=True)

    # ---------------- About
    with tabs[7]:
        st.markdown(f"""
### Problem statement
A school wants three angles on the same student data, answered by three different kinds of machine learning,
then combined into one action per student.

| Question | Paradigm | Target |
|---|---|---|
| How much will the student score? | **Regression** | `Final_Exam_Score` |
| Do they need intervention now? | **Classification** | `At_Risk` (Yes/No) |
| What kind of learner are they? | **Clustering** | none, personas are discovered |
| What should the school do? | **Decision engine** | Risk Level + Flag + Recommendation |

### Design decisions
- **Independent models.** All three use the same inputs; no model feeds another. Outputs meet only in the decision engine.
- **Selection by cross-validation, not the test set.** Linear Regression won on merit because the data is nearly linear.
- **Recall over accuracy** for `At_Risk` (about 19% positives). Models are class-weighted and compared on recall / F2.
- **k chosen, not assumed.** Elbow and silhouette are both shown; k = {clu['k']} yields distinct, nameable personas.
- **Persona names come after clustering** from the standardized profile heat-map. The instructor key is only used to
  check the result afterwards (ARI {clu['ari']:.2f}).
- **Recommendations are persona-aware.** The same red flag means a sleep/study timetable for a Sleep-Deprived Crammer
  and an attendance contract for a Struggling / Low Engagement student.

### Engineered features
`Engagement_Index` (attendance + study + extracurricular composite), `Study_Sleep_Ratio` (cramming signature),
`Support_Adjusted_GPA` (GPA scaled by parental support).

### Run locally
```
pip install streamlit pandas numpy scikit-learn matplotlib seaborn xgboost
streamlit run student_ml_project.py      # dashboard
python student_ml_project.py             # text report + charts in ./output
```
""")


# =============================================================================
# ENTRY POINT: detect whether we are running under Streamlit
# =============================================================================
def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _running_in_streamlit():
    run_dashboard()
elif __name__ == "__main__":
    run_cli()
