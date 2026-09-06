"""
=====================================================================================
  EMPLOYEE ATTRITION & WORKFORCE ANALYTICS
  Capstone 3 : fair-salary regression, attrition classification, engagement personas  (single-file project)
=====================================================================================

Everything for the capstone lives in this ONE file, including the dataset (embedded,
compressed) and an interactive Streamlit dashboard.

    REGRESSION      -> predicts Expected_Annual_Salary_k ($k)   (Linear Regression / Random Forest / XGBoost,
                       Department one-hot encoded)
    CLASSIFICATION  -> predicts Attrition (Yes / No)            (Logistic Regression / Decision Tree / Random Forest,
                       evaluated with accuracy, precision, recall, ROC-AUC)
    CLUSTERING      -> discovers engagement personas            (K-Means, elbow + silhouette, PCA)
    RETENTION ENGINE-> Retention-risk priority score + tier (🔴 Urgent 1:1 / 🟡 Check-in / 🟢 Develop & retain)
                       and a persona-aware HR action for every employee.
                       If an Actual_Salary_k column is present, the pay gap vs predicted fair salary is used too.

Two ways to run it
------------------
    streamlit run employee_ml_project.py        -> interactive dashboard (this is what you deploy)
    python employee_ml_project.py               -> full text report + charts written to ./output

Deploy on Streamlit Community Cloud
-----------------------------------
    1. Put this file and a requirements.txt in a GitHub repo.
    2. requirements.txt:  streamlit  pandas  numpy  scikit-learn  matplotlib  seaborn  xgboost
    3. share.streamlit.io -> New app -> pick the repo -> main file = employee_ml_project.py -> Deploy.
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
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, confusion_matrix,
                             f1_score, fbeta_score, make_scorer, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# =============================================================================
# CONFIGURATION
# =============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20

ID_COL = "EmployeeID"
REG_TARGET = "Expected_Annual_Salary_k"
CLF_TARGET = "Attrition"
ACTUAL_SALARY = "Actual_Salary_k"            # optional column: enables the underpaid check

CATEGORICAL = ["Department"]
NUMERIC_RAW = ["Age", "Years_at_Company", "Monthly_Hours_Worked", "Performance_Rating", "Satisfaction_Score",
               "Distance_From_Home_km", "Training_Hours_per_year", "Work_Life_Balance_Score"]
ENGINEERED = ["Overtime_Index", "Engagement_Index", "Tenure_x_Performance"]
NUMERIC = NUMERIC_RAW + ENGINEERED
FEATURES = NUMERIC + CATEGORICAL
RAW_FEATURES = NUMERIC_RAW + CATEGORICAL     # what an upload must contain

# Clustering uses the engagement-related features named in the brief (never the targets)
CLUSTER_FEATURES = ["Years_at_Company", "Monthly_Hours_Worked", "Performance_Rating", "Satisfaction_Score",
                    "Training_Hours_per_year", "Work_Life_Balance_Score"]

VALID_RANGES = {"Age": (16, 80), "Years_at_Company": (0, 60), "Monthly_Hours_Worked": (0, 400),
                "Performance_Rating": (1, 5), "Satisfaction_Score": (0, 10), "Distance_From_Home_km": (0, 500),
                "Training_Hours_per_year": (0, 1000), "Work_Life_Balance_Score": (0, 10)}

HOURS_WARN = 190           # monthly hours above this = overwork signal
SATISFACTION_WARN = 4.5
HIGH_PERFORMER = 4.0
UNDERPAID_PCT = -0.10      # actual salary 10%+ below predicted fair salary

TIER_RED, TIER_YELLOW, TIER_GREEN = "🔴 Urgent 1:1", "🟡 Check-in", "🟢 Develop & retain"
TIER_ORDER = [TIER_RED, TIER_YELLOW, TIER_GREEN]
TIER_COLORS = ["#d62728", "#f2c230", "#2ca02c"]

sns.set_theme(style="whitegrid")

# =============================================================================
# EMBEDDED DATASET (employee_capstone.csv, gzip + base64)
# =============================================================================
_DATA_B64 = "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"

# Instructor answer key: NEVER used for training, only to score the clustering afterwards
_ANSWER_KEY_B64 = "H4sIAAAAAAAC/22ZP29cNxDE+/sUV6VS8XZ3SC7bIEaSJn+QIK0hwAdFgH0KFAmGvn0EpPHu7xoDR5F843mznDfLD1/++fz0drn8/MPdn8+vl48frg/3D5cvl+vLx98uz/8+Xe9Pdvf96/P18un86+vLye/+eLncf3o7/3V5uTzfX09x98vl6/m784/PT18frw8ntd+jL5h9YN399Pjw9+e38//P/nTKPmN/i8COPt+sPdO8LIi+n3WQNrDnvPv99fHycn7/9+V91ckA07I8Zfe/O4C6dSQOPj0K3WowfDToPvsAgHr2TYA1jjYlADUciwrUUPmF9x548bH6Q/HmY7cpOvoUWZ9S3r6i/1nYAW9f/e1rlT0TC8DoANBRCmmAztFLaaCWQOqY2KWTOopMRyd0Hu0h0/qWE1BnJ3UKU7pQJ6DODnVmX1Mqf3WRrv7qF5CujnQVma7R/9zLaZU3vzrCtdtAouoTpZTeFwUW9cJPiDS7SHPhQb3yc+NsPb79/+2i0Y3TaZeS3x3k7i99o+h3f+mbxz2KyQ4e+gdotcM5FNU7cPIfg2smJi2MJAGBW7vlVYVha1Z1w6tM3IN2ZZMLO9NmBE3bMj9gp6T63bja5h7cSZg0MELgDuBerdZ70VkAc+AYswDmIOZqY/buY+XnrD/5URCJhwCtgFZVFaKQFVhDYYhiVkWsxRmJnesH1+jnrg2SO6qQB4U8oIZBHY8KF25mI7kxCJ4wX6Ot2XSsq6cFXc3mjS9FKnhSF7NKeLLuYG+2WHd0OFsV9ALNi6pYleZFvKviXTzcsjiHZZVwEmhWoClu2f3DsgKlxVlCvgk9bOphUw8betg8HzYcZJPeDQfZBE7jsw2Wnc7nh3GoVJ8fwX3EocF9Zt2HWYLG5wejj3UtO0Kam3NZcEgcQgYyhCDGNbfEpC4U96Jp95p9vbLcohqzmo86YTL+LSAgv96/2R1251GRRkUagfkAW53OA5QirnmA0aCAmdhczMF0PEdyc/XacxqeCznDaXqurDEbShigeEDBwzFSJTFYdMxwPqiLQcQDZA/W3UTd0fd8VnFMFh2Nz2l8zkTnc9Wt0Xmg7/niCbcIevWPN18Q9YI+FlFX6/Oa73yx/Gh9Xq3Ps+dQz0ox850nKYb7eZLhRBUmIcMBfZNkOqBvkLxRhJuaRuxz5j7flegNOe8OOo5CcxxWf7IXeXS0gcAXdL1A4ItjcW82p5j4wirkmvbCCNkA2WojzQYmsJNmxMuwF9aNJBx9P7/R+CNq77UXXlE7+3/oVAbcL5wkw/2Czcob3cog6AgOiU3NLuaAFUYAeBQ/CfpgCBUYMg6xyyrCZg8z2MQMAbcWJ5FxETwiYNRuZgzSzQgYiIAxwHZNgDFA9aCw6YQxSfekSmY/8IJuGLNKe95obVfMk9WI1mZM6Hr1T45AgzMWeV5EDCMMZsBYk+tANv0w0PKMBHA4YqDnGYkzJCvRiZOPfhhMhJG1FpPiYB6M3b/uYrMU4YaxK2Q4YSAKBqNg7BsXH6UHo2qFOjpaHY6bEtx8wAxFM1SNgGIE1JHYBtc0CICqViimPzH9qXqh4IXCPZ2Y/EQrFPue4n2d3DhUb5dghKpGKBqhmAPFSzvVnqe86iF4Dxa4CEPDU7RABe/DAjTz8k6Ig0IcFFqfUtUxL+/kWELMCILSwEjVMa1PaHyKzic4n5ADNcDzQO2NG9eOoJlJUPA/0f+EWz3hWk90P8H9NMk1s6DogJqoQ9zuqbZAVS/3tKoyYH1aUEbtfAquJ8Y/4YJPC0Bxy6dVsSZEwUs+JdhNqCIBOcltUhXZ7qMBOKGIDcwMgNr1dNuUQzU8wfC0J3cFw0h/2h3wOHDE/QeKMfsGrSIAAA=="
KEY_COL = "True_Engagement_Persona"


def _decode(b64):
    return pd.read_csv(io.BytesIO(gzip.decompress(base64.b64decode(b64))))


def load_embedded_data():
    return _decode(_DATA_B64)


def load_answer_key():
    return _decode(_ANSWER_KEY_B64)


# =============================================================================
# STEP 1-2  VALIDATE + CLEAN
# =============================================================================
def validate(df):
    problems = []
    missing = [c for c in [ID_COL] + RAW_FEATURES if c not in df.columns]
    if missing:
        problems.append(f"Missing required columns: {missing}")
    if len(df) == 0:
        problems.append("The file has no rows")
    return problems


def clean_data(df):
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
    imputed = int(df[NUMERIC_RAW].isnull().sum().sum())
    for col in NUMERIC_RAW:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    df["Department"] = df["Department"].astype(str).str.strip().str.title().replace({"Hr": "HR", "Nan": "Unknown"})
    if CLF_TARGET in df:
        df[CLF_TARGET] = df[CLF_TARGET].astype(str).str.strip().str.title()
        df["Attrition_Flag"] = (df[CLF_TARGET] == "Yes").astype(int)
    if REG_TARGET in df:
        df[REG_TARGET] = pd.to_numeric(df[REG_TARGET], errors="coerce")
        df = df.dropna(subset=[REG_TARGET])
    if ACTUAL_SALARY in df:
        df[ACTUAL_SALARY] = pd.to_numeric(df[ACTUAL_SALARY], errors="coerce")
    df = df.reset_index(drop=True)
    return df, {"rows_before": before, "rows_after": len(df), "duplicates_removed": before - len(df),
                "out_of_range_nulled": nulled, "values_imputed": imputed}


# =============================================================================
# STEP 3  EDA
# =============================================================================
def eda_figures(df):
    figs = {}
    num = NUMERIC_RAW + [REG_TARGET]
    corr = df[num].corr()

    fig, axes = plt.subplots(3, 3, figsize=(18, 11))
    for ax, col in zip(axes.ravel(), num):
        sns.histplot(df[col], kde=True, ax=ax, bins=25); ax.set_title(col)
    fig.suptitle("Feature distributions"); fig.tight_layout(); figs["01_distributions"] = fig

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title("Correlation matrix"); fig.tight_layout(); figs["02_correlation"] = fig

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    order = df.groupby("Department")[REG_TARGET].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="Department", y=REG_TARGET, order=order, ax=axes[0]); axes[0].set_title("Salary by department")
    sns.scatterplot(data=df, x="Years_at_Company", y=REG_TARGET, hue="Department", alpha=.7, ax=axes[1]); axes[1].set_title("Salary vs tenure")
    fig.tight_layout(); figs["03_salary_drivers"] = fig

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    df[CLF_TARGET].value_counts().plot.bar(ax=axes[0], color=["tab:blue", "tab:red"]); axes[0].set_title("Attrition balance")
    for ax, col in zip(axes[1:], ["Satisfaction_Score", "Work_Life_Balance_Score", "Monthly_Hours_Worked"]):
        sns.boxplot(data=df, x=CLF_TARGET, y=col, ax=ax, palette={"No": "tab:blue", "Yes": "tab:red"}); ax.set_title(f"{col} by attrition")
    fig.tight_layout(); figs["04_attrition"] = fig
    return figs


# =============================================================================
# STEP 4  FEATURE ENGINEERING + PREPROCESSOR (one-hot for Department)
# =============================================================================
def engineer_features(df):
    df = df.copy()
    df["Overtime_Index"] = (df["Monthly_Hours_Worked"] - 160) / 40             # 0 = standard month, 1 = +40h
    df["Engagement_Index"] = (df["Satisfaction_Score"] + df["Work_Life_Balance_Score"]) / 2 * df["Performance_Rating"] / 5
    df["Tenure_x_Performance"] = df["Years_at_Company"] * df["Performance_Rating"]
    return df


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


def feature_names(pipe):
    return [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]


# =============================================================================
# STEP 5  REGRESSION  (fair salary)
# =============================================================================
def run_regression(train, test):
    X_tr, y_tr, X_te, y_te = train[FEATURES], train[REG_TARGET], test[FEATURES], test[REG_TARGET]
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=3,
                                               random_state=RANDOM_STATE, n_jobs=-1),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.08, subsample=0.8,
                                         random_state=RANDOM_STATE, n_jobs=1)
    cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = [], {}
    for name, model in models.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
        cv_mae = -cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="neg_mean_absolute_error").mean()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({"Model": name, "CV MAE": cv_mae, "Test MAE": mean_absolute_error(y_te, pred),
                     "Test RMSE": np.sqrt(mean_squared_error(y_te, pred)), "Test R2": r2_score(y_te, pred)})
        fitted[name] = pipe
    table = pd.DataFrame(rows).set_index("Model").round(3)
    best = table["CV MAE"].idxmin()
    lin = fitted["Linear Regression"]
    coef = pd.Series(lin.named_steps["model"].coef_, index=feature_names(lin)).sort_values()

    fig, axes = plt.subplots(1, len(fitted), figsize=(6 * len(fitted), 5))
    for ax, (name, pipe) in zip(np.atleast_1d(axes), fitted.items()):
        p = pipe.predict(X_te); ax.scatter(y_te, p, alpha=.6)
        lim = [min(y_te.min(), p.min()) * .95, max(y_te.max(), p.max()) * 1.05]; ax.plot(lim, lim, "r--", lw=1)
        ax.set_xlabel("Actual salary ($k)"); ax.set_ylabel("Predicted fair salary ($k)")
        ax.set_title(f"{name}\nMAE=${mean_absolute_error(y_te, p):.1f}k  R2={r2_score(y_te, p):.3f}")
    fig.tight_layout()
    fig2, ax = plt.subplots(figsize=(8, 5.5))
    coef.plot.barh(ax=ax, color=np.where(coef > 0, "tab:green", "tab:red"))
    ax.set_title("Linear Regression coefficients ($k): numeric per 1 SD, Department one-hot"); fig2.tight_layout()
    return {"table": table, "best": best, "model": fitted[best], "coef": coef,
            "figs": {"05_regression_actual_vs_pred": fig, "05b_regression_coefficients": fig2}}


# =============================================================================
# STEP 6  CLASSIFICATION  (attrition)
# =============================================================================
def run_classification(train, test):
    X_tr, y_tr, X_te, y_te = train[FEATURES], train["Attrition_Flag"], test[FEATURES], test["Attrition_Flag"]
    models = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, class_weight="balanced",
                                                random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=3,
                                                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    f2 = make_scorer(fbeta_score, beta=2)
    rows, fitted, cms, rocs = [], {}, {}, {}
    for name, model in models.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
        cvres = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring={"recall": "recall", "f2": f2, "auc": "roc_auc"})
        pipe.fit(X_tr, y_tr)
        pred, proba = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]
        rows.append({"Model": name, "CV Recall": cvres["test_recall"].mean(), "CV F2": cvres["test_f2"].mean(),
                     "CV ROC-AUC": cvres["test_auc"].mean(),
                     "Accuracy": accuracy_score(y_te, pred), "Precision": precision_score(y_te, pred),
                     "Recall": recall_score(y_te, pred), "F1": f1_score(y_te, pred), "ROC-AUC": roc_auc_score(y_te, proba)})
        fitted[name] = pipe
        cms[name] = confusion_matrix(y_te, pred, labels=[0, 1])
        rocs[name] = roc_curve(y_te, proba)
    table = pd.DataFrame(rows).set_index("Model").round(3)
    best = table["CV F2"].idxmax()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred stay", "Pred leave"], yticklabels=["True stay", "True leave"])
        ax.set_title(f"{name}\nmissed leavers (FN): {cm[1, 0]} of {cm[1].sum()}")
    fig.tight_layout()
    fig2, ax = plt.subplots(figsize=(6, 5))
    for name, (fpr, tpr, _) in rocs.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC {table.loc[name, 'ROC-AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves on the test set"); ax.legend(); fig2.tight_layout()
    return {"table": table, "best": best, "model": fitted[best], "cms": cms,
            "baseline_accuracy": float(1 - y_te.mean()), "figs": {"06_confusion_matrices": fig, "06b_roc_curves": fig2}}


# =============================================================================
# STEP 7  CLUSTERING  (engagement personas)
# =============================================================================
def name_persona(z):
    """Interpret a cluster from its standardized profile (assigned AFTER clustering)."""
    ten, hrs, perf, sat, trn, wlb = (z["Years_at_Company"], z["Monthly_Hours_Worked"], z["Performance_Rating"],
                                     z["Satisfaction_Score"], z["Training_Hours_per_year"], z["Work_Life_Balance_Score"])
    if hrs > 1.0 and sat < -0.5:                 return "Burned Out"
    if perf < -0.8 and sat < -0.5:               return "Quiet Quitter"
    if ten > 1.0:                                return "Steady Veteran"
    if ten < -0.7 and trn > 0.8:                 return "New & Growing"
    if sat > 0.4 and perf > 0.3:                 return "Highly Engaged"
    return "Steady Contributor"


def run_clustering(df, k_override=None, answer_key=None):
    df = df.copy()
    scaler = StandardScaler().fit(df[CLUSTER_FEATURES])
    X = scaler.transform(df[CLUSTER_FEATURES])

    ks = list(range(2, 9)); inertia, sil = [], []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=RANDOM_STATE).fit(X)
        inertia.append(km.inertia_); sil.append(silhouette_score(X, km.labels_))
    ksel = pd.DataFrame({"k": ks, "inertia": np.round(inertia, 1), "silhouette": np.round(sil, 3)}).set_index("k")

    if k_override:
        k = k_override
    else:
        # largest k (<= 6) whose silhouette is within 0.06 of the best: the elbow bends at 4-5 and
        # k=5 separates Burned Out from Quiet Quitter, which need different HR actions
        best_sil = max(sil)
        k = max(kk for kk, s in zip(ks, sil) if s >= best_sil - 0.06 and kk <= 6)

    kmeans = KMeans(k, n_init=20, random_state=RANDOM_STATE).fit(X)
    df["Cluster"] = kmeans.labels_
    profile = df.groupby("Cluster")[CLUSTER_FEATURES + ["Age", "Distance_From_Home_km"]].mean().round(1)
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
    if REG_TARGET in df and "Attrition_Flag" in df:
        check = df.groupby("Persona").agg(n=("Cluster", "size"), mean_salary_k=(REG_TARGET, "mean"),
                                          attrition_share=("Attrition_Flag", "mean")).round(2).sort_values("attrition_share", ascending=False)
    ari = agree = None
    if answer_key is not None and ID_COL in df:
        m = df[[ID_COL, "Persona"]].merge(answer_key, on=ID_COL)
        if len(m):
            ari = adjusted_rand_score(m[KEY_COL], m["Persona"])
            agree = float((m["Persona"] == m[KEY_COL]).mean())

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
    ax.set_title(f"K-Means engagement personas (k={k}) in PCA space"); ax.legend(fontsize=8); fig.tight_layout(); figs["08_pca_clusters"] = fig

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(profile_z.rename(index=names), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Persona profiles (standardized: + above average, - below)"); fig.tight_layout(); figs["09_persona_heatmap"] = fig

    return {"df": df, "k": k, "ksel": ksel, "profile": profile, "profile_z": profile_z, "names": names,
            "check": check, "ari": ari, "agree": agree, "scaler": scaler, "kmeans": kmeans, "figs": figs}


# =============================================================================
# STEP 8  RETENTION-RISK ENGINE
# =============================================================================
PERSONA_ACTION = {
    "Burned Out": {
        TIER_RED: "Immediate manager check-in on workload and work-life balance; redistribute tasks this sprint. High performer at high risk of leaving.",
        TIER_YELLOW: "Review hours and on-call load; agree a recovery plan and protected time off.",
        TIER_GREEN: "Watch hours trend monthly; recognise effort but cap sustained overtime."},
    "Quiet Quitter": {
        TIER_RED: "Frank 1:1 on role fit and motivation; set a 90-day re-engagement plan with clear goals, or plan a managed exit.",
        TIER_YELLOW: "Skip-level conversation; find a project that matches their interests; revisit in 60 days.",
        TIER_GREEN: "Low-cost engagement: mentoring pairing and a stretch assignment."},
    "Steady Veteran": {
        TIER_RED: "Retention conversation on career path and pay; losing institutional knowledge is expensive.",
        TIER_YELLOW: "Offer a new challenge (mentoring, cross-team lead) to prevent stagnation.",
        TIER_GREEN: "Recognise tenure; use as mentor and knowledge anchor."},
    "New & Growing": {
        TIER_RED: "Onboarding rescue: pair with a buddy, clarify expectations, check pay against market for the role.",
        TIER_YELLOW: "30/60/90-day check-ins; make sure training converts into real responsibilities.",
        TIER_GREEN: "Keep investing in training; define a visible growth path."},
    "Highly Engaged": {
        TIER_RED: "Unexpected for this persona: check for external offers or a specific blocker; fast-track retention counter-plan.",
        TIER_YELLOW: "Career conversation; make sure recognition and pay keep pace with contribution.",
        TIER_GREEN: "Candidate for leadership development, high-visibility projects and succession planning."},
}
DEFAULT_ACTION = {TIER_RED: "Urgent manager 1:1.", TIER_YELLOW: "Schedule a check-in.", TIER_GREEN: "Retain as is."}


def decide(r):
    p, leave = r["Attrition_Prob"], r["Attrition_Pred"] == "Yes"
    high_perf = r["Performance_Rating"] >= HIGH_PERFORMER
    reasons, red_flags = [], 0
    if leave:                                         reasons.append(f"attrition model: Yes (p={p:.0%})")
    if high_perf:                                     reasons.append(f"high performer (rating {r['Performance_Rating']:.1f})")
    if r["Monthly_Hours_Worked"] > HOURS_WARN:        reasons.append(f"{r['Monthly_Hours_Worked']:.0f} h/month (> {HOURS_WARN})"); red_flags += 1
    if r["Satisfaction_Score"] < SATISFACTION_WARN:   reasons.append(f"satisfaction {r['Satisfaction_Score']:.1f} < {SATISFACTION_WARN}"); red_flags += 1
    underpaid = False
    if pd.notna(r.get("Pay_Gap_Pct", np.nan)):
        if r["Pay_Gap_Pct"] <= UNDERPAID_PCT:
            underpaid = True; reasons.append(f"paid {abs(r['Pay_Gap_Pct']):.0%} below predicted fair salary")
        elif r["Pay_Gap_Pct"] >= 0.10:
            reasons.append(f"paid {r['Pay_Gap_Pct']:.0%} above fair salary")

    if leave and (high_perf or underpaid or red_flags >= 1):  tier = TIER_RED
    elif leave or red_flags >= 2 or underpaid:               tier = TIER_YELLOW
    else:                                                    tier = TIER_GREEN
    table = next((t for k_, t in PERSONA_ACTION.items() if str(r["Persona"]).startswith(k_)), DEFAULT_ACTION)
    return pd.Series({"Tier": tier, "Why": "; ".join(reasons) or "engaged, no risk signals", "Recommendation": table[tier]})


def combine_and_prioritise(df, regressor, classifier):
    """Priority (0-100) = P(leave) x value-at-risk, where value = performance (+ pay-gap boost if available)."""
    df = df.copy()
    X = df[FEATURES]
    df["Predicted_Fair_Salary_k"] = regressor.predict(X).round(1)
    df["Attrition_Prob"] = classifier.predict_proba(X)[:, 1].round(3)
    df["Attrition_Pred"] = np.where(df["Attrition_Prob"] >= 0.5, "Yes", "No")
    if ACTUAL_SALARY in df and df[ACTUAL_SALARY].notna().any():
        df["Pay_Gap_Pct"] = ((df[ACTUAL_SALARY] - df["Predicted_Fair_Salary_k"]) / df["Predicted_Fair_Salary_k"]).round(3)
    else:
        df["Pay_Gap_Pct"] = np.nan
    value = (df["Performance_Rating"] / 5).clip(0, 1)
    underpaid_boost = np.where(df["Pay_Gap_Pct"].fillna(0) <= UNDERPAID_PCT, 1.25, 1.0)
    df["Priority_Score"] = (100 * df["Attrition_Prob"] * value * underpaid_boost).clip(0, 100).round(1)
    df[["Tier", "Why", "Recommendation"]] = df.apply(decide, axis=1)
    return df


def tiers_by_persona_fig(df):
    ct = pd.crosstab(df["Persona"], df["Tier"]).reindex(columns=TIER_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ct.plot.bar(stacked=True, ax=ax, color=TIER_COLORS)
    ax.set_title("Retention tier by engagement persona"); ax.set_ylabel("employees"); ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); return fig


FINAL_COLS = [ID_COL, "Department", "Predicted_Fair_Salary_k", "Pay_Gap_Pct", "Attrition_Pred", "Attrition_Prob",
              "Persona", "Priority_Score", "Tier", "Why", "Recommendation"]


# =============================================================================
# FULL PIPELINE
# =============================================================================
def run_pipeline(raw, k_override=None, use_answer_key=True):
    df, clean_info = clean_data(raw)
    df = engineer_features(df)
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["Attrition_Flag"])
    reg = run_regression(train, test)
    clf = run_classification(train, test)
    clu = run_clustering(df, k_override, load_answer_key() if use_answer_key else None)
    final = combine_and_prioritise(clu["df"], reg["model"], clf["model"])
    return {"clean_info": clean_info, "df": final, "train_rows": len(train), "test_rows": len(test),
            "reg": reg, "clf": clf, "clu": clu}


def score_new_employees(new_raw, res):
    df = new_raw.copy()
    df.columns = [c.strip() for c in df.columns]
    if CLF_TARGET not in df: df[CLF_TARGET] = "No"
    if REG_TARGET not in df: df[REG_TARGET] = 0.0
    df, _ = clean_data(df.assign(**{REG_TARGET: pd.to_numeric(df[REG_TARGET], errors="coerce").fillna(0)}))
    df = engineer_features(df)
    clu = res["clu"]
    df["Cluster"] = clu["kmeans"].predict(clu["scaler"].transform(df[CLUSTER_FEATURES]))
    df["Persona"] = df["Cluster"].map(clu["names"])
    return combine_and_prioritise(df, res["reg"]["model"], res["clf"]["model"])


# =============================================================================
# MODE 1: COMMAND-LINE REPORT
# =============================================================================
def banner(t):
    print("\n" + "=" * 90 + f"\n  {t}\n" + "=" * 90)


def run_cli():
    ap = argparse.ArgumentParser(description="Employee attrition & workforce analytics")
    ap.add_argument("--data", default=None, help="CSV path (default: embedded employee_capstone.csv)")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--out", default="output")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.data) if args.data else load_embedded_data()
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 60)

    banner("STEP 1  LOAD & UNDERSTAND THE DATA")
    print(f"Source : {args.data or 'embedded employee_capstone.csv'}\nShape  : {raw.shape[0]} rows x {raw.shape[1]} columns\n")
    print(raw.head(), "\n"); print("Missing values:\n", raw.isnull().sum(), "\n"); print("Duplicate rows:", raw.duplicated().sum())
    print("\nSummary statistics:\n", raw.describe().T.round(2))
    print("\nDepartments:", raw["Department"].value_counts().to_dict())
    share = (raw[CLF_TARGET].str.strip().str.title() == "Yes").mean()
    print(f"Attrition balance: {share:.1%} Yes / {1 - share:.1%} No")

    res = run_pipeline(raw, args.k)
    df, ci = res["df"], res["clean_info"]
    banner("STEP 2  DATA CLEANING")
    print(f"Rows {ci['rows_before']} -> {ci['rows_after']} | duplicates removed: {ci['duplicates_removed']} | "
          f"out-of-range nulled: {ci['out_of_range_nulled']} | imputed: {ci['values_imputed']}")
    banner("STEP 3  EXPLORATORY DATA ANALYSIS")
    corr = df[NUMERIC_RAW + [REG_TARGET]].corr()[REG_TARGET].drop(REG_TARGET)
    print("Correlation with salary:\n" + corr.sort_values(ascending=False).round(2).to_string())
    print("\nMean salary by department ($k):\n" + df.groupby("Department")[REG_TARGET].mean().round(1).sort_values(ascending=False).to_string())
    print("\nMean of each feature by Attrition:\n" + df.groupby(CLF_TARGET)[NUMERIC_RAW].mean().round(2).T.to_string())
    banner("STEP 4  FEATURE ENGINEERING + SPLIT")
    print("Engineered:", ENGINEERED, "| Department one-hot encoded inside the pipeline")
    print(f"Train {res['train_rows']} / Test {res['test_rows']} (stratified on attrition)")
    banner("STEP 5  REGRESSION  ->  Expected_Annual_Salary_k (fair salary)")
    print(res["reg"]["table"].to_string()); print(f"\n>> Selected: {res['reg']['best']}")
    print("\nLinear Regression coefficients ($k):\n" + res["reg"]["coef"].sort_values(ascending=False).round(1).to_string())
    banner("STEP 6  CLASSIFICATION  ->  Attrition")
    print(f"Baseline 'nobody leaves' accuracy = {res['clf']['baseline_accuracy']:.1%} with recall 0%\n")
    print(res["clf"]["table"].to_string())
    print("\nConfusion matrices [[TN FP] [FN TP]]  (FN = leavers we MISSED):")
    for n, cm in res["clf"]["cms"].items(): print(f"  {n:20s} {cm.tolist()}   missed {cm[1, 0]} of {cm[1].sum()}")
    print(f"\n>> Selected (recall-weighted F2): {res['clf']['best']}")
    clu = res["clu"]
    banner("STEP 7  CLUSTERING  ->  engagement personas")
    print("k selection:\n" + clu["ksel"].T.to_string()); print(f"\n>> Chosen k = {clu['k']}")
    print("\nPersona profiles:\n" + clu["profile"].to_string()); print("\nSanity check:\n" + clu["check"].to_string())
    if clu["ari"] is not None: print(f"\nCheck vs answer key: ARI = {clu['ari']:.3f}, name agreement = {clu['agree']:.1%}")
    banner("STEP 8  RETENTION-RISK ENGINE")
    if df["Pay_Gap_Pct"].isna().all():
        print("No Actual_Salary_k column in this file, so the underpaid check is skipped. Add that column to enable it.")
    print("Tier counts:\n" + df["Tier"].value_counts().to_string())
    print("\nTiers by persona:\n" + pd.crosstab(df["Persona"], df["Tier"]).to_string())
    banner("STEP 9  HR ACTION LIST (ranked)")
    ranked = df.sort_values("Priority_Score", ascending=False)[FINAL_COLS]
    ranked.to_csv(out / "hr_action_list.csv", index=False, encoding="utf-8-sig")
    print(ranked[[ID_COL, "Department", "Predicted_Fair_Salary_k", "Attrition_Pred", "Persona", "Priority_Score", "Tier"]].head(15).to_string(index=False))
    ex = df[df[ID_COL] == 390]; r = (ex if len(ex) else ranked).iloc[0]
    print(f"\nExample conclusion:\n  Employee #{r[ID_COL]} -> Predicted fair salary: ${r.Predicted_Fair_Salary_k:.0f}k · Attrition: {r.Attrition_Pred} · "
          f"Persona: {r.Persona}\n  Tier: {r.Tier} (priority {r.Priority_Score})\n  Why: {r.Why}\n  Recommendation: {r.Recommendation}")
    if not args.no_plots:
        figs = {**eda_figures(df), **res["reg"]["figs"], **res["clf"]["figs"], **clu["figs"], "10_tiers_by_persona": tiers_by_persona_fig(df)}
        for n, f in figs.items(): f.savefig(out / f"{n}.png", dpi=110); plt.close(f)
        print(f"\n{len(figs)} charts saved to {out}/")
    banner("SUMMARY")
    print(f"Regression: {res['reg']['best']} | Classification: {res['clf']['best']} | Clustering: K-Means k={clu['k']} -> "
          f"{sorted(df['Persona'].unique())}\nRetention engine: P(leave) x performance (x pay-gap) -> ranked HR action list ({out / 'hr_action_list.csv'})")


# =============================================================================
# MODE 2: STREAMLIT DASHBOARD
# =============================================================================
def run_dashboard():
    import streamlit as st
    st.set_page_config(page_title="Workforce Analytics", page_icon="🧑‍💼", layout="wide")

    @st.cache_resource(show_spinner="Training salary, attrition and persona models...")
    def train_all(k_override):
        return run_pipeline(load_embedded_data(), k_override)

    with st.sidebar:
        st.title("🧑‍💼 Controls")
        k_choice = st.selectbox("Number of personas (k)", ["auto (elbow + silhouette)", 3, 4, 5, 6], index=0)
        k_override = None if isinstance(k_choice, str) else int(k_choice)
        st.divider()
        st.markdown("**Score your own employees**")
        uploaded = st.file_uploader("Upload CSV", type=["csv"], help="Needs: " + ", ".join([ID_COL] + RAW_FEATURES))
        st.caption(f"Targets are optional. Add an `{ACTUAL_SALARY}` column to enable the underpaid check.")

    res = train_all(k_override)
    df, reg, clf, clu = res["df"], res["reg"], res["clf"], res["clu"]

    st.title("🧑‍💼 Employee Attrition & Workforce Analytics")
    st.caption("Regression (fair salary) + Classification (attrition risk) + Clustering (engagement persona) "
               "→ one retention-risk priority and HR action per employee.")
    tabs = st.tabs(["📊 Overview", "🔍 Data & EDA", "💵 Salary Regression", "🚪 Attrition Classification",
                    "🧩 Personas", "📋 HR Action List", "👤 Employee Detail", "ℹ️ About"])

    with tabs[0]:
        counts = df["Tier"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔴 Urgent 1:1", int(counts.get(TIER_RED, 0)))
        c2.metric("🟡 Check-in", int(counts.get(TIER_YELLOW, 0)))
        c3.metric("🟢 Develop & retain", int(counts.get(TIER_GREEN, 0)))
        c4.metric("Predicted leavers", int((df["Attrition_Pred"] == "Yes").sum()))
        m1, m2, m3 = st.columns(3)
        m1.metric("Regression", reg["best"], f"MAE ${reg['table'].loc[reg['best'], 'Test MAE']:.1f}k · R² {reg['table'].loc[reg['best'], 'Test R2']:.2f}")
        m2.metric("Classification", clf["best"], f"recall {clf['table'].loc[clf['best'], 'Recall']:.0%} · AUC {clf['table'].loc[clf['best'], 'ROC-AUC']:.2f}")
        m3.metric("Clustering", f"K-Means, k = {clu['k']}", f"silhouette {clu['ksel'].loc[clu['k'], 'silhouette']:.2f}")
        l, r_ = st.columns(2)
        with l:
            st.markdown("**Retention tier by persona**")
            st.bar_chart(pd.crosstab(df["Persona"], df["Tier"]).reindex(columns=TIER_ORDER, fill_value=0), color=TIER_COLORS)
        with r_:
            st.markdown("**Attrition rate by persona**")
            st.bar_chart((df.groupby("Persona")["Attrition_Prob"].mean() * 100).round(1))
        st.code("employee data → clean → features (+ one-hot Department) → [salary regression | attrition classifier | K-Means personas] "
                "→ priority = P(leave) × performance (× pay gap) → HR action list", language="text")

    with tabs[1]:
        ci = res["clean_info"]
        st.markdown(f"**Cleaning:** {ci['rows_before']} rows in, {ci['rows_after']} out · duplicates removed {ci['duplicates_removed']} · "
                    f"out-of-range nulled {ci['out_of_range_nulled']} · imputed {ci['values_imputed']}")
        st.dataframe(df[[ID_COL] + RAW_FEATURES + [REG_TARGET, CLF_TARGET]].head(20), hide_index=True, use_container_width=True)
        st.dataframe(df[NUMERIC_RAW + [REG_TARGET]].describe().T.round(2), use_container_width=True)
        st.markdown("**Mean salary by department ($k)**")
        st.dataframe(df.groupby("Department")[REG_TARGET].agg(["mean", "count"]).round(1).sort_values("mean", ascending=False), use_container_width=True)
        share = df["Attrition_Flag"].mean()
        st.info(f"Attrition: {share:.1%} Yes. Missing a leaver (false negative) means a resignation nobody saw coming, so recall and ROC-AUC matter more than accuracy.")
        for fig in eda_figures(df).values(): st.pyplot(fig); plt.close(fig)

    with tabs[2]:
        st.markdown("Target: `Expected_Annual_Salary_k`. `Department` is one-hot encoded inside the pipeline; numeric features are scaled. "
                    "Compared on 5-fold cross-validated MAE.")
        st.dataframe(reg["table"], use_container_width=True)
        st.success(f"Selected: **{reg['best']}**. Department and tenure carry most of the salary signal.")
        st.markdown(f"**Underpaid check:** if an `{ACTUAL_SALARY}` column is present, employees paid 10%+ below their predicted fair salary are flagged. "
                    "The capstone file has no actual-salary column, so upload one to see it in action.")
        for fig in reg["figs"].values(): st.pyplot(fig); plt.close(fig)

    with tabs[3]:
        st.markdown("Target: `Attrition`. Class-weighted models; selection by cross-validated **F2** (recall weighted 2×). Evaluated with accuracy, precision, recall and ROC-AUC.")
        st.dataframe(clf["table"], use_container_width=True)
        st.success(f"Selected: **{clf['best']}**. Baseline 'nobody leaves' accuracy would be {clf['baseline_accuracy']:.1%} with 0% recall.")
        for fig in clf["figs"].values(): st.pyplot(fig); plt.close(fig)

    with tabs[4]:
        st.markdown("Clustered on: " + ", ".join(f"`{c}`" for c in CLUSTER_FEATURES) + ". **No target used.** Names assigned after clustering.")
        st.dataframe(clu["ksel"].T, use_container_width=True)
        st.success(f"Chosen k = **{clu['k']}**. k=4 has a slightly higher silhouette but merges Burned Out with Quiet Quitter, which need opposite HR responses.")
        st.dataframe(clu["profile"], use_container_width=True)
        st.markdown("**Sanity check after clustering**"); st.dataframe(clu["check"], use_container_width=True)
        if clu["ari"] is not None:
            st.info(f"Check against the instructor answer key (never used in training): ARI = {clu['ari']:.3f}, name agreement = {clu['agree']:.1%}")
        for fig in clu["figs"].values(): st.pyplot(fig); plt.close(fig)

    with tabs[5]:
        st.markdown("**Priority** = P(leave) × performance/5 (×1.25 if underpaid). **Tier:** leave & (high performer or underpaid or red flag) → 🔴 · "
                    "leave, or 2 red flags, or underpaid → 🟡 · else 🟢. Red flags: >190 h/month, satisfaction < 4.5. Action depends on persona.")
        source_df = df
        if uploaded is not None:
            try:
                new_raw = pd.read_csv(uploaded); problems = validate(new_raw)
                if problems: st.error("; ".join(problems))
                else:
                    source_df = score_new_employees(new_raw, res); st.success(f"Scored {len(source_df)} uploaded employees.")
            except Exception as e:
                st.error(f"Could not read the upload: {e}")
        f1, f2_, f3 = st.columns(3)
        tier_sel = f1.multiselect("Tier", TIER_ORDER, default=TIER_ORDER)
        pers = sorted(source_df["Persona"].unique()); per_sel = f2_.multiselect("Persona", pers, default=pers)
        deps = sorted(source_df["Department"].unique()); dep_sel = f3.multiselect("Department", deps, default=deps)
        view = source_df[source_df["Tier"].isin(tier_sel) & source_df["Persona"].isin(per_sel) & source_df["Department"].isin(dep_sel)]
        view = view.sort_values("Priority_Score", ascending=False)[FINAL_COLS]
        st.dataframe(view, hide_index=True, use_container_width=True, height=450,
                     column_config={"Predicted_Fair_Salary_k": st.column_config.NumberColumn("Fair salary ($k)", format="%.1f"),
                                    "Pay_Gap_Pct": st.column_config.NumberColumn("Pay gap", format="%.0%"),
                                    "Attrition_Prob": st.column_config.NumberColumn("P(leave)", format="%.2f"),
                                    "Priority_Score": st.column_config.ProgressColumn("Priority", min_value=0, max_value=100, format="%.0f"),
                                    "Why": st.column_config.TextColumn("Why", width="large"),
                                    "Recommendation": st.column_config.TextColumn("HR action", width="large")})
        st.download_button("Download HR action list (CSV)", view.to_csv(index=False).encode("utf-8-sig"), "hr_action_list.csv", "text/csv")
        st.markdown("**Playbook by persona**")
        st.table(pd.DataFrame({"Persona": list(PERSONA_ACTION), "🔴 Urgent": [v[TIER_RED] for v in PERSONA_ACTION.values()],
                               "🟡 Check-in": [v[TIER_YELLOW] for v in PERSONA_ACTION.values()],
                               "🟢 Develop": [v[TIER_GREEN] for v in PERSONA_ACTION.values()]}).set_index("Persona"))
        st.session_state["view_ids"] = view[ID_COL].tolist(); st.session_state["source_df"] = source_df

    with tabs[6]:
        src = st.session_state.get("source_df", df)
        ids = st.session_state.get("view_ids", src[ID_COL].tolist()) or src[ID_COL].tolist()
        eid = st.selectbox("EmployeeID", ids, index=ids.index(390) if 390 in ids else 0)
        r = src[src[ID_COL] == eid].iloc[0]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Predicted fair salary", f"${r.Predicted_Fair_Salary_k:.0f}k",
                  None if pd.isna(r.Pay_Gap_Pct) else f"actual is {r.Pay_Gap_Pct:+.0%} vs fair")
        d2.metric("Attrition", r.Attrition_Pred, f"p = {r.Attrition_Prob:.0%}")
        d3.metric("Persona", r.Persona)
        d4.metric("Tier", r.Tier, f"priority {r.Priority_Score}")
        st.markdown(f"**Department:** {r.Department}"); st.markdown(f"**Why:** {r.Why}"); st.markdown(f"**Recommendation:** {r.Recommendation}")
        st.dataframe(src[src[ID_COL] == eid][[ID_COL] + RAW_FEATURES], hide_index=True, use_container_width=True)
        comp = pd.DataFrame({"this employee": r[CLUSTER_FEATURES].astype(float),
                             f"persona mean ({r.Persona})": src[src["Persona"] == r.Persona][CLUSTER_FEATURES].mean(),
                             "all employees": src[CLUSTER_FEATURES].mean()}).round(1)
        st.dataframe(comp, use_container_width=True)

    with tabs[7]:
        st.markdown(f"""
### Problem statement
An HR team wants one workforce dashboard that estimates a fair expected salary for each employee, flags who is at risk of resigning,
and groups employees into engagement personas, so leadership can act on retention with the right message for the right group.

| Question | Paradigm | Target |
|---|---|---|
| Fair salary? | **Regression** | `Expected_Annual_Salary_k` |
| Likely to leave? | **Classification** | `Attrition` (Yes/No) |
| What kind of employee? | **Clustering** | none, personas discovered |
| What should HR do? | **Retention engine** | Priority Score + Tier + Action |

### Design decisions
- `Department` is one-hot encoded inside the sklearn pipeline (as the brief requires); numeric features are scaled.
- Three independent models on the same inputs; outputs combine only in the retention engine.
- Attrition selection weights **recall**; ROC-AUC is reported and plotted as the brief asks.
- k chosen with elbow + silhouette (k = {clu['k']}); persona names come after clustering from the profile heat-map.
  Check vs the instructor key: ARI {clu['ari']:.2f}.
- **Priority = P(leave) × performance**, boosted when underpaid: a high performer about to leave ranks first,
  matching the brief's Employee #390 example. Underpaid detection needs an `{ACTUAL_SALARY}` column (not in the capstone file; upload one to use it).
- Actions differ by persona: Burned Out → workload review, Quiet Quitter → role-fit conversation, Highly Engaged → leadership development.

### Run locally
```
pip install streamlit pandas numpy scikit-learn matplotlib seaborn xgboost
streamlit run employee_ml_project.py
python employee_ml_project.py
```
""")


def _running_in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _running_in_streamlit():
    run_dashboard()
elif __name__ == "__main__":
    run_cli()
