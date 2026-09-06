"""
=====================================================================================
  RETAIL CUSTOMER INTELLIGENCE PLATFORM
  Capstone 2 : CLV prediction, churn classification, customer segmentation   (single-file project)
=====================================================================================

Everything for the capstone lives in this ONE file, including the dataset (embedded,
compressed) and an interactive Streamlit dashboard.

    REGRESSION      -> predicts Predicted_CLV_12mo ($)   (Linear Regression / Random Forest / XGBoost)
    CLASSIFICATION  -> predicts Churn_Risk (Yes / No)    (Logistic Regression / Decision Tree / Random Forest)
    CLUSTERING      -> discovers behavioural segments    (K-Means, elbow + silhouette, PCA)
    PRIORITY ENGINE -> Priority Score + tier (🔴 Retain now / 🟡 Re-engage / 🟢 Nurture & reward)
                       and a segment-aware marketing action for every customer.

Two ways to run it
------------------
    streamlit run retail_ml_project.py        -> interactive dashboard (this is what you deploy)
    python retail_ml_project.py               -> full text report + charts written to ./output

Deploy on Streamlit Community Cloud
-----------------------------------
    1. Put this file and a requirements.txt in a GitHub repo.
    2. requirements.txt:  streamlit  pandas  numpy  scikit-learn  matplotlib  seaborn  xgboost
    3. share.streamlit.io -> New app -> pick the repo -> main file = retail_ml_project.py -> Deploy.
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
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

ID_COL = "CustomerID"
REG_TARGET = "Predicted_CLV_12mo"
CLF_TARGET = "Churn_Risk"

RAW_FEATURES = ["Age", "Annual_Income_k", "Avg_Monthly_Spend", "Purchase_Frequency_per_month",
                "Days_Since_Last_Purchase", "Total_Lifetime_Orders", "Discount_Usage_Percent",
                "Email_Engagement_Score"]
ENGINEERED = ["Spend_to_Income_Ratio", "Recency_Score", "Engagement_x_Frequency"]
FEATURES = RAW_FEATURES + ENGINEERED

# Clustering uses the behavioural features named in the brief (never the targets)
CLUSTER_FEATURES = ["Annual_Income_k", "Avg_Monthly_Spend", "Purchase_Frequency_per_month",
                    "Days_Since_Last_Purchase", "Discount_Usage_Percent"]

VALID_RANGES = {"Age": (16, 100), "Annual_Income_k": (0, 1000), "Avg_Monthly_Spend": (0, 10000),
                "Purchase_Frequency_per_month": (0, 100), "Days_Since_Last_Purchase": (0, 3650),
                "Total_Lifetime_Orders": (0, 100000), "Discount_Usage_Percent": (0, 100),
                "Email_Engagement_Score": (0, 100)}

RECENCY_WARN = 60          # days since last purchase that counts as "going quiet"
ENGAGEMENT_WARN = 30       # email engagement below this = disengaged

TIER_RED, TIER_YELLOW, TIER_GREEN = "🔴 Retain now", "🟡 Re-engage", "🟢 Nurture & reward"
TIER_ORDER = [TIER_RED, TIER_YELLOW, TIER_GREEN]
TIER_COLORS = ["#d62728", "#f2c230", "#2ca02c"]

sns.set_theme(style="whitegrid")

# =============================================================================
# EMBEDDED DATASET (retail_capstone.csv, gzip + base64)
# =============================================================================
_DATA_B64 = "H4sIAAAAAAAC/12dW68kSXKc3/lbCoW4Xx4JkgIIUBIhSgL01FjMtpYLcmepmVkB/Pdy+8yz+lAA2dtzTmVlZoRfzM3No//mL7/+9uc/ff/l7//29dd/+P76659//svv/vXb3//8U/zw27+8/vr//uHbf/7zz7/987/++7d/+rfvP//+9Y9/+eWnf/7dr9+//adfvv+fv3z/+ad///Zv33/59id96PW3v/v3X7/90x9//un7t3/43a+/fXs+/Prvf/4tvvYf/vi/v//2x/ji//rL77//8uvrb//4609//svPv337H7/+7g/fv/3j919++v7zb6+/+9Pv/viv3/7u5z/ED/8UP/j2Tz/9+Zfvr3/85fvv//jTb99//+1v/uF/fqvtT39+/c0//+WXn7/9tz/++i9/VV9tvfp6t9c97/2q77Ff+z3ip/Gjft7rNff7vGq5873u67/8+a/ix+U16/u+znrPV3v3+I6uT993ea0TPxv7XeOStt9965L+GuV1mr5/tfjy+O7xqj3++5y4sOpep8dt2ojf76lrxmu01+QOrcTv47r72jW+ePPJolvw69XH+7bX//r+61/pv+K6+PjUI5T3WK879Pz648Yzr3d/7ftuh8/H28VXzXjuql+W+P994hMn/sLzxe92/Ljw6a1Px2vpPWaL71/v2uIPLuJ7+msVXVX0NlWvcV6j6jXisUssZn+XeF99/dCTdP0xvMDxMpvVunqJOfRQ8SolromF55pY0FjmFn+bWsZ4iBO/j0tqeY2uBxi6MH7zrrE6nTXS9+g6vuF5lfj13K+ulYxnWFqpo+2OrdRa69VjyeIr4yOxH1wStx56lfhVKXr4943XmLrHeFdftGQ7tYUVnM6D6SteS18a2xZm9S5hHjW27cZdeeAwm1jJvnPBqn7ywkza0bvEDshiin624p319rpLX73qqXXNlIE1LRkryr7XrhsezK3KXsqPtw8jiWet+j2b/z6ynPjY0B279rZddmXH0rErdccv9fv7unr7+Eta1I636TK32bUWdcT/4Ck1vuDoV1vPFnv6LuMlJxla96uniRUIw6n1vRZX3Li5bq1FjN8UnkHX6r5Vy9UKP/eLtIKXVO1UGZj3OfrcksfqzVo8XPxeZrPiBQcejNPLkHuYRt5naD3iPkVvo7/5KY6c7/pm8SVdjy13kv9zW9nX0jo0dlSL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"

# Instructor answer key: NEVER used for training, only to score the clustering afterwards
_ANSWER_KEY_B64 = "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"
KEY_COL = "True_Behavior_Segment"


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
    imputed = int(df[RAW_FEATURES].isnull().sum().sum())
    for col in RAW_FEATURES:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    if CLF_TARGET in df:
        df[CLF_TARGET] = df[CLF_TARGET].astype(str).str.strip().str.title()
        df["Churn_Flag"] = (df[CLF_TARGET] == "Yes").astype(int)
    if REG_TARGET in df:
        df[REG_TARGET] = pd.to_numeric(df[REG_TARGET], errors="coerce")
        df = df.dropna(subset=[REG_TARGET])
    df = df.reset_index(drop=True)
    return df, {"rows_before": before, "rows_after": len(df), "duplicates_removed": before - len(df),
                "out_of_range_nulled": nulled, "values_imputed": imputed}


# =============================================================================
# STEP 3  EDA
# =============================================================================
def eda_figures(df):
    figs = {}
    num = RAW_FEATURES + [REG_TARGET]
    corr = df[num].corr()

    fig, axes = plt.subplots(3, 3, figsize=(18, 11))
    for ax, col in zip(axes.ravel(), num):
        sns.histplot(df[col], kde=True, ax=ax, bins=25); ax.set_title(col)
    fig.suptitle("Feature distributions"); fig.tight_layout(); figs["01_distributions"] = fig

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title("Correlation matrix"); fig.tight_layout(); figs["02_correlation"] = fig

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, col in zip(axes, ["Avg_Monthly_Spend", "Purchase_Frequency_per_month", "Days_Since_Last_Purchase", "Email_Engagement_Score"]):
        sns.scatterplot(data=df, x=col, y=REG_TARGET, hue=CLF_TARGET, alpha=.7, ax=ax,
                        palette={"No": "tab:blue", "Yes": "tab:red"}); ax.set_title(f"{col} vs CLV")
    fig.tight_layout(); figs["03_scatter_vs_clv"] = fig

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    df[CLF_TARGET].value_counts().plot.bar(ax=axes[0], color=["tab:blue", "tab:red"]); axes[0].set_title("Churn_Risk balance")
    sns.boxplot(data=df, x=CLF_TARGET, y="Days_Since_Last_Purchase", ax=axes[1], palette={"No": "tab:blue", "Yes": "tab:red"})
    axes[1].set_title("Recency by churn")
    sns.boxplot(data=df, x=CLF_TARGET, y="Email_Engagement_Score", ax=axes[2], palette={"No": "tab:blue", "Yes": "tab:red"})
    axes[2].set_title("Email engagement by churn"); fig.tight_layout(); figs["04_churn"] = fig
    return figs


# =============================================================================
# STEP 4  FEATURE ENGINEERING
# =============================================================================
def engineer_features(df):
    df = df.copy()
    df["Spend_to_Income_Ratio"] = (df["Avg_Monthly_Spend"] * 12) / (df["Annual_Income_k"] * 1000 + 1)
    df["Recency_Score"] = 1 / (1 + df["Days_Since_Last_Purchase"] / 30)          # 1 = bought this month, -> 0 as they go quiet
    df["Engagement_x_Frequency"] = df["Email_Engagement_Score"] * df["Purchase_Frequency_per_month"] / 100
    return df


# =============================================================================
# STEP 5  REGRESSION  (CLV)
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
        pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
        cv_mae = -cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="neg_mean_absolute_error").mean()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({"Model": name, "CV MAE": cv_mae, "Test MAE": mean_absolute_error(y_te, pred),
                     "Test RMSE": np.sqrt(mean_squared_error(y_te, pred)), "Test R2": r2_score(y_te, pred)})
        fitted[name] = pipe
    table = pd.DataFrame(rows).set_index("Model").round(3)
    best = table["CV MAE"].idxmin()
    coef = pd.Series(fitted["Linear Regression"].named_steps["model"].coef_, index=FEATURES).sort_values()

    fig, axes = plt.subplots(1, len(fitted), figsize=(6 * len(fitted), 5))
    for ax, (name, pipe) in zip(np.atleast_1d(axes), fitted.items()):
        p = pipe.predict(X_te); ax.scatter(y_te, p, alpha=.6)
        lim = [0, max(y_te.max(), p.max()) * 1.05]; ax.plot(lim, lim, "r--", lw=1)
        ax.set_xlabel("Actual CLV ($)"); ax.set_ylabel("Predicted CLV ($)")
        ax.set_title(f"{name}\nMAE=${mean_absolute_error(y_te, p):.0f}  R2={r2_score(y_te, p):.3f}")
    fig.tight_layout()
    fig2, ax = plt.subplots(figsize=(8, 4.5))
    coef.plot.barh(ax=ax, color=np.where(coef > 0, "tab:green", "tab:red"))
    ax.set_title("Linear Regression: standardized coefficients ($ of CLV per 1 SD)"); fig2.tight_layout()
    return {"table": table, "best": best, "model": fitted[best], "coef": coef,
            "figs": {"05_regression_actual_vs_pred": fig, "05b_regression_coefficients": fig2}}


# =============================================================================
# STEP 6  CLASSIFICATION  (churn)
# =============================================================================
def run_classification(train, test):
    X_tr, y_tr, X_te, y_te = train[FEATURES], train["Churn_Flag"], test[FEATURES], test["Churn_Flag"]
    models = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, class_weight="balanced",
                                                random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=3,
                                                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    f2 = make_scorer(fbeta_score, beta=2)
    rows, fitted, cms = [], {}, {}
    for name, model in models.items():
        pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
        cvres = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring={"recall": "recall", "f2": f2})
        pipe.fit(X_tr, y_tr)
        pred, proba = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]
        rows.append({"Model": name, "CV Recall": cvres["test_recall"].mean(), "CV F2": cvres["test_f2"].mean(),
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
                    xticklabels=["Pred stay", "Pred churn"], yticklabels=["True stay", "True churn"])
        ax.set_title(f"{name}\nmissed churners (FN): {cm[1, 0]} of {cm[1].sum()}")
    fig.tight_layout()
    return {"table": table, "best": best, "model": fitted[best], "cms": cms,
            "baseline_accuracy": float(1 - y_te.mean()), "figs": {"06_confusion_matrices": fig}}


# =============================================================================
# STEP 7  CLUSTERING  (segments)
# =============================================================================
def name_segment(z):
    """Interpret a cluster from its standardized profile (assigned AFTER clustering)."""
    inc, spend, freq, rec, disc = (z["Annual_Income_k"], z["Avg_Monthly_Spend"], z["Purchase_Frequency_per_month"],
                                   z["Days_Since_Last_Purchase"], z["Discount_Usage_Percent"])
    if disc > 0.8:                                       return "Bargain Hunter"
    if spend > 0.5 and freq > 0.5 and rec < -0.3:        return "Loyal High Spender"
    if spend > 0.5 and rec > 0.2:                        return "At-Risk High Value"
    if spend < -0.5 and rec > 0.5:                       return "New / Low Engagement"
    return "Steady Moderate"


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
        # largest k (<= 6) whose silhouette is within 0.03 of the best: distinct but still interpretable
        best_sil = max(sil)
        k = max(kk for kk, s in zip(ks, sil) if s >= best_sil - 0.03 and kk <= 6)

    kmeans = KMeans(k, n_init=20, random_state=RANDOM_STATE).fit(X)
    df["Cluster"] = kmeans.labels_
    profile = df.groupby("Cluster")[CLUSTER_FEATURES + ["Email_Engagement_Score", "Total_Lifetime_Orders"]].mean().round(1)
    profile_z = pd.DataFrame(X, columns=CLUSTER_FEATURES).assign(Cluster=df["Cluster"]).groupby("Cluster").mean()
    names = {c: name_segment(r) for c, r in profile_z.iterrows()}
    seen = {}
    for c, n in list(names.items()):
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1: names[c] = f"{n} ({seen[n]})"
    df["Segment"] = df["Cluster"].map(names)
    profile["n"] = df["Cluster"].value_counts().sort_index()
    profile["Segment"] = pd.Series(names)

    check = None
    if REG_TARGET in df and "Churn_Flag" in df:
        check = df.groupby("Segment").agg(n=("Cluster", "size"), mean_CLV=(REG_TARGET, "mean"),
                                          churn_share=("Churn_Flag", "mean")).round(2).sort_values("mean_CLV", ascending=False)
    ari = agree = None
    if answer_key is not None and ID_COL in df:
        m = df[[ID_COL, "Segment"]].merge(answer_key, on=ID_COL)
        if len(m):
            ari = adjusted_rand_score(m[KEY_COL], m["Segment"])
            agree = float((m["Segment"] == m[KEY_COL]).mean())

    pca = PCA(2, random_state=RANDOM_STATE).fit(X)
    figs = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(ks, inertia, "o-"); axes[0].set_title("Elbow method"); axes[0].set_xlabel("k"); axes[0].set_ylabel("inertia")
    axes[1].plot(ks, sil, "o-", color="tab:orange"); axes[1].set_title("Silhouette score"); axes[1].set_xlabel("k")
    for ax in axes: ax.axvline(k, color="grey", ls="--", lw=1)
    fig.tight_layout(); figs["07_elbow_silhouette"] = fig

    pcs, cent = pca.transform(X), pca.transform(kmeans.cluster_centers_)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=df["Segment"], alpha=.75, ax=ax)
    ax.scatter(cent[:, 0], cent[:, 1], marker="X", s=200, c="black", label="centroids")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)"); ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
    ax.set_title(f"K-Means customer segments (k={k}) in PCA space"); ax.legend(fontsize=8); fig.tight_layout(); figs["08_pca_clusters"] = fig

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(profile_z.rename(index=names), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Segment profiles (standardized: + above average, - below)"); fig.tight_layout(); figs["09_segment_heatmap"] = fig

    return {"df": df, "k": k, "ksel": ksel, "profile": profile, "profile_z": profile_z, "names": names,
            "check": check, "ari": ari, "agree": agree, "scaler": scaler, "kmeans": kmeans, "figs": figs}


# =============================================================================
# STEP 8  PRIORITY ENGINE
# =============================================================================
SEGMENT_ACTION = {
    "Loyal High Spender": {
        TIER_RED: "Personal outreach from account team plus a VIP win-back offer; losing this customer costs the most.",
        TIER_YELLOW: "Early-warning: send a personalised 'we miss you' with a curated bundle before the gap widens.",
        TIER_GREEN: "Reward, don't discount: loyalty tier upgrade, early access to launches, referral bonus."},
    "At-Risk High Value": {
        TIER_RED: "Highest-priority retention offer now: high future value about to be lost. Phone/email within 48h.",
        TIER_YELLOW: "Reactivation campaign with a time-limited incentive and a reminder of past favourites.",
        TIER_GREEN: "Keep warm with premium content; monitor recency monthly."},
    "Bargain Hunter": {
        TIER_RED: "Targeted coupon on categories they already buy; avoid blanket discounts that erode margin.",
        TIER_YELLOW: "Flash-sale and clearance alerts; cheap to retain, price is the lever.",
        TIER_GREEN: "Include in seasonal promo lists only; nudge toward own-brand value lines."},
    "New / Low Engagement": {
        TIER_RED: "Low value and leaving: one automated win-back email, then stop spending on them.",
        TIER_YELLOW: "Onboarding journey: 3-email welcome series, first-repeat-purchase incentive.",
        TIER_GREEN: "Standard newsletter; measure open rate before investing further."},
    "Steady Moderate": {
        TIER_RED: "Retention offer sized to their spend level; ask for feedback on why they've gone quiet.",
        TIER_YELLOW: "Cross-sell campaign based on purchase history to lift frequency.",
        TIER_GREEN: "Maintain cadence; test a loyalty-programme invite to grow into high spender."},
}
DEFAULT_ACTION = {TIER_RED: "Retention offer.", TIER_YELLOW: "Re-engagement campaign.", TIER_GREEN: "Standard nurture."}


def decide(r, clv_hi, clv_mid):
    clv, churn, p = r["Predicted_CLV"], r["Churn_Pred"] == "Yes", r["Churn_Prob"]
    reasons = []
    if churn:                   reasons.append(f"churn model: Yes (p={p:.0%})")
    if clv >= clv_hi:           reasons.append(f"high predicted CLV ${clv:,.0f} (top 40%)")
    elif clv < clv_mid:         reasons.append(f"low predicted CLV ${clv:,.0f}")
    quiet = r["Days_Since_Last_Purchase"] > RECENCY_WARN
    if quiet:                   reasons.append(f"{r['Days_Since_Last_Purchase']:.0f} days since last purchase")
    if r["Email_Engagement_Score"] < ENGAGEMENT_WARN: reasons.append(f"email engagement {r['Email_Engagement_Score']:.0f} < {ENGAGEMENT_WARN}")

    if churn and clv >= clv_mid:            tier = TIER_RED       # value at stake + likely to leave
    elif churn or quiet:                    tier = TIER_YELLOW    # leaving but low value, or going quiet
    else:                                   tier = TIER_GREEN
    table = next((t for s, t in SEGMENT_ACTION.items() if str(r["Segment"]).startswith(s)), DEFAULT_ACTION)
    return pd.Series({"Tier": tier, "Why": "; ".join(reasons) or "healthy, active customer",
                      "Recommendation": table[tier]})


def combine_and_prioritise(df, regressor, classifier, clv_ref=None):
    """Priority Score (0-100) = churn probability x relative CLV: high value about to be lost ranks first."""
    df = df.copy()
    X = df[FEATURES]
    df["Predicted_CLV"] = np.clip(regressor.predict(X).round(0), 0, None)
    df["Churn_Prob"] = classifier.predict_proba(X)[:, 1].round(3)
    df["Churn_Pred"] = np.where(df["Churn_Prob"] >= 0.5, "Yes", "No")
    ref = clv_ref if clv_ref is not None else df["Predicted_CLV"]
    clv_hi, clv_mid, clv_max = ref.quantile(0.6), ref.quantile(0.3), ref.max()
    df["Priority_Score"] = (100 * df["Churn_Prob"] * (df["Predicted_CLV"] / clv_max).clip(0, 1)).round(1)
    df[["Tier", "Why", "Recommendation"]] = df.apply(lambda r: decide(r, clv_hi, clv_mid), axis=1)
    return df


def tiers_by_segment_fig(df):
    ct = pd.crosstab(df["Segment"], df["Tier"]).reindex(columns=TIER_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ct.plot.bar(stacked=True, ax=ax, color=TIER_COLORS)
    ax.set_title("Priority tier by customer segment"); ax.set_ylabel("customers"); ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); return fig


FINAL_COLS = [ID_COL, "Predicted_CLV", "Churn_Pred", "Churn_Prob", "Segment", "Priority_Score", "Tier", "Why", "Recommendation"]


# =============================================================================
# FULL PIPELINE
# =============================================================================
def run_pipeline(raw, k_override=None, use_answer_key=True):
    df, clean_info = clean_data(raw)
    df = engineer_features(df)
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["Churn_Flag"])
    reg = run_regression(train, test)
    clf = run_classification(train, test)
    clu = run_clustering(df, k_override, load_answer_key() if use_answer_key else None)
    final = combine_and_prioritise(clu["df"], reg["model"], clf["model"])
    return {"clean_info": clean_info, "df": final, "train_rows": len(train), "test_rows": len(test),
            "reg": reg, "clf": clf, "clu": clu}


def score_new_customers(new_raw, res):
    df = new_raw.copy()
    df.columns = [c.strip() for c in df.columns]
    if CLF_TARGET not in df: df[CLF_TARGET] = "No"
    if REG_TARGET not in df: df[REG_TARGET] = 0.0
    df, _ = clean_data(df.assign(**{REG_TARGET: pd.to_numeric(df[REG_TARGET], errors="coerce").fillna(0)}))
    df = engineer_features(df)
    clu = res["clu"]
    df["Cluster"] = clu["kmeans"].predict(clu["scaler"].transform(df[CLUSTER_FEATURES]))
    df["Segment"] = df["Cluster"].map(clu["names"])
    return combine_and_prioritise(df, res["reg"]["model"], res["clf"]["model"], clv_ref=res["df"]["Predicted_CLV"])


# =============================================================================
# MODE 1: COMMAND-LINE REPORT
# =============================================================================
def banner(t):
    print("\n" + "=" * 90 + f"\n  {t}\n" + "=" * 90)


def run_cli():
    ap = argparse.ArgumentParser(description="Retail customer intelligence: CLV / churn / segments")
    ap.add_argument("--data", default=None, help="CSV path (default: embedded retail_capstone.csv)")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--out", default="output")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.data) if args.data else load_embedded_data()
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 60)

    banner("STEP 1  LOAD & UNDERSTAND THE DATA")
    print(f"Source : {args.data or 'embedded retail_capstone.csv'}\nShape  : {raw.shape[0]} rows x {raw.shape[1]} columns\n")
    print(raw.head(), "\n"); print("Missing values:\n", raw.isnull().sum(), "\n"); print("Duplicate rows:", raw.duplicated().sum())
    print("\nSummary statistics:\n", raw.describe().T.round(2))
    share = (raw[CLF_TARGET].str.strip().str.title() == "Yes").mean()
    print(f"\nChurn_Risk balance: {share:.1%} Yes / {1 - share:.1%} No")

    res = run_pipeline(raw, args.k)
    df, ci = res["df"], res["clean_info"]
    banner("STEP 2  DATA CLEANING")
    print(f"Rows {ci['rows_before']} -> {ci['rows_after']} | duplicates removed: {ci['duplicates_removed']} | "
          f"out-of-range nulled: {ci['out_of_range_nulled']} | imputed: {ci['values_imputed']}")
    banner("STEP 3  EXPLORATORY DATA ANALYSIS")
    corr = df[RAW_FEATURES + [REG_TARGET]].corr()[REG_TARGET].drop(REG_TARGET)
    print("Correlation with Predicted_CLV_12mo:\n" + corr.sort_values(ascending=False).round(2).to_string())
    print("\nMean of each feature by Churn_Risk:\n" + df.groupby(CLF_TARGET)[RAW_FEATURES].mean().round(2).T.to_string())
    banner("STEP 4  FEATURE ENGINEERING + SPLIT")
    print("Engineered:", ENGINEERED, f"| Train {res['train_rows']} / Test {res['test_rows']} (stratified on churn)")
    banner("STEP 5  REGRESSION  ->  Predicted_CLV_12mo")
    print(res["reg"]["table"].to_string()); print(f"\n>> Selected: {res['reg']['best']}")
    print("\nStandardized coefficients:\n" + res["reg"]["coef"].sort_values(ascending=False).round(1).to_string())
    banner("STEP 6  CLASSIFICATION  ->  Churn_Risk")
    print(f"Baseline 'nobody churns' accuracy = {res['clf']['baseline_accuracy']:.1%} with recall 0%\n")
    print(res["clf"]["table"].to_string())
    print("\nConfusion matrices [[TN FP] [FN TP]]  (FN = churners we MISSED, the costly error):")
    for n, cm in res["clf"]["cms"].items(): print(f"  {n:20s} {cm.tolist()}   missed {cm[1, 0]} of {cm[1].sum()}")
    print(f"\n>> Selected (recall-weighted F2): {res['clf']['best']}")
    clu = res["clu"]
    banner("STEP 7  CLUSTERING  ->  customer segments")
    print("k selection:\n" + clu["ksel"].T.to_string()); print(f"\n>> Chosen k = {clu['k']}")
    print("\nSegment profiles:\n" + clu["profile"].to_string()); print("\nSanity check:\n" + clu["check"].to_string())
    if clu["ari"] is not None: print(f"\nCheck vs answer key: ARI = {clu['ari']:.3f}, name agreement = {clu['agree']:.1%}")
    banner("STEP 8  PRIORITY ENGINE")
    print("Tier counts:\n" + df["Tier"].value_counts().to_string())
    print("\nTiers by segment:\n" + pd.crosstab(df["Segment"], df["Tier"]).to_string())
    banner("STEP 9  RANKED MARKETING ACTION LIST")
    ranked = df.sort_values("Priority_Score", ascending=False)[FINAL_COLS]
    ranked.to_csv(out / "ranked_marketing_action_list.csv", index=False, encoding="utf-8-sig")
    print(ranked[[ID_COL, "Predicted_CLV", "Churn_Pred", "Segment", "Priority_Score", "Tier"]].head(15).to_string(index=False))
    ex = df[df[ID_COL] == 58]; r = (ex if len(ex) else ranked).iloc[0]
    print(f"\nExample conclusion:\n  Customer #{r[ID_COL]} -> Predicted CLV: ${r.Predicted_CLV:,.0f} · Churn_Risk: {r.Churn_Pred} · "
          f"Segment: {r.Segment}\n  Tier: {r.Tier} (priority {r.Priority_Score})\n  Why: {r.Why}\n  Recommendation: {r.Recommendation}")
    print("\nCampaign plan by segment:")
    for seg, tbl in SEGMENT_ACTION.items():
        if seg in df["Segment"].values: print(f"  {seg:22s} -> {tbl[TIER_GREEN]}")
    if not args.no_plots:
        figs = {**eda_figures(df), **res["reg"]["figs"], **res["clf"]["figs"], **clu["figs"], "10_tiers_by_segment": tiers_by_segment_fig(df)}
        for n, f in figs.items(): f.savefig(out / f"{n}.png", dpi=110); plt.close(f)
        print(f"\n{len(figs)} charts saved to {out}/")
    banner("SUMMARY")
    print(f"Regression: {res['reg']['best']} | Classification: {res['clf']['best']} | Clustering: K-Means k={clu['k']} -> "
          f"{sorted(df['Segment'].unique())}\nPriority engine: churn probability x CLV -> ranked action list ({out / 'ranked_marketing_action_list.csv'})")


# =============================================================================
# MODE 2: STREAMLIT DASHBOARD
# =============================================================================
def run_dashboard():
    import streamlit as st
    st.set_page_config(page_title="Retail Customer Intelligence", page_icon="🛒", layout="wide")

    @st.cache_resource(show_spinner="Training CLV, churn and segmentation models...")
    def train_all(k_override):
        return run_pipeline(load_embedded_data(), k_override)

    with st.sidebar:
        st.title("🛒 Controls")
        k_choice = st.selectbox("Number of segments (k)", ["auto (elbow + silhouette)", 3, 4, 5, 6], index=0)
        k_override = None if isinstance(k_choice, str) else int(k_choice)
        st.divider()
        st.markdown("**Score your own customers**")
        uploaded = st.file_uploader("Upload CSV", type=["csv"], help="Needs: " + ", ".join([ID_COL] + RAW_FEATURES))
        st.caption("Targets are optional in an upload.")

    res = train_all(k_override)
    df, reg, clf, clu = res["df"], res["reg"], res["clf"], res["clu"]

    st.title("🛒 Retail Customer Intelligence Platform")
    st.caption("Regression (12-month CLV) + Classification (churn risk) + Clustering (behavioural segment) "
               "→ one priority score and marketing action per customer.")
    tabs = st.tabs(["📊 Overview", "🔍 Data & EDA", "💰 CLV Regression", "🚪 Churn Classification",
                    "🧩 Segments", "📋 Action List", "👤 Customer Detail", "ℹ️ About"])

    with tabs[0]:
        counts = df["Tier"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔴 Retain now", int(counts.get(TIER_RED, 0)))
        c2.metric("🟡 Re-engage", int(counts.get(TIER_YELLOW, 0)))
        c3.metric("🟢 Nurture & reward", int(counts.get(TIER_GREEN, 0)))
        c4.metric("CLV at risk (churn=Yes)", f"${df.loc[df.Churn_Pred == 'Yes', 'Predicted_CLV'].sum():,.0f}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Regression", reg["best"], f"MAE ${reg['table'].loc[reg['best'], 'Test MAE']:.0f} · R² {reg['table'].loc[reg['best'], 'Test R2']:.2f}")
        m2.metric("Classification", clf["best"], f"recall {clf['table'].loc[clf['best'], 'Recall']:.0%} · AUC {clf['table'].loc[clf['best'], 'ROC-AUC']:.2f}")
        m3.metric("Clustering", f"K-Means, k = {clu['k']}", f"silhouette {clu['ksel'].loc[clu['k'], 'silhouette']:.2f}")
        l, r_ = st.columns(2)
        with l:
            st.markdown("**Priority tier by segment**")
            st.bar_chart(pd.crosstab(df["Segment"], df["Tier"]).reindex(columns=TIER_ORDER, fill_value=0), color=TIER_COLORS)
        with r_:
            st.markdown("**Predicted CLV by segment (mean $)**")
            st.bar_chart(df.groupby("Segment")["Predicted_CLV"].mean().round(0))
        st.code("customer data → clean → features → [CLV regression | churn classifier | K-Means segments] "
                "→ priority = P(churn) × CLV → ranked action list", language="text")

    with tabs[1]:
        ci = res["clean_info"]
        st.markdown(f"**Cleaning:** {ci['rows_before']} rows in, {ci['rows_after']} out · duplicates removed {ci['duplicates_removed']} · "
                    f"out-of-range nulled {ci['out_of_range_nulled']} · imputed {ci['values_imputed']}")
        st.dataframe(df[[ID_COL] + RAW_FEATURES + [REG_TARGET, CLF_TARGET]].head(20), hide_index=True, use_container_width=True)
        st.dataframe(df[RAW_FEATURES + [REG_TARGET]].describe().T.round(2), use_container_width=True)
        share = df["Churn_Flag"].mean()
        st.info(f"Churn_Risk: {share:.1%} Yes. A missed churner (false negative) is a customer lost silently, while a false alarm "
                f"only costs one offer, so recall is weighted above precision.")
        for fig in eda_figures(df).values(): st.pyplot(fig); plt.close(fig)

    with tabs[2]:
        st.markdown("Target: `Predicted_CLV_12mo` ($). Compared on 5-fold cross-validated MAE; test set used once.")
        st.dataframe(reg["table"], use_container_width=True)
        st.success(f"Selected: **{reg['best']}**. CLV is almost linear in monthly spend (r ≈ 0.97), so the linear model is hard to beat.")
        for fig in reg["figs"].values(): st.pyplot(fig); plt.close(fig)

    with tabs[3]:
        st.markdown("Target: `Churn_Risk`. Class-weighted models, selected by cross-validated **F2** (recall weighted 2×).")
        st.dataframe(clf["table"], use_container_width=True)
        st.success(f"Selected: **{clf['best']}**. Baseline 'nobody churns' accuracy would be {clf['baseline_accuracy']:.1%} with 0% recall.")
        st.markdown("**Why false negatives are the costly error:** a missed churner is a high-CLV customer who leaves without any offer; "
                    "a false positive is one unnecessary discount. The confusion matrices below highlight the missed churners.")
        for fig in clf["figs"].values(): st.pyplot(fig); plt.close(fig)

    with tabs[4]:
        st.markdown("Clustered on: " + ", ".join(f"`{c}`" for c in CLUSTER_FEATURES) + ". **No target used.** Names assigned after clustering.")
        st.dataframe(clu["ksel"].T, use_container_width=True)
        st.success(f"Chosen k = **{clu['k']}**")
        st.dataframe(clu["profile"], use_container_width=True)
        st.markdown("**Sanity check after clustering**"); st.dataframe(clu["check"], use_container_width=True)
        if clu["ari"] is not None:
            st.info(f"Check against the instructor answer key (never used in training): ARI = {clu['ari']:.3f}, name agreement = {clu['agree']:.1%}")
        for fig in clu["figs"].values(): st.pyplot(fig); plt.close(fig)

    with tabs[5]:
        st.markdown("**Priority Score** = P(churn) × (CLV / max CLV) × 100. **Tier:** churn & CLV above 30th pct → 🔴 · "
                    "churn or >60 days quiet → 🟡 · else 🟢. Recommendation depends on the segment.")
        source_df = df
        if uploaded is not None:
            try:
                new_raw = pd.read_csv(uploaded); problems = validate(new_raw)
                if problems: st.error("; ".join(problems))
                else:
                    source_df = score_new_customers(new_raw, res); st.success(f"Scored {len(source_df)} uploaded customers.")
            except Exception as e:
                st.error(f"Could not read the upload: {e}")
        f1, f2_, f3 = st.columns(3)
        tier_sel = f1.multiselect("Tier", TIER_ORDER, default=TIER_ORDER)
        segs = sorted(source_df["Segment"].unique()); seg_sel = f2_.multiselect("Segment", segs, default=segs)
        churn_sel = f3.multiselect("Churn prediction", ["Yes", "No"], default=["Yes", "No"])
        view = source_df[source_df["Tier"].isin(tier_sel) & source_df["Segment"].isin(seg_sel) & source_df["Churn_Pred"].isin(churn_sel)]
        view = view.sort_values("Priority_Score", ascending=False)[FINAL_COLS]
        st.dataframe(view, hide_index=True, use_container_width=True, height=450,
                     column_config={"Predicted_CLV": st.column_config.NumberColumn("Predicted CLV", format="$%.0f"),
                                    "Churn_Prob": st.column_config.NumberColumn("P(churn)", format="%.2f"),
                                    "Priority_Score": st.column_config.ProgressColumn("Priority", min_value=0, max_value=100, format="%.0f"),
                                    "Why": st.column_config.TextColumn("Why", width="large"),
                                    "Recommendation": st.column_config.TextColumn("Marketing action", width="large")})
        st.download_button("Download ranked action list (CSV)", view.to_csv(index=False).encode("utf-8-sig"), "ranked_marketing_action_list.csv", "text/csv")
        st.markdown("**Campaign plan by segment**")
        st.table(pd.DataFrame({"Segment": list(SEGMENT_ACTION), "Retention (🔴)": [v[TIER_RED] for v in SEGMENT_ACTION.values()],
                               "Re-engage (🟡)": [v[TIER_YELLOW] for v in SEGMENT_ACTION.values()],
                               "Nurture (🟢)": [v[TIER_GREEN] for v in SEGMENT_ACTION.values()]}).set_index("Segment"))
        st.session_state["view_ids"] = view[ID_COL].tolist(); st.session_state["source_df"] = source_df

    with tabs[6]:
        src = st.session_state.get("source_df", df)
        ids = st.session_state.get("view_ids", src[ID_COL].tolist()) or src[ID_COL].tolist()
        cid = st.selectbox("CustomerID", ids, index=ids.index(58) if 58 in ids else 0)
        r = src[src[ID_COL] == cid].iloc[0]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Predicted CLV (12 mo)", f"${r.Predicted_CLV:,.0f}")
        d2.metric("Churn risk", r.Churn_Pred, f"p = {r.Churn_Prob:.0%}")
        d3.metric("Segment", r.Segment)
        d4.metric("Tier", r.Tier, f"priority {r.Priority_Score}")
        st.markdown(f"**Why:** {r.Why}"); st.markdown(f"**Recommendation:** {r.Recommendation}")
        st.dataframe(src[src[ID_COL] == cid][[ID_COL] + RAW_FEATURES], hide_index=True, use_container_width=True)
        comp = pd.DataFrame({"this customer": r[CLUSTER_FEATURES + ["Email_Engagement_Score"]].astype(float),
                             f"segment mean ({r.Segment})": src[src["Segment"] == r.Segment][CLUSTER_FEATURES + ["Email_Engagement_Score"]].mean(),
                             "all customers": src[CLUSTER_FEATURES + ["Email_Engagement_Score"]].mean()}).round(1)
        st.dataframe(comp, use_container_width=True)

    with tabs[7]:
        st.markdown(f"""
### Problem statement
An online retailer wants a single view of each customer: how much are they worth over the next year, are they about to churn,
and which behavioural segment do they belong to, so marketing knows exactly what to do for each one.

| Question | Paradigm | Target |
|---|---|---|
| Future value? | **Regression** | `Predicted_CLV_12mo` |
| About to churn? | **Classification** | `Churn_Risk` (Yes/No) |
| What kind of customer? | **Clustering** | none, segments discovered |
| What should marketing do? | **Priority engine** | Priority Score + Tier + Action |

### Design decisions
- Three independent models on the same inputs; outputs combine only in the priority engine.
- Selection by cross-validation; Linear Regression wins CLV because spend explains it almost linearly.
- Churn selection weights **recall**: false negatives (missed churners) are the expensive error.
- k chosen with elbow + silhouette (k = {clu['k']}); segments named after clustering from the profile heat-map.
- **Priority = P(churn) × CLV**: the top of the list is high value about to be lost, exactly as the brief's example (Customer #58).
- Actions differ by segment: a Bargain Hunter gets a targeted coupon, a Loyal High Spender gets recognition, not discounts.

### Run locally
```
pip install streamlit pandas numpy scikit-learn matplotlib seaborn xgboost
streamlit run retail_ml_project.py
python retail_ml_project.py
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
