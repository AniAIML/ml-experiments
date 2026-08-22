import os
import io
import pathlib
import requests
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from fpdf import FPDF

# ---------------------------------------------------
# CLASS LABELS
# ---------------------------------------------------
# 0 = Non-Diabetic
# 1 = Prediabetes Stage 1 (mild)
# 2 = Prediabetes Stage 2 (impaired glucose tolerance)
# 3 = Diabetic

CLASS_LABELS = {
    0: "Non-Diabetic",
    1: "Prediabetes Stage 1",
    2: "Prediabetes Stage 2",
    3: "Diabetic"
}

# ---------------------------------------------------
# GEMINI API HELPER (with automatic model fallback)
# ---------------------------------------------------

GEMINI_MODEL_PRIMARY = "gemini-3.7-flash"   # highest free-tier model as of Aug 2026
GEMINI_MODEL_FALLBACK = "gemini-2.5-flash"  # stable fallback if primary is unavailable
GEMINI_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def get_gemini_api_key():
    """
    Resolve the Gemini API key with production-safe priority, returning (key, source).
    source is one of: "secrets", "secrets-nested", "env", "" (not found).

    1. st.secrets["GEMINI_API_KEY"] - top-level key in secrets.toml / Streamlit Cloud Secrets.
    2. One level of nesting, e.g. secrets.toml written as:
           [gemini]
           GEMINI_API_KEY = "..."
       (covers the common mistake of putting the key under a section header).
    3. Environment variable GEMINI_API_KEY.
    4. Not found -> caller falls back to a manual sidebar box (local testing only).
    """
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"], "secrets"
        for _, value in st.secrets.items():
            if isinstance(value, dict) and "GEMINI_API_KEY" in value:
                return value["GEMINI_API_KEY"], "secrets-nested"
    except Exception:
        # st.secrets raises if no secrets.toml/Secrets panel is configured at all
        pass

    env_key = os.environ.get("GEMINI_API_KEY", "")
    if env_key:
        return env_key, "env"
    return "", ""


def _secrets_diagnostic():
    """Returns a human-readable summary of what st.secrets currently sees, with no key values exposed."""
    try:
        keys = list(st.secrets.keys())
        if not keys:
            return "Streamlit secrets are configured but currently empty."
        return f"Top-level secret keys found: {', '.join(keys)}"
    except Exception as e:
        return f"No secrets.toml / Secrets panel detected at all ({e})."


def _call_gemini(api_key: str, prompt: str, model: str) -> str:
    url = GEMINI_URL_TEMPLATE.format(model=model)
    response = requests.post(
        url,
        params={"key": api_key},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def get_gemini_recommendation(api_key: str, stage_label: str, patient: dict):
    """
    Tries GEMINI_MODEL_PRIMARY first; if it errors (404 / unavailable / rate-limited),
    automatically retries with GEMINI_MODEL_FALLBACK.
    Returns (advice_text, model_used_or_None).
    """
    prompt = f"""
You are a medical wellness assistant (not a replacement for a doctor).
A screening ML model has classified a patient's diabetes risk stage as: "{stage_label}".

Patient details:
- Age: {patient['age']}
- BMI: {patient['bmi']}
- Glucose level: {patient['glu']}
- Blood Pressure: {patient['bp']}
- Pregnancies: {patient['preg']}

Based on this stage, provide a concise, structured response with:
1. **Remediation steps** - 3-4 practical lifestyle/medical actions appropriate for this stage.
2. **Diet suggestions** - a short list of foods to favor and foods to limit.
3. **When to see a doctor** - be direct: for "Diabetic" or "Prediabetes Stage 2", clearly
   recommend prompt medical consultation; for "Prediabetes Stage 1", recommend monitoring
   and a check-up; for "Non-Diabetic", note this is general wellness advice only.

Keep it under 200 words, use markdown bullet points, and include a brief disclaimer that
this is general guidance, not a medical diagnosis.
"""

    try:
        text = _call_gemini(api_key, prompt, GEMINI_MODEL_PRIMARY)
        return text, GEMINI_MODEL_PRIMARY
    except Exception as primary_error:
        try:
            text = _call_gemini(api_key, prompt, GEMINI_MODEL_FALLBACK)
            return text, GEMINI_MODEL_FALLBACK
        except Exception as fallback_error:
            return (
                f"⚠️ Could not fetch AI recommendation.\n\n"
                f"- Primary model ({GEMINI_MODEL_PRIMARY}) error: {primary_error}\n"
                f"- Fallback model ({GEMINI_MODEL_FALLBACK}) error: {fallback_error}",
                None,
            )


# ---------------------------------------------------
# PDF REPORT GENERATOR
# ---------------------------------------------------

def build_pdf_report(name, patient, predicted_label, confidence, advice_text, model_used):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Diabetes Risk Assessment Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated for: {name or 'N/A'}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Patient Details", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for label, val in [
        ("Pregnancies", patient["preg"]), ("Glucose", patient["glu"]),
        ("Blood Pressure", patient["bp"]), ("BMI", patient["bmi"]),
        ("Age", patient["age"]),
    ]:
        pdf.cell(0, 7, f"{label}: {val}", ln=True)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Prediction Result", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, f"Predicted Stage: {predicted_label}", ln=True)
    pdf.cell(0, 7, f"Model Confidence: {confidence:.1f}%", ln=True)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "AI-Generated Recommendations", ln=True)
    pdf.set_font("Helvetica", "", 10)
    clean_advice = advice_text.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 6, clean_advice)
    if model_used:
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 6, f"(Recommendations generated by {model_used})", ln=True)

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 5, "Disclaimer: This report provides general wellness guidance only "
                         "and is not a medical diagnosis. Always consult a licensed physician.")

    return bytes(pdf.output(dest="S"))


# ---------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------

st.set_page_config(page_title="Diabetes Predictor System", page_icon="🩺", layout="wide")

st.title("🩺 Diabetes Predictor System")
st.write("AI Based Diabetes Prediction using Machine Learning")
st.caption("Predicts 4 stages: Non-Diabetic, Prediabetes Stage 1, Prediabetes Stage 2, Diabetic")

# ---------------------------------------------------
# SESSION STATE (prediction history)
# ---------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------
# GEMINI API KEY INPUT
# ---------------------------------------------------

with st.sidebar:
    st.subheader("⚙️ Gemini API Settings")
    resolved_key, key_source = get_gemini_api_key()

    if resolved_key:
        gemini_api_key = resolved_key
        label = {"secrets": "Streamlit secrets", "secrets-nested": "Streamlit secrets (nested section)",
                  "env": "environment variable"}[key_source]
        st.success(f"🔒 API key loaded from {label}.")
        if key_source == "secrets-nested":
            st.caption(
                "Note: your key is under a section header in secrets.toml. It works, but "
                "for clarity consider moving it to the top level as `GEMINI_API_KEY = \"...\"`."
            )
    else:
        st.warning("No API key found in secrets or environment.")
        with st.expander("🔧 Why isn't my secret being found?"):
            st.code(_secrets_diagnostic(), language="text")
            st.markdown(
                "**Checklist:**\n"
                "- In Streamlit Cloud: Manage app → Settings → Secrets, add exactly:\n"
                "  ```toml\n  GEMINI_API_KEY = \"your-key-here\"\n  ```\n"
                "- Click **Save**, then **reboot the app** — secrets don't apply until it restarts.\n"
                "- Key name must match exactly (case-sensitive): `GEMINI_API_KEY`.\n"
                "- Locally: create `.streamlit/secrets.toml` with the same line, in the "
                "same folder as this script (and add it to `.gitignore`)."
            )
        gemini_api_key = st.text_input(
            "Gemini API Key (manual entry - local testing only)",
            type="password",
            help="For production, use Streamlit Cloud's Secrets panel instead."
        )

    st.caption(f"Primary model: `{GEMINI_MODEL_PRIMARY}`  \nFallback model: `{GEMINI_MODEL_FALLBACK}`")
    st.caption(
        "Never commit real keys into code - use Streamlit secrets or an environment variable."
    )

    st.divider()
    st.subheader("🕘 Session History")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download History (CSV)",
            data=csv_bytes,
            file_name="prediction_history.csv",
            mime="text/csv",
        )
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No predictions yet this session.")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

CSV_PATH = pathlib.Path(__file__).parent / "diabetes_multiclass.csv"

if not CSV_PATH.exists():
    st.error(
        f"⚠️ Dataset file not found at `{CSV_PATH.name}`.\n\n"
        "Make sure `diabetes_multiclass.csv` is committed to the same GitHub repo/folder "
        "as this script, then redeploy. Streamlit Cloud only sees files that are actually "
        "pushed to your repository."
    )
    st.stop()

data = pd.read_csv(CSV_PATH)

# ---------------------------------------------------
# SPLIT DATA
# ---------------------------------------------------

x = data.drop(columns='Outcome')
y = data['Outcome']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------
# LOGISTIC REGRESSION
# ---------------------------------------------------

lg = LogisticRegression(max_iter=1000)
lg.fit(x_train, y_train)

l_pred = lg.predict(x_test)

asc1 = accuracy_score(y_test, l_pred)

# ---------------------------------------------------
# DECISION TREE
# ---------------------------------------------------

dt = tree.DecisionTreeClassifier()
dt.fit(x_train, y_train)

pred = dt.predict(x_test)

asc2 = accuracy_score(y_test, pred)

# ---------------------------------------------------
# RANDOM FOREST
# ---------------------------------------------------

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

rfc_pred = rfc.predict(x_test)

asc3 = accuracy_score(y_test, rfc_pred)

# ---------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------

st.subheader("📊 Model Accuracy Comparison")

fig, ax = plt.subplots(figsize=(4.5, 2.8), dpi=120)

models = ["Logistic\nRegression", "Decision\nTree", "Random\nForest"]
accuracies = [asc1*100, asc2*100, asc3*100]

ax.bar(models, accuracies, color=["#4C72B0", "#DD8452", "#55A868"])

ax.set_xlabel("Models", fontsize=9)
ax.set_ylabel("Accuracy %", fontsize=9)
ax.set_title("Accuracy Comparison (4-Class)", fontsize=10)
ax.tick_params(labelsize=8)
fig.tight_layout()

chart_col, _ = st.columns([1, 1])
with chart_col:
    st.pyplot(fig, use_container_width=False)

# ---------------------------------------------------
# CHOOSE BEST MODEL
# ---------------------------------------------------

d = {
    "Logistic Regression": [asc1, "LR", lg],
    "Decision Tree": [asc2, "DT", dt],
    "Random Forest": [asc3, "RF", rfc]
}

x1 = list(d.keys())
y1 = list(d.values())

high = 0

for i in range(len(y1)):
    if y1[i][0] > high:
        high = y1[i][0]
        model = x1[i]
        model1 = y1[i][1]
        best_model_obj = y1[i][2]

st.success(f"✅ Best Model Selected: {model} ({high*100:.1f}% accuracy)")

# ---------------------------------------------------
# ADVANCED: FEATURE IMPORTANCE
# ---------------------------------------------------

with st.expander("🔍 What drives this prediction? (Feature Importance)"):
    if hasattr(best_model_obj, "feature_importances_"):
        importances = best_model_obj.feature_importances_
    elif hasattr(best_model_obj, "coef_"):
        importances = np.mean(np.abs(best_model_obj.coef_), axis=0)
    else:
        importances = None

    if importances is not None:
        imp_df = pd.DataFrame({
            "Feature": x.columns,
            "Importance": importances
        }).sort_values("Importance", ascending=True)

        fig_imp, ax_imp = plt.subplots(figsize=(4.5, 3), dpi=120)
        ax_imp.barh(imp_df["Feature"], imp_df["Importance"], color="#8172B2")
        ax_imp.set_xlabel("Relative Importance", fontsize=9)
        ax_imp.set_title(f"Feature Importance - {model}", fontsize=10)
        ax_imp.tick_params(labelsize=8)
        fig_imp.tight_layout()

        imp_col, _ = st.columns([1, 1])
        with imp_col:
            st.pyplot(fig_imp, use_container_width=False)
        st.caption(
            "Shows which health parameters most influence the model's staging decision."
        )
    else:
        st.info("Feature importance not available for this model type.")

# ---------------------------------------------------
# USER INPUTS
# ---------------------------------------------------

st.subheader("🧾 Enter Patient Details")

name = st.text_input("Enter Name")

col1, col2 = st.columns(2)
with col1:
    preg = st.number_input("Pregnancies", min_value=0)
    glu = st.number_input("Glucose Level", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    skin = st.number_input("Skin Thickness", min_value=0)
with col2:
    ins = st.number_input("Insulin Level", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=1)

# ---------------------------------------------------
# ADVANCED: BMI QUICK CALCULATOR
# ---------------------------------------------------

with st.expander("🧮 Don't know your BMI? Calculate it here"):
    hcol1, hcol2 = st.columns(2)
    with hcol1:
        height_cm = st.number_input("Height (cm)", min_value=0.0, value=0.0)
    with hcol2:
        weight_kg = st.number_input("Weight (kg)", min_value=0.0, value=0.0)
    if height_cm > 0 and weight_kg > 0:
        calc_bmi = weight_kg / ((height_cm / 100) ** 2)
        st.info(f"Calculated BMI: **{calc_bmi:.1f}** — enter this value in the BMI field above.")

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if st.button("Predict Diabetes", type="primary"):

    input_data = pd.DataFrame(
        [[preg, glu, bp, skin, ins, bmi, dpf, age]],
        columns=x.columns
    )

    if model1 == "LR":
        result = lg.predict(input_data)
        proba = lg.predict_proba(input_data)[0]
    elif model1 == "DT":
        result = dt.predict(input_data)
        proba = dt.predict_proba(input_data)[0]
    else:
        result = rfc.predict(input_data)
        proba = rfc.predict_proba(input_data)[0]

    predicted_class = int(result[0])
    predicted_label = CLASS_LABELS[predicted_class]
    confidence = float(proba[predicted_class]) * 100

    st.subheader("Prediction Result")

    if predicted_class == 0:
        st.success(f"✅ {name}: {predicted_label}")
    elif predicted_class == 1:
        st.info(f"🟡 {name}: {predicted_label}")
    elif predicted_class == 2:
        st.warning(f"🟠 {name}: {predicted_label}")
    else:
        st.error(f"⚠️ {name}: {predicted_label}")

    # -------------------------------------------
    # ADVANCED: PREDICTION CONFIDENCE BREAKDOWN
    # (plain-language first, chart tucked away for anyone curious)
    # -------------------------------------------

    st.subheader("🤔 How sure is the AI about this?")

    proba_df = pd.DataFrame({
        "Stage": [CLASS_LABELS[i] for i in range(len(proba))],
        "Probability (%)": proba * 100
    }).sort_values("Probability (%)", ascending=False).reset_index(drop=True)

    top_label = proba_df.loc[0, "Stage"]
    top_pct = proba_df.loc[0, "Probability (%)"]
    second_label = proba_df.loc[1, "Stage"]
    second_pct = proba_df.loc[1, "Probability (%)"]
    gap = top_pct - second_pct

    if gap >= 30:
        verdict = "The AI is quite confident about this result."
        verdict_icon = "✅"
    elif gap >= 15:
        verdict = "The AI leans towards this result, but it isn't fully certain."
        verdict_icon = "🟡"
    else:
        verdict = "This is a close call — the AI sees two results as almost equally likely."
        verdict_icon = "🟠"

    st.markdown(
        f"### {verdict_icon} {verdict}\n\n"
        f"Out of everything it checked, the AI is most confident this is "
        f"**{top_label}** ({top_pct:.0f} out of 100 chance)."
    )

    if gap < 30:
        st.markdown(
            f"But it also thinks there's a real chance it could instead be "
            f"**{second_label}** ({second_pct:.0f} out of 100 chance). "
            f"When the AI is this unsure, it's worth double-checking with an actual "
            f"blood test rather than relying on the app alone."
        )

    with st.expander("📊 See the full breakdown (for the curious)"):
        st.caption(
            "The AI doesn't just pick one answer — it rates how likely each of the "
            "4 possible results is, and these always add up to 100%. The tallest bar "
            "is the one shown as the main result above."
        )
        fig_proba, ax_proba = plt.subplots(figsize=(4.5, 2.8), dpi=120)
        stage_colors = {
            "Non-Diabetic": "#55A868", "Prediabetes Stage 1": "#4C72B0",
            "Prediabetes Stage 2": "#DD8452", "Diabetic": "#C44E52"
        }
        bar_colors = [stage_colors[s] for s in proba_df["Stage"]]
        ax_proba.bar(proba_df["Stage"], proba_df["Probability (%)"], color=bar_colors)
        ax_proba.set_ylabel("Out of 100 chance", fontsize=9)
        ax_proba.set_title("How likely is each possible result?", fontsize=10)
        ax_proba.tick_params(labelsize=7)
        plt.setp(ax_proba.get_xticklabels(), rotation=15, ha="right")
        fig_proba.tight_layout()

        proba_col, _ = st.columns([1, 1])
        with proba_col:
            st.pyplot(fig_proba, use_container_width=False)

    # -----------------------------------------------
    # GEMINI-POWERED REMEDIATION SUGGESTIONS
    # -----------------------------------------------

    st.subheader("💡 Recommended Next Steps")

    advice = None
    model_used = None

    if not gemini_api_key:
        st.warning(
            "Enter a Gemini API key in the sidebar to get personalized diet and "
            "remediation suggestions here."
        )
    else:
        with st.spinner(f"Generating personalized recommendations (trying {GEMINI_MODEL_PRIMARY})..."):
            patient_info = {
                "age": age, "bmi": bmi, "glu": glu, "bp": bp, "preg": preg
            }
            advice, model_used = get_gemini_recommendation(gemini_api_key, predicted_label, patient_info)
        st.markdown(advice)
        if model_used:
            st.caption(f"Generated using `{model_used}`" +
                       (" (fallback model used)" if model_used == GEMINI_MODEL_FALLBACK else ""))

    st.caption(
        "⚠️ This tool provides general wellness guidance only and is not a medical "
        "diagnosis. Always consult a licensed physician for treatment decisions."
    )

    # -----------------------------------------------
    # ADVANCED: DOWNLOADABLE PDF REPORT
    # -----------------------------------------------

    if advice:
        patient_info = {"preg": preg, "glu": glu, "bp": bp, "bmi": bmi, "age": age}
        pdf_bytes = build_pdf_report(name, patient_info, predicted_label, confidence, advice, model_used)
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"diabetes_report_{name or 'patient'}.pdf",
            mime="application/pdf",
        )

    # -----------------------------------------------
    # ADVANCED: SESSION HISTORY LOGGING
    # -----------------------------------------------

    st.session_state.history.append({
        "Name": name,
        "Glucose": glu,
        "BMI": bmi,
        "Age": age,
        "Predicted Stage": predicted_label,
        "Confidence %": round(confidence, 1),
    })
