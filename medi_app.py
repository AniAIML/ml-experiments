import os
import streamlit as st
import pandas as pd

# ----------------------------------------------------------------------------
# GEMINI (Google Gen AI) SDK — optional import so the app still runs (in
# "offline" mode, showing only your saved list) even if the package or the
# API key is missing.
# ----------------------------------------------------------------------------
try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_SDK_AVAILABLE = True
except ImportError:
    GENAI_SDK_AVAILABLE = False

# Tried in this order — the first one that actually responds on your key/region
# is used, and remembered for the rest of the session so we don't re-probe on
# every click. Newer, cheaper "Flash" models first, then the well-established
# 2.5 Flash as a safety-net fallback.
GEMINI_MODEL_CHAIN = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
]

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="আমার মেডিসিন গাইড | My Medicine Guide",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# DATA
# Add / edit entries here. Each entry: emoji, problem (English), problem
# (Bangla), medicine name(s). Add a "notes" field any time you want to jot
# down something extra about when/why you take it — totally optional.
# ----------------------------------------------------------------------------
ADULT_DATA = [
    {"emoji": "🤕", "problem_en": "Headache", "problem_bn": "মাথা ব্যথা", "medicine": "Greenil"},
    {"emoji": "🌡️", "problem_en": "Fever", "problem_bn": "জ্বর", "medicine": "Paracetamol 500 mg"},
    {"emoji": "🤕", "problem_en": "Abdominal pain", "problem_bn": "পেট ব্যথা", "medicine": "Drotin"},
    {"emoji": "🤢", "problem_en": "Vomiting", "problem_bn": "বমি", "medicine": "Zofer / Ondem"},
    {"emoji": "🌡️", "problem_en": "Fever + antibiotic", "problem_bn": "জ্বর + অ্যান্টিবায়োটিক", "medicine": "Furixim-CV 625"},
    {"emoji": "🍽️", "problem_en": "Digestion problem", "problem_bn": "হজমের সমস্যা", "medicine": "Unienzyme / Digene / Pan-D"},
    {"emoji": "💪", "problem_en": "Body, hand & leg pain", "problem_bn": "গা, হাত ও পা ব্যথা", "medicine": "Pyrigesic / Paracetamol"},
    {"emoji": "🩸", "problem_en": "Cholesterol", "problem_bn": "কোলেস্টেরল", "medicine": "Lpikicard 160"},
    {"emoji": "🌀", "problem_en": "Dizziness", "problem_bn": "মাথা ঘোরা", "medicine": "Stemetil MD"},
    {"emoji": "🤧", "problem_en": "Cold", "problem_bn": "সর্দি", "medicine": "Livcet 5"},
    {"emoji": "💩", "problem_en": "Loose motion / Diarrhea", "problem_bn": "পাতলা পায়খানা", "medicine": "Norflox-TZ RF / Ambzyme"},
    {"emoji": "💨", "problem_en": "Gas", "problem_bn": "গ্যাস", "medicine": "Cyra-D"},
    {"emoji": "🌡️", "problem_en": "Fever (higher strength)", "problem_bn": "জ্বর", "medicine": "Paracetamol 650 mg"},
]

CHILDREN_DATA = [
    # Add children's medicines here later, same format as ADULT_DATA, e.g.:
    # {"emoji": "🌡️", "problem_en": "Fever", "problem_bn": "জ্বর", "medicine": "Paracetamol Syrup"},
]

# ----------------------------------------------------------------------------
# UI TEXT — every label in English + বাংলা together, so it stays readable for
# anyone, regardless of which language they're more comfortable in.
# ----------------------------------------------------------------------------
TXT = {
    "app_title": "🩺 আমার মেডিসিন গাইড | My Medicine Guide",
    "app_caption": "নিজের পুরনো ব্যবহার করা ওষুধ সহজে খুঁজে নিন, আর AI থেকে সহজ ভাষায় আরও বিস্তারিত জানুন।"
                   " | Quick lookup for medicines you already use — plus simple AI-powered details.",
    "disclaimer": (
        "⚠️ <b>সতর্কতা / Disclaimer:</b> এই তালিকা শুধু নিজের ব্যক্তিগত রেফারেন্সের জন্য, এটি কোনও প্রেসক্রিপশন নয়। "
        "নতুন কোনও সমস্যা, বাচ্চাদের ওষুধ, বা এলার্জি/গর্ভাবস্থার মতো বিশেষ পরিস্থিতিতে অবশ্যই একজন যোগ্য ডাক্তার বা "
        "ফার্মাসিস্টের পরামর্শ নিন। AI-এর দেওয়া তথ্যও সাধারণ শিক্ষামূলক তথ্য মাত্র, চিকিৎসকের বিকল্প নয়।<br>"
        "This list is for personal reference only and is <b>not a prescription</b>. Always consult a qualified "
        "doctor or pharmacist for new problems, children's dosing, allergies, or pregnancy. The AI explanations "
        "below are general educational information only, not a medical diagnosis."
    ),
    "sidebar_header": "🔎 খুঁজুন | Find your medicine",
    "section_label": "সেকশন বাছুন | Select section:",
    "search_label": "খুঁজুন (সমস্যা বা ওষুধের নাম) | Search (problem or medicine)",
    "problem_label": "সমস্যা বাছুন | Or pick a problem:",
    "lang_label": "🌐 AI-এর উত্তরের ভাষা | AI answer language",
    "style_label": "📝 বিস্তারিত মাত্রা | Detail level",
    "no_children": "👶 এখনও বাচ্চাদের ওষুধের তথ্য যোগ করা হয়নি। | No children's medicine data added yet.",
    "no_match": "😕 কিছু পাওয়া যায়নি, অন্য শব্দ দিয়ে খুঁজে দেখুন। | No matching medicine found. Try a different term.",
    "ai_btn": "🤖 AI থেকে বিস্তারিত জানুন | Get AI details",
    "ai_regenerate": "🔄 আবার লিখুন | Regenerate",
    "ai_missing_key": (
        "🔑 AI ফিচার চালু করতে Streamlit **Secrets**-এ `GEMINI_API_KEY` যোগ করুন। "
        "Add your Gemini API key as `GEMINI_API_KEY` in Streamlit's Secrets manager to enable AI details."
    ),
    "ai_error": "⚠️ AI থেকে উত্তর আনা যায়নি | Could not fetch an AI response right now:",
    "freeform_header": "🗣️ নিজের সমস্যা লিখে AI-কে জিজ্ঞেস করুন | Ask AI about your own problem",
    "freeform_placeholder": "যেমন: গলা ব্যথা আর কাশি হচ্ছে দুদিন ধরে... | e.g. sore throat and cough for 2 days...",
    "freeform_btn": "🚀 AI-কে জিজ্ঞেস করুন | Ask AI",
    "copy_header": "📋 তালিকা কপি/ফরওয়ার্ড করুন | Copy / forward this list",
    "table_header": "📑 পুরো তালিকা দেখুন | View full table",
}

LANGUAGE_OPTIONS = {
    "বাংলা (Simple Bangla)": "simple_bangla",
    "বাংলা + English (Mixed)": "mixed",
    "English": "english",
}
STYLE_OPTIONS = {
    "সংক্ষিপ্ত | Short": "short",
    "বিস্তারিত | Detailed": "detailed",
}

# ----------------------------------------------------------------------------
# STYLING
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .med-card {
        background-color: #f7f9fc;
        border: 1px solid #e3e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .med-card h4 { margin: 0 0 6px 0; color: #1f2937; }
    .med-name { font-size: 1.15rem; font-weight: 700; color: #0f5132; }
    .bn-text { color: #6b7280; font-size: 0.9rem; }
    .disclaimer-box {
        background-color: #fff3cd;
        border: 1px solid #ffe69c;
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 0.92rem;
        color: #664d03;
    }
    .ai-box {
        background-color: #eef6ff;
        border: 1px solid #bfdcff;
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 6px;
        margin-bottom: 14px;
        font-size: 0.97rem;
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# GEMINI CLIENT SETUP
# ----------------------------------------------------------------------------
def get_gemini_client():
    """Reads GEMINI_API_KEY from Streamlit Secrets first, then env vars."""
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or not GENAI_SDK_AVAILABLE:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


gemini_client = get_gemini_client()
AI_ENABLED = gemini_client is not None

# Shared "house style" instruction so every AI answer stays safe, simple and
# on-topic no matter which language/detail level is chosen.
SYSTEM_INSTRUCTION = (
    "You are a friendly medicine-information helper inside a personal reference app used by ordinary "
    "people in rural West Bengal / Bangladesh, including people with little formal education. "
    "Explain things the way you would to a family member: short sentences, everyday words, no jargon "
    "(and if you must use a medical term, explain it in brackets). Never invent exact dosages for a "
    "specific person, never tell someone to stop a doctor-prescribed medicine, and always end by gently "
    "reminding them to see a doctor or pharmacist for anything serious, persistent, or for children, "
    "pregnancy, or allergies. You give general educational information only — never a diagnosis or a "
    "personal prescription."
)


def _language_instruction(language_code: str) -> str:
    return {
        "simple_bangla": "Answer ONLY in simple, everyday Bangla (সহজ বাংলা), as if talking to a villager. "
                         "Avoid English words where a common Bangla word exists.",
        "mixed": "Answer in a natural Bangla+English mix (Banglish), the way people actually text each other "
                 "in West Bengal/Bangladesh — simple Bangla sentences with common English words kept as-is.",
        "english": "Answer in simple, plain English, short sentences, no medical jargon.",
    }.get(language_code, "Answer in simple Bangla and English both.")


def _detail_instruction(style_code: str) -> str:
    if style_code == "short":
        return "Keep the WHOLE answer under 80 words, 4 short bullet points maximum."
    return "Give a clear, well-organised answer of about 150-220 words using the section headings requested."


def _generate_with_fallback(_client, prompt: str, system_instruction: str, temperature: float):
    """Tries each model in GEMINI_MODEL_CHAIN in order, returns (text, model_used, error).

    Remembers the first model that works (per session) in st.session_state so later
    calls don't re-probe models that are unavailable on this key/region every time.
    """
    chain = GEMINI_MODEL_CHAIN
    preferred = st.session_state.get("_working_gemini_model")
    if preferred and preferred in chain:
        chain = [preferred] + [m for m in chain if m != preferred]

    last_error = None
    for model_name in chain:
        try:
            response = _client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                ),
            )
            text = (response.text or "").strip()
            if text:
                st.session_state["_working_gemini_model"] = model_name
                return text, model_name, None
            last_error = "empty response"
        except Exception as exc:  # noqa: BLE001 — try the next model in the chain
            last_error = f"{model_name}: {exc}"
            continue
    return None, None, last_error


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def ask_gemini_medicine_details(_client, medicine: str, problem_en: str, problem_bn: str,
                                 language_code: str, style_code: str):
    """Cached call: explains one medicine/problem pair. Returns (text, error)."""
    if _client is None:
        return None, "no_key"

    prompt = f"""{_language_instruction(language_code)}
{_detail_instruction(style_code)}

A user of this personal medicine-reference app takes **{medicine}** for the problem
"{problem_en}" (Bangla: {problem_bn}).

Cover these points, using a short heading (with emoji) for each:
1. 💊 এটা কী কাজে লাগে / What it's used for — plain explanation of the problem and how this medicine helps.
2. 🧪 উপাদান/গোত্র / Composition & type — the generic/active ingredient and drug class, in simple words.
3. 🔁 বিকল্প / Similar alternatives — 2-3 other common brand names with the same generic ingredient that are
   typically available in local pharmacies in West Bengal/Bangladesh (mention this is for awareness, not a
   substitution instruction).
4. ⚠️ সতর্কতা / Precautions — the most important common precautions or side effects to watch for.
5. 🚨 কখন ডাক্তার দেখাবেন / When to see a doctor immediately — clear red-flag signs.

Do not state an exact mg dose for the person to take; only describe, in general terms, that this is
typically an over-the-counter/common medicine for this use if that's true.
"""
    text, _model_used, error = _generate_with_fallback(_client, prompt, SYSTEM_INSTRUCTION, temperature=0.4)
    return text, error


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def ask_gemini_freeform(_client, user_text: str, language_code: str, style_code: str):
    """Cached call: user describes their own problem in free text."""
    if _client is None:
        return None, "no_key"

    prompt = f"""{_language_instruction(language_code)}
{_detail_instruction(style_code)}

A user of a personal medicine-reference app (based in West Bengal/Bangladesh) describes this problem:
"{user_text}"

Respond with short headed sections (use emoji headings):
1. 🩺 সম্ভাব্য কারণ / What this might commonly be — 1-2 likely common, non-scary explanations.
2. 💊 সাধারণ ওষুধ / Commonly used medicine types — describe general categories (e.g. "a paracetamol-type
   fever reducer") rather than a single specific brand/dose, since you cannot examine the person.
3. 🏠 বাড়িতে যা করতে পারেন / What can help at home — simple, safe self-care steps.
4. 🚨 কখন ডাক্তার দেখাবেন / When to see a doctor immediately — clear red-flag signs for this symptom.

Be encouraging and calm. Make clear this is general information, not a diagnosis, and a doctor visit is
recommended if symptoms are severe, unusual, persistent beyond a couple of days, or involve a child,
pregnancy, or a known allergy.
"""
    text, _model_used, error = _generate_with_fallback(_client, prompt, SYSTEM_INSTRUCTION, temperature=0.5)
    return text, error


# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.title(TXT["app_title"])
st.caption(TXT["app_caption"])
st.markdown(f'<div class="disclaimer-box">{TXT["disclaimer"]}</div>', unsafe_allow_html=True)
st.write("")

# ----------------------------------------------------------------------------
# SIDEBAR: SETTINGS + SEARCH
# ----------------------------------------------------------------------------
st.sidebar.header(TXT["sidebar_header"])

section = st.sidebar.radio(TXT["section_label"], ["👤 Adult", "🧒 Children"], index=0)

data = ADULT_DATA if section == "👤 Adult" else CHILDREN_DATA
df = pd.DataFrame(data)

search_query = st.sidebar.text_input(TXT["search_label"], "")

problem_options = ["All"] + sorted(df["problem_en"].tolist()) if not df.empty else ["All"]
selected_problem = st.sidebar.selectbox(TXT["problem_label"], problem_options)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI সেটিংস | AI settings")
language_choice_label = st.sidebar.selectbox(TXT["lang_label"], list(LANGUAGE_OPTIONS.keys()))
style_choice_label = st.sidebar.selectbox(TXT["style_label"], list(STYLE_OPTIONS.keys()))
language_code = LANGUAGE_OPTIONS[language_choice_label]
style_code = STYLE_OPTIONS[style_choice_label]

if AI_ENABLED:
    working_model = st.session_state.get("_working_gemini_model")
    if working_model:
        st.sidebar.success(f"✅ AI সংযুক্ত | AI connected ({working_model})")
    else:
        st.sidebar.success("✅ AI সংযুক্ত | AI connected")
else:
    st.sidebar.warning(TXT["ai_missing_key"])

# ----------------------------------------------------------------------------
# FILTER LOGIC
# ----------------------------------------------------------------------------
filtered_df = df.copy()
if not filtered_df.empty:
    if selected_problem != "All":
        filtered_df = filtered_df[filtered_df["problem_en"] == selected_problem]
    if search_query.strip():
        q = search_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df["problem_en"].str.lower().str.contains(q)
            | filtered_df["problem_bn"].str.contains(search_query.strip())
            | filtered_df["medicine"].str.lower().str.contains(q)
        ]

# ----------------------------------------------------------------------------
# MAIN DISPLAY
# ----------------------------------------------------------------------------
if section == "🧒 Children" and df.empty:
    st.info(TXT["no_children"])
else:
    st.subheader(f"{section} — {'All Problems' if selected_problem == 'All' else selected_problem}")

    if filtered_df.empty:
        st.warning(TXT["no_match"])
    else:
        for i, row in filtered_df.reset_index(drop=True).iterrows():
            st.markdown(
                f"""
                <div class="med-card">
                    <h4>{row['emoji']} {row['problem_en']} <span class="bn-text">/ {row['problem_bn']}</span></h4>
                    <div class="med-name">💊 {row['medicine']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            state_key = f"ai_details__{section}__{row['medicine']}__{row['problem_en']}__{language_code}__{style_code}"

            with st.expander(TXT["ai_btn"]):
                if not AI_ENABLED:
                    st.info(TXT["ai_missing_key"])
                else:
                    show_regenerate = state_key in st.session_state
                    fetch = st.button(
                        TXT["ai_regenerate"] if show_regenerate else TXT["ai_btn"],
                        key=f"btn__{state_key}",
                    )
                    if fetch:
                        ask_gemini_medicine_details.clear() if show_regenerate else None
                        with st.spinner("AI লিখছে... | AI is thinking..."):
                            text, error = ask_gemini_medicine_details(
                                gemini_client, row["medicine"], row["problem_en"], row["problem_bn"],
                                language_code, style_code,
                            )
                        st.session_state[state_key] = (text, error)

                    if state_key in st.session_state:
                        text, error = st.session_state[state_key]
                        if error:
                            st.error(f"{TXT['ai_error']} {error}")
                        elif text:
                            st.markdown(f'<div class="ai-box">{text}</div>', unsafe_allow_html=True)

        with st.expander(TXT["copy_header"]):
            lines = [
                f"{r['emoji']} {r['problem_en']} ({r['problem_bn']}): {r['medicine']}"
                for _, r in filtered_df.iterrows()
            ]
            st.code("\n".join(lines), language=None)

# ----------------------------------------------------------------------------
# FREEFORM "ASK AI" SECTION
# ----------------------------------------------------------------------------
st.markdown("---")
st.subheader(TXT["freeform_header"])

if not AI_ENABLED:
    st.info(TXT["ai_missing_key"])
else:
    user_symptom = st.text_area(" ", placeholder=TXT["freeform_placeholder"], label_visibility="collapsed")
    if st.button(TXT["freeform_btn"]) and user_symptom.strip():
        with st.spinner("AI লিখছে... | AI is thinking..."):
            text, error = ask_gemini_freeform(gemini_client, user_symptom.strip(), language_code, style_code)
        if error:
            st.error(f"{TXT['ai_error']} {error}")
        elif text:
            st.markdown(f'<div class="ai-box">{text}</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# FULL TABLE VIEW (optional, collapsible)
# ----------------------------------------------------------------------------
if not df.empty:
    with st.expander(TXT["table_header"]):
        st.dataframe(
            df.rename(
                columns={"emoji": "", "problem_en": "Problem", "problem_bn": "সমস্যা", "medicine": "Medicine"}
            ),
            use_container_width=True,
            hide_index=True,
        )

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.write("")
st.markdown("---")
st.caption(
    "💡 নতুন ওষুধ/সমস্যা যোগ করতে app.py-এর উপরে ADULT_DATA / CHILDREN_DATA লিস্টে একটি নতুন লাইন যোগ করুন। "
    "| To add more medicines, edit the ADULT_DATA / CHILDREN_DATA lists at the top of app.py."
)
st.caption("⚠️ শুধুমাত্র ব্যক্তিগত রেফারেন্সের জন্য — ডাক্তারের পরামর্শের বিকল্প নয়। | For reference only — not a substitute for professional medical advice.")
