import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="US Stock High Price Predictor", page_icon="📈", layout="wide")

st.title("📈 US Stock — Next-Day High Price Predictor")
st.write(
    "Supervised Regression — predicts the **next trading day's High price** "
    "using live data pulled from Yahoo Finance via `yfinance`."
)

# =====================================================
# SIDEBAR — TICKER INPUT
# =====================================================

st.sidebar.header("⚙️ Settings")

ticker_input = st.sidebar.text_input(
    "Enter a US stock ticker",
    value="AAPL",
    help="Examples: AAPL, NVDA, MSFT, AMZN, GOOGL",
).strip().upper()

period = st.sidebar.selectbox(
    "Historical data window",
    ["2y", "3y", "5y", "10y", "max"],
    index=2,
    help="More history generally means a more stable model.",
)

st.sidebar.markdown("**📈 Regression Models:** Linear Regression, Decision Tree, Random Forest — all 3 always train and are compared below.")

n_estimators = st.sidebar.slider(
    "Random Forest trees (n_estimators)",
    50, 300, 150, step=50,
    help="Only affects the Random Forest model's setting.",
)
st.sidebar.caption("Linear Regression and Decision Tree use default settings — no tuning needed. All 3 models always train and compare below.")

run_button = st.sidebar.button("🚀 Fetch Data & Train Model", use_container_width=True)

st.sidebar.caption(
    "All data is pulled live from Yahoo Finance — nothing is hardcoded. "
    "No investment advice; educational demo only."
)

# =====================================================
# DATA FETCH
# =====================================================


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    raw = yf.download(ticker, period=period, auto_adjust=False, progress=False)

    if raw is None or raw.empty:
        return pd.DataFrame()

    # yfinance can return a MultiIndex column frame (esp. for single-ticker
    # calls on newer versions) — flatten it down to plain OHLCV columns.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()
    raw = raw.rename(columns={"Date": "Date"})

    keep_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in raw.columns]
    df = raw[keep_cols].copy()

    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).sort_values("Date").reset_index(drop=True)
    return df


# =====================================================
# FEATURE ENGINEERING
# =====================================================


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds model-ready features.

    Target  : Next_High -> tomorrow's High price (shift Close/High forward)
    Features: today's OHLCV + the latest 5 closing prices (Close_lag1..lag5)
              + a couple of simple derived features (daily range, 5-day
              rolling average close) that regressions typically benefit from.
    """
    data = df.copy()

    # Latest 5 closing prices as explicit lag features
    for lag in range(1, 6):
        data[f"Close_lag{lag}"] = data["Close"].shift(lag)

    # A couple of light derived features
    data["Daily_Range"] = data["High"] - data["Low"]
    data["Close_MA5"] = data["Close"].rolling(window=5).mean()

    # Target: next trading day's High price
    data["Next_High"] = data["High"].shift(-1)

    data = data.dropna().reset_index(drop=True)
    return data


FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "Close_lag1", "Close_lag2", "Close_lag3", "Close_lag4", "Close_lag5",
    "Daily_Range", "Close_MA5",
]

# =====================================================
# MAIN
# =====================================================

if run_button or "last_ticker" in st.session_state:

    if run_button:
        st.session_state["last_ticker"] = ticker_input
        st.session_state["last_period"] = period

    active_ticker = st.session_state.get("last_ticker", ticker_input)
    active_period = st.session_state.get("last_period", period)

    with st.spinner(f"Fetching {active_ticker} data from Yahoo Finance..."):
        raw_df = fetch_data(active_ticker, active_period)

    if raw_df.empty:
        st.error(
            f"❌ No data returned for ticker **{active_ticker}**. "
            "Check that it's a valid US ticker (e.g. AAPL, NVDA, MSFT, AMZN, GOOGL) and try again."
        )
        st.stop()

    feat_df = build_features(raw_df)

    if len(feat_df) < 50:
        st.warning(
            "⚠️ Not much usable history after feature engineering "
            f"({len(feat_df)} rows). Try a longer period for a more reliable model."
        )

    # -------------------------------------------------
    # LATEST SNAPSHOT
    # -------------------------------------------------

    latest = raw_df.iloc[-1]
    latest_5_closes = raw_df["Close"].tail(5).tolist()
    historical_high = raw_df["High"].max()

    st.subheader(f"🧾 Latest Snapshot — {active_ticker}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Latest Open", f"${latest['Open']:.2f}")
    c2.metric("Latest Close", f"${latest['Close']:.2f}")
    c3.metric("Latest High", f"${latest['High']:.2f}")
    c4.metric("Latest Low", f"${latest['Low']:.2f}")
    c5.metric("Latest Volume", f"{int(latest['Volume']):,}")

    st.markdown("**Latest 5 Closing Prices:**  " + "  ·  ".join(f"${p:.2f}" for p in latest_5_closes))
    st.markdown(f"**All-Time High (in fetched window):** ${historical_high:.2f}  (as of {active_period} history)")

    with st.expander("👁️ View Raw OHLCV Data"):
        st.dataframe(raw_df, use_container_width=True)

    st.divider()

    # -------------------------------------------------
    # PRICE HISTORY CHART
    # -------------------------------------------------

    st.subheader(f"📉 {active_ticker} — Close & High Price History")

    fig0, ax0 = plt.subplots(figsize=(11, 3.5))
    ax0.plot(raw_df["Date"], raw_df["Close"], color="steelblue", linewidth=1.1, label="Close")
    ax0.plot(raw_df["Date"], raw_df["High"], color="seagreen", linewidth=0.8, alpha=0.6, label="High")
    ax0.fill_between(raw_df["Date"], raw_df["Close"], alpha=0.08, color="steelblue")
    ax0.set_xlabel("Date")
    ax0.set_ylabel("Price (USD)")
    ax0.set_title(f"{active_ticker} — Close vs High ({active_period} history)")
    ax0.legend(fontsize=8)
    ax0.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig0)

    st.divider()

    # -------------------------------------------------
    # TIME-SERIES AWARE TRAIN/TEST SPLIT
    # No shuffling — earliest 80% trains, most recent 20% tests.
    # -------------------------------------------------

    x = feat_df[FEATURE_COLS]
    y = feat_df["Next_High"]

    split_idx = int(len(feat_df) * 0.8)
    x_train, x_test = x.iloc[:split_idx], x.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = feat_df["Date"].iloc[split_idx:]

    if len(x_test) < 5:
        st.warning("⚠️ Very small test window — metrics may be noisy. Consider a longer history period.")

    # -------------------------------------------------
    # TRAIN MODELS
    # -------------------------------------------------

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=n_estimators, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        results[name] = {
            "model": model,
            "pred": pred,
            "mae": mean_absolute_error(y_test, pred),
            "rmse": np.sqrt(mean_squared_error(y_test, pred)),
            "r2": r2_score(y_test, pred),
        }

    best_name = max(results, key=lambda k: results[k]["r2"])
    best = results[best_name]

    # -------------------------------------------------
    # METRICS TABLE
    # -------------------------------------------------

    st.subheader("📊 Regression Metrics (Time-Series Test Split)")

    st.markdown(
        """
        | Metric | What it means |
        |--------|--------------|
        | **MAE** | Average dollar error between predicted and actual High — lower is better |
        | **RMSE** | Like MAE but penalizes large misses more — lower is better |
        | **R² Score** | How much of the price variation the model explains — closer to 1.0 is better |
        """
    )

    metrics_df = pd.DataFrame({
        "Model": list(results.keys()),
        "MAE ($)": [round(results[m]["mae"], 4) for m in results],
        "RMSE ($)": [round(results[m]["rmse"], 4) for m in results],
        "R² Score": [round(results[m]["r2"], 4) for m in results],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.divider()

    # -------------------------------------------------
    # BAR CHART — R² COMPARISON ACROSS ALL 3 MODELS
    # -------------------------------------------------

    st.subheader("📊 R² Score Comparison (Higher = Better)")

    fig_bar, ax_bar = plt.subplots()

    model_names = list(results.keys())
    r2_scores = [results[m]["r2"] for m in model_names]
    bar_colors = ["steelblue", "tomato", "seagreen"]

    bars = ax_bar.bar(model_names, r2_scores, color=bar_colors, edgecolor="black", linewidth=0.7)

    for bar in bars:
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.4f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax_bar.set_xlabel("Models")
    ax_bar.set_ylabel("R² Score")
    ax_bar.set_title(f"{active_ticker} — R² Score Comparison (Next-Day High)")
    y_top = max(1.1, max(r2_scores) + 0.15) if r2_scores else 1.1
    ax_bar.set_ylim(min(0, min(r2_scores) - 0.05), y_top)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_bar)

    st.success(f"✅ Best Model: **{best_name}**  |  R² Score: {best['r2']:.4f}")

    st.divider()

    # -------------------------------------------------
    # ACTUAL vs PREDICTED — PER-MODEL LINE CHARTS
    # One subplot per model, all sharing the same
    # chronological test-window x-axis (dates_test).
    # -------------------------------------------------

    st.subheader("📈 Actual vs Predicted High — Comparative Line Charts")
    st.caption(
        "The blue line is the real next-day High from the test window. "
        "The coloured line is what each model predicted. "
        "Closer the lines → better the model."
    )

    plot_info = [
        ("Linear Regression", "tomato"),
        ("Decision Tree", "darkorange"),
        ("Random Forest", "seagreen"),
    ]

    fig_multi, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for ax, (name, color) in zip(axes, plot_info):
        pred = results[name]["pred"]
        ax.plot(dates_test, y_test.values, color="steelblue", linewidth=1.2,
                label="Actual High", zorder=2)
        ax.plot(dates_test, pred, color=color, linewidth=1.0,
                linestyle="--", label=f"Predicted ({name})", zorder=3, alpha=0.85)
        ax.fill_between(dates_test, y_test.values, pred, alpha=0.12, color=color, label="Error gap")
        ax.set_title(f"{name}  (R² = {results[name]['r2']:.4f})", fontsize=11, fontweight="bold")
        ax.set_ylabel("High Price ($)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    axes[-1].set_xlabel("Date (test window, chronological order)")
    plt.suptitle(f"{active_ticker} — Actual vs Predicted Next-Day High",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    st.pyplot(fig_multi)

    st.divider()

    # -------------------------------------------------
    # SIDE-BY-SIDE OVERLAY — all 3 models on one chart
    # -------------------------------------------------

    st.subheader("📊 All Models Overlaid on One Chart")
    st.caption("Easier to compare which model tracks the actual High most closely.")

    fig_overlay, ax_overlay = plt.subplots(figsize=(12, 4.5))

    ax_overlay.plot(dates_test, y_test.values, color="steelblue", linewidth=2,
                     label="Actual High", zorder=4)
    ax_overlay.plot(dates_test, results["Linear Regression"]["pred"], color="tomato",
                     linewidth=1, linestyle="--", label="Linear Regression", alpha=0.8)
    ax_overlay.plot(dates_test, results["Decision Tree"]["pred"], color="darkorange",
                     linewidth=1, linestyle=":", label="Decision Tree", alpha=0.8)
    ax_overlay.plot(dates_test, results["Random Forest"]["pred"], color="seagreen",
                     linewidth=1.2, linestyle="-.", label="Random Forest", alpha=0.85)

    ax_overlay.set_title(f"{active_ticker} — Actual vs All Predicted High — Overlaid",
                          fontsize=12, fontweight="bold")
    ax_overlay.set_xlabel("Date (test window, chronological order)")
    ax_overlay.set_ylabel("High Price ($)")
    ax_overlay.legend(fontsize=9)
    ax_overlay.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    st.pyplot(fig_overlay)

    st.divider()

    # -------------------------------------------------
    # NEXT-DAY PREDICTION
    # -------------------------------------------------

    st.subheader(f"🔮 Predicted Next Trading Day High — {active_ticker}")

    last_row = feat_df[FEATURE_COLS].iloc[[-1]]
    predicted_high = best["model"].predict(last_row)[0]
    current_high = latest["High"]
    diff = predicted_high - current_high
    pct_move = diff / current_high * 100
    direction = "📈 UP" if diff >= 0 else "📉 DOWN"

    p1, p2, p3 = st.columns(3)
    p1.metric("Latest High", f"${current_high:.2f}")
    p2.metric("Predicted Next-Day High", f"${predicted_high:.2f}", f"{diff:+.2f}")
    p3.metric("Expected Move", f"{pct_move:+.2f}%", direction)

    if diff >= 0:
        st.success(
            f"🔮 Predicted next trading day High for **{active_ticker}**: **${predicted_high:.2f}** "
            f"({direction} by ${abs(diff):.2f} / {abs(pct_move):.2f}% vs. latest High)"
        )
    else:
        st.error(
            f"🔮 Predicted next trading day High for **{active_ticker}**: **${predicted_high:.2f}** "
            f"({direction} by ${abs(diff):.2f} / {abs(pct_move):.2f}% vs. latest High)"
        )

    st.caption(f"Prediction generated using: {best_name} (highest R² on time-series test split)")

    # All-model comparison for the same next-day prediction
    st.markdown("**All 3 model predictions for next-day High:**")
    compare_df = pd.DataFrame({
        "Model": list(results.keys()),
        "Predicted High ($)": [round(results[m]["model"].predict(last_row)[0], 2) for m in results],
    })
    compare_df["Change vs Latest High"] = compare_df["Predicted High ($)"].apply(
        lambda v: f"{'▲' if v >= current_high else '▼'} ${abs(v - current_high):.2f}"
    )
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    st.warning(
        "⚠️ Disclaimer: This is a Machine Learning demonstration for educational purposes only. "
        "Stock prices are affected by many external factors not captured in this model. "
        "Do NOT use this for actual investment decisions."
    )

else:
    st.info("👈 Enter a US stock ticker in the sidebar and click **Fetch Data & Train Model** to begin.")

# =====================================================
# FOOTER
# =====================================================

st.divider()
st.caption("US Stock High Price Predictor  |  Live Yahoo Finance data via yfinance  |  Time-Series Regression Demo")
