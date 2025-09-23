import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt
import requests

# Title
st.title("💱 Exchange Rate Anomaly Detector")

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_excel("Exchange Rates 2017 to 2025.xlsx", usecols="D:F")
    df.columns = ["EUR", "GBP", "USD"]
    df = df.dropna()
    df = df[df.applymap(lambda x: isinstance(x, (int, float)))]
    df["SGD"] = 1.0  # Add synthetic SGD column
    df.index = pd.RangeIndex(start=0, stop=len(df), step=1)
    return df

df = load_data()

# 📈 Exploratory Data Analysis
st.header("📈 Exploratory Data Analysis")

# Summary statistics for EUR, GBP, USD only
st.subheader("Descriptive Statistics")
st.dataframe(df[["EUR", "GBP", "USD"]].describe().T)

# Line chart
st.subheader("Historical Exchange Rate Trends")
selected_currencies = st.multiselect("Select currencies to plot", df.columns.tolist(), default=df.columns.tolist())
st.line_chart(df[selected_currencies])

# Correlation matrix
st.subheader("Currency Correlation Matrix")
corr = df.corr()
sns.set(style="white")
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Boxplots (smaller, side-by-side, excluding SGD)
st.subheader("Distribution & Outliers")

cols_to_plot = ["EUR", "GBP", "USD"]
fig, axes = plt.subplots(1, len(cols_to_plot), figsize=(12, 3))

for i, currency in enumerate(cols_to_plot):
    sns.boxplot(data=df, x=currency, ax=axes[i])
    axes[i].set_title(f"{currency} Distribution")

st.pyplot(fig)

# 🧠 Train models
@st.cache_resource
def train_models(df):
    models = {}
    for currency in df.columns:
        model = IsolationForest(contamination=0.01, random_state=42)
        model.fit(df[[currency]])
        models[currency] = model
    return models

models = train_models(df)

FALLBACK_RATES = {"EUR": 1.51, "GBP": 1.73, "USD": 1.28, "SGD": 1.00}

def get_current_exchange_rates():
    url = "https://api.exchangerate.host/latest?base=SGD"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return {
            "EUR": float(data["rates"].get("EUR", FALLBACK_RATES["EUR"])),
            "GBP": float(data["rates"].get("GBP", FALLBACK_RATES["GBP"])),
            "USD": float(data["rates"].get("USD", FALLBACK_RATES["USD"])),
            "SGD": 1.0
        }
    except Exception:
        return FALLBACK_RATES.copy()

# Only fetch rates ONCE per session.
if "default_rates" not in st.session_state or not isinstance(st.session_state["default_rates"], dict):
    st.session_state["default_rates"] = get_current_exchange_rates()

default_rates = st.session_state["default_rates"]

# Defensive: ensure all needed keys exist
for k in ["EUR", "GBP", "USD", "SGD"]:
    if k not in default_rates:
        default_rates[k] = FALLBACK_RATES[k]

# 📥 Sidebar inputs with live defaults
st.sidebar.header("📥 Enter Today's Exchange Rates")
user_input = {}
for currency in ["EUR", "GBP", "USD", "SGD"]:
    default_val = float(default_rates.get(currency, FALLBACK_RATES[currency]))
    user_input[currency] = st.sidebar.number_input(
        f"{currency}",
        min_value=0.0,
        format="%.4f",
        value=default_val,
        key=f"input_{currency}"
    )

# 🔍 Predict anomalies
user_df = pd.DataFrame([user_input])
# Predict anomalies and get confidence scores (the further from 0, the higher the confidence)
anomalies = {}
confidence_scores = {}
for cur in df.columns:
    pred = models[cur].predict(user_df[[cur]])[0]
    score = models[cur].decision_function(user_df[[cur]])[0]  # Higher = more normal, Lower = more anomalous
    anomalies[cur] = pred
    confidence_scores[cur] = score

anomalous = [cur for cur, pred in anomalies.items() if pred == -1]

# Display anomaly result with confidence level only for flagged currencies
st.subheader("🔍 Anomaly Detection Result & Confidence")

if anomalous:
    for cur in anomalous:
        conf = confidence_scores[cur]
        st.error(f"⚠️ {cur}: Anomaly detected at {user_input[cur]} (Confidence Score: {conf:.3f})")
else:
    st.success("✅ No anomalies detected in the entered exchange rates.")

# 💱 Arbitrage logic: dynamic and robust
if anomalous:
    st.sidebar.markdown("### 💱 Enter Pairwise Exchange Rates")

    # Prepare a dictionary of all current exchange rates for autofill
    pairwise_defaults = {}
    for c1 in df.columns:
        for c2 in df.columns:
            if c1 == c2:
                continue
            # Compute the default rate from current exchange rates if possible
            # c1 to c2: rate_c2/rate_c1
            try:
                pairwise_defaults[f"{c1}_{c2}"] = float(default_rates[c2]) / float(default_rates[c1])
            except Exception:
                pairwise_defaults[f"{c1}_{c2}"] = 1.0

    for base in anomalous:
        st.sidebar.markdown(f"**Exchange rates for {base}**")
        others = [c for c in df.columns if c != base]
        arb_input = {}

        # Collect all pairwise rates involving base and others, with smart default values
        for a in others:
            key1 = f"{base}_{a}"
            key2 = f"{a}_{base}"
            arb_input[key1] = st.sidebar.number_input(
                f"{base} → {a}",
                min_value=0.0001,
                format="%.4f",
                value=pairwise_defaults.get(key1, 1.0),
                key=f"arb_{key1}"
            )
            arb_input[key2] = st.sidebar.number_input(
                f"{a} → {base}",
                min_value=0.0001,
                format="%.4f",
                value=pairwise_defaults.get(key2, 1.0),
                key=f"arb_{key2}"
            )

        # Also collect rates between others (non-base), with smart default values
        for i in range(len(others)):
            for j in range(len(others)):
                if i != j:
                    key = f"{others[i]}_{others[j]}"
                    arb_input[key] = st.sidebar.number_input(
                        f"{others[i]} → {others[j]}",
                        min_value=0.0001,
                        format="%.4f",
                        value=pairwise_defaults.get(key, 1.0),
                        key=f"arb_{key}"
                    )

        # Generate all valid 3-step loops: base → A → B → base
        st.subheader(f"💡 Arbitrage Opportunities for {base}")
        best_path = None
        best_profit = 1.0

        for a in others:
            for b in others:
                if a != b:
                    try:
                        rate1 = arb_input[f"{base}_{a}"]
                        rate2 = arb_input[f"{a}_{b}"]
                        rate3 = arb_input[f"{b}_{base}"]
                        product = rate1 * rate2 * rate3
                        if product > 1.01:
                            st.warning(f"Arbitrage path: {base} → {a} → {b} → {base} | Profit multiplier: {round(product, 4)}")
                            if product > best_profit:
                                best_profit = product
                                best_path = (base, a, b, base)
                    except KeyError:
                        continue  # Skip incomplete paths

        if best_path:
            st.success(f"✅ Recommended arbitrage path: {' → '.join(best_path)} | Profit multiplier: {round(best_profit, 4)}")
        else:
            st.info(f"No arbitrage paths found for {base}.")

# 📊 Optional: Show historical data for EUR, GBP, USD only
with st.expander("📊 Show Historical Exchange Rates"):
    st.dataframe(df[["EUR", "GBP", "USD"]].describe().T)

