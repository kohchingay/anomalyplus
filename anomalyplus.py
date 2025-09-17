import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt

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

# Boxplots
st.subheader("Distribution & Outliers")
for currency in df.columns:
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=currency, ax=ax)
    ax.set_title(f"{currency} Distribution")
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

# 📥 Sidebar inputs
st.sidebar.header("📥 Enter Today's Exchange Rates")
user_input = {}
for currency in df.columns:
    user_input[currency] = st.sidebar.number_input(f"{currency}", min_value=0.0, format="%.4f")

# 🔍 Predict anomalies
user_df = pd.DataFrame([user_input])
anomalies = {cur: models[cur].predict(user_df[[cur]])[0] for cur in df.columns}
anomalous = [cur for cur, pred in anomalies.items() if pred == -1]

# Display anomaly result
st.subheader("🔍 Anomaly Detection Result")
if anomalous:
    st.error(f"⚠️ Anomaly detected in: {', '.join(anomalous)}")
    for cur in anomalous:
        st.write(f"- {cur}: {user_input[cur]}")
else:
    st.success("✅ No anomalies detected in the entered exchange rates.")

# 💱 Arbitrage logic
if anomalous:
    st.sidebar.markdown("### 💱 Enter Pairwise Exchange Rates")
    arb_input = {}
    for base in anomalous:
        others = [c for c in df.columns if c != base]
        st.sidebar.markdown(f"**{base} vs others**")
        for target in others:
            key = f"{base}_{target}"
            arb_input[key] = st.sidebar.number_input(f"{base} → {target}", min_value=0.0001, format="%.4f")

    # Build arbitrage paths
    st.subheader("💡 Arbitrage Opportunities")
    for base in anomalous:
        others = [c for c in df.columns if c != base]
        for a in others:
            for b in others:
                if a != b and a != base and b != base:
                    try:
                        rate1 = arb_input[f"{base}_{a}"]
                        rate2 = arb_input[f"{a}_{b}"]
                        rate3 = arb_input[f"{b}_{base}"]
                        product = rate1 * rate2 * rate3
                        if product > 1.01:
                            st.warning(f"Arbitrage path: {base} → {a} → {b} → {base} | Profit multiplier: {round(product, 4)}")
                    except KeyError:
                        continue

# 📊 Optional: Show historical data for EUR, GBP, USD only
with st.expander("📊 Show Historical Exchange Rates"):
    st.dataframe(df[["EUR", "GBP", "USD"]].describe().T)

