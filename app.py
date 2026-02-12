import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(page_title="SecurePay", layout="wide")

# =========================
# Sidebar
# =========================
st.sidebar.title("SecurePay")
st.sidebar.write("AI-based Transaction Anomaly Detection")

st.sidebar.markdown("### About")
st.sidebar.write(
    "Upload a transaction dataset to detect suspicious or anomalous financial behaviour using Isolation Forest."
)

st.sidebar.markdown("### Expected Columns")
st.sidebar.code("""
amount
amount_deviation
txn_velocity
behaviour_score
""")

# =========================
# Header
# =========================
st.title("🔍 SecurePay — Anomaly Detection System")
st.write("Upload your dataset and detect suspicious financial transactions.")

# =========================
# Upload
# =========================
file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    required_cols = ["amount", "amount_deviation", "txn_velocity", "behaviour_score"]

    # =========================
    # Validate Columns
    # =========================
    if not all(col in df.columns for col in required_cols):
        st.error("Dataset must contain required columns.")
        st.stop()

    # =========================
    # Feature Preparation
    # =========================
    X = df[required_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =========================
    # Isolation Forest
    # =========================
    model = IsolationForest(
        n_estimators=100,
        contamination="auto",   # FIXED
        random_state=42
    )

    model.fit(X_scaled)

    df["anomaly"] = model.predict(X_scaled)
    df["anomaly"] = df["anomaly"].apply(lambda x: 1 if x == -1 else 0)

    # =========================
    # Metrics
    # =========================
    anomaly_count = df["anomaly"].sum()
    total = len(df)
    rate = (anomaly_count / total) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", total)
    col2.metric("Detected Anomalies", anomaly_count)
    col3.metric("Anomaly Rate (%)", f"{rate:.2f}")

    # =========================
    # Suspicious Transactions
    # =========================
    st.subheader("⚠ Detected Suspicious Transactions")
    suspicious = df[df["anomaly"] == 1]

    if len(suspicious) > 0:
        st.dataframe(suspicious.head(200))
    else:
        st.success("No suspicious transactions detected.")

    # =========================
    # Distribution Chart
    # =========================
    st.subheader("Anomaly Distribution")

    fig = px.histogram(df, x="anomaly", nbins=2)
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # Scatter Behaviour Map
    # =========================
    st.subheader("Behaviour Analysis")

    fig2 = px.scatter(
        df.sample(min(5000, len(df))),
        x="amount_deviation",
        y="txn_velocity",
        color="anomaly",
        title="Deviation vs Velocity"
    )

    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Upload a CSV file to begin anomaly detection.")

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown("© 2026 SecurePay | Financial Anomaly Detection Prototype")