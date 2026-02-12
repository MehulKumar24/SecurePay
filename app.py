import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import numpy as np

st.set_page_config(page_title="SecurePay", layout="wide")

st.markdown("""
<style>
.title {font-size:36px;font-weight:600;margin-bottom:5px;}
.blue {color:#3b82f6;}
.subtitle {color:#9aa0a6;margin-bottom:18px;font-size:16px;}
.footer {text-align:center;color:#9aa0a6;font-size:13px;margin-top:20px;}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("SecurePay")
    st.caption("Financial Behaviour Intelligence")

    st.markdown("**Overview**")
    st.write("Detects abnormal financial behaviour using behavioural anomaly modelling.")

    st.markdown("**Capabilities**")
    st.write("- Behaviour anomaly detection\n- Risk pattern discovery\n- Transaction irregularity monitoring")

    st.markdown("---")
    st.write("SecurePay • 2026")
    st.write("Author: Mehul Kumar")

# Header
st.markdown('<div class="title"><span class="blue">SecurePay</span> — Suspicious Transaction Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Intelligent Financial Anomaly Detection</div>', unsafe_allow_html=True)

file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    with st.spinner("Processing dataset..."):
        df = pd.read_csv(file)

        # Auto-fix column formatting (lowercase + remove spaces)
        df.columns = df.columns.str.strip().str.lower()

        required_cols = ["amount", "amount_deviation", "txn_velocity", "behaviour_score"]

        if not all(col in df.columns for col in required_cols):
            st.error("Dataset missing required columns.")
            st.stop()

        X = df[required_cols].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
        model.fit(X_scaled)

        df["anomaly_flag"] = model.predict(X_scaled)
        df["anomaly_flag"] = df["anomaly_flag"].apply(lambda x: 1 if x == -1 else 0)

    total = len(df)
    anomalies = int(df["anomaly_flag"].sum())
    rate = (anomalies / total) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Suspicious Detected", anomalies)
    c3.metric("Anomaly Rate", f"{rate:.2f}%")

    st.markdown("---")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")

    st.subheader("Detected Suspicious Transactions")
    suspicious = df[df["anomaly_flag"] == 1]

    if len(suspicious) > 0:
        st.dataframe(suspicious.head(300), use_container_width=True)

        csv = suspicious.to_csv(index=False).encode("utf-8")
        st.download_button("Download Suspicious Transactions", csv, "suspicious_transactions.csv", "text/csv")
    else:
        st.success("No suspicious behaviour detected.")

    st.markdown("---")

    st.subheader("Anomaly Distribution")
    fig1 = px.histogram(df, x="anomaly_flag", nbins=2)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    st.subheader("Behaviour Analysis")
    sample_df = df.sample(min(5000, len(df)))
    fig2 = px.scatter(sample_df, x="amount_deviation", y="txn_velocity", color="anomaly_flag", opacity=0.7)
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Upload a CSV file to begin analysis.")

st.markdown("---")
st.markdown('<div class="footer">© 2026 SecurePay — Intelligent Financial Anomaly Detection • Mehul Kumar</div>', unsafe_allow_html=True)