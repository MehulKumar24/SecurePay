import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(page_title="SecurePay", layout="wide")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("SecurePay")
    st.caption("Financial Behaviour Intelligence")

    st.markdown("### Detection Mode")
    st.write("Real-world adaptive anomaly detection")

    st.markdown("### Model")
    st.write("IsolationForest + Behaviour Rules")

    st.markdown("### Filtering")
    st.write("• Weak anomalies removed")
    st.write("• Behaviour aware")
    st.write("• Realistic fraud assumption")

    st.markdown("---")
    st.write("SecurePay • 2026")

# ---------------- CSS ----------------
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0b0f1a;
    color: #e6edf3;
    font-family: "Segoe UI", Roboto, sans-serif;
}
.main .block-container {
    max-width: 1200px;
}
.title {
    font-size: 42px;
    font-weight: 600;
    text-align: center;
    color: #3b82f6;
}
.subtitle {
    text-align: center;
    color: #9aa4b2;
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">SecurePay</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Intelligent Financial Anomaly Detection System</div>', unsafe_allow_html=True)

# ---------------- FILE UPLOAD ----------------
file = st.file_uploader("Upload Transaction Dataset (CSV)", type=["csv"])

if file is not None:

    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    required_cols = ["txn_amount", "amount_deviation", "txn_velocity", "behavior_score"]
    if any(col not in df.columns for col in required_cols):
        st.error("Dataset must contain: txn_amount, amount_deviation, txn_velocity, behavior_score")
        st.stop()

    df["raw_amount_deviation"] = pd.to_numeric(df["amount_deviation"], errors="coerce")

    X = df[required_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=250,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    df["anomaly_flag"] = (model.predict(X_scaled) == -1).astype(int)
    df["anomaly_score"] = model.decision_function(X_scaled)

    # -------- REAL-BANK FILTER --------
    dev_thresh = df["raw_amount_deviation"].quantile(0.95)
    vel_thresh = df["txn_velocity"].quantile(0.90)
    beh_thresh = 0.40

    df["real_anomaly"] = (
        (df["anomaly_flag"] == 1) &
        (df["raw_amount_deviation"] >= dev_thresh) &
        (df["txn_velocity"] >= vel_thresh) &
        (df["behavior_score"] <= beh_thresh)
    ).astype(int)

    # -------- SEVERITY --------
    def severity(score):
        if score < -0.25:
            return "High"
        elif score < -0.08:
            return "Medium"
        else:
            return "Low"

    df["severity"] = df["anomaly_score"].apply(severity)

    # -------- RISK SCORE --------
    df["risk_score"] = (
        (df["raw_amount_deviation"] * 0.45) +
        (df["txn_velocity"] * 0.35) +
        ((1 - df["behavior_score"]) * 0.20)
    )

    # -------- WHY FLAGGED (Explainability) --------
    def reason(row):
        r = []
        if row["raw_amount_deviation"] >= dev_thresh:
            r.append("High Deviation")
        if row["txn_velocity"] >= vel_thresh:
            r.append("High Velocity")
        if row["behavior_score"] <= beh_thresh:
            r.append("Abnormal Behaviour")
        return ", ".join(r)

    df["flag_reason"] = df.apply(reason, axis=1)

    # -------- METRICS --------
    total = len(df)
    anomalies = int(df["real_anomaly"].sum())
    rate = (anomalies / total) * 100 if total else 0

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Real Suspicious Detected", anomalies)
    c3.metric("Anomaly Rate", f"{rate:.2f}%")

    # -------- PREVIEW --------
    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # -------- SUSPICIOUS --------
    st.markdown("---")
    st.subheader("Real Suspicious Transactions")

    suspicious = df[df["real_anomaly"] == 1]

    if not suspicious.empty:
        st.dataframe(
            suspicious.sort_values("risk_score", ascending=False).head(300),
            use_container_width=True
        )

        csv = suspicious.to_csv(index=False).encode("utf-8")
        st.download_button("Download Suspicious Transactions", csv, "suspicious_transactions.csv")

    else:
        st.success("No strong anomalies detected (real-world filtering applied).")

    # -------- VISUAL --------
    st.markdown("---")
    st.subheader("Anomaly Distribution")
    fig1 = px.histogram(df, x="real_anomaly", template="plotly_dark")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")
    st.subheader("Behaviour Analysis")
    sample_df = df.sample(min(4000, len(df)))
    fig2 = px.scatter(
        sample_df,
        x="raw_amount_deviation",
        y="txn_velocity",
        color="real_anomaly",
        template="plotly_dark",
        opacity=0.7
    )
    st.plotly_chart(fig2, use_container_width=True)

    # -------- TOP RISK (FIXED: HIGH → LOW) --------
    st.markdown("---")
    st.subheader("Top Risk Transactions (High → Low)")
    top_risk = df.sort_values("risk_score", ascending=False).head(20)
    st.dataframe(top_risk, use_container_width=True)

else:
    st.info("Upload a CSV file to begin analysis.")

st.markdown("---")
st.markdown("© 2026 SecurePay — Intelligent Financial Anomaly Detection")