import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(page_title="SecurePay", layout="wide")

# ---------- DARK UI STYLE ----------
st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #0b0f1a;
    color: #e6edf3;
    font-family: "Segoe UI", Roboto, sans-serif;
}

/* Title */
.title {
    font-size: 40px;
    font-weight: 600;
    text-align: center;
    margin-top: 30px;
    margin-bottom: 5px;
}

.blue { color: #3b82f6; }

.subtitle {
    text-align: center;
    color: #9aa4b2;
    font-size: 15px;
    margin-bottom: 40px;
}

/* Upload Box */
[data-testid="stFileUploader"] {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 35px;
    width: 55%;
    margin: auto;
    margin-top: 40px;
    box-shadow: 0 0 0 1px rgba(59,130,246,0.06);
}

/* Button */
button[kind="secondary"] {
    background-color: #1f2937 !important;
    color: #e5e7eb !important;
    border-radius: 8px !important;
    border: 1px solid #374151 !important;
}

button[kind="secondary"]:hover {
    border: 1px solid #3b82f6 !important;
    color: #3b82f6 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #111827;
    border-radius: 10px;
    padding: 14px;
    border: 1px solid #1f2937;
}

/* Tables */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    border: 1px solid #1f2937;
}

/* Footer */
.footer {
    text-align: center;
    color: #6b7280;
    font-size: 13px;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title"><span class="blue">SecurePay</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Intelligent Financial Anomaly Detection System</div>', unsafe_allow_html=True)

# ---------- FILE UPLOAD ----------
file = st.file_uploader("Upload Transaction Dataset (CSV)", type=["csv"])

if file is not None:
    with st.spinner("Processing dataset..."):
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower()

        required_cols = [
            "txn_amount",
            "amount_deviation",
            "txn_velocity",
            "behavior_score"
        ]

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        X = df[required_cols].copy().fillna(0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(n_estimators=150, contamination=0.03, random_state=42)
        model.fit(X_scaled)

        df["anomaly_flag"] = model.predict(X_scaled)
        df["anomaly_flag"] = df["anomaly_flag"].apply(lambda x: 1 if x == -1 else 0)

    # ---------- METRICS ----------
    st.markdown("---")
    total = len(df)
    anomalies = int(df["anomaly_flag"].sum())
    rate = (anomalies / total) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Suspicious Detected", anomalies)
    c3.metric("Anomaly Rate", f"{rate:.2f}%")

    # ---------- DATA PREVIEW ----------
    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # ---------- SUSPICIOUS ----------
    st.markdown("---")
    st.subheader("Detected Suspicious Transactions")
    suspicious = df[df["anomaly_flag"] == 1]

    if len(suspicious) > 0:
        st.dataframe(suspicious.head(300), use_container_width=True)
        csv = suspicious.to_csv(index=False).encode("utf-8")
        st.download_button("Download Suspicious Transactions", csv, "suspicious_transactions.csv", "text/csv")
    else:
        st.success("No suspicious behaviour detected.")

    # ---------- DISTRIBUTION ----------
    st.markdown("---")
    st.subheader("Anomaly Distribution")
    fig1 = px.histogram(df, x="anomaly_flag", nbins=2, template="plotly_dark")
    st.plotly_chart(fig1, use_container_width=True)

    # ---------- SCATTER ----------
    st.markdown("---")
    st.subheader("Behaviour Analysis")
    sample_df = df.sample(min(5000, len(df)))
    fig2 = px.scatter(
        sample_df,
        x="amount_deviation",
        y="txn_velocity",
        color="anomaly_flag",
        opacity=0.7,
        template="plotly_dark"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ---------- RISK SCORE ----------
    st.markdown("---")
    st.subheader("Top Risk Transactions")

    df["risk_score"] = (
        (df["amount_deviation"] * 0.5) +
        (df["txn_velocity"] * 0.3) +
        (df["behavior_score"] * 0.2)
    )

    top_risk = df.sort_values("risk_score", ascending=False).head(20)
    st.dataframe(top_risk, use_container_width=True)

else:
    st.info("Upload a CSV file to begin analysis.")

# ---------- FOOTER ----------
st.markdown('<div class="footer">© 2026 SecurePay — Intelligent Financial Anomaly Detection</div>', unsafe_allow_html=True)