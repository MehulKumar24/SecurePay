import importlib
import io
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_recall_curve

px: Any = None
go: Any = None
plt: Any = None
PdfPages: Any = None

try:
    px = importlib.import_module("plotly.express")
    go = importlib.import_module("plotly.graph_objects")
    PLOTLY_AVAILABLE = True
except ModuleNotFoundError:
    px = None
    go = None
    PLOTLY_AVAILABLE = False

try:
    plt = importlib.import_module("matplotlib.pyplot")
    PdfPages = importlib.import_module("matplotlib.backends.backend_pdf").PdfPages
    MATPLOTLIB_AVAILABLE = True
except ModuleNotFoundError:
    plt = None
    PdfPages = None
    MATPLOTLIB_AVAILABLE = False

REQUIRED_COLUMNS = [
    "txn_id",
    "txn_hour",
    "txn_amount",
    "amount_deviation",
    "txn_velocity",
    "behavior_score",
    "payment_channel",
    "risk_flag",
]

MODEL_FEATURES = [
    "txn_hour",
    "txn_amount",
    "amount_deviation",
    "txn_velocity",
    "behavior_score",
]

SEVERITY_ORDER = ["Critical", "High", "Medium", "Low"]
SEVERITY_COLORS = {
    "Critical": "#ef4444",
    "High": "#f97316",
    "Medium": "#facc15",
    "Low": "#22c55e",
}

PLOT_CONFIG = {"displaylogo": False, "responsive": True}

st.set_page_config(page_title="SecurePay | Bank Risk Console", page_icon=":bank:", layout="wide")

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
    }
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(30, 64, 175, 0.18), transparent 35%),
            radial-gradient(circle at 80% 0%, rgba(8, 145, 178, 0.12), transparent 32%),
            #0b1220;
        color: #e5edf6;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
    }
    .title {
        font-size: 2.15rem;
        font-weight: 760;
        color: #f8fafc;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #9fb1c7;
        margin-bottom: 1rem;
    }
    .ticker {
        width: 100%;
        overflow: hidden;
        border: 1px solid #27364d;
        border-radius: 12px;
        background: rgba(13, 23, 38, 0.95);
        margin: 0.5rem 0 1rem 0;
    }
    .ticker span {
        display: inline-block;
        white-space: nowrap;
        color: #f8fafc;
        padding: 0.5rem 0;
        padding-left: 100%;
        animation: ticker 24s linear infinite;
        font-size: 0.93rem;
    }
    @keyframes ticker {
        0% { transform: translateX(0); }
        100% { transform: translateX(-100%); }
    }
    .status-row {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-top: 0.2rem;
    }
    .status-chip {
        border: 1px solid #2c3f58;
        background: rgba(15, 23, 42, 0.95);
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        font-size: 0.82rem;
        color: #dbe6f3;
    }
    .footer {
        text-align: center;
        margin-top: 1.4rem;
        padding-top: 1rem;
        border-top: 1px solid #223145;
        color: #8ca0b7;
        font-size: 0.84rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def add_reason_codes(frame: pd.DataFrame) -> pd.DataFrame:
    q99_amount = frame["txn_amount"].quantile(0.99)

    reason_codes: List[str] = []
    for row in frame[
        [
            "txn_hour",
            "amount_deviation",
            "txn_velocity",
            "behavior_score",
            "txn_amount",
            "high_risk",
        ]
    ].to_dict(orient="records"):
        reasons = []
        txn_hour = float(row["txn_hour"]) if pd.notna(row["txn_hour"]) else np.nan
        amount_deviation = float(row["amount_deviation"]) if pd.notna(row["amount_deviation"]) else np.nan
        txn_velocity = float(row["txn_velocity"]) if pd.notna(row["txn_velocity"]) else np.nan
        behavior_score = float(row["behavior_score"]) if pd.notna(row["behavior_score"]) else np.nan
        txn_amount = float(row["txn_amount"]) if pd.notna(row["txn_amount"]) else np.nan
        high_risk = int(row["high_risk"]) if pd.notna(row["high_risk"]) else 0

        if pd.notna(txn_hour) and (txn_hour <= 5 or txn_hour >= 22):
            reasons.append("Off-hour activity")
        if pd.notna(amount_deviation) and amount_deviation >= 1.0:
            reasons.append("Amount deviation spike")
        if pd.notna(txn_velocity) and txn_velocity >= 1.2:
            reasons.append("High transaction velocity")
        if pd.notna(behavior_score) and behavior_score >= 0.65:
            reasons.append("Behavior score elevated")
        if pd.notna(txn_amount) and txn_amount >= q99_amount:
            reasons.append("Large amount outlier")
        if not reasons and high_risk == 1:
            reasons.append("Composite model anomaly")
        reason_codes.append(", ".join(reasons) if reasons else "Within expected pattern")

    frame["reason_codes"] = reason_codes
    return frame


def compute_drift_by_batch(frame: pd.DataFrame, batch_size: int = 250) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame({"batch": [0], "drift_score": [0.0]})

    drift_base = frame[MODEL_FEATURES].copy()
    drift_base = drift_base.fillna(drift_base.median(numeric_only=True))
    drift_base["batch"] = (np.arange(len(drift_base)) // max(batch_size, 1)).astype(int)

    batch_means = drift_base.groupby("batch", as_index=True)[MODEL_FEATURES].mean()
    if batch_means.shape[0] <= 1:
        return pd.DataFrame({"batch": [0], "drift_score": [0.0]})

    baseline_mean = batch_means.iloc[0]
    baseline_std = drift_base[MODEL_FEATURES].std().replace(0, 1)
    drift = ((batch_means - baseline_mean).abs() / baseline_std).mean(axis=1)

    return drift.reset_index(name="drift_score")


def make_alert_ticker(frame: pd.DataFrame) -> str:
    critical = frame[frame["severity"] == "Critical"].sort_values("risk_score", ascending=False).head(10)
    if critical.empty:
        return "No critical alerts right now. Monitoring all channels in dark mode operations view."

    events = []
    for row in critical.itertuples(index=False):
        events.append(
            f"Txn {row.txn_id} | Risk {row.risk_score:.1f} | Amount {row.txn_amount:,.0f} | Channel {row.payment_channel}"
        )
    return "  |  ".join(events)


def build_pdf_snapshot(metrics: Dict[str, str], risk_values: pd.Series) -> Optional[bytes]:
    if not MATPLOTLIB_AVAILABLE:
        return None

    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        summary_fig, summary_ax = plt.subplots(figsize=(11.7, 8.3))
        summary_ax.axis("off")
        summary_ax.text(
            0.02,
            0.96,
            "SecurePay Executive Snapshot",
            fontsize=20,
            fontweight="bold",
            va="top",
        )
        summary_ax.text(
            0.02,
            0.90,
            "\n".join([f"{label}: {value}" for label, value in metrics.items()]),
            fontsize=12,
            va="top",
            linespacing=1.5,
        )
        pdf.savefig(summary_fig, bbox_inches="tight")
        plt.close(summary_fig)

        hist_fig, hist_ax = plt.subplots(figsize=(11.7, 4.3))
        hist_ax.hist(risk_values, bins=28, color="#1d4ed8", edgecolor="#e5edf6")
        hist_ax.set_title("Risk Score Distribution")
        hist_ax.set_xlabel("Risk Score")
        hist_ax.set_ylabel("Transaction Count")
        hist_ax.grid(alpha=0.25)
        pdf.savefig(hist_fig, bbox_inches="tight")
        plt.close(hist_fig)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


@st.cache_data(show_spinner=False)
def load_demo_data() -> pd.DataFrame:
    return pd.read_csv("securepay_txn_demo.csv")


st.markdown("<div class='title'>SecurePay Risk Console</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Bank-style anomaly monitoring with high-contrast dark UI, model health, and case operations.</div>",
    unsafe_allow_html=True,
)
if not PLOTLY_AVAILABLE:
    st.warning("Interactive charts are in fallback mode. Install `plotly` for full chart experience.")
if not MATPLOTLIB_AVAILABLE:
    st.info("PDF export is disabled. Install `matplotlib` to enable executive PDF snapshots.")

with st.sidebar:
    st.markdown("### Control Center")
    role_view = st.selectbox("Role View", ["Manager", "Analyst"], index=0)
    source = st.radio("Data Source", ["Upload CSV", "Demo data"], index=0)

    uploaded_file = None
    if source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])
    else:
        st.caption("Using local demo file: securepay_txn_demo.csv")

    st.markdown("---")
    st.markdown("### Model Tuning")
    contamination = st.slider("Contamination", min_value=0.005, max_value=0.10, value=0.015, step=0.005)
    n_estimators = st.slider("Estimators", min_value=50, max_value=400, value=180, step=10)
    risk_threshold = st.slider("High-risk score threshold", min_value=50, max_value=99, value=82, step=1)
    random_seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42, step=1)

    st.markdown("---")
    st.markdown("### Session")
    auto_refresh = st.checkbox("Auto-refresh")
    refresh_seconds = st.slider("Refresh every (sec)", min_value=10, max_value=120, value=30, step=5)
    if auto_refresh:
        autorefresh_fn = getattr(st, "autorefresh", None)
        if callable(autorefresh_fn):
            autorefresh_fn(interval=refresh_seconds * 1000, key="securepay-refresh")
        else:
            st.caption("Auto-refresh unavailable in this Streamlit version.")

if source == "Upload CSV":
    if uploaded_file is None:
        st.info("Upload a CSV with the exact 8 expected columns. Showing demo data for now.")
        raw_df = load_demo_data()
    else:
        raw_df = pd.read_csv(uploaded_file)
else:
    raw_df = load_demo_data()

missing_columns = [col for col in REQUIRED_COLUMNS if col not in raw_df.columns]
if missing_columns:
    st.error(f"Dataset missing required columns: {', '.join(missing_columns)}")
    st.stop()

df = raw_df[REQUIRED_COLUMNS].copy()
for feature in MODEL_FEATURES + ["risk_flag"]:
    df[feature] = pd.to_numeric(df[feature], errors="coerce")
df["txn_id"] = df["txn_id"].astype(str)
df["payment_channel"] = df["payment_channel"].astype(str)

validity_mask = (
    df["txn_hour"].between(0, 23)
    & (df["txn_amount"] >= 0)
    & (df["txn_velocity"] >= 0)
    & df["behavior_score"].between(0, 1)
    & df["risk_flag"].isin([0, 1])
)
completeness = 1.0 - df[REQUIRED_COLUMNS].isna().mean().mean()
validity = validity_mask.fillna(False).mean()
data_quality_score = round(100 * (0.6 * completeness + 0.4 * validity), 2)

model_frame = df[MODEL_FEATURES].fillna(df[MODEL_FEATURES].median(numeric_only=True))
model = IsolationForest(
    n_estimators=n_estimators,
    contamination=contamination,
    random_state=int(random_seed),
)
model.fit(model_frame)

raw_anomaly_score = -model.decision_function(model_frame)
if np.isclose(raw_anomaly_score.max(), raw_anomaly_score.min()):
    normalized_risk = np.full_like(raw_anomaly_score, 50.0)
else:
    normalized_risk = 100 * ((raw_anomaly_score - raw_anomaly_score.min()) / (raw_anomaly_score.max() - raw_anomaly_score.min()))

df["iforest_flag"] = (model.predict(model_frame) == -1).astype(int)
df["risk_score"] = np.round(normalized_risk, 2)
df["high_risk"] = ((df["risk_score"] >= risk_threshold) | (df["iforest_flag"] == 1)).astype(int)
df["risk_flag"] = df["risk_flag"].fillna(0).clip(0, 1).astype(int)

df["severity"] = pd.cut(
    df["risk_score"],
    bins=[-0.001, 40, 70, 85, 100],
    labels=["Low", "Medium", "High", "Critical"],
    include_lowest=True,
).astype(str)
df["severity"] = pd.Categorical(df["severity"], categories=SEVERITY_ORDER, ordered=True)

df["alert_generated"] = (df["risk_score"] >= 70).astype(int)
df["alert_triaged"] = (df["risk_score"] >= 75).astype(int)
df["alert_confirmed"] = (df["risk_score"] >= 85).astype(int)
df["alert_blocked"] = (df["risk_score"] >= 92).astype(int)
df["open_case"] = ((df["alert_confirmed"] == 1) & (df["alert_blocked"] == 0)).astype(int)

latency_ms = (
    90
    + (df["txn_velocity"].fillna(0) * 55)
    + (df["behavior_score"].fillna(0) * 45)
    + (df["txn_amount"].fillna(0) / max(float(df["txn_amount"].median()), 1.0) * 12)
)
df["latency_ms_est"] = latency_ms.clip(lower=60, upper=1500).round(2)
df["sla_breach"] = (df["latency_ms_est"] > 450).astype(int)
df = add_reason_codes(df)

with st.sidebar:
    st.markdown("---")
    st.markdown("### Global Filters")

    min_amount = float(df["txn_amount"].min())
    max_amount = float(df["txn_amount"].max())
    amount_range = st.slider(
        "Amount Range",
        min_value=float(np.floor(min_amount)),
        max_value=float(np.ceil(max_amount)),
        value=(float(np.floor(min_amount)), float(np.ceil(max_amount))),
        step=10.0,
    )

    hour_range = st.slider("Hour Range", min_value=0, max_value=23, value=(0, 23), step=1)

    channel_options = sorted(df["payment_channel"].dropna().unique().tolist())
    selected_channels = st.multiselect("Payment Channel", options=channel_options, default=channel_options)

    selected_severity = st.multiselect("Severity", options=SEVERITY_ORDER, default=SEVERITY_ORDER)
    only_high_risk = st.checkbox("Show only high-risk transactions", value=False)

filtered = df[
    (df["txn_amount"].between(amount_range[0], amount_range[1]))
    & (df["txn_hour"].between(hour_range[0], hour_range[1]))
    & (df["payment_channel"].isin(selected_channels))
    & (df["severity"].astype(str).isin(selected_severity))
].copy()

if only_high_risk:
    filtered = filtered[filtered["high_risk"] == 1].copy()

if filtered.empty:
    st.warning("No rows match the current filters.")
    st.stop()

total_txn = len(filtered)
suspicious_count = int(filtered["high_risk"].sum())
normal_count = total_txn - suspicious_count
total_value = float(filtered["txn_amount"].sum())
suspicious_value = float(filtered.loc[filtered["high_risk"] == 1, "txn_amount"].sum())
high_risk_rate = safe_div(suspicious_count * 100, total_txn)
blocked_count = int(filtered["alert_blocked"].sum())
blocked_value = float(filtered.loc[filtered["alert_blocked"] == 1, "txn_amount"].sum())
prevented_loss = blocked_value * 0.75
open_alerts = int(filtered["alert_generated"].sum() - filtered["alert_blocked"].sum())
open_cases = int(filtered["open_case"].sum())
sla_breach_count = int(filtered["sla_breach"].sum())
avg_latency = float(filtered["latency_ms_est"].mean())
p95_latency = float(filtered["latency_ms_est"].quantile(0.95))

true_y = filtered["risk_flag"].astype(int)
pred_y = filtered["high_risk"].astype(int)
tn, fp, fn, tp = confusion_matrix(true_y, pred_y, labels=[0, 1]).ravel()
precision = safe_div(tp, tp + fp)
recall = safe_div(tp, tp + fn)
false_positive_rate = safe_div(fp, fp + tn)

funnel_data = pd.DataFrame(
    {
        "Stage": ["Generated", "Triaged", "Confirmed", "Blocked"],
        "Count": [
            int(filtered["alert_generated"].sum()),
            int(filtered["alert_triaged"].sum()),
            int(filtered["alert_confirmed"].sum()),
            int(filtered["alert_blocked"].sum()),
        ],
    }
)

drift_data = compute_drift_by_batch(filtered)
drift_latest = float(drift_data["drift_score"].iloc[-1])

risk_status = "GREEN"
if high_risk_rate >= 8:
    risk_status = "RED"
elif high_risk_rate >= 4:
    risk_status = "AMBER"

sla_status = "GREEN"
if safe_div(sla_breach_count * 100, total_txn) >= 5:
    sla_status = "RED"
elif safe_div(sla_breach_count * 100, total_txn) >= 2:
    sla_status = "AMBER"

drift_status = "GREEN"
if drift_latest >= 0.60:
    drift_status = "RED"
elif drift_latest >= 0.30:
    drift_status = "AMBER"

st.markdown(f"<div class='ticker'><span>{make_alert_ticker(filtered)}</span></div>", unsafe_allow_html=True)
st.markdown(
    (
        "<div class='status-row'>"
        f"<div class='status-chip'>Risk Posture: {risk_status}</div>"
        f"<div class='status-chip'>SLA Health: {sla_status}</div>"
        f"<div class='status-chip'>Drift Watch: {drift_status}</div>"
        f"<div class='status-chip'>Role: {role_view}</div>"
        "</div>"
    ),
    unsafe_allow_html=True,
)

st.markdown("")
kpi_row_1 = st.columns(5)
kpi_row_1[0].metric("Total Transactions", f"{total_txn:,}")
kpi_row_1[1].metric("Total Transaction Value", f"{total_value:,.2f}")
kpi_row_1[2].metric("Suspicious Transactions", f"{suspicious_count:,}")
kpi_row_1[3].metric("Suspicious Value", f"{suspicious_value:,.2f}")
kpi_row_1[4].metric("High-Risk Rate", f"{high_risk_rate:.2f}%")

kpi_row_2 = st.columns(5)
kpi_row_2[0].metric("Blocked Suspicious", f"{blocked_count:,}")
kpi_row_2[1].metric("Estimated Prevented Loss", f"{prevented_loss:,.2f}")
kpi_row_2[2].metric("False Positive Rate", f"{false_positive_rate * 100:.2f}%")
kpi_row_2[3].metric("Precision", f"{precision * 100:.2f}%")
kpi_row_2[4].metric("Recall", f"{recall * 100:.2f}%")

kpi_row_3 = st.columns(5)
kpi_row_3[0].metric("Avg Detection Latency (est.)", f"{avg_latency:.1f} ms")
kpi_row_3[1].metric("P95 Latency (est.)", f"{p95_latency:.1f} ms")
kpi_row_3[2].metric("Open Alerts", f"{open_alerts:,}")
kpi_row_3[3].metric("Open Cases", f"{open_cases:,}")
kpi_row_3[4].metric("Data Quality Score", f"{data_quality_score:.2f}%")

summary_payload = {
    "Transactions": f"{total_txn:,}",
    "Total Value": f"{total_value:,.2f}",
    "High-Risk Txns": f"{suspicious_count:,}",
    "High-Risk Rate": f"{high_risk_rate:.2f}%",
    "Precision": f"{precision * 100:.2f}%",
    "Recall": f"{recall * 100:.2f}%",
    "Data Quality": f"{data_quality_score:.2f}%",
}
pdf_report = build_pdf_snapshot(summary_payload, filtered["risk_score"])

export_col_1, export_col_2, export_col_3 = st.columns(3)
export_col_1.download_button(
    "Download Scored Data (CSV)",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="securepay_scored_transactions.csv",
    mime="text/csv",
)
export_col_2.download_button(
    "Download Suspicious Cases (CSV)",
    data=filtered[filtered["high_risk"] == 1].to_csv(index=False).encode("utf-8"),
    file_name="securepay_suspicious_transactions.csv",
    mime="text/csv",
)
if pdf_report is not None:
    export_col_3.download_button(
        "Download Executive Snapshot (PDF)",
        data=pdf_report,
        file_name="securepay_executive_snapshot.pdf",
        mime="application/pdf",
    )
else:
    export_col_3.caption("PDF export unavailable (missing matplotlib).")

tab_exec, tab_ops, tab_model, tab_cases = st.tabs(
    ["Executive", "Risk Operations", "Model Health", "Case Console"]
)

with tab_exec:
    if not PLOTLY_AVAILABLE:
        left, right = st.columns(2)
        with left:
            severity_counts = (
                filtered["severity"]
                .astype(str)
                .value_counts()
                .reindex(["Low", "Medium", "High", "Critical"], fill_value=0)
            )
            st.markdown("#### Risk Severity Distribution")
            st.bar_chart(severity_counts)

        with right:
            hourly_counts = (
                filtered.groupby(["txn_hour", "high_risk"], as_index=False)
                .size()
                .rename(columns={"size": "count"})
                .pivot(index="txn_hour", columns="high_risk", values="count")
                .fillna(0)
                .rename(columns={0: "Normal", 1: "High Risk"})
            )
            st.markdown("#### Hourly Trend")
            st.line_chart(hourly_counts)

        left, right = st.columns(2)
        with left:
            channel_counts = (
                filtered.groupby(["payment_channel", "high_risk"], as_index=False)
                .size()
                .rename(columns={"size": "count"})
                .pivot(index="payment_channel", columns="high_risk", values="count")
                .fillna(0)
                .rename(columns={0: "Normal", 1: "High Risk"})
            )
            st.markdown("#### Channel Breakdown")
            st.bar_chart(channel_counts)

        with right:
            box_data = filtered[["severity", "txn_amount"]].copy()
            st.markdown("#### Amount by Severity (table fallback)")
            st.dataframe(
                box_data.groupby("severity", observed=False)["txn_amount"].describe().reset_index(),
                width="stretch",
            )
    else:
        left, right = st.columns(2)

        with left:
            risk_dist = px.histogram(
                filtered,
                x="risk_score",
                nbins=30,
                color="severity",
                color_discrete_map=SEVERITY_COLORS,
                title="Risk Score Distribution",
                template="plotly_dark",
            )
            risk_dist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e5edf6",
                legend_title_text="Severity",
            )
            st.plotly_chart(risk_dist, width="stretch", config=PLOT_CONFIG)

        with right:
            hourly = (
                filtered.groupby(["txn_hour", "high_risk"], as_index=False)
                .size()
                .rename(columns={"size": "count"})
            )
            hourly["class"] = hourly["high_risk"].map({0: "Normal", 1: "High Risk"})
            hourly_fig = px.line(
                hourly,
                x="txn_hour",
                y="count",
                color="class",
                markers=True,
                title="Hourly Suspicious Trend vs Normal Baseline",
                color_discrete_map={"Normal": "#22c55e", "High Risk": "#ef4444"},
                template="plotly_dark",
            )
            hourly_fig.update_layout(
                xaxis_title="Transaction Hour",
                yaxis_title="Transaction Count",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e5edf6",
            )
            st.plotly_chart(hourly_fig, width="stretch", config=PLOT_CONFIG)

        left, right = st.columns(2)
        with left:
            stacked = (
                filtered.groupby(["payment_channel", "high_risk"], as_index=False)
                .size()
                .rename(columns={"size": "count"})
            )
            stacked["class"] = stacked["high_risk"].map({0: "Normal", 1: "High Risk"})
            channel_fig = px.bar(
                stacked,
                x="payment_channel",
                y="count",
                color="class",
                barmode="stack",
                title="Anomalies by Payment Channel",
                color_discrete_map={"Normal": "#3b82f6", "High Risk": "#ef4444"},
                template="plotly_dark",
            )
            channel_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e5edf6",
                xaxis_title="Payment Channel",
                yaxis_title="Transaction Count",
            )
            st.plotly_chart(channel_fig, width="stretch", config=PLOT_CONFIG)

        with right:
            box_fig = px.box(
                filtered,
                x="severity",
                y="txn_amount",
                color="severity",
                category_orders={"severity": SEVERITY_ORDER},
                color_discrete_map=SEVERITY_COLORS,
                title="Transaction Amount by Risk Severity",
                template="plotly_dark",
            )
            box_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e5edf6",
                xaxis_title="Severity",
                yaxis_title="Transaction Amount",
                showlegend=False,
            )
            st.plotly_chart(box_fig, width="stretch", config=PLOT_CONFIG)

with tab_ops:
    if not PLOTLY_AVAILABLE:
        left, right = st.columns(2)
        with left:
            st.markdown("#### Alert Funnel (table fallback)")
            st.dataframe(funnel_data, width="stretch", hide_index=True)
        with right:
            heatmap_data = (
                filtered.pivot_table(
                    index="payment_channel",
                    columns="txn_hour",
                    values="high_risk",
                    aggfunc="sum",
                    fill_value=0,
                )
                .sort_index()
                .sort_index(axis=1)
            )
            st.markdown("#### Heatmap Data (fallback)")
            st.dataframe(heatmap_data, width="stretch")

        left, right = st.columns(2)
        with left:
            top_risky = filtered[filtered["high_risk"] == 1].sort_values("txn_amount", ascending=False).head(15)
            st.markdown("#### Top Risky Transactions")
            st.dataframe(top_risky[["txn_id", "txn_amount", "risk_score", "payment_channel"]], width="stretch")
        with right:
            exposure = suspicious_value
            prevented = prevented_loss
            residual = max(exposure - prevented, 0.0)
            st.markdown("#### Loss Waterfall Summary")
            summary_table = pd.DataFrame(
                {
                    "Metric": ["Potential Exposure", "Prevented Loss", "Residual Risk"],
                    "Value": [exposure, prevented, residual],
                }
            )
            st.dataframe(summary_table, width="stretch", hide_index=True)
    else:
        left, right = st.columns(2)

        with left:
            funnel_fig = go.Figure(
                go.Funnel(
                    y=funnel_data["Stage"],
                    x=funnel_data["Count"],
                    textinfo="value+percent previous",
                    marker=dict(color=["#38bdf8", "#0ea5e9", "#f97316", "#ef4444"]),
                )
            )
            funnel_fig.update_layout(
                title="Alert Funnel: Generated -> Triaged -> Confirmed -> Blocked",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e5edf6",
            )
            st.plotly_chart(funnel_fig, width="stretch", config=PLOT_CONFIG)

        with right:
            heatmap_data = (
                filtered.pivot_table(
                    index="payment_channel",
                    columns="txn_hour",
                    values="high_risk",
                    aggfunc="sum",
                    fill_value=0,
                )
                .sort_index()
                .sort_index(axis=1)
            )
            heatmap_fig = px.imshow(
                heatmap_data,
                color_continuous_scale="Reds",
                title="High-Risk Heatmap: Payment Channel x Hour",
                labels={"x": "Transaction Hour", "y": "Payment Channel", "color": "High-Risk Count"},
                aspect="auto",
                template="plotly_dark",
            )
            heatmap_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e5edf6",
            )
            st.plotly_chart(heatmap_fig, width="stretch", config=PLOT_CONFIG)

        left, right = st.columns(2)
        with left:
            top_risky = filtered[filtered["high_risk"] == 1].sort_values("txn_amount", ascending=False).head(15)
            if top_risky.empty:
                st.info("No high-risk transactions in current filter for Pareto analysis.")
            else:
                pareto = top_risky[["txn_id", "txn_amount"]].copy()
                pareto["cum_pct"] = pareto["txn_amount"].cumsum() / pareto["txn_amount"].sum() * 100
                pareto_fig = go.Figure()
                pareto_fig.add_trace(
                    go.Bar(x=pareto["txn_id"], y=pareto["txn_amount"], name="Amount at Risk", marker_color="#f97316")
                )
                pareto_fig.add_trace(
                    go.Scatter(
                        x=pareto["txn_id"],
                        y=pareto["cum_pct"],
                        name="Cumulative %",
                        yaxis="y2",
                        marker_color="#22c55e",
                    )
                )
                pareto_fig.update_layout(
                    title="Top Risky Transactions (Pareto by Value)",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e5edf6",
                    xaxis=dict(title="Transaction ID"),
                    yaxis=dict(title="Amount at Risk"),
                    yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 100]),
                    legend=dict(orientation="h"),
                )
                st.plotly_chart(pareto_fig, width="stretch", config=PLOT_CONFIG)

        with right:
            exposure = suspicious_value
            prevented = prevented_loss
            residual = max(exposure - prevented, 0.0)

            waterfall_fig = go.Figure(
                go.Waterfall(
                    name="Exposure",
                    orientation="v",
                    measure=["absolute", "relative", "total"],
                    x=["Potential Exposure", "Prevented Loss", "Residual Risk"],
                    y=[exposure, -prevented, residual],
                    connector={"line": {"color": "#94a3b8"}},
                    increasing={"marker": {"color": "#ef4444"}},
                    decreasing={"marker": {"color": "#22c55e"}},
                    totals={"marker": {"color": "#3b82f6"}},
                )
            )
            waterfall_fig.update_layout(
                title="Potential Loss vs Prevented Loss",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e5edf6",
                yaxis_title="Amount",
            )
            st.plotly_chart(waterfall_fig, width="stretch", config=PLOT_CONFIG)

with tab_model:
    left, right = st.columns(2)

    if not PLOTLY_AVAILABLE:
        with left:
            cm_df = pd.DataFrame(
                [[tn, fp], [fn, tp]],
                index=["Actual Normal", "Actual High-Risk"],
                columns=["Pred Normal", "Pred High-Risk"],
            )
            st.markdown("#### Confusion Matrix (table fallback)")
            st.dataframe(cm_df, width="stretch")

        with right:
            if true_y.nunique() <= 1:
                st.info("Precision-Recall curve requires both classes in current filter.")
            else:
                pr_precision, pr_recall, _ = precision_recall_curve(true_y, filtered["risk_score"] / 100.0)
                pr_df = pd.DataFrame({"Recall": pr_recall, "Precision": pr_precision}).set_index("Recall")
                st.markdown("#### Precision-Recall (line fallback)")
                st.line_chart(pr_df)

        st.markdown("#### Drift Trend")
        st.line_chart(drift_data.set_index("batch"))
    else:
        with left:
            cm = np.array([[tn, fp], [fn, tp]])
            cm_fig = px.imshow(
                cm,
                x=["Pred Normal", "Pred High-Risk"],
                y=["Actual Normal", "Actual High-Risk"],
                text_auto=True,
                color_continuous_scale="Blues",
                title="Confusion Matrix",
                template="plotly_dark",
            )
            cm_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e5edf6",
            )
            st.plotly_chart(cm_fig, width="stretch", config=PLOT_CONFIG)

        with right:
            if true_y.nunique() <= 1:
                st.info("Precision-Recall curve requires both classes in current filter.")
            else:
                pr_precision, pr_recall, pr_thresholds = precision_recall_curve(true_y, filtered["risk_score"] / 100.0)
                pr_fig = go.Figure()
                pr_fig.add_trace(
                    go.Scatter(
                        x=pr_recall,
                        y=pr_precision,
                        mode="lines",
                        name="PR Curve",
                        line=dict(color="#38bdf8", width=2),
                    )
                )

                if len(pr_thresholds) > 0:
                    threshold_norm = risk_threshold / 100.0
                    idx = int(np.argmin(np.abs(pr_thresholds - threshold_norm)))
                    pr_fig.add_trace(
                        go.Scatter(
                            x=[pr_recall[idx]],
                            y=[pr_precision[idx]],
                            mode="markers",
                            name=f"Current threshold ({risk_threshold})",
                            marker=dict(size=11, color="#ef4444", symbol="diamond"),
                        )
                    )

                pr_fig.update_layout(
                    title="Precision-Recall Curve",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e5edf6",
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                )
                st.plotly_chart(pr_fig, width="stretch", config=PLOT_CONFIG)

        drift_fig = px.line(
            drift_data,
            x="batch",
            y="drift_score",
            markers=True,
            title="Batch Drift Score vs Initial Baseline",
            template="plotly_dark",
        )
        drift_fig.update_layout(
            xaxis_title="Batch Number",
            yaxis_title="Drift Score",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e5edf6",
        )
        st.plotly_chart(drift_fig, width="stretch", config=PLOT_CONFIG)

    model_health_row = st.columns(4)
    model_health_row[0].metric("Model Contamination", f"{contamination:.3f}")
    model_health_row[1].metric("Current Risk Threshold", str(risk_threshold))
    model_health_row[2].metric("Latest Drift Score", f"{drift_latest:.3f}")
    model_health_row[3].metric("Normal Transactions", f"{normal_count:,}")

with tab_cases:
    st.markdown("#### Case Queue")
    cases = filtered[filtered["high_risk"] == 1].sort_values("risk_score", ascending=False).copy()

    if cases.empty:
        st.success("No high-risk transactions in current view.")
    else:
        view_columns = [
            "txn_id",
            "risk_score",
            "severity",
            "txn_amount",
            "payment_channel",
            "txn_hour",
            "reason_codes",
        ]
        if role_view == "Manager":
            manager_view = (
                cases.groupby("payment_channel", as_index=False)
                .agg(
                    high_risk_count=("txn_id", "count"),
                    total_value_at_risk=("txn_amount", "sum"),
                    avg_risk_score=("risk_score", "mean"),
                )
                .sort_values("high_risk_count", ascending=False)
            )
            st.dataframe(manager_view, width="stretch")
        else:
            st.dataframe(cases[view_columns], width="stretch", hide_index=True)

        selected_txn = st.selectbox("Drill-down transaction", options=cases["txn_id"].tolist())
        selected_row = cases[cases["txn_id"] == selected_txn].iloc[0]

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Risk Score", f"{selected_row['risk_score']:.2f}")
        d2.metric("Amount", f"{selected_row['txn_amount']:,.2f}")
        d3.metric("Hour", int(selected_row["txn_hour"]))
        d4.metric("Channel", selected_row["payment_channel"])

        st.markdown(f"**Why flagged:** `{selected_row['reason_codes']}`")

st.markdown(
    """
    <div class='footer'>
        (c) 2026 SecurePay | Intelligent Transaction Monitoring System<br>
        Academic Research Prototype - Developed by Mehul Kumar
    </div>
    """,
    unsafe_allow_html=True,
)
