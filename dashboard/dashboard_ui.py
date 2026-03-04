"""Streamlit dashboard for bank agents."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

try:
    from sklearn.linear_model import LinearRegression
except Exception:
    LinearRegression = None

from backend.duplicate_detection import find_similar_complaints
from backend.pipeline import bootstrap_from_sample, process_single_complaint
from backend.response_generator import generate_response_draft
from backend.root_cause_analysis import summarize_root_causes
from backend.sla_tracking import find_sla_alerts, get_sla_state

st.set_page_config(page_title="AI Banking Complaint Dashboard", page_icon="🏦", layout="wide")

db = bootstrap_from_sample()
rows = db.fetch_complaints()
df = pd.DataFrame(rows)

st.title("AI Powered Unified Customer Complaint Communication Dashboard")
st.caption("Bank agent console for multi-channel complaint triage and GenAI-assisted resolution.")

if df.empty:
    st.warning("No complaint data found.")
    st.stop()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Complaints", len(df))
kpi2.metric("Escalated", int((df["escalated"] == 1).sum()))
kpi3.metric("Critical/High", int(df["severity"].isin(["critical", "high"]).sum()))
kpi4.metric("Open", int(df["status"].isin(["open", "escalated"]).sum()))

st.subheader("New Complaint Intake (Multi-channel)")
with st.form("new_complaint_form", clear_on_submit=True):
    c1, c2, c3 = st.columns(3)
    source_channel = c1.selectbox("Source Channel", ["email", "website_form", "chatbot", "social_media", "call_center"])
    customer_id = c2.text_input("Customer ID", value="CUST-NEW")
    created_at = c3.text_input("Created At (ISO, optional)", value="")
    complaint_text = st.text_area("Complaint Text")
    submitted = st.form_submit_button("Ingest & Analyze")
    if submitted and complaint_text.strip():
        payload = {
            "source_channel": source_channel,
            "customer_id": customer_id,
            "complaint_text": complaint_text,
            "created_at": created_at or None,
        }
        new_id = process_single_complaint(db, payload)
        st.success(f"Complaint #{new_id} ingested, analyzed, and response drafted.")
        st.rerun()

st.subheader("Complaint Table & Filters")
f1, f2, f3, f4 = st.columns(4)
selected_categories = f1.multiselect("Category", sorted([x for x in df["category"].dropna().unique()]), default=[])
selected_sentiments = f2.multiselect("Sentiment", sorted([x for x in df["sentiment"].dropna().unique()]), default=[])
selected_severity = f3.multiselect("Severity", sorted([x for x in df["severity"].dropna().unique()]), default=[])
selected_channels = f4.multiselect("Channel", sorted([x for x in df["source_channel"].dropna().unique()]), default=[])

filtered = df.copy()
if selected_categories:
    filtered = filtered[filtered["category"].isin(selected_categories)]
if selected_sentiments:
    filtered = filtered[filtered["sentiment"].isin(selected_sentiments)]
if selected_severity:
    filtered = filtered[filtered["severity"].isin(selected_severity)]
if selected_channels:
    filtered = filtered[filtered["source_channel"].isin(selected_channels)]

st.dataframe(
    filtered[["id", "source_channel", "customer_id", "category", "product", "sentiment", "severity", "root_cause", "status", "created_at", "sla_hours"]],
    width="stretch",
)

st.subheader("Analytics")
a1, a2 = st.columns(2)
a1.write("Most Common Categories")
a1.bar_chart(filtered["category"].value_counts())
a2.write("Sentiment Distribution")
a2.bar_chart(filtered["sentiment"].value_counts())

trend_df = filtered.copy()
trend_df["created_at"] = pd.to_datetime(trend_df["created_at"], errors="coerce")
trend_df = trend_df.dropna(subset=["created_at"])
if not trend_df.empty:
    daily_counts = trend_df.groupby(trend_df["created_at"].dt.date).size().reset_index(name="count")
    daily_counts["created_at"] = pd.to_datetime(daily_counts["created_at"])
    daily_counts = daily_counts.set_index("created_at")
    st.write("Complaint Trend (Daily)")
    st.line_chart(daily_counts["count"])

    horizon = st.slider("Forecast Horizon (days)", 3, 30, 7, key="forecast_horizon")
    if len(daily_counts) >= 2:
        y = daily_counts["count"].values.astype(float)
        if LinearRegression is not None:
            x = np.arange(len(daily_counts)).reshape(-1, 1)
            model = LinearRegression().fit(x, y)
            future_x = np.arange(len(daily_counts), len(daily_counts) + horizon).reshape(-1, 1)
            preds = np.maximum(model.predict(future_x), 0)
        else:
            preds = np.repeat(np.mean(y), horizon)
        last_date = daily_counts.index.max()
        forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon)
        forecast = pd.DataFrame({"date": forecast_dates, "predicted_count": np.round(preds, 2)})
        forecast = forecast.set_index("date")
        st.write("Complaint Trend Prediction")
        st.line_chart(forecast["predicted_count"])
        st.dataframe(forecast.reset_index(), width="stretch")

st.write("Root Cause Patterns")
root_summary = summarize_root_causes(rows)
st.dataframe(pd.DataFrame(root_summary), width="stretch")

st.subheader("SLA Tracking & Severity Alerts")
sla_alerts = find_sla_alerts(rows)
if sla_alerts:
    st.error(f"{len(sla_alerts)} complaints are nearing or breaching SLA.")
    st.dataframe(
        pd.DataFrame(sla_alerts)[["id", "severity", "status", "elapsed_hours", "remaining_hours", "sla_state"]],
        width="stretch",
    )
else:
    st.success("No active SLA risk alerts.")

st.subheader("360 Degree Complaint View")
selected_id = st.selectbox("Select complaint ID", sorted(df["id"].tolist()))
selected = db.get_complaint(int(selected_id))

if selected:
    s1, s2 = st.columns(2)
    s1.write("Complaint Text")
    s1.info(selected["complaint_text"])
    s1.write("Suggested Response")
    s1.success(selected.get("response_draft") or "No response draft available")

    state = get_sla_state(selected["created_at"], int(selected.get("sla_hours") or 24), selected.get("resolved_at"))
    s2.json(
        {
            "category": selected.get("category"),
            "product": selected.get("product"),
            "sentiment": selected.get("sentiment"),
            "severity": selected.get("severity"),
            "root_cause": selected.get("root_cause"),
            "key_issues": selected.get("key_issues"),
            "status": selected.get("status"),
            "sla_state": state["sla_state"],
            "elapsed_hours": state["elapsed_hours"],
            "remaining_hours": state["remaining_hours"],
            "escalated": bool(selected.get("escalated")),
        }
    )

    timeline = db.fetch_timeline(int(selected_id))
    st.write("Timeline History")
    st.dataframe(pd.DataFrame(timeline), width="stretch")

    similar = db.fetch_similar_complaints(int(selected_id))
    st.write("Related / Similar Complaints")
    if similar:
        st.dataframe(pd.DataFrame(similar), width="stretch")
    else:
        semantic_candidates = [
            {"id": int(r["id"]), "complaint_text": r["complaint_text"]}
            for r in rows
            if int(r["id"]) != int(selected_id)
        ]
        semantic = find_similar_complaints(selected["complaint_text"], semantic_candidates, threshold=0.72, top_k=3)
        if semantic:
            st.dataframe(pd.DataFrame(semantic), width="stretch")
        else:
            st.info("No semantic matches above threshold.")

    if st.button("Regenerate Response Draft"):
        response = generate_response_draft(
            complaint_text=selected["complaint_text"],
            category=selected.get("category") or "other",
            product=selected.get("product") or "account",
            sentiment=selected.get("sentiment") or "neutral",
            severity=selected.get("severity") or "medium",
            key_issues=selected.get("key_issues") or "",
        )
        db.update_response(int(selected_id), response["draft"], response["mode"])
        db.add_timeline_event(int(selected_id), "RESPONSE_REGENERATED", "Agent regenerated GenAI draft")
        st.success(f"Response regenerated with mode: {response['mode']}")
        st.rerun()

st.subheader("Regulatory Reporting")
if st.button("Generate Regulatory Summary"):
    now = datetime.utcnow().isoformat(timespec="seconds")
    total = len(rows)
    severe = int(sum(1 for r in rows if r.get("severity") in {"high", "critical"}))
    escalated = int(sum(1 for r in rows if int(r.get("escalated") or 0) == 1))
    breached = len([x for x in sla_alerts if x["sla_state"] == "breached"])
    top_root = root_summary[0]["root_cause"] if root_summary else "n/a"
    report = (
        f"Regulatory Complaint Summary ({now} UTC)\n"
        f"- Total complaints: {total}\n"
        f"- High/Critical complaints: {severe}\n"
        f"- Escalated complaints: {escalated}\n"
        f"- SLA breaches: {breached}\n"
        f"- Dominant root cause pattern: {top_root}\n"
        "- Controls: NLP categorization, sentiment/severity scoring, semantic duplicate checks, and GenAI agent-assisted drafting are active.\n"
    )
    st.code(report)
    st.download_button("Download Report", data=report, file_name="regulatory_report.txt")
