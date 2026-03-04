import streamlit as st
import pandas as pd
from utils import get_sentiment, categorize, severity, generate_response

st.title("AI Powered Customer Complaint Dashboard")

data = pd.read_csv("complaints.csv")

data["sentiment"] = data["complaint_text"].apply(get_sentiment)
data["category"] = data["complaint_text"].apply(categorize)
data["severity"] = data["complaint_text"].apply(severity)

st.subheader("Complaint Table")
st.dataframe(data)

st.subheader("Complaint Categories")
st.bar_chart(data["category"].value_counts())

st.subheader("Sentiment Distribution")
st.bar_chart(data["sentiment"].value_counts())

st.subheader("Generate AI Response")

complaint_id = st.number_input("Enter Complaint ID", min_value=1, max_value=len(data), step=1)

if st.button("Generate Response"):
    complaint_text = data[data["id"] == complaint_id]["complaint_text"].values[0]
    st.write("Complaint:")
    st.info(complaint_text)

    response = generate_response(complaint_text)

    st.write("Suggested Response:")
    st.success(response)

st.subheader("Upload New Complaint")

new_complaint = st.text_input("Enter complaint text")

if st.button("Analyze Complaint"):
    if new_complaint:
        result = {
            "Complaint": new_complaint,
            "Category": categorize(new_complaint),
            "Sentiment": get_sentiment(new_complaint),
            "Severity": severity(new_complaint),
            "Suggested Response": generate_response(new_complaint)
        }

        st.write(result)