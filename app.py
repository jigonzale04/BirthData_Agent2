# ==============================
# IMPORTS (MUST BE FIRST)
# ==============================

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import re

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(layout="wide")

# ==============================
# HELPER FUNCTIONS (DEFINE FIRST)
# ==============================

def classify_intent(user_input: str) -> str:
    text = user_input.lower()

    pandas_keywords = [
        "table",
        "breakdown",
        "monthly",
        "show rows",
        "create table",
        "list",
        "display",
        "data for",
    ]

    llm_keywords = [
        "why",
        "compare",
        "trend",
        "interpret",
        "implication",
        "explain",
        "analysis",
        "insight",
    ]

    if any(k in text for k in pandas_keywords):
        return "pandas"

    if any(k in text for k in llm_keywords):
        return "llm"

    return "llm"


def extract_state(user_input, df):
    for state in df["state_of_residence"].unique():
        if state.lower() in user_input.lower():
            return state
    return None


# ==============================
# DATA LOADING
# ==============================

@st.cache_data
def load_data():
    return pd.read_csv("Provisional_Natality_2025_CDC.csv")

df = load_data()
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
df["births"] = pd.to_numeric(df["births"], errors="coerce")
df = df.dropna(subset=["births"])

# ==============================
# TITLE
# ==============================

st.title("Provisional Natality Data Dashboard")
st.subheader("Birth Analysis by State and Gender")

# ==============================
# SIDEBAR FILTERS
# ==============================

st.sidebar.header("Filters")

states = sorted(df["state_of_residence"].unique())
genders = sorted(df["sex_of_infant"].unique())
months = sorted(df["month"].unique())

state_sel = st.sidebar.multiselect("State", ["All"] + states, default=["All"])
gender_sel = st.sidebar.multiselect("Gender", ["All"] + genders, default=["All"])
month_sel = st.sidebar.multiselect("Month", ["All"] + months, default=["All"])

filtered = df.copy()

if "All" not in state_sel:
    filtered = filtered[filtered["state_of_residence"].isin(state_sel)]

if "All" not in gender_sel:
    filtered = filtered[filtered["sex_of_infant"].isin(gender_sel)]

if "All" not in month_sel:
    filtered = filtered[filtered["month"].isin(month_sel)]

# ==============================
# CHART
# ==============================

agg = (
    filtered.groupby(["state_of_residence", "sex_of_infant"])["births"]
    .sum()
    .reset_index()
)

fig = px.bar(
    agg,
    x="state_of_residence",
    y="births",
    color="sex_of_infant",
    title="Total Births by State and Gender",
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Filtered Records")
st.dataframe(filtered, use_container_width=True)

# ==============================
# AI SECTION
# ==============================

st.markdown("---")
st.header("AI Data Analyst")

user_input = st.chat_input("Ask an analytical question about the data...")

if user_input:

    st.chat_message("user").markdown(user_input)

    intent = classify_intent(user_input)

    # ======================
    # PANDAS ROUTE
    # ======================

    if intent == "pandas":

        matched_state = extract_state(user_input, df)

        if matched_state:

            table_df = (
                df[df["state_of_residence"] == matched_state]
                .groupby("month")["births"]
                .sum()
                .reset_index()
            )

            st.chat_message("assistant").markdown(
                f"Here is the monthly breakdown for **{matched_state}**:"
            )

            st.dataframe(table_df.sort_values("month"))

        else:
            st.chat_message("assistant").markdown(
                "Please specify a valid state."
            )

    # ======================
    # LLM ROUTE
    # ======================

    else:

        total_births = int(filtered["births"].sum())

        state_totals = (
            filtered.groupby("state_of_residence")["births"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .to_dict()
        )

        context = {
            "total_births": total_births,
            "top_states": state_totals,
        }

        system_prompt = f"""
You are a senior data analyst.

Use ONLY the dataset context provided.
Do not fabricate numbers.
Provide concise executive insights.

Dataset Context:
{json.dumps(context)}
"""

        try:
            api_key = st.secrets["GROQ_API_KEY"]

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 400,
                },
                timeout=30,
            )

            result = response.json()["choices"][0]["message"]["content"]

        except Exception as e:
            result = f"Error: {str(e)}"

        st.chat_message("assistant").markdown(result)
